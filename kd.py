import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import json
import numpy as np
from collections import defaultdict
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG 
# ─────────────────────────────────────────────
class Config:
    teacher_model   = "gpt2"
    student_model   = "distilgpt2"

    max_length      = 128
    batch_size      = 4
    epochs          = 8
    lr              = 3e-5
    warmup_ratio    = 0.1
    weight_decay    = 0.01

    temperature     = 2.5

    alpha           = 0.6   # KD
    beta            = 0.3   # CE
    gamma           = 0.05  # Hidden
    delta           = 0.05  # Attention

    grad_accum_steps = 2
    max_grad_norm   = 1.0

    subsample       = 20000
    patience        = 3
    label_smoothing = 0.05

    output_dir      = "distilled_student"
    metrics_json    = "training_metrics.json"

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(cfg.output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# TOKENIZER & MODELS
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model)
tokenizer.pad_token = tokenizer.eos_token

teacher = AutoModelForCausalLM.from_pretrained(
    cfg.teacher_model,
    output_hidden_states=True,
    output_attentions=True,
).to(device).eval()

student = AutoModelForCausalLM.from_pretrained(
    cfg.student_model,
    output_hidden_states=True,
    output_attentions=True,
).to(device)

# Freeze first 2 layers (stability)
for name, param in student.named_parameters():
    if "h.0" in name or "h.1" in name:
        param.requires_grad = False

# Projection
projection = nn.Linear(teacher.config.n_embd, student.config.n_embd, bias=False).to(device)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
val_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

if len(dataset) > cfg.subsample:
    dataset = dataset.select(range(cfg.subsample))

def tokenize(ex):
    return tokenizer(ex["text"], truncation=True,
                     padding="max_length", max_length=cfg.max_length)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
val_set = val_set.map(tokenize, batched=True, remove_columns=["text"])

for ds in [dataset, val_set]:
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

dataset = dataset.filter(lambda e: e["attention_mask"].sum() > 1)
val_set = val_set.filter(lambda e: e["attention_mask"].sum() > 1)

train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=cfg.batch_size)

# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
def distillation_loss(s_logits, t_logits, T):
    p_s = F.log_softmax(s_logits / T, dim=-1)
    p_t = F.softmax(t_logits / T, dim=-1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

def cross_entropy_loss(logits, targets):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=cfg.label_smoothing
    )

# 🔥 FIXED layer mapping
def hidden_state_loss(s_layers, t_layers):
    s_len = len(s_layers)
    t_len = len(t_layers)
    loss = 0.0

    for i in range(s_len):
        t_idx = int(i * (t_len - 1) / (s_len - 1))
        t_proj = projection(t_layers[t_idx].detach())
        loss += F.mse_loss(s_layers[i], t_proj)

    return loss / s_len

def attention_loss(s_attn, t_attn):
    s_len = len(s_attn)
    t_len = len(t_attn)
    loss = 0.0

    for i in range(s_len):
        t_idx = int(i * (t_len - 1) / (s_len - 1))

        sa = s_attn[i].mean(dim=1)
        ta = t_attn[t_idx].detach().mean(dim=1)

        sa = F.normalize(sa.view(sa.size(0), -1), dim=-1)
        ta = F.normalize(ta.view(ta.size(0), -1), dim=-1)

        loss += F.mse_loss(sa, ta)

    return loss / s_len

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def top_k_accuracy(logits, targets, k=5):
    preds = logits.topk(k, dim=-1).indices
    correct = preds.eq(targets.unsqueeze(-1)).any(-1)
    mask = targets != tokenizer.pad_token_id
    return (correct & mask).float().sum() / mask.sum()

def top_1_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    mask = targets != tokenizer.pad_token_id
    return ((preds == targets) & mask).float().sum() / mask.sum()

def perplexity(logits, targets):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=tokenizer.pad_token_id
    )
    return math.exp(loss.item())

# ─────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────
params = list(student.parameters()) + list(projection.parameters())
optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

total_steps = len(train_loader) * cfg.epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    int(total_steps * cfg.warmup_ratio),
    total_steps
)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
best_val = float("inf")
metrics = defaultdict(list)

for epoch in range(cfg.epochs):
    student.train()
    total_loss = 0

    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(bar):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            t_out = teacher(ids, attention_mask=mask)

        s_out = student(ids, attention_mask=mask)

        loss = (
            cfg.alpha * distillation_loss(s_out.logits, t_out.logits, cfg.temperature) +
            cfg.beta  * cross_entropy_loss(s_out.logits, ids) +
            cfg.gamma * hidden_state_loss(s_out.hidden_states, t_out.hidden_states) +
            cfg.delta * attention_loss(s_out.attentions, t_out.attentions)
        )

        loss = loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # ── VALIDATION ──
    student.eval()
    val_loss = val_acc1 = val_acc5 = val_ppl = 0

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            out = student(ids, attention_mask=mask)

            val_loss += cross_entropy_loss(out.logits, ids).item()
            val_acc1 += top_1_accuracy(out.logits, ids).item()
            val_acc5 += top_k_accuracy(out.logits, ids).item()
            val_ppl  += perplexity(out.logits, ids)

    n = len(val_loader)
    val_loss /= n
    val_acc1 /= n
    val_acc5 /= n
    val_ppl  /= n

    print(f"Val Loss={val_loss:.4f} | PPL={val_ppl:.2f} | Top1={val_acc1*100:.2f}% | Top5={val_acc5*100:.2f}%")

    if val_loss < best_val:
        best_val = val_loss
        student.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print("✓ Best model saved")

# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = student.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("\nSamples:")
print(generate("Artificial intelligence is"))
