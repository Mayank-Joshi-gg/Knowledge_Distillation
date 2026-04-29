Knowledge Distillation-based GPT Student Model Implementation

Description:
This project implements a knowledge distillation framework using a teacher-student architecture based on GPT models. The teacher model (GPT-2) transfers knowledge to a smaller student model (DistilGPT-2) using a combination of loss functions including distillation loss, cross-entropy loss, hidden-state alignment loss, and attention alignment loss. The pipeline includes dataset loading (WikiText-2), preprocessing, tokenization, training with gradient accumulation, and evaluation using perplexity, top-k accuracy, and top-1 accuracy. The trained student model is capable of efficient text generation with reduced computational cost while preserving much of the teacher’s performance.

PSEUDOCODE :-

BEGIN

1. Load Configuration
   - Set hyperparameters (epochs, batch size, learning rate, temperature, etc.)

2. Load Models
   - Load Teacher Model (pretrained GPT-2)
   - Load Student Model (DistilGPT-2)
   - Freeze selected layers if required
   - Create projection layer for hidden state alignment

3. Load Dataset
   - Download WikiText dataset
   - Tokenize text into input IDs and attention masks
   - Filter invalid/empty samples
   - Create DataLoader for training and validation

4. Define Loss Functions
   - Distillation Loss (KL Divergence between teacher and student logits)
   - Cross-Entropy Loss (student prediction vs ground truth)
   - Hidden State Loss (MSE between teacher and student hidden states)
   - Attention Loss (alignment between attention maps)

5. Initialize Optimizer and Scheduler

6. TRAINING LOOP:
   FOR each epoch:
       FOR each batch in training data:
           - Forward pass teacher (no gradient)
           - Forward pass student
           - Compute all losses
           - Combine weighted losses
           - Backpropagate gradients
           - Apply gradient clipping
           - Update optimizer and scheduler

7. VALIDATION LOOP:
   - Evaluate model on validation set
   - Compute loss, perplexity, top-1 accuracy, top-5 accuracy
   - Save best model

8. SAVE MODEL
   - Store trained student model and tokenizer

9. TEXT GENERATION
   - Take input prompt
   - Generate output using trained student model

END

