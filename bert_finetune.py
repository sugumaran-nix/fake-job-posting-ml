"""
bert_finetune.py — Fine-tune DistilBERT on fake job postings
==============================================================

Usage (GitHub Codespaces, has 32GB):
  1. pip install torch transformers datasets scikit-learn onnx onnxruntime
  2. Put fake_job_postings.csv in data/
  3. python bert_finetune.py

Output:
  - models/bert_finetuned.pt (PyTorch weights)
  - models/bert_tokenizer/ (saved tokenizer)
  - models/bert_onnx_quantized.onnx (for inference on Render)
  - bert_results.json (metrics: accuracy, F1-fraud, etc.)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512                    # BERT max sequence length
BATCH_SIZE = 16                     # Free tier: fit in ~6GB VRAM
EPOCHS = 3                          # Usually 2–3 for fine-tuning
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/fake_job_postings.csv"
OUTPUT_DIR = "models"

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")


# ── Load & preprocess data ─────────────────────────────────────────────
print("\n1. Loading data...")
if not os.path.exists(DATA_PATH):
    print(f"❌ {DATA_PATH} not found. Download from Kaggle.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Class distribution:\n{df['fraudulent'].value_counts()}")

# Combine text columns
TEXT_COLS = ["title", "company_profile", "description", "requirements",
             "benefits", "location", "employment_type", "required_experience",
             "required_education", "industry", "function"]

df["text"] = df[[c for c in TEXT_COLS if c in df.columns]].fillna("").apply(
    lambda row: " ".join(str(v) for v in row if str(v).strip()), axis=1
)

# Truncate to MAX_LENGTH tokens (rough estimate: words × 1.3)
df["text"] = df["text"].str[:int(MAX_LENGTH * 1.3)]

X = df["text"].values
y = df["fraudulent"].values

print(f"Text samples: {len(X)}")
print(f"Label distribution: {np.bincount(y)}")


# ── Tokenize ───────────────────────────────────────────────────────────
print("\n2. Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    X.tolist(),
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

print(f"Input IDs shape: {encodings['input_ids'].shape}")
print(f"Attention mask shape: {encodings['attention_mask'].shape}")

# Create dataset
dataset = TensorDataset(
    encodings["input_ids"],
    encodings["attention_mask"],
    torch.tensor(y, dtype=torch.long),
)

# Train/val/test split: 70/10/20
train_size = int(0.70 * len(dataset))
val_size = int(0.10 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# ── Load model ─────────────────────────────────────────────────────────
print("\n3. Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


# ── Training ───────────────────────────────────────────────────────────
print("\n4. Fine-tuning...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    # Train
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"  Avg Train Loss: {avg_loss:.4f}")
    
    # Validate
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"  Avg Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")


# ── Evaluate on test set ───────────────────────────────────────────────
print("\n5. Evaluating on test set...")
model.eval()
test_preds = []
test_probs = []
test_labels = []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (fraud)
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="binary"
)
roc_auc = roc_auc_score(test_labels, test_probs)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Precision (Fraud): {precision:.4f}")
print(f"Recall (Fraud): {recall:.4f}")
print(f"F1 (Fraud): {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

results = {
    "model": MODEL_NAME,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "test_accuracy": float(test_acc),
    "test_precision_fraud": float(precision),
    "test_recall_fraud": float(recall),
    "test_f1_fraud": float(f1),
    "test_roc_auc": float(roc_auc),
    "train_size": train_size,
    "val_size": val_size,
    "test_size": test_size,
}


# ── Save model ─────────────────────────────────────────────────────────
print("\n6. Saving...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PyTorch weights
model.save_pretrained(os.path.join(OUTPUT_DIR, "bert_finetuned"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "bert_tokenizer"))
print(f"✓ Saved PyTorch model to {OUTPUT_DIR}/bert_finetuned")

# Results
with open(os.path.join(OUTPUT_DIR, "bert_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results to {OUTPUT_DIR}/bert_results.json")

print("\n" + "=" * 60)
print("FINE-TUNING COMPLETE")
print("=" * 60)
print(f"Model F1 (Fraud class): {f1:.4f}")
print(f"Next: Convert to ONNX with bert_to_onnx.py")
print(f"Then: Update app.py ml_predict() to use BERT")
