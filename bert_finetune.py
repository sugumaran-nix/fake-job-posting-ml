"""
bert_finetune.py — Fine-tune DistilBERT (zero CSV, loads from HuggingFace Hub)
================================================================================
Run in GitHub Codespaces — no dataset download needed.

Setup:
  pip install torch transformers datasets huggingface_hub onnx onnxruntime

Then run:
  python bert_finetune.py
  python bert_to_onnx.py   ← converts + pushes to your HF model repo

Dataset loaded automatically from:
  victor/real-or-fake-fake-jobposting-prediction (HuggingFace Hub)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
HF_DATASET    = "victor/real-or-fake-fake-jobposting-prediction"
MAX_LENGTH    = 256          # 256 sufficient for job text; 512 doubles RAM usage
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR    = "models"

TEXT_COLS = [
    "title", "company_profile", "description",
    "requirements", "benefits", "location",
    "employment_type", "required_experience",
    "required_education", "industry", "function",
]

print(f"Device  : {DEVICE}")
print(f"Dataset : {HF_DATASET} (loading from HuggingFace Hub — no CSV needed)")
print(f"Model   : {MODEL_NAME}")
print()

# ── 1. Load dataset from HuggingFace Hub ──────────────────────────────
print("1. Loading dataset from HuggingFace Hub...")
import pandas as pd

df = pd.read_csv(
    f"hf://datasets/{HF_DATASET}/fake_job_postings.csv"
)
print(f"   Shape: {df.shape}")
print(f"   Class distribution:\n{df['fraudulent'].value_counts().to_string()}")

# ── 2. Build combined text feature ────────────────────────────────────
print("2. Building combined text feature...")
df["text"] = (
    df[[c for c in TEXT_COLS if c in df.columns]]
    .fillna("")
    .apply(lambda row: " ".join(str(v) for v in row if str(v).strip()), axis=1)
)
# Rough truncation before tokenizer (save RAM)
df["text"] = df["text"].str[:int(MAX_LENGTH * 4)]

X = df["text"].values
y = df["fraudulent"].values
print(f"   Samples: {len(X)}  |  Fraud: {y.sum()}  |  Legit: {(y==0).sum()}")

# ── 3. Tokenize ────────────────────────────────────────────────────────
print("3. Tokenizing (this takes ~2 minutes on CPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encodings = tokenizer(
    X.tolist(),
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
print(f"   Input shape: {encodings['input_ids'].shape}")

dataset = TensorDataset(
    encodings["input_ids"],
    encodings["attention_mask"],
    torch.tensor(y, dtype=torch.long),
)

train_size = int(0.70 * len(dataset))
val_size   = int(0.10 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)
print(f"   Train: {train_size}  Val: {val_size}  Test: {test_size}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── 4. Load model ──────────────────────────────────────────────────────
print("4. Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
print(f"   Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ── 5. Train ───────────────────────────────────────────────────────────
print("5. Fine-tuning...")
optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps  = len(train_loader) * EPOCHS
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    print(f"\n   Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for i, (ids, mask, labels) in enumerate(train_loader):
        ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(ids, attention_mask=mask, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"   Batch {i+1}/{len(train_loader)}  loss={loss.item():.4f}")

    # Validation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for ids, mask, labels in val_loader:
            out   = model(ids.to(DEVICE), attention_mask=mask.to(DEVICE))
            preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            truths.extend(labels.numpy())

    val_acc = accuracy_score(truths, preds)
    print(f"   Avg loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

# ── 6. Evaluate on test set ────────────────────────────────────────────
print("\n6. Evaluating on test set...")
model.eval()
preds, probs, truths = [], [], []
with torch.no_grad():
    for ids, mask, labels in test_loader:
        out   = model(ids.to(DEVICE), attention_mask=mask.to(DEVICE))
        p     = F.softmax(out.logits, dim=1)
        preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        truths.extend(labels.numpy())

prec, rec, f1, _ = precision_recall_fscore_support(truths, preds, average="binary")
auc = roc_auc_score(truths, probs)

print(f"   Accuracy : {accuracy_score(truths, preds):.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   F1 (fraud): {f1:.4f}")
print(f"   ROC-AUC  : {auc:.4f}")

# ── 7. Save PyTorch model + tokenizer ─────────────────────────────────
print("\n7. Saving...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(f"{OUTPUT_DIR}/bert_finetuned")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/bert_tokenizer")

results = {
    "model": MODEL_NAME, "epochs": EPOCHS, "max_length": MAX_LENGTH,
    "test_accuracy": float(accuracy_score(truths, preds)),
    "test_precision_fraud": float(prec), "test_recall_fraud": float(rec),
    "test_f1_fraud": float(f1), "test_roc_auc": float(auc),
    "train_size": train_size, "val_size": val_size, "test_size": test_size,
}
with open(f"{OUTPUT_DIR}/bert_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*55}")
print(f"DONE — F1(fraud)={f1:.4f}  ROC-AUC={auc:.4f}")
print(f"Next: python bert_to_onnx.py")
print(f"{'='*55}")
