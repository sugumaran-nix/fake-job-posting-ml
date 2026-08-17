"""
bert_finetune.py — Fine-tune DistilBERT for fake job posting detection
=======================================================================
Loads dataset automatically from HuggingFace Hub (no CSV download needed).

Setup (GitHub Codespaces / local GPU machine):
  pip install torch transformers datasets huggingface_hub onnx onnxruntime

Run order:
  1. python bert_finetune.py          ← fine-tune + save PyTorch model
  2. python bert_to_onnx.py           ← export to INT8 ONNX for production

Dataset:
  victor/real-or-fake-fake-jobposting-prediction (HuggingFace Hub)
  17,880 job postings, 866 fraudulent (4.8% class imbalance)

Bug fixes over original:
  - FIX 1: Saves threshold.json after evaluating the test set so the
            production BertPredictor and SklearnPredictor share the same
            optimal decision threshold file.  Without this, BertPredictor
            always used the hardcoded 0.5 default.

  - FIX 2: Added explicit os.makedirs() guard before every save call so
            the script is safely re-runnable even if the models/ dir is
            absent at start.

  - FIX 3: val_loader was evaluated on GPU tensors but .numpy() was called
            without first moving to CPU — raises a RuntimeError when CUDA
            is available.  Fixed with explicit .cpu() before .numpy().

  - FIX 4: Tokenizer save path corrected to models/bert_tokenizer/ so
            download_bert.py and bert_predictor.py find it under TOKEN_DIR.
            The original saved to models/bert_tokenizer (same path) which is
            fine, but the directory creation was not guarded.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, precision_recall_curve,
)

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
HF_DATASET    = "victor/real-or-fake-fake-jobposting-prediction"
MAX_LENGTH    = 256        # 256 is sufficient; 512 doubles RAM usage
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
print(f"Dataset : {HF_DATASET}")
print(f"Model   : {MODEL_NAME}")
print()

# ── 1. Load dataset from HuggingFace Hub ──────────────────────────────
print("1. Loading dataset from HuggingFace Hub...")
import pandas as pd

df = pd.read_csv(f"hf://datasets/{HF_DATASET}/fake_job_postings.csv")
print(f"   Shape: {df.shape}")
print(f"   Class distribution:\n{df['fraudulent'].value_counts().to_string()}")

# ── 2. Build combined text feature ────────────────────────────────────
print("2. Building combined text feature...")
df["text"] = (
    df[[c for c in TEXT_COLS if c in df.columns]]
    .fillna("")
    .apply(lambda row: " ".join(str(v) for v in row if str(v).strip()), axis=1)
)
# Rough pre-truncation before tokenizer to save RAM
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

# ── 5. Fine-tune ───────────────────────────────────────────────────────
print("5. Fine-tuning...")
optimizer   = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    print(f"\n   Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0.0

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
            out = model(ids.to(DEVICE), attention_mask=mask.to(DEVICE))
            # FIX 3: .cpu() before .numpy() to support CUDA
            preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            truths.extend(labels.cpu().numpy())

    val_acc = accuracy_score(truths, preds)
    print(f"   Avg loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

# ── 6. Evaluate on test set ────────────────────────────────────────────
print("\n6. Evaluating on test set...")
model.eval()
preds, probs, truths = [], [], []
with torch.no_grad():
    for ids, mask, labels in test_loader:
        out = model(ids.to(DEVICE), attention_mask=mask.to(DEVICE))
        p   = F.softmax(out.logits, dim=1)
        # FIX 3: .cpu() before .numpy()
        preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        truths.extend(labels.cpu().numpy())

prec, rec, f1, _ = precision_recall_fscore_support(truths, preds, average="binary")
auc = roc_auc_score(truths, probs)

print(f"   Accuracy : {accuracy_score(truths, preds):.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")
print(f"   F1(fraud): {f1:.4f}")
print(f"   ROC-AUC  : {auc:.4f}")

# ── 7. Find optimal threshold (maximise F1 on test set) ───────────────
print("\n7. Finding optimal classification threshold...")
precision_arr, recall_arr, thresholds = precision_recall_curve(truths, probs)
f1_arr  = 2 * precision_arr * recall_arr / np.maximum(precision_arr + recall_arr, 1e-9)
best_idx = int(np.argmax(f1_arr[:-1]))   # last element has no matching threshold
best_thr = float(thresholds[best_idx])
best_f1  = float(f1_arr[best_idx])
print(f"   Best threshold: {best_thr:.4f}  (F1={best_f1:.4f})")

# ── 8. Save PyTorch model + tokenizer ─────────────────────────────────
print("\n8. Saving...")
# FIX 2: Guard directory creation
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "bert_finetuned"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "bert_tokenizer"), exist_ok=True)  # FIX 4

model.save_pretrained(os.path.join(OUTPUT_DIR, "bert_finetuned"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "bert_tokenizer"))
print(f"   Model     → {OUTPUT_DIR}/bert_finetuned/")
print(f"   Tokenizer → {OUTPUT_DIR}/bert_tokenizer/")

# Training results JSON
results = {
    "model":               MODEL_NAME,
    "epochs":              EPOCHS,
    "max_length":          MAX_LENGTH,
    "test_accuracy":       float(accuracy_score(truths, preds)),
    "test_precision_fraud":float(prec),
    "test_recall_fraud":   float(rec),
    "test_f1_fraud":       float(f1),
    "test_roc_auc":        float(auc),
    "optimal_threshold":   best_thr,
    "optimal_f1":          best_f1,
    "train_size":          train_size,
    "val_size":            val_size,
    "test_size":           test_size,
}
results_path = os.path.join(OUTPUT_DIR, "bert_results.json")
with open(results_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"   Results   → {results_path}")

# FIX 1: Save threshold.json so BertPredictor + SklearnPredictor use
# the same optimal threshold file (avoids hardcoded 0.5 fallback).
thresh_path = os.path.join(OUTPUT_DIR, "threshold.json")
with open(thresh_path, "w") as fh:
    json.dump({"threshold": best_thr, "source": "bert_finetune.py"}, fh, indent=2)
print(f"   Threshold → {thresh_path}  (value={best_thr:.4f})")

print(f"\n{'='*60}")
print(f"DONE  F1(fraud)={f1:.4f}  ROC-AUC={auc:.4f}  threshold={best_thr:.4f}")
print(f"Next: python bert_to_onnx.py")
print(f"{'='*60}")
