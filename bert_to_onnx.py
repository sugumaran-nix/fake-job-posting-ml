"""
bert_to_onnx.py — Convert fine-tuned DistilBERT to quantized ONNX
===================================================================
Run AFTER bert_finetune.py.

PyTorch DistilBERT: ~300MB (too heavy for Render free tier)
ONNX quantized:     ~80MB  (fits on 512MB Render free tier)

Usage:
  python bert_to_onnx.py

Output:
  models/bert_onnx.onnx           (full ONNX, ~240MB — deleted after quantization)
  models/bert_onnx_quantized.onnx (INT8 quantized, ~80MB) ← used in production
  models/bert_onnx_meta.json      (size, opset, max_length)
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR  = "models/bert_finetuned"
TOKEN_DIR  = "models/bert_tokenizer"
ONNX_PATH  = "models/bert_onnx.onnx"
QUANT_PATH = "models/bert_onnx_quantized.onnx"
MAX_LENGTH = 512

# ── 1. Load fine-tuned PyTorch model ──────────────────────────────────
print("1. Loading fine-tuned model...")
if not os.path.exists(MODEL_DIR):
    print(f"ERROR: {MODEL_DIR} not found. Run bert_finetune.py first.")
    exit(1)

model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKEN_DIR)
model.eval()
print(f"   Model loaded from {MODEL_DIR}")

# ── 2. Dummy input for ONNX tracing ───────────────────────────────────
print("2. Creating dummy input...")
dummy_text = (
    "Software Engineer position at Infosys Bangalore. "
    "Responsibilities include backend API development and code reviews. "
    "Requirements: 2 years Python experience, BSc CS degree. "
    "Salary 8 LPA. Apply with resume at careers.infosys.com."
)
enc = tokenizer(
    dummy_text,
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
dummy_ids  = enc["input_ids"]
dummy_mask = enc["attention_mask"]
print(f"   Input shape: {dummy_ids.shape}")

# ── 3. Export to ONNX ─────────────────────────────────────────────────
print("3. Exporting to ONNX (this takes ~60 seconds)...")
os.makedirs("models", exist_ok=True)

with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits":         {0: "batch_size"},
        },
    )

size_mb = os.path.getsize(ONNX_PATH) / 1e6
print(f"   Saved: {ONNX_PATH} ({size_mb:.1f} MB)")

# ── 4. INT8 quantization ───────────────────────────────────────────────
print("4. Quantizing to INT8 (~80MB target)...")
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input=ONNX_PATH,
    model_output=QUANT_PATH,
    weight_type=QuantType.QInt8,
)

quant_mb = os.path.getsize(QUANT_PATH) / 1e6
print(f"   Quantized: {QUANT_PATH} ({quant_mb:.1f} MB)")
print(f"   Reduction: {size_mb:.0f}MB → {quant_mb:.0f}MB "
      f"({(1 - quant_mb / size_mb) * 100:.0f}% smaller)")

# ── 5. Verify ONNX vs PyTorch output ──────────────────────────────────
print("5. Verifying ONNX output matches PyTorch...")
import onnxruntime as rt

sess     = rt.InferenceSession(QUANT_PATH, providers=["CPUExecutionProvider"])
onnx_out = sess.run(
    ["logits"],
    {"input_ids": dummy_ids.numpy(), "attention_mask": dummy_mask.numpy()},
)[0]

with torch.no_grad():
    pt_out = model(dummy_ids, attention_mask=dummy_mask).logits.numpy()

max_diff = float(np.abs(onnx_out - pt_out).max())
print(f"   Max logit diff (PT vs ONNX): {max_diff:.6f}")
if max_diff < 0.1:
    print("   Output matches within tolerance — OK")
else:
    print(f"   WARNING: diff {max_diff:.4f} is large. "
          "Try QuantType.QUInt8 if predictions diverge in production.")

# ── 6. Remove unquantized ONNX (saves ~160MB) ─────────────────────────
os.remove(ONNX_PATH)
print(f"   Removed full ONNX ({size_mb:.0f}MB) — keeping quantized only")

# ── 7. Save metadata ───────────────────────────────────────────────────
meta = {
    "onnx_path":      QUANT_PATH,
    "size_mb":        round(quant_mb, 1),
    "max_length":     MAX_LENGTH,
    "base_model":     "distilbert-base-uncased",
    "quantization":   "INT8 dynamic",
    "opset_version":  14,
    "max_diff_vs_pt": round(max_diff, 6),
}
with open("models/bert_onnx_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 55)
print("ONNX EXPORT COMPLETE")
print("=" * 55)
print(f"Production model : {QUANT_PATH} ({quant_mb:.1f} MB)")
print(f"Render free tier : 512MB RAM — {quant_mb:.0f}MB model "
      f"+ ~60MB runtime = ~{quant_mb + 60:.0f}MB total — fits OK")
print("Next step        : python -c \"from utils.bert_predictor "
      "import BertPredictor; p=BertPredictor(); print(p.predict('test'))\"")
