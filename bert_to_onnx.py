"""
bert_to_onnx.py — Convert fine-tuned DistilBERT to ONNX + push to HuggingFace Hub
====================================================================================
Run AFTER bert_finetune.py.

Set HF_MODEL_REPO before running:
  export HF_MODEL_REPO=your-username/jobguard-bert
  python bert_to_onnx.py

The ONNX model is pushed to your HF Hub model repo so Render can
download it at build time — no large files in GitHub needed.

Output:
  models/bert_onnx_quantized.onnx  (local, ~80MB)
  HuggingFace Hub: {HF_MODEL_REPO}/bert_onnx_quantized.onnx
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
MAX_LENGTH = 256

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()

# ── 1. Load model ──────────────────────────────────────────────────────
print("1. Loading fine-tuned model...")
if not os.path.exists(MODEL_DIR):
    print(f"ERROR: {MODEL_DIR} not found. Run bert_finetune.py first.")
    exit(1)

model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKEN_DIR)
model.eval()
print(f"   Loaded from {MODEL_DIR}")

# ── 2. Dummy input ─────────────────────────────────────────────────────
print("2. Creating dummy input...")
dummy = (
    "Software Engineer position at Infosys Bangalore. Responsibilities include "
    "REST API development and code reviews. Requirements: 2 years Python, BSc CS. "
    "Salary 8 LPA. Apply at careers.infosys.com with resume."
)
enc  = tokenizer(dummy, max_length=MAX_LENGTH, padding="max_length",
                 truncation=True, return_tensors="pt")
ids  = enc["input_ids"]
mask = enc["attention_mask"]

# ── 3. Export to ONNX ─────────────────────────────────────────────────
print("3. Exporting to ONNX (~60 seconds)...")
os.makedirs("models", exist_ok=True)
with torch.no_grad():
    torch.onnx.export(
        model, (ids, mask), ONNX_PATH,
        export_params=True, opset_version=14,
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

# ── 4. INT8 quantize ───────────────────────────────────────────────────
print("4. Quantizing to INT8...")
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(ONNX_PATH, QUANT_PATH, weight_type=QuantType.QInt8)
quant_mb = os.path.getsize(QUANT_PATH) / 1e6
print(f"   Quantized: {QUANT_PATH} ({quant_mb:.1f} MB)")
print(f"   Reduction: {size_mb:.0f}MB → {quant_mb:.0f}MB")

# ── 5. Verify ──────────────────────────────────────────────────────────
print("5. Verifying output...")
import onnxruntime as rt
sess     = rt.InferenceSession(QUANT_PATH, providers=["CPUExecutionProvider"])
onnx_out = sess.run(["logits"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
with torch.no_grad():
    pt_out = model(ids, attention_mask=mask).logits.numpy()
diff = float(np.abs(onnx_out - pt_out).max())
print(f"   Max diff (PT vs ONNX): {diff:.6f} {'OK' if diff < 0.1 else 'WARNING: large diff'}")

os.remove(ONNX_PATH)
print(f"   Removed full ONNX ({size_mb:.0f}MB)")

# ── 6. Save metadata ───────────────────────────────────────────────────
meta = {
    "onnx_path": QUANT_PATH, "size_mb": round(quant_mb, 1),
    "max_length": MAX_LENGTH, "base_model": "distilbert-base-uncased",
    "quantization": "INT8 dynamic", "opset_version": 14,
    "max_diff_vs_pt": round(diff, 6),
}
with open("models/bert_onnx_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ── 7. Push to HuggingFace Hub ─────────────────────────────────────────
if HF_MODEL_REPO:
    print(f"\n6. Pushing to HuggingFace Hub: {HF_MODEL_REPO} ...")
    try:
        from huggingface_hub import HfApi, login

        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            login(token=hf_token)
        else:
            print("   HF_TOKEN not set — attempting anonymous push (may fail for private repos)")

        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
            print(f"   Repo: https://huggingface.co/{HF_MODEL_REPO}")
        except Exception as e:
            print(f"   Repo already exists or error: {e}")

        # Upload files
        for local_path, remote_path in [
            (QUANT_PATH,                   "bert_onnx_quantized.onnx"),
            ("models/bert_onnx_meta.json", "bert_onnx_meta.json"),
            ("models/bert_results.json",   "bert_results.json"),
        ]:
            if os.path.exists(local_path):
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=HF_MODEL_REPO,
                    repo_type="model",
                )
                print(f"   Uploaded: {remote_path}")

        # Also upload tokenizer files
        for fname in os.listdir("models/bert_tokenizer"):
            api.upload_file(
                path_or_fileobj=f"models/bert_tokenizer/{fname}",
                path_in_repo=f"bert_tokenizer/{fname}",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
        print(f"   Uploaded: bert_tokenizer/")

        print(f"\n   SUCCESS — model at https://huggingface.co/{HF_MODEL_REPO}")
        print(f"   Render will download it via download_bert.py at build time.")

    except Exception as e:
        print(f"   HF push failed: {e}")
        print(f"   Manually upload {QUANT_PATH} to HuggingFace Hub.")
else:
    print("\n6. Skipping HF Hub push (HF_MODEL_REPO not set)")
    print(f"   Set: export HF_MODEL_REPO=your-username/jobguard-bert")
    print(f"   Then: export HF_TOKEN=hf_xxxxx")
    print(f"   Then rerun this script to push.")
    print(f"\n   Alternative: commit {QUANT_PATH} to GitHub directly")
    print(f"   ({quant_mb:.0f}MB is under GitHub's 100MB limit)")

print(f"\n{'='*55}")
print(f"ONNX COMPLETE — {quant_mb:.1f}MB at {QUANT_PATH}")
print(f"{'='*55}")
