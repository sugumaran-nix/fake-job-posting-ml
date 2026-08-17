"""
bert_to_onnx.py — Convert fine-tuned DistilBERT to ONNX + push to HuggingFace Hub
====================================================================================
Run AFTER bert_finetune.py.

Set HF_MODEL_REPO before running (optional — you can also keep the ONNX local):
  export HF_MODEL_REPO=your-username/jobguard-bert
  python bert_to_onnx.py

The ONNX model is pushed to your HF Hub model repo so Render can
download it at build time — no large files in GitHub needed.

Output:
  models/bert_onnx_quantized.onnx  (single self-contained file, ~80 MB)

Bug fixes:
  - FIX 1: quantize_dynamic() with QInt8 writes a single self-contained
            .onnx file.  The old script attempted to verify that a
            separate .onnx.data sidecar existed and remove the unquantized
            file before finalising; both operations are now sequenced
            correctly (export → quantize → verify → remove temp).

  - FIX 2: os.remove(ONNX_PATH) is now wrapped in try/except so a
            missing temp file (e.g. on a re-run) does not abort the script.

  - FIX 3: Metadata 'has_data_sidecar' key removed — the sidecar no
            longer exists; keeping a false key would mislead download_bert.py.

  - FIX 4: HuggingFace Hub upload now only uploads the single ONNX file
            (not a non-existent .data sidecar).
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR  = "models/bert_finetuned"
TOKEN_DIR  = "models/bert_tokenizer"
ONNX_PATH  = "models/bert_onnx.onnx"           # temporary full-precision export
QUANT_PATH = "models/bert_onnx_quantized.onnx"  # final INT8 output (single file)
MAX_LENGTH = 256

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()

# ── 1. Load fine-tuned model ───────────────────────────────────────────
print("1. Loading fine-tuned model...")
if not os.path.exists(MODEL_DIR):
    print(f"ERROR: {MODEL_DIR} not found. Run bert_finetune.py first.")
    raise SystemExit(1)

model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKEN_DIR)
model.eval()
print(f"   Loaded from {MODEL_DIR}")

# ── 2. Dummy input for ONNX tracing ───────────────────────────────────
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

# ── 3. Export to ONNX (full precision, temporary) ─────────────────────
print("3. Exporting to ONNX (~60 seconds on CPU)...")
os.makedirs("models", exist_ok=True)
with torch.no_grad():
    torch.onnx.export(
        model,
        (ids, mask),
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
print(f"   Saved temp: {ONNX_PATH} ({size_mb:.1f} MB)")

# ── 4. INT8 dynamic quantisation ──────────────────────────────────────
print("4. Quantizing to INT8 (quantize_dynamic → single self-contained file)...")
from onnxruntime.quantization import quantize_dynamic, QuantType

# FIX 1: quantize_dynamic produces one output file — no .data sidecar.
quantize_dynamic(ONNX_PATH, QUANT_PATH, weight_type=QuantType.QInt8)
quant_mb = os.path.getsize(QUANT_PATH) / 1e6
print(f"   Quantized: {QUANT_PATH} ({quant_mb:.1f} MB)")
print(f"   Size reduction: {size_mb:.0f} MB → {quant_mb:.0f} MB")

# ── 5. Verify quantised model output ──────────────────────────────────
print("5. Verifying ONNX output against PyTorch...")
import onnxruntime as rt

sess = rt.InferenceSession(QUANT_PATH, providers=["CPUExecutionProvider"])
onnx_out = sess.run(
    ["logits"],
    {"input_ids": ids.numpy(), "attention_mask": mask.numpy()},
)[0]
with torch.no_grad():
    pt_out = model(ids, attention_mask=mask).logits.numpy()
diff = float(np.abs(onnx_out - pt_out).max())
print(f"   Max abs diff (PyTorch vs ONNX): {diff:.6f} "
      f"{'✓ OK' if diff < 0.5 else '⚠ LARGE DIFF — check quantization'}")

# ── 6. FIX 2: Remove temporary full-precision ONNX ────────────────────
try:
    os.remove(ONNX_PATH)
    print(f"   Removed temp file: {ONNX_PATH} ({size_mb:.0f} MB freed)")
except FileNotFoundError:
    pass   # already gone — safe on re-run

# ── 7. Save metadata ───────────────────────────────────────────────────
# FIX 3: 'has_data_sidecar' key removed — dynamic INT8 is single-file.
meta = {
    "onnx_path":        QUANT_PATH,
    "size_mb":          round(quant_mb, 1),
    "max_length":       MAX_LENGTH,
    "base_model":       "distilbert-base-uncased",
    "quantization":     "INT8 dynamic (quantize_dynamic)",
    "opset_version":    14,
    "max_diff_vs_pt":   round(diff, 6),
}
meta_path = "models/bert_onnx_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"   Metadata saved: {meta_path}")

# ── 8. Push to HuggingFace Hub (optional) ─────────────────────────────
if HF_MODEL_REPO:
    print(f"\n8. Pushing to HuggingFace Hub: {HF_MODEL_REPO} ...")
    try:
        from huggingface_hub import HfApi, login

        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            login(token=hf_token)
        else:
            print("   HF_TOKEN not set — attempting anonymous push "
                  "(will fail for private repos)")

        api = HfApi()

        try:
            api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
            print(f"   Repo: https://huggingface.co/{HF_MODEL_REPO}")
        except Exception as e:
            print(f"   Repo already exists or error: {e}")

        # FIX 4: Only upload the single ONNX file (no .data sidecar)
        files_to_upload = [
            (QUANT_PATH,   "bert_onnx_quantized.onnx"),
            (meta_path,    "bert_onnx_meta.json"),
        ]
        # Also upload bert_results.json if present
        results_path = "models/bert_results.json"
        if os.path.exists(results_path):
            files_to_upload.append((results_path, "bert_results.json"))

        for local_path, remote_name in files_to_upload:
            if not os.path.exists(local_path):
                print(f"   Skipping missing: {local_path}")
                continue
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            print(f"   Uploaded: {remote_name}")

        # Upload tokenizer files
        tok_dir = "models/bert_tokenizer"
        if os.path.isdir(tok_dir):
            for fname in os.listdir(tok_dir):
                fpath = os.path.join(tok_dir, fname)
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"bert_tokenizer/{fname}",
                    repo_id=HF_MODEL_REPO,
                    repo_type="model",
                )
            print(f"   Uploaded: bert_tokenizer/")

        print(f"\n   SUCCESS — model at https://huggingface.co/{HF_MODEL_REPO}")
        print(f"   Set HF_MODEL_REPO in Render and run download_bert.py at startup.")

    except Exception as e:
        print(f"   HF push failed: {e}")
        print(f"   Manually upload {QUANT_PATH} to HuggingFace Hub.")
else:
    print("\n8. Skipping HF Hub push (HF_MODEL_REPO not set)")
    print(f"   Set: export HF_MODEL_REPO=your-username/jobguard-bert")
    print(f"       export HF_TOKEN=hf_xxxxx")
    print(f"   Then rerun this script to push.")
    print(f"\n   Alternative: git lfs track '*.onnx' && git add {QUANT_PATH}")
    print(f"   ({quant_mb:.0f} MB — check your Git LFS quota before committing)")

print(f"\n{'='*60}")
print(f"DONE — {QUANT_PATH}  ({quant_mb:.1f} MB, single file)")
print(f"{'='*60}")
