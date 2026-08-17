"""
download_bert.py — Download ONNX model from HuggingFace Hub at Render startup
==============================================================================
Called by CMD in Dockerfile before gunicorn starts.
Env vars from Render are available at this point.

Required env var:
  HF_MODEL_REPO = your-username/jobguard-bert

Optional:
  HF_TOKEN = hf_xxxxx  (only if repo is private)

Bug fixes:
  - FIX 1: Removed download of bert_onnx_quantized.onnx.data — INT8 dynamic
            quantization (quantize_dynamic) produces a single self-contained
            .onnx file.  Attempting to download the non-existent .data sidecar
            caused an HTTP 404 from HF Hub, which was swallowed and then
            triggered a misleading "files incomplete" warning, making the app
            fall back to sklearn even when the ONNX model was fully present.

  - FIX 2: Completeness check now only verifies the single ONNX file size
            (≥ 30 MB for quantized DistilBERT) instead of requiring both
            .onnx + .onnx.data totalling > 50 MB.

  - FIX 3: Added vocab.txt as an optional tokenizer file to download so the
            HuggingFace tokenizer loads without hitting the network at runtime.
"""

import os
import sys

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()

if not HF_MODEL_REPO:
    print("[download_bert] HF_MODEL_REPO not set — skipping. App will use sklearn fallback.")
    sys.exit(0)

print(f"[download_bert] Downloading BERT model from: {HF_MODEL_REPO}")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[download_bert] huggingface_hub not installed. "
          "Add 'huggingface-hub>=0.20.0' to requirements.txt")
    sys.exit(0)

os.makedirs("models", exist_ok=True)
os.makedirs("models/bert_tokenizer", exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

# FIX 1: Only one ONNX file — no .data sidecar with quantize_dynamic().
# Tokenizer files: tokenizer_config.json + tokenizer.json cover DistilBERT
# (WordPiece). vocab.txt is optional but prevents a network fetch at runtime.
FILES_TO_DOWNLOAD = [
    # (filename on HF Hub,                              local destination)
    ("bert_onnx_quantized.onnx",                        "models/bert_onnx_quantized.onnx"),
    ("bert_onnx_meta.json",                             "models/bert_onnx_meta.json"),
    ("bert_results.json",                               "models/bert_results.json"),
    ("bert_tokenizer/tokenizer_config.json",            "models/bert_tokenizer/tokenizer_config.json"),
    ("bert_tokenizer/tokenizer.json",                   "models/bert_tokenizer/tokenizer.json"),
    ("bert_tokenizer/vocab.txt",                        "models/bert_tokenizer/vocab.txt"),     # FIX 3
    ("bert_tokenizer/special_tokens_map.json",          "models/bert_tokenizer/special_tokens_map.json"),
]

# Files that are truly optional (HF Hub 404 is acceptable)
OPTIONAL_FILES = {
    "models/bert_onnx_meta.json",
    "models/bert_results.json",
    "models/bert_tokenizer/vocab.txt",
    "models/bert_tokenizer/special_tokens_map.json",
}

download_ok = True

for remote_path, local_path in FILES_TO_DOWNLOAD:
    # Skip if already present (avoid re-downloading on container restart)
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"[download_bert] Already exists: {local_path} ({size_mb:.1f} MB)")
        continue

    print(f"[download_bert] Downloading {remote_path} ...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=remote_path,
            repo_type="model",
            token=HF_TOKEN,
            local_dir="models",
        )
        size_mb = os.path.getsize(downloaded) / 1e6
        print(f"[download_bert] Saved: {downloaded} ({size_mb:.1f} MB)")
    except Exception as e:
        if local_path in OPTIONAL_FILES:
            print(f"[download_bert] Optional file not found (skipping): {remote_path} — {e}")
        else:
            print(f"[download_bert] ERROR downloading required file {remote_path}: {e}")
            download_ok = False

# ── FIX 2: Verify completeness using single ONNX file ─────────────────
onnx_path = "models/bert_onnx_quantized.onnx"
onnx_ok   = os.path.exists(onnx_path)
onnx_mb   = os.path.getsize(onnx_path) / 1e6 if onnx_ok else 0.0

print(f"[download_bert] onnx={onnx_ok} size={onnx_mb:.1f} MB")

if onnx_ok and onnx_mb >= 30 and download_ok:
    print("[download_bert] BERT model ready — app will use BertPredictor.")
else:
    if not onnx_ok:
        print("[download_bert] WARNING: bert_onnx_quantized.onnx not found.")
    elif onnx_mb < 30:
        print(f"[download_bert] WARNING: ONNX file looks too small ({onnx_mb:.1f} MB < 30 MB).")
    print("[download_bert] App will fall back to sklearn predictor.")
