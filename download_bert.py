"""
download_bert.py — Download ONNX model from HuggingFace Hub at Render startup
==============================================================================
Called by CMD in Dockerfile before gunicorn starts.
Env vars from Render are available at this point.

Required env var:
  HF_MODEL_REPO = Sugum4r4n/jobguard-bert

Optional:
  HF_TOKEN = hf_xxxxx  (only if repo is private)
"""

import os
import sys

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()

if not HF_MODEL_REPO:
    print("[download_bert] HF_MODEL_REPO not set — skipping. Using sklearn fallback.")
    sys.exit(0)

print(f"[download_bert] Downloading BERT model from: {HF_MODEL_REPO}")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[download_bert] huggingface_hub not installed. Add 'huggingface-hub' to requirements.txt")
    sys.exit(0)

os.makedirs("models", exist_ok=True)
os.makedirs("models/bert_tokenizer", exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

# Files to download: both ONNX files are required
FILES_TO_DOWNLOAD = [
    # (filename on HF Hub, local path)
    ("bert_onnx_quantized.onnx",      "models/bert_onnx_quantized.onnx"),
    ("bert_onnx_quantized.onnx.data", "models/bert_onnx_quantized.onnx.data"),
    ("bert_onnx_meta.json",           "models/bert_onnx_meta.json"),
    ("bert_results.json",             "models/bert_results.json"),
    ("bert_tokenizer/tokenizer_config.json", "models/bert_tokenizer/tokenizer_config.json"),
    ("bert_tokenizer/tokenizer.json",        "models/bert_tokenizer/tokenizer.json"),
]

try:
    for remote_path, local_path in FILES_TO_DOWNLOAD:
        # Skip if already downloaded (avoid re-downloading on restart)
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"[download_bert] Already exists: {local_path} ({size_mb:.1f} MB)")
            continue

        print(f"[download_bert] Downloading {remote_path} ...")
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=remote_path,
            repo_type="model",
            token=HF_TOKEN,
            local_dir="models",
        )
        size_mb = os.path.getsize(downloaded) / 1e6
        print(f"[download_bert] Saved: {downloaded} ({size_mb:.1f} MB)")

    # Verify both ONNX files present
    onnx      = os.path.exists("models/bert_onnx_quantized.onnx")
    onnx_data = os.path.exists("models/bert_onnx_quantized.onnx.data")
    total_mb  = (
        os.path.getsize("models/bert_onnx_quantized.onnx") / 1e6 +
        os.path.getsize("models/bert_onnx_quantized.onnx.data") / 1e6
    ) if onnx and onnx_data else 0

    print(f"[download_bert] onnx={onnx} onnx.data={onnx_data} total={total_mb:.1f} MB")

    if onnx and onnx_data and total_mb > 50:
        print("[download_bert] BERT model ready — app will use BertPredictor.")
    else:
        print("[download_bert] WARNING: files incomplete — falling back to sklearn.")

except Exception as e:
    print(f"[download_bert] Download failed: {e}")
    print("[download_bert] App will use sklearn fallback.")
