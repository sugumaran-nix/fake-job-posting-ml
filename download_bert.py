"""
download_bert.py — Download ONNX model from HuggingFace Hub at Render build time
==================================================================================
Called by Render build command ONLY when HF_MODEL_REPO is set.
If HF_MODEL_REPO is not set, the app falls back to sklearn automatically.

Render build command (in render.yaml / Render dashboard):
  pip install -r requirements.txt && python nltk_setup.py && python download_bert.py

Environment variables needed on Render:
  HF_MODEL_REPO = your-username/jobguard-bert   (required for BERT)
  HF_TOKEN      = hf_xxxxx                       (only if repo is private)

If HF_MODEL_REPO is not set, this script exits silently and the app
uses sklearn (Linear SVM) as the prediction backend.
"""

import os
import sys

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()

if not HF_MODEL_REPO:
    print("[download_bert] HF_MODEL_REPO not set — skipping BERT download.")
    print("[download_bert] App will use sklearn backend (Linear SVM).")
    sys.exit(0)

print(f"[download_bert] Downloading BERT model from: {HF_MODEL_REPO}")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("[download_bert] huggingface_hub not installed. "
          "Add 'huggingface-hub' to requirements.txt")
    sys.exit(0)

os.makedirs("models", exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

try:
    # Download ONNX model
    print("[download_bert] Downloading bert_onnx_quantized.onnx ...")
    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="bert_onnx_quantized.onnx",
        repo_type="model",
        token=HF_TOKEN,
        local_dir="models",
    )
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"[download_bert] ONNX saved: {local_path} ({size_mb:.1f} MB)")

    # Download metadata
    for fname in ["bert_onnx_meta.json", "bert_results.json"]:
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=fname,
                repo_type="model",
                token=HF_TOKEN,
                local_dir="models",
            )
            print(f"[download_bert] Downloaded: {fname}")
        except Exception:
            pass  # optional files

    # Download tokenizer
    print("[download_bert] Downloading tokenizer ...")
    os.makedirs("models/bert_tokenizer", exist_ok=True)

    # tokenizer files
    for fname in [
        "bert_tokenizer/tokenizer_config.json",
        "bert_tokenizer/vocab.txt",
        "bert_tokenizer/tokenizer.json",
        "bert_tokenizer/special_tokens_map.json",
    ]:
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=fname,
                repo_type="model",
                token=HF_TOKEN,
                local_dir="models",
            )
        except Exception:
            pass

    # Verify tokenizer exists
    tok_files = os.listdir("models/bert_tokenizer")
    print(f"[download_bert] Tokenizer files: {tok_files}")

    print("[download_bert] BERT model ready — app will use BertPredictor backend.")

except Exception as e:
    print(f"[download_bert] Download failed: {e}")
    print("[download_bert] App will fall back to sklearn backend.")
    # Don't exit(1) — let the app start with sklearn fallback
