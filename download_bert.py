import os, sys
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
if not HF_MODEL_REPO:
    print("[download_bert] HF_MODEL_REPO not set — skipping."); sys.exit(0)
print(f"[download_bert] Downloading from: {HF_MODEL_REPO}")
from huggingface_hub import snapshot_download
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
os.makedirs("models/bert_finetuned", exist_ok=True)
os.makedirs("models/bert_tokenizer", exist_ok=True)
try:
    snapshot_download(repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN,
                      local_dir="models", ignore_patterns=["*.onnx", "*.onnx.data"])
    print("[download_bert] Download complete.")
except Exception as e:
    print(f"[download_bert] ERROR: {e}"); sys.exit(1)
model_ok = os.path.exists("models/bert_finetuned/config.json")
token_ok  = os.path.exists("models/bert_tokenizer/tokenizer.json")
if model_ok and token_ok:
    print("[download_bert] BERT model ready.")
else:
    print(f"[download_bert] WARNING: model_ok={model_ok} token_ok={token_ok}")
