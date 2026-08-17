import os, json, logging, numpy as np
logger = logging.getLogger(__name__)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "bert_finetuned")
TOKEN_DIR   = os.path.join(BASE_DIR, "models", "bert_tokenizer")
THRESH_PATH = os.path.join(BASE_DIR, "models", "threshold.json")

class BertPredictor:
    def __init__(self):
        self._model=None; self._tokenizer=None
        self._max_len=256; self._threshold=0.5
        self._model_name="DistilBERT (Fine-tuned)"
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_DIR):
            raise RuntimeError(f"PyTorch model not found: {MODEL_DIR}")
        if not os.path.exists(TOKEN_DIR):
            raise RuntimeError(f"Tokenizer not found: {TOKEN_DIR}")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._device=torch.device("cpu")
        self._tokenizer=AutoTokenizer.from_pretrained(TOKEN_DIR)
        self._model=AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self._model.to(self._device); self._model.eval()
        if os.path.exists(THRESH_PATH):
            try:
                with open(THRESH_PATH) as f:
                    self._threshold=float(json.load(f).get("threshold", 0.5))
            except: pass

    def predict(self, text):
        if not self._model: return {"error": "Model not initialised."}
        try:
            import torch, torch.nn.functional as F
            enc=self._tokenizer(text, max_length=self._max_len, padding="max_length",
                                truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits=self._model(enc["input_ids"].to(self._device),
                                   attention_mask=enc["attention_mask"].to(self._device)).logits
                probs=F.softmax(logits, dim=1)[0].cpu().numpy()
        except Exception as exc:
            return {"error": f"Inference error: {exc}"}
        fraud_prob=float(probs[1]); legit_prob=float(probs[0])
        is_fraud=fraud_prob>=self._threshold
        confidence=round(max(fraud_prob, legit_prob)*100, 2)
        return {"label":"Fraudulent" if is_fraud else "Legitimate","is_fraud":is_fraud,
                "confidence":confidence,"fraud_prob":round(fraud_prob*100,2),
                "legit_prob":round(legit_prob*100,2),"model_name":self._model_name,
                "explanation":self._explain(text, fraud_prob, is_fraud)}

    def _explain(self, text, fraud_prob, is_fraud):
        try:
            from utils.explainer import _match_patterns, _build_reasons
            patterns=_match_patterns(text)
            reasons=_build_reasons([], [], patterns, fraud_prob, is_fraud)
        except: patterns=[]; reasons=[]
        reasons.append({"icon":"fa-robot","text":f"DistilBERT assigned {fraud_prob*100:.1f}% fraud probability.",
                        "type":"model","sev":"high" if fraud_prob>0.8 else "medium" if fraud_prob>0.5 else "low"})
        return {"top_fraud_words":[],"top_legit_words":[],"highlighted_html":"",
                "fraud_patterns":patterns,"reasons":reasons[:6],"model_name":self._model_name,
                "decision_score":fraud_prob,"n_fraud_tokens":0,"n_legit_tokens":0}

    @property
    def is_loaded(self): return self._model is not None

    def warmup(self):
        self.predict("Software engineer position requirements experience")
