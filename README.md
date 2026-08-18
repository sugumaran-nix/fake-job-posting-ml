# JobGuard — Fake Job Posting Detector

> Paste a job posting, get an instant verdict. ML-powered classification with token-level explainability and a 10-signal URL fraud scorer.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![DistilBERT](https://img.shields.io/badge/DistilBERT-FF6B35?style=for-the-badge&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![Render](https://img.shields.io/badge/Deployed_on_Render-46E3B7?style=for-the-badge&logo=render&logoColor=111827)

**[🚀 Live Demo](https://jobguard-8vur.onrender.com)** · **[HuggingFace Model](https://huggingface.co/Sugum4r4n/jobguard-bert)**

---

## ✨ Features

- **5 classifiers** — Logistic Regression, Random Forest, Linear SVM, Naive Bayes, DistilBERT (ONNX INT8)
- **Runtime model switching** — hot-swap the active model without restarting the server
- **URL fraud scorer** — 10-signal heuristic (HTTPS, TLD, entropy, typosquatting, digit density…)
- **Company name analysis** — scam keyword + brand impersonation detection
- **Explainability** — top fraud/legit tokens + matched fraud patterns per prediction
- **REST JSON API** — `/api/predict` with per-request model override
- **Prediction history** — SQLite-backed, viewable and clearable via UI
- **Health endpoint** — `/health` for uptime monitoring

---

## 📊 Model Performance

Trained on 17,880 EMSCAD listings (866 fraud · 17,014 legit · 4.8:1 class imbalance). Best model selected by **fraud-class F1**, not overall accuracy.

| Model | Fraud F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| **Linear SVM** ✅ | **0.8757** | 98.83% | 0.9838 |
| Logistic Regression | 0.8051 | 97.87% | 0.9859 |
| Random Forest | 0.7372 | 97.99% | 0.9856 |
| DistilBERT (ONNX INT8) | fine-tuned | — | — |
| Naive Bayes | 0.5105 | 96.09% | 0.9362 |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Framework | Flask 3.0 + Gunicorn |
| ML | scikit-learn, TF-IDF (10k features, bigrams) |
| Deep Learning | DistilBERT fine-tuned → ONNX INT8 quantized |
| NLP | NLTK, contractions, stemming |
| Storage | SQLite (predictions) |
| Deployment | Render |

---

## 📁 Project Structure

```
fake-job-posting-ml/
├── app.py                      # Flask web application
├── train.py                    # ML training pipeline
├── analyzer.py                 # URL + company fraud heuristics
├── bert_finetune.py            # DistilBERT fine-tuning script
├── bert_to_onnx.py             # ONNX INT8 quantization
├── requirements.txt
│
├── utils/
│   ├── preprocessing.py        # Text cleaning pipeline
│   ├── model_router.py         # Runtime model hot-swap
│   ├── bert_predictor.py       # ONNX inference wrapper
│   ├── evaluation.py           # Metrics, plots, model registry
│   └── explainer.py            # Feature-importance explainability
│
├── data/
│   ├── fake_job_postings.csv   # ← you supply this (see Setup)
│   └── predictions.db          # SQLite DB (auto-created)
│
├── models/                     # Auto-created by train.py
│   ├── model.pkl               # Best classical classifier
│   ├── vectorizer.pkl          # Fitted TF-IDF
│   ├── bert_onnx_quantized.onnx
│   ├── bert_tokenizer/
│   ├── model_registry.json
│   ├── metrics.json
│   └── model_metadata.json
│
├── notebooks/                  # EDA
├── static/                     # CSS, JS, evaluation plots
├── templates/                  # Jinja2 HTML templates
├── tests/                      # pytest suite
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- The EMSCAD dataset from Kaggle

### 1. Clone & install

```bash
git clone https://github.com/sugumaran-nix/fake-job-posting-ml.git
cd fake-job-posting-ml
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download `fake_job_postings.csv` from Kaggle and place it in `data/`:

```
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
```

### 3. Set environment variables

```bash
cp env.example .env
# Edit .env — set FLASK_SECRET_KEY to a random string
```

### 4. Train the models

```bash
python train.py
```

Produces all `.pkl` files, `metrics.json`, `model_registry.json`, and evaluation plots in `static/images/`.

### 5. (Optional) Fine-tune DistilBERT

```bash
python bert_finetune.py     # fine-tune on the dataset
python bert_to_onnx.py      # quantize to ONNX INT8
```

### 6. Run the app

```bash
# Development
python app.py

# Production
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

Open [http://localhost:5000](http://localhost:5000)

> **Note:** Use `-w 1` (single worker). Active-model state is held in process memory — multi-worker setups require Redis or DB-backed state for model switching to propagate across workers.

---

## 🐳 Docker

```bash
docker build -t jobguard .
docker run -p 5000:5000 \
  -e FLASK_SECRET_KEY=your_secret_here \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  jobguard
```

Or with Compose:

```bash
docker-compose up --build
```

---

## 🧠 How It Works

### ML Pipeline

```
Raw job posting fields
        ↓
  Text combination (title + company + description + requirements)
        ↓
  NLP preprocessing
    • HTML entity decoding
    • Contraction expansion  ("don't" → "do not")
    • Lowercasing, punctuation removal
    • Stopword removal + stemming
        ↓
  TF-IDF Vectorisation
    • 10,000 features, bigrams, sublinear TF, min_df=2
        ↓
  Missingness features (salary empty? profile empty? benefits empty?)
        ↓
  Classifier  (Linear SVM default · or DistilBERT ONNX)
        ↓
  Confidence score + token-level explanation
```

### URL + Company Heuristics

`analyzer.py` runs independently of the ML model and scores the company website across 10 signals:

1. HTTPS / SSL
2. Raw IP address instead of domain
3. Free hosting platform (Wix, Weebly, GitHub Pages…)
4. High-risk TLD (`.tk`, `.xyz`, `.top`…)
5. Domain name Shannon entropy (auto-generated randomness)
6. Scam keywords in URL (`earn`, `quickmoney`, `joining-fee`…)
7. Subdomain depth > 3 levels
8. Typosquatting / brand impersonation (13 major brands)
9. Digit density in domain name
10. Domain length anomaly

Company name scored separately for scam keywords, brand impersonation, vague naming patterns, and excessive legal suffixes.

**Combined score:** URL (60%) + Company (40%) → `low / medium / high` risk.

---

## 🌐 Web Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Homepage with stats |
| `/classify` | GET | Job posting input form |
| `/predict` | POST | Submit posting for classification |
| `/history` | GET | Prediction history |
| `/models` | GET | Model comparison + switcher |
| `/about` | GET | Training metadata |
| `/select-model` | POST | Switch active model |
| `/clear_history` | POST | Delete all prediction records |
| `/health` | GET | Health check JSON |

---

## 🔌 REST API

### `POST /api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Data Entry Operator",
    "company": "Global Earn Solutions",
    "description": "Work from home. Guaranteed ₹5000/day. No experience needed.",
    "requirements": "",
    "website": "http://earn4u.tk",
    "model": "Linear SVM"
  }'
```

**Response:**

```json
{
  "prediction": "Fraudulent",
  "is_fraud": true,
  "confidence": 94.7,
  "model_used": "Linear SVM",
  "url_risk": "high",
  "combined_score": 72.4,
  "explanation": {
    "top_fraud_words": ["guaranteed", "earn", "home"],
    "top_legit_words": [],
    "fraud_patterns": ["..."],
    "reasons": ["..."],
    "decision_score": 2.31
  }
}
```

### `GET /api/models`

Returns all available models with accuracy, F1, AUC, and cross-validation scores.

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | *(insecure dev default)* | **Required in production.** Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `PORT` | `5000` | Port to bind |

---

## 🚨 Known Limitations

- **Single-worker only** for runtime model switching. Multi-worker Gunicorn requires Redis or DB-backed active model state.
- The ±confidence adjustment based on URL risk is a heuristic with no statistical grounding — it can suppress a legitimate posting flagged by a suspicious URL.
- Classical models (TF-IDF + sklearn) do not understand semantic meaning. A carefully written fake posting using legitimate vocabulary may evade detection.
- No live URL reachability check — URL analysis is purely syntactic/structural.

---

## 📄 License

MIT
