# 🕵️ Fake Job Posting Detector

A machine-learning web application that classifies job postings as **Fraudulent** or **Legitimate** using NLP text classification combined with automated URL and company-name fraud heuristics.

Built with Flask · scikit-learn · TF-IDF · SQLite

---

## ✨ Features

| Feature | Details |
|---|---|
| **4 ML Classifiers** | Logistic Regression, Random Forest, Linear SVM, Naive Bayes |
| **Runtime model switching** | Swap active model without restarting Flask |
| **URL fraud analysis** | 10-signal heuristic scorer (HTTPS, TLD, entropy, typosquatting…) |
| **Company name analysis** | Scam keyword + brand impersonation detection |
| **Explainability** | Top fraud/legit words + matched fraud patterns per prediction |
| **REST JSON API** | `/api/predict` with per-request model override |
| **Prediction history** | SQLite-backed, viewable + clearable via UI |
| **Health endpoint** | `/health` for uptime monitoring |

---

## 📂 Project Structure

```
fake-job-posting-ml/
├── app.py                      # Flask web application (v4)
├── train.py                    # ML training pipeline (v2)
├── analyzer.py                 # URL + company fraud heuristics
├── fix_db.py                   # DB schema migration helper
├── requirements.txt            # Python dependencies
│
├── utils/
│   ├── preprocessing.py        # Text cleaning pipeline
│   ├── evaluation.py           # Metrics, plots, model registry
│   └── explainer.py            # Feature-importance explainability
│
├── data/
│   ├── fake_job_postings.csv   # ← you supply this (see Setup)
│   └── predictions.db          # SQLite DB (auto-created)
│
├── models/                     # Auto-created by train.py
│   ├── model.pkl               # Best classifier
│   ├── vectorizer.pkl          # Fitted TF-IDF
│   ├── model_registry.json     # name → path mapping
│   ├── metrics.json            # All model scores
│   ├── model_metadata.json     # Training metadata
│   └── active_model.json       # Persisted model selection
│
├── notebooks/                  # Exploratory analysis
├── static/                     # CSS, JS, evaluation plots
├── templates/                  # Jinja2 HTML templates
│
├── tests/
│   ├── test_analyzer.py        # Unit tests for analyzer.py
│   ├── test_app.py             # Flask route tests
│   └── test_preprocessing.py  # Unit tests for preprocessing
│
├── .env.example                # Environment variable template
├── .gitignore
└── Dockerfile
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/suguamaran-nix/fake-job-posting-ml.git
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
cp .env.example .env
# Edit .env — set FLASK_SECRET_KEY to a random string
```

### 4. Train the models

```bash
python train.py
```

This produces all `.pkl` files, `metrics.json`, `model_registry.json`, and evaluation plots in `static/images/`.

### 5. Run the app

```bash
# Development
python app.py

# Production (recommended)
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

Open http://localhost:5000

> **Note:** Use `-w 1` (single worker) with Gunicorn. The active-model state is held in process memory; multi-worker setups require a shared store (Redis/DB) for model switching to propagate across workers.

---

## 🐳 Docker

```bash
docker build -t fake-job-ml .
docker run -p 5000:5000 \
  -e FLASK_SECRET_KEY=your_secret_here \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fake-job-ml
```

Or with docker-compose:

```bash
docker-compose up --build
```

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
| `/select-model` | POST | Switch active model (form: `model_name`) |
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
    "model": "Logistic Regression"
  }'
```

**Response:**

```json
{
  "prediction": "Fraudulent",
  "is_fraud": true,
  "confidence": 94.7,
  "model_used": "Logistic Regression",
  "url_risk": "high",
  "combined_score": 72.4,
  "explanation": {
    "top_fraud_words": ["guaranteed", "earn", "home"],
    "top_legit_words": [],
    "fraud_patterns": [...],
    "reasons": [...],
    "decision_score": 2.31
  }
}
```

### `GET /api/models`

Returns all available models with their accuracy, F1, AUC, and cross-validation scores.

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
    • Contraction expansion ("don't" → "do not")
    • Lowercasing, punctuation removal
    • Stopword removal + stemming
        ↓
  TF-IDF Vectorisation
    • 10,000 features, bigrams, sublinear TF, min_df=2
        ↓
  Missingness features (salary empty? profile empty? benefits empty?)
        ↓
  Classifier (default: best Fraud-F1 model from training)
        ↓
  Confidence score + top feature explanation
```

### URL + Company Heuristics

The `analyzer.py` module runs independently of the ML model and scores the company website URL across 10 signals:

1. HTTPS / SSL check  
2. Raw IP address instead of domain  
3. Free hosting platform (Wix, Weebly, GitHub Pages, etc.)  
4. High-risk TLD (`.tk`, `.xyz`, `.top`, etc.)  
5. Domain name Shannon entropy (auto-generated randomness)  
6. Scam keywords in URL (`earn`, `quickmoney`, `joining-fee`, etc.)  
7. Subdomain depth (> 3 levels)  
8. Typosquatting / brand impersonation (13 major brands)  
9. Digit density in domain name  
10. Domain length anomaly  

Company name is scored separately for scam keywords, brand impersonation, vague naming patterns, and excessive legal suffixes.

**Combined score:** URL (60%) + Company (40%) → `low / medium / high` risk.

### Model Selection

Training selects the best model by **fraud-class F1 score** (not overall accuracy), which is correct for this heavily imbalanced dataset (~4.8% fraud rate).

---

## 📊 Dataset

| Field | Value |
|---|---|
| Source | [Kaggle — Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| Rows | ~17,880 |
| Fraudulent | ~866 (~4.8%) |
| Features used | title, company_profile, description, requirements, benefits, salary_range, department |

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
| `FLASK_SECRET_KEY` | *(insecure dev default)* | **Must be set in production.** Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `PORT` | `5000` | Port to bind |

---

## 🚨 Known Limitations

- **Single-worker only** for runtime model switching. With multiple Gunicorn workers, only the worker that receives `/select-model` will switch — others retain the old model. For multi-worker production, store the active model name in the SQLite DB or Redis and reload per-request.
- The ±confidence adjustment based on URL risk is a heuristic with no statistical grounding. It can suppress a legitimate posting flagged by a suspicious URL.
- The ML models are classical (TF-IDF + sklearn). They do not understand semantic meaning. A carefully written fake posting using legitimate vocabulary may evade detection.
- No live URL reachability check — URL analysis is purely syntactic/structural.

---

## 🗺️ Roadmap

- [ ] Threshold tuning (precision-recall curve, not default 0.5)
- [ ] BERT/sentence-transformer embeddings
- [ ] Live URL reachability + WHOIS age check
- [ ] Docker Compose with Redis for multi-worker model state
- [ ] GitHub Actions CI with pytest + linting

---

## 📄 License

MIT
