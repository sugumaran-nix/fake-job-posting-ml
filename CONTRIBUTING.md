# Contributing & Setup Guide

## Dataset

The training dataset (`fake_job_postings.csv`) is **not included** in this repo.
It is the EMSCAD dataset published by the University of the Aegean.

**Download via Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction
unzip real-or-fake-fake-jobposting-prediction.zip -d data/
```

Or download manually from:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Place the file at `data/fake_job_postings.csv` before running `train.py`.

**Citation:**
> Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017).
> Automatic Detection of Online Recruitment Frauds: Characteristics,
> Methods, and a Public Dataset.
> *Future Internet*, 9(1), 6.

## Local Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK data
python nltk_setup.py

# Train models (requires dataset above)
python train.py

# Run dev server
FLASK_ENV=development python app.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Model Binary Policy

`.pkl` model files are listed in `.gitignore` and must **not** be committed.
They are generated locally by running `train.py`. This is intentional:

- Binary model files are large (up to 8 MB each)
- They embed the exact scikit-learn version used for training
- Anyone cloning the repo must train their own models from the dataset

## Admin Routes

`/select-model` and `/clear_history` are protected by `ADMIN_TOKEN`.
Set it in your `.env` file (copy `env.example` to `.env` first).
Leave it empty for local development.

## Architecture Notes

- The app runs with **gunicorn -w 1** (single worker) by default.
  Model selection is persisted to `models/active_model.json` so all
  workers read the same selection, but the in-process model cache
  only updates per-worker. For true multi-worker setups, consider
  a shared Redis cache or load models per-request.
- Rate limiting uses `memory://` storage by default.
  Set `RATELIMIT_STORAGE_URI=redis://localhost:6379/0` for multi-worker.
