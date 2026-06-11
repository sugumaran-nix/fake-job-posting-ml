# 🔍 Fraud Detection in Job Postings
### using NLP and Machine Learning

> **MCA Final Year Project**  
> Student : **Sugumaran** | Reg No : **722924622043** | Guide : **Ms. Kavipriya**  
> Sri Venkateswara College of Computer Applications and Management  
> Anna University — Apr / May 2026

---

## 📋 Project Overview

An end-to-end intelligent system that automatically identifies **fraudulent job postings**
by analysing their text using Natural Language Processing and a trained Machine Learning
classifier, deployed as a **Flask web application**.

---

## 🏗️ Project Structure

```
fraud_project/
│
├── app.py                  ← Main Flask web application
├── train.py                ← ML model training script
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── templates/
│   ├── base.html           ← Master layout (navbar + footer)
│   ├── index.html          ← Home page
│   ├── classify.html       ← Job input form
│   ├── result.html         ← Prediction result
│   ├── history.html        ← Past predictions log
│   └── about.html          ← Project info
│
├── static/
│   ├── css/
│   │   └── style.css       ← Full stylesheet
│   └── js/
│       └── main.js         ← Animations & interactions
│
├── models/                 ← Auto-created by train.py
│   ├── model.pkl           ← Trained classifier
│   └── vectorizer.pkl      ← Fitted TF-IDF vectorizer
│
├── data/                   ← Dataset & SQLite DB
│   ├── fake_job_postings.csv  ← EMSCAD dataset (download separately)
│   └── predictions.db      ← Auto-created SQLite history
│
└── notebooks/
    └── EDA.ipynb           ← Exploratory Data Analysis notebook
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download `fake_job_postings.csv` from:  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction  
Place it in the `data/` folder.

### 3. Train the model
```bash
python train.py
```
This creates `models/model.pkl` and `models/vectorizer.pkl`.

### 4. Run the web app
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🧠 NLP Pipeline

```
Raw Text → Concatenate → Lowercase → Strip HTML → Tokenise
        → Remove Stopwords → Lemmatise → TF-IDF → ML Model → Verdict
```

---

## 📊 Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression ★| 98.2%    | 96.4%     | 94.8%  | 95.6%    |
| SVM (Linear)         | 97.9%    | 95.8%     | 94.2%  | 95.0%    |
| Random Forest        | 97.5%    | 95.1%     | 93.2%  | 94.1%    |
| Naïve Bayes          | 96.8%    | 93.5%     | 90.1%  | 91.7%    |

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Web Framework**: Flask 2.x
- **NLP**: NLTK (tokenisation, stopwords, lemmatisation)
- **ML**: scikit-learn (TF-IDF + Logistic Regression)
- **Data**: pandas, NumPy
- **Storage**: SQLite (history), pickle (model)
- **Frontend**: HTML5, CSS3, Jinja2, Vanilla JS

---

## 🔌 API Endpoints

| Method | Route             | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | Home page                            |
| GET    | `/classify`       | Job input form                       |
| POST   | `/predict`        | Submit job for classification        |
| GET    | `/history`        | View past predictions                |
| GET    | `/about`          | Project information                  |
| POST   | `/clear_history`  | Clear all history records            |
| POST   | `/api/predict`    | JSON API for programmatic access     |
| GET    | `/health`         | Health check                         |

### JSON API Example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Data Entry","description":"Pay fee to register. Earn ₹1 lakh from home."}'
```

---

## 👨‍🎓 Author

| Field       | Details                                               |
|-------------|-------------------------------------------------------|
| Name        | Sugumaran                                             |
| Reg No      | 722924622043                                          |
| Guide       | Ms. Kavipriya                                         |
| Department  | Computer Applications                                 |
| College     | Sri Venkateswara College of Computer Applications     |
| University  | Anna University                                       |
| Exam        | Apr / May 2026                                        |
