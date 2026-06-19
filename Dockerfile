FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Download all NLTK data used by preprocessing (wordnet for lemmatizer,
# punkt/punkt_tab for tokenizer, stopwords, omw-1.4 for wordnet support).
COPY nltk_setup.py .
RUN python nltk_setup.py

# Copy source
COPY . .

# Create dirs that train.py / app.py write to
RUN mkdir -p data models static/images

EXPOSE 5000

# Single worker — required for in-process model switching.
# See README "Known Limitations" for multi-worker notes.
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 120 --access-logfile - app:app"]
