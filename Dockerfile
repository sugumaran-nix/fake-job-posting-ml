FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Download NLTK data
COPY nltk_setup.py .
RUN python nltk_setup.py

# Copy source
COPY . .

# Create dirs that app.py writes to
RUN mkdir -p data models static/images

# Download BERT ONNX model from HuggingFace Hub if HF_MODEL_REPO is set.
# If not set, script exits silently and app falls back to sklearn.
# ARG allows the env var to be read at build time.
ARG HF_MODEL_REPO
ARG HF_TOKEN
ENV HF_MODEL_REPO=$HF_MODEL_REPO
ENV HF_TOKEN=$HF_TOKEN
RUN python download_bert.py

EXPOSE 5000

CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 120 --preload --access-logfile - app:app"]
