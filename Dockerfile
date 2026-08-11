FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Download NLTK data
COPY nltk_setup.py .
RUN python nltk_setup.py

# Copy source
COPY . .

# Create dirs
RUN mkdir -p data models static/images

EXPOSE 5000

# download_bert.py runs at STARTUP (not build time) so Render env vars are available.
# If HF_MODEL_REPO is not set it exits silently and app uses sklearn fallback.
CMD ["sh", "-c", "python download_bert.py && gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 120 --preload --access-logfile - app:app"]
