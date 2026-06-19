FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Download NLTK data used by preprocessing
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy source
COPY . .

# Create dirs that train.py / app.py write to
RUN mkdir -p data models static/images

EXPOSE 5000

# Single worker — required for in-process model switching.
# See README "Known Limitations" for multi-worker notes.
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 120 --access-logfile - app:app"]
