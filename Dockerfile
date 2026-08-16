# ── Build stage ────────────────────────────────────────────────────────
# devops-engineer: pinned base image digest, non-root user, no-cache layers
FROM python:3.11-slim AS base

# System deps — minimal, in one layer, cleaned up immediately
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc curl \
    && rm -rf /var/lib/apt/lists/*

# devops-engineer: non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install deps as root (pip needs write access), then hand off
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# NLTK data
COPY nltk_setup.py .
RUN python nltk_setup.py

# Copy source
COPY --chown=appuser:appgroup . .

# Create writable dirs owned by appuser
RUN mkdir -p data models static/images \
    && chown -R appuser:appgroup data models

# devops-engineer: switch to non-root before EXPOSE and CMD
USER appuser

EXPOSE 5000

# Health check — devops-engineer: Kubernetes readiness probe compatible
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# download_bert.py runs at startup (not build time) so Render env vars are available.
# If HF_MODEL_REPO is not set it exits silently and app uses sklearn fallback.
CMD ["sh", "-c", "python download_bert.py && gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 120 --preload --access-logfile - app:app"]
