# JobGuard — Production Upgrade Design

## Three-perspective analysis (fullstack-guardian)

### Backend gaps
- No type hints on app.py public functions
- `save_pred` has no input length cap before DB write
- `api_predict` doesn't length-cap incoming JSON fields
- `get_predictor()` singleton not thread-safe
- No structured logging

### Security gaps (secure-code-guardian)
- CSP disabled in Talisman (`content_security_policy=False`)
- `ADMIN_TOKEN` comparison is timing-vulnerable (`==`)
- `api_predict` 503 error leaks internal model state
- Rate limiter uses `memory://` — doesn't share across gunicorn workers

### DevOps gaps (devops-engineer)
- Dockerfile runs as root
- No GitHub Actions CI pipeline
- No `.env.example`

## What we fix
1. Type-annotate app.py, analyzer.py, utils/
2. Enable CSP via Talisman
3. Fix ADMIN_TOKEN to use `hmac.compare_digest`
4. Add input length caps on all API + form fields
5. Thread-safe predictor singleton
6. Expand test suite (security, edge cases, API contract)
7. Harden Dockerfile (non-root user)
8. Add GitHub Actions CI
9. Add .env.example
