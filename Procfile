web: python nltk_setup.py && python -c "import app; app.init_db()" && gunicorn app:app --workers 1 --timeout 120 --preload --bind 0.0.0.0:$PORT
