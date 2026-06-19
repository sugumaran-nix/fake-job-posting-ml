"""Pre-download all NLTK resources used by utils/preprocessing.py.
Run once at Docker build time so the app never tries to download
resources at request time (which can fail silently in production).
"""
import nltk

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
        print(f"Downloaded NLTK resource: {resource}")
    except Exception as e:
        # Some resources (e.g. punkt_tab) may not exist on older nltk
        # versions' download index — skip rather than fail the build.
        print(f"Skipping NLTK resource {resource}: {e}")
