"""
utils/preprocessing.py
=======================
Single source of truth for NLP text preprocessing.

Pipeline (UPGRADED v2):
  raw text
    → HTML tag strip (handles nested/malformed tags)
    → HTML entity decode  (&amp; → &, &nbsp; → space, etc.)
    → Contraction expansion  (don't → do not, we're → we are)
    → Lowercase
    → URL removal           (http/www links stripped entirely)
    → Punctuation removal   (keep only a-z and spaces)
    → Whitespace collapse
    → NLTK tokenise
    → Stopword removal + single-char token drop
    → Lemmatise
    → Rejoin → clean string

Changes from v1:
  ✅ HTML entity decoding (was missed before — &amp; stayed in tokens)
  ✅ Contraction expansion (don't → do not — better vocab coverage)
  ✅ URL removal before lowercasing (avoids tokenising domain garbage)
  ✅ batch_preprocess() for efficient DataFrame processing
"""

import re
import html as html_module
import nltk

# ── NLTK downloads (silent, idempotent) ───────────────────────────────
for _resource in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Optional: contractions library (graceful fallback if not installed)
try:
    import contractions as _contractions_lib
    _CONTRACTIONS_AVAILABLE = True
except ImportError:
    _CONTRACTIONS_AVAILABLE = False

# Module-level singletons — initialised once per process
_STOPWORDS  = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Pre-compiled regex patterns (compile once for performance)
_RE_HTML_TAGS  = re.compile(r"<[^>]+>")
_RE_URLS       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_NON_ALPHA  = re.compile(r"[^a-z\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """
    Clean and normalise a raw job-posting text string.

    Steps
    -----
    1.  Guard against non-string input (NaN, None, numbers)
    2.  Strip HTML tags
    3.  Decode HTML entities   (&amp; → &, &nbsp; → space, &#x27; → ')
    4.  Expand contractions    (don't → do not, we're → we are)
    5.  Remove URLs
    6.  Lowercase
    7.  Remove punctuation / digits / special chars
    8.  Collapse whitespace
    9.  Tokenise with NLTK word_tokenize
    10. Drop stopwords and single-character tokens
    11. Lemmatise
    12. Rejoin into clean string
    """
    if not isinstance(text, str):
        return ""

    # 1. Strip HTML tags
    text = _RE_HTML_TAGS.sub(" ", text)

    # 2. Decode HTML entities (&amp; &nbsp; &#39; etc.)
    text = html_module.unescape(text)

    # 3. Expand contractions (don't → do not, it's → it is)
    if _CONTRACTIONS_AVAILABLE:
        try:
            text = _contractions_lib.fix(text)
        except Exception:
            pass

    # 4. Remove URLs before lowercasing
    text = _RE_URLS.sub(" ", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Keep only letters and whitespace
    text = _RE_NON_ALPHA.sub(" ", text)

    # 7. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 8. Tokenise → filter stopwords & short tokens → lemmatise
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in word_tokenize(text)
        if tok not in _STOPWORDS and len(tok) > 1
    ]

    return " ".join(tokens)


def batch_preprocess(texts, show_progress: bool = False):
    """Preprocess an iterable of strings, optionally showing progress."""
    results = []
    for i, t in enumerate(texts):
        results.append(preprocess(t))
        if show_progress and i > 0 and i % 1000 == 0:
            print(f"  ... preprocessed {i} rows", flush=True)
    return results


def build_combined_text(row, cols: list) -> str:
    """Concatenate selected columns from a dict/Series into one string."""
    parts = []
    for col in cols:
        if isinstance(row, dict):
            val = row.get(col, "")
        else:
            val = getattr(row, col, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts)


# Column definitions (exported for train.py)
TEXT_COLS     = ["title", "company_profile", "description", "requirements"]
ALL_TEXT_COLS = ["title", "company_profile", "description",
                 "requirements", "benefits", "department"]
