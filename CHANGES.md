# Upgrade Changelog — v2

## Files Modified

### `utils/preprocessing.py` ✅ Enhanced NLP Pipeline
| Step | Before | After |
|------|--------|-------|
| HTML stripping | Regex tag removal | Tags + **HTML entity decode** (`&amp;` → `&`, `&nbsp;` → space) |
| Contractions | ❌ Not handled | ✅ `don't → do not`, `we're → we are` (via `contractions` lib, graceful fallback) |
| URL removal | ❌ URLs left in text | ✅ `http://` and `www.` links stripped before tokenisation |
| Regex | Compiled per-call | **Pre-compiled** at module level (faster, one-time cost) |
| Batch API | ❌ No batch function | ✅ `batch_preprocess(texts, show_progress)` added |

### `utils/evaluation.py` ✅ ROC Curves + Model Registry
**New functions:**
- `plot_roc_curves(results)` — overlays all model ROC curves in one chart → `static/images/roc_curves.png`
- `save_all_models(results)` — saves each trained model as `models/<name>.pkl`
- `load_model_registry()` — loads `models/model_registry.json` (name → path map)
- `load_model_by_name(name)` — loads a specific model from registry by name

**Modified:**
- `compute_metrics()` — now runs 5-fold cross-validation, stores `cv_f1_mean` + `cv_f1_std` and raw `_y_test`/`_y_score` arrays for ROC plotting
- `save_metrics()` — excludes non-JSON-serialisable keys (`model`, `_y_test`, `_y_score`)

### `train.py` ✅ Saves All Models + ROC Plot
**New:**
- `save_all_models(results)` called → every classifier saved individually
- `plot_roc_curves(results)` called → ROC curve image generated
- Cross-validation F1 printed in results table (mean ± std)
- `metadata.json` now includes `all_models` list

### `app.py` ✅ Runtime Model Switching + New API Endpoints
**New routes:**
- `POST /select-model` — hot-swap active model without Flask restart; form field: `model_name`
- `GET  /api/models` — JSON list of all trained models with metrics and active status

**Modified:**
- `_active_state` dict — mutable container for hot-swappable model + name
- `_load_active_model()` — restores last selection from `models/active_model.json` on startup
- `switch_model(name)` — atomic model swap, persists to `active_model.json`
- `/api/predict` — accepts optional `model` field for per-request model override
- `/health` — now reports `active_model` name and `vectorizer_ok`
- `ml_predict()` — uses `_active_state["name"]` as model name in response
- `/models` route — passes `active_model` and `registry` to template

### `templates/models.html` ✅ Major UI Upgrades
**New sections:**
- **Model Switcher** — one-click buttons to switch active model (highlights active vs best)
- **ROC Curves** — new `<img>` block displaying `roc_curves.png`
- **CV F1 column** — cross-validation score (mean ± std) added to metrics table
- **Active badge** — separate "🔀 Active" badge distinct from "★ Best F1"
- **NLP Pipeline card** — lists all 6 preprocessing steps in Training Config
- **Model Registry card** — shows registry size, hot-swap status, API endpoints
- **REST API Reference table** — documents all 4 endpoints with descriptions

### `static/css/style.css` ✅ New CSS Classes
- `.badge-info` — blue info badge for "Best F1" indicator
- `.row-selected` — green row highlight for best (but not active) model
- `.model-switcher-bar` — flex layout for model switch buttons

### `requirements.txt` ✅ Added
- `contractions>=0.1.73`

---

## New Files Generated at Runtime (after `python train.py`)
| File | Description |
|------|-------------|
| `models/logistic_regression.pkl` | Logistic Regression model |
| `models/random_forest.pkl` | Random Forest model |
| `models/linear_svm.pkl` | Linear SVM model |
| `models/naive_bayes.pkl` | Naive Bayes model |
| `models/model_registry.json` | Name → path registry |
| `models/active_model.json` | Persisted active model selection |
| `static/images/roc_curves.png` | ROC curve overlay chart |

---

## How to Use the New Features

### Switch Active Model (Web UI)
1. Run `python train.py` to train and save all models
2. Open `http://localhost:5000/models`
3. Click any model button in the "Active Inference Model" switcher
4. All subsequent predictions use the selected model — no restart needed

### Switch Active Model (API)
```bash
# List all models
curl http://localhost:5000/api/models

# Switch to Random Forest
curl -X POST http://localhost:5000/select-model \
  -d "model_name=Random Forest"

# Predict with a specific model (per-request override)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Work from home, guaranteed income!", "model": "Naive Bayes"}'
```
