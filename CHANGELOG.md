# Changelog

All notable changes to this project will be documented in this file.  
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-03-13

### Added
- `train_demo.py` — self-contained synthetic-data training script; runs without
  HuggingFace internet access; called automatically by the dashboard if the bundle
  is missing or corrupt (self-healing behaviour).
- `requirements-ci.txt` — separate CI/dev dependency file (`ruff`, `pytest-cov`,
  `datasets`); keeps the Streamlit Cloud `requirements.txt` lean.
- `.python-version` — pins Python 3.11 for pyenv / Streamlit Cloud.
- `runtime.txt` — `python-3.11.9` for Streamlit Cloud runtime selection.
- `packages.txt` — empty apt-get packages file for Streamlit Cloud compatability.
- `.streamlit/config.toml` — dark-theme config and CORS / XSRF settings.
- `Dockerfile` — production-ready `python:3.11-slim` image with health check.
- Two new tests: `test_generate_synthetic_dataset` and `test_train_demo_quick`.

### Changed
- **CI workflow** (`ci.yml`) fully upgraded:
  - Python matrix: `3.11` + `3.12` (was `3.10` + `3.11`)
  - `checkout@v4`, `setup-python@v5` (was v4/v4)
  - `ruff` replaces `flake8` + `black` for linting
  - `codecov/codecov-action@v5` (was v4)
  - New `docker` job: `docker/setup-buildx-action@v3` +
    `docker/build-push-action@v5` (build-only, no push)
  - CI now installs from `requirements-ci.txt`
- `requirements.txt` — tightened version bounds, removed dev-only packages
  (`black`, `isort`, `flake8`, `matplotlib`, `seaborn`, `kaleido`,
   `huggingface-hub`, `datasets`).  Added upper bounds for cloud stability.
- `setup.py` — fixed package discovery for the actual `src` package layout and
  aligned runtime dependencies with `requirements.txt` so editable installs are
  cloud-safe and do not pull CI-only packages.
- `config.py` — removed deprecated `multi_class: "auto"` and
  `use_label_encoder: False` from model param grids.
- `src/models/trainer.py` — removed XGBoost `use_label_encoder` filter
  (deprecated params cleaned from config directly).
- Streamlit dashboard (`app/dashboard.py`) — self-healing bundle load via
  `train_demo.ensure_bundle()`; sidebar shows synthetic-data warning;
  `load_bundle()` now returns `(bundle, status)` tuple.
- `Makefile` — switched linting to `ruff` to match CI and removed stale
  `black`/`isort`/`flake8` commands.
- `Dockerfile` — added `curl` for the healthcheck and fixed the exec-form
  Streamlit entrypoint.
- `README.md` — removed duplicated sections, replaced placeholders with the
  actual repository URL, and updated setup, Docker, Streamlit, and CI docs.

### Fixed
- `test_end_to_end_training` — `LogisticRegression(multi_class="auto")` raised
  `TypeError` on scikit-learn ≥ 1.6 (parameter removed). Fixed to remove arg.

---

## [1.0.0] — 2026-03-13

### Added
- Initial project release.
- Full ML pipeline: data loader (`src/data/loader.py`), preprocessor,
  feature engineering (TF-IDF + 12 stat features), 5-model training with
  `RandomizedSearchCV`, evaluation bundle, Streamlit dashboard with 5 tabs,
  CLI predict script, EDA notebook, GitHub Actions CI.
