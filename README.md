# Prompt Source Intelligence

End-to-end NLP classification project that trains, evaluates, and deploys a Streamlit dashboard to predict the source category of OpenOrca-style prompts using reproducible data pipelines, model comparison, analytics, and live inference.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/Muhammad-Farooq13/Prompt-source-intelligence/actions/workflows/ci.yml/badge.svg)

## Problem Statement

This project predicts which instruction source a prompt most likely came from using OpenOrca-style text alone. It is framed as a multi-class NLP classification problem and packaged as a portfolio-ready workflow: data preparation, feature engineering, cross-validated modeling, evaluation, deployment, and live inference.

## Repository Structure

```text
Prompt-source-intelligence/
├── app/dashboard.py
├── src/data/
├── src/features/
├── src/models/
├── src/visualization/
├── tests/
├── train.py
├── train_demo.py
├── predict.py
├── requirements.txt
├── requirements-ci.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Setup

```bash
git clone https://github.com/Muhammad-Farooq13/Prompt-source-intelligence.git
cd Prompt-source-intelligence

python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

For local testing and CI tools:

```bash
pip install -r requirements-ci.txt
```

## Training

Full training uses the Hugging Face dataset:

```bash
python train.py
python train.py --quick
```

Cloud-safe demo training generates synthetic OpenOrca-like data and writes the dashboard bundle:

```bash
python train_demo.py
python train_demo.py --quick
python train_demo.py --rebuild
```

The dashboard bundle is saved to `models/demo_bundle.pkl`. The full pipeline bundle is saved to `models/model_bundle.pkl`.

## Streamlit App

Run locally:

```bash
streamlit run app/dashboard.py
```

The app includes five tabs:

1. Overview
2. Model Results
3. Analytics
4. Pipeline/API
5. Predict

If the bundle is missing or corrupt, the app will automatically rebuild it using `train_demo.py`.

## Prediction API

CLI examples:

```bash
python predict.py --text "Explain the difference between Git and GitHub."
python predict.py --interactive
python predict.py --file questions.txt --json
```

Python usage:

```python
from predict import load_bundle, predict_single

bundle = load_bundle()
result = predict_single("What is recursion?", bundle=bundle)
print(result["predicted_class"])
print(result["probabilities"])
```

## Docker

```bash
docker build -t prompt-source-intelligence .
docker run -p 8501:8501 prompt-source-intelligence
```

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
ruff check src/ app/ train.py predict.py train_demo.py --line-length 100
```

Current local result: `25 passed`.

## CI/CD

The GitHub Actions workflow includes:

1. Python matrix on 3.11 and 3.12
2. `actions/checkout@v4`
3. `actions/setup-python@v5`
4. `ruff` linting
5. `pytest` with `coverage.xml`
6. `codecov/codecov-action@v5`
7. Docker build verification with Buildx v3 and build-push-action v5

## Model Results

The dashboard renders the latest metrics from the saved bundle, including:

1. Accuracy
2. Weighted F1
3. Macro F1
4. Precision and recall
5. Confusion matrix
6. Feature importance
7. Live prediction confidence/risk view

Run `python train_demo.py` or `python train.py` to generate updated scores for your environment.

## Deployment Notes

Streamlit Cloud compatibility files are included:

1. `.python-version`
2. `runtime.txt`
3. `packages.txt`
4. `.streamlit/config.toml`

Runtime dependencies are separated from CI-only tooling to reduce Streamlit Cloud install risk.

## Author

Muhammad Farooq  
Email: mfarooqshafee333@gmail.com  
GitHub: https://github.com/Muhammad-Farooq13

## License

MIT. See `LICENSE`.
