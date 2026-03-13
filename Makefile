.PHONY: help install data train evaluate serve test lint clean

PYTHON   := python
PIP      := pip
STREAMLIT := streamlit

help:
	@echo ""
	@echo "OpenOrca Query Intelligence — available commands:"
	@echo "  make install      Install all Python dependencies"
	@echo "  make data         Download & preprocess the OpenOrca dataset"
	@echo "  make train        Train all models and save the model bundle"
	@echo "  make evaluate     Print evaluation metrics for the saved bundle"
	@echo "  make serve        Launch the Streamlit dashboard"
	@echo "  make test         Run unit tests with coverage"
	@echo "  make lint         Run ruff checks"
	@echo "  make clean        Remove generated artefacts"
	@echo ""

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) -c "from src.data.loader import download_and_save; download_and_save()"

train:
	$(PYTHON) train.py

evaluate:
	$(PYTHON) -c "\
	import joblib, json; \
	b = joblib.load('models/model_bundle.pkl'); \
	print(json.dumps({m: {k: round(float(v),4) for k,v in s.items()} \
	       for m, s in b['metrics'].items()}, indent=2))"

serve:
	$(STREAMLIT) run app/dashboard.py --server.port 8501

predict:
	$(PYTHON) predict.py --text "$(TEXT)"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ app/ train.py predict.py train_demo.py --line-length 100

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage
