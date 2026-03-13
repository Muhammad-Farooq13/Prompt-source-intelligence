# Entry point for Streamlit Cloud deployment.
# Streamlit Cloud looks for streamlit_app.py at the repo root.
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_path(str(Path(__file__).resolve().parent / "app" / "dashboard.py"), run_name="__main__")
