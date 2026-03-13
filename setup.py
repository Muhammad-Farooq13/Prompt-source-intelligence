from setuptools import setup, find_packages

setup(
    name="prompt-source-intelligence",
    version="1.1.0",
    description="End-to-end NLP prompt source classifier with Streamlit deployment",
    author="Muhammad Farooq",
    author_email="mfarooqshafee333@gmail.com",
    url="https://github.com/Muhammad-Farooq-13/Prompt-source-intelligence",
    python_requires=">=3.11",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.26.0,<3.0",
        "pandas>=2.0.0,<3.0",
        "scikit-learn>=1.3.0,<2.0",
        "xgboost>=2.0.0,<4.0",
        "lightgbm>=4.0.0,<5.0",
        "textstat>=0.7.3",
        "plotly>=5.17.0,<7.0",
        "streamlit>=1.28.0",
        "streamlit-option-menu>=0.3.6",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "pyarrow>=13.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
