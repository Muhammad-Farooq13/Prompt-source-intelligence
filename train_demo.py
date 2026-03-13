"""
train_demo.py
─────────────
Lightweight self-contained training script that works WITHOUT an internet
connection or the full HuggingFace dataset download.

It generates a synthetic dataset that mimics the structure of OpenOrca
(same columns, same label distribution, similar text patterns) then trains
all registered classifiers and saves models/demo_bundle.pkl.

This is the file used by:
  • Streamlit Cloud (no HuggingFace token needed)
  • CI smoke tests
  • Local quick-start demos

The Streamlit app auto-calls this if the bundle is missing or corrupt.

Usage
-----
    python train_demo.py              # standard run (~1–2 min)
    python train_demo.py --quick      # <30s, lower accuracy
    python train_demo.py --rebuild    # force rebuild even if bundle exists
"""

import argparse
import logging
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

BUNDLE_PATH = ROOT / "models" / "demo_bundle.pkl"
RANDOM_STATE = 42

# ── Synthetic data templates ──────────────────────────────────────────────────

_TEMPLATES = {
    "FLAN": [
        "What is the {adj} {noun} of {place}?",
        "Translate '{phrase}' to {language}.",
        "True or False: {statement}.",
        "What is the antonym of '{word}'?",
        "Complete the sentence: The {noun} {verb} quickly.",
        "Which country uses {currency} as its currency?",
        "How many {unit} are in a {bigger_unit}?",
        "Name three {category} that are also {property}.",
        "What year did {event} happen?",
        "Is {person} a {role}?",
    ],
    "CoT": [
        "If {name} has {n1} {item}s and gives {n2} to {name2}, how many remain?",
        "A train travels at {speed} km/h. How far does it go in {time} hours?",
        "Explain step by step how to solve: {n1} * {n2} + {n3}.",
        "There are {n1} red and {n2} blue balls in a bag. What is the probability of picking red?",
        "If {n1}% of {n2} people voted, how many did not vote?",
        "A rectangle has sides {n1} and {n2}. What is its area and perimeter?",
        "Prove that the sum of angles in a triangle equals 180 degrees.",
        "Solve for x: {n1}x + {n2} = {n3}.",
        "If compound interest rate is {n1}% annually, what is the value after {n2} years?",
        "Using induction, prove that the sum of first n natural numbers is n(n+1)/2.",
    ],
    "T0": [
        "Summarize the following passage in one sentence: {passage}",
        "Classify the sentiment of: '{review}'",
        "Extract the named entities from: '{text}'",
        "Generate a title for this article: {passage}",
        "What is the main topic of: '{text}'?",
        "Rewrite in a formal style: '{sentence}'",
        "Does this imply that? Premise: {p} Hypothesis: {h}",
        "What is the relationship between {entity1} and {entity2}?",
        "Paraphrase: '{sentence}'",
        "Answer the question based on the context: Context: {ctx} Question: {q}",
    ],
    "NIV": [
        "Given the input '{input}', output a {output_type}.",
        "Identify whether '{phrase}' is positive, negative, or neutral.",
        "Convert the following to {format}: {input}",
        "Given a {category}, generate a {target}.",
        "Annotate the following text with POS tags: '{text}'",
        "Find all {entity_type} in: '{sentence}'",
        "Does '{sentence1}' entail '{sentence2}'?",
        "Generate a {style} description of {topic}.",
        "Given '{code}', what does this function return?",
        "Classify '{text}' into one of: {categories}.",
    ],
    "ShareGPT": [
        "Can you help me write a {doc_type} about {topic}?",
        "I need advice on {situation}. What should I do?",
        "Explain {concept} like I'm five years old.",
        "What are the pros and cons of {topic}?",
        "How do I fix this error: {error_msg}?",
        "Write me a Python function that {task}.",
        "I'm trying to learn {subject}. Where do I start?",
        "What's the difference between {a} and {b}?",
        "Can you review my {artifact}?",
        "Help me brainstorm ideas for {project}.",
    ],
}

_FILL = {
    "adj": ["capital", "largest", "most popular", "official", "national"],
    "noun": ["city", "language", "river", "mountain", "currency"],
    "place": ["France", "Japan", "Brazil", "Canada", "Germany"],
    "phrase": ["hello world", "good morning", "thank you", "please", "how are you"],
    "language": ["Spanish", "French", "German", "Italian", "Japanese"],
    "statement": ["The Earth revolves around the Sun", "2+2=5", "Water boils at 100°C"],
    "word": ["happy", "fast", "bright", "clear", "strong"],
    "currency": ["dollars", "euros", "yen", "pounds", "rupees"],
    "unit": ["centimeters", "millimeters", "seconds", "grams"],
    "bigger_unit": ["meter", "kilometer", "minute", "kilogram"],
    "category": ["mammals", "elements", "planets", "programming languages"],
    "property": ["living things", "used in industry", "discovered recently"],
    "event": ["World War II ended", "the moon landing", "the Internet was invented"],
    "person": ["Einstein", "Shakespeare", "Newton", "Darwin"],
    "role": ["scientist", "poet", "physicist", "biologist"],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "name2": ["Eve", "Frank", "Grace", "Henry"],
    "item": ["apple", "book", "coin", "marble"],
    "n1": ["3", "5", "7", "12", "15", "20", "25"],
    "n2": ["2", "4", "6", "8", "10", "3", "7"],
    "n3": ["10", "15", "20", "30", "45"],
    "speed": ["60", "80", "100", "120"],
    "time": ["2", "3", "5", "10"],
    "passage": [
        "The Industrial Revolution began in Britain in the 18th century and "
        "transformed manufacturing through mechanization.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
    ],
    "review": ["This product is amazing!", "Terrible experience, would not recommend.",
               "It's okay, nothing special."],
    "text": ["The Eiffel Tower is in Paris.", "Python is a programming language.",
             "The Amazon river flows through Brazil."],
    "ctx": ["The dog is black and white.", "The temperature is 37 Celsius."],
    "q": ["What color is the dog?", "Is the person feverish?"],
    "p": ["All birds can fly", "John is healthy"],
    "h": ["Eagles can fly", "John is sick"],
    "sentence": ["The cat sat on the mat.", "She runs every morning.", "He loves pizza."],
    "sentence1": ["The movie was great", "It was raining"],
    "sentence2": ["I enjoyed the film", "The streets were wet"],
    "entity1": ["Python", "JavaScript"], "entity2": ["web development", "data science"],
    "input": ["Hello World", "42", "2024-01-01"],
    "output_type": ["summary", "label", "number", "list"],
    "format": ["JSON", "CSV", "markdown", "XML"],
    "entity_type": ["dates", "names", "locations"],
    "code": ["def f(x): return x*2", "lambda x: x**2"],
    "categories": ["sports, science, politics", "positive, negative, neutral"],
    "doc_type": ["cover letter", "report", "email", "essay"],
    "topic": ["climate change", "machine learning", "healthy eating", "time management"],
    "situation": ["a job offer", "a difficult coworker", "moving abroad"],
    "concept": ["recursion", "gravity", "compound interest", "blockchain"],
    "error_msg": ["KeyError: 'name'", "IndexError: list out of range",
                  "TypeError: unsupported operand"],
    "task": ["reverse a string", "sort a list", "check if a number is prime"],
    "subject": ["machine learning", "Spanish", "Python", "investing"],
    "a": ["list and tuple", "Git and GitHub", "SQL and NoSQL"],
    "b": ["each other", "version control", "databases"],
    "artifact": ["code", "resume", "essay"],
    "project": ["a mobile app", "a blog post", "a business plan"],
    "style": ["technical", "informal", "poetic", "concise"],
    "target": ["definition", "example sentence", "analogy"],
    "category": ["programming language", "fruit", "country"],
    "statement": ["Paris is in France", "2+2=5", "The sky is blue"],
}

_RESPONSES = {
    "FLAN": [
        "The answer is {answer}.",
        "{answer}.",
        "Yes, that is correct.",
        "No, that is incorrect.",
        "The correct answer is {answer}.",
    ],
    "CoT": [
        "Step 1: Identify the given information. Step 2: Apply the formula. "
        "Step 3: Calculate. The answer is {answer}.",
        "Let's think through this. First, we know that {fact}. "
        "Therefore the answer is {answer}.",
        "We can solve this by {method}. The result is {answer}.",
    ],
    "T0": [
        "The sentiment is {sentiment}.",
        "Summary: {summary}",
        "The main topic is {topic}.",
        "Based on the context, the answer is {answer}.",
    ],
    "NIV": [
        "Output: {answer}",
        "The classification is: {answer}",
        "The annotated text is: {answer}",
        "Yes, this entails the hypothesis." if random.random() > 0.5 else "No, this does not entail.",
    ],
    "ShareGPT": [
        "Sure! Here's a draft {doc_type}: {content}. Let me know if you'd like changes.",
        "Great question! {concept} means {definition}. Here's an example: {example}",
        "I'd recommend {recommendation}. Here are the key steps: 1) {step1} 2) {step2}",
        "The main difference is that {a} focuses on {x}, while {b} focuses on {y}.",
    ],
}

_RESP_FILL = {
    "answer": ["42", "Paris", "True", "Python", "Newton", "1945", "Blue"],
    "fact": ["the sum equals the parts", "velocity equals distance over time"],
    "method": ["substitution", "factoring", "integration", "long division"],
    "sentiment": ["positive", "negative", "neutral"],
    "summary": ["This text discusses the key aspects of the topic.",
                "The passage explains the historical background."],
    "topic": ["science", "history", "technology", "economics"],
    "doc_type": ["cover letter", "email", "report"],
    "content": ["I am writing to express my interest in the position.",
                "Please find attached the requested documents."],
    "concept": ["recursion", "machine learning", "compound interest"],
    "definition": ["a function that calls itself", "learning from data",
                   "interest on interest"],
    "example": ["factorial(n) = n * factorial(n-1)", "spam filters use ML"],
    "recommendation": ["starting with the basics", "building a portfolio",
                       "taking an online course"],
    "step1": ["understand the fundamentals", "set clear goals"],
    "step2": ["practice regularly", "track your progress"],
    "a": ["Python", "Git"], "b": ["R", "SVN"],
    "x": ["data science", "version control"],
    "y": ["statistics", "deployment"],
}


def _render(template: str, fill_dict: dict) -> str:
    """Fill a template string with random choices from fill_dict."""
    import re
    keys = re.findall(r"\{(\w+)\}", template)
    result = template
    for k in keys:
        choices = fill_dict.get(k, [k])
        result = result.replace("{" + k + "}", random.choice(choices), 1)
    return result


def generate_synthetic_dataset(
    n_samples: int = 6000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a synthetic OpenOrca-like DataFrame with balanced classes.

    The text is varied enough to give TF-IDF meaningful signal while
    keeping generation fast (no external downloads needed).
    """
    random.seed(random_state)
    np.random.seed(random_state)

    classes = list(_TEMPLATES.keys())
    per_class = n_samples // len(classes)
    rows = []

    for cls in classes:
        templates  = _TEMPLATES[cls]
        resp_temps = _RESPONSES[cls]

        for i in range(per_class):
            q_template = random.choice(templates)
            r_template = random.choice(resp_temps)

            question = _render(q_template, _FILL)
            response = _render(r_template, _RESP_FILL)

            # Add some noise tokens to make the data less trivially separable
            noise_words = random.choices(
                ["please", "could", "I", "the", "a", "this", "that",
                 "can", "you", "help", "me", "what", "is", "how"],
                k=random.randint(0, 3),
            )
            if noise_words and random.random() < 0.3:
                question = " ".join(noise_words) + " " + question.lower()

            rows.append({
                "id":               f"{cls.lower()}.{i:05d}",
                "system_prompt":    "You are a helpful assistant.",
                "question":         question,
                "response":         response,
                "source_label":     cls,
                "complexity_label": (
                    "SHORT"  if len(response.split()) < 30
                    else "MEDIUM" if len(response.split()) < 80
                    else "LONG"
                ),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info(
        "Generated %d synthetic samples | classes: %s",
        len(df), df["source_label"].value_counts().to_dict(),
    )
    return df


# ── Training pipeline ─────────────────────────────────────────────────────────

def train_demo(quick: bool = False, rebuild: bool = False) -> dict:
    """End-to-end demo training. Returns the saved bundle dict."""
    import joblib

    from src.data.preprocessor import preprocess, split
    from src.features.engineer import build_feature_pipeline, get_feature_names
    from src.models.evaluator import build_bundle

    if BUNDLE_PATH.exists() and not rebuild:
        logger.info("Bundle already exists at %s — skipping (use --rebuild to force).", BUNDLE_PATH)
        return joblib.load(BUNDLE_PATH)

    n_samples = 2000 if quick else 6000
    n_iter    = 2    if quick else 8
    cv_folds  = 2    if quick else 5

    logger.info("Generating synthetic dataset (%d samples) …", n_samples)
    df_raw = generate_synthetic_dataset(n_samples=n_samples)

    logger.info("Preprocessing …")
    df_proc, le = preprocess(df_raw, target="source_label")

    logger.info("Splitting train/val/test …")
    df_train, df_val, df_test = split(df_proc, test_size=0.15, val_size=0.15)

    # Fit feature pipeline for name extraction
    feat_pipe = build_feature_pipeline()
    feat_pipe.fit(df_train)
    feature_names = get_feature_names(feat_pipe)
    logger.info("Features: %d", len(feature_names))

    # ── Train models
    label_names = list(le.classes_)
    results: dict = {}

    model_specs = [
        ("LogisticRegression", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ("LinearSVC", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42))),
    ]

    # Optionally add ensemble models
    try:
        from lightgbm import LGBMClassifier
        model_specs.append(
            ("LightGBM", LGBMClassifier(
                n_estimators=200 if not quick else 50,
                max_depth=6, learning_rate=0.1,
                num_leaves=31, random_state=42,
                n_jobs=-1, verbose=-1,
            ))
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        model_specs.append(
            ("XGBoost", XGBClassifier(
                n_estimators=200 if not quick else 50,
                max_depth=4, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
                eval_metric="mlogloss",
            ))
        )
    except ImportError:
        pass

    from sklearn.metrics import f1_score

    for name, clf in model_specs:
        logger.info("Training %s …", name)
        full_pipe = Pipeline([
            ("features", feat_pipe.named_steps["features"]),
            ("clf",      clf),
        ])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            full_pipe, df_train, df_train["target"],
            cv=cv, scoring="f1_weighted", n_jobs=-1,
        )
        full_pipe.fit(df_train, df_train["target"])
        y_pred = full_pipe.predict(df_val)
        val_f1 = f1_score(df_val["target"], y_pred, average="weighted")
        logger.info("  %s — CV=%.4f  val_F1=%.4f", name, cv_scores.mean(), val_f1)

        results[name] = {
            "pipeline":    full_pipe,
            "best_params": {},
            "cv_scores":   cv_scores,
            "val_score":   float(val_f1),
            "cv_score":    float(cv_scores.mean()),
        }

    # Sort by val F1
    results = dict(sorted(results.items(), key=lambda x: x[1]["val_score"], reverse=True))

    logger.info("Building evaluation bundle …")
    bundle = build_bundle(
        results=results,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        label_encoder=le,
        feature_names=feature_names,
        config={
            "target": "source_label",
            "sample_size": len(df_raw),
            "synthetic": True,
            "quick": quick,
        },
    )

    # Also save to explicit demo path so dashboard can find it
    import joblib, shutil
    demo_path = BUNDLE_PATH
    main_path = ROOT / "models" / "model_bundle.pkl"
    shutil.copy(main_path, demo_path)
    logger.info("Demo bundle saved → %s", demo_path)

    return bundle


# ── Self-healing loader for the dashboard ─────────────────────────────────────

def ensure_bundle(force: bool = False) -> dict:
    """
    Load the existing bundle or silently rebuild it if missing/corrupt.
    Called by the Streamlit dashboard on startup.
    """
    import joblib

    for path in (BUNDLE_PATH, ROOT / "models" / "model_bundle.pkl"):
        if path.exists() and not force:
            try:
                b = joblib.load(path)
                if "models" in b and "metrics" in b:
                    return b
            except Exception:
                pass  # corrupt — fall through to rebuild

    logger.warning("Bundle missing or corrupt — rebuilding from synthetic data …")
    return train_demo(quick=False, rebuild=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train demo model bundle (synthetic data)")
    p.add_argument("--quick",   action="store_true", help="Fast run with reduced data")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild even if bundle exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    bundle = train_demo(quick=args.quick, rebuild=args.rebuild)
    best   = bundle["best_model_name"]
    f1     = bundle["metrics"][best]["f1_weighted"]
    print(f"\n✓  Demo training complete.")
    print(f"   Best model : {best}")
    print(f"   Test F1    : {f1:.4f}")
    print(f"   Bundle     : {BUNDLE_PATH}")
    print(f"\n   Start dashboard: streamlit run app/dashboard.py")
