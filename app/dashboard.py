"""
app/dashboard.py
────────────────
Streamlit dashboard for the OpenOrca Query Intelligence project.

Tabs
────
  1. Overview          — project summary, dataset stats, class distribution
  2. Model Comparison  — metrics table, grouped bar chart, CV box plots
  3. Analytics         — text length distributions, per-class deep-dives
  4. Pipeline & API    — architecture diagram, feature overview, code snippets
  5. 🔮 Predict        — live classification form with probability bars

Run with:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.visualization.plots import (
    metrics_summary_table,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_feature_importance,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_prediction_proba,
    plot_text_length_by_class,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenOrca Query Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bundle loader (cached, self-healing) ──────────────────────────────────────
BUNDLE_PATH     = ROOT / "models" / "demo_bundle.pkl"
FALLBACK_PATH   = ROOT / "models" / "model_bundle.pkl"


@st.cache_resource(show_spinner="Loading model bundle …")
def load_bundle():
    """Try every known bundle path; rebuild from synthetic data if all fail."""
    for path in (BUNDLE_PATH, FALLBACK_PATH):
        if path.exists():
            try:
                b = joblib.load(path)
                if "models" in b and "metrics" in b:
                    return b, None
            except Exception as exc:
                pass  # corrupt — try next

    # Self-heal: rebuild with synthetic data
    try:
        from train_demo import train_demo
        # Use quick rebuild on cloud/startup to avoid long first-load times.
        b = train_demo(quick=True, rebuild=True)
        return b, "rebuilt"
    except Exception as exc:
        return None, str(exc)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 OpenOrca\nQuery Intelligence")
    st.markdown("---")
    st.markdown(
        """
        **Multi-class NLP classifier** trained on the
        [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
        dataset.

        **Task:** Predict the *source* of a query  
        (FLAN · T0 · CoT · NIV · ShareGPT)  
        from the question text alone.

        ---
        **Stack:** scikit-learn · XGBoost · LightGBM · Streamlit  
        """
    )

    bundle, bundle_status = load_bundle()
    if bundle:
        if bundle_status == "rebuilt":
            st.info("Bundle rebuilt from synthetic data ✓")
        else:
            st.success("Bundle loaded ✓")
        st.caption(f"Trained: {bundle.get('training_date', 'N/A')[:10]}")
        if bundle.get("config", {}).get("synthetic"):
            st.caption("🔴 Using synthetic demo data")
        best = bundle["best_model_name"]
        best_f1 = bundle["metrics"][best]["f1_weighted"]
        st.metric("Best Model", best, f"F1={best_f1:.4f}")
        st.metric("Classes", len(bundle["label_names"]))
    else:
        st.error(
            f"Could not load or rebuild the model bundle.\n\n"
            f"{bundle_status}\n\n"
            "Run `python train_demo.py` manually."
        )

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_overview, tab_models, tab_analytics, tab_pipeline, tab_predict = st.tabs([
    "📊 Overview",
    "🏆 Model Comparison",
    "🔍 Analytics",
    "⚙️ Pipeline & API",
    "🔮 Predict / Classify",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Project Overview", divider="blue")

    st.markdown(
        """
        ### Problem Statement
        Given a **user question** from the OpenOrca dataset, predict which **instruction
        dataset** it originally came from (FLAN, T0, CoT, NIV, …).  
        This is a **multi-class text classification** problem — a fundamental NLP task
        that demonstrates the full ML lifecycle: data ingestion → feature engineering →
        model comparison → deployment.

        ### Why This Matters
        - Understanding *source* distributions helps evaluate dataset quality & bias  
        - A deployed classifier can automatically tag incoming prompts for routing  
        - The pipeline generalises to any text classification domain
        """
    )

    if bundle is None:
        st.info("Train the models to see live dataset statistics here.")
        st.stop()

    analytics   = bundle["analytics"]
    sample_df   = bundle["sample_df"]
    label_names = bundle["label_names"]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",  f"{analytics['total_samples']:,}")
    c2.metric("Training Set",   f"{analytics['train_size']:,}")
    c3.metric("Test Set",       f"{analytics['test_size']:,}")
    c4.metric("Classes",        len(label_names))

    st.markdown("---")
    col_dist, col_stats = st.columns([1, 1])

    with col_dist:
        fig = plot_class_distribution(
            analytics["class_distribution"], "Query Source Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        if "complexity_distribution" in analytics and analytics["complexity_distribution"]:
            fig2 = plot_class_distribution(
                analytics["complexity_distribution"], "Response Complexity Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("#### Dataset Sample")
            st.dataframe(
                sample_df[["question", "label_name"]]
                .rename(columns={"question": "Question", "label_name": "Source"})
                .head(10),
                use_container_width=True,
            )

    st.markdown("#### Sample Questions by Class")
    selected_class = st.selectbox("Select a class to preview", options=label_names)
    mask = sample_df["label_name"] == selected_class
    st.dataframe(
        sample_df[mask][["question", "label_name"]].head(8)
        .rename(columns={"question": "Question", "label_name": "Source"}),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.header("Model Comparison", divider="green")

    if bundle is None:
        st.info("Run training first to see model results.")
        st.stop()

    # Summary table
    df_summary = metrics_summary_table(bundle["metrics"])
    st.markdown("#### Test-Set Performance Summary")
    st.dataframe(
        df_summary.style.background_gradient(
            cmap="YlGn", subset=["F1 Weighted", "Accuracy", "F1 Macro"]
        ).format("{:.4f}", subset=df_summary.columns[1:]),
        use_container_width=True,
    )

    # Grouped bar comparison
    st.plotly_chart(
        plot_model_comparison(bundle["metrics"]),
        use_container_width=True,
    )

    # CV box plots
    if bundle.get("cv_results"):
        st.plotly_chart(
            plot_cv_scores(bundle["cv_results"]),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    model_sel = st.selectbox(
        "Select model", options=list(bundle["models"].keys()), key="cm_model"
    )
    if model_sel in bundle["confusion_matrices"]:
        fig_cm = plot_confusion_matrix(
            bundle["confusion_matrices"][model_sel],
            bundle["label_names"],
            title=f"Confusion Matrix — {model_sel}",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Per-Class Metrics")
    if model_sel in bundle["metrics"]:
        report = bundle["metrics"][model_sel].get("classification_report", {})
        fig_pc = plot_per_class_metrics(report, model_name=model_sel)
        st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("#### Feature Importance")
    if bundle.get("feature_importance"):
        fi_models = list(bundle["feature_importance"].keys())
        fi_sel = st.selectbox("Select model", fi_models, key="fi_sel")
        if fi_sel in bundle["feature_importance"]:
            fig_fi = plot_feature_importance(
                bundle["feature_importance"][fi_sel],
                title=f"Top Features — {fi_sel}",
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("Data Analytics", divider="orange")

    if bundle is None:
        st.info("Run training first.")
        st.stop()

    sample_df = bundle["sample_df"]

    # Text length plots
    st.markdown("#### Question Length by Class")
    fig_len = plot_text_length_by_class(sample_df, label_col="label_name")
    st.plotly_chart(fig_len, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Word Count Distribution")
        sample_df["word_count"] = sample_df["question"].str.split().str.len()
        fig_wc = px.histogram(
            sample_df, x="word_count", color="label_name",
            nbins=60, barmode="overlay", opacity=0.65,
            title="Question Word Count per Class",
            labels={"word_count": "Word Count", "label_name": "Source"},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_wc.update_layout(height=380)
        st.plotly_chart(fig_wc, use_container_width=True)

    with col2:
        st.markdown("#### Question Length (chars)")
        sample_df["char_len"] = sample_df["question"].str.len()
        fig_cl = px.box(
            sample_df, x="label_name", y="char_len",
            color="label_name",
            title="Character Length by Source",
            labels={"char_len": "Char Length", "label_name": "Source"},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_cl.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_cl, use_container_width=True)

    if "complexity_label" in sample_df.columns:
        st.markdown("#### Complexity × Source Cross-tab")
        ct = pd.crosstab(sample_df["label_name"], sample_df["complexity_label"])
        fig_cross = px.imshow(
            ct, text_auto=True, aspect="auto",
            title="Response Complexity per Query Source",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        st.plotly_chart(fig_cross, use_container_width=True)

    # ── Filterable data table
    st.markdown("#### Explore the Data")
    filt_class = st.multiselect(
        "Filter by class", options=bundle["label_names"],
        default=bundle["label_names"][:2],
    )
    filt_df = sample_df[sample_df["label_name"].isin(filt_class)] if filt_class else sample_df
    st.dataframe(
        filt_df[["question", "label_name"]].head(200)
        .rename(columns={"question": "Question", "label_name": "Source"}),
        use_container_width=True, height=320,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PIPELINE & API
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.header("Pipeline & API", divider="violet")

    st.markdown(
        """
        ### System Architecture

        ```
        ┌──────────────────────────────────────────────────────────────┐
        │                   OpenOrca Raw Dataset                        │
        │          (HuggingFace  ·  ~4.3M instruction pairs)           │
        └───────────────────────────┬──────────────────────────────────┘
                                    │  src/data/loader.py
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │           Preprocessing  (src/data/preprocessor.py)          │
        │  • HTML/URL stripping  • Length filtering  • Label encoding  │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │        Feature Engineering  (src/features/engineer.py)       │
        │  TF-IDF (25k features, 1-2 grams)  +  12 stat features      │
        │              FeatureUnion  →  sparse matrix                   │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │          Model Training  (src/models/trainer.py)             │
        │  RandomizedSearchCV  ·  StratifiedKFold (5-fold)             │
        │  LogisticRegression · RandomForest · XGBoost                 │
        │  LightGBM · LinearSVC (CalibratedCV)                         │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │         Evaluation & Bundle  (src/models/evaluator.py)       │
        │  Accuracy · F1 · ROC-AUC · Confusion Matrix · FI            │
        │       Saved to  models/model_bundle.pkl  (joblib)           │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
        ┌─────────────────────┐          ┌─────────────────────┐
        │  Streamlit Dashboard│          │   CLI — predict.py  │
        │  app/dashboard.py   │          │  --text / --file /  │
        │                     │          │  --interactive       │
        └─────────────────────┘          └─────────────────────┘
        ```
        """
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Statistical Features")
        feat_table = pd.DataFrame({
            "Feature": [
                "q_char_len", "q_word_count", "q_sentence_count",
                "q_avg_word_len", "q_unique_word_ratio", "q_punct_count",
                "q_digit_count", "q_upper_ratio", "q_question_mark_count",
                "q_has_code", "resp_word_count", "resp_sentence_count",
            ],
            "Description": [
                "Character length of question",
                "Word count of question",
                "Estimated sentence count",
                "Average word length",
                "Type-token ratio",
                "Punctuation mark count",
                "Digit character count",
                "Uppercase character ratio",
                "Number of '?' characters",
                "Contains code snippet (1/0)",
                "Word count of model response",
                "Sentence count of response",
            ],
        })
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### TF-IDF Configuration")
        if bundle:
            tfidf_cfg = bundle.get("config", {}).get("feature_config", {})
            st.json({
                "max_features": tfidf_cfg.get("tfidf_max_features", 25000),
                "ngram_range":  tfidf_cfg.get("tfidf_ngram_range",  [1, 2]),
                "sublinear_tf": tfidf_cfg.get("tfidf_sublinear_tf", True),
                "min_df":       tfidf_cfg.get("tfidf_min_df", 3),
                "max_df":       tfidf_cfg.get("tfidf_max_df", 0.95),
            })

    st.markdown("---")
    st.markdown("#### Python API — Programmatic Usage")
    st.code(
        '''
from predict import load_bundle, predict_single, predict_batch

# Load the trained bundle once
bundle = load_bundle("models/model_bundle.pkl")

# Single prediction
result = predict_single(
    "Explain the proof of the Pythagorean theorem step by step.",
    bundle=bundle,
)
print(result["predicted_class"])   # → "CoT"
print(result["probabilities"])     # {"CoT": 0.72, "FLAN": 0.15, ...}

# Batch prediction
questions = ["What is 2+2?", "Write a poem about the ocean."]
results = predict_batch(questions, bundle=bundle)
''',
        language="python",
    )

    st.markdown("#### CLI Usage")
    st.code(
        """
# Single question
python predict.py --text "Describe the water cycle."

# File with one question per line
python predict.py --file questions.txt --model LightGBM

# Interactive REPL
python predict.py --interactive
""",
        language="bash",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT / CLASSIFY  🔮
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("🔮 Live Query Classifier", divider="rainbow")

    if bundle is None:
        st.warning(
            "Model bundle not found. Please run `python train.py` to train "
            "the models, then refresh this page."
        )
        st.stop()

    st.markdown(
        "Type any question below and the classifier will predict which "
        "instruction dataset it most likely originates from."
    )

    # ── Model selector
    model_options = list(bundle["models"].keys())
    best_idx      = model_options.index(bundle["best_model_name"])
    chosen_model  = st.selectbox(
        "Model", model_options, index=best_idx,
        help="Select which trained model to run inference with.",
    )

    # ── Input form
    with st.form("prediction_form", clear_on_submit=False):
        user_input = st.text_area(
            "Enter your question",
            placeholder="e.g.  What are the main causes of World War I?",
            height=140,
        )
        submitted = st.form_submit_button("🔮 Classify", use_container_width=True)

    if submitted:
        if not user_input.strip():
            st.error("Please enter a question before classifying.")
        else:
            with st.spinner("Running inference …"):
                from predict import predict_single as _predict
                result = _predict(user_input, bundle=bundle, model_name=chosen_model)

            pred_class = result["predicted_class"]
            proba_dict = result["probabilities"]
            best_prob  = max(proba_dict.values())

            # ── Result hero
            res_col, info_col = st.columns([3, 2])
            with res_col:
                st.success(f"### Predicted Class: **{pred_class}**")
                conf_color = "green" if best_prob > 0.6 else "orange" if best_prob > 0.4 else "red"
                st.markdown(
                    f"<span style='font-size:1.1rem'>Confidence: "
                    f"<b style='color:{conf_color}'>{best_prob:.1%}</b></span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"*Model: {chosen_model}*")

            with info_col:
                st.markdown("**Class Descriptions**")
                class_desc = {
                    "FLAN":     "FLAN (Fine-tuned LAnguage Net) style instructions",
                    "T0":       "T0 multi-task training prompts (BigScience)",
                    "CoT":      "Chain-of-Thought reasoning questions",
                    "NIV":      "Natural Instructions V2 diverse NLP tasks",
                    "ShareGPT": "Real human–GPT conversations",
                    "CoD":      "Code description / programming tasks",
                }
                desc = class_desc.get(pred_class, "Model instruction dataset source")
                st.info(f"**{pred_class}**: {desc}")

            # ── Probability bar chart
            proba_arr  = np.array([proba_dict.get(l, 0.0) for l in bundle["label_names"]])
            fig_proba  = plot_prediction_proba(proba_arr, bundle["label_names"])
            st.plotly_chart(fig_proba, use_container_width=True)

            # ── Input summary expander
            with st.expander("📋 Input Summary & raw probabilities"):
                st.markdown(f"**Raw input length:** {len(user_input)} characters · "
                            f"{len(user_input.split())} words")
                st.markdown("**Full probability distribution:**")
                prob_df = pd.DataFrame(
                    {"Class": list(proba_dict.keys()), "Probability": list(proba_dict.values())}
                ).sort_values("Probability", ascending=False).reset_index(drop=True)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # ── Example questions
    st.markdown("---")
    st.markdown("#### Try these example questions")
    examples = {
        "FLAN style": "What is the Spanish word for 'happy'?",
        "Chain-of-Thought": "If I have 5 apples and give 2 to my friend, then buy 3 more, how many do I have?",
        "NIV style": "Summarize the following passage in one sentence: The quick brown fox …",
        "Coding": "Write a Python function to compute the Fibonacci sequence using recursion.",
        "Creative / T0": "Write a short poem about autumn leaves falling in a city park.",
    }
    for label, ex_text in examples.items():
        if st.button(f"**{label}**  — {ex_text[:60]}…" if len(ex_text) > 60 else f"**{label}**  — {ex_text}"):
            with st.spinner("Classifying …"):
                from predict import predict_single as _predict2
                r = _predict2(ex_text, bundle=bundle, model_name=chosen_model)
            st.info(
                f"**Predicted:** {r['predicted_class']}  "
                f"(confidence {max(r['probabilities'].values()):.1%})"
            )
