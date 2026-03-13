"""
src/visualization/plots.py
──────────────────────────
Plotly-based plotting helpers used by both the training reports and the
Streamlit dashboard.  Every function returns a plotly Figure object so
callers decide how to render or save it.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2


# ── Model comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(metrics: dict) -> go.Figure:
    """
    Grouped bar chart comparing Accuracy, F1 Weighted, F1 Macro, Precision,
    Recall across all models.
    """
    metric_keys = [
        ("accuracy",           "Accuracy"),
        ("f1_weighted",        "F1 Weighted"),
        ("f1_macro",           "F1 Macro"),
        ("precision_weighted", "Precision"),
        ("recall_weighted",    "Recall"),
    ]
    model_names = list(metrics.keys())
    fig = go.Figure()

    for key, label in metric_keys:
        values = [metrics[m].get(key, 0) for m in model_names]
        fig.add_trace(go.Bar(
            name=label, x=model_names, y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis=dict(title="Score", range=[0, 1.05]),
        xaxis_title="Model",
        legend_title="Metric",
        height=480,
        template="plotly_white",
    )
    return fig


def plot_cv_scores(cv_scores: dict) -> go.Figure:
    """Box-plot of cross-validation score distributions per model."""
    fig = go.Figure()
    for i, (name, scores) in enumerate(cv_scores.items()):
        if len(scores) == 0:
            continue
        fig.add_trace(go.Box(
            y=scores, name=name,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
        ))
    fig.update_layout(
        title="Cross-Validation Score Distribution",
        yaxis_title="CV Score (Weighted F1)",
        height=420,
        template="plotly_white",
    )
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, label_names: list, title: str = "") -> go.Figure:
    """Annotated heatmap of a confusion matrix (normalised by true class)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    text = [[f"{cm_norm[i,j]:.2f}<br>({cm[i,j]})"
             for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=cm_norm[::-1],           # flip so (0,0) is top-left
        x=label_names,
        y=label_names[::-1],
        text=text[::-1],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=title or "Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        template="plotly_white",
    )
    return fig


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(df_imp: pd.DataFrame, title: str = "") -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    df_plot = df_imp.sort_values("importance").tail(20)
    fig = px.bar(
        df_plot, x="importance", y="feature", orientation="h",
        title=title or "Top Feature Importances",
        color="importance",
        color_continuous_scale="teal",
        template="plotly_white",
    )
    fig.update_layout(height=520, yaxis_title="", coloraxis_showscale=False)
    return fig


# ── Class distribution ────────────────────────────────────────────────────────

def plot_class_distribution(dist: dict, title: str = "Class Distribution") -> go.Figure:
    labels = list(dist.keys())
    values = list(dist.values())
    fig = px.pie(
        names=labels, values=values, title=title,
        color_discrete_sequence=PALETTE,
        template="plotly_white",
        hole=0.35,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(height=420)
    return fig


# ── Text length distribution ──────────────────────────────────────────────────

def plot_text_length_by_class(df: pd.DataFrame, label_col: str = "label_name") -> go.Figure:
    """Box plot of question character length grouped by class."""
    df = df.copy()
    df["q_len"] = df["question"].str.len()
    fig = px.box(
        df, x=label_col, y="q_len", color=label_col,
        title="Question Length Distribution by Class",
        labels={"q_len": "Character Length", label_col: "Class"},
        color_discrete_sequence=PALETTE,
        template="plotly_white",
    )
    fig.update_layout(height=440, showlegend=False)
    return fig


# ── Per-class metrics ─────────────────────────────────────────────────────────

def plot_per_class_metrics(report: dict, model_name: str = "") -> go.Figure:
    """
    Grouped bar chart of precision / recall / F1 per class from a
    classification_report dict (as returned by sklearn).
    """
    rows = {
        k: v for k, v in report.items()
        if isinstance(v, dict) and "f1-score" in v
        and k not in ("macro avg", "weighted avg", "accuracy")
    }
    classes   = list(rows.keys())
    precision = [rows[c]["precision"] for c in classes]
    recall    = [rows[c]["recall"]    for c in classes]
    f1        = [rows[c]["f1-score"]  for c in classes]

    fig = go.Figure()
    for vals, name in [(precision, "Precision"), (recall, "Recall"), (f1, "F1")]:
        fig.add_trace(go.Bar(
            name=name, x=classes, y=vals,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title=f"Per-Class Metrics — {model_name}",
        yaxis=dict(title="Score", range=[0, 1.1]),
        xaxis_title="Class",
        height=440,
        template="plotly_white",
    )
    return fig


# ── Probability bar (single prediction) ──────────────────────────────────────

def plot_prediction_proba(proba: np.ndarray, label_names: list) -> go.Figure:
    """Horizontal bar chart of class probabilities for a single prediction."""
    idx  = np.argsort(proba)
    fig  = go.Figure(go.Bar(
        x=proba[idx], y=[label_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=proba[idx],
            colorscale="Teal",
        ),
        text=[f"{p:.1%}" for p in proba[idx]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis=dict(title="Probability", range=[0, 1.1]),
        yaxis_title="",
        height=360,
        template="plotly_white",
    )
    return fig


# ── Training / val / test metric summary table ────────────────────────────────

def metrics_summary_table(metrics: dict) -> pd.DataFrame:
    """Convert the full metrics dict to a tidy summary DataFrame."""
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "Model":      model_name,
            "Accuracy":   round(m.get("accuracy", 0), 4),
            "F1 Weighted": round(m.get("f1_weighted", 0), 4),
            "F1 Macro":   round(m.get("f1_macro", 0), 4),
            "Precision":  round(m.get("precision_weighted", 0), 4),
            "Recall":     round(m.get("recall_weighted", 0), 4),
            "ROC-AUC":    round(m.get("roc_auc_ovr", float("nan")), 4),
        })
    return pd.DataFrame(rows).sort_values("F1 Weighted", ascending=False)
