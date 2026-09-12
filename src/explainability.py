"""
Model explainability: per-shipment risk-factor attribution.

Uses SHAP TreeExplainer when the trained model is tree-based (Random Forest /
HistGradientBoosting). Falls back to a model-agnostic permutation-style
"contribution" approximation (feature importance x deviation-from-typical)
if SHAP is unavailable or unsupported for the fitted model type, so the app
never breaks regardless of which model won model selection.
"""

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _readable_feature_name(raw_name: str) -> str:
    name = raw_name
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("_", " ")
    return name


def _transform(pipeline, X: pd.DataFrame):
    prep = pipeline.named_steps["prep"]
    return prep.transform(X)


def explain_instance(model_payload: dict, X_row: pd.DataFrame, top_k: int = 5) -> list:
    """Return top_k contributing factors [{feature, value, impact, direction}] for one shipment."""
    pipeline = model_payload["pipeline"]
    clf = pipeline.named_steps["clf"]
    feature_names = list(pipeline.named_steps["prep"].get_feature_names_out())

    X_t = _transform(pipeline, X_row)

    contributions = None
    if _SHAP_AVAILABLE and clf.__class__.__name__ in ("RandomForestClassifier", "HistGradientBoostingClassifier"):
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_t)
            if isinstance(sv, list):  # RF binary -> [class0, class1]
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:  # (n, features, classes)
                sv = sv[:, :, 1]
            contributions = sv[0]
        except Exception:
            contributions = None

    if contributions is None:
        contributions = _fallback_contributions(pipeline, X_t)

    order = np.argsort(-np.abs(contributions))[:top_k]
    factors = []
    for idx in order:
        factors.append({
            "feature": _readable_feature_name(feature_names[idx]),
            "impact": float(contributions[idx]),
            "direction": "increases risk" if contributions[idx] > 0 else "decreases risk",
        })
    return factors


def _fallback_contributions(pipeline, X_t: np.ndarray) -> np.ndarray:
    """Model-agnostic approximation: global feature importance (or |coef|) scaled
    by how far this instance's value sits from the training-set mean for that
    (encoded) feature. Used when SHAP is unavailable/unsupported."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
    else:
        importance = np.ones(X_t.shape[1])

    row = X_t[0]
    baseline = np.zeros_like(row)  # standardized/one-hot space -> 0 is the "typical/absent" reference
    deviation = row - baseline
    return importance * deviation


def top_factors_for_dataframe(model_payload: dict, X: pd.DataFrame, top_k: int = 5) -> list:
    """Convenience: explain each row in X, returns list of factor-lists."""
    return [explain_instance(model_payload, X.iloc[[i]], top_k=top_k) for i in range(len(X))]


def global_feature_importance(model_payload: dict, X_sample: pd.DataFrame = None,
                               y_sample: pd.Series = None, top_k: int = 15) -> pd.DataFrame:
    """Global feature importance. Prefers mean |SHAP value| over a sample
    (works for any model type, incl. HistGradientBoosting which has no
    native feature_importances_). Falls back to native importances / |coef|,
    then to sklearn permutation importance if X_sample/y_sample are given."""
    pipeline = model_payload["pipeline"]
    clf = pipeline.named_steps["clf"]
    feature_names = list(pipeline.named_steps["prep"].get_feature_names_out())
    importance = None

    if _SHAP_AVAILABLE and X_sample is not None and clf.__class__.__name__ in (
        "RandomForestClassifier", "HistGradientBoostingClassifier"
    ):
        try:
            sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
            X_t = _transform(pipeline, sample)
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_t)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            importance = np.abs(sv).mean(axis=0)
        except Exception:
            importance = None

    if importance is None and hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif importance is None and hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])

    if importance is None and X_sample is not None and y_sample is not None:
        from sklearn.inspection import permutation_importance
        sample_idx = X_sample.sample(min(500, len(X_sample)), random_state=42).index
        result = permutation_importance(
            pipeline, X_sample.loc[sample_idx], y_sample.loc[sample_idx],
            n_repeats=5, random_state=42, scoring="roc_auc",
        )
        importance = result.importances_mean
        feature_names = list(X_sample.columns)  # permutation importance is on raw features

    if importance is None:
        importance = np.zeros(len(feature_names))

    df = pd.DataFrame({
        "Feature": [_readable_feature_name(f) for f in feature_names],
        "Importance": importance,
    }).sort_values("Importance", ascending=False).head(top_k).reset_index(drop=True)
    return df
