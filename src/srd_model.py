"""
LR-level delay-risk model for SRD Logistics.

Predicts whether an LR will miss its target days, using only information known at
booking time (route, hub/via, business, consignor, weight, booking time, target days).
Trained on delivered LRs; used to score LRs still in the network and to explain why an
LR is at risk (model-agnostic one-feature-at-a-time ablation).
"""

import datetime as dt
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "srd_delay_risk_model.joblib")

NUMERIC = ["Weight_KG", "Articles", "Distance_KM", "Target_Days", "Book_Hour", "Book_DOW", "Book_Month"]
CATEGORICAL = ["Source", "Destination", "Via", "Main_Hub", "Business", "Consignor"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Delayed"


def add_booking_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["Book_Hour"] = out["Booking_DT"].dt.hour
    out["Book_DOW"] = out["Booking_DT"].dt.dayofweek
    out["Book_Month"] = out["Booking_DT"].dt.month
    return out


def _pipe(clf):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=30), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def train(d: pd.DataFrame, random_state: int = 42) -> dict:
    """d: prepared LR frame (srd_analytics.prepare). Trains on delivered LRs only."""
    t = add_booking_features(d[d["Delivered"]])
    X, y = t[FEATURES], t[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    cands = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Gradient Boosting (HGB)": HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08,
                                                                  class_weight="balanced", random_state=random_state),
    }
    res = {}
    for name, clf in cands.items():
        p = _pipe(clf).fit(Xtr, ytr)
        proba = p.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        res[name] = {"pipeline": p, "metrics": {
            "roc_auc": round(float(roc_auc_score(yte, proba)), 4), "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, pred, zero_division=0)), 4), "f1": round(float(f1_score(yte, pred, zero_division=0)), 4)}}
    best = max(res, key=lambda k: res[k]["metrics"]["roc_auc"])
    return {
        "pipeline": res[best]["pipeline"], "model_name": best, "metrics": res[best]["metrics"],
        "all_model_metrics": {k: v["metrics"] for k, v in res.items()},
        "base_rate": round(float(y.mean()), 4), "training_records": len(Xtr), "test_records": len(Xte),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "baseline": {c: (float(X[c].median()) if c in NUMERIC else X[c].mode().iloc[0]) for c in FEATURES},
        "X_test": Xte, "y_test": yte,
    }


def save(payload: dict, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(payload, path)


def load(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


def score(payload: dict, d: pd.DataFrame) -> np.ndarray:
    return payload["pipeline"].predict_proba(add_booking_features(d)[FEATURES])[:, 1]


def risk_level(p: float) -> str:
    return "LOW" if p < 0.15 else ("MEDIUM" if p < 0.30 else ("HIGH" if p < 0.50 else "CRITICAL"))


def global_importance(payload: dict, n: int = 5000, seed: int = 0) -> pd.DataFrame:
    X, y = payload["X_test"], payload["y_test"]
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    r = permutation_importance(payload["pipeline"], X.iloc[idx], y.iloc[idx], scoring="roc_auc", n_repeats=3, random_state=seed)
    out = pd.DataFrame({"Feature": FEATURES, "Importance": r.importances_mean}).sort_values("Importance", ascending=False)
    out["Importance"] = out["Importance"].clip(lower=0).round(4)
    return out.reset_index(drop=True)


def explain_lr(payload: dict, row: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Risk change when each feature is swapped for a typical (baseline) value."""
    base = add_booking_features(row)[FEATURES].iloc[[0]]
    p0 = float(payload["pipeline"].predict_proba(base)[:, 1][0])
    variants = []
    for f in FEATURES:
        v = base.copy()
        v[f] = payload["baseline"][f]
        variants.append(v)
    pv = payload["pipeline"].predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    out = pd.DataFrame({"Feature": FEATURES, "Value": [base[f].iloc[0] for f in FEATURES], "Typical": [payload["baseline"][f] for f in FEATURES],
                        "Risk_Impact": p0 - pv})
    out["Direction"] = np.where(out["Risk_Impact"] > 0, "Raises risk", "Lowers risk")
    return out.reindex(out["Risk_Impact"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)
