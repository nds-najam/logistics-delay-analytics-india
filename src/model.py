"""
Delay-prediction model: feature engineering, training, evaluation, and
model selection. Predicts Delay_Flag using only information that is known
(or reasonably estimable) at booking/dispatch time -- i.e. no leakage from
post-delivery fields such as realized component hours or Delay_Hours itself.
"""

import os
import datetime as dt

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "delay_risk_model.joblib")

TARGET = "Delay_Flag"

# Features known/estimable at booking or dispatch time (no post-delivery leakage)
NUMERIC_FEATURES = [
    "Package_Weight", "Package_Volume", "Distance_KM", "Promised_Delivery_Days",
    "Number_of_Handoffs", "Warehouse_Capacity_Utilization",
]
CATEGORICAL_FEATURES = [
    "Origin_Region", "Destination_Region", "Origin_Warehouse", "Destination_Warehouse",
    "Shipping_Mode", "Carrier", "Service_Type", "Package_Type", "Customer_Priority",
    "Weather_Condition", "Traffic_Level", "Vehicle_Availability", "Driver_Availability",
    "Address_Quality",
]
BINARY_FEATURES = [
    "Holiday_Flag", "Weekend_Flag", "Peak_Season_Flag", "Strike_Disruption_Flag",
    "COD_Flag", "Monsoon_Disruption_Flag",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def prepare_features(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def _build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def _candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting (HGB)": HistGradientBoostingClassifier(
            max_iter=250, max_depth=8, learning_rate=0.08, class_weight="balanced", random_state=42,
        ),
    }


def _evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    for name, clf in _candidate_models().items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, y_pred, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[name] = {
            "pipeline": pipe, "metrics": metrics,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        }

    # Model selection: weight recall (missing a delay is costlier than a false alarm),
    # but still require solid overall discrimination (ROC-AUC) and balance (F1).
    def score(m):
        return 0.45 * m["metrics"]["recall"] + 0.35 * m["metrics"]["roc_auc"] + 0.20 * m["metrics"]["f1"]

    best_name = max(results, key=lambda k: score(results[k]))

    return {
        "results": results,
        "best_model_name": best_name,
        "X_test": X_test, "y_test": y_test,
        "training_records": len(X_train),
        "test_records": len(X_test),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "feature_list": ALL_FEATURES,
    }


def get_feature_names(pipeline: Pipeline) -> list:
    prep = pipeline.named_steps["prep"]
    return list(prep.get_feature_names_out())


def save_best_model(training_output: dict, path: str = MODEL_PATH):
    best_name = training_output["best_model_name"]
    best = training_output["results"][best_name]
    payload = {
        "pipeline": best["pipeline"],
        "model_name": best_name,
        "metrics": best["metrics"],
        "all_model_metrics": {k: v["metrics"] for k, v in training_output["results"].items()},
        "roc_curves": {k: v["roc_curve"] for k, v in training_output["results"].items()},
        "trained_at": training_output["trained_at"],
        "training_records": training_output["training_records"],
        "test_records": training_output["test_records"],
        "feature_list": training_output["feature_list"],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(payload, path)
    return payload


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict_risk(model_payload: dict, input_df: pd.DataFrame) -> np.ndarray:
    pipeline = model_payload["pipeline"]
    return pipeline.predict_proba(input_df[ALL_FEATURES])[:, 1]


def risk_level(prob: float) -> str:
    if prob < 0.25:
        return "LOW"
    elif prob < 0.5:
        return "MEDIUM"
    elif prob < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


if __name__ == "__main__":
    from data_processing import load_and_clean

    df, _, _ = load_and_clean()
    out = train_and_evaluate(df)
    print("Model comparison:")
    for name, res in out["results"].items():
        print(f"  {name}: {res['metrics']}")
    print("\nBest model:", out["best_model_name"])
    payload = save_best_model(out)
    print("Saved to", MODEL_PATH)
