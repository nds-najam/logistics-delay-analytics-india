"""
Data loading, quality assessment, and cleaning pipeline.
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logistics_data.csv")

DATE_COLS = ["Order_Date", "Pickup_Date", "Expected_Delivery_Date", "Actual_Delivery_Date"]

CITY_CANONICAL = {
    "mumbai": "Mumbai", "bombay": "Mumbai", " mumbai": "Mumbai",
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
    "new delhi": "Delhi", "delhi": "Delhi",
    "cochin": "Kochi", "kochi": "Kochi",
}

CARRIER_CANONICAL = {
    "carrier a": "Carrier A", "carrier-a": "Carrier A",
    "carrier b": "Carrier B", "carrier_b": "Carrier B",
    "carrier c": "Carrier C",
}


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _canonicalize(series: pd.Series, mapping: dict) -> pd.Series:
    s = series.astype(str).str.strip()
    lowered = s.str.lower()
    mapped = lowered.map(mapping)
    return mapped.where(mapped.notna(), s)


def assess_data_quality(df: pd.DataFrame) -> dict:
    """Compute a data-quality report BEFORE cleaning."""
    n = len(df)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    dup_id = df["Shipment_ID"].duplicated().sum() if "Shipment_ID" in df.columns else 0
    dup_full = df.duplicated().sum()

    outlier_flags = pd.Series(False, index=df.index)
    outlier_counts = {}
    for col, (lo, hi) in {
        "Package_Weight": (0, 100), "Distance_KM": (0, 6000), "Delay_Hours": (0, 400),
    }.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            outlier_counts[col] = int(mask.sum())
            outlier_flags |= mask.fillna(False)

    total_cells = df.shape[0] * df.shape[1]
    completeness = 100 * (1 - df.isna().sum().sum() / total_cells)

    inconsistent = {}
    for col in ["Origin_City", "Destination_City", "Carrier"]:
        if col in df.columns:
            inconsistent[col] = int(df[col].astype(str).str.strip().nunique() -
                                     df[col].astype(str).str.strip().str.title().nunique())

    return {
        "n_records": n,
        "n_columns": df.shape[1],
        "missing_by_column": missing.to_dict(),
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_shipment_ids": int(dup_id),
        "duplicate_full_rows": int(dup_full),
        "outlier_rows_total": int(outlier_flags.sum()),
        "outlier_counts_by_column": outlier_counts,
        "data_completeness_pct": round(completeness, 2),
        "inconsistent_category_signal": inconsistent,
    }


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw dataset and return (clean_df, cleaning_report)."""
    report = {"steps": []}
    df = df.copy()
    n0 = len(df)

    # 1. remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["steps"].append(f"Removed {before - len(df)} exact duplicate rows.")

    # 2. remove duplicate shipment IDs (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["Shipment_ID"], keep="first")
    report["steps"].append(f"Removed {before - len(df)} duplicate Shipment_ID rows.")

    # 3. canonicalize inconsistent categorical labels
    df["Origin_City"] = _canonicalize(df["Origin_City"], CITY_CANONICAL)
    df["Destination_City"] = _canonicalize(df["Destination_City"], CITY_CANONICAL).str.strip()
    df["Origin_City"] = df["Origin_City"].str.strip()
    df["Carrier"] = _canonicalize(df["Carrier"], CARRIER_CANONICAL)
    report["steps"].append("Canonicalized inconsistent City/Carrier labels (case, whitespace, aliases).")

    # 4. clip / null-out outliers using domain-plausible bounds, then impute
    outlier_bounds = {"Package_Weight": (0.01, 100), "Distance_KM": (1, 6000), "Delay_Hours": (0, 400)}
    n_outliers_fixed = 0
    for col, (lo, hi) in outlier_bounds.items():
        mask = (df[col] < lo) | (df[col] > hi)
        n_outliers_fixed += int(mask.sum())
        df.loc[mask, col] = np.nan
    report["steps"].append(f"Flagged {n_outliers_fixed} extreme outlier values as missing for re-imputation.")

    # 5. impute missing values
    numeric_impute_cols = ["Package_Weight", "Package_Volume", "Distance_KM", "Delay_Hours"]
    for col in numeric_impute_cols:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    categorical_impute_cols = ["Weather_Condition", "Traffic_Level", "Vehicle_Availability",
                                "Driver_Availability", "Address_Quality", "Customer_Priority"]
    for col in categorical_impute_cols:
        if col in df.columns and df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "Unknown"
            df[col] = df[col].fillna(fill_val)
    report["steps"].append("Imputed missing numeric values with column median and categorical values with mode.")

    # 6. drop rows with missing critical fields (dates) that cannot be safely imputed
    before = len(df)
    df = df.dropna(subset=["Order_Date", "Expected_Delivery_Date"])
    critical_dropped = before - len(df)
    report["steps"].append(f"Dropped {critical_dropped} rows missing critical date fields.")

    # 7. re-derive Actual_Delivery_Date where missing, from component hours (self-consistency)
    if "Actual_Delivery_Date" in df.columns:
        missing_actual = df["Actual_Delivery_Date"].isna()
        if missing_actual.any():
            derived = (df.loc[missing_actual, "Pickup_Date"] + pd.to_timedelta(
                df.loc[missing_actual, "Warehouse_Processing_Time_Hours"] +
                df.loc[missing_actual, "Transit_Time_Hours"] +
                df.loc[missing_actual, "Sorting_Time_Hours"] +
                df.loc[missing_actual, "Last_Mile_Time_Hours"], unit="h"
            ))
            df.loc[missing_actual, "Actual_Delivery_Date"] = derived
            report["steps"].append(f"Re-derived {int(missing_actual.sum())} missing Actual_Delivery_Date values from component times.")

    # 8. recompute Delay_Flag / Delay_Hours for consistency after cleaning
    delay_hours_raw = (df["Actual_Delivery_Date"] - df["Expected_Delivery_Date"]).dt.total_seconds() / 3600.0
    df["Delay_Flag"] = (delay_hours_raw > 0).astype(int)
    df["Delay_Hours"] = np.clip(delay_hours_raw, 0, None).round(2)
    df.loc[df["Delay_Flag"] == 0, "Delay_Category"] = "On Time"

    report["records_before"] = n0
    report["records_after"] = len(df)
    report["records_removed"] = n0 - len(df)
    return df.reset_index(drop=True), report


def load_and_clean(path: str = DATA_PATH):
    raw = load_raw_data(path)
    quality_report = assess_data_quality(raw)
    clean, cleaning_report = clean_data(raw)
    return clean, quality_report, cleaning_report


if __name__ == "__main__":
    raw = load_raw_data()
    qr = assess_data_quality(raw)
    print("=== Data Quality Report (raw) ===")
    for k, v in qr.items():
        print(k, ":", v)
    clean, cr = clean_data(raw)
    print("\n=== Cleaning Report ===")
    for k, v in cr.items():
        print(k, ":", v)
    print("\nClean shape:", clean.shape)
    print("Delay rate after cleaning:", round(clean["Delay_Flag"].mean() * 100, 2), "%")
