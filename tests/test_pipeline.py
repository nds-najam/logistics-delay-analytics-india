"""
Basic tests for the DataQ logistics delay analytics pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_generator as dg
import data_processing as dp
import delay_analysis as da
import model as mdl
import recommendations as rec


@pytest.fixture(scope="module")
def raw_small():
    return dg.generate_synthetic_data(n_records=5000, seed=1)


@pytest.fixture(scope="module")
def raw_with_quality_issues(raw_small):
    return dg.inject_data_quality_issues(raw_small, seed=2)


@pytest.fixture(scope="module")
def clean_small(raw_with_quality_issues):
    clean, report = dp.clean_data(raw_with_quality_issues)
    return clean, report


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
class TestDataGeneration:
    def test_record_count(self, raw_small):
        assert len(raw_small) == 5000

    def test_required_columns_present(self, raw_small):
        required = [
            "Shipment_ID", "Order_Date", "Expected_Delivery_Date", "Actual_Delivery_Date",
            "Delay_Flag", "Delay_Hours", "Delay_Category", "Delay_Reason", "Distance_KM",
        ]
        for col in required:
            assert col in raw_small.columns

    def test_delay_flag_consistent_with_dates(self, raw_small):
        diff_hours = (raw_small["Actual_Delivery_Date"] - raw_small["Expected_Delivery_Date"]).dt.total_seconds() / 3600
        expected_flag = (diff_hours > 0).astype(int)
        assert (raw_small["Delay_Flag"] == expected_flag).all()

    def test_delay_hours_non_negative(self, raw_small):
        assert (raw_small["Delay_Hours"] >= 0).all()

    def test_on_time_shipments_have_no_delay_category(self, raw_small):
        on_time = raw_small.loc[raw_small["Delay_Flag"] == 0]
        assert (on_time["Delay_Category"] == "On Time").all()

    def test_delayed_shipments_have_named_category(self, raw_small):
        delayed = raw_small.loc[raw_small["Delay_Flag"] == 1]
        assert delayed["Delay_Category"].isin(dg.DELAY_CAUSES).all()

    def test_delay_rate_in_plausible_range(self, raw_small):
        rate = raw_small["Delay_Flag"].mean()
        assert 0.10 < rate < 0.55

    def test_distance_positive(self, raw_small):
        assert (raw_small["Distance_KM"] > 0).all()

    def test_reproducible_with_seed(self):
        a = dg.generate_synthetic_data(n_records=500, seed=99)
        b = dg.generate_synthetic_data(n_records=500, seed=99)
        pd.testing.assert_frame_equal(a, b)

    def test_quality_issues_injected(self, raw_with_quality_issues, raw_small):
        assert raw_with_quality_issues.isna().sum().sum() > 0
        assert len(raw_with_quality_issues) > len(raw_small)  # duplicates appended


# --------------------------------------------------------------------------
# Data cleaning
# --------------------------------------------------------------------------
class TestDataCleaning:
    def test_quality_report_detects_issues(self, raw_with_quality_issues):
        report = dp.assess_data_quality(raw_with_quality_issues)
        assert report["total_missing_cells"] > 0
        assert report["duplicate_full_rows"] > 0

    def test_cleaning_removes_duplicates(self, clean_small):
        clean, report = clean_small
        assert clean.duplicated().sum() == 0
        assert clean["Shipment_ID"].duplicated().sum() == 0

    def test_cleaning_reduces_missing_values(self, raw_with_quality_issues, clean_small):
        clean, _ = clean_small
        raw_missing = raw_with_quality_issues.isna().sum().sum()
        clean_missing = clean.isna().sum().sum()
        assert clean_missing < raw_missing

    def test_city_names_canonicalized(self, clean_small):
        clean, _ = clean_small
        assert "mumbai" not in clean["Origin_City"].values
        assert "MUMBAI" not in clean["Origin_City"].values

    def test_carrier_names_canonicalized(self, clean_small):
        clean, _ = clean_small
        valid_carriers = set(dg.CARRIERS)
        assert set(clean["Carrier"].unique()).issubset(valid_carriers)

    def test_no_extreme_outliers_remain(self, clean_small):
        clean, _ = clean_small
        assert clean["Package_Weight"].max() <= 100
        assert clean["Distance_KM"].max() <= 6000

    def test_delay_flag_recomputed_after_cleaning(self, clean_small):
        clean, _ = clean_small
        diff_hours = (clean["Actual_Delivery_Date"] - clean["Expected_Delivery_Date"]).dt.total_seconds() / 3600
        expected_flag = (diff_hours > 0).astype(int)
        assert (clean["Delay_Flag"] == expected_flag).all()


# --------------------------------------------------------------------------
# Delay analysis
# --------------------------------------------------------------------------
class TestDelayAnalysis:
    def test_kpis_keys(self, clean_small):
        clean, _ = clean_small
        kpis = da.compute_kpis(clean)
        for key in ["total_shipments", "delayed_shipments", "delay_rate", "avg_delay_hours", "on_time_pct"]:
            assert key in kpis

    def test_kpis_empty_dataframe(self):
        empty = pd.DataFrame(columns=["Delay_Flag", "Delay_Hours", "Actual_Delivery_Days"])
        kpis = da.compute_kpis(empty)
        assert kpis["total_shipments"] == 0

    def test_delay_by_cause_sums_to_100(self, clean_small):
        clean, _ = clean_small
        causes = da.delay_by_cause(clean)
        if len(causes):
            assert abs(causes["Share_of_Delayed_Pct"].sum() - 100) < 1.0

    def test_generate_key_insights_returns_strings(self, clean_small):
        clean, _ = clean_small
        insights = da.generate_key_insights(clean)
        assert len(insights) > 0
        assert all(isinstance(i, str) for i in insights)

    def test_route_performance_columns(self, clean_small):
        clean, _ = clean_small
        routes = da.route_performance(clean)
        assert "Delay_Rate_Pct" in routes.columns
        assert "Route" in routes.columns

    def test_warehouse_performance_ranks_by_delay(self, clean_small):
        clean, _ = clean_small
        wh = da.warehouse_performance(clean)
        assert wh["Delay_Rate_Pct"].is_monotonic_decreasing


# --------------------------------------------------------------------------
# Feature engineering / model
# --------------------------------------------------------------------------
class TestModel:
    def test_prepare_features_no_leakage_columns(self, clean_small):
        clean, _ = clean_small
        X, y = mdl.prepare_features(clean)
        leaky_cols = ["Delay_Hours", "Delay_Category", "Delay_Reason", "Actual_Delivery_Date",
                      "Warehouse_Processing_Time_Hours", "Transit_Time_Hours", "Sorting_Time_Hours",
                      "Last_Mile_Time_Hours", "Pickup_Delay_Hours", "EWay_Bill_Delay_Hours"]
        for col in leaky_cols:
            assert col not in X.columns
        assert set(y.unique()).issubset({0, 1})

    def test_train_and_evaluate_returns_all_models(self, clean_small):
        clean, _ = clean_small
        out = mdl.train_and_evaluate(clean, test_size=0.3)
        assert set(out["results"].keys()) == {"Logistic Regression", "Random Forest", "Gradient Boosting (HGB)"}
        assert out["best_model_name"] in out["results"]

    def test_model_metrics_in_valid_range(self, clean_small):
        clean, _ = clean_small
        out = mdl.train_and_evaluate(clean, test_size=0.3)
        for name, res in out["results"].items():
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                assert 0.0 <= res["metrics"][metric] <= 1.0

    def test_predict_risk_returns_probabilities(self, clean_small):
        clean, _ = clean_small
        out = mdl.train_and_evaluate(clean, test_size=0.3)
        best = out["results"][out["best_model_name"]]
        payload = {"pipeline": best["pipeline"]}
        X, _ = mdl.prepare_features(clean.head(10))
        probs = mdl.predict_risk(payload, X)
        assert len(probs) == 10
        assert np.all((probs >= 0) & (probs <= 1))

    def test_risk_level_thresholds(self):
        assert mdl.risk_level(0.1) == "LOW"
        assert mdl.risk_level(0.4) == "MEDIUM"
        assert mdl.risk_level(0.6) == "HIGH"
        assert mdl.risk_level(0.9) == "CRITICAL"


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
class TestRecommendations:
    def test_generate_recommendations_structure(self, clean_small):
        clean, _ = clean_small
        recs = rec.generate_recommendations(clean)
        for r in recs:
            for key in ["issue", "evidence", "recommendation", "expected_impact", "priority"]:
                assert key in r
            assert r["priority"] in ("HIGH", "MEDIUM", "LOW")

    def test_generate_recommendations_empty_df(self):
        empty = pd.DataFrame(columns=["Delay_Flag"])
        assert rec.generate_recommendations(empty) == []

    def test_shipment_recommendations_non_empty(self):
        row = {
            "Warehouse_Capacity_Utilization": 95, "Driver_Availability": "Low",
            "Traffic_Level": "Severe", "Number_of_Handoffs": 5, "Address_Quality": "Poor",
            "COD_Flag": 1, "Strike_Disruption_Flag": 0, "Weather_Condition": "Clear",
            "Monsoon_Disruption_Flag": 0,
        }
        recs = rec.shipment_recommendations([], row)
        assert len(recs) >= 5

    def test_shipment_recommendations_fallback(self):
        row = {
            "Warehouse_Capacity_Utilization": 50, "Driver_Availability": "High",
            "Traffic_Level": "Low", "Number_of_Handoffs": 1, "Address_Quality": "Good",
            "COD_Flag": 0, "Strike_Disruption_Flag": 0, "Weather_Condition": "Clear",
            "Monsoon_Disruption_Flag": 0,
        }
        recs = rec.shipment_recommendations([], row)
        assert len(recs) == 1
        assert "No single dominant risk driver" in recs[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
