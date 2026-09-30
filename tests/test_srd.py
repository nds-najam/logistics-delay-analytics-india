"""
Tests for the SRD Logistics LR-level pipeline.
Run with:  pytest tests/test_srd.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import srd_analytics as an
import srd_generator as gen
import srd_live as lv
import srd_model as mdl

TIME_CHAIN = ["Booking_DT", "Collection_DT", "Dispatch_DT", "Hub_Receipt_DT", "Hub_Dispatch_DT",
              "Dest_Arrival_DT", "Dest_Received_DT", "Delivery_DT"]


@pytest.fixture(scope="module")
def raw():
    return gen.generate_lr_data(n_records=6000, seed=3)


@pytest.fixture(scope="module")
def prepared(raw):
    return an.prepare(raw, gen.default_targets(), dict(an.DEFAULT_RULES))


# ---------------- generator ----------------
def test_schema_and_uniqueness(raw):
    for c in ["LR_No", "Source", "Destination", "Via", "Main_Hub", "Business", "Consignor", "Consignee", "Weight_KG",
              "Status", "Delay_Reason", "Condition_Remark", "Operational_Remark"] + TIME_CHAIN:
        assert c in raw.columns
    assert raw["LR_No"].is_unique
    assert (raw["Source"] != raw["Destination"]).all()


def test_timestamps_ordered_and_not_in_future(raw):
    for a, b in zip(TIME_CHAIN[:-1], TIME_CHAIN[1:]):
        both = raw[a].notna() & raw[b].notna()
        assert (raw.loc[both, b] >= raw.loc[both, a]).all(), (a, b)
        assert (raw[b].isna() | raw[a].notna()).all()  # a later milestone never exists without the earlier one
    for c in TIME_CHAIN:
        assert (raw[c].dropna() <= gen.ASOF).all()


def test_direct_routes_have_no_via_timestamps(raw):
    direct = raw["Via"] == "Direct"
    assert raw.loc[direct, "Via_Arrival_DT"].isna().all()


def test_status_matches_last_milestone(raw):
    assert (raw.loc[raw["Status"] == "Delivered", "Delivery_DT"].notna()).all()
    assert (raw.loc[raw["Status"] == "Booked", "Collection_DT"].isna()).all()


def test_route_reference_covers_all_pairs():
    ref = gen.route_reference()
    n = len(gen.NETWORK_CITIES)
    assert len(ref) == n * (n - 1)
    assert (ref["Target_Days"] >= 1).all()


# ---------------- analytics ----------------
def test_summarize_identities(prepared):
    s = an.summarize(prepared, ["Source"])
    assert (s["On_Track"] + s["Delayed"] == s["Volume"]).all()
    assert np.allclose(s["Actual_Pct"], (100 * s["On_Track"] / s["Volume"]).round(2), atol=0.01)
    assert np.allclose(s["Variation_Pct"], (s["Actual_Pct"] - s["Target_Pct"]).round(2), atol=0.02)
    assert s["Volume"].sum() == len(prepared)


def test_overall_summary(prepared):
    s = an.summarize(prepared, [])
    assert len(s) == 1 and s.loc[0, "Volume"] == len(prepared)


def test_target_modes(raw):
    t = gen.default_targets()
    fixed = an.prepare(raw, t, dict(an.DEFAULT_RULES), target_mode="fixed", target_value=6)
    assert (fixed["Target_Days"] == 6).all()
    bench = an.prepare(raw, t, dict(an.DEFAULT_RULES))
    off = an.prepare(raw, t, dict(an.DEFAULT_RULES), target_mode="offset", target_value=-1)
    assert (off["Target_Days"] == (bench["Target_Days"] - 1).clip(lower=1)).all()
    # tighter target can only reduce service %
    assert an.summarize(off, [])["Actual_Pct"].iloc[0] <= an.summarize(bench, [])["Actual_Pct"].iloc[0]


def test_missing_route_uses_default_target(raw):
    t = gen.default_targets().iloc[1:]  # drop one route
    d = an.prepare(raw, t, dict(an.DEFAULT_RULES))
    assert d["Target_Days"].notna().all() and d["Target_Service_Pct"].notna().all()


def test_target_sensitivity_monotonic(prepared):
    s = an.target_sensitivity(prepared, range(2, 11))
    assert s["Service_Pct"].is_monotonic_increasing
    assert s["Delayed"].is_monotonic_decreasing


def test_delay_point_consistent_with_recorded_reason(prepared):
    d = prepared[prepared["Delayed"] & (prepared["Delay_Reason"] != "") & (prepared["Delay_Point"] != an.NO_POINT)
                 & ~prepared["Delay_Reason"].str.startswith("Shortage")]
    ok = [r in [x[0] for x in gen.STAGE_REASONS[p]] for p, r in zip(d["Delay_Point"], d["Delay_Reason"])]
    assert len(d) > 50 and np.mean(ok) > 0.93


def test_stage_delay_summary_totals(prepared):
    sd = an.stage_delay_summary(prepared)
    assert sd["Delayed_LRs"].sum() == int(prepared["Delayed"].sum())
    assert abs(sd["Pct_of_Delayed"].sum() - 100) < 0.5


def test_reason_summary_percentages(prepared):
    rs = an.reason_summary(prepared)
    assert abs(rs["Pct_of_Delayed"].sum() - 100) < 0.5
    by = an.reason_summary(prepared, by="Delay_Point")
    assert np.allclose(by.groupby("Delay_Point")["Pct_of_Delayed"].sum(), 100, atol=0.5)


def test_period_labels():
    s = pd.Series(pd.to_datetime(["2025-10-01", "2025-10-10", "2025-10-11", "2025-10-21", "2026-03-31", "2026-04-01"]))
    lab10, _ = an.period_columns(s, "10-Day")
    assert lab10.iloc[0] == lab10.iloc[1] and lab10.iloc[1] != lab10.iloc[2] and lab10.iloc[2] != lab10.iloc[3]
    lab15, _ = an.period_columns(s, "15-Day")
    assert lab15.iloc[2] == lab15.iloc[1] and lab15.iloc[3] != lab15.iloc[2]
    fy, _ = an.period_columns(s, "Yearly (FY)")
    assert fy.iloc[0] == "FY 2025-26" and fy.iloc[4] == "FY 2025-26" and fy.iloc[5] == "FY 2026-27"
    wk, key = an.period_columns(s, "Weekly")
    assert (key.dt.dayofweek == 0).all()


@pytest.mark.parametrize("kind", an.PERIOD_KINDS)
def test_period_summary_conserves_volume(prepared, kind):
    ps = an.period_summary(prepared, kind)
    assert ps["Volume"].sum() == len(prepared)
    assert ps["Period_Key"].is_monotonic_increasing


def test_compare_periods(prepared):
    a = (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-30"))
    b = (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-31"))
    cp = an.compare_periods(prepared, a, b)
    assert cp.loc[0, "Volume_Current"] > 0 and cp.loc[0, "Volume_Previous"] > 0
    same = an.compare_periods(prepared, a, a, by="Source")
    assert (same["Actual_Pct_Change"].abs() < 1e-9).all()


def test_hub_movement_partitions_lrs(prepared):
    h = an.hub_movement(prepared, dict(an.DEFAULT_RULES))
    assert h["Hub_Stage"].isin(an.HUB_STAGES).all() and len(h) == len(prepared)
    assert (h.loc[h["Hub_Stage"] == an.HUB_STAGES[3], "Hub_Dispatch_DT"].notna()).all()
    assert h.loc[h["Hub_Stage"] != an.HUB_STAGES[3], "Age_Hrs"].notna().all()
    hs = an.hub_summary(h, "Main_Hub")
    assert hs["Booked_LRs"].sum() == len(h)


def test_lr_journey(prepared):
    delivered = prepared[prepared["Delivered"] & (prepared["Via"] != "Direct")].iloc[0]
    jr = an.lr_journey(delivered)
    assert jr["Reached"].all() and jr["Milestone"].iloc[0] == "Booking" and jr["Milestone"].iloc[-1] == "Delivery"
    open_lr = prepared[~prepared["Delivered"]].iloc[0]
    assert not an.lr_journey(open_lr)["Reached"].all()


def test_suggestions_shape(prepared):
    rules = dict(an.DEFAULT_RULES, min_route_volume=10, suggest_weeks_below=2)
    sg = an.suggestions(prepared, rules)
    if len(sg):
        assert (sg["Variation_Pct"] < 0).all()
        assert sg["Suggestion"].str.len().min() > 10
        assert set(sg["Priority"].astype(str)) <= {"HIGH", "MEDIUM", "LOW"}


def test_targets_and_rules_roundtrip(tmp_path):
    p = str(tmp_path / "t.csv")
    t = gen.default_targets().head(5)
    t.loc[t.index[0], "Target_Days"] = 9
    an.save_targets(t, p)
    back = an.load_targets(p)
    assert back.loc[0, "Target_Days"] == 9
    rp = str(tmp_path / "r.json")
    an.save_rules({"min_route_volume": 77}, rp)
    assert an.load_rules(rp)["min_route_volume"] == 77
    assert an.load_rules(str(tmp_path / "missing.json"))["min_route_volume"] == an.DEFAULT_RULES["min_route_volume"]


# ---------------- live ----------------
def test_live_reports(prepared):
    R = lv.live_reports(prepared, gen.ASOF)
    assert len(R["bookings_today"]) > 0
    assert R["daily_booking"]["by_source"]["LRs"].sum() == len(R["bookings_today"])
    assert set(R["lag_lead_source"]["Performance"]) <= {"Leading", "Lagging"}
    assert len(R["hub_movement"]) == prepared["Main_Hub"].nunique()


def test_get_source_defaults_to_simulated(monkeypatch):
    monkeypatch.delenv("SRD_API_BASE_URL", raising=False)
    assert isinstance(lv.get_source(), lv.SimulatedSource)
    monkeypatch.setenv("SRD_API_BASE_URL", "http://example.invalid/api")
    assert isinstance(lv.get_source(), lv.HttpSource)


# ---------------- model ----------------
def test_model_trains_and_explains(prepared):
    payload = mdl.train(prepared)
    assert 0.5 < payload["metrics"]["roc_auc"] <= 1.0
    row = prepared[~prepared["Delivered"]].head(1)
    p = mdl.score(payload, row)
    assert 0 <= p[0] <= 1
    ex = mdl.explain_lr(payload, row, top_k=3)
    assert len(ex) == 3 and set(ex["Direction"]) <= {"Raises risk", "Lowers risk"}
    assert mdl.risk_level(0.05) == "LOW" and mdl.risk_level(0.9) == "CRITICAL"
