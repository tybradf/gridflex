import numpy as np
import pandas as pd

from gridflex.models.baselines import seasonal_naive_predict
from gridflex.models.zone_aggregate import run_zone_aggregate_backtest
from gridflex.store.db import upsert


def _seed_constant_zones(con, zone_vals: dict, n=24 * 90 + 5 * 24 + 168 + 20, pjm_bias=1.0):
    periods = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    for z, v in zone_vals.items():
        upsert(con, "subba_demand", pd.DataFrame({
            "period": periods, "subba": [z] * n, "parent": ["PJM"] * n, "value": [v] * n,
        }))
        upsert(con, "weather", pd.DataFrame({
            "period": periods, "subba": [z] * n,
            "temperature_2m": [20.0] * n, "relative_humidity_2m": [50.0] * n,
            "wind_speed_10m": [5.0] * n, "shortwave_radiation": [100.0] * n,
        }))
    system_total = sum(zone_vals.values())
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": [system_total] * n,
    }))
    upsert(con, "pjm_forecast", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["DF"] * n,
        "value": [system_total * pjm_bias] * n,
    }))
    return periods


def test_aggregate_matches_exact_known_values(tmp_db):
    """On constant per-zone demand, seasonal-naive predicts exactly right —
    summed across zones, the aggregate should match true system demand
    exactly (0.0 error), and PJM's known +5% bias should score exactly 5.0
    MAPE. This is the tz-stripping bug's regression test — it previously
    produced all-NaN 'ours' metrics."""
    from gridflex.store.db import get_connection
    con = get_connection()
    zone_vals = {"PE": 1000.0, "CE": 2000.0, "AE": 3000.0}
    _seed_constant_zones(con, zone_vals, pjm_bias=1.05)

    result = run_zone_aggregate_backtest(
        con, zones=list(zone_vals), predict_fn=seasonal_naive_predict,
        n_splits=5, test_size_hours=24, min_train_hours=24 * 90,
    )
    con.close()

    assert not result["ours_mape"].isna().any(), "regression: tz-stripping bug produced NaN metrics"
    assert (result["ours_mape"] < 0.01).all()
    assert (abs(result["pjm_mape"] - 5.0) < 0.01).all()
    assert result["zone_coverage_ok"].all()


def test_anchor_prevents_frontier_mismatch_crash(tmp_db):
    """Regression test for the actual crash: pjm_demand fresher than
    subba_demand caused ALL 20 zones to show 0 test rows in the final
    fold (a frontier mismatch, not a zone-specific data-quality issue),
    which crashed on a shape-mismatch broadcast error. The anchor must
    trim fold windows to the safe frontier across ALL required sources."""
    from gridflex.store.db import get_connection
    from gridflex.models.zone_aggregate import _safe_backtest_anchor

    con = get_connection()
    n_zone = 24 * 90 + 5 * 24 + 168 + 20
    zone_periods = pd.date_range("2024-01-01", periods=n_zone, freq="h", tz="UTC")
    n_pjm = n_zone + 30  # pjm_demand extends 30h FURTHER than subba_demand
    pjm_periods = pd.date_range("2024-01-01", periods=n_pjm, freq="h", tz="UTC")

    zone_vals = {"PE": 1000.0, "CE": 2000.0}
    for z, v in zone_vals.items():
        upsert(con, "subba_demand", pd.DataFrame({
            "period": zone_periods, "subba": [z] * n_zone, "parent": ["PJM"] * n_zone, "value": [v] * n_zone,
        }))
        upsert(con, "weather", pd.DataFrame({
            "period": zone_periods, "subba": [z] * n_zone,
            "temperature_2m": [20.0] * n_zone, "relative_humidity_2m": [50.0] * n_zone,
            "wind_speed_10m": [5.0] * n_zone, "shortwave_radiation": [100.0] * n_zone,
        }))
    system_total = sum(zone_vals.values())
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": pjm_periods, "respondent": ["PJM"] * n_pjm, "type": ["D"] * n_pjm, "value": [system_total] * n_pjm,
    }))
    upsert(con, "pjm_forecast", pd.DataFrame({
        "period": pjm_periods, "respondent": ["PJM"] * n_pjm, "type": ["DF"] * n_pjm, "value": [system_total * 1.05] * n_pjm,
    }))

    anchor = _safe_backtest_anchor(con, list(zone_vals))
    assert anchor == zone_periods.max()  # limited by zone data, NOT pjm_demand's later frontier

    result = run_zone_aggregate_backtest(  # must not raise
        con, zones=list(zone_vals), predict_fn=seasonal_naive_predict,
        n_splits=5, test_size_hours=24, min_train_hours=24 * 90,
    )
    con.close()
    assert result["zone_coverage_ok"].all()


def test_mid_history_gap_flagged_without_affecting_other_folds(tmp_db):
    """The anchor fix (found via a real crash) correctly EXCLUDES trailing-
    frontier mismatches entirely, rather than scoring them with a warning —
    a fold missing even one zone isn't a fair 20-zone-vs-PJM comparison.
    So the partial-coverage warning path is for a DIFFERENT case: a gap in
    the MIDDLE of one zone's history, which doesn't affect the frontier at
    all and therefore isn't caught by the anchor."""
    from gridflex.store.db import get_connection
    con = get_connection()
    zone_vals = {"PE": 1000.0, "CE": 2000.0}
    periods = _seed_constant_zones(con, zone_vals, pjm_bias=1.0)

    # Find fold 2's test window WITHOUT any sabotage first, so we know
    # exactly which historical range to damage.
    from gridflex.models.zone_aggregate import run_zone_aggregate_backtest
    from gridflex.models.backtest import calendar_fold_windows
    from gridflex.features.build import build_training_table

    reference_df = build_training_table(con)
    windows = calendar_fold_windows(reference_df, n_splits=5, test_size_hours=24, min_train_hours=24 * 90)
    _, test_start, test_end = windows[2]  # a MIDDLE fold, not the last one

    # Delete PE's data specifically within fold 2's test window — PE's
    # trailing/frontier data (used by the anchor) is untouched.
    con.execute("""
        DELETE FROM subba_demand
        WHERE subba = 'PE' AND period >= ? AND period <= ?
    """, [test_start, test_end])

    result = run_zone_aggregate_backtest(
        con, zones=list(zone_vals), predict_fn=seasonal_naive_predict,
        n_splits=5, test_size_hours=24, min_train_hours=24 * 90,
    )
    con.close()

    assert not result["zone_coverage_ok"].iloc[2]  # the damaged middle fold
    assert result["zone_coverage_ok"].iloc[0]       # earlier folds unaffected
    assert result["zone_coverage_ok"].iloc[1]
    assert result["zone_coverage_ok"].iloc[3]       # later folds unaffected
    assert result["zone_coverage_ok"].iloc[4]        # frontier fold unaffected
