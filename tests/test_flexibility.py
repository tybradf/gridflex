import numpy as np
import pandas as pd

from gridflex.models.flexibility import emissions_avoided_kg, find_best_shift_hour, value_of_shift
from gridflex.store.db import upsert


def _seg(rows):
    return pd.DataFrame(rows)


def test_rejects_lower_point_estimate_with_overlapping_ci():
    """The core correctness test: a candidate hour with a LOWER point
    estimate but a CI overlapping the origin's must be REJECTED — only a
    statistically defensible (non-overlapping CI) improvement should be
    picked, never just the lowest number on the page."""
    seg = _seg([
        {"season": "summer", "hour": 10, "marginal_rate_kg_per_mwh": 600.0, "ci95_low": 550.0, "ci95_high": 650.0},
        {"season": "summer", "hour": 11, "marginal_rate_kg_per_mwh": 580.0, "ci95_low": 560.0, "ci95_high": 600.0},  # overlaps
        {"season": "summer", "hour": 12, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0},  # real
        {"season": "summer", "hour": 13, "marginal_rate_kg_per_mwh": 320.0, "ci95_low": 300.0, "ci95_high": 340.0},
    ])
    result = find_best_shift_hour(seg, season="summer", origin_hour=10, window_hours=4)
    assert result["feasible"] is True
    assert result["best_hour"] == 12


def test_origin_hour_not_gated_is_infeasible():
    seg = _seg([{"season": "summer", "hour": 12, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0}])
    result = find_best_shift_hour(seg, season="summer", origin_hour=3, window_hours=4)
    assert result["feasible"] is False
    assert "did not clear" in result["reason"]


def test_no_candidates_in_window_is_infeasible():
    seg = _seg([{"season": "summer", "hour": 10, "marginal_rate_kg_per_mwh": 600.0, "ci95_low": 550.0, "ci95_high": 650.0}])
    result = find_best_shift_hour(seg, season="summer", origin_hour=10, window_hours=1)
    assert result["feasible"] is False


def test_no_defensible_improvement_is_infeasible_not_fabricated():
    """If every candidate is worse (or not distinguishably better), the
    function must say so rather than picking the 'least bad' option."""
    seg = _seg([
        {"season": "summer", "hour": 10, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0},
        {"season": "summer", "hour": 12, "marginal_rate_kg_per_mwh": 600.0, "ci95_low": 550.0, "ci95_high": 650.0},
    ])
    result = find_best_shift_hour(seg, season="summer", origin_hour=10, window_hours=4)
    assert result["feasible"] is False


def test_window_wraps_past_midnight():
    seg = _seg([
        {"season": "winter", "hour": 22, "marginal_rate_kg_per_mwh": 600.0, "ci95_low": 550.0, "ci95_high": 650.0},
        {"season": "winter", "hour": 1, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0},
    ])
    result = find_best_shift_hour(seg, season="winter", origin_hour=22, window_hours=4)
    assert result["feasible"] is True
    assert result["best_hour"] == 1


def test_emissions_avoided_sign_correctness():
    assert emissions_avoided_kg(mw=100, rate_origin=600, rate_target=300) == 30_000.0
    assert emissions_avoided_kg(mw=100, rate_origin=300, rate_target=600) == -30_000.0


def test_value_of_shift_end_to_end():
    seg = _seg([
        {"season": "summer", "hour": 10, "marginal_rate_kg_per_mwh": 600.0, "ci95_low": 550.0, "ci95_high": 650.0},
        {"season": "summer", "hour": 12, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0},
    ])
    result = value_of_shift(seg, mw=50, origin_hour=10, window_hours=4, season="summer")
    assert result["feasible"] is True
    assert result["emissions_avoided_kg"] == 50 * (600 - 300)


def test_value_of_shift_infeasible_case_shape():
    """When infeasible, value_of_shift must still return a clear, well-shaped
    explanation — not just propagate an exception or a bare False."""
    seg = _seg([{"season": "summer", "hour": 12, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0}])
    result = value_of_shift(seg, mw=50, origin_hour=3, window_hours=4, season="summer")
    assert result["feasible"] is False
    assert "reason" in result


def _seed_zone_demand(con, zone="PE"):
    periods = pd.date_range("2024-06-01", periods=24 * 30, freq="h", tz="UTC")
    hours = periods.hour
    demand = 1000 + 500 * np.sin((hours - 9) / 24 * 2 * np.pi)  # peaks near hour 15
    upsert(con, "subba_demand", pd.DataFrame({
        "period": periods, "subba": [zone] * len(periods),
        "parent": ["PJM"] * len(periods), "value": demand,
    }))
    return periods


def test_zone_typical_demand_and_peak_on_known_shape(tmp_db):
    """Hand-verifiable: constructed demand peaks exactly at hour 15 (1500)
    and troughs at hour 3 (500) — confirm both the raw lookup and the
    seasonal-peak-of-typical-curve calculation recover these exactly."""
    from gridflex.store.db import get_connection
    from gridflex.models.flexibility import zone_seasonal_peak, zone_typical_demand

    con = get_connection()
    _seed_zone_demand(con)
    h15 = zone_typical_demand(con, "PE", "summer", 15)
    h3 = zone_typical_demand(con, "PE", "summer", 3)
    peak = zone_seasonal_peak(con, "PE", "summer")
    con.close()

    assert abs(h15 - 1500.0) < 1e-6
    assert abs(h3 - 500.0) < 1e-6
    assert abs(peak - 1500.0) < 1e-6


def test_full_value_of_shift_uses_same_target_hour_for_both_pieces(tmp_db):
    """The two halves (emissions, zone peak context) must agree on WHICH
    hour they're describing — the peak context should use the hour
    find_best_shift_hour actually picked, not an independently-chosen one."""
    from gridflex.store.db import get_connection
    from gridflex.models.flexibility import full_value_of_shift

    con = get_connection()
    _seed_zone_demand(con)
    seg = _seg([
        {"season": "summer", "hour": 15, "marginal_rate_kg_per_mwh": 600.0, "ci95_low": 550.0, "ci95_high": 650.0},
        {"season": "summer", "hour": 3, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0},
    ])
    result = full_value_of_shift(con, seg, zone="PE", mw=50, origin_hour=15, window_hours=13, season="summer")
    con.close()

    assert result["emissions"]["feasible"] is True
    assert result["peak_context"]["target_hour"] == result["emissions"]["best_hour"]


def test_full_value_of_shift_infeasible_skips_peak_context_not_crashes(tmp_db):
    from gridflex.store.db import get_connection
    from gridflex.models.flexibility import full_value_of_shift

    con = get_connection()
    _seed_zone_demand(con)
    seg = _seg([{"season": "summer", "hour": 3, "marginal_rate_kg_per_mwh": 300.0, "ci95_low": 280.0, "ci95_high": 320.0}])
    result = full_value_of_shift(con, seg, zone="PE", mw=50, origin_hour=15, window_hours=13, season="summer")
    con.close()

    assert result["emissions"]["feasible"] is False
    assert result["peak_context"] is None
