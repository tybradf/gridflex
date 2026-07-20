import pandas as pd

from gridflex.models.flexibility import emissions_avoided_kg, find_best_shift_hour, value_of_shift


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
