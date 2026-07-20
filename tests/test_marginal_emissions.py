import numpy as np
import pandas as pd

from gridflex.config import EMISSION_FACTORS_KG_PER_MWH
from gridflex.models.marginal_emissions import compute_deltas, estimate_marginal_emissions_rate
from gridflex.store.db import upsert


def test_regression_recovers_known_true_marginal_rate(tmp_db):
    """The core correctness test: construct data where NG is the ONLY fuel
    responding to demand changes (baseload held perfectly flat), so the
    TRUE marginal rate is exactly NG's emission factor by construction —
    confirm the regression recovers it exactly."""
    from gridflex.store.db import get_connection
    con = get_connection()

    n = 500
    periods = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    np.random.seed(0)
    demand = 100_000 + 20_000 * np.sin(np.arange(n) / 24 * 2 * np.pi) + np.random.normal(0, 1000, n)
    ng_generation = demand - 50_000  # baseload = 50,000 flat; NG absorbs all variation

    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": demand,
    }))
    fuel_rows = []
    for i, p in enumerate(periods):
        fuel_rows.append({"period": p, "respondent": "PJM", "fueltype": "NG", "value": ng_generation[i]})
        fuel_rows.append({"period": p, "respondent": "PJM", "fueltype": "COL", "value": 30_000.0})
        fuel_rows.append({"period": p, "respondent": "PJM", "fueltype": "NUC", "value": 20_000.0})
    upsert(con, "fuel_mix", pd.DataFrame(fuel_rows))

    deltas = compute_deltas(con)
    result = estimate_marginal_emissions_rate(deltas)
    con.close()

    true_rate = EMISSION_FACTORS_KG_PER_MWH["NG"]
    assert abs(result["marginal_rate_kg_per_mwh"] - true_rate) < 0.01
    assert result["r2"] > 0.999


def test_gap_spanning_delta_excluded_legitimate_neighbor_kept(tmp_db):
    """A real data gap must not contaminate the regression with a spurious
    multi-hour delta — but the legitimate delta immediately BEFORE the gap
    must still be included."""
    from gridflex.store.db import get_connection
    con = get_connection()

    n = 100
    periods = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    np.random.seed(1)
    demand = 100_000 + np.random.normal(0, 1000, n)
    ng = demand - 50_000

    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": demand,
    }))
    upsert(con, "fuel_mix", pd.DataFrame(
        [{"period": p, "respondent": "PJM", "fueltype": "NG", "value": ng[i]} for i, p in enumerate(periods)]
    ))

    gap_period = periods[50]
    con.execute(f"DELETE FROM pjm_demand WHERE period = '{gap_period}'")
    con.execute(f"DELETE FROM fuel_mix WHERE period = '{gap_period}'")

    deltas = compute_deltas(con)
    con.close()

    present = set(deltas["period"])
    assert periods[49] in present   # legitimate 48->49 delta, unaffected by the gap
    assert periods[50] not in present  # deleted, cannot exist
    assert periods[51] not in present  # gap-spanning 50->51 delta, must be excluded


def test_standard_error_matches_independent_numpy_polyfit():
    """Cross-checks our slope SE/CI against numpy.polyfit's independent
    covariance computation — a completely separate numerical method,
    not just re-deriving the same formula twice."""
    np.random.seed(5)
    n = 700
    x = np.random.uniform(-5000, 5000, n)
    true_slope, true_intercept = 450.0, 100.0
    y = true_slope * x + true_intercept + np.random.normal(0, 500_000, n)

    df = pd.DataFrame({"delta_demand": x, "delta_emissions": y})
    result = estimate_marginal_emissions_rate(df)

    coeffs, cov = np.polyfit(x, y, deg=1, cov=True)
    polyfit_slope_se = np.sqrt(cov[0, 0])

    assert abs(result["marginal_rate_kg_per_mwh"] - coeffs[0]) < 0.01
    assert abs(result["slope_se"] - polyfit_slope_se) < 0.01
    assert result["ci95_low"] < result["marginal_rate_kg_per_mwh"] < result["ci95_high"]


def test_raises_on_too_few_data_points():
    tiny = pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        "delta_demand": [100.0] * 5,
        "delta_emissions": [40000.0] * 5,
    })
    try:
        estimate_marginal_emissions_rate(tiny)
        assert False, "should have raised"
    except ValueError:
        pass


def test_season_boundaries():
    from gridflex.models.marginal_emissions import _season
    assert _season(12) == _season(1) == _season(2) == "winter"
    assert _season(3) == _season(4) == _season(5) == "spring"
    assert _season(6) == _season(7) == _season(8) == "summer"
    assert _season(9) == _season(10) == _season(11) == "fall"


def test_segmentation_recovers_distinct_regime_rates_global_regression_blends(tmp_db):
    """The key differentiator test: construct data where winter's TRUE
    marginal fuel is coal (~1000 kg/MWh) and summer's is gas (~410
    kg/MWh) — confirm segmentation recovers BOTH distinct rates, while a
    global regression necessarily blends them into one number in between.
    This is what actually enables a 'flexibility is worth Nx more at time
    A than B' claim — the global number alone cannot support it."""
    from gridflex.models.marginal_emissions import estimate_marginal_emissions_by_segment
    from gridflex.store.db import get_connection

    con = get_connection()
    winter_periods = pd.date_range("2024-01-01", periods=700, freq="h", tz="UTC")
    summer_periods = pd.date_range("2024-07-01", periods=700, freq="h", tz="UTC")

    np.random.seed(0)
    def build(periods, marginal_fuel):
        n = len(periods)
        demand = 100_000 + 15_000 * np.sin(np.arange(n) / 24 * 2 * np.pi) + np.random.normal(0, 500, n)
        marginal_gen = demand - 50_000
        demand_rows = pd.DataFrame({
            "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": demand,
        })
        fuel_rows = [{"period": p, "respondent": "PJM", "fueltype": marginal_fuel, "value": marginal_gen[i]}
                     for i, p in enumerate(periods)]
        fuel_rows += [{"period": p, "respondent": "PJM", "fueltype": "NUC", "value": 20_000.0} for p in periods]
        return demand_rows, pd.DataFrame(fuel_rows)

    wd, wf = build(winter_periods, "COL")
    sd, sf = build(summer_periods, "NG")
    upsert(con, "pjm_demand", pd.concat([wd, sd]))
    upsert(con, "fuel_mix", pd.concat([wf, sf]))

    deltas = compute_deltas(con)
    segmented = estimate_marginal_emissions_by_segment(deltas, min_n=20)
    con.close()

    winter_avg = segmented[segmented["season"] == "winter"]["marginal_rate_kg_per_mwh"].mean()
    summer_avg = segmented[segmented["season"] == "summer"]["marginal_rate_kg_per_mwh"].mean()

    assert abs(winter_avg - EMISSION_FACTORS_KG_PER_MWH["COL"]) < 5
    assert abs(summer_avg - EMISSION_FACTORS_KG_PER_MWH["NG"]) < 5
    assert abs(winter_avg - summer_avg) > 400


def test_r2_gate_excludes_noisy_segment_that_clears_n_gate(tmp_db):
    """Regression test for a real finding: a segment can clear min_n and
    still be an untrustworthy fit (real example: winter hour=6 had n=691
    but r2=0.08 and a physically-impossible rate of 1,987 kg/MWh,
    exceeding coal's own emission factor). Constructs one segment with a
    real signal and one with pure noise (delta_emissions uncorrelated with
    delta_demand) — both clear min_n, only the real-signal one should
    survive min_r2."""
    from gridflex.models.marginal_emissions import estimate_marginal_emissions_by_segment
    from gridflex.store.db import get_connection

    con = get_connection()
    clean_periods = pd.date_range("2024-01-01", periods=700, freq="h", tz="UTC")   # winter
    noisy_periods = pd.date_range("2024-07-01", periods=700, freq="h", tz="UTC")   # summer

    np.random.seed(2)
    n = 700
    # Clean segment: NG absorbs demand variation deterministically -> high r2
    clean_demand = 100_000 + 15_000 * np.sin(np.arange(n) / 24 * 2 * np.pi)
    clean_ng = clean_demand - 50_000

    # Noisy segment: demand varies, but generation is essentially RANDOM,
    # uncorrelated with demand -> r2 should be near zero regardless of n.
    noisy_demand = 100_000 + 15_000 * np.sin(np.arange(n) / 24 * 2 * np.pi)
    noisy_ng = 50_000 + np.random.normal(0, 20_000, n)  # no real relationship to demand

    demand_df = pd.concat([
        pd.DataFrame({"period": clean_periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": clean_demand}),
        pd.DataFrame({"period": noisy_periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": noisy_demand}),
    ])
    fuel_df = pd.concat([
        pd.DataFrame({"period": clean_periods, "respondent": ["PJM"] * n, "fueltype": ["NG"] * n, "value": clean_ng}),
        pd.DataFrame({"period": noisy_periods, "respondent": ["PJM"] * n, "fueltype": ["NG"] * n, "value": noisy_ng}),
    ])
    upsert(con, "pjm_demand", demand_df)
    upsert(con, "fuel_mix", fuel_df)

    deltas = compute_deltas(con)
    result = estimate_marginal_emissions_by_segment(deltas, min_n=20, min_r2=0.15)
    con.close()

    winter_segments = result[result["season"] == "winter"]
    summer_segments = result[result["season"] == "summer"]
    assert len(winter_segments) > 0, "the real-signal (clean) segments should survive"
    # Pure noise can spuriously clear the r2 gate by chance in a small
    # fraction of buckets — but the plausibility ceiling must catch those,
    # since a spurious fit on noise tends to produce an implausible rate
    # (observed: ~10^16 kg/MWh). Combined, both noise segments must be excluded.
    assert len(summer_segments) == 0, (
        "noise-only segments must be excluded by r2 AND/OR the plausibility "
        "ceiling working together — if this fails, check whether a spurious "
        "fit is producing a rate that's implausible but somehow still "
        "within max_abs_rate"
    )


def test_segments_below_min_n_are_skipped_not_crashed(tmp_db):
    """A segment with too little data must be skipped (logged), not crash
    the whole run or silently produce an untrustworthy slope."""
    from gridflex.models.marginal_emissions import estimate_marginal_emissions_by_segment
    from gridflex.store.db import get_connection

    con = get_connection()
    # Only 50 hours of data total -> most (season, hour) buckets get 0-3 points
    n = 50
    periods = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    demand = 100_000 + np.random.normal(0, 500, n)
    ng = demand - 50_000
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": demand,
    }))
    upsert(con, "fuel_mix", pd.DataFrame(
        [{"period": p, "respondent": "PJM", "fueltype": "NG", "value": ng[i]} for i, p in enumerate(periods)]
    ))

    deltas = compute_deltas(con)
    result = estimate_marginal_emissions_by_segment(deltas, min_n=100)  # deliberately impossible to satisfy
    con.close()

    assert result.empty  # every segment skipped, but no crash
