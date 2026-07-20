"""
Week 4, block 4.1 — marginal emissions estimator.

carbon/average.py gives AVERAGE carbon intensity (the blended fuel mix's
rate). What a flexibility decision actually needs is MARGINAL: if 1 MW of
load shifts into or out of an hour, which plant responds on the margin, and
what's its emissions rate? These can differ enormously — a mix that looks
clean on average (lots of nuclear/hydro baseload) can still have a dirty
marginal responder, because baseload plants don't ramp to follow small
demand changes; fast-ramping gas peakers do.

Standard approach: regress Δ(total emissions) on Δ(total demand) across
hour-to-hour changes. The slope approximates the marginal responding fuel's
emissions rate.

CRITICAL: deltas must only be computed between genuinely adjacent hours
(period gap == exactly 1 hour) — the same lesson from detect_spike_rows.
A naive .diff() across a real data gap produces a spurious delta that has
nothing to do with a real marginal response, and would silently corrupt
the regression's slope estimate.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from gridflex.config import EMISSION_FACTORS_KG_PER_MWH

log = logging.getLogger(__name__)


def hourly_total_emissions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Total system emissions (kg CO2) per hour — sum across fuel types of
    generation (MWh) x emission factor (kg CO2/MWh). Distinct from
    carbon/average.py's INTENSITY (kg CO2/MWh, a rate); this is a total.
    """
    case_expr = " ".join(
        f"WHEN fueltype = '{ft}' THEN {factor}"
        for ft, factor in EMISSION_FACTORS_KG_PER_MWH.items()
    )
    query = f"""
        SELECT period, SUM(value * (CASE {case_expr} ELSE 0 END)) AS total_emissions_kg
        FROM fuel_mix
        GROUP BY period
        ORDER BY period
    """
    return con.execute(query).fetchdf()


def compute_deltas(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Hour-to-hour (Δemissions, Δdemand) pairs, restricted to genuinely
    adjacent hours only (period gap == exactly 1 hour). Rows spanning a
    real data gap are dropped, not silently included with a spurious delta.
    """
    demand = con.execute("SELECT period, value AS demand FROM pjm_demand ORDER BY period").fetchdf()
    emissions = hourly_total_emissions(con)

    df = demand.merge(emissions, on="period", how="inner").sort_values("period").reset_index(drop=True)

    gap_hours = df["period"].diff().dt.total_seconds() / 3600
    df["delta_demand"] = df["demand"].diff()
    df["delta_emissions"] = df["total_emissions_kg"].diff()

    real_adjacent = gap_hours == 1
    return df.loc[real_adjacent, ["period", "delta_demand", "delta_emissions"]].reset_index(drop=True)


def estimate_marginal_emissions_rate(delta_df: pd.DataFrame) -> dict:
    """Fits Δemissions = slope * Δdemand + intercept via OLS. The slope
    (kg CO2 / MWh) IS the marginal emissions rate estimate — the change in
    emissions per unit change in demand, i.e. the rate of whatever fuel
    responds on the margin.

    Also returns the slope's standard error and 95% confidence interval —
    a DIFFERENT question from R2. R2 measures how well the model predicts
    individual hourly outcomes (dragged down by genuine per-hour noise:
    renewables intermittency, untracked imports/exports, dispatch
    decisions unrelated to load). The slope's precision is a separate
    question, and with n in the many hundreds, it can be tight even when
    R2 is moderate. This is what actually matters for comparing whether
    two segments' rates are genuinely different — not just their point
    estimates, but whether their confidence intervals separate at all.
    """
    if len(delta_df) < 10:
        raise ValueError(
            f"Only {len(delta_df)} adjacent-hour pairs available — too few "
            f"to fit a meaningful regression."
        )

    x = delta_df["delta_demand"].values.astype(float)
    y = delta_df["delta_emissions"].values.astype(float)
    n = len(x)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(x.reshape(-1, 1), y))

    residuals = y - (slope * x + intercept)
    dof = n - 2
    sigma2 = np.sum(residuals ** 2) / dof
    sxx = np.sum((x - x.mean()) ** 2)
    slope_se = float(np.sqrt(sigma2 / sxx))

    t_crit = float(stats.t.ppf(0.975, dof))
    ci95_low = slope - t_crit * slope_se
    ci95_high = slope + t_crit * slope_se

    return {
        "marginal_rate_kg_per_mwh": slope,
        "intercept_kg": intercept,
        "r2": r2,
        "n": n,
        "slope_se": slope_se,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }


def _season(month: int) -> str:
    """Meteorological seasons: Winter=Dec-Feb, Spring=Mar-May,
    Summer=Jun-Aug, Fall=Sep-Nov."""
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall"}[month]


def estimate_marginal_emissions_by_segment(
    delta_df: pd.DataFrame, min_n: int = 100, min_r2: float = 0.15, max_abs_rate: float = 3000.0
) -> pd.DataFrame:
    """Fits a SEPARATE regression per (season, hour-of-day) bucket, rather
    than one global slope. Motivated directly by the global regression's
    R^2 ~= 0.49 on real data: the marginal responder genuinely differs by
    regime (different plants on the margin at 3am in April vs. 6pm in
    July), so a single system-wide slope necessarily averages over real
    regime-switching. This is what actually enables a claim like "a MW of
    flexibility is worth Nx more at time A than time B" — the global
    number alone cannot support that claim, since it IS one fixed number.

    THREE quality gates, not one:
    - min_n: enough points to attempt a regression at all.
    - min_r2: enough EXPLANATORY POWER to trust the slope (real example:
      (winter, hour=6) had n=691, comfortably over min_n, but r2=0.08 and
      a "rate" of 1,987 kg/MWh — exceeding coal's own emission factor,
      the highest in EMISSION_FACTORS_KG_PER_MWH).
    - max_abs_rate: a plausibility ceiling on the rate ITSELF. Found
      necessary because r2 alone is not sufficient — pure noise can
      spuriously clear an r2 threshold by chance in a small fraction of
      buckets (confirmed: with ~29 points per hour-bucket, r2=0.15 is only
      marginally above what noise alone produces sometimes), and near-
      singular regression on small noisy samples can then produce
      astronomically large slopes (observed: ~10^16 kg/MWh from a
      constructed pure-noise test case). Default 3000 kg/MWh — generous
      headroom above coal's ~1000 kg/MWh (the dirtiest fuel in
      EMISSION_FACTORS_KG_PER_MWH) for legitimate estimation error,
      while ruling out numerically-degenerate results.

    Segments failing any gate are skipped (logged with the specific
    reason), never silently dropped or trusted at face value.
    """
    df = delta_df.copy()
    df["season"] = df["period"].dt.month.map(_season)
    df["hour"] = df["period"].dt.hour

    rows = []
    for (season, hour), group in df.groupby(["season", "hour"]):
        if len(group) < min_n:
            log.warning(
                "segment (season=%s, hour=%d): only %d pairs, need >= %d — skipped (n)",
                season, hour, len(group), min_n,
            )
            continue
        result = estimate_marginal_emissions_rate(group)
        if result["r2"] < min_r2:
            log.warning(
                "segment (season=%s, hour=%d): r2=%.3f < %.2f (rate would have "
                "been %.1f kg/MWh) — skipped (r2), fit too weak to trust",
                season, hour, result["r2"], min_r2, result["marginal_rate_kg_per_mwh"],
            )
            continue
        if abs(result["marginal_rate_kg_per_mwh"]) > max_abs_rate:
            log.warning(
                "segment (season=%s, hour=%d): rate %.1f kg/MWh exceeds plausibility "
                "ceiling %.1f despite r2=%.3f clearing the gate — skipped (plausibility), "
                "likely a near-singular fit on a small/noisy sample",
                season, hour, result["marginal_rate_kg_per_mwh"], max_abs_rate, result["r2"],
            )
            continue
        rows.append({"season": season, "hour": hour, **result})

    if not rows:
        return pd.DataFrame(columns=[
            "season", "hour", "marginal_rate_kg_per_mwh", "intercept_kg", "r2", "n"
        ])
    return pd.DataFrame(rows).sort_values(["season", "hour"]).reset_index(drop=True)
