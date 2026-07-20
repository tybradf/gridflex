"""
Week 4, block 4.3 — the flexible-demand value engine.

Given a flexible load currently scheduled for some hour, with a shiftable
window of H hours, find the statistically defensible best hour to move it
to and compute the emissions avoided.

CRITICAL DESIGN CONSTRAINT: only recommend a shift where the improvement is
real, not noise. A candidate hour must (a) have cleared all three quality
gates from block 4.1 (n, r2, plausibility) to even be considered, and
(b) have a 95% CI that does NOT overlap the origin hour's CI — otherwise
the "optimal" recommendation would be built on a difference indistinguishable
from statistical noise, exactly the failure mode the whole segmentation
exercise was built to avoid.

Emissions rate here is SYSTEM-WIDE, not zone-specific (fuel_mix only exists
at the system level — see README known limitations). So the emissions-
avoided calculation is the same regardless of which zone the flexible load
is in. Zone-level demand only enters for peak-reduction (a genuinely
separate question), kept as a distinct function rather than blended into
one number.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def _ci_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return not (a_low > b_high or b_low > a_high)


def find_best_shift_hour(
    segment_df: pd.DataFrame, season: str, origin_hour: int, window_hours: int
) -> dict:
    """Evaluates every candidate hour in [origin_hour, origin_hour+window_hours)
    (wrapping mod 24) for the given season, using the three-gate-validated
    segment_df (output of estimate_marginal_emissions_by_segment). Picks the
    candidate with the LOWEST marginal rate among those whose CI does NOT
    overlap the origin hour's CI — a statistically defensible improvement,
    not just a lower point estimate.

    Returns a dict with 'feasible': bool and an explanation either way —
    never a fabricated "best hour" built on a difference that isn't real.
    """
    season_df = segment_df[segment_df["season"] == season]
    origin_row = season_df[season_df["hour"] == origin_hour]

    if origin_row.empty:
        return {
            "feasible": False,
            "reason": f"origin hour {origin_hour} ({season}) did not clear the "
            f"quality gates — no reliable baseline rate to compare against.",
        }
    origin = origin_row.iloc[0]

    candidate_hours = [(origin_hour + h) % 24 for h in range(window_hours)]
    candidates = season_df[season_df["hour"].isin(candidate_hours)]
    candidates = candidates[candidates["hour"] != origin_hour]

    if candidates.empty:
        return {
            "feasible": False,
            "reason": f"no candidate hour in the {window_hours}h window cleared "
            f"the quality gates.",
            "origin_rate": float(origin["marginal_rate_kg_per_mwh"]),
        }

    defensible = candidates[
        candidates.apply(
            lambda r: not _ci_overlap(
                r["ci95_low"], r["ci95_high"], origin["ci95_low"], origin["ci95_high"]
            )
            and r["marginal_rate_kg_per_mwh"] < origin["marginal_rate_kg_per_mwh"],
            axis=1,
        )
    ]

    if defensible.empty:
        return {
            "feasible": False,
            "reason": f"no candidate hour in the {window_hours}h window showed a "
            f"statistically defensible improvement over hour {origin_hour} "
            f"(either not cleaner, or the difference wasn't distinguishable "
            f"from noise via non-overlapping confidence intervals).",
            "origin_rate": float(origin["marginal_rate_kg_per_mwh"]),
        }

    best = defensible.loc[defensible["marginal_rate_kg_per_mwh"].idxmin()]
    return {
        "feasible": True,
        "origin_hour": origin_hour,
        "origin_rate": float(origin["marginal_rate_kg_per_mwh"]),
        "best_hour": int(best["hour"]),
        "best_rate": float(best["marginal_rate_kg_per_mwh"]),
        "rate_reduction_kg_per_mwh": float(origin["marginal_rate_kg_per_mwh"] - best["marginal_rate_kg_per_mwh"]),
    }


def emissions_avoided_kg(mw: float, rate_origin: float, rate_target: float) -> float:
    """kg CO2 avoided by shifting `mw` of load from a rate_origin-kg/MWh
    hour to a rate_target-kg/MWh hour. Positive = avoided (target cleaner);
    negative = added (target dirtier — e.g. shifting the wrong direction)."""
    return mw * (rate_origin - rate_target)


def value_of_shift(
    segment_df: pd.DataFrame, mw: float, origin_hour: int, window_hours: int, season: str
) -> dict:
    """Top-level entry point: finds the best defensible shift hour and
    computes the emissions value, or explains clearly why no defensible
    shift exists rather than fabricating one."""
    shift = find_best_shift_hour(segment_df, season, origin_hour, window_hours)
    if not shift["feasible"]:
        return {"feasible": False, "mw": mw, "season": season, **shift}

    avoided = emissions_avoided_kg(mw, shift["origin_rate"], shift["best_rate"])
    return {
        "feasible": True,
        "mw": mw,
        "season": season,
        "origin_hour": shift["origin_hour"],
        "best_hour": shift["best_hour"],
        "origin_rate_kg_per_mwh": shift["origin_rate"],
        "best_rate_kg_per_mwh": shift["best_rate"],
        "emissions_avoided_kg": avoided,
    }


SEASON_MONTHS = {
    "winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "fall": [9, 10, 11],
}


def zone_typical_demand(con: duckdb.DuckDBPyConnection, zone: str, season: str, hour: int) -> float:
    """Average zone demand at a given (season, hour) — a simple TYPICAL-load
    proxy, not a full peak-day distributional analysis. Uses the same season
    boundaries as the marginal-emissions segmentation (block 4.1) for
    consistency between the two halves of "value."
    """
    months = SEASON_MONTHS[season]
    placeholders = ",".join("?" * len(months))
    result = con.execute(f"""
        SELECT AVG(value) FROM subba_demand
        WHERE subba = ? AND EXTRACT(hour FROM period) = ?
          AND EXTRACT(month FROM period) IN ({placeholders})
    """, [zone, hour, *months]).fetchone()[0]
    if result is None:
        raise ValueError(f"No subba_demand data for zone={zone}, season={season}, hour={hour}")
    return float(result)


def zone_seasonal_peak(con: duckdb.DuckDBPyConnection, zone: str, season: str) -> float:
    """Peak of the TYPICAL daily curve (max of the 24 hourly averages) — NOT
    the all-time absolute peak, which would be dominated by rare extreme
    events and inconsistent with using averages for the hourly comparison.
    """
    return max(zone_typical_demand(con, zone, season, h) for h in range(24))


def peak_relief(
    con: duckdb.DuckDBPyConnection, zone: str, mw: float, origin_hour: int, target_hour: int, season: str
) -> dict:
    """Zone-level context for a shift — deliberately SEPARATE from
    emissions (which is system-wide, not zone-specific — fuel_mix doesn't
    exist per zone, see README known limitations). Reports typical demand
    at both hours and how much of the zone's typical seasonal peak the
    shifted MW represents — context for a grid operator's capacity/
    reliability question, distinct from a sustainability team's emissions
    question.
    """
    origin_demand = zone_typical_demand(con, zone, season, origin_hour)
    target_demand = zone_typical_demand(con, zone, season, target_hour)
    seasonal_peak = zone_seasonal_peak(con, zone, season)

    return {
        "zone": zone,
        "season": season,
        "origin_hour": origin_hour,
        "origin_typical_demand_mw": origin_demand,
        "target_hour": target_hour,
        "target_typical_demand_mw": target_demand,
        "zone_seasonal_peak_mw": seasonal_peak,
        "origin_pct_of_seasonal_peak": origin_demand / seasonal_peak * 100,
        "mw_shifted": mw,
        "mw_shifted_pct_of_peak": mw / seasonal_peak * 100,
    }


def full_value_of_shift(
    con: duckdb.DuckDBPyConnection,
    segment_df: pd.DataFrame,
    zone: str,
    mw: float,
    origin_hour: int,
    window_hours: int,
    season: str,
) -> dict:
    """Combines emissions value (system-wide) and peak-relief context
    (zone-level) into one result — kept as two clearly labeled sub-dicts,
    not blended into a single fuzzy number, since they answer genuinely
    different stakeholder questions.
    """
    emissions = value_of_shift(segment_df, mw, origin_hour, window_hours, season)
    result = {"emissions": emissions}

    if emissions["feasible"]:
        result["peak_context"] = peak_relief(
            con, zone, mw, origin_hour, emissions["best_hour"], season
        )
    else:
        result["peak_context"] = None

    return result
