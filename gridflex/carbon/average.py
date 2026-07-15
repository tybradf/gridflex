"""
Block 2.1 — average carbon intensity, system-wide (PJM as a whole).

NOTE this is system-wide, not zone-level — see README known limitations
(EIA-930 only reports fuel mix at the balancing-authority level). Every zone
gets the same carbon intensity value for a given hour; what varies spatially
in this project is DEMAND, not (yet) carbon intensity. True zone-level carbon
needs EPA CAMPD unit-level emissions — a documented Week 4 stretch goal.

Carbon intensity (kg CO2/MWh) = generation-weighted average of each fuel
type's emission factor, i.e.:
    sum(generation_i * factor_i) / sum(generation_i)   for each hour
"""

from __future__ import annotations

import duckdb
import pandas as pd

from gridflex.config import EMISSION_FACTORS_KG_PER_MWH


def carbon_intensity_by_hour(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Returns a DataFrame with one row per hour: period, carbon_intensity
    (kg CO2/MWh), and total_generation (MWh, for sanity-checking / weighting
    context). Computed via SQL directly against fuel_mix for efficiency at
    scale, rather than pulling ~500k rows into pandas first.
    """
    # Build a SQL CASE expression from the emission factors dict rather than
    # a join to a tiny lookup table — simpler for a fixed, small factor set.
    case_expr = " ".join(
        f"WHEN fueltype = '{ft}' THEN {factor}"
        for ft, factor in EMISSION_FACTORS_KG_PER_MWH.items()
    )

    query = f"""
        SELECT
            period,
            SUM(value * (CASE {case_expr} ELSE 0 END)) / NULLIF(SUM(value), 0)
                AS carbon_intensity,
            SUM(value) AS total_generation
        FROM fuel_mix
        GROUP BY period
        ORDER BY period
    """
    return con.execute(query).fetchdf()


def latest_carbon_intensity(con: duckdb.DuckDBPyConnection, hours: int = 48) -> pd.DataFrame:
    """Same as carbon_intensity_by_hour but limited to the most recent N
    hours — what the dashboard actually needs, not the full history."""
    df = carbon_intensity_by_hour(con)
    return df.tail(hours).reset_index(drop=True)
