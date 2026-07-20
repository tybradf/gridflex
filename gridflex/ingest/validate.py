"""
Schema validation (block 3.1) — pandera, lazy validation.

Design: validate-and-filter, not validate-and-crash. A nightly cron
shouldn't die because EIA published one bad reading in one zone — but a bad
row silently entering the store (like the ~1.5e9 MW rows found manually in
Session 2) is worse. So: pandera runs in `lazy=True` mode, which collects
ALL failing rows instead of stopping at the first one; we then drop exactly
those rows (by index) before upsert, and log everything dropped and why.

This directly encodes the outliers found manually in Session 2
(DOM ~1.5e9 MW, CE ~142k MW) as an automatic, permanent, loud check.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandera.pandas as pa

from gridflex.config import PLAUSIBLE_RANGES

log = logging.getLogger(__name__)


def _schema_for(table: str) -> pa.DataFrameSchema:
    lo, hi = PLAUSIBLE_RANGES.get(table, (0, float("inf")))
    return pa.DataFrameSchema(
        {
            "period": pa.Column(pa.DateTime, nullable=False),
            "value": pa.Column(
                float,
                checks=pa.Check.in_range(lo, hi, error=f"value outside plausible [{lo}, {hi}]"),
                nullable=False,
                coerce=True,
            ),
        },
        strict=False,  # other columns (subba, respondent, fueltype, ...) pass through unchecked
    )


def detect_spike_rows(
    df: pd.DataFrame, value_col: str = "value", threshold: float = 30_000
) -> pd.Series:
    """Flags CONTEXTUAL outliers: a single row whose value spikes sharply
    away from BOTH immediate neighbors and reverts — distinct from
    PLAUSIBLE_RANGES' absolute check. A spike can land well inside a
    plausible range and still be obviously wrong given its neighbors.
    Found via a real example: 215,682 MW sandwiched between two ~70,000 MW
    readings (73,999 -> 215,682 -> 68,741) — well inside the pjm_demand
    [0, 250000] plausible range, but physically impossible as a genuine
    hour-to-hour swing. This would have silently corrupted a
    marginal-emissions Delta-demand regression if left uncaught.

    df must be sorted by period ascending (caller's responsibility — this
    function re-sorts defensively). Only flags rows where BOTH neighbors
    are genuinely adjacent (period gap == exactly 1 hour) — a row next to a
    real DATA GAP is never flagged, since the delta there isn't a
    meaningful hourly comparison (see the Week 4 positional-vs-calendar
    lesson: naive .diff() across a gap produces a spurious huge delta that
    has nothing to do with a real spike).
    """
    df = df.sort_values("period").reset_index(drop=True)

    gap_before = df["period"].diff().dt.total_seconds() / 3600
    gap_after = -df["period"].diff(-1).dt.total_seconds() / 3600

    delta_in = df[value_col].diff()
    delta_out = -df[value_col].diff(-1)

    real_neighbors = (gap_before == 1) & (gap_after == 1)
    is_spike = (
        real_neighbors
        & (delta_in.abs() > threshold)
        & (delta_out.abs() > threshold)
        & (pd.Series(np.sign(delta_in)) != pd.Series(np.sign(delta_out)))
    )
    return is_spike.fillna(False)


def validate_and_filter(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Validate df against table's schema. Rows that fail are dropped and
    logged loudly; rows that pass are returned. Never raises — a bad row
    should never take down the whole ingest run."""
    if df.empty:
        return df

    schema = _schema_for(table)
    n0 = len(df)

    try:
        schema.validate(df, lazy=True)
        return df  # everything passed
    except pa.errors.SchemaErrors as err:
        bad_idx = set(err.failure_cases["index"].dropna().astype(int))
        clean = df.drop(index=[i for i in bad_idx if i in df.index])

        cols = [c for c in ("period", "subba", "value") if c in df.columns]
        log.error(
            "%s: pandera rejected %d/%d row(s):\n%s",
            table, len(bad_idx), n0, df.loc[list(bad_idx), cols].to_string(),
        )
        return clean
