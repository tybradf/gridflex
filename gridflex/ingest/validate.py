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
