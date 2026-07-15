"""
Block 3.4 — the benchmark comparison.

Attaches PJM's own published day-ahead forecast (DF) to the same table used
for backtesting, so it can be scored through the IDENTICAL harness as our
own models — apples to apples, not two different measurement approaches.

CRITICAL: pjm_forecast_mw is deliberately NOT added to FEATURE_COLUMNS. If
LightGBM could see PJM's own forecast as an input feature, it could partially
just learn to copy it — that would quietly invalidate the "independent
model" comparison. This column exists ONLY to be read directly as a
competing prediction, exactly like lag_168h is for seasonal_naive_predict.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def attach_pjm_forecast(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    forecast = con.execute("""
        SELECT period, value AS pjm_forecast_mw
        FROM pjm_forecast
        ORDER BY period
    """).fetchdf()
    return df.merge(forecast, on="period", how="left")


def pjm_forecast_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Reads PJM's own forecast for the test period directly — mirrors how
    seasonal_naive_predict reads lag_168h. Fails loudly rather than silently
    if PJM's forecast has gaps in this window, since a comparison scored
    over missing benchmark data would be misleading, not just incomplete."""
    if "pjm_forecast_mw" not in test_df.columns:
        raise ValueError("pjm_forecast_mw missing — did attach_pjm_forecast run?")
    vals = test_df["pjm_forecast_mw"]
    if vals.isna().any():
        window = (
            f"({test_df['period'].min()} .. {test_df['period'].max()})"
            if "period" in test_df.columns else "(period column not present)"
        )
        raise ValueError(
            f"PJM's own forecast has {vals.isna().sum()} missing value(s) in this "
            f"test window {window} — benchmark comparison would be misleading. "
            f"Investigate before scoring."
        )
    return vals.values
