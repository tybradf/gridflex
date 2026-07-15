"""
Block 3.6 — live forecast generation and the scoreboard.

This is what makes the project genuinely LIVE rather than a static backtest:
generate_forecast() predicts the next N hours using actual forecast weather
(Open-Meteo's /v1/forecast endpoint), not historical/archive weather — the
gap flagged back when we first found the archive endpoint's inherent lag.

live_scoreboard() reuses compute_metrics() from gridflex.models.backtest
directly — the same measurement used for the whole Week 3 backtest, so the
live scoreboard and the offline evaluation are provably the same yardstick,
not two implementations that could quietly disagree.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb
import lightgbm as lgb
import pandas as pd

from gridflex.features.build import (
    DEGREE_DAY_BASE_C,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    _zone_weights,
    build_training_table,
    weighted_weather_from_df,
)
from gridflex.ingest.weather import fetch_forecast
from gridflex.models.backtest import compute_metrics
from gridflex.store.db import upsert

_HOLIDAYS_CACHE = None


def _future_calendar_features(periods: pd.DatetimeIndex) -> pd.DataFrame:
    import holidays
    global _HOLIDAYS_CACHE
    if _HOLIDAYS_CACHE is None:
        _HOLIDAYS_CACHE = holidays.US(years=range(2018, 2028))
    holiday_dates = pd.to_datetime(list(_HOLIDAYS_CACHE.keys()))

    df = pd.DataFrame({"period": periods})
    df["hour"] = df["period"].dt.hour
    df["dow"] = df["period"].dt.dayofweek
    df["month"] = df["period"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["period"].dt.date.astype("datetime64[ns]").isin(holiday_dates).astype(int)
    return df


def generate_forecast(
    con: duckdb.DuckDBPyConnection,
    horizon_hours: int = 24,
    weather_fetch_fn: Callable[..., pd.DataFrame] = fetch_forecast,
) -> pd.DataFrame:
    """Trains on all available history, predicts the next horizon_hours
    using live forecast weather. weather_fetch_fn is injectable so tests
    never hit the real Open-Meteo API — same pattern used for the EIA
    client's pagination tests back in Week 1.
    """
    train_df = build_training_table(con)  # weights_as_of=None: live deployment
    # wants the freshest zone weights, unlike backtesting where leakage matters.

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                               num_leaves=31, verbosity=-1)
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    last_known = train_df["period"].max()
    future_periods = pd.date_range(
        last_known + pd.Timedelta(hours=1), periods=horizon_hours, freq="h", tz="UTC"
    )

    # Lag features: looked up directly from actual demand history — always
    # available since every lag length (24h/48h/168h) is longer than any
    # gap between "now" and the forecast horizon.
    demand_by_period = con.execute("SELECT period, value FROM pjm_demand").fetchdf()
    demand_lookup = demand_by_period.set_index("period")["value"]

    future = _future_calendar_features(future_periods)
    for lag_h, col in [(24, "lag_24h"), (48, "lag_48h"), (168, "lag_168h")]:
        future[col] = [demand_lookup.get(p - pd.Timedelta(hours=lag_h)) for p in future_periods]

    missing_lags = future[["lag_24h", "lag_48h", "lag_168h"]].isna().any(axis=1).sum()
    if missing_lags:
        raise ValueError(
            f"{missing_lags} forecast row(s) missing lag data — demand history "
            f"doesn't reach far enough back. Run ingest before forecasting."
        )

    # Weather: LIVE forecast, not archive — the actual gap this block exists to close.
    weather_df = weather_fetch_fn(past_days=1, forecast_days=max(2, horizon_hours // 24 + 1))
    weights = _zone_weights(con)  # freshest weights for live use
    weighted = weighted_weather_from_df(weather_df, weights)
    future = future.merge(weighted, on="period", how="left")

    future["hdd"] = (DEGREE_DAY_BASE_C - future["temp_mean"]).clip(lower=0)
    future["cdd"] = (future["temp_mean"] - DEGREE_DAY_BASE_C).clip(lower=0)

    missing_weather = future["temp_mean"].isna().sum()
    if missing_weather:
        raise ValueError(
            f"{missing_weather} forecast row(s) missing weather — the forecast "
            f"horizon may extend beyond what weather_fetch_fn returned."
        )

    predictions = model.predict(future[FEATURE_COLUMNS])
    return pd.DataFrame({
        "period": future_periods,
        "predicted_demand": predictions,
        "generated_at": pd.Timestamp.now(tz="UTC"),
    })


def store_forecast(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return upsert(con, "forecasts", df)


def live_scoreboard(con: duckdb.DuckDBPyConnection, days: int = 7) -> dict:
    """Joins stored forecasts to actuals (now known) and PJM's own DF for the
    SAME periods, and scores both through compute_metrics — the identical
    function used throughout Week 3's backtesting, so this is provably the
    same yardstick, not a separate live-only measurement.
    """
    joined = con.execute("""
        WITH anchor AS (SELECT MAX(period) AS t FROM forecasts)
        SELECT f.period, f.predicted_demand, d.value AS actual, df.value AS pjm_forecast
        FROM forecasts f
        JOIN pjm_demand d ON f.period = d.period
        JOIN pjm_forecast df ON f.period = df.period
        CROSS JOIN anchor
        WHERE f.period > anchor.t - INTERVAL (?) DAY
        ORDER BY f.period
    """, [days]).fetchdf()

    if joined.empty:
        return {"n_scored": 0, "ours": None, "pjm": None, "rows": []}

    ours = compute_metrics(joined["actual"].values, joined["predicted_demand"].values)
    pjm = compute_metrics(joined["actual"].values, joined["pjm_forecast"].values)
    return {
        "n_scored": len(joined),
        "ours": ours,
        "pjm": pjm,
        "rows": joined.to_dict(orient="records"),
    }
