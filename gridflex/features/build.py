"""
Block 3.1 (+ Week 3 feature backlog) — feature engineering for system-level
demand forecasting.

Target: pjm_demand (system-wide) — NOT zone-level. PJM's own published
forecast (DF), our Week 3 benchmark, only exists at the system level, so the
credible head-to-head comparison has to be built here first. Zone-level
forecasting is a Week 4 extension for the flexibility engine, without an
external benchmark to score against — a different kind of validation.

Weather covariates are DEMAND-WEIGHTED averages across the 20 zones (each
zone's fixed weight = its share of total historical average demand), not a
naive unweighted mean — a zone carrying 15% of PJM's demand should influence
the system-wide weather signal more than one carrying 0.5%. This replaces
the earlier documented simplification from block 3.1.

Includes humidity, wind speed, and solar radiation (previously ingested but
unused), heating/cooling degree days (HDD/CDD, base 18.33C/65F — the
standard load-forecasting transform, more physically meaningful than raw
temperature), and short lags (t-1h, t-2h) alongside the existing seasonal
lags (24h, 48h, 168h).
"""

from __future__ import annotations

import duckdb
import holidays
import pandas as pd

US_HOLIDAYS = holidays.US(years=range(2018, 2028))

DEGREE_DAY_BASE_C = 18.33  # 65F — standard load-forecasting reference temp


def _zone_weights(con: duckdb.DuckDBPyConnection, as_of: pd.Timestamp | None = None) -> pd.Series:
    """Each zone's fixed weight = its share of PJM's historical average
    demand. A static weight vector (not recomputed per-hour) — simple and
    stable; a zone's relative size doesn't meaningfully shift hour to hour.

    as_of: if given, only uses subba_demand up to this timestamp. Without
    this, weights computed from the FULL history (including data inside
    every backtest fold) is a mild leak — found during a comprehensive
    leakage audit. In practice the impact is small (utility territory sizes
    are stable year to year), but it's real, so it's fixed rather than
    just documented.
    """
    where = "WHERE period <= ?" if as_of is not None else ""
    params = [as_of] if as_of is not None else []
    avg_by_zone = con.execute(f"""
        SELECT subba, AVG(value) AS avg_demand
        FROM subba_demand
        {where}
        GROUP BY subba
    """, params).fetchdf().set_index("subba")["avg_demand"]
    return avg_by_zone / avg_by_zone.sum()


def weighted_weather_from_df(weather_df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Demand-weighted average of each weather variable across zones, per
    hour. Works on ANY zone-level weather dataframe with columns (period,
    subba, temperature_2m, relative_humidity_2m, wind_speed_10m,
    shortwave_radiation) — used for BOTH historical archive data (training,
    from the DB) and live forecast data (inference, from Open-Meteo's
    forecast endpoint). One shared implementation, not two that could
    silently drift apart and disagree.
    """
    if weights.empty:
        raise ValueError(
            "Zone weights are empty — subba_demand has no rows. "
            "Demand-weighted weather depends on subba_demand; ensure "
            "zone-level demand has been ingested first."
        )

    w = weather_df.copy()
    w["_weight"] = w["subba"].map(weights).fillna(0.0)

    def wavg(col: str) -> pd.Series:
        num = (w[col] * w["_weight"]).groupby(w["period"]).sum()
        den = w.groupby("period")["_weight"].sum()
        return num / den.replace(0, pd.NA)

    out = pd.DataFrame({
        "temp_mean": wavg("temperature_2m"),
        "humidity_mean": wavg("relative_humidity_2m"),
        "wind_mean": wavg("wind_speed_10m"),
        "solar_mean": wavg("shortwave_radiation"),
    }).reset_index()
    return out.sort_values("period").reset_index(drop=True)


def _weighted_weather(con: duckdb.DuckDBPyConnection, weights: pd.Series) -> pd.DataFrame:
    """Thin wrapper: pulls archive weather from the DB, delegates the actual
    weighting math to weighted_weather_from_df (the shared implementation).
    """
    weather_df = con.execute("""
        SELECT period, subba, temperature_2m, relative_humidity_2m,
               wind_speed_10m, shortwave_radiation
        FROM weather
        ORDER BY period
    """).fetchdf()
    return weighted_weather_from_df(weather_df, weights)


def _assemble_features(demand_df: pd.DataFrame, weather_agg_df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature assembly: calendar features, HDD/CDD, lags, dropna.
    Used by BOTH build_training_table (system-level) and
    build_zone_training_table (per-zone, Week 4) — one implementation, not
    two that could quietly drift apart, same philosophy as
    weighted_weather_from_df.

    demand_df: columns (period, demand).
    weather_agg_df: columns (period, temp_mean, humidity_mean, wind_mean,
    solar_mean) — either the demand-weighted system aggregate, or a single
    zone's own raw weather, renamed to the same column names so this
    function (and FEATURE_COLUMNS, and the model code) doesn't need to know
    or care which.
    """
    df = demand_df.merge(weather_agg_df, on="period", how="left")
    df = df.sort_values("period").reset_index(drop=True)

    df["hdd"] = (DEGREE_DAY_BASE_C - df["temp_mean"]).clip(lower=0)
    df["cdd"] = (df["temp_mean"] - DEGREE_DAY_BASE_C).clip(lower=0)

    df["hour"] = df["period"].dt.hour
    df["dow"] = df["period"].dt.dayofweek  # 0=Mon
    df["month"] = df["period"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["period"].dt.date.astype("datetime64[ns]").isin(
        pd.to_datetime(list(US_HOLIDAYS.keys()))
    ).astype(int)

    # INVARIANT: every lag length here must be >= the backtest's
    # test_size_hours (default 24h, see gridflex/models/backtest.py), or it
    # silently leaks within-horizon actuals that a real day-ahead forecast
    # would never have. lag_1h/lag_2h were tried and REMOVED after producing
    # an implausibly good backtest result — PJM's real forecast horizon is
    # ~24h, never 1-2h.
    df["lag_24h"] = df["demand"].shift(24)
    df["lag_48h"] = df["demand"].shift(48)
    df["lag_168h"] = df["demand"].shift(168)

    n_before = len(df)
    # Check ALL lag columns, not just lag_168h — see the Week 3 dropna
    # symmetry fix. A null mid-series creates NaN in three different
    # downstream rows (one per lag length), not the same row three times.
    lag_cols = [c for c in df.columns if c.startswith("lag_") and c.endswith("h")]
    df = df.dropna(subset=[*lag_cols, "demand"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        import logging
        logging.getLogger(__name__).info(
            "_assemble_features: dropped %d row(s) lacking full lag history "
            "(expected — needs 168h of prior data)", n_dropped
        )

    return df


def build_training_table(
    con: duckdb.DuckDBPyConnection, weights_as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """One row per hour: system-wide target demand, calendar features,
    lags, demand-weighted system-wide weather.

    weights_as_of: passed to _zone_weights — for a leak-free backtest, pass
    a cutoff strictly before the earliest test fold begins (see
    safe_weights_cutoff() below).
    """
    demand = con.execute("""
        SELECT period, value AS demand
        FROM pjm_demand
        ORDER BY period
    """).fetchdf()

    weights = _zone_weights(con, as_of=weights_as_of)
    weather = _weighted_weather(con, weights)

    return _assemble_features(demand, weather)


def build_zone_training_table(con: duckdb.DuckDBPyConnection, zone: str) -> pd.DataFrame:
    """One row per hour: THIS ZONE's demand, calendar features, lags, and
    THIS ZONE's OWN weather (not the system-wide demand-weighted average —
    Philadelphia's demand responds to Philadelphia's temperature, not PJM's
    blended one). Week 4: zone-level forecasting for the flexibility engine.

    Uses the exact same _assemble_features as the system-level table, so
    FEATURE_COLUMNS and every model/backtest function work unchanged on
    either — only the demand and weather SOURCE differs.
    """
    demand = con.execute("""
        SELECT period, value AS demand
        FROM subba_demand
        WHERE subba = ?
        ORDER BY period
    """, [zone]).fetchdf()

    weather = con.execute("""
        SELECT period, temperature_2m AS temp_mean,
               relative_humidity_2m AS humidity_mean,
               wind_speed_10m AS wind_mean,
               shortwave_radiation AS solar_mean
        FROM weather
        WHERE subba = ?
        ORDER BY period
    """, [zone]).fetchdf()

    return _assemble_features(demand, weather)


FEATURE_COLUMNS = [
    "hour", "dow", "month", "is_weekend", "is_holiday",
    "temp_mean", "hdd", "cdd", "humidity_mean", "wind_mean", "solar_mean",
    "lag_24h", "lag_48h", "lag_168h",
]
TARGET_COLUMN = "demand"


def safe_weights_cutoff(
    con: duckdb.DuckDBPyConnection, n_splits: int = 5, test_size_hours: int = 24
) -> pd.Timestamp:
    """Latest timestamp zone weights can safely see without leaking into any
    backtest fold — strictly before the EARLIEST test fold begins. Mirrors
    the fold arithmetic in gridflex/models/backtest.py's walk_forward_splits
    (test windows are the last n_splits * test_size_hours hours of data).
    """
    max_period = con.execute("SELECT MAX(period) FROM pjm_demand").fetchone()[0]
    max_period = pd.Timestamp(max_period)
    if max_period.tzinfo is None:
        max_period = max_period.tz_localize("UTC")
    else:
        max_period = max_period.tz_convert("UTC")
    return max_period - pd.Timedelta(hours=n_splits * test_size_hours)
