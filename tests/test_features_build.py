import pandas as pd

from gridflex.features.build import DEGREE_DAY_BASE_C, FEATURE_COLUMNS, build_training_table
from gridflex.store.db import upsert


def _seed(con, n_hours=200, start="2024-06-25T00:00"):
    periods = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    demand_df = pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n_hours,
        "type": ["D"] * n_hours, "value": [100_000.0 + i for i in range(n_hours)],
    })
    upsert(con, "pjm_demand", demand_df)

    # Deliberately UNEQUAL zone demand shares (PE=100, CE=300 -> weights
    # 0.25/0.75) so tests can actually distinguish demand-weighted averaging
    # from a naive unweighted mean, rather than accidentally matching both.
    subba_df = pd.DataFrame({
        "period": list(periods) * 2,
        "subba": ["PE"] * n_hours + ["CE"] * n_hours,
        "parent": ["PJM"] * n_hours * 2,
        "value": [100.0] * n_hours + [300.0] * n_hours,
    })
    upsert(con, "subba_demand", subba_df)

    weather_df = pd.DataFrame({
        "period": list(periods) * 2,
        "subba": ["PE"] * n_hours + ["CE"] * n_hours,
        "temperature_2m": [20.0] * n_hours + [30.0] * n_hours,
        "relative_humidity_2m": [50.0] * n_hours * 2,
        "wind_speed_10m": [5.0] * n_hours * 2,
        "shortwave_radiation": [100.0] * n_hours * 2,
    })
    upsert(con, "weather", weather_df)
    return periods


def test_lag_features_are_exact(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed(con)
    df = build_training_table(con)
    con.close()

    row0 = df.iloc[0]
    assert row0["lag_24h"] == row0["demand"] - 24
    assert row0["lag_48h"] == row0["demand"] - 48
    assert row0["lag_168h"] == row0["demand"] - 168


def test_no_lag_shorter_than_default_test_horizon():
    """The actual bug found via a suspiciously-good backtest result:
    lag_1h/lag_2h leaked within-horizon actuals a real day-ahead forecast
    would never have. This is the permanent guard — any lag feature must be
    >= DEFAULT_TEST_SIZE_HOURS, or it's unsafe for the DF benchmark
    comparison regardless of how good it makes the numbers look."""
    from gridflex.models.backtest import walk_forward_splits
    import inspect
    default_test_size = inspect.signature(walk_forward_splits).parameters["test_size_hours"].default

    lag_cols = [c for c in FEATURE_COLUMNS if c.startswith("lag_") and c.endswith("h")]
    assert lag_cols, "expected at least one lag feature"
    for col in lag_cols:
        hours = int(col.removeprefix("lag_").removesuffix("h"))
        assert hours >= default_test_size, (
            f"{col} ({hours}h) is shorter than the default test horizon "
            f"({default_test_size}h) — this WILL leak within-horizon actuals"
        )


def test_drops_rows_without_full_lag_history(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed(con, n_hours=200)
    df = build_training_table(con)
    con.close()
    assert len(df) == 200 - 168


def test_holiday_flag_fires_on_july_4th_only(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed(con)
    df = build_training_table(con)
    con.close()

    july4 = df[df["period"].dt.date == pd.Timestamp("2024-07-04").date()]
    july3 = df[df["period"].dt.date == pd.Timestamp("2024-07-03").date()]
    assert (july4["is_holiday"] == 1).all()
    assert (july3["is_holiday"] == 0).all()


def test_weather_is_demand_weighted_not_unweighted(tmp_db):
    """PE (weight 0.25, temp 20) + CE (weight 0.75, temp 30) should give
    27.5, NOT the naive unweighted mean of 25.0 — this is the test that
    actually distinguishes the demand-weighted implementation from the old
    simplification it replaced."""
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed(con)
    df = build_training_table(con)
    con.close()

    assert (abs(df["temp_mean"] - 27.5) < 1e-6).all()
    assert not (abs(df["temp_mean"] - 25.0) < 1e-6).all(), (
        "temp_mean matches the OLD unweighted mean — weighting isn't being applied"
    )


def test_hdd_cdd_computed_from_weighted_temp(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed(con)  # weighted temp = 27.5, above the 18.33 base -> CDD only
    df = build_training_table(con)
    con.close()

    expected_cdd = 27.5 - DEGREE_DAY_BASE_C
    assert (abs(df["cdd"] - expected_cdd) < 1e-6).all()
    assert (df["hdd"] == 0).all()


def test_humidity_wind_solar_present_and_weighted(tmp_db):
    """These were ingested since Week 1 but unused as features until now —
    confirm they're actually present and computed."""
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed(con)
    df = build_training_table(con)
    con.close()

    for col, pe_val, ce_val in [
        ("humidity_mean", 50.0, 50.0),
        ("wind_mean", 5.0, 5.0),
        ("solar_mean", 100.0, 100.0),
    ]:
        expected = 0.25 * pe_val + 0.75 * ce_val
        assert (abs(df[col] - expected) < 1e-6).all(), col
