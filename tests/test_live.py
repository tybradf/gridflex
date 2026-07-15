import numpy as np
import pandas as pd

from gridflex.models.live import generate_forecast, live_scoreboard, store_forecast
from gridflex.store.db import upsert


def _seed_history(con, n=24 * 90 + 200, seed=2):
    periods = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hour, dow = periods.hour.values, periods.dayofweek.values
    np.random.seed(seed)
    demand = (100_000 + 20_000 * np.sin((hour - 6) / 24 * 2 * np.pi)
              - 15_000 * (dow >= 5) + np.random.normal(0, 500, n))

    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": demand,
    }))
    upsert(con, "subba_demand", pd.DataFrame({
        "period": list(periods) * 2, "subba": ["PE"] * n + ["CE"] * n,
        "parent": ["PJM"] * n * 2, "value": [100.0] * n + [300.0] * n,
    }))
    upsert(con, "weather", pd.DataFrame({
        "period": list(periods) * 2, "subba": ["PE"] * n + ["CE"] * n,
        "temperature_2m": list(20 + 10 * np.sin((hour - 6) / 24 * 2 * np.pi)) * 2,
        "relative_humidity_2m": [50.0] * n * 2, "wind_speed_10m": [5.0] * n * 2,
        "shortwave_radiation": [100.0] * n * 2,
    }))
    return periods, demand


def _fake_weather_fetch(last_period):
    def fn(**kwargs):
        future = pd.date_range(last_period + pd.Timedelta(hours=1), periods=48, freq="h", tz="UTC")
        fh = future.hour.values
        return pd.DataFrame({
            "period": list(future) * 2, "subba": ["PE"] * 48 + ["CE"] * 48,
            "temperature_2m": list(20 + 10 * np.sin((fh - 6) / 24 * 2 * np.pi)) * 2,
            "relative_humidity_2m": [50.0] * 96, "wind_speed_10m": [5.0] * 96,
            "shortwave_radiation": [100.0] * 96,
        })
    return fn


def test_generate_forecast_shape_and_horizon(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    periods, _ = _seed_history(con)

    fc = generate_forecast(con, horizon_hours=24, weather_fetch_fn=_fake_weather_fetch(periods.max()))
    con.close()

    assert len(fc) == 24
    assert fc["period"].min() == periods.max() + pd.Timedelta(hours=1)
    assert not fc["predicted_demand"].isna().any()


def test_generate_forecast_raises_on_missing_weather(tmp_db):
    """Fails loudly rather than silently forecasting with incomplete
    weather data — a real failure mode if the forecast horizon extends
    beyond what the weather API returns."""
    from gridflex.store.db import get_connection
    con = get_connection()
    periods, _ = _seed_history(con)

    def broken_weather_fetch(**kwargs):
        return pd.DataFrame(columns=["period", "subba", "temperature_2m",
                                      "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"])

    try:
        generate_forecast(con, horizon_hours=24, weather_fetch_fn=broken_weather_fetch)
        assert False, "should have raised on missing weather"
    except ValueError:
        pass
    con.close()


def test_store_and_score_end_to_end(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    periods, demand = _seed_history(con)

    fc = generate_forecast(con, horizon_hours=24, weather_fetch_fn=_fake_weather_fetch(periods.max()))
    store_forecast(con, fc)

    future_periods = fc["period"]
    np.random.seed(3)
    actual_vals = demand[-1] + np.random.normal(0, 200, 24)
    pjm_vals = actual_vals + np.random.normal(0, 300, 24)
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": future_periods, "respondent": ["PJM"] * 24, "type": ["D"] * 24, "value": actual_vals,
    }))
    upsert(con, "pjm_forecast", pd.DataFrame({
        "period": future_periods, "respondent": ["PJM"] * 24, "type": ["DF"] * 24, "value": pjm_vals,
    }))

    score = live_scoreboard(con, days=7)
    con.close()

    assert score["n_scored"] == 24
    assert set(score["ours"].keys()) == {"mae", "rmse", "mape", "n"}
    assert set(score["pjm"].keys()) == {"mae", "rmse", "mape", "n"}


def test_scoreboard_empty_when_no_forecasts_stored(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()
    _seed_history(con)
    score = live_scoreboard(con, days=7)
    con.close()
    assert score["n_scored"] == 0
    assert score["ours"] is None
