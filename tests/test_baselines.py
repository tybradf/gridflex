import numpy as np
import pandas as pd

from gridflex.models.backtest import run_backtest
from gridflex.models.baselines import make_lightgbm_predict_fn, seasonal_naive_predict


def test_seasonal_naive_is_exactly_lag_168h():
    test_df = pd.DataFrame({"lag_168h": [1.0, 2.0, 3.0]})
    preds = seasonal_naive_predict(None, test_df)
    assert (preds == test_df["lag_168h"].values).all()


def test_seasonal_naive_requires_lag_column():
    test_df = pd.DataFrame({"other_col": [1.0]})
    try:
        seasonal_naive_predict(None, test_df)
        assert False, "should have raised"
    except ValueError:
        pass


def _synthetic_seasonal_df(seed=0):
    np.random.seed(seed)
    n = 24 * 90 + 5 * 48 + 168 + 20  # account for the 168 rows lost to lag dropna
    periods = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    hour, dow = periods.hour.values, periods.dayofweek.values
    demand = (100_000 + 20_000 * np.sin((hour - 6) / 24 * 2 * np.pi)
              - 15_000 * (dow >= 5) + np.random.normal(0, 500, n))
    df = pd.DataFrame({"period": periods, "demand": demand})
    df["hour"], df["dow"], df["month"] = hour, dow, periods.month
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_holiday"] = 0
    df["temp_mean"] = 20 + 10 * np.sin((hour - 6) / 24 * 2 * np.pi)
    df.loc[df.index[-12:], "temp_mean"] = np.nan  # real trailing weather-lag gap
    df["hdd"] = (18.33 - df["temp_mean"]).clip(lower=0)
    df["cdd"] = (df["temp_mean"] - 18.33).clip(lower=0)
    df["humidity_mean"] = 50.0
    df["wind_mean"] = 5.0
    df["solar_mean"] = 100.0
    df["lag_24h"] = df["demand"].shift(24)
    df["lag_48h"] = df["demand"].shift(48)
    df["lag_168h"] = df["demand"].shift(168)
    return df.dropna(subset=["lag_168h", "demand"]).reset_index(drop=True)


def test_baselines_beat_naive_mean_floor_on_seasonal_data():
    """Both real baselines must dramatically beat 'predict the training
    mean' on data with obvious seasonal structure — if they don't, something
    is broken (wrong features passed, model not actually fitting, etc.)."""
    df = _synthetic_seasonal_df()

    def naive_mean_model(train_df, test_df):
        return np.full(len(test_df), train_df["demand"].mean())

    naive = run_backtest(df, "demand", naive_mean_model, n_splits=5, test_size_hours=48)
    seasonal = run_backtest(df, "demand", seasonal_naive_predict, n_splits=5, test_size_hours=48)

    assert seasonal["mae"].mean() < naive["mae"].mean() * 0.5


def test_lightgbm_handles_nan_features_and_beats_floor():
    """Confirms LightGBM trains/predicts cleanly through NaN weather values
    (the real trailing-lag gap found in this project) without crashing, and
    that it actually learns the seasonal pattern rather than degenerating."""
    df = _synthetic_seasonal_df()

    def naive_mean_model(train_df, test_df):
        return np.full(len(test_df), train_df["demand"].mean())

    naive = run_backtest(df, "demand", naive_mean_model, n_splits=5, test_size_hours=48)
    lgb_results = run_backtest(df, "demand", make_lightgbm_predict_fn(), n_splits=5, test_size_hours=48)

    assert lgb_results["mae"].mean() < naive["mae"].mean() * 0.5
    assert not lgb_results["mae"].isna().any()
