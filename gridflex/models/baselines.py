"""
Block 3.3 — baselines.

seasonal_naive_predict: predicts demand at hour t as demand at t-168h (same
hour, same day-of-week, one week prior). This is the standard baseline for
load forecasting with weekly seasonality — NOT t-24h, because a plain
24h-ago baseline gets weekday/weekend transitions wrong (e.g. predicting
Monday from Sunday's very different demand shape). Any real model needs to
beat this, or it isn't earning its complexity.

lightgbm_predict: gradient-boosted trees on the calendar + lag + weather
features from build.py. Handles NaN features (e.g. the weather archive's
trailing lag, see Week 3 diagnosis) natively — no imputation needed.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridflex.features.build import FEATURE_COLUMNS, TARGET_COLUMN


def seasonal_naive_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """The floor. Doesn't even need train_df — lag_168h (built in 3.1) IS
    the prediction by definition of what seasonal-naive means."""
    if "lag_168h" not in test_df.columns:
        raise ValueError("seasonal_naive_predict requires lag_168h — did build_training_table run?")
    return test_df["lag_168h"].values


def make_lightgbm_predict_fn(**lgb_params):
    """Returns a predict_fn closure so hyperparameters can be configured
    without changing the harness call site. Default params are deliberately
    modest (small model) — this is a baseline, not the final tuned model."""
    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "verbosity": -1,
        **lgb_params,
    }

    def predict_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        model = lgb.LGBMRegressor(**params)
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        return model.predict(test_df[FEATURE_COLUMNS])

    return predict_fn


# A ready-to-use default instance for convenience, matching how
# seasonal_naive_predict is used directly in run_backtest calls.
lightgbm_predict = make_lightgbm_predict_fn()
