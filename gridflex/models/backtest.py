"""
Block 3.2 — backtest harness.

Design: everything here is generic over (a) the target series and (b) the
prediction function. That's deliberate — it lets the exact same harness and
metrics score:
  - baseline models (3.3)
  - the deep model (3.5)
  - PJM's own published forecast (3.4) — by wrapping "predict = the DF
    column" as a trivial model, so the incumbent gets scored through the
    IDENTICAL code path as our own models. That's what makes the head-to-head
    comparison fair rather than apples-to-oranges.

Splitting: EXPANDING walk-forward windows, strictly time-ordered. Train set
grows each fold; test set is always strictly after train — this is the
single easiest mistake to make in forecasting (shuffling time series data
leaks future information into training), so it's asserted, not just hoped for.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size_hours: int = 24,  # a true single-issuance day-ahead horizon —
    # see gridflex/features/build.py module docstring for why this matters:
    # a lag feature of length L is only leak-free when L >= test_size_hours.
    min_train_hours: int = 24 * 90,  # 90 days minimum before the first fold
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns a list of (train_idx, test_idx) index arrays into df, assumed
    already sorted by period ascending. Expanding window: fold i's train set
    is everything before fold i's test window (grows each fold); test
    windows are consecutive, non-overlapping blocks at the END of the data
    (most recent, realistic — not scattered through history).
    """
    n = len(df)
    if n < min_train_hours + n_splits * test_size_hours:
        raise ValueError(
            f"Not enough data: {n} rows, need at least "
            f"{min_train_hours + n_splits * test_size_hours} for "
            f"{n_splits} splits of {test_size_hours}h with {min_train_hours}h min train."
        )

    splits = []
    # Test windows are the LAST n_splits * test_size_hours rows, walked forward.
    first_test_start = n - n_splits * test_size_hours
    for i in range(n_splits):
        test_start = first_test_start + i * test_size_hours
        test_end = test_start + test_size_hours
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)

        assert train_idx.max() < test_idx.min(), "leakage: train overlaps test"
        splits.append((train_idx, test_idx))

    return splits


def calendar_fold_windows(
    reference_df: pd.DataFrame,
    n_splits: int = 5,
    test_size_hours: int = 24,
    min_train_hours: int = 24 * 90,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Derives fold boundaries as (train_end, test_start, test_end)
    TIMESTAMPS from a reference dataframe — not row positions. Critical for
    cross-zone aggregation (Week 4): walk_forward_splits is positional, so
    independently backtesting 20 zones with slightly different row counts
    (one zone missing a data-quality-flagged hour another doesn't) would
    silently misalign "fold 3" across zones to different real calendar
    hours. Deriving the windows ONCE here, then slicing every zone's data
    by these same timestamps (see slice_by_window), makes alignment exact
    regardless of any per-zone row-count differences.
    """
    reference_df = reference_df.sort_values("period").reset_index(drop=True)
    splits = walk_forward_splits(reference_df, n_splits, test_size_hours, min_train_hours)
    windows = []
    for train_idx, test_idx in splits:
        train_end = reference_df["period"].iloc[train_idx[-1]]
        test_start = reference_df["period"].iloc[test_idx[0]]
        test_end = reference_df["period"].iloc[test_idx[-1]]
        windows.append((train_end, test_start, test_end))
    return windows


def slice_by_window(
    df: pd.DataFrame, train_end: pd.Timestamp, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slices ANY dataframe (e.g. a single zone's table) by calendar
    timestamp against a fold window derived elsewhere — the counterpart to
    calendar_fold_windows. Train = everything up to and including
    train_end; test = the closed interval [test_start, test_end]."""
    train_df = df[df["period"] <= train_end]
    test_df = df[(df["period"] >= test_start) & (df["period"] <= test_end)]
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100),
        "n": int(len(y_true)),
    }


def run_backtest(
    df: pd.DataFrame,
    target_col: str,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    n_splits: int = 5,
    test_size_hours: int = 24,
    min_train_hours: int = 24 * 90,
) -> pd.DataFrame:
    """Runs walk-forward backtesting and returns one row per fold with
    metrics, plus the raw predictions/actuals for later inspection.

    predict_fn(train_df, test_df) -> np.ndarray of predictions, same length
    and order as test_df. This is the only thing that changes between a
    seasonal-naive baseline, LightGBM, a deep model, or the PJM benchmark.

    IMPORTANT: default test_size_hours=24 matches a true single-issuance
    day-ahead forecast (PJM's DF is issued once, ~24h ahead, with zero
    knowledge of anything that happens during the forecast horizon). Any
    lag feature must have length >= test_size_hours or it silently leaks
    within-horizon actuals — found the hard way via an implausibly good
    LightGBM result (see tests/test_features_build.py for the guard).
    """
    df = df.sort_values("period").reset_index(drop=True)
    splits = walk_forward_splits(df, n_splits, test_size_hours, min_train_hours)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        preds = predict_fn(train_df, test_df)
        metrics = compute_metrics(test_df[target_col].values, preds)
        rows.append({
            "fold": fold,
            "train_end": train_df["period"].max(),
            "test_start": test_df["period"].min(),
            "test_end": test_df["period"].max(),
            **metrics,
        })

    return pd.DataFrame(rows)
