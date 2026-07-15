import numpy as np
import pandas as pd

from gridflex.models.backtest import compute_metrics, run_backtest, walk_forward_splits


def _synthetic_df(n=24 * 90 + 5 * 48 + 10):
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "demand": np.arange(n, dtype=float),
    })


def test_no_leakage_across_all_folds():
    """The single most important property of the harness: test data must
    never leak into train. Asserted explicitly, not just hoped for."""
    df = _synthetic_df()
    splits = walk_forward_splits(df, n_splits=5, test_size_hours=48, min_train_hours=24 * 90)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        assert len(test_idx) == 48


def test_train_window_expands_each_fold():
    df = _synthetic_df()
    splits = walk_forward_splits(df, n_splits=5, test_size_hours=48, min_train_hours=24 * 90)
    train_sizes = [len(tr) for tr, _ in splits]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[1] - train_sizes[0] == 48


def test_insufficient_data_raises_clear_error():
    df = _synthetic_df(n=100)  # far too little for the default min_train_hours
    try:
        walk_forward_splits(df, n_splits=5, test_size_hours=48, min_train_hours=24 * 90)
        assert False, "should have raised"
    except ValueError:
        pass


def test_compute_metrics_matches_hand_calculation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    m = compute_metrics(y_true, y_pred)
    assert abs(m["mae"] - (10 + 10 + 30) / 3) < 1e-9
    assert abs(m["rmse"] - np.sqrt((10**2 + 10**2 + 30**2) / 3)) < 1e-9
    assert abs(m["mape"] - np.mean([10/100, 10/200, 30/300]) * 100) < 1e-9


def test_run_backtest_is_model_agnostic():
    """The harness must work identically for ANY predict_fn — this is what
    lets baselines, the deep model, and PJM's own DF benchmark all be
    scored through the same code path in Week 3.4."""
    df = _synthetic_df()

    def naive_mean_model(train_df, test_df):
        return np.full(len(test_df), train_df["demand"].mean())

    results = run_backtest(df, target_col="demand", predict_fn=naive_mean_model,
                            n_splits=5, test_size_hours=48, min_train_hours=24 * 90)
    assert len(results) == 5
    assert (results["n"] == 48).all()
    assert {"mae", "rmse", "mape"}.issubset(results.columns)
