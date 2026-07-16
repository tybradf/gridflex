"""
Week 4, block 4.2 — zone-level forecasting, aggregated and benchmarked
against PJM's system-wide DF.

Trains an independent model per zone per fold, sums the per-zone
predictions into one system-level aggregate per hour, and scores that
aggregate against BOTH true system actuals and PJM's own DF — reusing
compute_metrics unchanged, directly comparable to Week 3's headline table.

Uses calendar_fold_windows/slice_by_window (NOT walk_forward_splits run
independently per zone) specifically to avoid silently misaligning zones
with different row counts — see gridflex/models/backtest.py and the tests
proving this holds even when a zone has a gap inside a test window.

Ground truth for scoring is TRUE system actuals (pjm_demand), not the sum
of zone actuals — even though those are nearly identical (~0.08% gap from
transmission losses, confirmed in Week 1). PJM's DF is also scored against
the same true system actuals, so both sides of the comparison share
identical ground truth.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import duckdb
import numpy as np
import pandas as pd

from gridflex.config import PJM_SUBBA
from gridflex.features.build import build_training_table, build_zone_training_table
from gridflex.models.backtest import calendar_fold_windows, compute_metrics, slice_by_window

log = logging.getLogger(__name__)


def _safe_backtest_anchor(con: duckdb.DuckDBPyConnection, zones: list[str]) -> pd.Timestamp:
    """The latest period safe to include in a zone-aggregate backtest fold —
    must not exceed the freshness of subba_demand for ANY zone, pjm_demand
    (system actual), or pjm_forecast (DF). Otherwise fold windows can
    extend into a range some required series doesn't cover yet.

    Found via a real crash: pjm_demand was fresher than subba_demand at
    ingest time, so folds derived purely from pjm_demand extended past what
    zone-level data covered — EVERY zone showed 0 test rows for the final
    fold (not a zone-specific data-quality issue; a frontier mismatch, same
    class of bug as the fuel_mix/subba_demand cadence gap in the README,
    just between a different pair of streams).
    """
    from gridflex.store.db import max_period

    candidates = [max_period(con, "pjm_demand"), max_period(con, "pjm_forecast")]
    for z in zones:
        zmax = con.execute(
            "SELECT MAX(period) FROM subba_demand WHERE subba = ?", [z]
        ).fetchone()[0]
        if zmax is not None:
            zmax = pd.Timestamp(zmax)
            zmax = zmax.tz_localize("UTC") if zmax.tzinfo is None else zmax.tz_convert("UTC")
            candidates.append(zmax)

    valid = [c for c in candidates if c is not None]
    if not valid:
        raise ValueError("No data available in any required table — cannot backtest.")
    return min(valid)


def run_zone_aggregate_backtest(
    con: duckdb.DuckDBPyConnection,
    zones: list[str] = PJM_SUBBA,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray] | None = None,
    n_splits: int = 5,
    test_size_hours: int = 24,
    min_train_hours: int = 24 * 90,
) -> pd.DataFrame:
    if predict_fn is None:
        from gridflex.models.baselines import make_lightgbm_predict_fn
        predict_fn = make_lightgbm_predict_fn()

    # Reference used to derive calendar fold windows — trimmed to the safe
    # frontier across ALL required sources (see _safe_backtest_anchor),
    # not just pjm_demand, which can be misleadingly fresher than the
    # zone-level data folds actually depend on.
    anchor = _safe_backtest_anchor(con, zones)
    reference_df = build_training_table(con)
    reference_df = reference_df[reference_df["period"] <= anchor].reset_index(drop=True)
    windows = calendar_fold_windows(reference_df, n_splits, test_size_hours, min_train_hours)

    zone_tables = {z: build_zone_training_table(con, z) for z in zones}

    system_actual = con.execute(
        "SELECT period, value AS demand FROM pjm_demand ORDER BY period"
    ).fetchdf()
    pjm_forecast = con.execute(
        "SELECT period, value AS pjm_forecast FROM pjm_forecast ORDER BY period"
    ).fetchdf()

    rows = []
    for fold, (train_end, test_start, test_end) in enumerate(windows):
        agg_pred, agg_actual, coverage = None, None, {}

        for zone, zdf in zone_tables.items():
            train_df, test_df = slice_by_window(zdf, train_end, test_start, test_end)
            coverage[zone] = len(test_df)
            if test_df.empty:
                continue
            preds = predict_fn(train_df, test_df)
            pred_s = pd.Series(preds, index=test_df["period"])
            actual_s = pd.Series(test_df["demand"].values, index=test_df["period"])
            agg_pred = pred_s if agg_pred is None else agg_pred.add(pred_s, fill_value=0)
            agg_actual = actual_s if agg_actual is None else agg_actual.add(actual_s, fill_value=0)

        incomplete = {z: n for z, n in coverage.items() if n < test_size_hours}
        if incomplete:
            log.warning(
                "fold %d: incomplete zone coverage %s — aggregate prediction "
                "for the affected hours is understated (missing zones "
                "contribute 0, not their true value), which penalizes OUR "
                "score, not PJM's. Not a bug in the comparison, but a real "
                "data-quality caveat for that fold.", fold, incomplete,
            )
        if agg_pred is None or agg_pred.empty:
            raise ValueError(
                f"fold {fold}: EVERY zone returned 0 test rows for window "
                f"{test_start}..{test_end}. This should be prevented by "
                f"_safe_backtest_anchor — if you're seeing this, check "
                f"whether zones= includes a zone with no data at all."
            )

        window_mask = (system_actual["period"] >= test_start) & (system_actual["period"] <= test_end)
        true_system = system_actual.loc[window_mask].set_index("period")["demand"]
        aligned_pred = agg_pred.reindex(true_system.index) if agg_pred is not None else pd.Series(dtype=float)

        pjm_mask = (pjm_forecast["period"] >= test_start) & (pjm_forecast["period"] <= test_end)
        true_pjm = pjm_forecast.loc[pjm_mask].set_index("period")["pjm_forecast"]
        aligned_pjm = true_pjm.reindex(true_system.index)

        ours = compute_metrics(true_system.values, aligned_pred.values)
        pjm = compute_metrics(true_system.values, aligned_pjm.values)

        rows.append({
            "fold": fold, "train_end": train_end, "test_start": test_start, "test_end": test_end,
            "zone_coverage_ok": not bool(incomplete),
            "ours_mae": ours["mae"], "ours_mape": ours["mape"],
            "pjm_mae": pjm["mae"], "pjm_mape": pjm["mape"],
        })

    return pd.DataFrame(rows)
