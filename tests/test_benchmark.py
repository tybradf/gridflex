import numpy as np
import pandas as pd

from gridflex.features.build import FEATURE_COLUMNS
from gridflex.models.benchmark import attach_pjm_forecast, pjm_forecast_predict
from gridflex.store.db import upsert


def test_pjm_forecast_mw_excluded_from_feature_columns():
    """CRITICAL: if this ever regresses, LightGBM could partially learn to
    copy PJM's own forecast, quietly invalidating the whole comparison."""
    assert "pjm_forecast_mw" not in FEATURE_COLUMNS


def test_attach_pjm_forecast_join_is_exact(tmp_db):
    from gridflex.store.db import get_connection

    con = get_connection()
    periods = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    upsert(con, "pjm_forecast", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * 5, "type": ["DF"] * 5,
        "value": [100.0, 200.0, 300.0, 400.0, 500.0],
    }))

    df = pd.DataFrame({"period": periods, "demand": [1.0] * 5})
    result = attach_pjm_forecast(df, con)
    con.close()

    assert list(result["pjm_forecast_mw"]) == [100.0, 200.0, 300.0, 400.0, 500.0]


def test_pjm_forecast_predict_raises_on_missing_values():
    """Fails loudly rather than silently scoring a comparison over gaps in
    PJM's own published forecast, which would be misleading, not just incomplete."""
    test_df = pd.DataFrame({"pjm_forecast_mw": [100.0, np.nan, 300.0]})
    try:
        pjm_forecast_predict(None, test_df)
        assert False, "should have raised on missing benchmark values"
    except ValueError:
        pass


def test_pjm_forecast_predict_returns_exact_values():
    test_df = pd.DataFrame({"pjm_forecast_mw": [100.0, 200.0, 300.0]})
    preds = pjm_forecast_predict(None, test_df)
    assert list(preds) == [100.0, 200.0, 300.0]
