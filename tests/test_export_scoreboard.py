import json

import pandas as pd

from gridflex.config import PJM_SUBBA
from gridflex.features import export
from gridflex.models.live import store_forecast
from gridflex.store.db import upsert


def test_scoreboard_and_upcoming_forecast_split_correctly(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()

    periods = pd.date_range("2026-07-13T00:00", periods=72, freq="h", tz="UTC")
    rows = [{"period": p, "subba": z, "parent": "PJM", "value": 5000.0}
            for p in periods for z in PJM_SUBBA]
    upsert(con, "subba_demand", pd.DataFrame(rows))
    upsert(con, "fuel_mix", pd.DataFrame(
        [{"period": p, "respondent": "PJM", "fueltype": "NG", "value": 50000.0} for p in periods]
    ))

    past = periods[:48]  # actuals exist -> scoreable
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": past, "respondent": ["PJM"] * 48, "type": ["D"] * 48, "value": [100_000.0] * 48,
    }))
    upsert(con, "pjm_forecast", pd.DataFrame({
        "period": past, "respondent": ["PJM"] * 48, "type": ["DF"] * 48, "value": [102_000.0] * 48,
    }))

    fc = pd.DataFrame({"period": periods, "predicted_demand": [99_000.0] * 72,
                        "generated_at": pd.Timestamp.now(tz="UTC")})
    store_forecast(con, fc)
    con.close()

    export.run(hours=48)
    payload = json.loads((export.SITE_DATA_DIR / "latest.json").read_text())
    (export.SITE_DATA_DIR / "latest.json").unlink()  # clean up test artifact

    assert payload["scoreboard"]["n_scored"] == 48
    assert len(payload["forecast_upcoming"]) == 24
    assert payload["scoreboard"]["ours"]["mae"] == 1000.0
    # every value must be JSON-serializable-clean (already proven by json.loads
    # succeeding above, but double check no raw Timestamp objects leaked through)
    assert all(isinstance(r["period"], str) for r in payload["scoreboard"]["rows"])
    assert all(isinstance(r["period"], str) for r in payload["forecast_upcoming"])


def test_export_flexibility_data_shapes_correctly(tmp_db):
    from gridflex.store.db import get_connection
    con = get_connection()

    n = 24 * 730  # 2 years -- enough per-bucket coverage to clear min_n
    periods = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    import numpy as np
    np.random.seed(0)
    demand = 100_000 + 20_000 * np.sin(np.arange(n) / 24 * 2 * np.pi) + np.random.normal(0, 500, n)
    ng = demand - 50_000

    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * n, "type": ["D"] * n, "value": demand,
    }))
    upsert(con, "fuel_mix", pd.DataFrame(
        [{"period": p, "respondent": "PJM", "fueltype": "NG", "value": ng[i]} for i, p in enumerate(periods)]
    ))
    upsert(con, "subba_demand", pd.DataFrame({
        "period": list(periods) * 2, "subba": ["PE"] * n + ["CE"] * n,
        "parent": ["PJM"] * n * 2, "value": [1000.0] * n + [2000.0] * n,
    }))

    result = export.export_flexibility_data(con)
    con.close()

    assert len(result["segments"]) > 0
    assert len(result["zone_typical_demand"]) == 2 * 4 * 24  # 2 zones x 4 seasons x 24 hours
    # must be genuinely JSON-serializable, not just Python-dict-shaped
    json.dumps(result)
