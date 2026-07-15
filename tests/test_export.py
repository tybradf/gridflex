import pandas as pd

from gridflex.features import export
from gridflex.store.db import upsert


def test_export_anchors_demand_and_carbon_to_same_window(tmp_db):
    """Regression test for a real bug: subba_demand and fuel_mix can have
    different freshness (found in practice: fuel_mix ran ~23h ahead of
    subba_demand). Each export independently windowing to 'its own last N
    hours' created a false gap. Both must be anchored to a shared endpoint."""
    from gridflex.store.db import get_connection

    con = get_connection()

    demand_periods = pd.date_range("2026-07-05T00:00", "2026-07-14T04:00", freq="h", tz="UTC")
    fuel_periods = pd.date_range("2026-07-05T00:00", "2026-07-15T03:00", freq="h", tz="UTC")

    subba_df = pd.DataFrame({
        "period": list(demand_periods),
        "subba": ["PE"] * len(demand_periods),
        "parent": ["PJM"] * len(demand_periods),
        "value": [5000.0] * len(demand_periods),
    })
    fuel_df = pd.DataFrame({
        "period": fuel_periods,
        "respondent": ["PJM"] * len(fuel_periods),
        "fueltype": ["NG"] * len(fuel_periods),
        "value": [10000.0] * len(fuel_periods),
    })
    upsert(con, "subba_demand", subba_df)
    upsert(con, "fuel_mix", fuel_df)

    anchor = export._shared_anchor(con)
    demand_records = export.export_zone_demand(con, hours=48, anchor=anchor)
    carbon_records = export.export_carbon(con, hours=48, anchor=anchor)
    con.close()

    demand_periods_out = {r["period"] for r in demand_records}
    carbon_periods_out = {r["period"] for r in carbon_records}

    assert demand_periods_out - carbon_periods_out == set(), (
        "demand periods missing from carbon — the anchor-alignment bug regressed"
    )
