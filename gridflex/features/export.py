"""
Block 2.2 — export site data as JSON.

No server exists (see README architecture) — GitHub Pages serves static
files, so "live" means this script runs on the same cron as ingest and
overwrites these JSON files, which the frontend just fetches directly.

Kept deliberately small: the dashboard needs the last ~48h, not full
history, so payload size stays tiny (KBs, not MBs) regardless of how much
data has accumulated in DuckDB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridflex.carbon.average import carbon_intensity_by_hour
from gridflex.config import PJM_SUBBA_NAMES, ROOT
from gridflex.store.db import get_connection

SITE_DATA_DIR = ROOT / "site" / "data"


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: timestamps as ISO strings, NaN as null."""
    out = df.copy()
    for col in out.select_dtypes(include=["datetime64[ns, UTC]", "datetimetz"]).columns:
        out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.loads(out.to_json(orient="records"))


def _shared_anchor(con) -> pd.Timestamp | None:
    """The common end timestamp both exports must respect.

    BUG FOUND: fuel_mix and subba_demand can have different freshness (in
    practice, fuel_mix ran ~23h AHEAD of subba_demand at one point). If each
    export independently takes "my own last N hours," their windows land on
    different calendar ranges and create a false gap — periods that exist in
    one series but fall entirely outside the other's window, even though the
    underlying data isn't actually missing for that range. Anchoring both to
    the OLDER of the two max periods keeps them on the same calendar window.
    """
    from gridflex.store.db import max_period

    demand_max = max_period(con, "subba_demand")
    fuel_max = max_period(con, "fuel_mix")
    if demand_max is None or fuel_max is None:
        return demand_max or fuel_max
    return min(demand_max, fuel_max)


def export_zone_demand(con, hours: int = 48, anchor: pd.Timestamp | None = None) -> list[dict]:
    if anchor is None:
        anchor = _shared_anchor(con)
    df = con.execute("""
        SELECT period, subba, value AS demand_mw
        FROM subba_demand
        WHERE period <= ? AND period > ? - INTERVAL (?) HOUR
        ORDER BY period
    """, [anchor, anchor, hours]).fetchdf()
    return _df_to_records(df)


def export_carbon(con, hours: int = 48, anchor: pd.Timestamp | None = None) -> list[dict]:
    if anchor is None:
        anchor = _shared_anchor(con)
    full = carbon_intensity_by_hour(con)
    if full.empty:
        return []
    mask = (full["period"] <= anchor) & (full["period"] > anchor - pd.Timedelta(hours=hours))
    return _df_to_records(full.loc[mask])


def export_zone_metadata() -> list[dict]:
    """Static zone list — name, code — the frontend needs to label markers.
    Rarely changes; exported alongside the time-series data for simplicity
    rather than a separate hand-maintained frontend file."""
    from gridflex.config import ZONE_COORDS

    return [
        {"code": code, "name": PJM_SUBBA_NAMES[code], "lat": lat, "lon": lon}
        for code, (lat, lon) in ZONE_COORDS.items()
    ]


def run(hours: int = 48) -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = get_connection()

    anchor = _shared_anchor(con)  # computed once, shared — see _shared_anchor docstring

    payload = {
        "zone_demand": export_zone_demand(con, hours=hours, anchor=anchor),
        "carbon_intensity": export_carbon(con, hours=hours, anchor=anchor),
        "zones": export_zone_metadata(),
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    con.close()

    out_path = SITE_DATA_DIR / "latest.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} "
          f"({len(payload['zone_demand'])} demand rows, "
          f"{len(payload['carbon_intensity'])} carbon rows)")


if __name__ == "__main__":
    run()
