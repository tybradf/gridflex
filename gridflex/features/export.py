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


def export_forecast_upcoming(con) -> list[dict]:
    """Forecast hours that haven't happened yet — i.e. not yet scoreable.
    Straight from the forecasts table, no join needed."""
    df = con.execute("""
        SELECT f.period, f.predicted_demand
        FROM forecasts f
        LEFT JOIN pjm_demand d ON f.period = d.period
        WHERE d.value IS NULL
        ORDER BY f.period
    """).fetchdf()
    return _df_to_records(df)


def export_scoreboard(con, days: int = 7) -> dict:
    """Block 3.6 — reuses live_scoreboard() (which itself reuses
    compute_metrics() from the backtest harness) so the exported number is
    provably the same yardstick used throughout Week 3, not a separate
    export-time calculation that could quietly drift from it."""
    from gridflex.models.live import live_scoreboard

    score = live_scoreboard(con, days=days)
    if score["n_scored"] == 0:
        return score

    rows = []
    for r in score["rows"]:
        row = dict(r)
        row["period"] = row["period"].strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(row)
    score["rows"] = rows
    return score


def export_flexibility_data(con) -> dict:
    """Data needed for the CLIENT-SIDE flexibility engine (block 4.3's UI).
    The site is static (no server — see README architecture), so the
    shift-finding logic runs in JavaScript against precomputed tables, not
    by calling Python live. Two tables, kept small:

    - segments: marginal emissions rate + 95% CI per (season, hour),
      already three-gate-filtered by estimate_marginal_emissions_by_segment
      — the JS engine trusts these are pre-validated, doesn't re-derive them.
    - zone_typical_demand: AVG(demand) per (zone, season, hour) — one
      efficient SQL query for all 20 zones x 4 seasons x 24 hours (1,920
      rows) rather than 1,920 separate Python calls to zone_typical_demand().
      zone_seasonal_peak is NOT separately exported — the JS engine derives
      it as max(typical_demand) across the 24 hours, mirroring exactly how
      Python's zone_seasonal_peak() does it, one source of truth either way.
    """
    from gridflex.models.marginal_emissions import compute_deltas, estimate_marginal_emissions_by_segment

    deltas = compute_deltas(con)
    segments = estimate_marginal_emissions_by_segment(deltas)
    segments_out = _df_to_records(segments) if not segments.empty else []

    zone_demand = con.execute("""
        SELECT
            subba AS zone,
            CASE
                WHEN EXTRACT(month FROM period) IN (12, 1, 2) THEN 'winter'
                WHEN EXTRACT(month FROM period) IN (3, 4, 5) THEN 'spring'
                WHEN EXTRACT(month FROM period) IN (6, 7, 8) THEN 'summer'
                ELSE 'fall'
            END AS season,
            EXTRACT(hour FROM period)::INT AS hour,
            AVG(value) AS typical_demand_mw
        FROM subba_demand
        GROUP BY zone, season, hour
        ORDER BY zone, season, hour
    """).fetchdf()

    return {
        "segments": segments_out,
        "zone_typical_demand": zone_demand.to_dict(orient="records"),
    }


def run(hours: int = 48) -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = get_connection()

    anchor = _shared_anchor(con)  # computed once, shared — see _shared_anchor docstring

    payload = {
        "zone_demand": export_zone_demand(con, hours=hours, anchor=anchor),
        "carbon_intensity": export_carbon(con, hours=hours, anchor=anchor),
        "zones": export_zone_metadata(),
        "forecast_upcoming": export_forecast_upcoming(con),
        "scoreboard": export_scoreboard(con, days=7),
        "flexibility": export_flexibility_data(con),
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    con.close()

    out_path = SITE_DATA_DIR / "latest.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} "
          f"({len(payload['zone_demand'])} demand rows, "
          f"{len(payload['carbon_intensity'])} carbon rows, "
          f"{len(payload['forecast_upcoming'])} upcoming forecast rows, "
          f"scoreboard n_scored={payload['scoreboard']['n_scored']}, "
          f"flexibility segments={len(payload['flexibility']['segments'])})")


if __name__ == "__main__":
    run()
