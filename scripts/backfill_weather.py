"""
Session 2, block 2.5 — join weather to zone demand, sanity-check the join.

The check: temperature vs. demand should show the classic summer hockey-stick
(demand rises with temperature, driven by AC load) or a U-shape across a full
year (heating in winter, cooling in summer). If we don't see a positive
temp-demand relationship in July data, something is wrong with the join or
the timezone alignment — better to catch that now than after a model trains
quietly on garbage.

Run: python scripts/backfill_weather.py --start 2026-07-01 --end 2026-07-08
     python scripts/check_weather_join.py
"""

import typer

from gridflex.config import ZONE_COORDS
from gridflex.ingest.land import write_raw
from gridflex.ingest.weather import fetch_historical
from gridflex.store.db import get_connection, upsert

app = typer.Typer()


@app.command()
def run(start: str = typer.Option(...), end: str = typer.Option(...)) -> None:
    """Backfill historical weather for all PJM zones over [start, end]
    (YYYY-MM-DD dates) and upsert into the weather table."""
    print(f"Fetching weather for {len(ZONE_COORDS)} zones, {start}..{end} "
          f"(single multi-location request)...")
    df = fetch_historical(start, end)
    if df.empty:
        print("!!! 0 rows returned — check dates / Open-Meteo status")
        raise typer.Exit(1)

    print(f"{len(df)} rows returned. Columns: {list(df.columns)}")
    con = get_connection()
    write_raw(df, "weather")
    n = upsert(con, "weather", df)
    con.close()
    print(f"Upserted {n} rows into weather table.")


if __name__ == "__main__":
    app()
