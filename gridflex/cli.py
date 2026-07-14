"""
GridFlex CLI.

    python -m gridflex.cli ingest --start 2026-07-01 --end 2026-07-08

If --end is omitted, defaults to now (UTC).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import typer

from gridflex.ingest.eia import EIAClient
from gridflex.ingest.land import write_raw

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = typer.Typer(help="GridFlex data pipeline CLI.")


@app.command()
def status() -> None:
    """Quick sanity check: prints resolved config paths and PJM zone count.
    Also exists to keep this a genuine multi-command CLI — Typer silently
    collapses single-command apps into a no-subcommand-name calling
    convention, which is confusing once we add train/forecast/score later.
    """
    from gridflex.config import CURATED, DB_PATH, PJM_SUBBA, RAW

    typer.echo(f"RAW dir:      {RAW}")
    typer.echo(f"CURATED dir:  {CURATED}")
    typer.echo(f"DuckDB path:  {DB_PATH}")
    typer.echo(f"PJM zones:    {len(PJM_SUBBA)} configured")


@app.command()
def ingest(
    start: str = typer.Option(..., help="Start date, e.g. 2026-07-01 or 2026-07-01T00"),
    end: str = typer.Option(None, help="End date. Defaults to now (UTC) if omitted."),
) -> None:
    """Pull PJM demand, forecast, zone demand, and fuel mix for a date range,
    and land them as partitioned Parquet under data/raw/.
    """
    start_ts = _normalize(start)
    end_ts = _normalize(end) if end else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    log.info("Ingesting %s .. %s", start_ts, end_ts)

    with EIAClient() as eia:
        jobs = {
            "pjm_demand": lambda: eia.fetch_region("D", start_ts, end_ts),
            "pjm_forecast": lambda: eia.fetch_region("DF", start_ts, end_ts),
            "subba_demand": lambda: eia.fetch_subba_demand(start_ts, end_ts),
            "fuel_mix": lambda: eia.fetch_fuel_mix(start_ts, end_ts),
        }

        for dataset, fetch_fn in jobs.items():
            log.info("Fetching %s ...", dataset)
            df = fetch_fn()
            if df.empty:
                log.warning("  %s: 0 rows returned — skipping write", dataset)
                continue
            paths = write_raw(df, dataset)
            log.info("  %s: %d rows -> %s", dataset, len(df), paths)

    log.info("Done.")


def _normalize(date_str: str) -> str:
    """Accept plain dates (2026-07-01) or already-hourly timestamps
    (2026-07-01T00) and return the hourly format EIA expects."""
    return date_str if "T" in date_str else f"{date_str}T00"


if __name__ == "__main__":
    app()
