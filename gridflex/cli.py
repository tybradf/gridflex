"""
GridFlex CLI.

    python -m gridflex.cli ingest --start 2026-07-01 --end 2026-07-08

If --end is omitted, defaults to now (UTC).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
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
    start: str = typer.Option(
        None,
        help="Start date, e.g. 2026-07-01 or 2026-07-01T00. If omitted, resumes "
        "from each table's watermark (max stored period) minus a 72h overlap, "
        "to pick up EIA's revisions to recent data. Required for a fresh/empty DB.",
    ),
    end: str = typer.Option(None, help="End date. Defaults to now (UTC) if omitted."),
    overlap_hours: int = typer.Option(
        72, help="Hours to re-pull before the watermark, to catch EIA revisions."
    ),
) -> None:
    """Pull PJM demand, forecast, zone demand, and fuel mix; land raw Parquet
    under data/raw/, and upsert into DuckDB. Incremental by default: each
    dataset resumes from its own watermark unless --start is given explicitly.
    """
    from gridflex.store.db import get_connection, max_period, upsert

    end_ts = _normalize(end) if end else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    con = get_connection()

    with EIAClient() as eia:
        jobs = {
            "pjm_demand": lambda: eia.fetch_region("D", start_for("pjm_demand"), end_ts),
            "pjm_forecast": lambda: eia.fetch_region("DF", start_for("pjm_forecast"), end_ts),
            "subba_demand": lambda: eia.fetch_subba_demand(start_for("subba_demand"), end_ts),
            "fuel_mix": lambda: eia.fetch_fuel_mix(start_for("fuel_mix"), end_ts),
        }

        def start_for(table: str) -> str:
            if start:
                return _normalize(start)
            wm = max_period(con, table)
            if wm is None:
                raise typer.BadParameter(
                    f"'{table}' is empty and no --start given. "
                    "Provide --start for the initial backfill."
                )
            resume = wm - pd.Timedelta(hours=overlap_hours)
            return resume.strftime("%Y-%m-%dT%H")

        for dataset, fetch_fn in jobs.items():
            log.info("Fetching %s ...", dataset)
            df = fetch_fn()
            if df.empty:
                log.warning("  %s: 0 rows returned — skipping", dataset)
                continue
            raw_paths = write_raw(df, dataset)
            n = upsert(con, dataset, df)
            log.info("  %s: %d rows -> DuckDB, raw Parquet -> %s", dataset, n, raw_paths)

    con.close()
    log.info("Done.")


def _normalize(date_str: str) -> str:
    """Accept plain dates (2026-07-01) or already-hourly timestamps
    (2026-07-01T00) and return the hourly format EIA expects."""
    return date_str if "T" in date_str else f"{date_str}T00"


if __name__ == "__main__":
    app()
