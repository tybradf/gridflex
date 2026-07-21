"""
Session 2, block 2.3 — full historical backfill.

EIA-930 data starts 2019-01-01 for these routes (confirmed in block 1.2's
metadata output). We chunk by month rather than pulling ~8 years in one call:
EIA_PAGE_SIZE is 5000 rows/page, and one month of hourly data across 20 zones
is already ~14,600 rows (3 pages) — a single multi-year pull would mean
hundreds of sequential paginated requests in one HTTP session, and given the
latency you're already seeing, a failure 40 minutes in would lose everything.
Monthly chunks mean a crash only costs you re-running one month.

Run: python scripts/backfill.py [--from-year 2019]
"""

from __future__ import annotations

import logging

import pandas as pd
import typer

from gridflex.ingest.eia import EIAClient
from gridflex.ingest.land import write_raw
from gridflex.ingest.validate import validate_and_filter
from gridflex.store.db import get_connection, upsert

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = typer.Typer()


def month_ranges(start_year: int) -> list[tuple[str, str]]:
    """(start, end) hourly-format strings for each month from start_year through
    the current month, inclusive."""
    start = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    months = pd.date_range(start, now, freq="MS", tz="UTC")
    ranges = []
    for m in months:
        start = m.strftime("%Y-%m-%dT%H")
        end = (m + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%dT%H")
        ranges.append((start, end))
    return ranges


@app.command()
def run(from_year: int = 2019) -> None:
    con = get_connection()
    ranges = month_ranges(from_year)
    log.info("Backfilling %d months from %d", len(ranges), from_year)

    fetchers = {}

    with EIAClient() as eia:
        fetchers = {
            "pjm_demand": lambda s, e: eia.fetch_region("D", s, e),
            "pjm_forecast": lambda s, e: eia.fetch_region("DF", s, e),
            "subba_demand": lambda s, e: eia.fetch_subba_demand(s, e),
            "fuel_mix": lambda s, e: eia.fetch_fuel_mix(s, e),
        }

        for i, (start, end) in enumerate(ranges, 1):
            log.info("[%d/%d] %s .. %s", i, len(ranges), start, end)
            for dataset, fn in fetchers.items():
                try:
                    df = fn(start, end)
                except Exception:
                    log.exception(
                        "  %s FAILED for %s..%s — skipping this month, "
                        "rerun `backfill --from-year` later to patch gaps",
                        dataset, start, end,
                    )
                    continue
                if df.empty:
                    log.warning("  %s: 0 rows for %s..%s", dataset, start, end)
                    continue
                df = validate_and_filter(df, dataset)
                if df.empty:
                    log.warning("  %s: 0 rows survived validation for %s..%s", dataset, start, end)
                    continue
                write_raw(df, dataset)
                n = upsert(con, dataset, df)
                log.info("  %s: %d rows", dataset, n)

    con.close()
    log.info("Backfill complete.")


if __name__ == "__main__":
    app()
