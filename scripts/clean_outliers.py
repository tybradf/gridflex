"""
Retroactive data-quality cleanup — two passes.

Pass 1 (original, Session 2/3): reuses validate_and_filter (the SAME tested
function already wired into ingest and backfill) — absolute range + null
checks — single source of truth, not a separate implementation.

Pass 2 (Week 4): contextual spike detection (detect_spike_rows) — catches
single-hour spikes that land INSIDE the plausible range but are obviously
wrong relative to their immediate neighbors (found via a real example:
215,682 MW sandwiched between two ~70,000 MW readings — well inside
pjm_demand's [0, 250000] range, invisible to pass 1 entirely). Grouped by
zone (subba_demand) or fuel type (fuel_mix) where relevant — a spike check
must never compare one zone's demand to another's.

Neither pass runs automatically as part of ingest; this is deliberately a
separate, explicit step — a human should see what's being removed before
it happens. Pass 2 in particular is NOT yet wired into ongoing ingest
protection (see gridflex/ingest/validate.py's module docstring) — it needs
neighboring rows that a small incremental ingest batch may not have full
context for. This is a known, documented gap, not a solved problem.

Run: python scripts/clean_outliers.py           # reports only, no changes
     python scripts/clean_outliers.py --delete  # actually removes the rows
"""

import pandas as pd
import typer

from gridflex.ingest.validate import detect_spike_rows, validate_and_filter
from gridflex.store.db import SCHEMAS, get_connection

app = typer.Typer()

TABLES_WITH_VALUE = ["pjm_demand", "pjm_forecast", "subba_demand", "fuel_mix"]

# Grouping column for spike detection — a spike check must only compare a
# series to ITS OWN neighbors (one zone's demand, one fuel type's
# generation), never across zones/fuel types.
SPIKE_GROUP_COL = {
    "pjm_demand": None,
    "pjm_forecast": None,
    "subba_demand": "subba",
    "fuel_mix": "fueltype",
}
# Lower thresholds for zone/fuel-level series — smaller natural magnitude
# than system-wide totals. Physically-reasoned starting points, not
# statistically derived — worth revisiting if results look off.
SPIKE_THRESHOLDS = {
    "pjm_demand": 30_000,
    "pjm_forecast": 30_000,
    "subba_demand": 15_000,
    "fuel_mix": 10_000,
}


def _find_spikes(con, table: str) -> pd.DataFrame:
    group_col = SPIKE_GROUP_COL[table]
    threshold = SPIKE_THRESHOLDS[table]

    if group_col:
        df = con.execute(f"SELECT * FROM {table} ORDER BY {group_col}, period").fetchdf()
        if df.empty:
            return df
        parts = []
        for _, g in df.groupby(group_col):
            g = g.reset_index(drop=True)
            flags = detect_spike_rows(g, threshold=threshold)
            if flags.any():
                parts.append(g[flags])
        return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    else:
        df = con.execute(f"SELECT * FROM {table} ORDER BY period").fetchdf()
        if df.empty:
            return df
        flags = detect_spike_rows(df, threshold=threshold)
        return df[flags]


def _delete_by_key(con, table: str, rows: pd.DataFrame) -> int:
    key_cols = SCHEMAS[table]
    n_before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.register("_bad_rows", rows[key_cols])
    key_match = " AND ".join(f"{table}.{c} = _bad_rows.{c}" for c in key_cols)
    con.execute(f"DELETE FROM {table} WHERE EXISTS (SELECT 1 FROM _bad_rows WHERE {key_match})")
    con.unregister("_bad_rows")
    n_after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return n_before - n_after


@app.command()
def run(delete: bool = typer.Option(False, help="Actually delete flagged rows.")) -> None:
    con = get_connection()
    total_flagged = 0

    print("=== Pass 1: range + null validation ===")
    for table in TABLES_WITH_VALUE:
        df = con.execute(f"SELECT * FROM {table}").fetchdf()
        if df.empty:
            print(f"{table}: empty, skipping")
            continue

        cleaned = validate_and_filter(df.copy(), table)
        rejected = df.loc[df.index.difference(cleaned.index)]

        if rejected.empty:
            print(f"{table}: clean")
            continue

        total_flagged += len(rejected)
        print(f"\n{table}: {len(rejected)} row(s) rejected")
        print(rejected.to_string())

        if delete:
            n_deleted = _delete_by_key(con, table, rejected)
            print(f"  -> deleted {n_deleted} row(s) from {table}")

    print("\n=== Pass 2: contextual spike detection ===")
    for table in TABLES_WITH_VALUE:
        spikes = _find_spikes(con, table)
        if spikes.empty:
            print(f"{table}: no contextual spikes found")
            continue

        total_flagged += len(spikes)
        print(f"\n{table}: {len(spikes)} contextual spike(s) found")
        print(spikes.to_string())

        if delete:
            n_deleted = _delete_by_key(con, table, spikes)
            print(f"  -> deleted {n_deleted} row(s) from {table}")

    con.close()

    if total_flagged == 0:
        print("\nNo issues found. Nothing to do.")
    elif not delete:
        print(f"\n{total_flagged} row(s) flagged across all tables/passes. "
              f"Re-run with --delete to remove them.")


if __name__ == "__main__":
    app()
