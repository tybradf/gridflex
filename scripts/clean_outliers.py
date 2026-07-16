"""
Retroactive data-quality cleanup — reuses validate_and_filter (the SAME
tested function already wired into ingest and backfill) as the single
source of truth, rather than re-implementing range/null checks separately.

Originally built for the ~1.5e9 MW outliers (Session 2). Extended to also
catch NULL values after finding 116 null pjm_demand rows (Week 3) that
predate validate_and_filter's existence — the historical backfill ran
BEFORE block 3.1 (validation) was built, so these slipped through. Every
future ingest is already protected; this is purely retroactive.

This does NOT run automatically as part of ingest; it's deliberately a
separate, explicit step — a human should see what's being removed before
it happens.

Run: python scripts/clean_outliers.py           # reports only, no changes
     python scripts/clean_outliers.py --delete  # actually removes the rows
"""

import typer

from gridflex.ingest.validate import validate_and_filter
from gridflex.store.db import SCHEMAS, get_connection

app = typer.Typer()

TABLES_WITH_VALUE = ["pjm_demand", "pjm_forecast", "subba_demand", "fuel_mix"]


@app.command()
def run(delete: bool = typer.Option(False, help="Actually delete flagged rows.")) -> None:
    con = get_connection()
    total_flagged = 0

    for table in TABLES_WITH_VALUE:
        df = con.execute(f"SELECT * FROM {table}").fetchdf()
        if df.empty:
            print(f"{table}: empty, skipping")
            continue

        cleaned = validate_and_filter(df.copy(), table)
        rejected = df.loc[df.index.difference(cleaned.index)]

        if rejected.empty:
            print(f"{table}: clean, no rows rejected by current validation")
            continue

        total_flagged += len(rejected)
        print(f"\n{table}: {len(rejected)} row(s) would be rejected by current validation")
        print(rejected.to_string())

        if delete:
            key_cols = SCHEMAS[table]
            n_before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            con.register("_rejected", rejected[key_cols])
            key_match = " AND ".join(f"{table}.{c} = _rejected.{c}" for c in key_cols)
            con.execute(f"""
                DELETE FROM {table}
                WHERE EXISTS (SELECT 1 FROM _rejected WHERE {key_match})
            """)
            con.unregister("_rejected")
            n_after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  -> deleted {n_before - n_after} row(s) from {table}")

    con.close()

    if total_flagged == 0:
        print("\nNo issues found. Nothing to do.")
    elif not delete:
        print(f"\n{total_flagged} row(s) flagged across all tables. "
              f"Re-run with --delete to remove them.")


if __name__ == "__main__":
    app()
