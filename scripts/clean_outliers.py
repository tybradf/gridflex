"""
One-time cleanup for the outliers found in Session 2 (DOM ~1.5e9 MW, CE
~142k MW) — and reusable going forward as a manual audit tool.

This does NOT run automatically as part of ingest; it's deliberately a
separate, explicit step. Automatic silent deletion of "implausible" data
during ingest is dangerous (a real, extreme event could get silently
dropped). The pandera schema in block 3.1 will WARN/FAIL loudly on future
outliers instead of deleting them — a human should decide whether an
outlier is a data error or a genuine rare event.

Run: python scripts/clean_outliers.py           # reports only, no changes
     python scripts/clean_outliers.py --delete  # actually removes the rows
"""

import typer

from gridflex.config import PLAUSIBLE_RANGES
from gridflex.store.db import get_connection

app = typer.Typer()


@app.command()
def run(delete: bool = typer.Option(False, help="Actually delete flagged rows.")) -> None:
    con = get_connection()

    total_flagged = 0
    for table, (lo, hi) in PLAUSIBLE_RANGES.items():
        rows = con.execute(f"""
            SELECT * FROM {table}
            WHERE value < {lo} OR value > {hi}
            ORDER BY value DESC
        """).fetchdf()

        if rows.empty:
            print(f"{table}: no outliers outside [{lo}, {hi}]")
            continue

        print(f"\n{table}: {len(rows)} row(s) outside plausible range [{lo}, {hi}]")
        print(rows.to_string())
        total_flagged += len(rows)

        if delete:
            n_before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            con.execute(f"DELETE FROM {table} WHERE value < {lo} OR value > {hi}")
            n_after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  -> deleted {n_before - n_after} row(s) from {table}")

    con.close()

    if total_flagged == 0:
        print("\nNo outliers found. Nothing to do.")
    elif not delete:
        print(f"\n{total_flagged} row(s) flagged across all tables. "
              f"Re-run with --delete to remove them.")


if __name__ == "__main__":
    app()
