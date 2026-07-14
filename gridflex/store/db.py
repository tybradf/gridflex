"""
DuckDB store: schema + idempotent upsert.

Design choice: one table per dataset, with a composite primary key that
matches each dataset's natural grain. Upsert via INSERT ... ON CONFLICT DO
UPDATE, so re-running ingest for an overlapping window is always safe —
this is what lets us re-pull the last 72h every run (EIA revises recent
data) without creating duplicates. This is block 2.1; the watermark logic
that decides *what range* to re-pull is block 2.2, deliberately separate.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from gridflex.config import DB_PATH

# (table_name, primary_key_columns) — grain of each dataset.
SCHEMAS: dict[str, list[str]] = {
    "pjm_demand": ["period"],  # PJM system-level, one series (D)
    "pjm_forecast": ["period"],  # PJM system-level, one series (DF)
    "subba_demand": ["period", "subba"],  # zone-level
    "fuel_mix": ["period", "fueltype"],  # fuel-level
    "weather": ["period", "subba"],  # zone-level, joined 1:1 with subba_demand
}

_DDL = {
    "pjm_demand": """
        CREATE TABLE IF NOT EXISTS pjm_demand (
            period TIMESTAMPTZ NOT NULL,
            respondent VARCHAR,
            type VARCHAR,
            value DOUBLE,
            PRIMARY KEY (period)
        )
    """,
    "pjm_forecast": """
        CREATE TABLE IF NOT EXISTS pjm_forecast (
            period TIMESTAMPTZ NOT NULL,
            respondent VARCHAR,
            type VARCHAR,
            value DOUBLE,
            PRIMARY KEY (period)
        )
    """,
    "subba_demand": """
        CREATE TABLE IF NOT EXISTS subba_demand (
            period TIMESTAMPTZ NOT NULL,
            subba VARCHAR NOT NULL,
            parent VARCHAR,
            value DOUBLE,
            PRIMARY KEY (period, subba)
        )
    """,
    "fuel_mix": """
        CREATE TABLE IF NOT EXISTS fuel_mix (
            period TIMESTAMPTZ NOT NULL,
            respondent VARCHAR,
            fueltype VARCHAR NOT NULL,
            value DOUBLE,
            PRIMARY KEY (period, fueltype)
        )
    """,
    "weather": """
        CREATE TABLE IF NOT EXISTS weather (
            period TIMESTAMPTZ NOT NULL,
            subba VARCHAR NOT NULL,
            temperature_2m DOUBLE,
            relative_humidity_2m DOUBLE,
            wind_speed_10m DOUBLE,
            shortwave_radiation DOUBLE,
            PRIMARY KEY (period, subba)
        )
    """,
}


# Explicit column lists per table (matches the DDL). The EIA API returns
# extra descriptive columns we don't store — respondent-name, type-name,
# value-units — which also happen to contain hyphens and break as unquoted
# SQL identifiers. We select down to exactly these columns before upserting,
# rather than trusting df.columns.
TABLE_COLUMNS: dict[str, list[str]] = {
    "pjm_demand": ["period", "respondent", "type", "value"],
    "pjm_forecast": ["period", "respondent", "type", "value"],
    "subba_demand": ["period", "subba", "parent", "value"],
    "fuel_mix": ["period", "respondent", "fueltype", "value"],
    "weather": [
        "period", "subba", "temperature_2m",
        "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation",
    ],
}


def get_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    for ddl in _DDL.values():
        con.execute(ddl)
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Idempotent insert: re-running with overlapping rows updates rather than
    duplicates. Returns the number of rows in df (not rows actually changed —
    DuckDB doesn't cheaply expose that for ON CONFLICT)."""
    if df.empty:
        return 0
    if table not in SCHEMAS:
        raise ValueError(f"Unknown table {table!r}. Known: {list(SCHEMAS)}")

    key_cols = SCHEMAS[table]
    known_cols = TABLE_COLUMNS[table]

    missing_keys = [k for k in key_cols if k not in df.columns]
    if missing_keys:
        raise ValueError(f"{table}: incoming data missing key column(s) {missing_keys}")

    # Drop anything not in our schema (e.g. EIA's 'respondent-name',
    # 'type-name', 'value-units'); keep only columns we actually store,
    # and only those present in this particular df.
    cols = [c for c in known_cols if c in df.columns]
    df = df[cols]

    update_cols = [c for c in cols if c not in key_cols]

    con.register("_incoming", df)

    set_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
    conflict_cols = ", ".join(key_cols)
    col_list = ", ".join(cols)

    con.execute(f"""
        INSERT INTO {table} ({col_list})
        SELECT {col_list} FROM _incoming
        ON CONFLICT ({conflict_cols}) DO UPDATE SET {set_clause}
    """)
    con.unregister("_incoming")
    return len(df)


def row_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in SCHEMAS}


def max_period(con: duckdb.DuckDBPyConnection, table: str) -> pd.Timestamp | None:
    """Latest period currently stored — the watermark for incremental pulls.
    Used by block 2.2, defined here since it's a store-level query."""
    result = con.execute(f"SELECT MAX(period) FROM {table}").fetchone()[0]
    return pd.Timestamp(result, tz="UTC") if result is not None else None
