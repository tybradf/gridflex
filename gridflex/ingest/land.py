"""
Land raw EIA responses as partitioned Parquet under data/raw/{dataset}/{year}/.

This is deliberately dumb and append-only for now — Session 1 just proves data
can be pulled and land on disk in a sane layout. Tuesday's watermark/upsert
logic (block 2.2) is what makes re-running safe; don't build that here yet.
"""

from __future__ import annotations

import pandas as pd

from gridflex.config import RAW


def write_raw(df: pd.DataFrame, dataset: str) -> list[str]:
    """Partition a DataFrame by the year of its 'period' column and write one
    Parquet file per (dataset, year). Returns the list of paths written.

    Partitioning by year (not month/day) is deliberately coarse for now — with
    ~8 years of hourly data across 20 zones, year-partitioning keeps file count
    sane while still letting later code load a single year without scanning
    everything. We can repartition finer later if a year-file gets unwieldy.
    """
    if df.empty:
        return []

    out_dir = RAW / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for year, chunk in df.groupby(df["period"].dt.year):
        path = out_dir / f"{year}.parquet"
        # Append-safe for Session 1: if the file exists, concat + de-dupe on
        # (period + facet columns), then rewrite. This is O(file size) per
        # call, which is fine for now and gets replaced by DuckDB upserts
        # tomorrow — don't over-engineer this step today.
        if path.exists():
            existing = pd.read_parquet(path)
            key_cols = [c for c in chunk.columns if c != "value"]
            chunk = (
                pd.concat([existing, chunk], ignore_index=True)
                .drop_duplicates(subset=key_cols, keep="last")
                .sort_values("period")
            )
        chunk.to_parquet(path, index=False)
        written.append(str(path))

    return written
