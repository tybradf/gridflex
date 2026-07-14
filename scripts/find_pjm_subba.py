"""Discover PJM's actual subba codes by querying the DATA endpoint (not metadata),
filtered to parent=PJM, over a short recent window. This is the reliable way —
the facet metadata endpoint doesn't expose parent-child relationships."""

import httpx
from gridflex.config import EIA_API_KEY, EIA_BASE, PARENT_BA

url = f"{EIA_BASE}/electricity/rto/region-sub-ba-data/data"
params = {
    "api_key": EIA_API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[parent][]": PARENT_BA,
    "start": "2026-07-01T00",
    "end": "2026-07-02T00",
    "length": 5000,
}
r = httpx.get(url, params=params, timeout=30)
r.raise_for_status()
rows = r.json()["response"]["data"]

seen = {}
for row in rows:
    seen[row["subba"]] = row.get("subba-name", "")

print(f"{len(seen)} PJM subba zones found:\n")
for code, name in sorted(seen.items()):
    print(f"  {code:8} {name}")
print("\n>>> PJM_SUBBA =", sorted(seen.keys()))
