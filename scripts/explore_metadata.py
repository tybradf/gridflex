"""
Session 1, block 1.2 — Metadata reconnaissance.

Hits each EIA-930 route's metadata endpoint and enumerates the ACTUAL facets and
facet values. Do not hand-write facet codes from memory or from a blog post; the
API is the only source of truth, and a silently-wrong filter costs an hour later.

Run:  python scripts/explore_metadata.py
Then: paste the output back so we can populate config.PJM_SUBBA / config.FUEL_TYPES.
"""

import json
import sys

import httpx

from gridflex.config import EIA_API_KEY, EIA_BASE, PARENT_BA, ROUTES


def get(path: str, **params) -> dict:
    """GET an EIA v2 path and return the 'response' block."""
    params["api_key"] = EIA_API_KEY
    url = f"{EIA_BASE}/{path.lstrip('/')}"
    r = httpx.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("response", {})


def describe_route(name: str, path: str) -> None:
    print("\n" + "=" * 70)
    print(f"ROUTE: {name}  ->  {path}")
    print("=" * 70)

    meta = get(path)

    # What data columns does this route expose? (e.g. 'value' with units)
    print("\n-- frequencies --")
    for f in meta.get("frequency", []):
        print(f"   {f.get('id'):12} {f.get('description', '')}")

    print("\n-- data columns --")
    for col, info in (meta.get("data") or {}).items():
        print(f"   {col:16} units={info.get('units')}  {info.get('alias', '')}")

    print(f"\n-- date range: {meta.get('startPeriod')} .. {meta.get('endPeriod')} --")

    # Facets are the filterable dimensions. Enumerate each one's values.
    facets = [f["id"] for f in meta.get("facets", [])]
    print(f"\n-- facets: {facets} --")

    for facet_id in facets:
        try:
            fmeta = get(f"{path}/facet/{facet_id}")
        except httpx.HTTPStatusError as e:
            print(f"\n   [{facet_id}] could not fetch: {e}")
            continue

        values = fmeta.get("facets", [])
        print(f"\n   [{facet_id}] {len(values)} values")

        # For subba, only PJM's children matter. The response includes a parent
        # field on sub-BA routes — filter to it so we get PJM's ~20 zones, not
        # every sub-BA in the country.
        if facet_id == "subba":
            pjm = [v for v in values if v.get("parent") == PARENT_BA]
            print(f"       PJM children ({len(pjm)}):")
            for v in pjm:
                print(f"         {v.get('id'):8} {v.get('name', '')}")
            print("\n       >>> PJM_SUBBA = " + json.dumps(sorted(v["id"] for v in pjm)))

        elif facet_id == "fueltype":
            for v in values:
                print(f"         {v.get('id'):8} {v.get('name', '')}")
            print("\n       >>> FUEL_TYPES = " + json.dumps(sorted(v["id"] for v in values)))

        elif facet_id == "type":
            # On region-data this is the series type: D, DF, NG, TI, etc.
            for v in values:
                print(f"         {v.get('id'):8} {v.get('name', '')}")

        else:
            # respondent, timezone, etc. — just show a sample so we don't spam.
            for v in values[:8]:
                print(f"         {v.get('id'):8} {v.get('name', '')}")
            if len(values) > 8:
                print(f"         ... and {len(values) - 8} more")


def main() -> None:
    if not EIA_API_KEY:
        sys.exit("EIA_API_KEY not set. Put it in .env as EIA_API_KEY=...")

    for name, path in ROUTES.items():
        describe_route(name, path)

    print("Done")


if __name__ == "__main__":
    main()
