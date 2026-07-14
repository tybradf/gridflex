"""
EIA API v2 client for the three EIA-930 routes we need.

Confirmed against the live API (see scripts/explore_metadata.py and
scripts/find_pjm_subba.py output) — not guessed from docs or memory:
  - path pattern: {route}/data  (metadata lives at {route}, no suffix)
  - facet filters: facets[name][]=value  (repeatable for multiple values)
  - pagination: offset + length, length capped at 5000
  - region-data 'type' facet: D, DF, NG, TI  (DF = PJM's own forecast, our benchmark)
  - EIA's free-tier latency can be slow (~1 min observed for a single call) —
    timeouts and retries are generous on purpose, not a bug workaround.
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gridflex.config import EIA_API_KEY, EIA_BASE, EIA_PAGE_SIZE, PARENT_BA, ROUTES

log = logging.getLogger(__name__)

_RETRYABLE = (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError)


def _is_retryable_status(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        # Retry on rate-limit and server errors; don't retry on 4xx client errors
        # like a bad API key or malformed facet — those won't fix themselves.
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class EIAClient:
    def __init__(self, api_key: str | None = None, timeout: float = 90.0):
        self.api_key = api_key or EIA_API_KEY
        if not self.api_key:
            raise ValueError("EIA_API_KEY not set. Put it in .env as EIA_API_KEY=...")
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EIAClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _get_page(self, path: str, params: dict) -> dict:
        url = f"{EIA_BASE}/{path.lstrip('/')}/data"
        p = {**params, "api_key": self.api_key}
        r = self._client.get(url, params=p)
        if r.status_code >= 400 and not _is_retryable_status_from_response(r):
            # Fail fast and loud on client errors (bad facet, bad key, etc.)
            # instead of retrying something that can't succeed.
            log.error("EIA request failed (%s): %s", r.status_code, r.text[:500])
        r.raise_for_status()
        return r.json()["response"]

    def fetch(
        self,
        route: str,
        *,
        facets: dict[str, list[str]],
        start: str,
        end: str,
        frequency: str = "hourly",
        data_col: str = "value",
    ) -> pd.DataFrame:
        """Fetch all rows for a route/facet/date-range combination, paginating
        transparently. Returns a flat DataFrame — one row per (period, facet...).
        """
        path = ROUTES[route] if route in ROUTES else route
        params: dict = {
            "frequency": frequency,
            "data[0]": data_col,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": EIA_PAGE_SIZE,
        }
        for facet_name, values in facets.items():
            for i, v in enumerate(values):
                params[f"facets[{facet_name}][{i}]"] = v

        all_rows: list[dict] = []
        total: int | None = None

        while True:
            params["offset"] = len(all_rows)
            resp = self._get_page(path, params)
            rows = resp.get("data", [])
            all_rows.extend(rows)

            if total is None:
                total = int(resp.get("total", len(rows)))
                log.info("EIA %s: %s rows expected", route, total)

            if not rows or len(all_rows) >= total:
                break

        df = pd.DataFrame(all_rows)
        if df.empty:
            log.warning("EIA %s returned 0 rows for %s..%s facets=%s", route, start, end, facets)
            return df

        df["period"] = pd.to_datetime(df["period"], utc=(frequency == "hourly"))
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    # --- Convenience wrappers for our three routes ---

    def fetch_region(self, series_type: str, start: str, end: str) -> pd.DataFrame:
        """PJM-level series: D (demand), DF (forecast), NG (net gen), TI (interchange)."""
        return self.fetch(
            "region",
            facets={"respondent": [PARENT_BA], "type": [series_type]},
            start=start,
            end=end,
        )

    def fetch_subba_demand(self, start: str, end: str) -> pd.DataFrame:
        """Hourly demand for every PJM sub-BA zone in one call."""
        return self.fetch(
            "subba",
            facets={"parent": [PARENT_BA]},
            start=start,
            end=end,
        )

    def fetch_fuel_mix(self, start: str, end: str) -> pd.DataFrame:
        """PJM system-wide net generation by fuel type. NOTE: BA-level, not
        zone-level — see README known limitations."""
        return self.fetch(
            "fuel",
            facets={"respondent": [PARENT_BA]},
            start=start,
            end=end,
        )


def _is_retryable_status_from_response(r: httpx.Response) -> bool:
    return r.status_code == 429 or r.status_code >= 500
