"""Central configuration. Reads secrets from .env locally, from env vars in CI."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
DB_PATH = DATA / "gridflex.duckdb"

for _p in (RAW, CURATED):
    _p.mkdir(parents=True, exist_ok=True)

# --- EIA API ---
EIA_API_KEY = os.environ.get("EIA_API_KEY")
EIA_BASE = "https://api.eia.gov/v2"
EIA_PAGE_SIZE = 5000  # API hard max per request

# The three EIA-930 routes we need.
#   region-data      -> PJM-level demand (D), *PJM's own day-ahead forecast* (DF),
#                       net generation (NG), total interchange (TI). DF is our benchmark.
#   region-sub-ba-data -> hourly demand by PJM subregion (the ~20 zones). Our spatial layer.
#   fuel-type-data   -> hourly net generation by fuel type. BA-level only (see README limits).
ROUTES = {
    "region": "electricity/rto/region-data",
    "subba": "electricity/rto/region-sub-ba-data",
    "fuel": "electricity/rto/fuel-type-data",
}

PARENT_BA = "PJM"

# Populated in Session 1, block 1.2 from the live metadata/data endpoints.
# DO NOT hand-write these from memory — confirmed via scripts/find_pjm_subba.py
# and scripts/explore_metadata.py against the real API.

# TODO: paste output of `python scripts/find_pjm_subba.py` here.
PJM_SUBBA: list[str] = []

# Deduped by code from the fueltype facet (EIA's metadata has duplicate codes
# with inconsistent casing on their names, e.g. "Battery" vs "Battery storage" — 
# the codes themselves are the source of truth, not the display names).
FUEL_TYPES: list[str] = [
    "BAT",  # Battery storage
    "COL",  # Coal
    "GEO",  # Geothermal
    "NG",   # Natural gas
    "NUC",  # Nuclear
    "OES",  # Other energy storage
    "OIL",  # Petroleum
    "OTH",  # Other
    "PS",   # Pumped storage
    "SNB",  # Solar with integrated battery storage
    "SUN",  # Solar
    "UES",  # Unknown energy storage
    "UNK",  # Unknown
    "WAT",  # Hydro
    "WNB",  # Wind with integrated battery storage
    "WND",  # Wind
]

# Confirmed live: region-data 'type' facet = D (demand), TI (interchange),
# NG (net generation), DF (day-ahead demand forecast). DF is PJM's own
# published forecast — our production benchmark for Week 3.
SERIES_TYPES = {"demand": "D", "forecast": "DF", "net_gen": "NG", "interchange": "TI"}
