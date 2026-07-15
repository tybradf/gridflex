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

# --- Weather (Open-Meteo, no key required) ---
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

HOURLY_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

# KNOWN SIMPLIFICATION (documented in README): one representative city per
# zone, not a true population-weighted centroid. Good enough to capture
# regional weather variation (Chicago winters vs. DC summers); a real
# population-weighted centroid is a fair enhancement to note as future work,
# not a Week 1 blocker.
ZONE_COORDS: dict[str, tuple[float, float]] = {
    "AE": (39.3643, -74.4229),    # Atlantic City, NJ
    "AEP": (39.9612, -82.9988),   # Columbus, OH
    "AP": (40.4406, -79.9959),    # Pittsburgh, PA
    "ATSI": (41.0814, -81.5190),  # Akron, OH
    "BC": (39.2904, -76.6122),    # Baltimore, MD
    "CE": (41.8781, -87.6298),    # Chicago, IL
    "DAY": (39.7589, -84.1916),   # Dayton, OH
    "DEOK": (39.1031, -84.5120),  # Cincinnati, OH
    "DOM": (37.5407, -77.4360),   # Richmond, VA
    "DPL": (39.7391, -75.5398),   # Wilmington, DE
    "DUQ": (40.4406, -79.9959),   # Pittsburgh, PA
    "EKPC": (38.0406, -84.5037),  # Lexington, KY
    "JC": (40.7357, -74.1724),    # Newark, NJ
    "ME": (40.3356, -75.9269),    # Reading, PA
    "PE": (39.9526, -75.1652),    # Philadelphia, PA
    "PEP": (38.9072, -77.0369),   # Washington, DC
    "PL": (40.6084, -75.4902),    # Allentown, PA
    "PN": (40.3267, -78.9220),    # Johnstown, PA
    "PS": (40.4862, -74.4518),    # New Brunswick, NJ
    "RECO": (41.1128, -74.1494),  # Suffern, NY
}

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

# Confirmed live via scripts/find_pjm_subba.py — PJM's 20 sub-BA zones.
# Note: PE = PECO, i.e. Philadelphia. Worth calling out in the README/demo.
PJM_SUBBA: list[str] = [
    "AE", "AEP", "AP", "ATSI", "BC", "CE", "DAY", "DEOK", "DOM", "DPL",
    "DUQ", "EKPC", "JC", "ME", "PE", "PEP", "PL", "PN", "PS", "RECO",
]

PJM_SUBBA_NAMES: dict[str, str] = {
    "AE": "Atlantic Electric",
    "AEP": "American Electric Power",
    "AP": "Allegheny Power",
    "ATSI": "American Transmission Systems, Inc.",
    "BC": "Baltimore Gas & Electric",
    "CE": "Commonwealth Edison",
    "DAY": "Dayton Power & Light",
    "DEOK": "Duke Energy Ohio/Kentucky",
    "DOM": "Dominion Virginia Power",
    "DPL": "Delmarva Power & Light",
    "DUQ": "Duquesne Lighting Company",
    "EKPC": "East Kentucky Power Cooperative",
    "JC": "Jersey Central Power & Light",
    "ME": "Metropolitan Edison",
    "PE": "PECO Energy",  # Philadelphia
    "PEP": "Potomac Electric Power",
    "PL": "Pennsylvania Power & Light",
    "PN": "Pennsylvania Electric",
    "PS": "Public Service Electric & Gas",
    "RECO": "Rockland Electric (East)",
}

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

# Plausible value ranges (MW), used for outlier detection/validation.
# PJM system peak is ~165,000 MW (summer 2025/2026); largest individual zones
# (AEP, CE) peak in the 25-35k range historically. Bounds below are
# deliberately generous — wide enough to never clip a real extreme event,
# tight enough to catch the kind of ~10^9 MW data-entry glitches found in
# subba_demand (DOM, Oct 2021) during Session 2. Source of truth: found via
# `SELECT * FROM subba_demand WHERE value > 100000` turning up exactly 3
# rows out of 1.3M — i.e. real outliers, not a systematic unit bug.
PLAUSIBLE_RANGES = {
    "pjm_demand": (0, 250_000),
    "pjm_forecast": (0, 250_000),
    "subba_demand": (0, 60_000),
    "fuel_mix": (0, 150_000),  # single fuel type, system-wide
}

# Approximate emission factors (kg CO2/MWh), commonly-cited EPA eGRID-derived
# national averages. NOT plant-specific or region-specific — a real refinement
# would use EPA eGRID's published subregion/plant-level rates (or CAMPD
# unit-level data, same source noted for the zone-level carbon limitation).
# Documented here as an approximation, not hidden — see README known limitations.
EMISSION_FACTORS_KG_PER_MWH: dict[str, float] = {
    "COL": 1000,   # Coal
    "NG": 410,     # Natural gas
    "OIL": 760,    # Petroleum
    "NUC": 0,      # Nuclear — no combustion emissions
    "WAT": 0,      # Hydro
    "WND": 0,      # Wind
    "SUN": 0,      # Solar
    "GEO": 40,     # Geothermal — small but nonzero
    "BAT": 0,      # Battery storage — pass-through, not a primary source
    "PS": 0,       # Pumped storage — same
    "OES": 0,      # Other energy storage
    "SNB": 0,      # Solar + battery — treat as solar-dominant, 0
    "WNB": 0,      # Wind + battery — same logic
    "UES": 0,      # Unknown storage
    "OTH": 0,      # Other — unknown composition, conservative 0 rather than guessing
    "UNK": 0,      # Unknown
}
