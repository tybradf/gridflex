# GridFlex

Forecasting PJM grid load and carbon intensity, and valuing flexible demand by zone and hour.

**Status:** Week 1 — Data Foundation.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # add your free EIA API key: https://www.eia.gov/opendata/
python scripts/explore_metadata.py
```
