# GridFlex

**Forecasting PJM grid load and carbon intensity, and quantifying what a
megawatt of flexible demand is worth — by zone, by hour.**

Electricity must be generated the instant it's consumed, so the mix of
plants running at any moment sets both the price and the carbon intensity of
a kilowatt-hour — and that varies enormously by hour and by location. A
small but fast-growing slice of demand is **flexible**: EV charging, data
center compute, batteries, and industrial processes can often run now or
three hours from now without anyone noticing.

This project forecasts *when and where* the PJM grid (the 20-zone RTO
covering ~65M people, from Chicago to DC to Philadelphia) is dirtiest and
most strained, then quantifies what a megawatt of flexible demand is worth
if you move it — the central question facing a grid where data centers are
projected to drive peak demand up ~58% by 2046.

**Status:** Week 1 complete — automated ingestion pipeline is live and
running daily. Forecasting model, live dashboard, and the flexibility-value
engine are Weeks 2-4 (see [Roadmap](#roadmap) below).

## Live sanity check

![Temperature vs. zone demand, all 20 PJM zones and Philadelphia (PECO) highlighted](data/weather_join_sanity_check.png)

The classic V-shape: demand is lowest around 15-20°C (nothing running),
rises at both extremes (heating in the cold, AC in the heat). Confirms the
weather join is physically sound across ~1.3M zone-hours, 2019-2026.

## Data sources

| Source | What it provides | Access |
|---|---|---|
| [EIA API v2](https://www.eia.gov/opendata/) (EIA-930) | Hourly demand, PJM's own day-ahead forecast, generation by fuel type, by zone and system-wide, 2019-present | Free API key |
| [Open-Meteo](https://open-meteo.com) | Historical + forecast weather (temp, humidity, wind, solar radiation) | No key required |

## Architecture

No server, no paid hosting — a scheduled GitHub Action does the compute:

```
EIA API + Open-Meteo  ->  ingest client (paginated, retrying, validated)
                       ->  DuckDB (idempotent upsert, watermark-based incremental pulls)
                       ->  [persisted as a GitHub Release asset, not committed to git]
```

- **`gridflex/ingest/`** - EIA client, weather client, pandera-based validation
- **`gridflex/store/`** - DuckDB schema + idempotent upsert
- **`gridflex/cli.py`** - `gridflex ingest` (incremental, resumes from watermark), `gridflex status`
- **`.github/workflows/ingest.yml`** - daily cron; downloads DB from Release, updates it, re-uploads
- **`.github/workflows/test.yml`** - runs the test suite on every push/PR

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # add your free EIA API key: https://www.eia.gov/opendata/
python scripts/explore_metadata.py   # confirm live API access
python -m gridflex.cli status
```

## Testing

```bash
python -m pytest tests/ -v
```

14 tests covering pagination logic, idempotent upsert, watermark resume, and
schema validation - including direct regression tests for two real bugs
found during development (see [Known limitations](#known-limitations)).

## Automated daily updates (GitHub Actions)

The DuckDB file is persisted as a **GitHub Release asset**, not committed to
git - a daily-growing binary in git history would bloat the repo forever.
The `ingest.yml` workflow downloads it, updates it incrementally, re-uploads
it. `test.yml` runs the test suite on every push/PR - a separate concern.

**One-time setup, required before the cron will work:**

1. Add your EIA key as a repo secret: Settings -> Secrets and variables ->
   Actions -> New repository secret -> name it `EIA_API_KEY`.
2. Seed the initial Release with your already-backfilled local DB (the cron
   can only do *incremental* pulls - it can't do the full historical
   backfill itself without risking timeouts/rate limits):
   ```bash
   gh release create data-latest data/gridflex.duckdb \
     --title "GridFlex data snapshot" \
     --notes "Auto-updated by .github/workflows/ingest.yml"
   ```
   (Needs the `gh` CLI installed and authenticated - `brew install gh && gh auth login`
   on macOS - or do the equivalent via the GitHub web UI: Releases -> Draft a
   new release -> tag `data-latest` -> attach `data/gridflex.duckdb`.)
3. Confirm it worked: Actions tab -> Ingest -> Run workflow (manual trigger),
   watch the log.

After that, it runs daily on its own (06:17 UTC) with no further action.

## Known limitations

Named explicitly rather than hidden - these are real simplifications, not
bugs, and each has a clear path to fixing later:

- **Fuel mix is system-wide, not zone-level.** EIA-930 only reports
  generation-by-fuel at the balancing-authority level (all of PJM), not per
  zone. True zone-level carbon intensity needs EPA CAMPD hourly unit-level
  emissions (free, has plant coordinates) - a planned Week 4 enhancement.
- **Weather uses one representative city per zone** (e.g. Philadelphia for
  PE/PECO), not a true population-weighted centroid. Captures real regional
  variation (Chicago winters vs. DC summers) but is a known simplification.
- **EIA revises recent data** after initial publication. The ingest pipeline
  re-pulls and upserts the last 72 hours on every run to catch these
  revisions rather than assuming the first pull is final.
- **Raw EIA-930 data contains occasional data-entry errors** - several rows
  were found during development at up to ~2 *billion* MW (physically
  impossible; PJM's actual system peak is ~165,000 MW). These are now caught
  automatically by a pandera schema check before they ever reach the
  database (`gridflex/ingest/validate.py`), rather than requiring manual
  discovery.
- **~0.18% of historical demand data contained null values**, weakly
  correlated with DST transitions (4 of 6 clusters land exactly on real US
  DST dates, 2 don't - cause not fully confirmed, likely an intermittent
  EIA/PJM telemetry issue). These predated the pandera validation above and
  were cleaned retroactively (`scripts/clean_outliers.py`, which reuses the
  same validation logic rather than a separate implementation); all data
  ingested going forward is protected automatically.
- **The demand and fuel-mix data streams publish on independent, sometimes
  conflicting cadences** - which one is "behind" the other flips day to day
  (observed fuel-mix running ~23h ahead of demand on one occasion, and
  demand running ahead of fuel-mix on another). The live dashboard currently
  anchors both to whichever stream is older, to avoid a false gap where one
  series' data exists but the export window excludes it (see
  `gridflex/features/export.py`'s `_shared_anchor`). This keeps the two
  panels aligned but means the whole dashboard is only as fresh as its
  slowest input. A cleaner fix - each panel showing its own independent
  "as of" freshness rather than a forced shared anchor - is deferred to a
  future design pass.
- **The backtest evaluates against actual historical weather, not a
  forecast.** A real day-ahead deployment only has weather *forecasts*,
  with their own error - PJM's real `DF` had to contend with that; our
  backtest used hindsight-perfect weather. This is standard practice for
  evaluating model architecture, but it's a real advantage our backtest had
  that a live system wouldn't. Live inference (`gridflex/models/live.py`)
  correctly uses forecast weather, not archive weather, closing this gap
  for actual production use even though the backtest doesn't reflect it.

## Results (Week 3)

System-level demand forecasting, benchmarked against PJM's own published
day-ahead forecast (`DF`) via 5-fold walk-forward backtesting (no data
leakage - verified via a full audit, see commit history):

| Model | MAE (MW) | MAPE |
|---|---|---|
| Seasonal naive (demand 168h ago) | ~12,368 | 10.52% |
| LightGBM (calendar + weather + lags) | ~4,777 | 3.98% |
| **PJM's own day-ahead forecast** | ~4,093 | **3.67%** |

Our model narrowly trails PJM's own production forecasting system - a
mature RTO's system built over years, using data (confirmed outage
schedules, intraday weather nowcasts) we don't have access to. Getting
within ~8% relative of that on public data alone, built in days, is a
credible result on its own.

The differentiator: **PJM publishes no equivalent zone-level forecast at
all.** This project's zone-level demand forecasting (Week 4) fills a real
gap in what's publicly available, not just a comparison against an
incumbent.

## Roadmap

- **Week 2:** live public dashboard (PJM zone map, carbon intensity, GitHub Pages) ✅
- **Week 3:** forecasting models, benchmarked live against PJM's own published forecast ✅
- **Week 4:** marginal-emissions estimation + the flexible-demand value engine
