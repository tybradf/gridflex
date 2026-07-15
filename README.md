# GridFlex

Forecasting PJM grid load and carbon intensity, and valuing flexible demand by zone and hour.

**Status:** week 1 — data foundation.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # add your free EIA API key: https://www.eia.gov/opendata/
python scripts/explore_metadata.py
```

## Automated daily updates (GitHub Actions)

The DuckDB file is persisted as a **GitHub Release asset**, not committed to
git — a daily-growing binary in git history would bloat the repo forever.
The `ingest.yml` workflow downloads it, updates it incrementally, re-uploads
it. `test.yml` runs the test suite on every push/PR — a separate concern.

**One-time setup, required before the cron will work:**

1. Add your EIA key as a repo secret: Settings → Secrets and variables →
   Actions → New repository secret → name it `EIA_API_KEY`.
2. Seed the initial Release with your already-backfilled local DB (the cron
   can only do *incremental* pulls — it can't do the full historical
   backfill itself without risking timeouts/rate limits):
   ```bash
   gh release create data-latest data/gridflex.duckdb \
     --title "GridFlex data snapshot" \
     --notes "Auto-updated by .github/workflows/ingest.yml"
   ```
   (Needs the `gh` CLI installed and authenticated — `brew install gh && gh auth login`
   on macOS — or do the equivalent via the GitHub web UI: Releases → Draft a
   new release → tag `data-latest` → attach `data/gridflex.duckdb`.)
3. Confirm it worked: Actions tab → Ingest → Run workflow (manual trigger),
   watch the log.

After that, it runs daily on its own (06:17 UTC) with no further action.
