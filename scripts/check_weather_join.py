"""
Session 2, block 2.5 — the actual sanity check on the weather join.

Run AFTER scripts/backfill_weather.py has loaded some weather data.
Run: python scripts/check_weather_join.py
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; we're saving to file, not popping a window
import matplotlib.pyplot as plt

from gridflex.store.db import get_connection


def main() -> None:
    con = get_connection()

    joined = con.execute("""
        SELECT
            d.period,
            d.subba,
            d.value AS demand_mw,
            w.temperature_2m
        FROM subba_demand d
        JOIN weather w
          ON d.period = w.period AND d.subba = w.subba
    """).fetchdf()

    con.close()

    if joined.empty:
        print("!!! Join returned 0 rows. Likely causes:")
        print("    - weather table is empty (run scripts/backfill_weather.py first)")
        print("    - date ranges of subba_demand and weather don't overlap")
        print("    - timezone mismatch between the two tables")
        return

    print(f"{len(joined)} joined rows across {joined['subba'].nunique()} zones")

    corr = joined["temperature_2m"].corr(joined["demand_mw"])
    print(f"\nOverall temp-vs-demand correlation: {corr:+.3f}")
    print("(Expect positive in July — AC load rises with heat. A strongly")
    print(" negative or near-zero value here means investigate before Week 3.)")

    # Per-zone, since PECO (PE) is Philadelphia — worth calling out specifically.
    print("\nPer-zone correlation:")
    per_zone = joined.groupby("subba").apply(
        lambda g: g["temperature_2m"].corr(g["demand_mw"]), include_groups=False
    )
    print(per_zone.sort_values(ascending=False).to_string())

    # Save the scatter for a visual eyeball, focused on PE (Philadelphia) plus
    # overall — this is the plot the Week 1 plan calls for.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(joined["temperature_2m"], joined["demand_mw"], s=4, alpha=0.3)
    axes[0].set_title("All PJM zones")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Zone demand (MW)")

    pe = joined[joined["subba"] == "PE"]
    axes[1].scatter(pe["temperature_2m"], pe["demand_mw"], s=6, alpha=0.4, color="darkorange")
    axes[1].set_title("PE (Philadelphia / PECO)")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Zone demand (MW)")

    fig.tight_layout()
    out_path = "data/weather_join_sanity_check.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved scatter plot to {out_path} — eyeball it for the hockey-stick shape.")


if __name__ == "__main__":
    main()
