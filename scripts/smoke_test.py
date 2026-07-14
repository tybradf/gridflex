"""
Session 1, block 1.4 — pull real PJM data, verify shapes and units before
building anything else on top of this client.

Run: python scripts/smoke_test.py
"""

import logging

from gridflex.config import PJM_SUBBA
from gridflex.ingest.eia import EIAClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

START, END = "2026-07-01T00", "2026-07-08T00"  # one week


def main() -> None:
    with EIAClient() as eia:
        print("\n--- PJM system demand (D) ---")
        d = eia.fetch_region("D", START, END)
        print(d[["period", "respondent", "type", "value"]].head())
        print(f"rows={len(d)}  nulls={d['value'].isna().sum()}  "
              f"range=({d['value'].min():,.0f}, {d['value'].max():,.0f}) MWh")

        print("\n--- PJM's own day-ahead forecast (DF) — our benchmark ---")
        df_ = eia.fetch_region("DF", START, END)
        print(f"rows={len(df_)}  nulls={df_['value'].isna().sum() if not df_.empty else 'N/A'}")
        if df_.empty:
            print("  !!! DF returned nothing for this window — investigate before Week 3.")

        print("\n--- PJM sub-BA (zone) demand ---")
        sub = eia.fetch_subba_demand(START, END)
        found_zones = sorted(sub["subba"].unique())
        print(f"rows={len(sub)}  zones found={len(found_zones)}")
        missing = set(PJM_SUBBA) - set(found_zones)
        if missing:
            print(f"  !!! zones in config but missing from data: {missing}")
        print(sub[["period", "subba", "value"]].head())

        print("\n--- PJM fuel mix (system-wide) ---")
        fuel = eia.fetch_fuel_mix(START, END)
        print(f"rows={len(fuel)}  fuel types found={sorted(fuel['fueltype'].unique())}")
        print(fuel[["period", "fueltype", "value"]].head())

        # Sanity check: does system demand roughly equal the sum of zone demand
        # at a single hour? (Won't match exactly — losses, timing — but should
        # be in the same ballpark. This is the kind of check that catches a
        # units or join bug before it silently poisons everything downstream.)
        one_hour = sub["period"].iloc[0]
        zone_sum = sub.loc[sub["period"] == one_hour, "value"].sum()
        sys_val = d.loc[d["period"] == one_hour, "value"]
        if not sys_val.empty:
            print(f"\n--- Sanity check at {one_hour} ---")
            print(f"  sum of zone demand:   {zone_sum:,.0f} MWh")
            print(f"  system demand (D):    {sys_val.iloc[0]:,.0f} MWh")
            print(f"  ratio: {zone_sum / sys_val.iloc[0]:.2f} (expect close to 1.0)")


if __name__ == "__main__":
    main()
