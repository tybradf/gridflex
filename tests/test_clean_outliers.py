import numpy as np
import pandas as pd

from gridflex.store.db import upsert


def _seed_dirty_data(con):
    periods = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    values = [100_000.0, np.nan, 105_000.0, 1.5e9, 102_000.0,
              103_000.0, np.nan, 104_000.0, 106_000.0, 107_000.0]
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * 10, "type": ["D"] * 10, "value": values,
    }))

    sub_periods = periods[:5]
    upsert(con, "subba_demand", pd.DataFrame({
        "period": list(sub_periods) * 2,
        "subba": ["PE"] * 5 + ["CE"] * 5,
        "parent": ["PJM"] * 10,
        "value": [5000.0, np.nan, 5100.0, 5200.0, 5300.0]
                 + [12000.0, 12100.0, 12200.0, 12300.0, 12400.0],
    }))
    return periods


def test_report_only_makes_zero_changes(tmp_db):
    from gridflex.store.db import get_connection
    from scripts.clean_outliers import run as clean_run

    con = get_connection()
    _seed_dirty_data(con)
    n_before = con.execute("SELECT COUNT(*) FROM pjm_demand").fetchone()[0]
    con.close()

    clean_run(delete=False)

    con = get_connection()
    n_after = con.execute("SELECT COUNT(*) FROM pjm_demand").fetchone()[0]
    con.close()
    assert n_before == n_after == 10


def test_delete_removes_exactly_the_bad_rows(tmp_db):
    from gridflex.store.db import get_connection
    from scripts.clean_outliers import run as clean_run

    con = get_connection()
    _seed_dirty_data(con)
    con.close()

    clean_run(delete=True)

    con = get_connection()
    pjm_count = con.execute("SELECT COUNT(*) FROM pjm_demand").fetchone()[0]
    subba_count = con.execute("SELECT COUNT(*) FROM subba_demand").fetchone()[0]
    bad_pjm = con.execute(
        "SELECT COUNT(*) FROM pjm_demand WHERE value IS NULL OR value > 250000"
    ).fetchone()[0]
    con.close()

    assert pjm_count == 7  # 10 - 2 nulls - 1 outlier
    assert subba_count == 9  # 10 - 1 null
    assert bad_pjm == 0


def test_composite_key_delete_targets_only_the_bad_zone(tmp_db):
    """The critical correctness check for a destructive operation: deleting
    a null PE row at a given hour must NOT also delete CE's valid row at
    that SAME hour — a naive delete-by-period-only would get this wrong."""
    from gridflex.store.db import get_connection
    from scripts.clean_outliers import run as clean_run

    con = get_connection()
    _seed_dirty_data(con)
    con.close()

    clean_run(delete=True)

    con = get_connection()
    remaining = con.execute(
        "SELECT subba, value FROM subba_demand WHERE period = '2024-01-01 01:00:00+00'"
    ).fetchdf()
    con.close()

    assert len(remaining) == 1
    assert remaining.iloc[0]["subba"] == "CE"
    assert not remaining["value"].isna().any()


def test_spike_pass_catches_contextual_outlier_missed_by_range_check(tmp_db):
    """Week 4: a spike that lands INSIDE PLAUSIBLE_RANGES (invisible to
    pass 1) must still be caught by pass 2, and composite-key deletion must
    correctly target only the spiking zone."""
    from gridflex.store.db import get_connection
    from scripts.clean_outliers import run as clean_run

    con = get_connection()
    periods = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    values = [79156, 77330, 73999, 215682, 68741, 67380, 67152, 67222]
    upsert(con, "pjm_demand", pd.DataFrame({
        "period": periods, "respondent": ["PJM"] * 8, "type": ["D"] * 8, "value": values,
    }))
    pe_values = [1000, 1000, 1000, 20000, 1000, 1000, 1000, 1000]
    ce_values = [2000] * 8
    upsert(con, "subba_demand", pd.DataFrame({
        "period": list(periods) * 2, "subba": ["PE"] * 8 + ["CE"] * 8,
        "parent": ["PJM"] * 16, "value": pe_values + ce_values,
    }))
    con.close()

    clean_run(delete=True)

    con = get_connection()
    pjm_count = con.execute("SELECT COUNT(*) FROM pjm_demand").fetchone()[0]
    remaining_zone = con.execute(
        "SELECT subba, value FROM subba_demand WHERE period = '2024-01-01 03:00:00+00'"
    ).fetchdf()
    con.close()

    assert pjm_count == 7  # 8 - 1 spike
    assert len(remaining_zone) == 1
    assert remaining_zone.iloc[0]["subba"] == "CE"
    assert remaining_zone.iloc[0]["value"] == 2000.0
