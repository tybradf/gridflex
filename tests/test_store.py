import pandas as pd

from gridflex.store.db import get_connection, max_period, row_counts, upsert


def test_schema_creates_clean(tmp_db):
    con = get_connection()
    counts = row_counts(con)
    assert counts == {
        "pjm_demand": 0, "pjm_forecast": 0,
        "subba_demand": 0, "fuel_mix": 0, "weather": 0, "forecasts": 0,
    }
    con.close()


def test_upsert_is_idempotent(tmp_db, sample_subba_df):
    """Re-running the same upsert must not duplicate rows — this is the core
    guarantee that makes the 72h overlap re-pull safe."""
    clean = sample_subba_df[sample_subba_df["subba"] == "PE"]  # 2 clean rows

    con = get_connection()
    n1 = upsert(con, "subba_demand", clean)
    n2 = upsert(con, "subba_demand", clean)  # exact same data again
    total = con.execute("SELECT COUNT(*) FROM subba_demand").fetchone()[0]
    con.close()

    assert n1 == 2
    assert n2 == 2  # upsert() reports rows processed, not rows changed
    assert total == 2  # but the TABLE must still only have 2 rows


def test_upsert_updates_on_conflict(tmp_db):
    """A revised value for the same (period, subba) key should overwrite,
    not duplicate — this is what makes EIA's revisions-to-recent-data safe."""
    con = get_connection()
    df1 = pd.DataFrame({
        "period": pd.to_datetime(["2024-07-01T00:00"], utc=True),
        "subba": ["PE"], "parent": ["PJM"], "value": [5000.0],
    })
    df2 = pd.DataFrame({
        "period": pd.to_datetime(["2024-07-01T00:00"], utc=True),
        "subba": ["PE"], "parent": ["PJM"], "value": [5555.0],  # revised value
    })
    upsert(con, "subba_demand", df1)
    upsert(con, "subba_demand", df2)
    result = con.execute("SELECT value FROM subba_demand").fetchdf()
    con.close()

    assert len(result) == 1
    assert result["value"].iloc[0] == 5555.0


def test_upsert_drops_extra_hyphenated_columns(tmp_db):
    """Regression test for the real Session 2 bug: EIA's raw response
    includes columns like 'respondent-name' and 'value-units' that aren't in
    our schema and break as unquoted SQL identifiers if not filtered out."""
    con = get_connection()
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-07-01T00:00"], utc=True),
        "respondent": ["PJM"],
        "respondent-name": ["PJM Interconnection"],  # should be dropped
        "type": ["D"],
        "type-name": ["Demand"],                      # should be dropped
        "value": [100000.0],
        "value-units": ["megawatthours"],              # should be dropped
    })
    n = upsert(con, "pjm_demand", df)  # must not raise
    stored = con.execute("SELECT * FROM pjm_demand").fetchdf()
    con.close()

    assert n == 1
    assert set(stored.columns) == {"period", "respondent", "type", "value"}


def test_max_period_watermark(tmp_db, sample_subba_df):
    clean = sample_subba_df[sample_subba_df["subba"] == "PE"]
    con = get_connection()
    upsert(con, "subba_demand", clean)
    wm = max_period(con, "subba_demand")
    con.close()

    assert wm == pd.Timestamp("2024-07-01T01:00", tz="UTC")


def test_max_period_empty_table_returns_none(tmp_db):
    con = get_connection()
    wm = max_period(con, "subba_demand")
    con.close()
    assert wm is None
