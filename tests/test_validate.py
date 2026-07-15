import pandas as pd

from gridflex.ingest.validate import validate_and_filter


def test_drops_real_session2_outlier(sample_subba_df):
    """The exact regression case: DOM at 1.5e9 MW must be dropped, the two
    clean PE rows must survive."""
    clean = validate_and_filter(sample_subba_df, "subba_demand")
    assert len(clean) == 2
    assert set(clean["subba"]) == {"PE"}
    assert (clean["value"] < 60_000).all()


def test_passes_all_clean_data_unchanged():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-07-01T00:00", "2024-07-01T01:00"], utc=True),
        "subba": ["PE", "PE"],
        "parent": ["PJM", "PJM"],
        "value": [5200.0, 5100.0],
    })
    clean = validate_and_filter(df, "subba_demand")
    assert len(clean) == 2
    pd.testing.assert_frame_equal(
        clean.reset_index(drop=True), df.reset_index(drop=True)
    )


def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=["period", "subba", "parent", "value"])
    result = validate_and_filter(df, "subba_demand")
    assert result.empty


def test_unknown_table_falls_back_to_unbounded_range():
    """A table not in PLAUSIBLE_RANGES shouldn't crash — it should just skip
    the range check (only period-nullness is enforced)."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-07-01T00:00"], utc=True),
        "value": [999_999_999.0],
    })
    result = validate_and_filter(df, "some_new_table_not_yet_configured")
    assert len(result) == 1
