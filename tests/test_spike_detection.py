import numpy as np
import pandas as pd

from gridflex.ingest.validate import detect_spike_rows


def test_detects_real_replicated_spike():
    """Exact replication of the real 2020-04-10 pjm_demand find: a single
    row spiking to 215,682 MW between two ~70,000-75,000 MW readings —
    well inside PLAUSIBLE_RANGES, invisible to that check entirely."""
    periods = pd.date_range("2020-04-10T01:00", periods=8, freq="h", tz="UTC")
    values = [79156, 77330, 73999, 215682, 68741, 67380, 67152, 67222]
    df = pd.DataFrame({"period": periods, "value": values})
    flags = detect_spike_rows(df)
    assert flags.tolist() == [False, False, False, True, False, False, False, False]


def test_does_not_flag_genuine_sustained_ramp():
    periods = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    values = [70000, 90000, 110000, 130000, 150000, 160000]  # steady climb
    df = pd.DataFrame({"period": periods, "value": values})
    assert not detect_spike_rows(df, threshold=15000).any()


def test_does_not_flag_delta_spanning_a_real_gap():
    periods = pd.to_datetime(
        ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T05:00", "2024-01-01T06:00"], utc=True
    )
    values = [100000, 100500, 200000, 100200]  # big jump, but 4h gap, not 1h
    df = pd.DataFrame({"period": periods, "value": values})
    assert not detect_spike_rows(df).any()


def test_no_false_positives_on_smooth_realistic_curve():
    np.random.seed(0)
    hours = np.arange(200)
    smooth = 100_000 + 20_000 * np.sin(hours / 24 * 2 * np.pi) + np.random.normal(0, 500, 200)
    periods = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    df = pd.DataFrame({"period": periods, "value": smooth})
    assert detect_spike_rows(df).sum() == 0
