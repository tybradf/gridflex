import pandas as pd
import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the STORE module's DB_PATH at a throwaway file for this test only.

    Important: db.py does `from gridflex.config import DB_PATH`, which binds
    the name into gridflex.store.db's own namespace at import time. Patching
    gridflex.config.DB_PATH would NOT affect that already-bound name — we
    have to patch gridflex.store.db.DB_PATH directly, since that's the name
    get_connection() actually looks up at call time.
    """
    import gridflex.store.db as db

    db_path = tmp_path / "test.duckdb"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    return db_path


@pytest.fixture
def sample_subba_df():
    """Shaped like real subba_demand data: two clean PE rows plus the exact
    class of outlier found manually in Session 2 (DOM, ~1.5e9 MW)."""
    return pd.DataFrame({
        "period": pd.to_datetime(
            ["2024-07-01T00:00", "2024-07-01T01:00", "2024-07-01T00:00"], utc=True
        ),
        "subba": ["PE", "PE", "DOM"],
        "parent": ["PJM", "PJM", "PJM"],
        "value": [5200.0, 5100.0, 1_526_640_000.0],
    })
