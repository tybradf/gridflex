"""
Pagination logic tests. Per the block 1.3/3.2 design: mock at the
_get_page boundary rather than hitting the live EIA API in CI. _get_page is
assigned as a plain function on the instance (not the class), which bypasses
its @retry decorator entirely for these tests — that's intentional, we're
testing fetch()'s pagination LOOP here, not tenacity's retry behavior.
"""

from gridflex.ingest.eia import EIAClient


def _make_fake_pages(total: int, page_size: int):
    """Returns a fake _get_page(path, params) that serves `total` rows across
    pages of `page_size`, mimicking EIA's offset/total pagination contract."""

    def fake_get_page(path, params):
        offset = params["offset"]
        remaining = total - offset
        n = min(page_size, max(remaining, 0))
        rows = [
            {
                "period": f"2024-01-01T{(offset + i) % 24:02d}",
                "respondent": "PJM",
                "type": "D",
                "value": str(1000 + offset + i),
            }
            for i in range(n)
        ]
        return {"data": rows, "total": total}

    return fake_get_page


def test_pagination_collects_all_rows_across_multiple_pages():
    """15 total rows served 5 at a time across 3 pages — fetch() must collect
    all 15, not just the first page."""
    client = EIAClient(api_key="test-key")
    client._get_page = _make_fake_pages(total=15, page_size=5)

    df = client.fetch(
        "region",
        facets={"respondent": ["PJM"], "type": ["D"]},
        start="2024-01-01T00",
        end="2024-01-02T00",
    )

    assert len(df) == 15
    assert df["value"].is_monotonic_increasing  # 1000..1014, in request order
    client.close()


def test_pagination_handles_single_page_under_page_size():
    """3 total rows, page size 5 — should still work as a degenerate
    single-page case (this is the shape of most of our real 1-week pulls)."""
    client = EIAClient(api_key="test-key")
    client._get_page = _make_fake_pages(total=3, page_size=5)

    df = client.fetch(
        "region",
        facets={"respondent": ["PJM"], "type": ["D"]},
        start="2024-01-01T00",
        end="2024-01-02T00",
    )

    assert len(df) == 3
    client.close()


def test_empty_response_returns_empty_df():
    client = EIAClient(api_key="test-key")
    client._get_page = _make_fake_pages(total=0, page_size=5)

    df = client.fetch(
        "region",
        facets={"respondent": ["PJM"], "type": ["D"]},
        start="2024-01-01T00",
        end="2024-01-02T00",
    )

    assert df.empty
    client.close()


def test_period_parsed_as_utc_datetime():
    client = EIAClient(api_key="test-key")
    client._get_page = _make_fake_pages(total=2, page_size=5)

    df = client.fetch(
        "region",
        facets={"respondent": ["PJM"], "type": ["D"]},
        start="2024-01-01T00",
        end="2024-01-02T00",
    )

    assert str(df["period"].dt.tz) == "UTC"
    assert "datetime64" in str(df["period"].dtype)
