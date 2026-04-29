from datetime import date

from octts.tools.compare_top3_score_fields import (
    _build_day_cache_path,
    _load_day_cache,
    _forward_return,
    _looks_unfillable_limit_up,
    _pick_tradeable_codes,
    _write_day_cache,
)


class _FakeRepo:
    def __init__(self, bars):
        self._bars = bars

    def list_trading_dates(self, start_date: str, end_date: str):
        del start_date, end_date
        return ["20260407", "20260408", "20260409", "20260410", "20260413"]

    def get_daily(self, ts_code: str, trade_date: str):
        return self._bars.get((ts_code, trade_date))


def test_next_open_limit_up_is_unfillable() -> None:
    assert _looks_unfillable_limit_up(
        entry_bar={
            "open": 11.0,
            "high": 11.0,
            "low": 11.0,
            "close": 11.0,
            "pre_close": 10.0,
            "pct_chg": 10.0,
        },
        ts_code="600000.SH",
        name="浦发银行",
        market="主板",
    ) is True


def test_forward_return_uses_next_open_and_skips_unfillable_entry() -> None:
    repo = _FakeRepo(
        {
            ("000001.SZ", "20260408"): {
                "open": 11.0,
                "high": 11.0,
                "low": 11.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
            },
            ("000001.SZ", "20260409"): {"close": 11.5},
        }
    )

    result = _forward_return(
        repo,
        ts_code="000001.SZ",
        trade_day=date(2026, 4, 7),
        horizon=1,
        record={"name": "平安银行", "market": "主板"},
        entry_mode="next_open",
        require_fillable_entry=True,
    )

    assert result == {"return": None, "fillable": False}


def test_forward_return_uses_next_open_price_when_fillable() -> None:
    repo = _FakeRepo(
        {
            ("000001.SZ", "20260408"): {
                "open": 10.2,
                "high": 10.8,
                "low": 10.1,
                "close": 10.6,
                "pre_close": 10.0,
                "pct_chg": 6.0,
            },
            ("000001.SZ", "20260410"): {"close": 11.22},
        }
    )

    result = _forward_return(
        repo,
        ts_code="000001.SZ",
        trade_day=date(2026, 4, 7),
        horizon=3,
        record={"name": "平安银行", "market": "主板"},
        entry_mode="next_open",
        require_fillable_entry=True,
    )

    assert result["fillable"] is True
    assert result["return"] == 0.1


def test_pick_tradeable_codes_refills_after_unfillable_entry() -> None:
    repo = _FakeRepo(
        {
            ("000001.SZ", "20260408"): {
                "open": 11.0,
                "high": 11.0,
                "low": 11.0,
                "close": 11.0,
                "pre_close": 10.0,
                "pct_chg": 10.0,
            },
            ("000002.SZ", "20260408"): {
                "open": 10.1,
                "high": 10.4,
                "low": 10.0,
                "close": 10.3,
                "pre_close": 10.0,
                "pct_chg": 3.0,
            },
            ("000003.SZ", "20260408"): {
                "open": 10.2,
                "high": 10.5,
                "low": 10.1,
                "close": 10.4,
                "pre_close": 10.0,
                "pct_chg": 4.0,
            },
        }
    )

    picked = _pick_tradeable_codes(
        repo,
        date(2026, 4, 7),
        ["000001.SZ", "000002.SZ", "000003.SZ"],
        records={
            "000001.SZ": {"name": "顶板股", "market": "主板"},
            "000002.SZ": {"name": "可成交A", "market": "主板"},
            "000003.SZ": {"name": "可成交B", "market": "主板"},
        },
        top_k=2,
        entry_mode="next_open",
        require_fillable_entry=True,
        refill_unfillable=True,
    )

    assert picked == ["000002.SZ", "000003.SZ"]


def test_day_cache_round_trip(tmp_path) -> None:
    cache_path = _build_day_cache_path(
        str(tmp_path),
        trade_day=date(2026, 4, 7),
        candidate_limit=200,
        exclude_bj=False,
    )

    _write_day_cache(
        cache_path,
        payload={
            "trade_date": "2026-04-07",
            "stage1_candidate_codes": ["000001.SZ"],
            "stage2_top20_codes": ["000001.SZ"],
            "stage1_records": {"000001.SZ": {"name": "平安银行"}},
            "diagnostics": {"pool_size": 1},
        },
    )

    loaded = _load_day_cache(cache_path)

    assert loaded is not None
    assert loaded["trade_date"] == "2026-04-07"
    assert loaded["stage1_candidate_codes"] == ["000001.SZ"]
