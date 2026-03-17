from octts.config import Settings
from octts.clients.tushare_client import TushareClient


class RecordingTushareClient(TushareClient):
    def __init__(self) -> None:
        self._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
        self.calls: list[tuple[str, str | None]] = []

    def _fetch_daily(self, *, ts_code: str, trade_date: str | None) -> dict[str, object]:
        self.calls.append(("daily", trade_date))
        return {
            "trade_date": "20260316",
            "open": 10.0,
            "close": 10.2,
            "pct_chg": 1.5,
            "amount": 10200000,
            "high": 10.3,
            "low": 9.9,
        }

    def _fetch_daily_basic(self, *, ts_code: str, trade_date: str | None) -> dict[str, object]:
        self.calls.append(("daily_basic", trade_date))
        return {"volume_ratio": 1.1, "turnover_rate": 1.2}

    def _fetch_minute_summary(self, *, ts_code: str, trade_date: str | None) -> list[dict[str, object]]:
        self.calls.append(("minute_summary", trade_date))
        return []

    def _fetch_daily_summary(self, *, ts_code: str, trade_date: str | None) -> list[dict[str, object]]:
        self.calls.append(("daily_summary", trade_date))
        return [{"trade_date": "20260316", "close": 10.2}]

    def _fetch_weekly_summary(self, *, ts_code: str, trade_date: str | None) -> list[dict[str, object]]:
        self.calls.append(("weekly_summary", trade_date))
        return [{"trade_date": "20260313", "close": 10.2}]

    def _fetch_moneyflow_summary(self, *, ts_code: str, trade_date: str | None) -> dict[str, object]:
        self.calls.append(("moneyflow_summary", trade_date))
        return {"net_mf_amount": 1200}

    def _fetch_stock_name(self, ts_code: str) -> str:
        self.calls.append(("stock_name", None))
        return "PF Bank"


def test_build_snapshot_aligns_related_data_to_resolved_trade_date() -> None:
    client = RecordingTushareClient()

    snapshot = client._build_snapshot(
        ts_code="600000.SH",
        phase="review",
        trade_date=None,
        include_minute_summary=True,
    )

    assert snapshot.trade_date == "20260316"
    assert ("daily_basic", "20260316") in client.calls
    assert ("minute_summary", "20260316") in client.calls
    assert ("daily_summary", "20260316") in client.calls
    assert ("weekly_summary", "20260316") in client.calls
    assert ("moneyflow_summary", "20260316") in client.calls


def test_fetch_minute_summary_skips_non_current_trade_date(monkeypatch) -> None:
    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")

    class UnexpectedCall:
        def rt_min(self, *args, **kwargs):
            raise AssertionError("rt_min should not be called for stale trade dates")

    client._pro = UnexpectedCall()
    client._ts = UnexpectedCall()

    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260317")

    assert client._fetch_minute_summary(ts_code="600000.SH", trade_date="20260316") == []


def test_call_pro_bar_falls_back_when_api_kwarg_is_unsupported() -> None:
    client = object.__new__(TushareClient)
    client._pro = object()

    class ProBarCompat:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def pro_bar(self, **kwargs):
            self.calls.append(kwargs)
            if "api" in kwargs:
                raise TypeError("pro_bar() got an unexpected keyword argument 'api'")
            return {"ok": True, "kwargs": kwargs}

    client._ts = ProBarCompat()

    result = client._call_pro_bar(ts_code="600000.SH", asset="E", freq="D")

    assert result["ok"] is True
    assert len(client._ts.calls) == 2
    assert "api" in client._ts.calls[0]
    assert "api" not in client._ts.calls[1]


def test_fetch_daily_picks_latest_trade_date_when_frame_order_is_ascending() -> None:
    import pandas as pd

    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")

    class DailyOnly:
        def daily(self, *, ts_code: str, trade_date: str | None):
            del ts_code, trade_date
            return pd.DataFrame(
                [
                    {"trade_date": "20260312", "close": 59.04},
                    {"trade_date": "20260316", "close": 56.95},
                    {"trade_date": "20260317", "close": 52.84},
                ]
            )

    client._pro = DailyOnly()
    client._ts = object()

    record = client._fetch_daily(ts_code="301568.SZ", trade_date=None)

    assert record["trade_date"] == "20260317"
    assert record["close"] == 52.84
