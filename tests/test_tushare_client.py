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


def test_fetch_minute_summary_filters_today_and_sorts_ascending(monkeypatch) -> None:
    import pandas as pd

    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")

    class MinuteOnly:
        def rt_min(self, *, ts_code: str, freq: str):
            del ts_code, freq
            return pd.DataFrame(
                [
                    {"trade_time": "2026-03-17 14:30:00", "close": 9.9},
                    {"trade_time": "2026-03-18 10:30:00", "close": 10.4},
                    {"trade_time": "2026-03-18 09:30:00", "close": 10.1},
                ]
            )

    client._pro = MinuteOnly()
    client._ts = object()
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    rows = client._fetch_minute_summary(ts_code="600000.SH", trade_date="20260318")

    assert [item["trade_time"] for item in rows] == ["2026-03-18 09:30:00", "2026-03-18 10:30:00"]


def test_fetch_minute_summary_accepts_time_only_records_for_today(monkeypatch) -> None:
    import pandas as pd

    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")

    class MinuteOnly:
        def rt_min(self, *, ts_code: str, freq: str):
            del ts_code, freq
            return pd.DataFrame(
                [
                    {"trade_time": "10:30:00", "close": 10.4},
                    {"trade_time": "09:30:00", "close": 10.1},
                ]
            )

    client._pro = MinuteOnly()
    client._ts = object()
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    rows = client._fetch_minute_summary(ts_code="600000.SH", trade_date="20260318")

    assert [item["trade_time"] for item in rows] == ["09:30:00", "10:30:00"]


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


def test_fetch_daily_picks_latest_trade_date_when_frame_order_is_ascending(monkeypatch) -> None:
    import pandas as pd

    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    class DailyOnly:
        def __init__(self) -> None:
            self.daily_calls: list[str | None] = []

        def trade_cal(self, *, exchange: str, start_date: str, end_date: str, is_open: str):
            del exchange, start_date, end_date, is_open
            return pd.DataFrame(
                [
                    {"cal_date": "20260312"},
                    {"cal_date": "20260313"},
                    {"cal_date": "20260314"},
                    {"cal_date": "20260317"},
                    {"cal_date": "20260318"},
                ]
            )

        def daily(self, *, ts_code: str, trade_date: str | None):
            del ts_code
            self.daily_calls.append(trade_date)
            if trade_date == "20260318":
                return pd.DataFrame()
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
    assert client._pro.daily_calls == ["20260318", "20260317"]


def test_build_snapshot_uses_intraday_data_when_today_daily_is_unavailable(monkeypatch) -> None:
    client = RecordingTushareClient()
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    def fetch_daily(*, ts_code: str, trade_date: str | None) -> dict[str, object]:
        del ts_code, trade_date
        return {
            "trade_date": "20260317",
            "open": 10.0,
            "close": 10.0,
            "pct_chg": 0.0,
            "amount": 1000000,
            "high": 10.1,
            "low": 9.9,
        }

    def fetch_minute_summary(*, ts_code: str, trade_date: str | None) -> list[dict[str, object]]:
        del ts_code
        client.calls.append(("minute_summary", trade_date))
        return [
            {"trade_time": "2026-03-18 09:30:00", "open": 10.1, "high": 10.2, "low": 10.0, "close": 10.15, "amount": 1000},
            {"trade_time": "2026-03-18 10:00:00", "open": 10.15, "high": 10.4, "low": 10.1, "close": 10.3, "amount": 1200},
        ]

    client._fetch_daily = fetch_daily  # type: ignore[method-assign]
    client._fetch_minute_summary = fetch_minute_summary  # type: ignore[method-assign]

    snapshot = client._build_snapshot(
        ts_code="600000.SH",
        phase="review",
        trade_date="20260318",
        include_minute_summary=True,
    )

    assert snapshot.trade_date == "20260318"
    assert snapshot.open == 10.1
    assert snapshot.close == 10.3
    assert snapshot.high == 10.4
    assert snapshot.low == 10.0
    assert snapshot.amount == 2200
    assert round(snapshot.pct_chg or 0, 2) == 3.0
    assert ("daily_basic", "20260318") in client.calls


def test_build_snapshot_prefers_daily_data_after_close_when_today_daily_exists(monkeypatch) -> None:
    client = RecordingTushareClient()
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    def fetch_daily(*, ts_code: str, trade_date: str | None) -> dict[str, object]:
        del ts_code, trade_date
        return {
            "trade_date": "20260318",
            "open": 10.2,
            "close": 10.5,
            "pct_chg": 5.0,
            "amount": 5000000,
            "high": 10.6,
            "low": 10.1,
        }

    def fetch_minute_summary(*, ts_code: str, trade_date: str | None) -> list[dict[str, object]]:
        del ts_code
        client.calls.append(("minute_summary", trade_date))
        return [
            {"trade_time": "2026-03-18 09:30:00", "open": 10.1, "high": 10.2, "low": 10.0, "close": 10.15, "amount": 1000}
        ]

    client._fetch_daily = fetch_daily  # type: ignore[method-assign]
    client._fetch_minute_summary = fetch_minute_summary  # type: ignore[method-assign]

    snapshot = client._build_snapshot(
        ts_code="600000.SH",
        phase="review",
        trade_date="20260318",
        include_minute_summary=True,
    )

    assert snapshot.trade_date == "20260318"
    assert snapshot.close == 10.5
    assert snapshot.minute_summary == []


def test_build_snapshot_uses_realtime_quote_when_minute_data_is_unavailable(monkeypatch) -> None:
    client = RecordingTushareClient()
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    def fetch_daily(*, ts_code: str, trade_date: str | None) -> dict[str, object]:
        del ts_code, trade_date
        return {
            "trade_date": "20260317",
            "open": 57.28,
            "close": 52.84,
            "pct_chg": -7.2169,
            "amount": 260939.064,
            "high": 57.36,
            "low": 52.7,
        }

    def fetch_minute_summary(*, ts_code: str, trade_date: str | None) -> list[dict[str, object]]:
        del ts_code, trade_date
        client.calls.append(("minute_summary", trade_date))
        return []

    class QuotesOnly:
        def get_realtime_quotes(self, code: str):
            import pandas as pd

            assert code == "600000"
            return pd.DataFrame(
                [
                    {
                        "date": "2026-03-18",
                        "time": "11:30:00",
                        "open": "52.980",
                        "pre_close": "52.840",
                        "price": "54.680",
                        "high": "54.760",
                        "low": "52.850",
                        "amount": "128386958.740",
                    }
                ]
            )

    client._fetch_daily = fetch_daily  # type: ignore[method-assign]
    client._fetch_minute_summary = fetch_minute_summary  # type: ignore[method-assign]
    client._ts = QuotesOnly()

    snapshot = client._build_snapshot(
        ts_code="600000.SH",
        phase="review",
        trade_date="20260318",
        include_minute_summary=True,
    )

    assert snapshot.trade_date == "20260318"
    assert snapshot.open == 52.98
    assert snapshot.close == 54.68
    assert snapshot.high == 54.76
    assert snapshot.low == 52.85
    assert round(snapshot.amount or 0, 3) == 128386.959
    assert round(snapshot.pct_chg or 0, 4) == 3.4815
