from datetime import datetime
from pathlib import Path
from octts.config import Settings
from octts.clients.tushare_client import TushareClient
from typing import Any, Dict, List, Optional, Tuple


class RecordingTushareClient(TushareClient):
    def __init__(self) -> None:
        self._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
        self.calls: List[Tuple[str, Optional[str]]] = []
        self._ts = object()

    def _fetch_daily(self, *, ts_code: str, trade_date: Optional[str]) -> Dict[str, object]:
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

    def _fetch_daily_basic(self, *, ts_code: str, trade_date: Optional[str]) -> Dict[str, object]:
        self.calls.append(("daily_basic", trade_date))
        return {"volume_ratio": 1.1, "turnover_rate": 1.2}

    def _fetch_minute_summary(self, *, ts_code: str, trade_date: Optional[str]) -> List[Dict[str, object]]:
        self.calls.append(("minute_summary", trade_date))
        return []

    def _fetch_daily_summary(self, *, ts_code: str, trade_date: Optional[str]) -> List[Dict[str, object]]:
        self.calls.append(("daily_summary", trade_date))
        return [{"trade_date": "20260316", "close": 10.2}]

    def _fetch_weekly_summary(self, *, ts_code: str, trade_date: Optional[str]) -> List[Dict[str, object]]:
        self.calls.append(("weekly_summary", trade_date))
        return [{"trade_date": "20260313", "close": 10.2}]

    def _fetch_moneyflow_summary(self, *, ts_code: str, trade_date: Optional[str]) -> Dict[str, object]:
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
    assert ("minute_summary", "20260327") in client.calls
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


def test_call_pro_bar_returns_none_for_out_of_bounds_error(monkeypatch) -> None:
    client = object.__new__(TushareClient)
    client._pro = object()
    debug_calls: list[tuple[str, dict[str, object]]] = []

    class ProBarBroken:
        def pro_bar(self, **kwargs):
            del kwargs
            raise IndexError("single positional indexer is out-of-bounds")

    client._ts = ProBarBroken()
    monkeypatch.setattr(
        "octts.clients.tushare_client.logger.debug",
        lambda message, kwargs: debug_calls.append((message, kwargs.copy())),
    )

    result = client._call_pro_bar(ts_code="600000.SH", asset="E", freq="D")

    assert result is None
    assert debug_calls == [
        (
            "pro_bar returned no usable rows: %s",
            {"ts_code": "600000.SH", "asset": "E", "freq": "D"},
        )
    ]


def test_fetch_daily_batch_returns_empty_list_for_out_of_bounds_errors() -> None:
    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    calls: list[str] = []

    def fake_call_pro_bar(**kwargs):
        ts_code = kwargs["ts_code"]
        calls.append(ts_code)
        if ts_code == "000001.SZ":
            return None

        import pandas as pd

        return pd.DataFrame(
            [
                {"ts_code": ts_code, "trade_date": "20260318", "close": 10.2},
                {"ts_code": ts_code, "trade_date": "20260317", "close": 10.1},
            ]
        )

    client._call_pro_bar = fake_call_pro_bar

    result = client.fetch_daily_batch(
        ts_codes=["000001.SZ", "600000.SH"],
        start_date="20260301",
        end_date="20260318",
    )

    assert result == {
        "000001.SZ": [],
        "600000.SH": [
            {"ts_code": "600000.SH", "trade_date": "20260318", "close": 10.2},
            {"ts_code": "600000.SH", "trade_date": "20260317", "close": 10.1},
        ],
    }
    assert calls == ["000001.SZ", "600000.SH"]


def test_fetch_daily_picks_latest_trade_date_when_frame_order_is_ascending(monkeypatch) -> None:
    import pandas as pd

    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    class DailyOnly:
        def __init__(self) -> None:
            self.daily_calls: List[Optional[str]] = []

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

        def daily(self, *, ts_code: str, trade_date: Optional[str]):
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


def test_fetch_daily_basic_batch_falls_back_to_recent_trade_date(monkeypatch) -> None:
    import pandas as pd

    client = object.__new__(TushareClient)
    client._settings = Settings(TUSHARE_TOKEN="token", OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    monkeypatch.setattr(
        client,
        "_candidate_trade_dates",
        lambda *, target_trade_date: ["20260318", "20260317"],
    )

    class DailyBasicOnly:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def daily_basic(self, *, ts_code: str, trade_date: str, fields: Optional[str] = None):
            del fields
            self.calls.append((ts_code, trade_date))
            if trade_date == "20260318":
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "ts_code": "600000.SH",
                        "close": 10.2,
                        "turnover_rate": 1.5,
                        "volume_ratio": 1.3,
                        "total_mv": 2000000,
                    }
                ]
            )

    client._pro = DailyBasicOnly()
    client._ts = object()

    result = client.fetch_daily_basic_batch(ts_codes=["600000.SH"], trade_date="20260318")

    assert result["600000.SH"]["close"] == 10.2
    assert client._pro.calls == [
        ("600000.SH", "20260318"),
        ("600000.SH", "20260317"),
    ]


def test_build_snapshot_uses_intraday_data_when_today_daily_is_unavailable(monkeypatch) -> None:
    client = RecordingTushareClient()
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260318")

    def fetch_daily(*, ts_code: str, trade_date: Optional[str]) -> Dict[str, object]:
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

    def fetch_minute_summary(*, ts_code: str, trade_date: Optional[str]) -> List[Dict[str, object]]:
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

    def fetch_daily(*, ts_code: str, trade_date: Optional[str]) -> Dict[str, object]:
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

    def fetch_minute_summary(*, ts_code: str, trade_date: Optional[str]) -> List[Dict[str, object]]:
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

    def fetch_daily(*, ts_code: str, trade_date: Optional[str]) -> Dict[str, object]:
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

    def fetch_minute_summary(*, ts_code: str, trade_date: Optional[str]) -> List[Dict[str, object]]:
        del ts_code
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
    assert round(snapshot.pct_chg or 0, 4) == 3.4822


class ScreeningSnapshotClient(TushareClient):
    def __init__(self, history_dir: Path) -> None:
        self._settings = Settings(
            TUSHARE_TOKEN="token",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH="memory.json",
            OCTTS_HISTORY_DIR=str(history_dir),
        )
        self._ts = object()
        self.build_calls = 0

    def _resolve_target_trade_date(self, trade_date: Optional[str]) -> str:
        assert trade_date is not None
        return trade_date

    def fetch_stock_list(self, *, list_status: str = "L") -> List[Dict[str, Any]]:
        del list_status
        self.build_calls += 1
        return [
            {"ts_code": "600000.SH", "name": "PF Bank"},
            {"ts_code": "000001.SZ", "name": "SZ Bank"},
        ]

    def fetch_daily_basic_batch(self, *, ts_codes: List[str], trade_date: str) -> Dict[str, Dict[str, Any]]:
        del ts_codes, trade_date
        return {
            "600000.SH": {"ts_code": "600000.SH", "close": 10.5, "turnover_rate": 1.2, "total_mv": 1200000},
            "000001.SZ": {"ts_code": "000001.SZ", "close": 12.3, "turnover_rate": 0.9, "volume_ratio": 1.1},
        }


def test_screening_snapshot_rebuilds_today_cache_created_before_close(monkeypatch, tmp_path) -> None:
    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260327"
    snapshot_path = client._screening_snapshot_path(trade_date)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"trade_date":"20260327","created_at":"2026-03-27T15:00:00","stocks":[{"ts_code":"000001.SZ"}],"daily_basic":{"000001.SZ":{"turnover_rate":0.9}},"daily":{}}',
        encoding="utf-8",
    )

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 3, 27, 16, 30, 0)

        @classmethod
        def combine(cls, date, time_value):
            return datetime.combine(date, time_value)

        @classmethod
        def fromisoformat(cls, date_string):
            return datetime.fromisoformat(date_string)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return datetime.fromtimestamp(timestamp, tz=tz)

    monkeypatch.setattr("octts.clients.tushare_client.datetime", FrozenDateTime)
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: trade_date)

    snapshot = client.get_or_build_screening_snapshot(trade_date)

    assert client.build_calls == 1
    assert snapshot["stocks"] == [
        {"ts_code": "600000.SH", "name": "PF Bank"},
        {"ts_code": "000001.SZ", "name": "SZ Bank"},
    ]
    assert snapshot["daily_basic"] == {
        "600000.SH": {"ts_code": "600000.SH", "close": 10.5, "turnover_rate": 1.2, "total_mv": 1200000},
        "000001.SZ": {"ts_code": "000001.SZ", "close": 12.3, "turnover_rate": 0.9, "volume_ratio": 1.1},
    }


def test_screening_snapshot_rebuilds_invalid_today_cache_created_after_close(monkeypatch, tmp_path) -> None:
    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260327"
    snapshot_path = client._screening_snapshot_path(trade_date)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"trade_date":"20260327","created_at":"2026-03-27T16:05:00","stocks":[{"ts_code":"000001.SZ"}],"daily_basic":{"000001.SZ":{"turnover_rate":0.9}},"daily":{}}',
        encoding="utf-8",
    )

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 3, 27, 16, 30, 0)

        @classmethod
        def combine(cls, date, time_value):
            return datetime.combine(date, time_value)

        @classmethod
        def fromisoformat(cls, date_string):
            return datetime.fromisoformat(date_string)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return datetime.fromtimestamp(timestamp, tz=tz)

    monkeypatch.setattr("octts.clients.tushare_client.datetime", FrozenDateTime)
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: trade_date)

    snapshot = client.get_or_build_screening_snapshot(trade_date)

    assert client.build_calls == 1
    assert snapshot["stocks"] == [
        {"ts_code": "600000.SH", "name": "PF Bank"},
        {"ts_code": "000001.SZ", "name": "SZ Bank"},
    ]


def test_screening_snapshot_hits_valid_today_cache_created_after_close(monkeypatch, tmp_path) -> None:
    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260327"
    snapshot_path = client._screening_snapshot_path(trade_date)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"trade_date":"20260327","created_at":"2026-03-27T16:05:00","stocks":[{"ts_code":"000001.SZ","name":"SZ Bank"}],"daily_basic":{"000001.SZ":{"ts_code":"000001.SZ","close":12.3,"turnover_rate":0.9}},"daily":{}}',
        encoding="utf-8",
    )

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 3, 27, 16, 30, 0)

        @classmethod
        def combine(cls, date, time_value):
            return datetime.combine(date, time_value)

        @classmethod
        def fromisoformat(cls, date_string):
            return datetime.fromisoformat(date_string)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return datetime.fromtimestamp(timestamp, tz=tz)

    monkeypatch.setattr("octts.clients.tushare_client.datetime", FrozenDateTime)
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: trade_date)

    snapshot = client.get_or_build_screening_snapshot(trade_date)

    assert client.build_calls == 0
    assert snapshot["stocks"] == [{"ts_code": "000001.SZ", "name": "SZ Bank"}]


def test_screening_snapshot_uses_file_mtime_when_created_at_missing(monkeypatch, tmp_path) -> None:
    import os

    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260327"
    snapshot_path = client._screening_snapshot_path(trade_date)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"trade_date":"20260327","stocks":[{"ts_code":"000001.SZ","name":"SZ Bank"}],"daily_basic":{"000001.SZ":{"ts_code":"000001.SZ","close":12.3,"turnover_rate":0.9}},"daily":{}}',
        encoding="utf-8",
    )
    mtime = datetime(2026, 3, 27, 16, 10, 0).timestamp()
    os.utime(snapshot_path, (mtime, mtime))

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 3, 27, 16, 30, 0)

        @classmethod
        def combine(cls, date, time_value):
            return datetime.combine(date, time_value)

        @classmethod
        def fromisoformat(cls, date_string):
            return datetime.fromisoformat(date_string)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return datetime.fromtimestamp(timestamp, tz=tz)

    monkeypatch.setattr("octts.clients.tushare_client.datetime", FrozenDateTime)
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: trade_date)

    snapshot = client.get_or_build_screening_snapshot(trade_date)

    assert client.build_calls == 0
    assert snapshot["stocks"] == [{"ts_code": "000001.SZ", "name": "SZ Bank"}]


def test_screening_snapshot_rebuilds_invalid_historical_cache(monkeypatch, tmp_path) -> None:
    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260326"
    snapshot_path = client._screening_snapshot_path(trade_date)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"trade_date":"20260326","created_at":"2026-03-26T10:00:00","stocks":[{"ts_code":"000001.SZ"}],"daily_basic":{"000001.SZ":{"turnover_rate":0.9}},"daily":{}}',
        encoding="utf-8",
    )

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 3, 27, 16, 30, 0)

        @classmethod
        def combine(cls, date, time_value):
            return datetime.combine(date, time_value)

        @classmethod
        def fromisoformat(cls, date_string):
            return datetime.fromisoformat(date_string)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return datetime.fromtimestamp(timestamp, tz=tz)

    monkeypatch.setattr("octts.clients.tushare_client.datetime", FrozenDateTime)
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260327")

    snapshot = client.get_or_build_screening_snapshot(trade_date)

    assert client.build_calls == 1
    assert snapshot["stocks"] == [
        {"ts_code": "600000.SH", "name": "PF Bank"},
        {"ts_code": "000001.SZ", "name": "SZ Bank"},
    ]


def test_screening_snapshot_hits_valid_historical_cache(monkeypatch, tmp_path) -> None:
    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260326"
    snapshot_path = client._screening_snapshot_path(trade_date)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"trade_date":"20260326","created_at":"2026-03-26T10:00:00","stocks":[{"ts_code":"000001.SZ","name":"SZ Bank"}],"daily_basic":{"000001.SZ":{"ts_code":"000001.SZ","close":12.3,"turnover_rate":0.9}},"daily":{}}',
        encoding="utf-8",
    )

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            del tz
            return cls(2026, 3, 27, 16, 30, 0)

        @classmethod
        def combine(cls, date, time_value):
            return datetime.combine(date, time_value)

        @classmethod
        def fromisoformat(cls, date_string):
            return datetime.fromisoformat(date_string)

        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            return datetime.fromtimestamp(timestamp, tz=tz)

    monkeypatch.setattr("octts.clients.tushare_client.datetime", FrozenDateTime)
    monkeypatch.setattr("octts.clients.tushare_client._today_trade_date", lambda: "20260327")

    snapshot = client.get_or_build_screening_snapshot(trade_date)

    assert client.build_calls == 0
    assert snapshot["stocks"] == [{"ts_code": "000001.SZ", "name": "SZ Bank"}]


def test_screening_snapshot_skips_writing_invalid_build(tmp_path) -> None:
    client = ScreeningSnapshotClient(tmp_path)
    trade_date = "20260327"
    snapshot = {
        "trade_date": trade_date,
        "created_at": "2026-03-27T16:30:00",
        "stocks": [{"ts_code": "600000.SH", "name": "PF Bank"}],
        "daily_basic": {"600000.SH": {"turnover_rate": 1.2}},
        "daily": {},
    }

    client._write_screening_snapshot(trade_date, snapshot)

    assert client._read_screening_snapshot(trade_date) is None
