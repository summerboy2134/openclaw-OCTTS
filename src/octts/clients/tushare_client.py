from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from octts.config import Settings
from octts.schemas.backtest import DailyBar
from octts.schemas.report import AnalysisPhase, PriceSnapshot


class TushareClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.tushare_token:
            raise ValueError("TINYSHARE_TOKEN or TUSHARE_TOKEN is required.")

        try:
            import tinyshare as ts
        except ImportError as exc:
            try:
                import tushare as ts
            except ImportError:
                raise RuntimeError("tinyshare is not installed.") from exc

        ts.set_token(settings.tushare_token)
        self._ts = ts
        self._pro = ts.pro_api(settings.tushare_token)
        self._settings = settings

    def fetch_snapshot(
        self,
        *,
        ts_code: str,
        phase: AnalysisPhase,
        trade_date: str | None = None,
    ) -> PriceSnapshot:
        return self._build_snapshot(
            ts_code=ts_code,
            phase=phase,
            trade_date=trade_date,
            include_minute_summary=True,
        )

    def fetch_historical_snapshot(
        self,
        *,
        ts_code: str,
        phase: AnalysisPhase = "review",
        trade_date: str,
    ) -> PriceSnapshot:
        return self._build_snapshot(
            ts_code=ts_code,
            phase=phase,
            trade_date=trade_date,
            include_minute_summary=False,
        )

    def fetch_trading_dates(self, *, start_date: str, end_date: str) -> list[str]:
        df = self._pro.trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            is_open="1",
        )
        if df is None or df.empty:
            return []
        if "cal_date" not in df.columns:
            return []
        return sorted(str(item) for item in df["cal_date"].tolist())

    def fetch_daily_bars(self, *, ts_code: str, start_date: str, end_date: str) -> list[DailyBar]:
        try:
            df = self._ts.pro_bar(
                api=self._pro,
                ts_code=ts_code,
                asset="E",
                start_date=start_date,
                end_date=end_date,
                freq="D",
                adj="qfq",
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []

        records = [DailyBar.model_validate(_serialize_daily_bar(row)) for row in df.to_dict(orient="records")]
        records.sort(key=lambda item: item.trade_date)
        return records

    def _build_snapshot(
        self,
        *,
        ts_code: str,
        phase: AnalysisPhase,
        trade_date: str | None,
        include_minute_summary: bool,
    ) -> PriceSnapshot:
        daily = self._fetch_daily(ts_code=ts_code, trade_date=trade_date)
        daily_basic = self._fetch_daily_basic(ts_code=ts_code, trade_date=trade_date)
        minute_summary = self._fetch_minute_summary(ts_code=ts_code) if include_minute_summary else []
        daily_summary = self._fetch_daily_summary(ts_code=ts_code, trade_date=trade_date)
        weekly_summary = self._fetch_weekly_summary(ts_code=ts_code, trade_date=trade_date)
        moneyflow_summary = self._fetch_moneyflow_summary(ts_code=ts_code, trade_date=trade_date)
        name = self._fetch_stock_name(ts_code)

        return PriceSnapshot(
            ts_code=ts_code,
            name=name,
            trade_date=daily.get("trade_date"),
            open=_safe_float(daily.get("open")),
            close=_safe_float(daily.get("close")),
            pct_chg=_safe_float(daily.get("pct_chg")),
            vol_ratio=_safe_float(daily_basic.get("volume_ratio")),
            turnover_rate=_safe_float(daily_basic.get("turnover_rate")),
            amount=_safe_float(daily.get("amount")),
            high=_safe_float(daily.get("high")),
            low=_safe_float(daily.get("low")),
            minute_summary=minute_summary,
            daily_summary=daily_summary,
            weekly_summary=weekly_summary,
            moneyflow_summary={
                **moneyflow_summary,
                "phase": phase,
            },
        )

    def _fetch_daily(self, *, ts_code: str, trade_date: str | None) -> dict[str, Any]:
        df = self._pro.daily(ts_code=ts_code, trade_date=trade_date)
        if df.empty:
            # Fallback to a recent bar if same-day daily data is not ready yet.
            end_date = trade_date or datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=self._settings.default_lookback_days)).strftime("%Y%m%d")
            df = self._ts.pro_bar(
                api=self._pro,
                ts_code=ts_code,
                asset="E",
                start_date=start_date,
                end_date=end_date,
                freq="D",
                adj="qfq",
            )

        if df is None or df.empty:
            raise ValueError(f"No daily market data returned for {ts_code}.")

        return df.iloc[0].to_dict()

    def _fetch_daily_basic(self, *, ts_code: str, trade_date: str | None) -> dict[str, Any]:
        df = self._pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    def _fetch_minute_summary(self, *, ts_code: str) -> list[dict[str, Any]]:
        try:
            df = self._pro.rt_min(ts_code=ts_code, freq=self._settings.minute_freq)
        except Exception:
            df = None

        if df is None or df.empty:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=2)
            try:
                df = self._ts.pro_bar(
                    api=self._pro,
                    ts_code=ts_code,
                    asset="E",
                    start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    freq=self._settings.minute_freq.lower(),
                    adj="qfq",
                )
            except Exception:
                return []

        if df is None or df.empty:
            return []

        subset = df.head(16)
        return [row for row in subset.to_dict(orient="records")]

    def _fetch_daily_summary(self, *, ts_code: str, trade_date: str | None) -> list[dict[str, Any]]:
        end_date = trade_date or datetime.now().strftime("%Y%m%d")
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(self._settings.default_lookback_days, 20) * 2)).strftime(
            "%Y%m%d"
        )
        try:
            df = self._ts.pro_bar(
                api=self._pro,
                ts_code=ts_code,
                asset="E",
                start_date=start_date,
                end_date=end_date,
                freq="D",
                adj="qfq",
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return [row for row in df.head(max(self._settings.default_lookback_days, 20)).to_dict(orient="records")]

    def _fetch_weekly_summary(self, *, ts_code: str, trade_date: str | None) -> list[dict[str, Any]]:
        end_date = trade_date or datetime.now().strftime("%Y%m%d")
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=120)).strftime("%Y%m%d")
        try:
            df = self._ts.pro_bar(
                api=self._pro,
                ts_code=ts_code,
                asset="E",
                start_date=start_date,
                end_date=end_date,
                freq="W",
                adj="qfq",
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return [row for row in df.head(12).to_dict(orient="records")]

    def _fetch_moneyflow_summary(self, *, ts_code: str, trade_date: str | None) -> dict[str, Any]:
        try:
            df = self._pro.moneyflow(ts_code=ts_code, trade_date=trade_date)
        except Exception:
            return {}

        if df.empty:
            return {}

        record = df.iloc[0].to_dict()
        keys = [
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_md_amount",
            "sell_md_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ]
        return {key: record.get(key) for key in keys if key in record}

    def _fetch_stock_name(self, ts_code: str) -> str | None:
        try:
            df = self._pro.stock_basic(ts_code=ts_code)
        except Exception:
            return None

        if df.empty:
            return None
        return df.iloc[0].get("name")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _serialize_daily_bar(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts_code": row.get("ts_code"),
        "trade_date": str(row.get("trade_date")),
        "open": _safe_float(row.get("open")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),
        "close": _safe_float(row.get("close")),
        "pct_chg": _safe_float(row.get("pct_chg")),
        "vol": _safe_float(row.get("vol")),
        "amount": _safe_float(row.get("amount")),
    }
