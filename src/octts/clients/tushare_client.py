from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from octts.config import Settings
from octts.schemas.backtest import DailyBar
from octts.schemas.report import AnalysisPhase, PriceSnapshot

MAX_RECENT_TRADE_DATE_FALLBACKS = 3


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

        import tushare.pro.client as _client
        _client.DataApi._DataApi__http_url = "http://tushare.xyz"

        ts.set_token(settings.tushare_token)
        self._ts = ts
        self._pro = ts.pro_api(settings.tushare_token)
        self._settings = settings

    def fetch_snapshot(
        self,
        *,
        ts_code: str,
        phase: AnalysisPhase,
        trade_date: Optional[str] = None,
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
            df = self._call_pro_bar(
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
        trade_date: Optional[str],
        include_minute_summary: bool,
    ) -> PriceSnapshot:
        target_trade_date = self._resolve_target_trade_date(trade_date)
        daily = self._fetch_daily(ts_code=ts_code, trade_date=target_trade_date)
        resolved_trade_date = _normalize_trade_date_value(daily.get("trade_date")) or target_trade_date or trade_date
        daily_basic = self._fetch_daily_basic(ts_code=ts_code, trade_date=resolved_trade_date)
        minute_summary = (
            self._fetch_minute_summary(ts_code=ts_code, trade_date=resolved_trade_date) if include_minute_summary else []
        )
        daily_summary = self._fetch_daily_summary(ts_code=ts_code, trade_date=resolved_trade_date)
        weekly_summary = self._fetch_weekly_summary(ts_code=ts_code, trade_date=resolved_trade_date)
        moneyflow_summary = self._fetch_moneyflow_summary(ts_code=ts_code, trade_date=resolved_trade_date)
        name = self._fetch_stock_name(ts_code)

        return PriceSnapshot(
            ts_code=ts_code,
            name=name,
            trade_date=resolved_trade_date,
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

    def _resolve_target_trade_date(self, trade_date: Optional[str]) -> str:
        normalized_trade_date = _normalize_trade_date_value(trade_date)
        if normalized_trade_date:
            return normalized_trade_date

        today_trade_date = _today_trade_date()
        trading_dates = self._list_recent_trading_dates(end_date=today_trade_date)
        if trading_dates:
            return trading_dates[-1]
        return today_trade_date

    def _list_recent_trading_dates(self, *, end_date: str, lookback_days: int = 14) -> list[str]:
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(lookback_days, 1))).strftime("%Y%m%d")
        try:
            return self.fetch_trading_dates(start_date=start_date, end_date=end_date)
        except Exception:
            return []

    def _candidate_trade_dates(self, *, target_trade_date: str) -> list[str]:
        trading_dates = self._list_recent_trading_dates(end_date=target_trade_date)
        eligible_dates = [item for item in trading_dates if item <= target_trade_date]
        if eligible_dates:
            return list(reversed(eligible_dates[-(MAX_RECENT_TRADE_DATE_FALLBACKS + 1) :]))
        return [target_trade_date]

    def _fetch_daily(self, *, ts_code: str, trade_date: Optional[str]) -> dict[str, Any]:
        target_trade_date = self._resolve_target_trade_date(trade_date)
        candidate_trade_dates = self._candidate_trade_dates(target_trade_date=target_trade_date)
        oldest_allowed_trade_date = candidate_trade_dates[-1] if candidate_trade_dates else target_trade_date

        for candidate_trade_date in candidate_trade_dates:
            df = self._pro.daily(ts_code=ts_code, trade_date=candidate_trade_date)
            record = _pick_trade_date_record(
                df,
                target_trade_date=candidate_trade_date,
                oldest_allowed_trade_date=candidate_trade_date,
            )
            if record:
                return record

        # Fallback to a recent bar window when exact-date daily data is unavailable.
        end_date = target_trade_date
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(self._settings.default_lookback_days, 20))).strftime("%Y%m%d")
        df = self._call_pro_bar(
            ts_code=ts_code,
            asset="E",
            start_date=start_date,
            end_date=end_date,
            freq="D",
            adj="qfq",
        )
        record = _pick_trade_date_record(
            df,
            target_trade_date=target_trade_date,
            oldest_allowed_trade_date=oldest_allowed_trade_date,
        )
        if record:
            return record

        raise ValueError(f"No daily market data returned for {ts_code} near {target_trade_date}.")

    def _fetch_daily_basic(self, *, ts_code: str, trade_date: Optional[str]) -> dict[str, Any]:
        df = self._pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        record = _pick_trade_date_record(df, target_trade_date=trade_date, oldest_allowed_trade_date=trade_date)
        if not record:
            return {}
        return record

    def _fetch_minute_summary(self, *, ts_code: str, trade_date: Optional[str]) -> list[dict[str, Any]]:
        if not trade_date or trade_date != _today_trade_date():
            return []
        try:
            df = self._pro.rt_min(ts_code=ts_code, freq=self._settings.minute_freq)
        except Exception:
            df = None

        if df is None or df.empty:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=2)
            try:
                df = self._call_pro_bar(
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

    def _fetch_daily_summary(self, *, ts_code: str, trade_date: Optional[str]) -> list[dict[str, Any]]:
        end_date = trade_date or datetime.now().strftime("%Y%m%d")
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(self._settings.default_lookback_days, 20) * 2)).strftime(
            "%Y%m%d"
        )
        try:
            df = self._call_pro_bar(
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
        df = _sort_frame_by_trade_date_desc(df)
        return [row for row in df.head(max(self._settings.default_lookback_days, 20)).to_dict(orient="records")]

    def _fetch_weekly_summary(self, *, ts_code: str, trade_date: Optional[str]) -> list[dict[str, Any]]:
        end_date = trade_date or datetime.now().strftime("%Y%m%d")
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=120)).strftime("%Y%m%d")
        try:
            df = self._call_pro_bar(
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
        df = _sort_frame_by_trade_date_desc(df)
        return [row for row in df.head(12).to_dict(orient="records")]

    def _fetch_moneyflow_summary(self, *, ts_code: str, trade_date: Optional[str]) -> dict[str, Any]:
        try:
            df = self._pro.moneyflow(ts_code=ts_code, trade_date=trade_date)
        except Exception:
            return {}

        record = _pick_trade_date_record(df, target_trade_date=trade_date, oldest_allowed_trade_date=trade_date)
        if not record:
            return {}
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

    def _fetch_stock_name(self, ts_code: str) -> Optional[str]:
        try:
            df = self._pro.stock_basic(ts_code=ts_code)
        except Exception:
            return None

        if df.empty:
            return None
        return df.iloc[0].get("name")

    def _call_pro_bar(self, **kwargs):
        try:
            return self._ts.pro_bar(api=self._pro, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument 'api'" not in str(exc):
                raise
            return self._ts.pro_bar(**kwargs)


def _safe_float(value: Any) -> Optional[float]:
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


def _today_trade_date() -> str:
    return datetime.now().strftime("%Y%m%d")


def _normalize_trade_date_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if len(text) == 8 and text.isdigit() else None


def _pick_trade_date_record(df, *, target_trade_date: Optional[str], oldest_allowed_trade_date: Optional[str]) -> Optional[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return None

    normalized_target = _normalize_trade_date_value(target_trade_date)
    normalized_oldest = _normalize_trade_date_value(oldest_allowed_trade_date)
    best_record = None
    best_trade_date = None

    for record in df.to_dict(orient="records"):
        record_trade_date = _normalize_trade_date_value(record.get("trade_date"))
        if not record_trade_date:
            continue
        if normalized_target and record_trade_date > normalized_target:
            continue
        if normalized_oldest and record_trade_date < normalized_oldest:
            continue
        if best_trade_date is None or record_trade_date > best_trade_date:
            best_trade_date = record_trade_date
            best_record = record

    if best_record is not None:
        return best_record

    return None


def _sort_frame_by_trade_date_desc(df):
    columns = getattr(df, "columns", [])
    if "trade_date" not in columns or not hasattr(df, "sort_values"):
        return df
    try:
        return df.sort_values(by="trade_date", ascending=False)
    except Exception:
        return df
