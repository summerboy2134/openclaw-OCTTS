from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

from octts.config import Settings
from octts.schemas.backtest import DailyBar
from octts.schemas.report import AnalysisPhase, PriceSnapshot
from octts.services.market_raw_data_repository import MarketRawDataRepository

MAX_RECENT_TRADE_DATE_FALLBACKS = 3
PRO_BAR_EMPTY_RESULT_MESSAGES = (
    "single positional indexer is out-of-bounds",
    "out-of-bounds",
)
logger = logging.getLogger(__name__)
SCREENING_SNAPSHOT_REQUIRED_BASIC_FIELDS = {"ts_code", "close", "total_mv", "turnover_rate", "volume_ratio"}
SCREENING_SNAPSHOT_MIN_BASIC_FIELDS = 2
SCREENING_SNAPSHOT_VERSION = 3
SCREENING_SNAPSHOT_HISTORY_LOOKBACK_DAYS = 120
SCREENING_SNAPSHOT_MIN_HISTORY_ROWS = 20
SCREENING_HISTORY_THROTTLE_MIN_BACKFILL_DATES = 20
SCREENING_SNAPSHOT_CACHE_MODE = "slim_v1"
LATEST_TRADE_DATE_PROBE_CODE = "000001.SZ"
LATEST_TRADE_DATE_PROBE_CACHE_SECONDS = 600
LATEST_TRADE_DATE_PROBE_FAILURE_COOLDOWN_SECONDS = 180


class TushareClient:
    _latest_trade_date_probe_cache: Dict[Tuple[str, str, str], Tuple[float, Optional[str]]] = {}
    _latest_trade_date_probe_failure_until: Dict[Tuple[str, str, str], float] = {}

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
        _client.DataApi._DataApi__http_url = settings.tushare_base_url or "http://tushare.xyz"

        self._safe_set_token(ts, settings.tushare_token)
        self._ts = ts
        self._pro = ts.pro_api(settings.tushare_token)
        try:
            self._pro._DataApi__http_url = settings.tushare_base_url or "http://tushare.xyz"
        except Exception:
            pass
        self._settings = settings
        self._trading_dates_cache: Dict[Tuple[str, str], list[str]] = {}
        self._stock_list_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._raw_data_repo = MarketRawDataRepository(settings.database_url)

    def _safe_set_token(self, ts_module, token: str) -> None:
        try:
            ts_module.set_token(token)
        except Exception as exc:
            # Some wrappers persist the token under the user home directory. When that
            # write is blocked, authenticated requests can still succeed via pro_api.
            if "Operation not permitted" not in str(exc) and "Permission denied" not in str(exc):
                raise

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
        cache_key = (start_date, end_date)
        cached_dates = self._trading_dates_cache.get(cache_key)
        if cached_dates is not None:
            logger.info(
                "Tushare trade_cal cache hit: start_date=%s, end_date=%s, dates=%s",
                start_date,
                end_date,
                len(cached_dates),
            )
            return list(cached_dates)

        local_dates = self._raw_data_repo.list_trading_dates(start_date=start_date, end_date=end_date)
        if local_dates:
            logger.info(
                "Local trade_cal hit: start_date=%s, end_date=%s, dates=%s",
                start_date,
                end_date,
                len(local_dates),
            )
            self._trading_dates_cache[cache_key] = local_dates
            return list(local_dates)

        df = self._pro.trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            is_open="1",
        )
        if df is None or df.empty:
            self._trading_dates_cache[cache_key] = []
            return []
        if "cal_date" not in df.columns:
            self._trading_dates_cache[cache_key] = []
            return []
        rows = df.to_dict(orient="records")
        self._raw_data_repo.save_trade_calendar(rows, exchange="SSE")
        trading_dates = sorted(str(item) for item in df["cal_date"].tolist())
        self._trading_dates_cache[cache_key] = trading_dates
        return list(trading_dates)

    def resolve_latest_trade_date(
        self,
        *,
        end_date: Optional[str] = None,
        lookback_days: int = 7,
        probe_code: str = LATEST_TRADE_DATE_PROBE_CODE,
    ) -> str:
        """Resolve the latest trade date, validating local calendar with a light remote daily probe."""
        target_end_date = end_date or datetime.now().strftime("%Y%m%d")
        anchor_date = datetime.strptime(target_end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(lookback_days, 1))).strftime("%Y%m%d")
        local_dates = self.fetch_trading_dates(start_date=start_date, end_date=target_end_date)
        local_latest = local_dates[-1] if local_dates else None

        remote_latest = self._fetch_latest_daily_trade_date_probe(
            ts_code=probe_code,
            start_date=start_date,
            end_date=target_end_date,
        )
        if remote_latest and (local_latest is None or remote_latest > local_latest):
            logger.warning(
                "Local trade calendar lags remote daily probe: local_latest=%s, remote_latest=%s, probe_code=%s",
                local_latest,
                remote_latest,
                probe_code,
            )
            self._refresh_trade_calendar(start_date=start_date, end_date=target_end_date)
            return remote_latest

        return local_latest or remote_latest or target_end_date

    def _fetch_latest_daily_trade_date_probe(
        self,
        *,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[str]:
        cache_key = (ts_code, start_date, end_date)
        now = time.time()
        cached_probe = self._latest_trade_date_probe_cache.get(cache_key)
        if cached_probe and now - cached_probe[0] <= LATEST_TRADE_DATE_PROBE_CACHE_SECONDS:
            return cached_probe[1]

        failure_until = self._latest_trade_date_probe_failure_until.get(cache_key, 0.0)
        if failure_until > now:
            logger.debug(
                "Latest trade date probe skipped during cooldown: ts_code=%s, start_date=%s, end_date=%s, cooldown_remaining=%.0fs",
                ts_code,
                start_date,
                end_date,
                failure_until - now,
            )
            return None

        try:
            df = self._pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date",
            )
        except Exception as exc:
            self._latest_trade_date_probe_failure_until[cache_key] = (
                now + LATEST_TRADE_DATE_PROBE_FAILURE_COOLDOWN_SECONDS
            )
            logger.warning(
                "Latest trade date probe failed: ts_code=%s, start_date=%s, end_date=%s, cooldown=%ss, error=%s",
                ts_code,
                start_date,
                end_date,
                LATEST_TRADE_DATE_PROBE_FAILURE_COOLDOWN_SECONDS,
                exc,
            )
            return None
        if df is None or df.empty or "trade_date" not in df.columns:
            self._latest_trade_date_probe_cache[cache_key] = (now, None)
            return None
        values = [str(value) for value in df["trade_date"].tolist() if value]
        latest = max(values) if values else None
        self._latest_trade_date_probe_cache[cache_key] = (now, latest)
        self._latest_trade_date_probe_failure_until.pop(cache_key, None)
        return latest

    def _refresh_trade_calendar(self, *, start_date: str, end_date: str) -> None:
        try:
            df = self._pro.trade_cal(
                exchange="SSE",
                start_date=start_date,
                end_date=end_date,
                fields="exchange,cal_date,is_open,pretrade_date",
            )
        except Exception as exc:
            logger.warning(
                "Trade calendar refresh failed: start_date=%s, end_date=%s, error=%s",
                start_date,
                end_date,
                exc,
            )
            return
        if df is None or df.empty:
            return
        rows = df.to_dict(orient="records")
        inserted = self._raw_data_repo.save_trade_calendar(rows, exchange="SSE")
        self._trading_dates_cache.clear()
        logger.info(
            "Trade calendar refreshed from remote: start_date=%s, end_date=%s, rows=%s, inserted=%s",
            start_date,
            end_date,
            len(rows),
            inserted,
        )

    def fetch_daily_bars(self, *, ts_code: str, start_date: str, end_date: str) -> list[DailyBar]:
        rows = self.fetch_daily_batch(ts_codes=[ts_code], start_date=start_date, end_date=end_date).get(ts_code, [])
        records = [DailyBar.model_validate(_serialize_daily_bar(row)) for row in rows]
        records.sort(key=lambda item: item.trade_date)
        return records

    def fetch_daily_data(self, *, ts_code: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Return daily bars as dictionaries for legacy callers."""
        return [bar.model_dump(mode="json") for bar in self.fetch_daily_bars(ts_code=ts_code, start_date=start_date, end_date=end_date)]

    def fetch_stock_info(self, ts_code: str) -> Dict[str, Any]:
        """Fetch basic stock metadata."""
        try:
            df = self._pro.stock_basic(
                ts_code=ts_code,
                fields="ts_code,symbol,name,area,industry,market,list_date",
            )
        except Exception:
            return {"ts_code": ts_code}

        if df is None or df.empty:
            return {"ts_code": ts_code}
        return df.iloc[0].to_dict()

    def fetch_financial_indicators(self, ts_code: str) -> List[Dict[str, Any]]:
        """Fetch recent financial indicators for AI analysis."""
        try:
            df = self._pro.fina_indicator(
                ts_code=ts_code,
                fields=(
                    "ts_code,ann_date,end_date,eps,dt_eps,roe,roe_dt,op_income,op_income_yoy,"
                    "netprofit_yoy,dt_netprofit_yoy,netprofit_margin,grossprofit_margin,assets_turn,bps,ocfps"
                ),
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    def fetch_earnings_express(self, ts_code: str) -> List[Dict[str, Any]]:
        """Fetch recent earnings express rows for fast fundamental updates."""
        try:
            df = self._pro.express(
                ts_code=ts_code,
                fields=(
                    "ts_code,ann_date,end_date,revenue,operate_profit,total_profit,n_income,"
                    "diluted_eps,diluted_roe,yoy_sales,yoy_op,yoy_tp,yoy_dedu_np,yoy_net_profit,bps,perf_summary,update_flag"
                ),
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    def fetch_earnings_forecast(self, ts_code: str) -> List[Dict[str, Any]]:
        """Fetch recent earnings forecast rows for event-driven fundamental signals."""
        try:
            df = self._pro.forecast(
                ts_code=ts_code,
                fields=(
                    "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,"
                    "last_parent_net,first_ann_date,summary,change_reason"
                ),
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    def fetch_income_statements(self, ts_code: str) -> List[Dict[str, Any]]:
        """Fetch recent income statement rows for loss-risk checks."""
        try:
            df = self._pro.income(
                ts_code=ts_code,
                fields="ts_code,end_date,n_income_attr_p,profit_dedt",
            )
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    def fetch_moneyflow(self, ts_code: str, *, trade_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch recent moneyflow records."""
        try:
            if trade_date:
                start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
                end_date = trade_date
                df = self._pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = self._pro.moneyflow(ts_code=ts_code)
        except Exception:
            return []

        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    def fetch_company_profile(self, ts_code: str) -> Dict[str, Any]:
        """Fetch company profile for business summary generation."""
        try:
            df = self._pro.stock_company(
                ts_code=ts_code,
                fields="ts_code,exchange,chairman,manager,secretary,reg_capital,setup_date,province,city,website,email,employees,main_business,business_scope",
            )
        except Exception:
            return {}
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    def _build_snapshot(
        self,
        *,
        ts_code: str,
        phase: AnalysisPhase,
        trade_date: Optional[str],
        include_minute_summary: bool,
    ) -> PriceSnapshot:
        target_trade_date = self._resolve_target_trade_date(trade_date)
        minute_summary = (
            self._fetch_minute_summary(ts_code=ts_code, trade_date=target_trade_date) if include_minute_summary else []
        )
        daily = self._fetch_daily(ts_code=ts_code, trade_date=target_trade_date)
        resolved_trade_date = _normalize_trade_date_value(daily.get("trade_date")) or target_trade_date or trade_date
        intraday_daily = None
        if (
            include_minute_summary
            and target_trade_date == _today_trade_date()
            and resolved_trade_date != target_trade_date
            and minute_summary
        ):
            intraday_daily = _build_intraday_daily_record(
                minute_summary,
                trade_date=target_trade_date,
                previous_close=_safe_float(daily.get("close")),
            )
        if intraday_daily:
            daily = {**daily, **intraday_daily}
            resolved_trade_date = target_trade_date
        elif include_minute_summary and target_trade_date == _today_trade_date() and resolved_trade_date != target_trade_date:
            realtime_daily = self._fetch_realtime_daily(ts_code=ts_code, trade_date=target_trade_date, previous_daily=daily)
            if realtime_daily:
                daily = {**daily, **realtime_daily}
                resolved_trade_date = target_trade_date
        elif include_minute_summary and target_trade_date == _today_trade_date() and resolved_trade_date == target_trade_date:
            minute_summary = []
        daily_basic = self._fetch_daily_basic(ts_code=ts_code, trade_date=resolved_trade_date)
        daily_summary = self._fetch_daily_summary(ts_code=ts_code, trade_date=resolved_trade_date)
        weekly_summary = self._fetch_weekly_summary(ts_code=ts_code, trade_date=resolved_trade_date)
        moneyflow_summary = self._fetch_moneyflow_summary(ts_code=ts_code, trade_date=resolved_trade_date)
        financial_indicators = self.fetch_financial_indicators(ts_code)
        earnings_express = self.fetch_earnings_express(ts_code)
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
            financial_indicators=financial_indicators,
            earnings_express=earnings_express,
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
            local_record = self._raw_data_repo.get_daily(ts_code=ts_code, trade_date=candidate_trade_date)
            if local_record:
                return local_record

        for candidate_trade_date in candidate_trade_dates:
            df = self._pro.daily(ts_code=ts_code, trade_date=candidate_trade_date)
            rows = [] if df is None or getattr(df, "empty", True) else df.to_dict(orient="records")
            if rows:
                self._raw_data_repo.save_daily(rows)
            record = _pick_trade_date_record(
                df,
                target_trade_date=candidate_trade_date,
                oldest_allowed_trade_date=candidate_trade_date,
            )
            if record:
                return record

        end_date = target_trade_date
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(self._settings.default_lookback_days, 20))).strftime("%Y%m%d")
        rows = self.fetch_daily_batch(ts_codes=[ts_code], start_date=start_date, end_date=end_date).get(ts_code, [])
        if rows:
            record = _pick_trade_date_record(
                rows,
                target_trade_date=target_trade_date,
                oldest_allowed_trade_date=oldest_allowed_trade_date,
            )
            if record:
                return record

        raise ValueError(f"No daily market data returned for {ts_code} near {target_trade_date}.")

    def _fetch_daily_basic(self, *, ts_code: str, trade_date: Optional[str]) -> dict[str, Any]:
        normalized_trade_date = _normalize_trade_date_value(trade_date)
        if normalized_trade_date:
            local_record = self._raw_data_repo.get_daily_basic(ts_code=ts_code, trade_date=normalized_trade_date)
            if local_record:
                return local_record

        df = self._pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        rows = [] if df is None or getattr(df, "empty", True) else df.to_dict(orient="records")
        if rows:
            self._raw_data_repo.save_daily_basic(rows)
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
            return []

        records = [row for row in df.to_dict(orient="records")]
        minute_records = _normalize_minute_records(records, trade_date)
        return minute_records

    def _fetch_realtime_daily(
        self, *, ts_code: str, trade_date: str, previous_daily: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        fetch_quote = getattr(getattr(self, "_ts", None), "get_realtime_quotes", None)
        if fetch_quote is None:
            return None
        try:
            df = fetch_quote(_strip_exchange_suffix(ts_code))
        except Exception:
            return None
        if df is None or getattr(df, "empty", True):
            return None
        records = df.to_dict(orient="records")
        if not records:
            return None
        return _build_realtime_daily_record(records[0], trade_date=trade_date, previous_daily=previous_daily)

    def _fetch_daily_summary(self, *, ts_code: str, trade_date: Optional[str]) -> list[dict[str, Any]]:
        end_date = trade_date or datetime.now().strftime("%Y%m%d")
        anchor_date = datetime.strptime(end_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(self._settings.default_lookback_days, 20) * 2)).strftime(
            "%Y%m%d"
        )
        rows = self.fetch_daily_batch(ts_codes=[ts_code], start_date=start_date, end_date=end_date).get(ts_code, [])
        if not rows:
            return []
        return rows[: max(self._settings.default_lookback_days, 20)]

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
            return self._call_pro_bar_without_api(kwargs)
        except Exception as exc:
            if not self._is_empty_pro_bar_error(exc):
                self._log_pro_bar_failure(exc, kwargs)
                raise
            self._log_empty_pro_bar(kwargs)
            return None

    def _call_pro_bar_without_api(self, kwargs: dict[str, Any]):
        try:
            return self._ts.pro_bar(**kwargs)
        except Exception as exc:
            if not self._is_empty_pro_bar_error(exc):
                self._log_pro_bar_failure(exc, kwargs)
                raise
            self._log_empty_pro_bar(kwargs)
            return None

    def _log_pro_bar_failure(self, exc: Exception, kwargs: dict[str, Any]) -> None:
        logger.warning(
            "Tushare pro_bar failed: ts_code=%s, start_date=%s, end_date=%s, freq=%s, adj=%s, error=%s",
            kwargs.get("ts_code"),
            kwargs.get("start_date"),
            kwargs.get("end_date"),
            kwargs.get("freq"),
            kwargs.get("adj"),
            exc,
        )

    def _is_empty_pro_bar_error(self, exc: Exception) -> bool:
        message = str(exc)
        return any(fragment in message for fragment in PRO_BAR_EMPTY_RESULT_MESSAGES)

    def _log_empty_pro_bar(self, kwargs: dict[str, Any]) -> None:
        logger.debug("pro_bar returned no usable rows: %s", kwargs)
        ts_code = kwargs.get("ts_code")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        freq = kwargs.get("freq")
        if ts_code or start_date or end_date:
            logger.info(
                "Tushare pro_bar empty result: ts_code=%s, start_date=%s, end_date=%s, freq=%s",
                ts_code,
                start_date,
                end_date,
                freq,
            )

    def get_or_build_screening_snapshot(self, trade_date: str) -> Dict[str, Any]:
        normalized_trade_date = self._resolve_target_trade_date(trade_date)
        snapshot = self._read_screening_snapshot(normalized_trade_date)
        if snapshot is not None:
            stale_reason = self._screening_snapshot_stale_reason(normalized_trade_date, snapshot)
            if stale_reason is None:
                logger.info("Screening snapshot hit: trade_date=%s", normalized_trade_date)
                return snapshot
            logger.info(stale_reason)

        snapshot_path = self._screening_snapshot_path(normalized_trade_date)
        if snapshot_path.exists():
            logger.info(
                "Screening snapshot rebuild start: trade_date=%s, cache_path=%s, reason=%s",
                normalized_trade_date,
                snapshot_path,
                "stale_or_invalid_cache",
            )
        else:
            logger.info(
                "Screening snapshot build start: trade_date=%s, cache_path=%s, reason=%s",
                normalized_trade_date,
                snapshot_path,
                "cache_miss",
            )
        stocks = self.fetch_stock_list(list_status="L")
        ts_codes = [stock.get("ts_code") for stock in stocks if stock.get("ts_code")]
        daily_basic = self.fetch_daily_basic_batch(ts_codes=ts_codes, trade_date=normalized_trade_date)
        daily = self.fetch_screening_daily_history_batch(ts_codes=ts_codes, trade_date=normalized_trade_date)
        snapshot = {
            "snapshot_version": SCREENING_SNAPSHOT_VERSION,
            "trade_date": normalized_trade_date,
            "created_at": datetime.now().isoformat(),
            "stocks": stocks,
            "daily_basic": daily_basic,
            "daily": daily,
        }
        coverage = self._summarize_screening_daily_coverage(daily.values())
        logger.info(
            "Screening snapshot daily coverage summary: trade_date=%s, ge_2=%s, ge_14=%s, ge_20=%s, ge_60=%s, total_symbols=%s",
            normalized_trade_date,
            coverage["ge_2"],
            coverage["ge_14"],
            coverage["ge_20"],
            coverage["ge_60"],
            coverage["total"],
        )
        invalid_reason = self._screening_snapshot_invalid_reason(snapshot)
        if invalid_reason is None:
            self._write_screening_snapshot(normalized_trade_date, snapshot)
        else:
            logger.warning(
                "Screening snapshot build incomplete: trade_date=%s, reason=%s, skip_cache_write=true, stocks=%s, basic=%s, cached_daily=%s",
                normalized_trade_date,
                invalid_reason,
                len(stocks),
                len(daily_basic),
                len(snapshot["daily"]),
            )
        logger.info(
            "Screening snapshot build complete: trade_date=%s, stocks=%s, basic=%s, cached_daily=%s",
            normalized_trade_date,
            len(stocks),
            len(daily_basic),
            len(snapshot["daily"]),
        )
        return snapshot

    def _screening_snapshot_path(self, trade_date: str) -> Path:
        return Path(self._settings.history_dir_path) / "screening_cache" / f"{trade_date}.json"

    def _read_screening_snapshot(self, trade_date: str) -> Optional[Dict[str, Any]]:
        path = self._screening_snapshot_path(trade_date)
        if not path.exists():
            logger.info("Screening snapshot cache miss: trade_date=%s, cache_path=%s", trade_date, path)
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as exc:
            logger.warning("Screening snapshot read failed: %s (%s)", trade_date, exc)
            return None
        snapshot = self._hydrate_screening_snapshot_from_cache(trade_date, snapshot)
        invalid_reason = self._screening_snapshot_invalid_reason(snapshot)
        if invalid_reason is not None:
            logger.warning(
                "Screening snapshot invalid: trade_date=%s, reason=%s, rebuild=true",
                trade_date,
                invalid_reason,
            )
            return None
        return snapshot

    def _hydrate_screening_snapshot_from_cache(self, trade_date: str, snapshot: Any) -> Any:
        if not isinstance(snapshot, dict):
            return snapshot
        daily = snapshot.get("daily")
        if isinstance(daily, dict) and daily:
            return snapshot
        if snapshot.get("cache_mode") != SCREENING_SNAPSHOT_CACHE_MODE:
            return snapshot
        stocks = snapshot.get("stocks")
        if not isinstance(stocks, list):
            return snapshot
        ts_codes = [stock.get("ts_code") for stock in stocks if isinstance(stock, dict) and stock.get("ts_code")]
        if not ts_codes:
            return snapshot
        hydrated = dict(snapshot)
        hydrated["daily"] = self.fetch_screening_daily_history_batch(ts_codes=ts_codes, trade_date=trade_date)
        logger.info(
            "Screening snapshot hydrated from slim cache: trade_date=%s, stocks=%s, cached_daily=%s",
            trade_date,
            len(ts_codes),
            len(hydrated.get("daily") or {}),
        )
        return hydrated

    def _write_screening_snapshot(self, trade_date: str, snapshot: Dict[str, Any]) -> None:
        path = self._screening_snapshot_path(trade_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        persisted_snapshot = self._build_persisted_screening_snapshot(snapshot)
        with path.open("w", encoding="utf-8") as f:
            json.dump(persisted_snapshot, f, ensure_ascii=False)
        logger.info(
            "Screening snapshot cache write complete: trade_date=%s, cache_path=%s, stocks=%s, basic=%s, cached_daily=%s, cache_mode=%s",
            trade_date,
            path,
            len(snapshot.get("stocks", [])) if isinstance(snapshot.get("stocks"), list) else 0,
            len(snapshot.get("daily_basic", {})) if isinstance(snapshot.get("daily_basic"), dict) else 0,
            len(snapshot.get("daily", {})) if isinstance(snapshot.get("daily"), dict) else 0,
            persisted_snapshot.get("cache_mode"),
        )

    def _build_persisted_screening_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        daily = snapshot.get("daily") if isinstance(snapshot.get("daily"), dict) else {}
        daily_index = {
            ts_code: {
                "rows": len(rows),
                "start_date": str(rows[-1].get("trade_date") or "") if rows else None,
                "end_date": str(rows[0].get("trade_date") or "") if rows else None,
            }
            for ts_code, rows in daily.items()
            if isinstance(rows, list)
        }
        return {
            "snapshot_version": snapshot.get("snapshot_version"),
            "cache_mode": SCREENING_SNAPSHOT_CACHE_MODE,
            "trade_date": snapshot.get("trade_date"),
            "created_at": snapshot.get("created_at"),
            "stocks": snapshot.get("stocks") or [],
            "daily_basic": snapshot.get("daily_basic") or {},
            "daily_index": daily_index,
            "daily_coverage": self._summarize_screening_daily_coverage(daily.values()),
        }

    def _screening_snapshot_invalid_reason(self, snapshot: Any) -> Optional[str]:
        if not isinstance(snapshot, dict):
            return "snapshot_not_dict"

        if snapshot.get("snapshot_version") != SCREENING_SNAPSHOT_VERSION:
            return "snapshot_version_mismatch"

        stocks = snapshot.get("stocks")
        if not isinstance(stocks, list) or not stocks:
            return "stocks_missing"

        stock_codes = [stock.get("ts_code") for stock in stocks if isinstance(stock, dict) and stock.get("ts_code")]
        if not stock_codes:
            return "stocks_missing_ts_code"

        daily_basic = snapshot.get("daily_basic")
        if not isinstance(daily_basic, dict) or not daily_basic:
            return "daily_basic_missing"

        matched_basics = []
        for ts_code in stock_codes:
            basic = daily_basic.get(ts_code)
            if isinstance(basic, dict):
                matched_basics.append(basic)

        if not matched_basics:
            return "daily_basic_no_matching_ts_code"

        has_complete_basic = any(self._screening_snapshot_basic_complete(basic) for basic in matched_basics)
        if not has_complete_basic:
            return "daily_basic_missing_required_fields"

        daily = snapshot.get("daily")
        if not isinstance(daily, dict):
            return "daily_missing"
        matched_daily = [daily.get(ts_code) for ts_code in stock_codes if isinstance(daily.get(ts_code), list)]
        if not matched_daily:
            return "daily_no_matching_ts_code"
        coverage = self._summarize_screening_daily_coverage(matched_daily)
        if coverage["ge_14"] == 0:
            return "daily_history_insufficient_for_rsi"
        if coverage["ge_20"] == 0:
            return "daily_history_insufficient_for_price_position"
        if coverage["ge_20"] < max(50, min(len(stock_codes), len(stock_codes) // 5 or 1)):
            return "daily_history_coverage_too_shallow"

        return None

    def _screening_snapshot_basic_complete(self, basic: Dict[str, Any]) -> bool:
        fields = {key for key, value in basic.items() if value is not None and key in SCREENING_SNAPSHOT_REQUIRED_BASIC_FIELDS}
        return len(fields) >= SCREENING_SNAPSHOT_MIN_BASIC_FIELDS

    @staticmethod
    def _summarize_screening_daily_coverage(rows_iterable: Any) -> Dict[str, int]:
        counts = {
            "total": 0,
            "ge_2": 0,
            "ge_14": 0,
            "ge_20": 0,
            "ge_60": 0,
        }
        for rows in rows_iterable:
            if not isinstance(rows, list):
                continue
            row_count = len(rows)
            counts["total"] += 1
            if row_count >= 2:
                counts["ge_2"] += 1
            if row_count >= 14:
                counts["ge_14"] += 1
            if row_count >= 20:
                counts["ge_20"] += 1
            if row_count >= 60:
                counts["ge_60"] += 1
        return counts

    def _screening_snapshot_stale_reason(self, trade_date: str, snapshot: Dict[str, Any]) -> Optional[str]:
        if trade_date != _today_trade_date():
            return None
        now = datetime.now()
        close_cutoff = datetime.combine(now.date(), dt_time(hour=16))
        if now < close_cutoff:
            return None
        snapshot_time = self._screening_snapshot_created_at(trade_date, snapshot)
        if snapshot_time is None or snapshot_time >= close_cutoff:
            return None
        return (
            "Screening snapshot stale after close: trade_date=%s, snapshot_time=%s, close_cutoff=%s, rebuild=true"
            % (trade_date, snapshot_time.isoformat(), close_cutoff.isoformat())
        )

    def _screening_snapshot_created_at(self, trade_date: str, snapshot: Dict[str, Any]) -> Optional[datetime]:
        created_at = snapshot.get("created_at")
        if isinstance(created_at, str):
            try:
                return datetime.fromisoformat(created_at)
            except ValueError:
                pass
        path = self._screening_snapshot_path(trade_date)
        if not path.exists():
            return None
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None

    # 批量数据获取方法 - 用于股票筛选功能
    def fetch_stock_list(self, *, list_status: str = "L") -> List[Dict[str, Any]]:
        """
        获取股票列表

        Args:
            list_status: 上市状态 L上市 D退市 P暂停上市

        Returns:
            股票列表
        """
        cached_rows = self._stock_list_cache.get(list_status)
        if cached_rows is not None:
            logger.info("Tushare stock list cache hit: list_status=%s, rows=%s", list_status, len(cached_rows))
            return [dict(item) for item in cached_rows]

        started_at = time.time()
        logger.info("Tushare fetch_stock_list start: list_status=%s", list_status)
        try:
            df = self._pro.stock_basic(
                exchange="",
                list_status=list_status,
                fields="ts_code,symbol,name,area,industry,market,list_date"
            )
            if df is None or df.empty:
                logger.info("Tushare fetch_stock_list complete: 0 rows in %.2fs", time.time() - started_at)
                self._stock_list_cache[list_status] = []
                return []
            rows = df.to_dict(orient="records")
            self._stock_list_cache[list_status] = rows
            logger.info("Tushare fetch_stock_list complete: %s rows in %.2fs", len(rows), time.time() - started_at)
            return [dict(item) for item in rows]
        except Exception as exc:
            logger.error("Tushare fetch_stock_list failed after %.2fs: %s", time.time() - started_at, exc)
            return []

    def _throttle_screening_history_fetch_loop(
        self,
        *,
        fetch_name: str,
        processed_count: int,
        total_count: int,
    ) -> None:
        if processed_count <= 0:
            return
        if total_count < SCREENING_HISTORY_THROTTLE_MIN_BACKFILL_DATES:
            return
        sleep_every = max(int(self._settings.screening_history_throttle_every), 0)
        batch_sleep_seconds = max(float(self._settings.screening_history_throttle_sleep_seconds), 0.0)
        if sleep_every <= 0 or batch_sleep_seconds <= 0:
            return
        if processed_count % sleep_every != 0:
            return
        logger.info(
            "Screening history throttle: fetch=%s, processed_trade_dates=%s/%s, sleep_seconds=%.2f",
            fetch_name,
            processed_count,
            total_count,
            batch_sleep_seconds,
        )
        time.sleep(batch_sleep_seconds)

    def fetch_daily_basic_batch(
        self,
        *,
        ts_codes: List[str],
        trade_date: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取日线基础指标

        Args:
            ts_codes: 股票代码列表
            trade_date: 交易日期

        Returns:
            {ts_code: daily_basic_data} 的字典
        """
        started_at = time.time()
        logger.info(
            "Tushare fetch_daily_basic_batch start: %s symbols, trade_date=%s",
            len(ts_codes),
            trade_date,
        )
        local_result = self._raw_data_repo.get_daily_basic_batch_for_trade_date(ts_codes=ts_codes, trade_date=trade_date)
        result = dict(local_result)
        if len(result) == len(ts_codes):
            logger.info(
                "Local daily_basic batch hit: %s/%s symbols in %.2fs",
                len(result),
                len(ts_codes),
                time.time() - started_at,
            )
            return result

        candidate_trade_dates = self._candidate_trade_dates(target_trade_date=trade_date)
        batch_request_failures = 0
        fallback_attempts = 0
        fallback_hits = 0

        for candidate_trade_date in candidate_trade_dates:
            missing_codes = [ts_code for ts_code in ts_codes if ts_code not in result]
            if not missing_codes:
                break

            try:
                df = self._pro.daily_basic(
                    trade_date=candidate_trade_date,
                    fields="ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
                )
                if df is not None and not df.empty:
                    rows = df.to_dict(orient="records")
                    self._raw_data_repo.save_daily_basic(rows)
                    for row in rows:
                        ts_code = row.get("ts_code")
                        if ts_code in missing_codes and ts_code not in result:
                            result[ts_code] = row
            except Exception as exc:
                batch_request_failures += 1
                logger.warning(
                    "Tushare fetch_daily_basic_batch request failed: trade_date=%s, candidate_trade_date=%s, mode=trade_date_scan, error=%s",
                    trade_date,
                    candidate_trade_date,
                    exc,
                )

        missing_codes = [ts_code for ts_code in ts_codes if ts_code not in result]
        if missing_codes:
            logger.info(
                "Tushare fetch_daily_basic_batch fallback start: trade_date=%s, missing_symbols=%s",
                trade_date,
                len(missing_codes),
            )
        for ts_code in missing_codes:
            for candidate_trade_date in candidate_trade_dates:
                fallback_attempts += 1
                try:
                    df = self._pro.daily_basic(ts_code=ts_code, trade_date=candidate_trade_date)
                    if df is not None and not df.empty:
                        rows = df.to_dict(orient="records")
                        self._raw_data_repo.save_daily_basic(rows)
                        result[ts_code] = rows[0]
                        fallback_hits += 1
                        break
                except Exception:
                    continue

        logger.info(
            "Tushare fetch_daily_basic_batch complete: %s/%s symbols in %.2fs, batch_failures=%s, fallback_attempts=%s, fallback_hits=%s",
            len(result),
            len(ts_codes),
            time.time() - started_at,
            batch_request_failures,
            fallback_attempts,
            fallback_hits,
        )
        return result

    def fetch_daily_trade_date_batch(
        self,
        *,
        ts_codes: List[str],
        trade_date: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取指定交易日的日线数据

        Args:
            ts_codes: 股票代码列表
            trade_date: 交易日期

        Returns:
            {ts_code: [daily_data]} 的字典
        """
        started_at = time.time()
        logger.info(
            "Tushare fetch_daily_trade_date_batch start: %s symbols, trade_date=%s",
            len(ts_codes),
            trade_date,
        )
        result = self._raw_data_repo.get_daily_batch_for_trade_date(ts_codes=ts_codes, trade_date=trade_date)
        if not ts_codes:
            return result
        if sum(1 for items in result.values() if items) == len(ts_codes):
            logger.info(
                "Local daily trade-date batch hit: %s/%s symbols in %.2fs",
                len(ts_codes),
                len(ts_codes),
                time.time() - started_at,
            )
            return result

        candidate_trade_dates = self._candidate_trade_dates(target_trade_date=trade_date)
        remaining_codes = {ts_code for ts_code in ts_codes if not result.get(ts_code)}
        for candidate_trade_date in candidate_trade_dates:
            if not remaining_codes:
                break
            try:
                df = self._pro.daily(trade_date=candidate_trade_date)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            rows = _sort_frame_by_trade_date_desc(df).to_dict(orient="records")
            self._raw_data_repo.save_daily(rows)
            for row in rows:
                ts_code = row.get("ts_code")
                if ts_code in remaining_codes:
                    result[ts_code] = [row]
                    remaining_codes.remove(ts_code)

        non_empty = sum(1 for items in result.values() if items)
        logger.info(
            "Tushare fetch_daily_trade_date_batch complete: %s/%s symbols in %.2fs",
            non_empty,
            len(ts_codes),
            time.time() - started_at,
        )
        return result

    def fetch_screening_daily_history_batch(
        self,
        *,
        ts_codes: List[str],
        trade_date: str,
        lookback_days: int = SCREENING_SNAPSHOT_HISTORY_LOOKBACK_DAYS,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not ts_codes:
            return {}
        anchor_date = datetime.strptime(trade_date, "%Y%m%d")
        start_date = (anchor_date - timedelta(days=max(lookback_days, 1))).strftime("%Y%m%d")
        trading_dates = self.fetch_trading_dates(start_date=start_date, end_date=trade_date)
        if not trading_dates:
            return {ts_code: [] for ts_code in ts_codes}

        target_codes = set(ts_codes)
        raw_daily_rows = self._fetch_screening_daily_rows_by_trade_dates(
            ts_codes=target_codes,
            trading_dates=trading_dates,
        )
        raw_daily_basic_rows = self._fetch_screening_daily_basic_rows_by_trade_dates(
            ts_codes=target_codes,
            trading_dates=trading_dates,
        )
        adj_factor_rows = self._fetch_screening_adj_factor_rows_by_trade_dates(
            ts_codes=target_codes,
            trading_dates=trading_dates,
        )
        history = self._build_screening_qfq_history(
            ts_codes=ts_codes,
            trading_dates=trading_dates,
            raw_daily_rows=raw_daily_rows,
            adj_factor_rows=adj_factor_rows,
        )
        logger.info(
            "Screening qfq history coverage: retained_symbols=%s/%s, retained_rows=%s",
            sum(1 for rows in history.values() if rows),
            len(ts_codes),
            sum(len(rows) for rows in history.values()),
        )

        enriched_history: Dict[str, List[Dict[str, Any]]] = {}
        daily_basic_map = self.fetch_daily_basic_batch(ts_codes=ts_codes, trade_date=trade_date)
        for ts_code in ts_codes:
            rows = history.get(ts_code) or []
            latest_basic = daily_basic_map.get(ts_code) or {}
            normalized_rows: List[Dict[str, Any]] = []
            for index, row in enumerate(rows):
                merged_row = dict(row)
                trade_date_value = str(merged_row.get("trade_date") or "")
                historical_basic = raw_daily_basic_rows.get(ts_code, {}).get(trade_date_value, {})
                for field in ("turnover_rate", "volume_ratio", "close"):
                    historical_value = historical_basic.get(field)
                    if historical_value is not None and merged_row.get(field) is None:
                        merged_row[field] = historical_value
                if index == 0:
                    turnover_value = latest_basic.get("turnover_rate")
                    volume_ratio_value = latest_basic.get("volume_ratio")
                    if turnover_value is not None and merged_row.get("turnover_rate") is None:
                        merged_row["turnover_rate"] = turnover_value
                    if volume_ratio_value is not None and merged_row.get("volume_ratio") is None:
                        merged_row["volume_ratio"] = volume_ratio_value
                normalized_rows.append(merged_row)
            enriched_history[ts_code] = normalized_rows
        return enriched_history

    def _fetch_screening_daily_basic_rows_by_trade_dates(
        self,
        *,
        ts_codes: set[str],
        trading_dates: List[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        rows_by_code: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        local_rows = self._raw_data_repo.get_daily_basic_by_trade_dates(ts_codes=ts_codes, trading_dates=trading_dates)
        for ts_code, rows in local_rows.items():
            if rows:
                rows_by_code[ts_code].update(rows)

        missing_trade_dates = [
            trade_date for trade_date in trading_dates
            if any(trade_date not in rows_by_code.get(ts_code, {}) for ts_code in ts_codes)
        ]
        matched_symbols_by_date: Dict[str, int] = {}
        total_rows_by_date: Dict[str, int] = {}
        total_trade_dates = len(trading_dates)
        for index, trade_date in enumerate(missing_trade_dates, start=1):
            try:
                df = self._pro.daily_basic(
                    trade_date=trade_date,
                    fields="ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv",
                )
            except Exception as exc:
                logger.warning("Screening daily_basic batch fetch failed: trade_date=%s, error=%s", trade_date, exc)
                self._throttle_screening_history_fetch_loop(
                    fetch_name="daily_basic",
                    processed_count=index,
                    total_count=max(total_trade_dates, len(missing_trade_dates)),
                )
                continue
            if df is None or df.empty:
                matched_symbols_by_date[trade_date] = 0
                total_rows_by_date[trade_date] = 0
                self._throttle_screening_history_fetch_loop(
                    fetch_name="daily_basic",
                    processed_count=index,
                    total_count=max(total_trade_dates, len(missing_trade_dates)),
                )
                continue
            rows = df.to_dict(orient="records")
            self._raw_data_repo.save_daily_basic(rows)
            total_rows_by_date[trade_date] = len(rows)
            matched_count = 0
            for row in rows:
                ts_code = row.get("ts_code")
                if ts_code in ts_codes:
                    rows_by_code[ts_code][trade_date] = row
                    matched_count += 1
            matched_symbols_by_date[trade_date] = matched_count
            self._throttle_screening_history_fetch_loop(
                fetch_name="daily_basic",
                processed_count=index,
                total_count=max(total_trade_dates, len(missing_trade_dates)),
            )
        matched_symbols = sum(1 for rows in rows_by_code.values() if rows)
        latest_trade_date = trading_dates[-1] if trading_dates else None
        if latest_trade_date is not None:
            logger.info(
                "Screening daily_basic history coverage: latest_trade_date=%s, matched_symbols=%s/%s, matched_rows=%s, total_rows=%s",
                latest_trade_date,
                sum(1 for ts_code in ts_codes if latest_trade_date in rows_by_code.get(ts_code, {})),
                len(ts_codes),
                matched_symbols_by_date.get(latest_trade_date, sum(1 for ts_code in ts_codes if latest_trade_date in rows_by_code.get(ts_code, {}))),
                total_rows_by_date.get(latest_trade_date, 0),
            )
        logger.info(
            "Screening daily_basic history aggregate coverage: trading_dates=%s, symbols_with_any_rows=%s/%s",
            len(trading_dates),
            matched_symbols,
            len(ts_codes),
        )
        return rows_by_code

    def _fetch_screening_daily_rows_by_trade_dates(
        self,
        *,
        ts_codes: set[str],
        trading_dates: List[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        rows_by_code: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        local_rows = self._raw_data_repo.get_daily_by_trade_dates(ts_codes=ts_codes, trading_dates=trading_dates)
        for ts_code, rows in local_rows.items():
            if rows:
                rows_by_code[ts_code].update(rows)

        missing_trade_dates = [
            trade_date for trade_date in trading_dates
            if any(trade_date not in rows_by_code.get(ts_code, {}) for ts_code in ts_codes)
        ]
        matched_symbols_by_date: Dict[str, int] = {}
        total_rows_by_date: Dict[str, int] = {}
        total_trade_dates = len(trading_dates)
        for index, trade_date in enumerate(missing_trade_dates, start=1):
            try:
                df = self._pro.daily(trade_date=trade_date)
            except Exception as exc:
                logger.warning("Screening daily batch fetch failed: trade_date=%s, error=%s", trade_date, exc)
                self._throttle_screening_history_fetch_loop(
                    fetch_name="daily",
                    processed_count=index,
                    total_count=max(total_trade_dates, len(missing_trade_dates)),
                )
                continue
            if df is None or df.empty:
                matched_symbols_by_date[trade_date] = 0
                total_rows_by_date[trade_date] = 0
                self._throttle_screening_history_fetch_loop(
                    fetch_name="daily",
                    processed_count=index,
                    total_count=max(total_trade_dates, len(missing_trade_dates)),
                )
                continue
            rows = _sort_frame_by_trade_date_desc(df).to_dict(orient="records")
            self._raw_data_repo.save_daily(rows)
            total_rows_by_date[trade_date] = len(rows)
            matched_count = 0
            for row in rows:
                ts_code = row.get("ts_code")
                if ts_code in ts_codes:
                    rows_by_code[ts_code][trade_date] = row
                    matched_count += 1
            matched_symbols_by_date[trade_date] = matched_count
            self._throttle_screening_history_fetch_loop(
                fetch_name="daily",
                processed_count=index,
                total_count=max(total_trade_dates, len(missing_trade_dates)),
            )
        matched_symbols = sum(1 for rows in rows_by_code.values() if rows)
        latest_trade_date = trading_dates[-1] if trading_dates else None
        if latest_trade_date is not None:
            logger.info(
                "Screening daily batch coverage: latest_trade_date=%s, matched_symbols=%s/%s, matched_rows=%s, total_rows=%s",
                latest_trade_date,
                sum(1 for ts_code in ts_codes if latest_trade_date in rows_by_code.get(ts_code, {})),
                len(ts_codes),
                matched_symbols_by_date.get(latest_trade_date, sum(1 for ts_code in ts_codes if latest_trade_date in rows_by_code.get(ts_code, {}))),
                total_rows_by_date.get(latest_trade_date, 0),
            )
        logger.info(
            "Screening daily batch aggregate coverage: trading_dates=%s, symbols_with_any_rows=%s/%s",
            len(trading_dates),
            matched_symbols,
            len(ts_codes),
        )
        return rows_by_code

    def _fetch_screening_adj_factor_rows_by_trade_dates(
        self,
        *,
        ts_codes: set[str],
        trading_dates: List[str],
    ) -> Dict[str, Dict[str, float]]:
        factor_by_code: Dict[str, Dict[str, float]] = defaultdict(dict)
        local_rows = self._raw_data_repo.get_adj_factors_by_trade_dates(ts_codes=ts_codes, trading_dates=trading_dates)
        for ts_code, rows in local_rows.items():
            if rows:
                factor_by_code[ts_code].update(rows)

        missing_trade_dates = [
            trade_date for trade_date in trading_dates
            if any(trade_date not in factor_by_code.get(ts_code, {}) for ts_code in ts_codes)
        ]
        matched_symbols_by_date: Dict[str, int] = {}
        total_rows_by_date: Dict[str, int] = {}
        total_trade_dates = len(trading_dates)
        for index, trade_date in enumerate(missing_trade_dates, start=1):
            try:
                df = self._pro.adj_factor(trade_date=trade_date)
            except Exception as exc:
                logger.warning("Screening adj_factor batch fetch failed: trade_date=%s, error=%s", trade_date, exc)
                self._throttle_screening_history_fetch_loop(
                    fetch_name="adj_factor",
                    processed_count=index,
                    total_count=max(total_trade_dates, len(missing_trade_dates)),
                )
                continue
            if df is None or df.empty:
                matched_symbols_by_date[trade_date] = 0
                total_rows_by_date[trade_date] = 0
                self._throttle_screening_history_fetch_loop(
                    fetch_name="adj_factor",
                    processed_count=index,
                    total_count=max(total_trade_dates, len(missing_trade_dates)),
                )
                continue
            rows = df.to_dict(orient="records")
            self._raw_data_repo.save_adj_factor(rows)
            total_rows_by_date[trade_date] = len(rows)
            matched_count = 0
            for row in rows:
                ts_code = row.get("ts_code")
                if ts_code not in ts_codes:
                    continue
                adj_factor = _safe_float(row.get("adj_factor"))
                if adj_factor is not None:
                    factor_by_code[ts_code][trade_date] = adj_factor
                    matched_count += 1
            matched_symbols_by_date[trade_date] = matched_count
            self._throttle_screening_history_fetch_loop(
                fetch_name="adj_factor",
                processed_count=index,
                total_count=max(total_trade_dates, len(missing_trade_dates)),
            )
        matched_symbols = sum(1 for rows in factor_by_code.values() if rows)
        latest_trade_date = trading_dates[-1] if trading_dates else None
        if latest_trade_date is not None:
            logger.info(
                "Screening adj_factor batch coverage: latest_trade_date=%s, matched_symbols=%s/%s, matched_rows=%s, total_rows=%s",
                latest_trade_date,
                sum(1 for ts_code in ts_codes if latest_trade_date in factor_by_code.get(ts_code, {})),
                len(ts_codes),
                matched_symbols_by_date.get(latest_trade_date, sum(1 for ts_code in ts_codes if latest_trade_date in factor_by_code.get(ts_code, {}))),
                total_rows_by_date.get(latest_trade_date, 0),
            )
        logger.info(
            "Screening adj_factor aggregate coverage: trading_dates=%s, symbols_with_any_rows=%s/%s",
            len(trading_dates),
            matched_symbols,
            len(ts_codes),
        )
        return factor_by_code

    def _build_screening_qfq_history(
        self,
        *,
        ts_codes: List[str],
        trading_dates: List[str],
        raw_daily_rows: Dict[str, Dict[str, Dict[str, Any]]],
        adj_factor_rows: Dict[str, Dict[str, float]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        history: Dict[str, List[Dict[str, Any]]] = {ts_code: [] for ts_code in ts_codes}

        for ts_code in ts_codes:
            factor_map = adj_factor_rows.get(ts_code) or {}
            latest_factor = _pick_latest_available_adj_factor(factor_map, trading_dates)
            if latest_factor is None or latest_factor == 0:
                continue

            normalized_rows: List[Dict[str, Any]] = []
            seen_trade_dates: set[str] = set()
            for trade_date in reversed(trading_dates):
                raw_row = (raw_daily_rows.get(ts_code) or {}).get(trade_date)
                if not isinstance(raw_row, dict) or trade_date in seen_trade_dates:
                    continue
                row_factor = _pick_row_adj_factor(factor_map, trading_dates, trade_date)
                normalized_row = _normalize_screening_history_row(
                    raw_row=raw_row,
                    trade_date=trade_date,
                    current_adj_factor=latest_factor,
                    row_adj_factor=row_factor,
                )
                if normalized_row is None:
                    continue
                seen_trade_dates.add(trade_date)
                normalized_rows.append(normalized_row)
            history[ts_code] = normalized_rows
        return history

    def fetch_daily_batch(
        self,
        *,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取日线数据

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            {ts_code: [daily_data]} 的字典
        """
        started_at = time.time()
        logger.info(
            "Tushare fetch_daily_batch start: %s symbols, %s -> %s",
            len(ts_codes),
            start_date,
            end_date,
        )
        result = {ts_code: self._raw_data_repo.get_daily_range(ts_code=ts_code, start_date=start_date, end_date=end_date) for ts_code in ts_codes}
        missing_codes = [ts_code for ts_code, rows in result.items() if not rows]

        batch_size = 50
        for i in range(0, len(missing_codes), batch_size):
            batch_codes = missing_codes[i:i + batch_size]

            for ts_code in batch_codes:
                try:
                    df = self._call_pro_bar(
                        ts_code=ts_code,
                        asset="E",
                        start_date=start_date,
                        end_date=end_date,
                        freq="D",
                        adj="qfq"
                    )
                except Exception as exc:
                    logger.warning(
                        "Tushare fetch_daily_batch skip symbol after pro_bar failure: ts_code=%s, start_date=%s, end_date=%s, error=%s",
                        ts_code,
                        start_date,
                        end_date,
                        exc,
                    )
                    continue
                if df is None or df.empty:
                    logger.info(
                        "Tushare fetch_daily_batch empty symbol: ts_code=%s, start_date=%s, end_date=%s",
                        ts_code,
                        start_date,
                        end_date,
                    )
                    continue
                daily_rows = _sort_frame_by_trade_date_desc(df).to_dict(orient="records")
                raw_rows = [_build_raw_daily_row_from_qfq_row(row) for row in daily_rows]
                if raw_rows:
                    self._raw_data_repo.save_daily(raw_rows)
                result[ts_code] = daily_rows

        non_empty = sum(1 for items in result.values() if items)
        if non_empty == 0 and ts_codes:
            logger.warning(
                "Tushare fetch_daily_batch returned no data: symbols=%s, start_date=%s, end_date=%s",
                len(ts_codes),
                start_date,
                end_date,
            )
        logger.info(
            "Tushare fetch_daily_batch complete: %s/%s symbols with data in %.2fs",
            non_empty,
            len(ts_codes),
            time.time() - started_at,
        )
        return result


def _build_raw_daily_row_from_qfq_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts_code": row.get("ts_code"),
        "trade_date": row.get("trade_date"),
        "open": _safe_float(row.get("open")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),
        "close": _safe_float(row.get("close")),
        "pre_close": _safe_float(row.get("pre_close")),
        "change": _safe_float(row.get("change")),
        "pct_chg": _safe_float(row.get("pct_chg")),
        "vol": _safe_float(row.get("vol")),
        "amount": _safe_float(row.get("amount")),
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


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


def _normalize_minute_records(records: list[dict[str, Any]], trade_date: str) -> list[dict[str, Any]]:
    normalized: list[tuple[tuple[int, str], dict[str, Any]]] = []
    for record in records:
        record_trade_date = _extract_minute_trade_date(record, fallback_trade_date=trade_date)
        if record_trade_date != trade_date:
            continue
        sort_key = _minute_sort_key(record)
        normalized.append((sort_key, record))
    normalized.sort(key=lambda item: item[0])
    return [record for _, record in normalized]


def _build_intraday_daily_record(
    minute_records: list[dict[str, Any]], *, trade_date: str, previous_close: Optional[float]
) -> Optional[dict[str, Any]]:
    records = _normalize_minute_records(minute_records, trade_date)
    if not records:
        return None

    first = records[0]
    last = records[-1]
    first_open = _safe_float(first.get("open"))
    fallback_open = _safe_float(first.get("price")) or _safe_float(first.get("close")) or _safe_float(first.get("close_price"))
    close = _safe_float(last.get("close")) or _safe_float(last.get("price")) or _safe_float(last.get("close_price"))
    open_price = first_open if first_open is not None else fallback_open

    high_candidates = [_safe_float(item.get("high")) for item in records]
    low_candidates = [_safe_float(item.get("low")) for item in records]
    price_candidates = [
        _safe_float(item.get("price")) or _safe_float(item.get("close")) or _safe_float(item.get("close_price"))
        for item in records
    ]
    high = _max_defined(high_candidates + price_candidates)
    low = _min_defined(low_candidates + price_candidates)
    amount_values = [_safe_float(item.get("amount")) for item in records]
    amount = _sum_defined(amount_values)
    pct_chg = None
    if previous_close not in (None, 0) and close is not None:
        pct_chg = ((close - previous_close) / previous_close) * 100

    return {
        "trade_date": trade_date,
        "open": open_price,
        "close": close,
        "high": high,
        "low": low,
        "amount": amount,
        "pct_chg": pct_chg,
    }


def _build_realtime_daily_record(
    record: dict[str, Any], *, trade_date: str, previous_daily: Optional[dict[str, Any]] = None
) -> Optional[dict[str, Any]]:
    record_trade_date = _extract_realtime_trade_date(record)
    if record_trade_date != trade_date:
        return None

    previous_close = _safe_float(record.get("pre_close"))
    if previous_close is None and previous_daily is not None:
        previous_close = _safe_float(previous_daily.get("close"))
    close = _safe_float(record.get("price"))
    open_price = _safe_float(record.get("open"))
    high = _safe_float(record.get("high"))
    low = _safe_float(record.get("low"))
    amount = _safe_float(record.get("amount"))
    pct_chg = None
    if previous_close not in (None, 0) and close is not None:
        pct_chg = ((close - previous_close) / previous_close) * 100

    return {
        "trade_date": trade_date,
        "open": open_price,
        "close": close,
        "high": high,
        "low": low,
        "amount": amount / 1000 if amount is not None else None,
        "pct_chg": pct_chg,
    }


def _extract_minute_trade_date(record: dict[str, Any], fallback_trade_date: Optional[str] = None) -> Optional[str]:
    value = record.get("trade_date")
    normalized = _normalize_trade_date_value(value)
    if normalized:
        return normalized
    timestamp = _extract_minute_timestamp(record)
    if not timestamp:
        return None
    digits = "".join(ch for ch in timestamp if ch.isdigit())
    if len(digits) >= 8:
        candidate = digits[:8]
        return candidate if len(candidate) == 8 else None
    if fallback_trade_date and _looks_like_intraday_time(timestamp):
        return fallback_trade_date
    return None


def _extract_realtime_trade_date(record: dict[str, Any]) -> Optional[str]:
    date_text = str(record.get("date") or "").strip()
    if not date_text:
        return None
    digits = "".join(ch for ch in date_text if ch.isdigit())
    return digits if len(digits) == 8 else None


def _extract_minute_timestamp(record: dict[str, Any]) -> Optional[str]:
    for key in ("trade_time", "trade_datetime", "datetime", "time"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _minute_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    timestamp = _extract_minute_timestamp(record)
    if not timestamp:
        return (1, "")
    return (0, timestamp)


def _looks_like_intraday_time(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if ":" in text:
        return True
    digits = "".join(ch for ch in text if ch.isdigit())
    return len(digits) in {4, 6}


def _strip_exchange_suffix(ts_code: str) -> str:
    return ts_code.split(".", maxsplit=1)[0].strip()


def _max_defined(values: list[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _min_defined(values: list[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _sum_defined(values: list[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered)


def _pick_latest_available_adj_factor(factor_map: Dict[str, float], trading_dates: List[str]) -> Optional[float]:
    for trade_date in reversed(trading_dates):
        factor = _safe_float(factor_map.get(trade_date))
        if factor is not None and factor != 0:
            return factor
    return None


def _pick_row_adj_factor(factor_map: Dict[str, float], trading_dates: List[str], trade_date: str) -> Optional[float]:
    if not trading_dates:
        return None
    try:
        index = trading_dates.index(trade_date)
    except ValueError:
        return None
    for candidate_trade_date in reversed(trading_dates[: index + 1]):
        factor = _safe_float(factor_map.get(candidate_trade_date))
        if factor is not None and factor != 0:
            return factor
    return None


def _normalize_screening_history_row(
    *,
    raw_row: Dict[str, Any],
    trade_date: str,
    current_adj_factor: float,
    row_adj_factor: Optional[float],
) -> Optional[Dict[str, Any]]:
    close = _safe_float(raw_row.get("close"))
    row_factor = _safe_float(row_adj_factor)
    if close is None or row_factor is None or row_factor == 0:
        return None

    qfq_ratio = current_adj_factor / row_factor
    high = _safe_float(raw_row.get("high"))
    low = _safe_float(raw_row.get("low"))
    vol = _safe_float(raw_row.get("vol"))
    amount = _safe_float(raw_row.get("amount"))
    pct_chg = _safe_float(raw_row.get("pct_chg"))

    normalized_row = {
        "ts_code": raw_row.get("ts_code"),
        "trade_date": trade_date,
        "close": close * qfq_ratio,
        "high": (high if high is not None else close) * qfq_ratio,
        "low": (low if low is not None else close) * qfq_ratio,
        "vol": vol,
        "amount": amount,
        "pct_chg": pct_chg,
    }
    return normalized_row


def _sort_frame_by_trade_date_desc(df):
    columns = getattr(df, "columns", [])
    if "trade_date" not in columns or not hasattr(df, "sort_values"):
        return df
    try:
        return df.sort_values(by="trade_date", ascending=False)
    except Exception:
        return df
