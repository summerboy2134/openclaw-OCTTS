from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Optional, List, Dict

from octts.config import Settings
from octts.schemas.backtest import DailyBar
from octts.schemas.report import AnalysisPhase, PriceSnapshot

MAX_RECENT_TRADE_DATE_FALLBACKS = 3
PRO_BAR_EMPTY_RESULT_MESSAGES = (
    "single positional indexer is out-of-bounds",
    "out-of-bounds",
)
logger = logging.getLogger(__name__)
SCREENING_SNAPSHOT_REQUIRED_BASIC_FIELDS = {"ts_code", "close", "total_mv", "turnover_rate", "volume_ratio"}
SCREENING_SNAPSHOT_MIN_BASIC_FIELDS = 2


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

        self._safe_set_token(ts, settings.tushare_token)
        self._ts = ts
        self._pro = ts.pro_api(settings.tushare_token)
        self._settings = settings

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
                fields="ts_code,end_date,eps,dt_eps,roe,roe_dt,netprofit_margin,grossprofit_margin,assets_turn,op_income_yoy,netprofit_yoy",
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
            try:
                return self._ts.pro_bar(**kwargs)
            except Exception as fallback_exc:
                if not self._is_empty_pro_bar_error(fallback_exc):
                    raise
                self._log_empty_pro_bar(kwargs)
                return None
        except Exception as exc:
            if not self._is_empty_pro_bar_error(exc):
                raise
            self._log_empty_pro_bar(kwargs)
            return None

    def _is_empty_pro_bar_error(self, exc: Exception) -> bool:
        message = str(exc)
        return any(fragment in message for fragment in PRO_BAR_EMPTY_RESULT_MESSAGES)

    def _log_empty_pro_bar(self, kwargs: dict[str, Any]) -> None:
        logger.debug("pro_bar returned no usable rows: %s", kwargs)

    def get_or_build_screening_snapshot(self, trade_date: str) -> Dict[str, Any]:
        normalized_trade_date = self._resolve_target_trade_date(trade_date)
        snapshot = self._read_screening_snapshot(normalized_trade_date)
        if snapshot is not None:
            stale_reason = self._screening_snapshot_stale_reason(normalized_trade_date, snapshot)
            if stale_reason is None:
                logger.info("Screening snapshot hit: trade_date=%s", normalized_trade_date)
                return snapshot
            logger.info(stale_reason)

        logger.info("Screening snapshot build start: trade_date=%s", normalized_trade_date)
        stocks = self.fetch_stock_list(list_status="L")
        ts_codes = [stock.get("ts_code") for stock in stocks if stock.get("ts_code")]
        daily_basic = self.fetch_daily_basic_batch(ts_codes=ts_codes, trade_date=normalized_trade_date)
        snapshot = {
            "trade_date": normalized_trade_date,
            "created_at": datetime.now().isoformat(),
            "stocks": stocks,
            "daily_basic": daily_basic,
            "daily": {},
        }
        invalid_reason = self._screening_snapshot_invalid_reason(snapshot)
        if invalid_reason is None:
            self._write_screening_snapshot(normalized_trade_date, snapshot)
        else:
            logger.warning(
                "Screening snapshot build incomplete: trade_date=%s, reason=%s, skip_cache_write=true",
                normalized_trade_date,
                invalid_reason,
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
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                snapshot = json.load(f)
        except Exception as exc:
            logger.warning("Screening snapshot read failed: %s (%s)", trade_date, exc)
            return None
        invalid_reason = self._screening_snapshot_invalid_reason(snapshot)
        if invalid_reason is not None:
            logger.warning(
                "Screening snapshot invalid: trade_date=%s, reason=%s, rebuild=true",
                trade_date,
                invalid_reason,
            )
            return None
        return snapshot

    def _write_screening_snapshot(self, trade_date: str, snapshot: Dict[str, Any]) -> None:
        path = self._screening_snapshot_path(trade_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)

    def _screening_snapshot_invalid_reason(self, snapshot: Any) -> Optional[str]:
        if not isinstance(snapshot, dict):
            return "snapshot_not_dict"

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

        return None

    def _screening_snapshot_basic_complete(self, basic: Dict[str, Any]) -> bool:
        fields = {key for key, value in basic.items() if value is not None and key in SCREENING_SNAPSHOT_REQUIRED_BASIC_FIELDS}
        return len(fields) >= SCREENING_SNAPSHOT_MIN_BASIC_FIELDS

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
                return []
            rows = df.to_dict(orient="records")
            logger.info("Tushare fetch_stock_list complete: %s rows in %.2fs", len(rows), time.time() - started_at)
            return rows
        except Exception as exc:
            logger.error("Tushare fetch_stock_list failed after %.2fs: %s", time.time() - started_at, exc)
            return []

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
        result = {}
        candidate_trade_dates = self._candidate_trade_dates(target_trade_date=trade_date)

        # 分批处理，每批最多1000个，减少请求次数
        batch_size = 1000
        for i in range(0, len(ts_codes), batch_size):
            batch_codes = ts_codes[i:i + batch_size]

            for candidate_trade_date in candidate_trade_dates:
                missing_codes = [ts_code for ts_code in batch_codes if ts_code not in result]
                if not missing_codes:
                    break

                try:
                    df = self._pro.daily_basic(
                        ts_code=",".join(missing_codes),
                        trade_date=candidate_trade_date,
                        fields="ts_code,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,total_mv"
                    )
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            ts_code = row.get("ts_code")
                            if ts_code and ts_code not in result:
                                result[ts_code] = row.to_dict()
                except Exception:
                    pass

            # 如果批量失败或部分缺失，尝试单个获取
            missing_codes = [ts_code for ts_code in batch_codes if ts_code not in result]
            for ts_code in missing_codes:
                for candidate_trade_date in candidate_trade_dates:
                    try:
                        df = self._pro.daily_basic(ts_code=ts_code, trade_date=candidate_trade_date)
                        if df is not None and not df.empty:
                            result[ts_code] = df.iloc[0].to_dict()
                            break
                    except Exception:
                        continue

        logger.info(
            "Tushare fetch_daily_basic_batch complete: %s/%s symbols in %.2fs",
            len(result),
            len(ts_codes),
            time.time() - started_at,
        )
        return result

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
        result = {ts_code: [] for ts_code in ts_codes}

        # 分批处理
        batch_size = 50
        for i in range(0, len(ts_codes), batch_size):
            batch_codes = ts_codes[i:i + batch_size]

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
                except Exception:
                    continue
                if df is None or df.empty:
                    continue
                df = _sort_frame_by_trade_date_desc(df)
                result[ts_code] = df.to_dict(orient="records")

        non_empty = sum(1 for items in result.values() if items)
        logger.info(
            "Tushare fetch_daily_batch complete: %s/%s symbols with data in %.2fs",
            non_empty,
            len(ts_codes),
            time.time() - started_at,
        )
        return result


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


def _sort_frame_by_trade_date_desc(df):
    columns = getattr(df, "columns", [])
    if "trade_date" not in columns or not hasattr(df, "sort_values"):
        return df
    try:
        return df.sort_values(by="trade_date", ascending=False)
    except Exception:
        return df
