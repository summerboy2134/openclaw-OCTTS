"""Stock screener service for technical analysis based screening."""

import logging
import time
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from octts.config import Settings, get_settings
from octts.clients.tushare_client import TushareClient
from octts.schemas.screener import (
    ScreenCriteria,
    StockScreenItem,
    ScreenResult,
    ScreenPreset
)
from octts.indicators.technical import build_technical_snapshot


@lru_cache(maxsize=4)
def _get_cached_stock_list(list_status: str) -> tuple[Dict[str, Any], ...]:
    """Cache the exchange stock list across requests."""
    client = TushareClient(get_settings())
    stocks = client.fetch_stock_list(list_status=list_status)
    return tuple(dict(item) for item in stocks)


_SCREEN_RESULTS_CACHE: Dict[str, ScreenResult] = {}
logger = logging.getLogger(__name__)

_RECOMMENDATION_ORDER = {
    "avoid": 0,
    "weak_or_early": 1,
    "monitor": 2,
    "buy_watchlist": 3,
    "strong_buy_watchlist": 4,
}

_RISK_LEVEL_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _name_indicates_st(name: Any) -> bool:
    text = str(name or "").strip().upper().replace(" ", "")
    return bool(text) and (text.startswith("ST") or text.startswith("*ST") or "退" in text[:3])


class StockScreener:
    """股票筛选服务"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[TushareClient] = None,
    ):
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> TushareClient:
        if self._client is None:
            self._client = TushareClient(self.settings)
        return self._client

    def get_all_stocks(self) -> List[Dict[str, Any]]:
        """
        获取所有股票列表(缓存1天)

        Returns:
            股票列表
        """
        if self._client is not None:
            return self.client.fetch_stock_list(list_status="L")
        return [dict(item) for item in _get_cached_stock_list("L")]

    def clear_cache(self):
        """清除缓存"""
        _get_cached_stock_list.cache_clear()

    def screen(
        self,
        criteria: ScreenCriteria,
        trade_date: Optional[str] = None,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> ScreenResult:
        """
        执行股票筛选

        Args:
            criteria: 筛选条件
            trade_date: 指定交易日，默认为最近交易日

        Returns:
            筛选结果
        """
        start_time = time.time()
        screen_id = str(uuid.uuid4())

        resolved_trade_date = trade_date or self._get_latest_trade_date()
        snapshot = market_snapshot or {}
        all_stocks = snapshot.get("stocks") if market_snapshot else self.get_all_stocks()
        if not isinstance(all_stocks, list):
            all_stocks = []

        filtered_stocks = self._pre_filter_stocks(all_stocks, criteria)

        logger.info(
            "Stock screener %s start: %s stocks -> %s filtered, trade_date=%s",
            screen_id,
            len(all_stocks),
            len(filtered_stocks),
            resolved_trade_date,
        )

        # 获取股票数据
        stock_data = self._fetch_stock_data(
            [s["ts_code"] for s in filtered_stocks],
            resolved_trade_date,
            market_snapshot=market_snapshot,
        )

        filtered_stocks = self._pre_filter_by_basic_data(filtered_stocks, stock_data, criteria)
        filtered_ts_codes = [stock["ts_code"] for stock in filtered_stocks]
        stock_data = self._ensure_daily_data(
            filtered_ts_codes,
            stock_data,
            resolved_trade_date,
            market_snapshot=market_snapshot,
        )

        # 应用筛选条件
        screened_items = []
        failure_counts: Dict[str, int] = {}
        failure_samples: Dict[str, List[Dict[str, Any]]] = {}
        shallow_history_count = 0
        shallow_history_samples: List[Dict[str, Any]] = []
        for stock in filtered_stocks:
            ts_code = stock["ts_code"]
            if ts_code not in stock_data:
                failure_counts["missing_stock_data"] = failure_counts.get("missing_stock_data", 0) + 1
                if len(failure_samples.setdefault("missing_stock_data", [])) < 5:
                    failure_samples["missing_stock_data"].append({"ts_code": ts_code})
                continue

            item, evaluation_meta = self._evaluate_stock(
                stock,
                stock_data[ts_code],
                criteria,
                resolved_trade_date
            )

            if item is None:
                failure_counts["evaluation_failed"] = failure_counts.get("evaluation_failed", 0) + 1
                if len(failure_samples.setdefault("evaluation_failed", [])) < 5:
                    failure_samples["evaluation_failed"].append({"ts_code": ts_code})
                continue

            if evaluation_meta.get("shallow_history"):
                shallow_history_count += 1
                if len(shallow_history_samples) < 5:
                    shallow_history_samples.append(
                        {
                            "ts_code": item.ts_code,
                            "history_rows": evaluation_meta.get("history_rows"),
                            "latest_trade_date": evaluation_meta.get("latest_trade_date"),
                            "rsi": item.rsi,
                            "price_position_20d": item.price_position_20d,
                        }
                    )

            failure_reason = self._get_failure_reason(item, criteria)
            if failure_reason is None:
                screened_items.append(item)
                continue

            failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
            if len(failure_samples.setdefault(failure_reason, [])) < 5:
                failure_samples[failure_reason].append(
                    {
                        "ts_code": item.ts_code,
                        "technical_score": round(float(item.technical_score or 0.0), 2) if item.technical_score is not None else None,
                        "recommendation_score": round(float(item.recommendation_score or 0.0), 2) if item.recommendation_score is not None else None,
                        "pct_change": round(float(item.pct_change or 0.0), 2) if item.pct_change is not None else None,
                        "volume_ratio": round(float(item.volume_ratio or 0.0), 2) if item.volume_ratio is not None else None,
                        "turnover_rate": round(float(item.turnover_rate or 0.0), 2) if item.turnover_rate is not None else None,
                        "rsi": round(float(item.rsi or 0.0), 2) if item.rsi is not None else None,
                        "price_position_20d": round(float(item.price_position_20d or 0.0), 4) if item.price_position_20d is not None else None,
                    }
                )

        if failure_counts:
            logger.info(
                "Stock screener %s failure distribution: %s, samples=%s",
                screen_id,
                dict(sorted(failure_counts.items(), key=lambda entry: (-entry[1], entry[0]))),
                failure_samples,
            )
        if shallow_history_count:
            logger.info(
                "Stock screener %s technical snapshot shallow history summary: count=%s, samples=%s",
                screen_id,
                shallow_history_count,
                shallow_history_samples,
            )

        # 排序
        screened_items = self._sort_results(screened_items, criteria)

        # 分页
        total_count = len(screened_items)
        screened_items = screened_items[
            criteria.offset:criteria.offset + criteria.limit
        ]

        execution_time = time.time() - start_time
        logger.info(
            "Stock screener %s evaluated: stock_data=%s, matched=%s, execution_time=%.2fs",
            screen_id,
            len(stock_data),
            total_count,
            execution_time,
        )

        result = ScreenResult(
            screen_id=screen_id,
            criteria=criteria,
            stocks=screened_items,
            total_count=total_count,
            execution_time=execution_time
        )
        self._store_result(result)
        return result

    @staticmethod
    def get_presets() -> List[ScreenPreset]:
        """获取预设筛选策略"""
        return []

    @staticmethod
    def get_screen_result(screen_id: str) -> Optional[ScreenResult]:
        return _SCREEN_RESULTS_CACHE.get(screen_id)

    @staticmethod
    def _store_result(result: ScreenResult) -> None:
        _SCREEN_RESULTS_CACHE[result.screen_id] = result
        if len(_SCREEN_RESULTS_CACHE) > 100:
            oldest_screen_id = min(
                _SCREEN_RESULTS_CACHE,
                key=lambda item: _SCREEN_RESULTS_CACHE[item].screen_time,
            )
            _SCREEN_RESULTS_CACHE.pop(oldest_screen_id, None)

    def _get_latest_trade_date(self) -> str:
        """获取最近交易日"""
        today = datetime.now()
        try:
            return self.client.resolve_latest_trade_date(
                end_date=today.strftime("%Y%m%d"),
                lookback_days=7,
            )
        except Exception:
            logger.exception("Failed to resolve latest trade date with remote probe, fallback to local calendar")

        end_date = today.strftime("%Y%m%d")
        start_date = (today - timedelta(days=7)).strftime("%Y%m%d")
        trade_dates = self.client.fetch_trading_dates(start_date=start_date, end_date=end_date)
        return trade_dates[-1] if trade_dates else end_date

    def _pre_filter_stocks(
        self,
        stocks: List[Dict[str, Any]],
        criteria: ScreenCriteria
    ) -> List[Dict[str, Any]]:
        """初步筛选股票"""
        filtered = []

        for stock in stocks:
            ts_code = str(stock.get("ts_code") or "").upper()
            if criteria.exclude_st and _name_indicates_st(stock.get("name", "")):
                continue
            if criteria.exclude_bj and ts_code.endswith(".BJ"):
                continue
            if criteria.industries and stock.get("industry") not in criteria.industries:
                continue
            filtered.append(stock)

        return filtered

    def _pre_filter_by_basic_data(
        self,
        stocks: List[Dict[str, Any]],
        stock_data: Dict[str, Dict[str, Any]],
        criteria: ScreenCriteria,
    ) -> List[Dict[str, Any]]:
        filtered = []
        for stock in stocks:
            ts_code = stock.get("ts_code")
            if not ts_code:
                continue
            basic = stock_data.get(ts_code, {}).get("basic")
            if not basic:
                continue
            if not self._basic_data_meets_criteria(stock, basic, criteria):
                continue
            filtered.append(stock)
        return filtered

    def _basic_data_meets_criteria(
        self,
        stock: Dict[str, Any],
        basic: Dict[str, Any],
        criteria: ScreenCriteria,
    ) -> bool:
        close = self._to_float(basic.get("close"))
        turnover_rate = self._to_float(basic.get("turnover_rate"))
        volume_ratio = self._to_float(basic.get("volume_ratio"))
        market_cap = self._to_float(basic.get("total_mv"))
        pct_change = self._to_float(basic.get("pct_chg"))

        if criteria.price_min is not None and (close is None or close < criteria.price_min):
            return False
        if criteria.price_max is not None and (close is None or close > criteria.price_max):
            return False
        if criteria.turnover_rate_min is not None and (turnover_rate is None or turnover_rate < criteria.turnover_rate_min):
            return False
        if criteria.turnover_rate_max is not None and (turnover_rate is None or turnover_rate > criteria.turnover_rate_max):
            return False
        if criteria.volume_ratio_min is not None and (volume_ratio is None or volume_ratio < criteria.volume_ratio_min):
            return False
        if criteria.volume_ratio_max is not None and (volume_ratio is None or volume_ratio > criteria.volume_ratio_max):
            return False
        if criteria.market_cap_min is not None:
            if market_cap is None or market_cap / 10000 < criteria.market_cap_min:
                return False
        if criteria.market_cap_max is not None:
            if market_cap is None or market_cap / 10000 > criteria.market_cap_max:
                return False
        if criteria.pct_change_min is not None and pct_change is not None and pct_change < criteria.pct_change_min:
            return False
        if criteria.pct_change_max is not None and pct_change is not None and pct_change > criteria.pct_change_max:
            return False
        if criteria.industries and stock.get("industry") not in criteria.industries:
            return False
        if self._has_loss_risk(ts_code=stock.get("ts_code"), criteria=criteria):
            return False
        if criteria.require_positive_3d_moneyflow and not self._has_positive_3d_moneyflow(stock.get("ts_code")):
            return False
        return True

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _has_loss_risk(self, ts_code: Optional[str], criteria: ScreenCriteria) -> bool:
        if not ts_code or criteria.max_recent_loss_years is None:
            return False
        statements = self.client.fetch_income_statements(ts_code)
        if not statements:
            return False
        annual_records = []
        for item in statements:
            end_date = str(item.get("end_date") or "")
            if len(end_date) != 8 or not end_date.endswith("1231"):
                continue
            annual_records.append(item)
        annual_records.sort(key=lambda item: str(item.get("end_date") or ""), reverse=True)
        consecutive_losses = 0
        for item in annual_records:
            profit = self._to_float(item.get("profit_dedt"))
            if profit is None:
                profit = self._to_float(item.get("n_income_attr_p"))
            if profit is None:
                continue
            if profit < 0:
                consecutive_losses += 1
                if consecutive_losses > criteria.max_recent_loss_years:
                    return True
            else:
                break
        return False

    def _has_positive_3d_moneyflow(self, ts_code: Optional[str]) -> bool:
        if not ts_code:
            return False
        moneyflow = self.client.fetch_moneyflow(ts_code)
        if not moneyflow:
            return False
        recent_rows = sorted(moneyflow, key=lambda item: str(item.get("trade_date") or ""), reverse=True)[:3]
        if len(recent_rows) < 3:
            return False
        total_flow = 0.0
        for item in recent_rows:
            total_flow += self._to_float(item.get("net_mf_amount")) or 0.0
        return total_flow > 0

    @staticmethod
    def _hits_late_stage_risk_gate(
        *,
        pct_change: Optional[float],
        turnover_rate: Optional[float],
        volume_ratio: Optional[float],
        price_position: Optional[float],
        criteria: ScreenCriteria,
    ) -> bool:
        max_position = criteria.max_late_stage_price_position
        if max_position is None:
            return False
        if price_position is None or price_position > max_position:
            return True

        pct_change_value = pct_change or 0.0
        turnover_value = turnover_rate or 0.0
        volume_ratio_value = volume_ratio or 0.0
        high_position = price_position >= max(0.9, max_position - 0.03)
        overheated_move = pct_change_value >= 7.0
        weak_close = pct_change_value <= -1.0
        high_turnover = turnover_value >= 12.0
        abnormal_volume = volume_ratio_value >= 2.8

        if high_position and weak_close and high_turnover and abnormal_volume:
            return True
        if high_position and high_turnover and abnormal_volume and pct_change_value <= -2.0:
            return True
        if high_position and abnormal_volume and overheated_move and weak_close:
            return True
        return False

    def _fetch_stock_data(
        self,
        ts_codes: List[str],
        trade_date: str,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """批量获取股票数据"""
        logger.info(
            "Stock screener fetching data: %s symbols for %s",
            len(ts_codes),
            trade_date,
        )
        if market_snapshot is not None:
            basic_source = market_snapshot.get("daily_basic", {})
            daily_source = market_snapshot.get("daily", {})
            basic_data = {
                ts_code: basic_source[ts_code]
                for ts_code in ts_codes
                if ts_code in basic_source
            }
            daily_data = {
                ts_code: daily_source.get(ts_code, [])
                for ts_code in ts_codes
            }
            logger.info(
                "Stock screener reused snapshot basic data: basic=%s/%s, cached_daily=%s/%s",
                len(basic_data),
                len(ts_codes),
                sum(1 for items in daily_data.values() if items),
                len(ts_codes),
            )
        else:
            basic_start = time.time()
            basic_data = self.client.fetch_daily_basic_batch(
                ts_codes=ts_codes,
                trade_date=trade_date
            )
            logger.info(
                "Stock screener daily_basic fetched: %s/%s symbols in %.2fs",
                len(basic_data),
                len(ts_codes),
                time.time() - basic_start,
            )
            daily_data = {ts_code: [] for ts_code in ts_codes}

        result = {}
        for ts_code in ts_codes:
            if ts_code in basic_data:
                result[ts_code] = {
                    "basic": basic_data[ts_code],
                    "daily": daily_data.get(ts_code, [])
                }

        return result

    def _ensure_daily_data(
        self,
        ts_codes: List[str],
        stock_data: Dict[str, Dict[str, Any]],
        trade_date: str,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not ts_codes:
            return stock_data

        missing_ts_codes = [
            ts_code for ts_code in ts_codes
            if not stock_data.get(ts_code, {}).get("daily")
        ]
        if not missing_ts_codes:
            return stock_data

        end_date = trade_date
        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60)
        ).strftime("%Y%m%d")
        daily_start = time.time()
        fetched_daily = self.client.fetch_daily_batch(
            ts_codes=missing_ts_codes,
            start_date=start_date,
            end_date=end_date
        )
        fetched_count = sum(1 for items in fetched_daily.values() if items)
        missing_after_fetch = [
            ts_code for ts_code in missing_ts_codes
            if not fetched_daily.get(ts_code)
        ]
        logger.info(
            "Stock screener daily history fetched after prefilter: %s/%s symbols in %.2fs",
            fetched_count,
            len(missing_ts_codes),
            time.time() - daily_start,
        )
        if missing_after_fetch:
            logger.warning(
                "Stock screener missing daily history after fetch: count=%s, trade_date=%s, symbols=%s",
                len(missing_after_fetch),
                trade_date,
                ",".join(missing_after_fetch[:20]),
            )

        if market_snapshot is not None:
            daily_source = market_snapshot.setdefault("daily", {})
            for ts_code, rows in fetched_daily.items():
                daily_source[ts_code] = rows

        for ts_code in missing_ts_codes:
            if ts_code in stock_data:
                stock_data[ts_code]["daily"] = fetched_daily.get(ts_code, [])

        return stock_data

    def _evaluate_stock(
        self,
        stock_info: Dict[str, Any],
        stock_data: Dict[str, Any],
        criteria: ScreenCriteria,
        trade_date: str
    ) -> tuple[Optional[StockScreenItem], Dict[str, Any]]:
        """评估单个股票"""
        ts_code = stock_info["ts_code"]
        basic = stock_data.get("basic", {})
        daily_list = stock_data.get("daily", [])

        if not basic or not daily_list:
            return None, {}

        ordered_daily = sorted(
            daily_list,
            key=lambda item: str(item.get("trade_date", "")),
        )
        latest = ordered_daily[-1]

        closes = pd.Series([float(item["close"]) for item in ordered_daily])
        highs = pd.Series([
            float(item.get("high", item["close"])) for item in ordered_daily
        ])
        lows = pd.Series([
            float(item.get("low", item["close"])) for item in ordered_daily
        ])
        volumes = pd.Series([
            float(item["vol"]) for item in ordered_daily if item.get("vol") is not None
        ])

        snapshot = build_technical_snapshot(closes, highs, lows, volumes)
        evaluation_meta = {
            "shallow_history": snapshot.rsi is None or snapshot.price_position_20d is None,
            "history_rows": len(ordered_daily),
            "latest_trade_date": latest.get("trade_date"),
        }
        volume_ratio = float(basic.get("volume_ratio", 0)) if basic.get("volume_ratio") else snapshot.volume_ratio

        pct_change = self._to_float(latest.get("pct_chg"))
        if pct_change is None:
            pct_change = self._to_float(latest.get("pct_change"))

        item = StockScreenItem(
            ts_code=ts_code,
            name=stock_info.get("name", ""),
            close=float(basic.get("close", snapshot.close)),
            pct_change=pct_change,
            volume_ratio=volume_ratio,
            turnover_rate=float(basic.get("turnover_rate", 0)),
            rsi=snapshot.rsi,
            ma5=snapshot.ma5,
            ma10=snapshot.ma10,
            ma20=snapshot.ma20,
            ma60=snapshot.ma60,
            macd=snapshot.macd,
            macd_signal=snapshot.macd_signal,
            macd_histogram=snapshot.macd_histogram,
            price_position_20d=snapshot.price_position_20d,
            trend_status=snapshot.trend_status,
            momentum_status=snapshot.momentum_status,
            technical_score=snapshot.technical_score,
            trend_score=snapshot.trend_score,
            momentum_score=snapshot.momentum_score,
            volume_score=snapshot.volume_score,
            breakout_score=snapshot.breakout_score,
            risk_score=snapshot.risk_score,
            setup_quality_score=snapshot.setup_quality_score,
            recommendation_score=snapshot.recommendation_score,
            recommendation=snapshot.recommendation,
            setup_type=snapshot.setup_type,
            risk_level=snapshot.risk_level,
            entry_style=snapshot.entry_style,
            confidence=snapshot.confidence,
            risk_flags=list(snapshot.risk_flags),
            setup_notes=list(snapshot.setup_notes),
            distance_to_ma20_pct=snapshot.distance_to_ma20_pct,
            distance_to_ma60_pct=snapshot.distance_to_ma60_pct,
            breakout_strength=snapshot.breakout_strength,
            score=snapshot.recommendation_score,
            market_cap=float(basic.get("total_mv", 0)) / 10000 if basic.get("total_mv") else None,
            pe_ratio=float(basic.get("pe", 0)) if basic.get("pe") else None,
            industry=(
                stock_info.get("industry")
                if isinstance(stock_info.get("industry"), str)
                else None
            )
        )

        match_reasons = []
        if item.setup_type:
            match_reasons.append(f"形态={item.setup_type}")
        if item.recommendation:
            match_reasons.append(f"推荐={item.recommendation}")
        if criteria.rsi_max is not None and item.rsi is not None and item.rsi <= criteria.rsi_max:
            match_reasons.append(f"RSI={item.rsi:.1f}低于{criteria.rsi_max}")
        if criteria.volume_ratio_min is not None and item.volume_ratio >= criteria.volume_ratio_min:
            match_reasons.append(f"量比={item.volume_ratio:.1f}")
        if criteria.pct_change_min is not None and item.pct_change is not None and item.pct_change >= criteria.pct_change_min:
            match_reasons.append(f"涨幅={item.pct_change:.1f}%")
        if criteria.technical_score_min is not None and item.technical_score is not None and item.technical_score >= criteria.technical_score_min:
            match_reasons.append(f"技术评分={item.technical_score:.1f}")
        if criteria.recommendation_score_min is not None and item.recommendation_score is not None and item.recommendation_score >= criteria.recommendation_score_min:
            match_reasons.append(f"推荐评分={item.recommendation_score:.1f}")
        if criteria.setup_quality_score_min is not None and item.setup_quality_score is not None and item.setup_quality_score >= criteria.setup_quality_score_min:
            match_reasons.append(f"形态质量={item.setup_quality_score:.1f}")
        if criteria.require_bullish_ma_alignment and item.trend_status == "bullish":
            match_reasons.append("均线多头排列")
        if criteria.require_macd_above_signal and item.macd is not None and item.macd_signal is not None and item.macd >= item.macd_signal:
            match_reasons.append("MACD位于信号线上方")
        if criteria.price_position_min is not None and item.price_position_20d is not None and item.price_position_20d >= criteria.price_position_min:
            match_reasons.append(f"20日位置={item.price_position_20d:.2f}")
        if item.entry_style:
            match_reasons.append(f"入场风格={item.entry_style}")

        item.match_reasons = match_reasons

        return item, evaluation_meta

    def _get_failure_reason(
        self,
        item: StockScreenItem,
        criteria: ScreenCriteria,
    ) -> Optional[str]:
        """返回首个失败原因，便于统计策略失效分布。"""
        # 价格条件
        if criteria.price_min is not None and item.close < criteria.price_min:
            return "price_min"
        if criteria.price_max is not None and item.close > criteria.price_max:
            return "price_max"

        # 涨跌幅
        if criteria.pct_change_min is not None and (item.pct_change is None or item.pct_change < criteria.pct_change_min):
            return "pct_change_min"
        if criteria.pct_change_max is not None and (item.pct_change is None or item.pct_change > criteria.pct_change_max):
            return "pct_change_max"

        # 成交量
        if criteria.volume_ratio_min is not None and item.volume_ratio < criteria.volume_ratio_min:
            return "volume_ratio_min"
        if criteria.volume_ratio_max is not None and item.volume_ratio > criteria.volume_ratio_max:
            return "volume_ratio_max"

        # 换手率
        if criteria.turnover_rate_min is not None and item.turnover_rate < criteria.turnover_rate_min:
            return "turnover_rate_min"
        if criteria.turnover_rate_max is not None and item.turnover_rate > criteria.turnover_rate_max:
            return "turnover_rate_max"

        # RSI
        if criteria.rsi_min is not None:
            if item.rsi is None or item.rsi < criteria.rsi_min:
                return "rsi_min"
        if criteria.rsi_max is not None:
            if item.rsi is None or item.rsi > criteria.rsi_max:
                return "rsi_max"

        # 均线
        if criteria.ma5_above_ma20:
            if item.ma5 is None or item.ma20 is None or item.ma5 <= item.ma20:
                return "ma5_above_ma20"

        if criteria.price_above_ma5:
            if item.ma5 is None or item.close <= item.ma5:
                return "price_above_ma5"

        if criteria.price_above_ma60:
            if item.ma60 is None or item.close <= item.ma60:
                return "price_above_ma60"

        if criteria.require_bullish_ma_alignment:
            if item.trend_status != "bullish":
                return "require_bullish_ma_alignment"

        if criteria.require_macd_above_signal:
            if item.macd is None or item.macd_signal is None or item.macd < item.macd_signal:
                return "require_macd_above_signal"

        if criteria.technical_score_min is not None:
            if item.technical_score is None or item.technical_score < criteria.technical_score_min:
                return "technical_score_min"
        if criteria.recommendation_score_min is not None:
            if item.recommendation_score is None or item.recommendation_score < criteria.recommendation_score_min:
                return "recommendation_score_min"
        if criteria.setup_quality_score_min is not None:
            if item.setup_quality_score is None or item.setup_quality_score < criteria.setup_quality_score_min:
                return "setup_quality_score_min"
        if criteria.setup_types:
            if item.setup_type is None or item.setup_type not in criteria.setup_types:
                return "setup_types"
        if criteria.risk_level_max is not None:
            item_risk = _RISK_LEVEL_ORDER.get(item.risk_level or "high")
            max_risk = _RISK_LEVEL_ORDER.get(criteria.risk_level_max)
            if max_risk is None or item_risk > max_risk:
                return "risk_level_max"
        if criteria.recommendation_min is not None:
            item_rec = _RECOMMENDATION_ORDER.get(item.recommendation or "avoid")
            min_rec = _RECOMMENDATION_ORDER.get(criteria.recommendation_min)
            if min_rec is None or item_rec < min_rec:
                return "recommendation_min"

        if criteria.price_position_min is not None:
            if item.price_position_20d is None or item.price_position_20d < criteria.price_position_min:
                return "price_position_min"
        if criteria.price_position_max is not None:
            if item.price_position_20d is None or item.price_position_20d > criteria.price_position_max:
                return "price_position_max"
        if criteria.distance_to_ma60_pct_min is not None:
            if item.distance_to_ma60_pct is None or item.distance_to_ma60_pct < criteria.distance_to_ma60_pct_min:
                return "distance_to_ma60_pct_min"
        if criteria.distance_to_ma60_pct_max is not None:
            if item.distance_to_ma60_pct is None or item.distance_to_ma60_pct > criteria.distance_to_ma60_pct_max:
                return "distance_to_ma60_pct_max"
        if self._hits_late_stage_risk_gate(
            pct_change=item.pct_change,
            turnover_rate=item.turnover_rate,
            volume_ratio=item.volume_ratio,
            price_position=item.price_position_20d,
            criteria=criteria,
        ):
            return "late_stage_risk_gate"

        # 市值
        if criteria.market_cap_min is not None:
            if item.market_cap is None or item.market_cap < criteria.market_cap_min:
                return "market_cap_min"
        if criteria.market_cap_max is not None:
            if item.market_cap is None or item.market_cap > criteria.market_cap_max:
                return "market_cap_max"

        return None

    def _meets_criteria(
        self,
        item: StockScreenItem,
        criteria: ScreenCriteria
    ) -> bool:
        """检查是否满足筛选条件"""
        return self._get_failure_reason(item, criteria) is None

    def _sort_results(
        self,
        items: List[StockScreenItem],
        criteria: ScreenCriteria
    ) -> List[StockScreenItem]:
        """排序结果"""
        # 获取排序字段
        sort_key = criteria.sort_by
        reverse = criteria.sort_desc

        # 定义排序函数
        def get_sort_value(item: StockScreenItem):
            value = getattr(item, sort_key, None)
            if value is None:
                return float('-inf') if reverse else float('inf')
            if sort_key == "risk_level":
                return _RISK_LEVEL_ORDER.get(value, float('inf'))
            if sort_key == "recommendation":
                return _RECOMMENDATION_ORDER.get(value, float('-inf') if reverse else float('inf'))
            return value

        return sorted(items, key=get_sort_value, reverse=reverse)