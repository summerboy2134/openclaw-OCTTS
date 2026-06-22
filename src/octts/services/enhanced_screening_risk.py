"""Shared helpers for enhanced screening scheduler mixins."""

import html
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.services.stock_screener import StockScreener
from octts.services.regression_rerank_service import RegressionRerankResult
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_constants import *

logger = logging.getLogger(__name__)


class EnhancedScreeningRiskMixin:
    def _build_distribution_risk_map(
        self,
        stock_map: Dict[str, Any],
        *,
        market_snapshot: Optional[Dict[str, Any]] = None,
        trade_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        daily_history_map = {}
        if isinstance(market_snapshot, dict):
            raw_daily = market_snapshot.get("daily")
            if isinstance(raw_daily, dict):
                daily_history_map = raw_daily
        risk_map: Dict[str, Dict[str, Any]] = {}
        relay_limit_map = self.market_raw_data_repo.get_limit_list_by_trade_date(
            ts_codes=stock_map.keys(),
            trade_date=trade_date,
        ) if trade_date else {}
        relay_top_map = self.market_raw_data_repo.get_top_list_by_trade_date(
            ts_codes=stock_map.keys(),
            trade_date=trade_date,
        ) if trade_date else {}
        for code, stock in stock_map.items():
            daily_rows = daily_history_map.get(code) or []
            risk_map[code] = self._evaluate_distribution_risk(
                stock,
                daily_rows=daily_rows,
                relay_limit=relay_limit_map.get(code),
                relay_top_rows=relay_top_map.get(code) or [],
                trade_date=trade_date,
            )
            if code in {"002269.SZ", "301226.SZ", "600222.SH"}:
                logger.info(
                    "Risk evaluation for %s: daily_rows=%s, stock_turnover=%s, stock_volume_ratio=%s, stock_pct_change=%s, result=%s",
                    code,
                    len(daily_rows),
                    (stock.get("turnover_rate") if isinstance(stock, dict) else getattr(stock, "turnover_rate", None)),
                    (stock.get("volume_ratio") if isinstance(stock, dict) else getattr(stock, "volume_ratio", None)),
                    (stock.get("pct_change") if isinstance(stock, dict) else getattr(stock, "pct_change", None)),
                    risk_map[code],
                )
        return risk_map

    def _evaluate_distribution_risk(
        self,
        stock: Any,
        *,
        daily_rows: List[Dict[str, Any]],
        relay_limit: Optional[Dict[str, Any]] = None,
        relay_top_rows: Optional[List[Dict[str, Any]]] = None,
        trade_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(stock, dict):
            ts_code = str(stock.get("ts_code") or "").strip()
            raw_volume_ratio = self._safe_float(stock.get("volume_ratio"))
            volume_ratio = raw_volume_ratio or 0.0
            pct_change = self._safe_float(stock.get("pct_change"))
            if pct_change is None:
                pct_change = self._safe_float(stock.get("pct_chg")) or 0.0
            raw_price_position = self._safe_float(stock.get("price_position_20d"))
            price_position = raw_price_position or 0.0
            raw_turnover_rate = self._safe_float(stock.get("turnover_rate"))
            turnover_rate = raw_turnover_rate or 0.0
        else:
            ts_code = str(getattr(stock, "ts_code", "") or "").strip()
            raw_volume_ratio = self._safe_float(getattr(stock, "volume_ratio", None))
            volume_ratio = raw_volume_ratio or 0.0
            pct_change = self._safe_float(getattr(stock, "pct_change", None)) or 0.0
            raw_price_position = self._safe_float(getattr(stock, "price_position_20d", None))
            price_position = raw_price_position or 0.0
            raw_turnover_rate = self._safe_float(getattr(stock, "turnover_rate", None))
            turnover_rate = raw_turnover_rate or 0.0
        risk_data_missing_flags: List[str] = []
        if not daily_rows:
            risk_data_missing_flags.append("缺少日线历史，无法判断近期走势风险")
        elif len(daily_rows) < 5:
            risk_data_missing_flags.append("日线历史不足5日，无法完整判断近5日位置风险")
        if raw_price_position is None:
            risk_data_missing_flags.append("缺少20日价格位置")
        if raw_turnover_rate is None or raw_turnover_rate <= 0:
            risk_data_missing_flags.append("缺少有效换手率")
        if raw_volume_ratio is None:
            risk_data_missing_flags.append("缺少量比")
        moneyflow_summary = self._build_stock_moneyflow_summary(ts_code, trade_date=trade_date) if ts_code else None
        if moneyflow_summary and bool(moneyflow_summary.get("moneyflow_data_missing", False)):
            risk_data_missing_flags.append("缺少近3日资金流，无法判断承接质量")
        elif moneyflow_summary and bool(moneyflow_summary.get("moneyflow_data_stale", False)):
            risk_data_missing_flags.append("资金流数据不是当前交易日，承接质量可能滞后")
        moneyflow_3d_value = float((moneyflow_summary or {}).get("recent_3d_net_inflow") or 0.0)
        large_order_net_inflow = float((moneyflow_summary or {}).get("recent_large_order_net_inflow") or 0.0)
        super_large_order_net_inflow = float((moneyflow_summary or {}).get("recent_super_large_order_net_inflow") or 0.0)
        recent_runup_5d = self._build_recent_runup_5d(daily_rows)
        turnover_spike_ratio = self._build_turnover_spike_ratio(daily_rows, turnover_rate)
        relay_limit = relay_limit or {}
        relay_top_rows = relay_top_rows or []

        latest_row = daily_rows[0] if daily_rows else {}
        latest_open = self._safe_float(latest_row.get("open"))
        latest_high = self._safe_float(latest_row.get("high"))
        latest_low = self._safe_float(latest_row.get("low"))
        latest_close = self._safe_float(latest_row.get("close"))
        latest_turnover = self._safe_float(latest_row.get("turnover_rate")) or turnover_rate
        latest_weakening = pct_change <= -1.0
        if latest_open is not None and latest_close is not None and latest_close < latest_open:
            latest_weakening = True
        high_retrace = False
        if latest_high not in (None, 0) and latest_close is not None:
            high_retrace = ((latest_high - latest_close) / latest_high) >= 0.03
            if high_retrace:
                latest_weakening = True
        intraday_range_pct = None
        if latest_high not in (None, 0) and latest_low is not None and latest_close not in (None, 0):
            intraday_range_pct = (latest_high - latest_low) / latest_close * 100.0
        high_turnover_active = latest_turnover >= 12.0 or turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_HIGH
        failed_trend_signal = self._build_high_level_failed_trend_signal(daily_rows)
        failed_trend_flag = bool(failed_trend_signal.get("high_level_failed_trend_flag", False))
        deep_drawdown_rebound_signal = self._build_deep_drawdown_rebound_signal(daily_rows, pct_change=pct_change)
        deep_drawdown_rebound_flag = bool(deep_drawdown_rebound_signal.get("deep_drawdown_rebound_flag", False))
        theme_support_absent_flag = (
            price_position >= DISTRIBUTION_PRICE_POSITION_HIGH
            and recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH
            and moneyflow_3d_value <= 3000
            and not high_retrace
        )

        risk_score = 0.0
        risk_flags: List[str] = []
        open_times = relay_limit.get("open_times")
        try:
            open_times_value = int(open_times) if open_times is not None else 0
        except (TypeError, ValueError):
            open_times_value = 0
        limit_last_time = str(relay_limit.get("last_time") or "").strip()
        limit_first_time = str(relay_limit.get("first_time") or "").strip()
        top_net_amount = sum(self._safe_float(item.get("net_amount")) or 0.0 for item in relay_top_rows)
        top_net_rate_values = [self._safe_float(item.get("net_rate")) for item in relay_top_rows]
        top_net_rate_values = [value for value in top_net_rate_values if value is not None]
        top_net_rate = sum(top_net_rate_values) / len(top_net_rate_values) if top_net_rate_values else None
        if moneyflow_3d_value <= 0:
            risk_score += 1.5
            risk_flags.append("近3日资金未转正")
        elif moneyflow_3d_value < 5000:
            risk_score += 0.7
            risk_flags.append("近3日资金承接偏弱")
        if turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_VERY_HIGH:
            risk_score += 1.2
            risk_flags.append("换手较近期明显激增")
        elif turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_HIGH:
            risk_score += 0.8
            risk_flags.append("换手较近期抬升")
        if volume_ratio >= DISTRIBUTION_VOLUME_RATIO_HIGH:
            risk_score += 0.9
            risk_flags.append("当日量比偏高")
        if pct_change >= DISTRIBUTION_NEAR_LIMIT_UP_PCT_CHANGE_MIN:
            risk_score += 2.0
            risk_flags.append("当日涨幅接近涨停")
        elif pct_change >= DISTRIBUTION_STRONG_MOVE_PCT_CHANGE_MIN:
            risk_score += 1.0
            risk_flags.append("当日涨幅偏大")
        if recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH:
            risk_score += 1.0
            risk_flags.append("近5日累计涨幅偏大")
        if price_position >= DISTRIBUTION_PRICE_POSITION_HIGH:
            risk_score += 0.8
            risk_flags.append("价格处于20日高位")
        if latest_weakening:
            risk_score += 0.9
            risk_flags.append("最新日转弱")
        if high_retrace:
            risk_score += 0.8
            risk_flags.append("高位冲高回落")
        if open_times_value >= 3:
            risk_score += 1.4
            risk_flags.append("炸板次数过多")
        elif open_times_value >= 1:
            risk_score += 0.7
            risk_flags.append("存在开板分歧")
        if limit_last_time and limit_last_time.isdigit() and int(limit_last_time) >= 145000:
            risk_score += 0.8
            risk_flags.append("涨停封板偏晚")
        if limit_first_time and limit_first_time.isdigit() and int(limit_first_time) >= 140000:
            risk_score += 0.4
            risk_flags.append("首次封板偏晚")
        one_word_limit_flag = bool(
            relay_limit
            and pct_change >= ONE_WORD_LIMIT_PCT_CHANGE_MIN
            and open_times_value <= 0
            and intraday_range_pct is not None
            and intraday_range_pct <= ONE_WORD_LIMIT_MAX_INTRADAY_RANGE_PCT
            and latest_turnover <= ONE_WORD_LIMIT_LOW_TURNOVER_MAX
        )
        if one_word_limit_flag:
            risk_score += 0.5
            risk_flags.append("一字封死且换手偏低，次日可买性待确认")
        if top_net_amount < 0:
            risk_score += 0.9
            risk_flags.append("龙虎榜净卖出")
        if top_net_rate is not None and top_net_rate < 0:
            risk_score += 0.7
            risk_flags.append("龙虎榜净买占比转负")
        if large_order_net_inflow < 0:
            risk_score += 0.7
            risk_flags.append("大单资金承接偏弱")
        if super_large_order_net_inflow < 0:
            risk_score += 0.8
            risk_flags.append("超大单资金承接偏弱")

        late_stage_momentum_flag = (
            price_position >= DISTRIBUTION_PRICE_POSITION_HIGH
            and turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_HIGH
            and volume_ratio >= DISTRIBUTION_VOLUME_RATIO_HIGH
            and recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH
            and moneyflow_3d_value <= 0
        )
        high_level_pullback_flag = (
            price_position >= DISTRIBUTION_PRICE_POSITION_HIGH
            and recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH
            and latest_weakening
            and moneyflow_3d_value <= 5000
            and high_turnover_active
            and volume_ratio >= DISTRIBUTION_VOLUME_RATIO_HIGH
        )
        if late_stage_momentum_flag:
            risk_score += 1.0
            risk_flags.append("疑似末端分歧")
        if high_level_pullback_flag:
            risk_score += 1.1
            risk_flags.append("高位回调且承接不足")
        if deep_drawdown_rebound_flag:
            risk_score += 1.2
            risk_flags.append("前高大幅回撤后的涨停反抽")
        if theme_support_absent_flag:
            risk_score += 0.8
            risk_flags.append("高位运行但题材承接不足")
        if failed_trend_flag:
            risk_score += 1.0
            risk_flags.append("高位连续低开/收跌，趋势失效")
        if risk_score <= 0 and risk_data_missing_flags:
            risk_score += 0.5
            risk_flags.append("风险数据不完整，不能按零风险处理")
        elif risk_data_missing_flags:
            risk_score += 0.3
            risk_flags.append("风险数据不完整，评估从保守处理")

        distribution_risk_score = round(risk_score, 2)
        relay_veto = open_times_value >= 3 or (
            open_times_value >= 2 and limit_last_time.isdigit() and int(limit_last_time) >= 145000
        ) or (top_net_amount < 0 and top_net_rate is not None and top_net_rate <= -3.0)
        return {
            "distribution_risk_score": distribution_risk_score,
            "distribution_risk_flags": risk_flags,
            "risk_data_missing_flags": risk_data_missing_flags,
            "risk_data_incomplete": bool(risk_data_missing_flags),
            "sparse_history_flag": len(daily_rows or []) < 5,
            "moneyflow_3d_value": round(moneyflow_3d_value, 2),
            "large_order_net_inflow": round(large_order_net_inflow, 2),
            "super_large_order_net_inflow": round(super_large_order_net_inflow, 2),
            "price_position_20d": round(price_position, 6),
            "turnover_spike_ratio": round(turnover_spike_ratio, 2),
            "recent_runup_5d": round(recent_runup_5d, 2),
            "relay_open_times": open_times_value,
            "relay_limit_last_time": limit_last_time or None,
            "relay_limit_first_time": limit_first_time or None,
            "one_word_limit_flag": one_word_limit_flag,
            "intraday_range_pct": round(intraday_range_pct, 4) if intraday_range_pct is not None else None,
            "relay_top_net_amount": round(top_net_amount, 2),
            "relay_top_net_rate": round(top_net_rate, 2) if top_net_rate is not None else None,
            "late_stage_momentum_flag": late_stage_momentum_flag,
            "latest_weakening_flag": latest_weakening,
            "high_level_pullback_flag": high_level_pullback_flag,
            "theme_support_absent_flag": theme_support_absent_flag,
            "high_level_failed_trend_flag": failed_trend_flag,
            "high_level_failed_trend_signal": failed_trend_signal,
            "deep_drawdown_rebound_flag": deep_drawdown_rebound_flag,
            "deep_drawdown_rebound_signal": deep_drawdown_rebound_signal,
            "relay_candidate_veto": relay_veto,
            "candidate_risk_blocked": distribution_risk_score >= DISTRIBUTION_RISK_BLOCK_SCORE or relay_veto,
        }

    def _build_stock_moneyflow_summary(self, ts_code: str, *, trade_date: Optional[str] = None) -> Optional[Dict[str, float]]:
        resolved_trade_date = trade_date or self.screener._get_latest_trade_date()
        if not self._is_valid_trade_date_text(resolved_trade_date):
            rows = self.screener.client.fetch_moneyflow(ts_code)
            fallback_summary = self._summarize_fetched_moneyflow_rows(rows)
            if fallback_summary:
                return fallback_summary
            return {
                "recent_3d_net_inflow": 0.0,
                "recent_large_order_net_inflow": 0.0,
                "recent_super_large_order_net_inflow": 0.0,
                "positive_flag": 0.0,
                "moneyflow_data_missing": True,
                "moneyflow_data_stale": False,
            }
        summaries = self.market_raw_data_repo.get_moneyflow_summaries_by_trade_date(
            ts_codes=[ts_code],
            trade_date=resolved_trade_date,
            lookback_days=3,
        )
        summary = summaries.get(ts_code)
        if summary and not bool(summary.get("stale_for_trade_date", False)):
            return {
                "recent_3d_net_inflow": float(summary.get("recent_3d_net_inflow") or 0.0),
                "recent_large_order_net_inflow": float(summary.get("recent_large_order_net_inflow") or 0.0),
                "recent_super_large_order_net_inflow": float(summary.get("recent_super_large_order_net_inflow") or 0.0),
                "positive_flag": float(summary.get("positive_flag") or 0.0),
                "moneyflow_data_missing": False,
                "moneyflow_data_stale": False,
            }
        return {
            "recent_3d_net_inflow": 0.0,
            "recent_large_order_net_inflow": 0.0,
            "recent_super_large_order_net_inflow": 0.0,
            "positive_flag": 0.0,
            "moneyflow_data_missing": True,
            "moneyflow_data_stale": bool(summary.get("stale_for_trade_date", False)) if summary else False,
        }

    @staticmethod
    def _is_valid_trade_date_text(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        text = value.strip()
        if len(text) == 8 and text.isdigit():
            return True
        if len(text) == 10:
            try:
                datetime.strptime(text, "%Y-%m-%d")
                return True
            except ValueError:
                return False
        return False

    @classmethod
    def _summarize_fetched_moneyflow_rows(cls, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        recent_rows = sorted(rows, key=lambda item: str(item.get("trade_date") or ""), reverse=True)[:3]
        net_inflow = 0.0
        large_net_inflow = 0.0
        super_large_net_inflows: List[float] = []
        for item in recent_rows:
            net_inflow += cls._safe_float(item.get("net_mf_amount")) or 0.0
            large_net_inflow += (
                (cls._safe_float(item.get("buy_lg_amount")) or 0.0)
                - (cls._safe_float(item.get("sell_lg_amount")) or 0.0)
            )
            super_large_net_inflows.append(
                (cls._safe_float(item.get("buy_elg_amount")) or 0.0)
                - (cls._safe_float(item.get("sell_elg_amount")) or 0.0)
            )
        super_large_net_inflow = sum(super_large_net_inflows)
        return {
            "recent_3d_net_inflow": round(net_inflow, 2),
            "recent_large_order_net_inflow": round(large_net_inflow, 2),
            "recent_super_large_order_net_inflow": round(super_large_net_inflow, 2),
            "super_large_order_net_inflow_negative_days_3d": sum(1 for value in super_large_net_inflows if value < 0),
            "positive_flag": 1.0 if net_inflow > 0 else 0.0,
            "moneyflow_data_missing": False,
            "moneyflow_data_stale": False,
        }

    def _build_turnover_spike_ratio(self, daily_rows: List[Dict[str, Any]], current_turnover_rate: float) -> float:
        if not daily_rows or current_turnover_rate <= 0:
            return 0.0
        recent_rows = sorted(daily_rows, key=lambda item: str(item.get("trade_date") or ""), reverse=True)
        history_rates: List[float] = []
        for item in recent_rows[1:6]:
            value = self._safe_float(item.get("turnover_rate"))
            if value is not None and value > 0:
                history_rates.append(value)
        if not history_rates:
            return 0.0
        baseline = sum(history_rates) / len(history_rates)
        if baseline <= 0:
            return 0.0
        return current_turnover_rate / baseline

    def _build_recent_runup_5d(self, daily_rows: List[Dict[str, Any]]) -> float:
        if not daily_rows:
            return 0.0
        recent_rows = sorted(daily_rows, key=lambda item: str(item.get("trade_date") or ""))
        target_rows = recent_rows[-5:] if len(recent_rows) >= 5 else recent_rows
        total = 0.0
        for item in target_rows:
            total += self._safe_float(item.get("pct_chg")) or 0.0
        return total

    def _build_high_level_failed_trend_signal(self, daily_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(daily_rows or []) < 5:
            return {"high_level_failed_trend_flag": False}
        recent_rows = sorted(daily_rows, key=lambda item: str(item.get("trade_date") or ""))
        last_three = recent_rows[-3:]
        gap_down_days = 0
        down_close_days = 0
        for idx, row in enumerate(last_three):
            close_value = self._safe_float(row.get("close"))
            open_value = self._safe_float(row.get("open"))
            pct_change = self._safe_float(row.get("pct_chg"))
            if pct_change is None:
                pct_change = self._safe_float(row.get("pct_change"))
            if pct_change is not None and pct_change < 0:
                down_close_days += 1
            global_idx = len(recent_rows) - len(last_three) + idx
            if global_idx > 0 and open_value is not None:
                prev_close = self._safe_float(recent_rows[global_idx - 1].get("close"))
                if prev_close not in (None, 0) and (open_value - prev_close) / prev_close * 100.0 <= -0.5:
                    gap_down_days += 1
        recent_high = max(
            self._safe_float(row.get("high")) or 0.0
            for row in recent_rows[-10:]
        )
        latest_close = self._safe_float(recent_rows[-1].get("close")) or 0.0
        drawdown_pct = 0.0
        if recent_high > 0 and latest_close > 0:
            drawdown_pct = (recent_high - latest_close) / recent_high * 100.0
        recent_runup_5d = self._build_recent_runup_5d(recent_rows)
        previous_runup_5d = self._build_recent_runup_5d(recent_rows[:-2]) if len(recent_rows) >= 7 else 0.0
        high_level_context = bool(
            recent_runup_5d >= HIGH_LEVEL_FAILED_TREND_RECENT_RUNUP_MIN
            or previous_runup_5d >= HIGH_LEVEL_FAILED_TREND_RECENT_RUNUP_MIN
            or drawdown_pct >= HIGH_LEVEL_FAILED_TREND_DRAWDOWN_PCT_MIN
        )
        failed = bool(
            high_level_context
            and drawdown_pct >= HIGH_LEVEL_FAILED_TREND_DRAWDOWN_PCT_MIN
            and (
                gap_down_days >= HIGH_LEVEL_FAILED_TREND_GAP_DOWN_DAYS_MIN
                or down_close_days >= HIGH_LEVEL_FAILED_TREND_DOWN_DAYS_MIN
            )
        )
        return {
            "high_level_failed_trend_flag": failed,
            "failed_trend_gap_down_days_3d": gap_down_days,
            "failed_trend_down_close_days_3d": down_close_days,
            "failed_trend_drawdown_pct": round(drawdown_pct, 4),
            "failed_trend_recent_runup_5d": round(recent_runup_5d, 4),
            "failed_trend_previous_runup_5d": round(previous_runup_5d, 4),
        }

    def _build_deep_drawdown_rebound_signal(
        self,
        daily_rows: List[Dict[str, Any]],
        *,
        pct_change: float,
    ) -> Dict[str, Any]:
        if len(daily_rows or []) < 8:
            return {"deep_drawdown_rebound_flag": False}
        recent_rows = sorted(daily_rows, key=lambda item: str(item.get("trade_date") or ""))
        latest = recent_rows[-1]
        latest_close = self._safe_float(latest.get("close")) or 0.0
        if latest_close <= 0:
            return {"deep_drawdown_rebound_flag": False}
        prior_rows = recent_rows[:-1]
        recent_window = prior_rows[-20:] if len(prior_rows) > 20 else prior_rows
        prior_high = max((self._safe_float(row.get("high")) or 0.0) for row in recent_window)
        recent_low = min(
            (self._safe_float(row.get("low")) or self._safe_float(row.get("close")) or latest_close)
            for row in recent_window[-10:]
        )
        drawdown_from_high_pct = (prior_high - latest_close) / prior_high * 100.0 if prior_high > 0 else 0.0
        rebound_from_low_pct = (latest_close - recent_low) / recent_low * 100.0 if recent_low > 0 else 0.0
        flag = bool(
            pct_change >= TOP3_DEEP_DRAWDOWN_REBOUND_PCT_CHANGE_MIN
            and drawdown_from_high_pct >= TOP3_DEEP_DRAWDOWN_REBOUND_DRAWDOWN_MIN
            and rebound_from_low_pct >= TOP3_DEEP_DRAWDOWN_REBOUND_RECENT_LOW_REBOUND_MIN
        )
        return {
            "deep_drawdown_rebound_flag": flag,
            "deep_drawdown_rebound_drawdown_from_high_pct": round(drawdown_from_high_pct, 4),
            "deep_drawdown_rebound_from_low_pct": round(rebound_from_low_pct, 4),
            "deep_drawdown_rebound_prior_high": round(prior_high, 4) if prior_high else None,
            "deep_drawdown_rebound_recent_low": round(recent_low, 4) if recent_low else None,
            "deep_drawdown_rebound_pct_change": round(float(pct_change or 0.0), 4),
        }

    def _is_st_stock_for_final_veto(
        self,
        *,
        code: str,
        payload: Dict[str, Any],
        stock_name_map: Dict[str, str],
    ) -> bool:
        names = [
            stock_name_map.get(code),
            payload.get("name"),
            payload.get("stock_name"),
            payload.get("display_name"),
        ]
        basic_name = self._get_stock_basic_name(code)
        if basic_name:
            names.append(basic_name)
        return any(self._name_indicates_st(name) for name in names)

    def _get_stock_basic_name(self, code: str) -> Optional[str]:
        if not code:
            return None
        if self._stock_name_cache is None:
            self._stock_name_cache = {}
            try:
                session = DatabaseManager(self.settings.database_url).get_session()
                try:
                    for row in session.query(MarketStockBasic.ts_code, MarketStockBasic.name).all():
                        if row.ts_code and row.name:
                            self._stock_name_cache[str(row.ts_code).upper()] = str(row.name).strip()
                finally:
                    session.close()
            except Exception as exc:
                logger.warning("Failed to load stock basic names for ST veto: %s", exc)
        return self._stock_name_cache.get(str(code).upper())

    @classmethod
    def _name_indicates_st(cls, name: Any) -> bool:
        text = str(name or "").strip().upper().replace(" ", "")
        return bool(text) and (text.startswith("ST") or text.startswith("*ST") or "退" in text[:3])

    @classmethod
    def _get_top3_quality_floor_reason(cls, payload: Dict[str, Any]) -> Optional[str]:
        risk_adjusted_fusion = cls._safe_float(payload.get("risk_adjusted_fusion_score"))
        overall_score = cls._safe_float(payload.get("overall_score_norm"))
        if risk_adjusted_fusion is None or overall_score is None:
            return None
        if risk_adjusted_fusion >= TOP3_QUALITY_FLOOR_STRONG_FUSION_MIN:
            return None
        if risk_adjusted_fusion >= TOP3_QUALITY_FLOOR_FUSION_MIN and overall_score >= TOP3_QUALITY_FLOOR_OVERALL_MIN:
            return None
        if cls._has_top3_quality_floor_strong_confirmation(payload):
            return None
        if risk_adjusted_fusion < TOP3_QUALITY_FLOOR_FUSION_MIN and overall_score < TOP3_QUALITY_FLOOR_OVERALL_MIN:
            return "low_fusion_and_low_overall_quality"
        if risk_adjusted_fusion < TOP3_QUALITY_FLOOR_FUSION_MIN:
            return "low_risk_adjusted_fusion_quality"
        return "low_overall_quality_without_strong_confirmation"

    @classmethod
    def _has_top3_quality_floor_strong_confirmation(cls, payload: Dict[str, Any]) -> bool:
        risk_adjusted_fusion = cls._safe_float(payload.get("risk_adjusted_fusion_score"))
        moneyflow_score = cls._safe_float(payload.get("stage3_moneyflow_score"))
        moneyflow_3d_value = cls._safe_float(payload.get("moneyflow_3d_value"))
        large_order_net_inflow = cls._safe_float(payload.get("recent_large_order_net_inflow"))
        super_large_order_net_inflow = cls._safe_float(payload.get("recent_super_large_order_net_inflow"))
        close_auction_score = cls._safe_float(payload.get("stage3_close_auction_score"))
        strong_moneyflow = bool(
            moneyflow_score is not None
            and moneyflow_score >= TOP3_QUALITY_FLOOR_STRONG_MONEYFLOW_SCORE
            and (moneyflow_3d_value or 0.0) > TOP3_QUALITY_FLOOR_STRONG_MONEYFLOW_VALUE
            and (large_order_net_inflow or 0.0) > TOP3_QUALITY_FLOOR_STRONG_MONEYFLOW_VALUE
            and (super_large_order_net_inflow or 0.0) > TOP3_QUALITY_FLOOR_STRONG_MONEYFLOW_VALUE
        )
        strong_close_auction = bool(
            close_auction_score is not None
            and close_auction_score >= TOP3_QUALITY_FLOOR_STRONG_CLOSE_AUCTION_SCORE
            and not bool(payload.get("stage3_close_auction_veto", False))
            and not bool(payload.get("stage3_close_auction_missing", False))
        )
        if risk_adjusted_fusion is not None and risk_adjusted_fusion < TOP3_QUALITY_FLOOR_FUSION_MIN:
            return strong_moneyflow and strong_close_auction
        return strong_moneyflow or strong_close_auction

    @classmethod
    def _is_strong_quality_for_close_auction_soften(cls, payload: Dict[str, Any]) -> bool:
        risk_adjusted_fusion = cls._safe_float(payload.get("risk_adjusted_fusion_score")) or 0.0
        overall_score_norm = cls._safe_float(payload.get("overall_score_norm")) or 0.0
        moneyflow_score = cls._safe_float(payload.get("stage3_moneyflow_score")) or 0.0
        return bool(
            risk_adjusted_fusion >= CLOSE_AUCTION_SOFTEN_FUSION_MIN
            and overall_score_norm >= CLOSE_AUCTION_SOFTEN_OVERALL_MIN
            and moneyflow_score >= CLOSE_AUCTION_SOFTEN_MONEYFLOW_SCORE_MIN
        )

    @classmethod
    def _get_top3_extreme_risk_reason(cls, payload: Dict[str, Any]) -> Optional[str]:
        pct_change = cls._safe_float(payload.get("pct_change"))
        if pct_change is None:
            pct_change = cls._safe_float(payload.get("pct_chg")) or 0.0
        if pct_change >= TOP3_NEAR_LIMIT_UP_PCT_CHANGE_MIN:
            return "near_limit_up_pct_change"
        if bool(payload.get("candidate_risk_blocked", False)):
            return "candidate_risk_blocked"
        if bool(payload.get("relay_candidate_veto", False)):
            return "relay_candidate_veto"
        if bool(payload.get("stage3_moneyflow_veto", False)):
            return "stage3_moneyflow_veto"
        if bool(payload.get("stage3_close_auction_veto", False)):
            if cls._is_strong_quality_for_close_auction_soften(payload):
                payload["stage3_close_auction_veto_softened"] = True
            else:
                return "stage3_close_auction_veto"
        distribution_risk_score = payload.get("distribution_risk_score")
        if distribution_risk_score is not None and float(distribution_risk_score) >= DISTRIBUTION_RISK_BLOCK_SCORE:
            return "distribution_risk_score_block"
        if cls._is_weak_market_high_position_top3_veto(payload):
            return "weak_market_high_position_top3_veto"
        if cls._is_high_position_super_large_outflow_veto(payload):
            return "high_position_super_large_outflow_top3_veto"
        if cls._is_rebound_distribution_super_large_outflow_veto(payload):
            return "rebound_distribution_super_large_outflow_top3_veto"
        if bool(payload.get("deep_drawdown_rebound_flag", False)):
            return "deep_drawdown_rebound_top3_veto"
        # Keep failed-trend diagnostics observable without affecting the current Top3 selection path.
        # Re-enable only after a dedicated server-side 603399.SH window validates the trade-off.
        if False and cls._is_high_level_failed_trend_top3_veto(payload):
            return "high_level_failed_trend_top3_veto"
        distribution_risk_value = cls._safe_float(distribution_risk_score) or 0.0
        moneyflow_3d_value = cls._safe_float(payload.get("moneyflow_3d_value")) or 0.0
        large_order_net_inflow = cls._safe_float(payload.get("recent_large_order_net_inflow")) or 0.0
        super_large_order_net_inflow = cls._safe_float(payload.get("recent_super_large_order_net_inflow")) or 0.0
        weak_moneyflow = moneyflow_3d_value <= 0 or large_order_net_inflow < 0 or super_large_order_net_inflow < 0
        if bool(payload.get("late_stage_momentum_flag", False)) and weak_moneyflow:
            return "late_stage_momentum_top3_veto"
        if bool(payload.get("high_level_pullback_flag", False)) and weak_moneyflow:
            return "high_level_pullback_top3_veto"
        if bool(payload.get("unsupported_high_position_flag", False)) and weak_moneyflow:
            return "unsupported_high_position_weak_moneyflow_top3_veto"
        return None

    @classmethod
    def _is_high_position_super_large_outflow_veto(cls, payload: Dict[str, Any]) -> bool:
        price_position = cls._safe_float(payload.get("price_position_20d")) or 0.0
        recent_runup_5d = cls._safe_float(payload.get("recent_runup_5d")) or 0.0
        pct_change = cls._safe_float(payload.get("pct_change"))
        if pct_change is None:
            pct_change = cls._safe_float(payload.get("pct_chg")) or 0.0
        negative_days = cls._safe_float(
            payload.get("super_large_order_net_inflow_negative_days_3d")
        ) or 0.0
        accelerated_or_chased = (
            recent_runup_5d >= TOP3_HIGH_POSITION_EVENT_RUNUP_NEAR_THRESHOLD
            or pct_change >= TOP3_HIGH_POSITION_EVENT_PCT_CHANGE_MIN
        )
        return bool(
            price_position >= TOP3_HIGH_POSITION_EVENT_RISK_THRESHOLD
            and negative_days >= TOP3_SUPER_LARGE_OUTFLOW_NEGATIVE_DAYS_MIN
            and accelerated_or_chased
        )

    @classmethod
    def _is_rebound_distribution_super_large_outflow_veto(cls, payload: Dict[str, Any]) -> bool:
        moneyflow_3d_value = cls._safe_float(payload.get("moneyflow_3d_value")) or 0.0
        recent_super_large_order_net_inflow = cls._safe_float(
            payload.get("recent_super_large_order_net_inflow")
        ) or 0.0
        recent_runup_5d = cls._safe_float(payload.get("recent_runup_5d")) or 0.0
        pct_change = cls._safe_float(payload.get("pct_change"))
        if pct_change is None:
            pct_change = cls._safe_float(payload.get("pct_chg")) or 0.0
        negative_days = cls._safe_float(
            payload.get("super_large_order_net_inflow_negative_days_3d")
        ) or 0.0
        return bool(
            moneyflow_3d_value > 0
            and recent_super_large_order_net_inflow < 0
            and negative_days >= TOP3_SUPER_LARGE_OUTFLOW_NEGATIVE_DAYS_MIN
            and recent_runup_5d >= TOP3_REBOUND_DISTRIBUTION_RUNUP_MIN
            and pct_change >= TOP3_REBOUND_DISTRIBUTION_PCT_CHANGE_MIN
        )

    @classmethod
    def _is_high_level_failed_trend_top3_veto(cls, payload: Dict[str, Any]) -> bool:
        if not bool(payload.get("high_level_failed_trend_flag", False)):
            return False
        moneyflow_3d_value = cls._safe_float(payload.get("moneyflow_3d_value")) or 0.0
        large_order_net_inflow = cls._safe_float(payload.get("recent_large_order_net_inflow"))
        if large_order_net_inflow is None:
            large_order_net_inflow = cls._safe_float(payload.get("large_order_net_inflow")) or 0.0
        super_large_order_net_inflow = cls._safe_float(
            payload.get("recent_super_large_order_net_inflow")
        )
        if super_large_order_net_inflow is None:
            super_large_order_net_inflow = cls._safe_float(payload.get("super_large_order_net_inflow")) or 0.0
        weak_moneyflow = moneyflow_3d_value <= 0 or large_order_net_inflow < 0 or super_large_order_net_inflow < 0
        weak_price_action = bool(payload.get("latest_weakening_flag", False)) or bool(payload.get("stage3_close_auction_veto", False))
        if not weak_price_action:
            signal = payload.get("high_level_failed_trend_signal") or {}
            if isinstance(signal, dict):
                down_days = cls._safe_float(signal.get("failed_trend_down_close_days_3d")) or 0.0
                gap_days = cls._safe_float(signal.get("failed_trend_gap_down_days_3d")) or 0.0
                weak_price_action = bool(
                    down_days >= HIGH_LEVEL_FAILED_TREND_DOWN_DAYS_MIN
                    or gap_days >= HIGH_LEVEL_FAILED_TREND_GAP_DOWN_DAYS_MIN
                )
        return bool(weak_moneyflow and weak_price_action)

    @classmethod
    def _is_weak_market_high_position_top3_veto(cls, payload: Dict[str, Any]) -> bool:
        price_position = cls._safe_float(payload.get("price_position_20d")) or 0.0
        if price_position < TOP3_WEAK_MARKET_HIGH_POSITION_THRESHOLD:
            return False

        weak_market_flag = payload.get("weak_market_flag")
        if weak_market_flag is not None:
            if isinstance(weak_market_flag, str):
                return weak_market_flag.strip().lower() in {"1", "true", "yes", "y"}
            return bool(weak_market_flag)

        market_return_1d = cls._safe_float(payload.get("market_return_1d"))
        market_return_3d = cls._safe_float(payload.get("market_return_3d"))
        market_up_ratio_1d = cls._safe_float(payload.get("market_up_ratio_1d"))
        market_up_ratio_3d_avg = cls._safe_float(payload.get("market_up_ratio_3d_avg"))
        market_up_days_5d = cls._safe_float(payload.get("market_up_days_5d"))

        return bool(
            (market_return_1d is not None and market_return_1d < 0)
            or (market_return_3d is not None and market_return_3d < 0)
            or (market_up_ratio_1d is not None and market_up_ratio_1d < 0.45)
            or (market_up_ratio_3d_avg is not None and market_up_ratio_3d_avg < 0.45)
            or (market_up_days_5d is not None and market_up_days_5d <= 2)
        )

    @classmethod
    def _build_close_auction_signal(cls, payload: Dict[str, Any], auction_row: Dict[str, Any]) -> Dict[str, Any]:
        if not auction_row:
            return {
                "stage3_close_auction_score": 0.0,
                "stage3_close_auction_flags": [],
                "stage3_close_auction_risks": [],
                "stage3_close_auction_veto": False,
                "stage3_close_auction_missing": True,
            }

        close_price = cls._safe_float(payload.get("close"))
        auction_close = cls._first_numeric(
            auction_row,
            "price",
            "auction_price",
            "match_price",
            "final_price",
            "close",
        )
        auction_vwap = cls._first_numeric(auction_row, "vwap", "avg_price", "average_price")
        auction_price_for_signal = auction_vwap if auction_vwap is not None else auction_close
        auction_amount = cls._first_numeric(
            auction_row,
            "amount",
            "amt",
            "auction_amount",
            "match_amount",
        )
        auction_vol = cls._first_numeric(
            auction_row,
            "vol",
            "volume",
            "auction_vol",
            "match_vol",
        )
        daily_amount = cls._safe_float(payload.get("amount"))
        pct_change = cls._safe_float(payload.get("pct_change")) or 0.0

        price_deviation_pct = None
        if close_price not in (None, 0) and auction_price_for_signal is not None:
            price_deviation_pct = (auction_price_for_signal - close_price) / close_price * 100.0
        amount_ratio = cls._normalize_close_auction_amount_ratio(auction_amount, daily_amount)

        score = 0.0
        flags: List[str] = []
        risks: List[str] = []
        if price_deviation_pct is not None:
            if price_deviation_pct >= CLOSE_AUCTION_STRONG_DEVIATION_PCT:
                flags.append("close_auction_price_support")
            elif price_deviation_pct <= CLOSE_AUCTION_WEAK_DEVIATION_PCT:
                score -= 2.0
                risks.append("close_auction_price_weak")
        if amount_ratio is not None and amount_ratio >= CLOSE_AUCTION_HIGH_AMOUNT_RATIO:
            if price_deviation_pct is not None and price_deviation_pct >= 0:
                flags.append("close_auction_active_support")
            elif price_deviation_pct is not None:
                score -= 1.2
                risks.append("close_auction_active_distribution")

        veto = bool(
            price_deviation_pct is not None
            and price_deviation_pct <= CLOSE_AUCTION_FAKE_STRONG_DEVIATION_PCT
            and pct_change >= CLOSE_AUCTION_FAKE_STRONG_PCT_CHANGE_MIN
        )
        if veto:
            score -= 2.5
            risks.append("close_auction_fake_strength_veto")

        return {
            "close_auction_raw": dict(auction_row),
            "close_auction_price": round(auction_close, 4) if auction_close is not None else None,
            "close_auction_amount": round(auction_amount, 4) if auction_amount is not None else None,
            "close_auction_vol": round(auction_vol, 4) if auction_vol is not None else None,
            "close_auction_vwap": round(auction_vwap, 4) if auction_vwap is not None else None,
            "close_auction_price_deviation_pct": round(price_deviation_pct, 4) if price_deviation_pct is not None else None,
            "close_auction_amount_ratio": round(amount_ratio, 6) if amount_ratio is not None else None,
            "stage3_close_auction_score": round(score, 4),
            "stage3_close_auction_flags": flags,
            "stage3_close_auction_risks": risks,
            "stage3_close_auction_veto": veto,
            "stage3_close_auction_missing": False,
        }

    @classmethod
    def _first_numeric(cls, row: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            value = cls._safe_float(row.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _normalize_close_auction_amount_ratio(cls, 
        auction_amount: Optional[float],
        daily_amount: Optional[float],
    ) -> Optional[float]:
        if auction_amount in (None, 0) or daily_amount in (None, 0):
            return None
        auction_value = float(auction_amount)
        daily_value = float(daily_amount)
        if auction_value > daily_value and auction_value / 1000.0 <= daily_value:
            auction_value = auction_value / 1000.0
        ratio = auction_value / daily_value
        return ratio if 0 <= ratio <= 1.0 else None
