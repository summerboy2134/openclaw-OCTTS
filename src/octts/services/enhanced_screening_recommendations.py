"""Mixin helpers for enhanced screening scheduler."""

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
from octts.services.position_store import create_position_store
from octts.services.regression_rerank_service import RegressionRerankResult
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_constants import *

logger = logging.getLogger(__name__)


class EnhancedScreeningRecommendationsMixin:
    def _build_backfill_ai_analyses(
        self,
        stock_codes: List[str],
        *,
        screening_results: Dict[str, ScreenResult],
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        stock_map = self._build_screened_stock_map(screening_results)
        analyses: Dict[str, Dict[str, Any]] = {}
        for code in stock_codes:
            stock = stock_map.get(code)
            if stock is None:
                continue
            technical_score = float(getattr(stock, "technical_score", 0.0) or 0.0)
            recommendation_score = float(getattr(stock, "recommendation_score", 0.0) or 0.0)
            fundamental_score = self._estimate_backfill_fundamental_score(code)
            sentiment_score = self._estimate_backfill_sentiment_score(stock)
            news_score = 50.0
            base_score = round(technical_score * 0.7 + fundamental_score * 0.3, 2)
            overall_score = round(base_score + (sentiment_score - 50.0) * 0.08 + (news_score - 50.0) * 0.05, 2)
            overall_confidence = round(min(0.88, max(0.62, 0.64 + recommendation_score / 400.0)), 4)
            analyses[code] = {
                "ts_code": code,
                "name": getattr(stock, "name", code),
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "sentiment_score": sentiment_score,
                "news_score": news_score,
                "base_score": base_score,
                "overall_score": overall_score,
                "overall_confidence": overall_confidence,
                "technical_signal": getattr(stock, "trend_status", None) or getattr(stock, "momentum_status", None) or "",
                "summary": f"{getattr(stock, 'name', code)}回补样本使用结构化特征近似生成综合分，用于短线训练集构造。",
                "score_model": "backfill_structured_v1",
                "sentiment_adjustment": round((sentiment_score - 50.0) * 0.08, 2),
                "news_adjustment": round((news_score - 50.0) * 0.05, 2),
            }
        return analyses

    def _build_backfill_final_recommendations(
        self,
        *,
        trade_date: date,
        screening_results: Dict[str, ScreenResult],
        ai_analyses: Dict[str, Dict[str, Any]],
        market_snapshot: Optional[Dict[str, Any]] = None,
        candidate_codes: Optional[List[str]] = None,
        top20_limit: int = STAGE2_TOP20_LIMIT,
    ) -> Dict[str, Dict[str, Any]]:
        stock_map = self._build_screened_stock_map(screening_results)
        all_market_stock_map = self._build_all_market_stock_map(market_snapshot)
        industry_adjustments = self._build_industry_flow_adjustments(
            stock_map,
            all_market_stock_map=all_market_stock_map,
            market_snapshot=market_snapshot,
            trade_date=trade_date.strftime("%Y%m%d"),
        )
        distribution_risk_map = self._build_distribution_risk_map(
            stock_map,
            market_snapshot=market_snapshot,
            trade_date=trade_date.strftime("%Y%m%d"),
        )
        theme_support_map = self._build_theme_support_map(
            stock_map,
            news_clusters=[],
            news_hot_stocks=set(),
            industry_adjustments=industry_adjustments,
            distribution_risk_map=distribution_risk_map,
            screening_results=screening_results,
        )
        previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(trade_date)
        stage1_candidate_set = set(candidate_codes or ai_analyses.keys())
        previous_states = {
            item.get("ts_code"): item
            for item in self.store.load_recommendation_pool_state(trade_date=previous_trade_date)
            if previous_trade_date and item.get("ts_code")
        } if previous_trade_date else {}
        previous_top3_codes = set(self._get_previous_top3_codes(trade_date))

        final_recommendations: Dict[str, Dict[str, Any]] = {}
        stage2_scores: Dict[str, float] = {}
        for code, analysis in ai_analyses.items():
            stock = stock_map.get(code)
            if stock is None:
                continue
            confidence = float(analysis.get("overall_confidence") or 0.65)
            overall_score = float(analysis.get("overall_score") or 50.0)
            stock_recommendation_score = float(getattr(stock, "recommendation_score", 0.0) or 0.0)
            appearance_count = 0
            industry_adjustment = industry_adjustments.get(code, {})
            distribution_risk = distribution_risk_map.get(code, {})
            theme_support = theme_support_map.get(code, {})
            industry_heat_score = float(industry_adjustment.get("industry_heat_score") or 0.0)
            distribution_risk_score = float(distribution_risk.get("distribution_risk_score") or 0.0)
            contradiction_penalty = self._build_short_term_contradiction_penalty(
                {
                    **analysis,
                    **distribution_risk,
                    "recommendation_text": self._generate_recommendation(
                        stock_recommendation_score,
                        analysis,
                        distribution_risk=distribution_risk,
                    ),
                }
            )
            fundamental_bonus = self._build_light_fundamental_bonus(analysis)
            continuation_bias_score, continuation_positive_flags, continuation_negative_flags = self._build_continuation_bias(
                code,
                analysis,
                distribution_risk=distribution_risk,
                theme_support=theme_support,
                industry_adjustment=industry_adjustment,
                strategy_count=appearance_count,
                previous_state=previous_states.get(code),
                is_previous_top3=code in previous_top3_codes,
            )
            ranking_risk_penalty = round(
                max(
                    0.0,
                    distribution_risk_score * TOP10_RISK_PENALTY_MULTIPLIER
                    + contradiction_penalty
                    + (1.2 if bool(theme_support.get("unsupported_high_position_flag", False)) else 0.0),
                ),
                2,
            )
            adjusted_final_score = overall_score + industry_heat_score + fundamental_bonus["total_bonus"] - ranking_risk_penalty
            continuation_adjustment = min(max(continuation_bias_score, 0.0), 2.0)
            execution_score = round(
                max(
                    0.0,
                    stock_recommendation_score * 0.55
                    + adjusted_final_score * 0.30
                    + continuation_adjustment * 2.0,
                ) * confidence,
                4,
            )
            weighted_score = execution_score
            recommendation_text = self._generate_recommendation(
                weighted_score,
                analysis,
                distribution_risk=distribution_risk,
            )
            stage2_scores[code] = execution_score
            final_recommendations[code] = {
                "score": weighted_score,
                "overall_score": overall_score,
                "final_score": overall_score,
                "adjusted_final_score": adjusted_final_score,
                "weighted_score": weighted_score,
                "ranking_risk_penalty": ranking_risk_penalty,
                "fundamental_bonus": fundamental_bonus["total_bonus"],
                "fundamental_bonus_breakdown": fundamental_bonus["breakdown"],
                "continuation_bias_score": continuation_bias_score,
                "continuation_adjustment": continuation_adjustment,
                "continuation_positive_flags": continuation_positive_flags,
                "continuation_negative_flags": continuation_negative_flags,
                "ai_confidence": confidence,
                "overall_confidence": confidence,
                "ai_summary": analysis.get("summary", ""),
                "summary": analysis.get("summary", ""),
                "technical_signal": analysis.get("technical_signal", ""),
                "technical_score": analysis.get("technical_score"),
                "fundamental_score": analysis.get("fundamental_score"),
                "sentiment_score": analysis.get("sentiment_score"),
                "news_score": analysis.get("news_score"),
                "base_score": analysis.get("base_score"),
                "sentiment_adjustment": analysis.get("sentiment_adjustment"),
                "news_adjustment": analysis.get("news_adjustment"),
                "score_model": analysis.get("score_model"),
                "close": getattr(stock, "close", None),
                "pct_change": getattr(stock, "pct_change", None),
                "volume_ratio": getattr(stock, "volume_ratio", None),
                "turnover_rate": getattr(stock, "turnover_rate", None),
                "ma20": getattr(stock, "ma20", None),
                "price_position_20d": getattr(stock, "price_position_20d", None),
                "news_mentioned": False,
                "strategy_count": appearance_count,
                "industry": industry_adjustment.get("industry"),
                "industry_heat_score": industry_heat_score,
                "industry_flow_bias": industry_adjustment.get("industry_flow_bias", "中性"),
                "industry_positive_ratio": industry_adjustment.get("industry_positive_ratio"),
                "industry_3d_net_inflow": industry_adjustment.get("industry_3d_net_inflow"),
                "industry_flow_value": industry_adjustment.get("industry_flow_value"),
                "theme_support_score": theme_support.get("theme_support_score"),
                "theme_support_label": theme_support.get("theme_support_label"),
                "theme_support_sources": list(theme_support.get("theme_support_sources") or []),
                "unsupported_high_position_flag": bool(theme_support.get("unsupported_high_position_flag", False)),
                "leader_turnover_justified_flag": bool(theme_support.get("leader_turnover_justified_flag", False)),
                "distribution_risk_score": distribution_risk_score,
                "distribution_risk_flags": list(distribution_risk.get("distribution_risk_flags") or []),
                "risk_data_missing_flags": list(distribution_risk.get("risk_data_missing_flags") or []),
                "risk_data_incomplete": bool(distribution_risk.get("risk_data_incomplete", False)),
                "moneyflow_3d_value": distribution_risk.get("moneyflow_3d_value"),
                "turnover_spike_ratio": distribution_risk.get("turnover_spike_ratio"),
                "recent_runup_5d": distribution_risk.get("recent_runup_5d"),
                "relay_open_times": distribution_risk.get("relay_open_times"),
                "relay_limit_last_time": distribution_risk.get("relay_limit_last_time"),
                "relay_limit_first_time": distribution_risk.get("relay_limit_first_time"),
                "one_word_limit_flag": bool(distribution_risk.get("one_word_limit_flag", False)),
                "intraday_range_pct": distribution_risk.get("intraday_range_pct"),
                "relay_top_net_amount": distribution_risk.get("relay_top_net_amount"),
                "relay_top_net_rate": distribution_risk.get("relay_top_net_rate"),
                "late_stage_momentum_flag": bool(distribution_risk.get("late_stage_momentum_flag", False)),
                "latest_weakening_flag": bool(distribution_risk.get("latest_weakening_flag", False)),
                "high_level_pullback_flag": bool(distribution_risk.get("high_level_pullback_flag", False)),
                "theme_support_absent_flag": bool(distribution_risk.get("theme_support_absent_flag", False)),
                "candidate_risk_blocked": bool(distribution_risk.get("candidate_risk_blocked", False)),
                "relay_candidate_veto": bool(distribution_risk.get("relay_candidate_veto", False)),
                "recent_large_order_net_inflow": distribution_risk.get("large_order_net_inflow"),
                "recent_super_large_order_net_inflow": distribution_risk.get("super_large_order_net_inflow"),
                "action_bias": None,
                "recommendation_text": recommendation_text,
                "recommendation": recommendation_text,
            }
            logger.info(
                "Backfill threshold summary for %s: scores=%s, risk=%s, bonus=%s, continuation=%s, confidence=%s",
                code,
                {
                    "overall_score": round(float(overall_score or 0.0), 2),
                    "adjusted_final_score": round(float(adjusted_final_score or 0.0), 2),
                    "weighted_score": round(float(weighted_score or 0.0), 2),
                    "stock_recommendation_score": round(float(stock_recommendation_score or 0.0), 2),
                },
                {
                    "distribution_risk_score": round(float(distribution_risk_score or 0.0), 2),
                    "ranking_risk_penalty": round(float(ranking_risk_penalty or 0.0), 2),
                    "unsupported_high_position_flag": bool(theme_support.get("unsupported_high_position_flag", False)),
                    "distribution_risk_flags": list(distribution_risk.get("distribution_risk_flags") or []),
                },
                {
                    "total_bonus": fundamental_bonus.get("total_bonus"),
                    "breakdown": fundamental_bonus.get("breakdown") or {},
                },
                {
                    "continuation_bias_score": continuation_bias_score,
                    "positive_flags": continuation_positive_flags,
                    "negative_flags": continuation_negative_flags,
                },
                round(float(confidence or 0.0), 4),
            )
        stage2_ranked_codes = [
            code for code, _ in sorted(
                stage2_scores.items(),
                key=lambda item: (
                    bool(final_recommendations.get(item[0], {}).get("candidate_risk_blocked", False)),
                    -float(item[1] or 0.0),
                    item[0],
                ),
            )
            if code in stage1_candidate_set
        ]
        stage2_top20_codes = stage2_ranked_codes[:max(1, top20_limit)]
        final_recommendations = self._build_structured_stage_selection_metadata(
            recommendations=final_recommendations,
            rerank_metadata={},
            stage2_top20_codes=stage2_top20_codes,
            stage3_top3_codes=[],
        )
        return dict(
            sorted(final_recommendations.items(), key=lambda item: item[1]["score"], reverse=True)
        )

    @classmethod
    def _build_continuation_bias(
        cls,
        code: str,
        analysis: Dict[str, Any],
        *,
        distribution_risk: Dict[str, Any],
        theme_support: Dict[str, Any],
        industry_adjustment: Dict[str, Any],
        strategy_count: int,
        previous_state: Optional[Dict[str, Any]],
        is_previous_top3: bool,
    ) -> Any:
        positive_flags: List[str] = []
        negative_flags: List[str] = []
        score = 0.0

        moneyflow_3d_value = float(distribution_risk.get("moneyflow_3d_value") or 0.0)
        if moneyflow_3d_value > 0:
            score += 0.6
            positive_flags.append("3日资金承接为正")
            if moneyflow_3d_value >= 5000:
                score += 0.4
                positive_flags.append("3日资金承接偏强")
        elif moneyflow_3d_value < 0:
            score -= 1.0
            negative_flags.append("3日资金承接转弱")

        industry_heat_score = float(industry_adjustment.get("industry_heat_score") or 0.0)
        if industry_heat_score >= 2.0:
            score += 0.5
            positive_flags.append("板块热度支撑")
        elif industry_heat_score <= -0.5:
            score -= 0.6
            negative_flags.append("板块热度偏弱")

        technical_signal = str(analysis.get("technical_signal") or "")
        if cls._contains_any_keyword(technical_signal, ["突破", "走强", "多头", "放量", "启动"]):
            score += 0.4
            positive_flags.append("技术形态偏向延续")

        if bool(theme_support.get("leader_turnover_justified_flag", False)):
            score += 0.3
            positive_flags.append("龙头换手具备承接")

        large_order_net_inflow = float(distribution_risk.get("large_order_net_inflow") or 0.0)
        if large_order_net_inflow > 0:
            score += 0.3
            positive_flags.append("大单资金承接为正")
        elif large_order_net_inflow < 0:
            score -= 0.7
            negative_flags.append("大单资金承接转弱")

        super_large_order_net_inflow = float(distribution_risk.get("super_large_order_net_inflow") or 0.0)
        if super_large_order_net_inflow > 0:
            score += 0.4
            positive_flags.append("超大单资金承接为正")
        elif super_large_order_net_inflow < 0:
            score -= 0.8
            negative_flags.append("超大单资金承接转弱")

        relay_top_net_amount = float(distribution_risk.get("relay_top_net_amount") or 0.0)
        if relay_top_net_amount > 0:
            score += 0.2
            positive_flags.append("龙虎榜净买为正")
        elif relay_top_net_amount < 0:
            score -= 0.8
            negative_flags.append("龙虎榜净卖偏强")

        relay_open_times = int(distribution_risk.get("relay_open_times") or 0)
        if relay_open_times == 0 and str(distribution_risk.get("relay_limit_first_time") or "").strip():
            score += 0.3
            positive_flags.append("涨停结构干净")
        elif relay_open_times >= 2:
            score -= 1.0
            negative_flags.append("涨停结构分歧较大")

        recent_runup_5d = float(distribution_risk.get("recent_runup_5d") or 0.0)
        if recent_runup_5d >= 10.0:
            score -= 1.1
            negative_flags.append("近5日涨幅偏大")
        elif recent_runup_5d <= 6.0:
            score += 0.2
            positive_flags.append("短线位置未明显透支")

        turnover_spike_ratio = float(distribution_risk.get("turnover_spike_ratio") or 0.0)
        if turnover_spike_ratio >= 2.1:
            score -= 1.0
            negative_flags.append("换手放大过快")
        elif 0 < turnover_spike_ratio <= 1.35:
            score += 0.2
            positive_flags.append("换手节奏相对健康")

        distribution_risk_score = float(distribution_risk.get("distribution_risk_score") or 0.0)
        if distribution_risk_score >= 2.0:
            score -= 1.2
            negative_flags.append("分歧派发风险偏高")
        elif distribution_risk_score <= 1.0:
            score += 0.3
            positive_flags.append("分歧风险可控")

        if bool(distribution_risk.get("latest_weakening_flag", False)):
            score -= 1.0
            negative_flags.append("尾盘转弱")
        if bool(distribution_risk.get("high_level_pullback_flag", False)):
            score -= 1.1
            negative_flags.append("高位回落")
        if bool(distribution_risk.get("theme_support_absent_flag", False)):
            score -= 0.8
            negative_flags.append("题材支持不足")
        if bool(theme_support.get("unsupported_high_position_flag", False)):
            score -= 1.2
            negative_flags.append("高位缺乏板块支撑")

        if is_previous_top3:
            previous_score = float((previous_state or {}).get("recommendation_score") or 0.0)
            current_score = float(analysis.get("overall_score") or 0.0)
            score_change = (previous_state or {}).get("score_change")
            if score_change is not None and float(score_change) >= 0 and distribution_risk_score < TOP10_MAX_DISTRIBUTION_RISK_SCORE:
                score += 0.6
                positive_flags.append("昨日Top3延续未走坏")
            elif previous_score and current_score >= previous_score and not bool(distribution_risk.get("latest_weakening_flag", False)):
                score += 0.4
                positive_flags.append("昨日Top3强度延续")
            elif distribution_risk_score >= TOP10_MAX_DISTRIBUTION_RISK_SCORE or bool(distribution_risk.get("latest_weakening_flag", False)) or bool(distribution_risk.get("high_level_pullback_flag", False)):
                negative_flags.append("昨日Top3但今日风险走弱")

        if bool(distribution_risk.get("candidate_risk_blocked", False)):
            score = min(score, 0.0)

        score = round(max(-CONTINUATION_BIAS_MAX_ABS, min(CONTINUATION_BIAS_MAX_ABS, score)), 2)
        return score, positive_flags, negative_flags

    @classmethod
    def _build_light_fundamental_bonus(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
        signals = analysis.get("fundamental_signals") or {}
        if not isinstance(signals, dict):
            signals = {}

        def _to_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        reasons: List[str] = []
        fina_bonus = 0.0
        express_bonus = 0.0
        forecast_bonus = 0.0

        netprofit_yoy = _to_float(signals.get("netprofit_yoy"))
        dt_netprofit_yoy = _to_float(signals.get("dt_netprofit_yoy"))
        op_income_yoy = _to_float(signals.get("op_income_yoy"))
        roe = _to_float(signals.get("roe"))
        roe_dt = _to_float(signals.get("roe_dt"))
        grossprofit_margin = _to_float(signals.get("grossprofit_margin"))
        netprofit_margin = _to_float(signals.get("netprofit_margin"))
        ocfps = _to_float(signals.get("ocfps"))

        express_yoy_net_profit = _to_float(signals.get("express_yoy_net_profit"))
        express_yoy_sales = _to_float(signals.get("express_yoy_sales"))
        express_diluted_roe = _to_float(signals.get("express_diluted_roe"))
        express_diluted_eps = _to_float(signals.get("express_diluted_eps"))

        forecast_type = str(signals.get("forecast_type") or "").strip()
        forecast_p_change_min = _to_float(signals.get("forecast_p_change_min"))
        forecast_p_change_max = _to_float(signals.get("forecast_p_change_max"))

        if netprofit_yoy is not None:
            if netprofit_yoy >= 50:
                fina_bonus += 1.1
                reasons.append("净利润同比高增")
            elif netprofit_yoy >= 20:
                fina_bonus += 0.7
                reasons.append("净利润同比改善")
            elif netprofit_yoy < -20:
                fina_bonus -= 0.8
                reasons.append("净利润同比走弱")

        if dt_netprofit_yoy is not None:
            if dt_netprofit_yoy >= 20:
                fina_bonus += 0.9
                reasons.append("扣非净利改善")
            elif dt_netprofit_yoy < -20:
                fina_bonus -= 0.8
                reasons.append("扣非净利承压")

        if op_income_yoy is not None:
            if op_income_yoy >= 15:
                fina_bonus += 0.5
                reasons.append("营收同比增长")
            elif op_income_yoy < -10:
                fina_bonus -= 0.5
                reasons.append("营收同比转弱")

        effective_roe = roe if roe is not None else roe_dt
        if effective_roe is not None:
            if effective_roe >= 12:
                fina_bonus += 0.6
                reasons.append("ROE质量较好")
            elif effective_roe < 5:
                fina_bonus -= 0.5
                reasons.append("ROE偏弱")

        if grossprofit_margin is not None and grossprofit_margin >= 25:
            fina_bonus += 0.2
        if netprofit_margin is not None and netprofit_margin >= 8:
            fina_bonus += 0.2
        if ocfps is not None:
            if ocfps > 0:
                fina_bonus += 0.5
                reasons.append("经营现金流为正")
            else:
                fina_bonus -= 0.5
                reasons.append("经营现金流承压")

        if express_yoy_net_profit is not None:
            if express_yoy_net_profit >= 40:
                express_bonus += 1.4
                reasons.append("快报净利润同比高增")
            elif express_yoy_net_profit >= 15:
                express_bonus += 0.8
                reasons.append("快报净利润改善")
            elif express_yoy_net_profit < -20:
                express_bonus -= 0.8
                reasons.append("快报净利润走弱")

        if express_yoy_sales is not None:
            if express_yoy_sales >= 20:
                express_bonus += 0.6
                reasons.append("快报营收增速较快")
            elif express_yoy_sales >= 5:
                express_bonus += 0.3
            elif express_yoy_sales < -10:
                express_bonus -= 0.4
                reasons.append("快报营收转弱")

        if express_diluted_roe is not None:
            if express_diluted_roe >= 12:
                express_bonus += 0.4
            elif express_diluted_roe < 5:
                express_bonus -= 0.3
        if express_diluted_eps is not None:
            if express_diluted_eps > 0:
                express_bonus += 0.3
            elif express_diluted_eps < 0:
                express_bonus -= 0.2

        normalized_forecast_type = forecast_type.lower()
        if any(keyword in forecast_type for keyword in ["预增", "略增", "扭亏", "续盈"]) or any(
            keyword in normalized_forecast_type for keyword in ["preincrease", "turnaround", "profit"]
        ):
            forecast_bonus += 0.7
            reasons.append("业绩预告偏正面")
        elif any(keyword in forecast_type for keyword in ["预减", "首亏", "续亏", "略减"]) or any(
            keyword in normalized_forecast_type for keyword in ["predecrease", "loss"]
        ):
            forecast_bonus -= 0.7
            reasons.append("业绩预告偏谨慎")

        forecast_growth_anchor = forecast_p_change_max if forecast_p_change_max is not None else forecast_p_change_min
        if forecast_growth_anchor is not None:
            if forecast_growth_anchor >= 100:
                forecast_bonus += 0.8
                reasons.append("预告上修幅度大")
            elif forecast_growth_anchor >= 30:
                forecast_bonus += 0.4
            elif forecast_growth_anchor <= -50:
                forecast_bonus -= 0.8
                reasons.append("预告下修压力大")
            elif forecast_growth_anchor <= -20:
                forecast_bonus -= 0.4

        fina_bonus = round(max(-2.0, min(3.0, fina_bonus)), 2)
        express_bonus = round(max(-1.5, min(2.0, express_bonus)), 2)
        forecast_bonus = round(max(-1.0, min(LIGHT_FUNDAMENTAL_FORECAST_BONUS_CAP, forecast_bonus)), 2)
        total_bonus = round(
            max(
                LIGHT_FUNDAMENTAL_BONUS_MIN,
                min(LIGHT_FUNDAMENTAL_BONUS_MAX, fina_bonus + express_bonus + forecast_bonus),
            ),
            2,
        )
        return {
            "total_bonus": total_bonus,
            "breakdown": {
                "fina_indicator_bonus": fina_bonus,
                "express_bonus": express_bonus,
                "forecast_bonus": forecast_bonus,
                "reasons": reasons[:6],
            },
        }

    def _generate_recommendation(
        self,
        score: float,
        analysis: Dict[str, Any],
        distribution_risk: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成操作建议"""
        technical_signal = str(analysis.get("technical_signal") or "").strip()
        risk_payload = distribution_risk or {}
        risk_blocked = bool(risk_payload.get("candidate_risk_blocked", False))
        risk_score = float(risk_payload.get("distribution_risk_score") or 0.0)
        confidence = float(analysis.get("overall_confidence") or 0.0)

        if risk_blocked:
            return "等待确认：短线分歧偏大，暂不追高"
        if score >= 82 and confidence >= 0.7 and risk_score < 1.2:
            return "优先关注：多维度共振，可作为当日重点跟踪"
        if score >= 72 and risk_score < 2.0:
            if self._contains_any_keyword(technical_signal, ["突破", "走强", "多头", "放量", "启动"]):
                return "建议跟踪：趋势延续性较好，可等回踩或放量确认"
            return "建议跟踪：强度尚可，关注盘中承接与量价确认"
        if score >= 60:
            return "等待确认：具备一定弹性，先观察是否形成一致性"
        return "暂不参与：当前胜率与盈亏比暂不占优"

    def _get_top_stocks(
        self,
        screening_results: Dict[str, ScreenResult],
        limit: int = 10
    ) -> List[str]:
        """获取TOP股票列表（两层筛选）"""
        stock_scores = {}

        # 第一层：统计每只股票出现次数和得分
        for strategy_id, result in screening_results.items():
            if not result:
                continue

            for i, stock in enumerate(result.stocks):
                technical_score = float(stock.technical_score or 0.0)
                recommendation_score = float(stock.recommendation_score or stock.score or 0.0)
                pct_change = float(stock.pct_change or 0.0) if stock.pct_change is not None else 0.0
                volume_ratio = float(stock.volume_ratio or 0.0)
                current_record = {
                    "stock": stock,
                    "technical_score": technical_score,
                    "recommendation_score": recommendation_score,
                    "pct_change": pct_change,
                    "volume_ratio": volume_ratio,
                    "rsi": stock.rsi,
                }
                if stock.ts_code not in stock_scores:
                    stock_scores[stock.ts_code] = {
                        "count": 0,
                        "total_score": 0,
                        "best_rank": float('inf'),
                        "technical_score": technical_score,
                        "recommendation_score": recommendation_score,
                        "pct_change": pct_change,
                        "volume_ratio": volume_ratio,
                        "rsi": stock.rsi,
                        "technical_score_min": technical_score,
                        "technical_score_max": technical_score,
                        "divergence_score": 0.0,
                        "strategy_consistency_label": "单策略命中",
                        "representative_stock": stock,
                    }
                else:
                    aggregate = stock_scores[stock.ts_code]
                    aggregate["technical_score_min"] = min(aggregate["technical_score_min"], technical_score)
                    aggregate["technical_score_max"] = max(aggregate["technical_score_max"], technical_score)
                    representative_score = float(aggregate["recommendation_score"] or 0.0)
                    representative_technical = float(aggregate["technical_score"] or 0.0)
                    if (
                        recommendation_score > representative_score
                        or (
                            recommendation_score == representative_score
                            and technical_score > representative_technical
                        )
                    ):
                        aggregate["representative_stock"] = stock
                        aggregate["technical_score"] = technical_score
                        aggregate["recommendation_score"] = recommendation_score
                        aggregate["pct_change"] = pct_change
                        aggregate["volume_ratio"] = volume_ratio
                        aggregate["rsi"] = stock.rsi

                stock_scores[stock.ts_code]["count"] += 1
                stock_scores[stock.ts_code]["total_score"] += (
                    100 - i * 2  # 排名越靠前分数越高
                )
                stock_scores[stock.ts_code]["best_rank"] = min(
                    stock_scores[stock.ts_code]["best_rank"],
                    i
                )
                divergence_score = max(
                    0.0,
                    float(stock_scores[stock.ts_code]["technical_score_max"] or 0.0)
                    - float(stock_scores[stock.ts_code]["technical_score_min"] or 0.0),
                )
                stock_scores[stock.ts_code]["divergence_score"] = divergence_score
                if stock_scores[stock.ts_code]["count"] <= 1:
                    stock_scores[stock.ts_code]["strategy_consistency_label"] = "单策略命中"
                elif divergence_score >= 20:
                    stock_scores[stock.ts_code]["strategy_consistency_label"] = "存在分歧"
                else:
                    stock_scores[stock.ts_code]["strategy_consistency_label"] = "多策略一致"

        # 第二层：仅做轻量风险过滤，优先保障第一阶段候选池召回
        # 优先级：出现次数 > 技术评分 > 成交量 > 涨幅
        filtered_stocks = []
        reject_reasons = {
            "technical_score": 0,
            "volume_ratio": 0,
            "rsi": 0,
        }
        reject_samples = {
            "technical_score": [],
            "volume_ratio": [],
            "rsi": [],
        }
        for code, scores in stock_scores.items():
            # 基础筛选：至少出现在一个策略中
            if scores["count"] < 1:
                continue

            # 技术评分只拦截明显过弱的标的
            if scores["technical_score"] < 30:
                reject_reasons["technical_score"] += 1
                if len(reject_samples["technical_score"]) < 5:
                    reject_samples["technical_score"].append(
                        {
                            "ts_code": code,
                            "technical_score": round(float(scores["technical_score"] or 0.0), 2),
                            "volume_ratio": round(float(scores["volume_ratio"] or 0.0), 2),
                            "rsi": round(float(scores["rsi"] or 0.0), 2) if scores["rsi"] is not None else None,
                            "count": scores["count"],
                        }
                    )
                continue

            # 成交量仅剔除显著缩量的标的
            if scores["volume_ratio"] < 0.7:
                reject_reasons["volume_ratio"] += 1
                if len(reject_samples["volume_ratio"]) < 5:
                    reject_samples["volume_ratio"].append(
                        {
                            "ts_code": code,
                            "technical_score": round(float(scores["technical_score"] or 0.0), 2),
                            "volume_ratio": round(float(scores["volume_ratio"] or 0.0), 2),
                            "rsi": round(float(scores["rsi"] or 0.0), 2) if scores["rsi"] is not None else None,
                            "count": scores["count"],
                        }
                    )
                continue

            # RSI 只过滤极端过热/极端过冷
            if scores["rsi"] is not None:
                if scores["rsi"] > 90 or scores["rsi"] < 10:
                    reject_reasons["rsi"] += 1
                    if len(reject_samples["rsi"]) < 5:
                        reject_samples["rsi"].append(
                            {
                                "ts_code": code,
                                "technical_score": round(float(scores["technical_score"] or 0.0), 2),
                                "volume_ratio": round(float(scores["volume_ratio"] or 0.0), 2),
                                "rsi": round(float(scores["rsi"] or 0.0), 2),
                                "count": scores["count"],
                            }
                        )
                    continue

            filtered_stocks.append((code, scores))

        logger.info(
            "Baseline helper filters: technical_score=%s, volume_ratio=%s, rsi=%s, samples=%s",
            reject_reasons["technical_score"],
            reject_reasons["volume_ratio"],
            reject_reasons["rsi"],
            reject_samples,
        )
        divergence_samples = [
            {
                "ts_code": code,
                "count": scores["count"],
                "technical_score_min": round(float(scores["technical_score_min"] or 0.0), 2),
                "technical_score_max": round(float(scores["technical_score_max"] or 0.0), 2),
                "representative_technical_score": round(float(scores["technical_score"] or 0.0), 2),
                "representative_recommendation_score": round(float(scores["recommendation_score"] or 0.0), 2),
            }
            for code, scores in stock_scores.items()
            if scores["count"] > 1 and abs(float(scores["technical_score_max"] or 0.0) - float(scores["technical_score_min"] or 0.0)) >= 20
        ]
        if divergence_samples:
            logger.info(
                "Baseline helper divergences: %s",
                divergence_samples[:10],
            )

        # 排序：多策略共振 > 技术评分 > 成交量 > 涨幅
        sorted_stocks = sorted(
            filtered_stocks,
            key=lambda x: (
                -x[1]["count"],           # 出现次数越多越好
                -x[1]["technical_score"], # 技术评分越高越好
                -x[1]["volume_ratio"],    # 成交量越大越好
                -x[1]["pct_change"],      # 涨幅越大越好
            )
        )

        logger.info(
            f"Baseline helper screening: {len(stock_scores)} candidates → "
            f"{len(filtered_stocks)} qualified → {min(len(sorted_stocks), limit)} selected"
        )

        return [code for code, _ in sorted_stocks[:limit]]

    def _save_recommendation_run(
        self,
        *,
        trade_date: str,
        screening_results: Dict[str, ScreenResult],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
        report_id: Optional[str],
        rerank_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.settings.use_database:
            return

        trade_day = datetime.strptime(trade_date, "%Y%m%d").date()
        pool_states = self.store.list_recommendation_pool(trade_date=trade_day)
        items = self._build_recommendation_items(
            trade_date=trade_date,
            screening_results=screening_results,
            ai_analyses=ai_analyses,
            final_recommendations=final_recommendations,
            pool_states=pool_states,
            rerank_metadata=rerank_metadata or {},
        )
        self.store.save_recommendation_run(
            run_id=f"rec-{trade_date}",
            trade_date=trade_day,
            candidate_count=len(pool_states),
            final_count=len(items),
            report_id=report_id,
            items=items,
            generated_at=datetime.now(),
        )

    def _build_recommendation_items(
        self,
        *,
        trade_date: str,
        screening_results: Dict[str, ScreenResult],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
        pool_states: List[Dict[str, Any]],
        rerank_metadata: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        stock_map = {}
        trade_day = datetime.strptime(trade_date, "%Y%m%d").date()
        for result in screening_results.values():
            if not result:
                continue
            for stock in result.stocks:
                stock_map.setdefault(stock.ts_code, stock)

        state_map = {item.get("ts_code"): item for item in pool_states}
        items = []
        sorted_states = sorted(pool_states, key=self._frontlist_sort_key)
        for state in sorted_states:
            code = state.get("ts_code")
            stock = stock_map.get(code)
            analysis = ai_analyses.get(code, {})
            info = final_recommendations.get(code, {})
            rerank_info = rerank_metadata.get(code, {})
            source_tag = state.get("source_tag") or WINDOW_RECOMMENDATION_TAG
            item_payload = {
                "ts_code": code,
                "name": state.get("name") or getattr(stock, "name", "") or analysis.get("name") or "",
                "recommend_rank": state.get("recommend_rank"),
                "recommend_score": state.get("recommendation_score", info.get("score")),
                "recommendation_score": state.get("recommendation_score", info.get("score")),
                "priority_score": state.get("priority_score"),
                "frontlist_rank": state.get("frontlist_rank"),
                "hit_streak_days": state.get("hit_streak_days", 0),
                "miss_streak_days": state.get("miss_streak_days", 0),
                "tracking_status": state.get("tracking_status"),
                "in_frontlist": state.get("in_frontlist", False),
                "llm_focus_level": state.get("llm_focus_level"),
                "source_tag": source_tag,
                "is_repeat_pick": bool(state.get("is_repeat_pick", False)),
                "ai_confidence": state.get("ai_confidence", info.get("ai_confidence", analysis.get("overall_confidence"))),
                "strategy_count": state.get("strategy_count", info.get("strategy_count", 0)),
                "news_mentioned": state.get("news_mentioned", info.get("news_mentioned", False)),
                "industry": state.get("industry", info.get("industry")),
                "industry_heat_score": state.get("industry_heat_score", info.get("industry_heat_score")),
                "industry_flow_bias": state.get("industry_flow_bias", info.get("industry_flow_bias")),
                "score_change": state.get("score_change"),
                "previous_recommendation_score": state.get("previous_recommendation_score"),
                "technical_signal": state.get("technical_signal") or info.get("technical_signal") or analysis.get("technical_signal") or "",
                "recommendation_text": state.get("recommendation_text") or info.get("recommendation") or analysis.get("recommendation") or analysis.get("summary") or "",
                "status": "tracking" if source_tag == "昨日延续" else (state.get("tracking_status") or "new"),
                "tracking_days": max(state.get("hit_streak_days", 0), 0),
                "trade_date": trade_day,
                "entry_price": state.get("entry_price", getattr(stock, "close", None)),
                "score_mode": state.get("score_model") or rerank_info.get("score_mode"),
                "rerank_pool_rank": rerank_info.get("rerank_pool_rank"),
                "rerank_blend_score": rerank_info.get("blend_score"),
                "rerank_model_score": rerank_info.get("model_score"),
                "rerank_rule_score": rerank_info.get("rule_score"),
                "rerank_rule_weight": rerank_info.get("rule_weight"),
                "rerank_model_target": rerank_info.get("model_target"),
                "selection_stage": state.get("selection_stage"),
                "selection_reason": state.get("selection_reason"),
                "selection_reason_components": state.get("selection_reason_components") or {},
                "structured_rank_score": state.get("structured_rank_score"),
                "structured_rank_position": state.get("structured_rank_position"),
            }
            item_payload.update(self._build_unified_score_fields(item_payload, state, analysis, info))
            items.append(item_payload)
        return items

    def _build_placeholder_ai_analyses(
        self,
        codes: List[str],
        screening_results: Dict[str, ScreenResult],
        structured_recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        stock_map = self._build_screened_stock_map(screening_results)
        stock_name_map = self._build_stock_name_map(screening_results)
        analyses: Dict[str, Dict[str, Any]] = {}
        for code in codes:
            recommendation = structured_recommendations.get(code, {})
            stock = stock_map.get(code)
            analyses[code] = {
                "name": stock_name_map.get(code) or getattr(stock, "name", code),
                "summary": recommendation.get("summary") or recommendation.get("recommendation_text") or "",
                "recommendation": recommendation.get("recommendation") or recommendation.get("recommendation_text") or "",
                "technical_signal": recommendation.get("technical_signal") or getattr(stock, "trend_status", "") or "",
                "overall_score": recommendation.get("overall_score") or recommendation.get("adjusted_final_score") or recommendation.get("score") or 0.0,
                "overall_confidence": recommendation.get("overall_confidence") or recommendation.get("ai_confidence") or 0.65,
                "technical_score": recommendation.get("technical_score") or getattr(stock, "technical_score", None),
                "fundamental_score": recommendation.get("fundamental_score"),
                "sentiment_score": recommendation.get("sentiment_score"),
                "news_score": recommendation.get("news_score"),
                "base_score": recommendation.get("base_score"),
                "sentiment_adjustment": recommendation.get("sentiment_adjustment"),
                "news_adjustment": recommendation.get("news_adjustment"),
                "score_model": recommendation.get("score_model") or "stage_structured_fallback",
            }
        return analyses

    def _merge_recommendation_payloads(
        self,
        *,
        base_recommendations: Dict[str, Dict[str, Any]],
        llm_recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        ranking_protected_keys = {
            "score",
            "weighted_score",
            "recommendation_score",
            "recommend_score",
            "priority_score",
            "overall_score",
            "final_score",
            "adjusted_final_score",
            "stage3_final_score",
            "final_selection_score",
            "stage3_moneyflow_score",
            "structured_rank_score",
            "structured_rank_position",
            "model_score_norm",
            "overall_score_norm",
            "fusion_70_30",
            "top3_ranking_strategy",
            "selection_stage",
            "selection_reason",
            "selection_reason_components",
            "recommend_rank",
            "frontlist_rank",
            "rerank_pool_rank",
            "top3_extreme_risk_blocked",
            "top3_extreme_risk_reason",
            "top3_st_excluded",
            "top3_st_excluded_reason",
        }
        merged = {code: dict(payload) for code, payload in base_recommendations.items()}
        for code, payload in llm_recommendations.items():
            if code not in merged:
                continue
            combined = dict(merged[code])
            for key, value in payload.items():
                if key in ranking_protected_keys:
                    if value not in (None, "", [], {}):
                        combined[f"llm_{key}"] = value
                    continue
                combined[key] = value
            combined["selection_stage"] = merged[code].get("selection_stage")
            combined["selection_reason"] = merged[code].get("selection_reason")
            combined["selection_reason_components"] = merged[code].get("selection_reason_components") or {}
            merged[code] = combined
        return dict(sorted(merged.items(), key=lambda item: item[1].get("score", 0.0), reverse=True))

    def _build_structured_stage_selection_metadata(
        self,
        *,
        recommendations: Dict[str, Dict[str, Any]],
        rerank_metadata: Dict[str, Dict[str, Any]],
        stage2_top20_codes: List[str],
        stage3_top3_codes: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        structured_rank_map = self._build_stage_rank_maps(list(recommendations.keys()))
        stage2_rank_map = self._build_stage_rank_maps(stage2_top20_codes)
        stage3_rank_map = self._build_stage_rank_maps(stage3_top3_codes)
        for code, payload in recommendations.items():
            rerank_rank = rerank_metadata.get(code, {}).get("rerank_pool_rank") or payload.get("rerank_pool_rank")
            final_selection_score = self._first_defined_value(
                payload.get("final_selection_score"),
                payload.get("stage3_final_score"),
                payload.get("structured_rank_score"),
                payload.get("score"),
            )
            selection_reason_components = {
                **dict(payload.get("selection_reason_components") or {}),
                "structured_score": round(float(payload.get("score") or 0.0), 4),
                "final_selection_score": round(float(final_selection_score or 0.0), 4),
                "continuation_bias": round(float(payload.get("continuation_bias_score") or 0.0), 4),
                "risk_penalty": round(float(payload.get("ranking_risk_penalty") or 0.0), 4),
                "rerank_rank": rerank_rank,
                "top3_extreme_risk_blocked": bool(payload.get("top3_extreme_risk_blocked", False)),
                "top3_extreme_risk_reason": payload.get("top3_extreme_risk_reason"),
                "top3_st_excluded": bool(payload.get("top3_st_excluded", False)),
                "top3_st_excluded_reason": payload.get("top3_st_excluded_reason"),
                "stage2_rank": stage2_rank_map.get(code),
                "stage3_rank": stage3_rank_map.get(code),
            }
            payload["structured_rank_position"] = stage3_rank_map.get(code) or stage2_rank_map.get(code) or structured_rank_map.get(code)
            payload["structured_rank_score"] = final_selection_score
            payload["final_selection_score"] = final_selection_score
            if payload.get("top3_extreme_risk_blocked"):
                payload["selection_stage"] = "model_top100_extreme_risk_veto"
            elif payload.get("top3_st_excluded"):
                payload["selection_stage"] = "model_top100_st_refill_excluded"
            else:
                payload["selection_stage"] = "stage3_final_top3" if code in stage3_top3_codes else ("stage2_top20_pre_moneyflow" if code in stage2_top20_codes else "stage1_candidate_pool")
            payload["selection_reason_components"] = selection_reason_components
            payload["selection_reason"] = (
                f"model_rank={selection_reason_components['rerank_rank']}; "
                f"structured_score={selection_reason_components['structured_score']:.2f}; "
                f"final_selection_score={selection_reason_components['final_selection_score']:.2f}; "
                f"continuation_bias={selection_reason_components['continuation_bias']:.2f}; "
                f"risk_penalty={selection_reason_components['risk_penalty']:.2f}; "
                f"top3_extreme_risk={selection_reason_components['top3_extreme_risk_blocked']}; "
                f"top3_st_excluded={selection_reason_components['top3_st_excluded']}; "
                f"rerank_rank={selection_reason_components['rerank_rank']}; "
                f"stage2_rank={selection_reason_components['stage2_rank']}; "
                f"stage3_rank={selection_reason_components['stage3_rank']}"
            )
        return recommendations

    def _build_analysis_target_codes(
        self,
        *,
        trade_date: date,
        candidate_codes: List[str],
        screening_results: Dict[str, ScreenResult],
    ) -> List[str]:
        del trade_date, screening_results
        return list(dict.fromkeys(candidate_codes[:TODAY_TOP_LIMIT]))

    def _get_previous_top3_codes(self, trade_date: date) -> List[str]:
        previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(trade_date)
        if not previous_trade_date:
            return []
        previous_states = self.store.load_recommendation_pool_state(trade_date=previous_trade_date)
        previous_top3 = [
            item for item in previous_states
            if item.get("source_tag") == "今日Top3" and item.get("ts_code")
        ]
        previous_top3 = sorted(
            previous_top3,
            key=lambda item: (
                -(item.get("priority_score") or 0.0),
                -(item.get("recommendation_score") or 0.0),
                item.get("ts_code") or "",
            ),
        )[:TODAY_TOP_LIMIT]
        return [item.get("ts_code") for item in previous_top3 if item.get("ts_code")]

    def _filter_out_tracked_and_holding_codes(self, candidate_codes: List[str]) -> List[str]:
        settings = self.settings
        blocked_codes = {code.strip().upper() for code in settings.stock_pool}
        position_store = create_position_store(settings)
        filtered_codes: List[str] = []
        for code in candidate_codes:
            normalized = code.strip().upper()
            if normalized in blocked_codes:
                continue
            if position_store.get_status(normalized) == "holding":
                continue
            filtered_codes.append(normalized)
        return filtered_codes

    @staticmethod
    def _calculate_streaks(previous_state: Optional[Dict[str, Any]], is_displayed: bool) -> Dict[str, int]:
        if is_displayed:
            return {
                "hit_streak_days": int((previous_state or {}).get("hit_streak_days", 0)) + 1,
                "miss_streak_days": 0,
            }
        return {
            "hit_streak_days": 0,
            "miss_streak_days": int((previous_state or {}).get("miss_streak_days", 0)) + 1,
        }

    @staticmethod
    def _classify_tracking_status(is_displayed: bool, source_tag: str) -> str:
        if not is_displayed:
            return "shadow"
        if source_tag == "今日Top3":
            return "active"
        if source_tag == "昨日延续":
            return "tracking"
        return "candidate"

    @staticmethod
    def _classify_llm_focus_level(source_tag: str) -> str:
        if source_tag == "今日Top3":
            return "high"
        if source_tag == "昨日延续":
            return "medium"
        return "medium"

    @staticmethod
    def _apply_repeat_pick_confidence_bonus(confidence: Optional[float], is_repeat_pick: bool) -> Optional[float]:
        if confidence is None:
            return None
        if not is_repeat_pick:
            return round(float(confidence), 4)
        return round(min(float(confidence) + REPEAT_CONFIDENCE_BONUS, MAX_CONFIDENCE), 4)

    @staticmethod
    def _is_code_like_name(code: str, value: Any) -> bool:
        name = str(value or "").strip()
        if not name:
            return True
        normalized_code = str(code or "").strip().upper()
        normalized_name = name.upper()
        return normalized_name == normalized_code or normalized_name == normalized_code.split(".")[0]

    def _load_stock_name_cache(self) -> Dict[str, str]:
        if self._stock_name_cache is not None:
            return self._stock_name_cache
        db = DatabaseManager(self.settings.database_url)
        session = db.get_session()
        try:
            rows = session.query(MarketStockBasic.ts_code, MarketStockBasic.name).all()
            self._stock_name_cache = {
                str(ts_code).strip().upper(): str(name).strip()
                for ts_code, name in rows
                if str(ts_code or "").strip() and not self._is_code_like_name(str(ts_code), name)
            }
        except Exception:
            logger.exception("Failed to load stock basic names")
            self._stock_name_cache = {}
        finally:
            session.close()
        return self._stock_name_cache

    def _resolve_stock_name(
        self,
        code: str,
        stock: Any,
        recommendation: Dict[str, Any],
        current_state: Optional[Dict[str, Any]],
        historical_state: Optional[Dict[str, Any]],
    ) -> str:
        normalized_code = str(code or "").strip().upper()
        candidates = [
            getattr(stock, "name", None),
            (current_state or {}).get("name"),
            recommendation.get("name"),
            recommendation.get("stock_name"),
            recommendation.get("display_name"),
            (historical_state or {}).get("name"),
            self._load_stock_name_cache().get(normalized_code),
        ]
        for candidate in candidates:
            name = str(candidate or "").strip()
            if name and not self._is_code_like_name(normalized_code, name):
                return name
        return normalized_code

    @staticmethod
    def _has_real_ai_overall_score(payload: Optional[Dict[str, Any]]) -> bool:
        if not payload:
            return False
        if payload.get("overall_score") is None and payload.get("base_score") is None:
            return False
        ai_markers = (
            "base_score",
            "overall_confidence",
            "technical_score",
            "fundamental_score",
            "sentiment_score",
            "news_score",
            "score_model",
        )
        return any(payload.get(marker) is not None for marker in ai_markers)

    @classmethod
    def _resolve_real_overall_score(
        cls,
        current_payload: Optional[Dict[str, Any]],
        *historical_payloads: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if cls._has_real_ai_overall_score(current_payload):
            value = current_payload.get("overall_score")
            if value is None:
                value = current_payload.get("base_score")
            if value is not None:
                return float(value)
        for payload in historical_payloads:
            if cls._has_real_ai_overall_score(payload):
                value = payload.get("overall_score")
                if value is None:
                    value = payload.get("base_score")
                if value is not None:
                    return float(value)
        return None

    @classmethod
    def _is_authoritative_today_top_state(cls, item: Optional[Dict[str, Any]]) -> bool:
        if not item or item.get("source_tag") != "今日Top3":
            return False
        if item.get("recommend_rank") is None:
            return False
        selection_stage = str(item.get("selection_stage") or "")
        if selection_stage:
            return selection_stage == "stage3_final_top3"
        return item.get("final_selection_score") is not None or item.get("structured_rank_position") is not None

    @staticmethod
    def _is_st_excluded_pool_state(item: Dict[str, Any]) -> bool:
        selection_stage = str((item or {}).get("selection_stage") or "").lower()
        components = (item or {}).get("selection_reason_components") or {}
        if not isinstance(components, dict):
            components = {}
        return bool(components.get("top3_st_excluded", False)) or "st_veto" in selection_stage or "st_refill_excluded" in selection_stage

    @staticmethod
    def _today_top_sort_key(item: Dict[str, Any]) -> Any:
        return (
            item.get("recommend_rank") is None,
            int(item.get("recommend_rank") or 9999),
            item.get("structured_rank_position") is None,
            int(item.get("structured_rank_position") or 9999),
            -float(item.get("final_selection_score") or item.get("stage3_final_score") or item.get("structured_rank_score") or 0.0),
            -float(item.get("recommendation_score") or 0.0),
            -float(item.get("overall_score") or item.get("priority_score") or 0.0),
            item.get("ts_code") or "",
        )

    @staticmethod
    def _frontlist_sort_key(item: Dict[str, Any]) -> Any:
        return (
            item.get("frontlist_rank") is None,
            int(item.get("frontlist_rank") or 9999),
            item.get("structured_rank_position") is None,
            int(item.get("structured_rank_position") or 9999),
            -float(item.get("final_selection_score") or item.get("stage3_final_score") or item.get("structured_rank_score") or 0.0),
            -float(item.get("recommendation_score") or item.get("final_display_recommendation_score") or 0.0),
            -float(item.get("overall_score") or item.get("priority_score") or 0.0),
            item.get("ts_code") or "",
        )

    def _is_recommendation_pool_st_excluded(
        self,
        *,
        code: str,
        payload: Dict[str, Any],
        stock_name_map: Dict[str, str],
    ) -> bool:
        selection_stage = str(payload.get("selection_stage") or "").lower()
        if bool(payload.get("top3_st_excluded", False)):
            return True
        if "st_veto" in selection_stage or "st_refill_excluded" in selection_stage:
            return True
        return self._is_st_stock_for_final_veto(
            code=code,
            payload=payload,
            stock_name_map=stock_name_map,
        )

    def _is_top3_fallback_eligible(
        self,
        *,
        code: str,
        payload: Dict[str, Any],
        stock_name_map: Dict[str, str],
    ) -> bool:
        selection_stage = str(payload.get("selection_stage") or "")
        if selection_stage != "stage2_top20_pre_moneyflow":
            return False
        if self._is_recommendation_pool_st_excluded(
            code=code,
            payload=payload,
            stock_name_map=stock_name_map,
        ):
            return False
        hard_veto_fields = (
            "candidate_risk_blocked",
            "top3_extreme_risk_blocked",
            "stage3_moneyflow_veto",
        )
        if any(bool(payload.get(field, False)) for field in hard_veto_fields):
            return False
        if bool(payload.get("stage3_close_auction_veto", False)) and not bool(payload.get("stage3_close_auction_veto_softened", False)):
            return False
        if self._get_top3_extreme_risk_reason(payload):
            return False
        return True

    @classmethod
    def _select_authoritative_today_top_states(
        cls,
        pool_states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked_states = sorted(
            [
                item for item in pool_states
                if item.get("ts_code") and not cls._is_st_excluded_pool_state(item)
            ],
            key=cls._today_top_sort_key,
        )
        authoritative_states = [
            item for item in ranked_states
            if cls._is_authoritative_today_top_state(item)
        ]
        if authoritative_states:
            return authoritative_states[:TODAY_TOP_LIMIT]
        fallback_states = [
            item for item in ranked_states
            if item.get("source_tag") == "今日Top3"
            and item.get("recommend_rank") is not None
            and (
                cls._has_real_ai_overall_score(item)
                or item.get("final_selection_score") is not None
                or item.get("structured_rank_position") is not None
            )
        ]
        return fallback_states[:TODAY_TOP_LIMIT]

    def _build_recommendation_pool_states(
        self,
        *,
        trade_date: date,
        screening_results: Dict[str, ScreenResult],
        final_recommendations: Dict[str, Dict[str, Any]],
        candidate_codes: List[str],
        rerank_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[TrackedRecommendationState]:
        previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(trade_date)
        rerank_metadata = rerank_metadata or {}
        previous_states = {
            item.get("ts_code"): item
            for item in self.store.load_recommendation_pool_state(trade_date=previous_trade_date)
        } if previous_trade_date else {}
        stock_map: Dict[str, Any] = {}
        strategy_counts: Dict[str, int] = {}
        for result in screening_results.values():
            if not result:
                continue
            for stock in result.stocks:
                stock_map.setdefault(stock.ts_code, stock)
                strategy_counts[stock.ts_code] = strategy_counts.get(stock.ts_code, 0) + 1
        stock_name_map = self._build_stock_name_map(screening_results)

        model_candidate_codes = [
            code for code in candidate_codes[:MODEL_RISK_REVIEW_POOL_LIMIT]
            if code in final_recommendations
        ]
        eligible_model_candidate_codes = [
            code for code in model_candidate_codes
            if not self._is_recommendation_pool_st_excluded(
                code=code,
                payload=final_recommendations.get(code, {}) or {},
                stock_name_map=stock_name_map,
            )
        ]
        rerank_rank_map = {
            code: index for index, code in enumerate(model_candidate_codes, start=1)
        }
        stage3_selected_codes = [
            code for code in eligible_model_candidate_codes
            if str((final_recommendations.get(code, {}) or {}).get("selection_stage") or "") == "stage3_final_top3"
        ]
        today_top_codes = sorted(
            stage3_selected_codes,
            key=lambda code: (
                int(float((final_recommendations.get(code, {}) or {}).get("structured_rank_position") or 9999)),
                -float((final_recommendations.get(code, {}) or {}).get("final_selection_score") or (final_recommendations.get(code, {}) or {}).get("stage3_final_score") or 0.0),
                int(float((final_recommendations.get(code, {}) or {}).get("rerank_pool_rank") or rerank_rank_map.get(code) or 9999)),
                code,
            ),
        )[:TODAY_TOP_LIMIT]
        if not today_top_codes:
            today_top_codes = [
                code for code in eligible_model_candidate_codes
                if self._is_top3_fallback_eligible(
                    code=code,
                    payload=final_recommendations.get(code, {}) or {},
                    stock_name_map=stock_name_map,
                )
            ][:TODAY_TOP_LIMIT]
        display_codes = list(dict.fromkeys(
            today_top_codes
            + [
                code for code in sorted(
                    eligible_model_candidate_codes,
                    key=lambda candidate_code: (
                        str((final_recommendations.get(candidate_code, {}) or {}).get("selection_stage") or "") != "stage2_top20_pre_moneyflow",
                        int(float((final_recommendations.get(candidate_code, {}) or {}).get("structured_rank_position") or 9999)),
                        -float((final_recommendations.get(candidate_code, {}) or {}).get("final_selection_score") or (final_recommendations.get(candidate_code, {}) or {}).get("stage3_final_score") or 0.0),
                        int(float((final_recommendations.get(candidate_code, {}) or {}).get("rerank_pool_rank") or rerank_rank_map.get(candidate_code) or 9999)),
                        candidate_code,
                    ),
                )
                if code not in today_top_codes
            ]
        ))[:TOP_RECOMMENDATION_LIMIT]
        previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(trade_date)
        previous_frontlist = self.store.list_recommendation_pool(
            trade_date=previous_trade_date,
            front_only=True,
            limit=TOP_RECOMMENDATION_LIMIT,
        ) if previous_trade_date else []
        previous_frontlist_map = {
            item.get("ts_code"): item for item in previous_frontlist if item.get("ts_code")
        }
        previous_score_map = {
            code: float(item.get("recommendation_score") or item.get("recommend_score") or item.get("score") or 0.0)
            for code, item in previous_frontlist_map.items()
        }
        previous_top3_codes = self._get_previous_top3_codes(trade_date)
        display_code_set = set(display_codes)
        display_rank_map = {code: index for index, code in enumerate(display_codes, start=1)}
        continuation_codes = [code for code in previous_top3_codes if code]
        rerank_top_codes = set(today_top_codes)
        merged_display_codes = list(dict.fromkeys(display_codes + eligible_model_candidate_codes + [
            code for code in continuation_codes
            if code not in rerank_rank_map and code in final_recommendations
            and not self._is_recommendation_pool_st_excluded(
                code=code,
                payload=final_recommendations.get(code, {}) or {},
                stock_name_map=stock_name_map,
            )
        ]))
        states_payload: List[Dict[str, Any]] = []
        for display_rank, code in enumerate(merged_display_codes, start=1):
            stock = stock_map.get(code)
            recommendation = final_recommendations.get(code, {})
            previous_state = previous_states.get(code)
            previous_front_state = previous_frontlist_map.get(code)
            is_today_top = code in today_top_codes
            is_continuation = code in continuation_codes and not is_today_top
            source_tag = "今日Top3" if is_today_top else ("昨日延续" if is_continuation else WINDOW_RECOMMENDATION_TAG)
            overall_score = self._resolve_real_overall_score(
                recommendation,
                previous_state,
                previous_front_state,
            )
            top3_risk_penalty = float(recommendation.get("distribution_risk_score") or 0.0) * TOP3_RISK_PENALTY_MULTIPLIER
            contradiction_penalty = self._build_short_term_contradiction_penalty(recommendation)
            base_recommendation_score = float(
                recommendation.get("weighted_score")
                or recommendation.get("recommendation_score")
                or recommendation.get("score")
                or getattr(stock, "recommendation_score", None)
                or (previous_state or {}).get("recommendation_score")
                or (previous_front_state or {}).get("recommendation_score")
                or 0.0
            )
            recommendation_score = round(
                max(
                    0.0,
                    base_recommendation_score
                    - (top3_risk_penalty if is_today_top else 0.0)
                    - (contradiction_penalty if is_today_top else 0.0),
                ),
                2,
            )
            raw_display_confidence = recommendation.get("display_confidence")
            raw_overall_confidence = recommendation.get("overall_confidence")
            raw_ai_confidence = recommendation.get("ai_confidence")
            analysis_confidence = raw_display_confidence
            if analysis_confidence is None:
                analysis_confidence = raw_overall_confidence
            if analysis_confidence is None:
                analysis_confidence = raw_ai_confidence
            if analysis_confidence is None and previous_state is not None:
                analysis_confidence = previous_state.get("display_confidence")
                if analysis_confidence is None:
                    analysis_confidence = previous_state.get("overall_confidence")
                if analysis_confidence is None:
                    analysis_confidence = previous_state.get("ai_confidence")
            if analysis_confidence is None and previous_front_state is not None:
                analysis_confidence = previous_front_state.get("display_confidence")
                if analysis_confidence is None:
                    analysis_confidence = previous_front_state.get("overall_confidence")
                if analysis_confidence is None:
                    analysis_confidence = previous_front_state.get("ai_confidence")
            if analysis_confidence is None and stock is not None:
                confidence_label = getattr(stock, "confidence", None)
                confidence_map = {"high": 0.8, "medium": 0.65, "low": 0.5}
                analysis_confidence = confidence_map.get(str(confidence_label).lower())
            selected_confidence = round(float(analysis_confidence), 4) if analysis_confidence is not None else None
            display_confidence = round(float(raw_display_confidence), 4) if raw_display_confidence is not None else selected_confidence
            overall_confidence = round(float(raw_overall_confidence), 4) if raw_overall_confidence is not None else selected_confidence
            ai_confidence_base = round(float(raw_ai_confidence), 4) if raw_ai_confidence is not None else selected_confidence
            ai_confidence = self._apply_repeat_pick_confidence_bonus(ai_confidence_base, is_continuation)
            streaks = self._calculate_streaks(previous_state, True)
            in_frontlist = code in display_code_set
            entered_frontlist = in_frontlist and not bool((previous_state or {}).get("in_frontlist"))
            previous_score = previous_score_map.get(code)
            divergence_score = float(recommendation.get("divergence_score") or 0.0)
            strategy_consistency_label = str(
                recommendation.get("strategy_consistency_label")
                or ("单策略命中" if int(strategy_counts.get(code, 0) or 0) <= 1 else ("存在分歧" if divergence_score >= 20 else "多策略一致"))
            )
            if "distribution_risk_flags" in recommendation:
                current_distribution_risk_flags = list(recommendation.get("distribution_risk_flags") or [])
            else:
                current_distribution_risk_flags = list((previous_state or {}).get("distribution_risk_flags") or [])
            current_distribution_risk_score = (
                recommendation.get("distribution_risk_score")
                if "distribution_risk_score" in recommendation
                else (previous_state or {}).get("distribution_risk_score")
            )
            final_selection_score = self._first_defined_value(
                recommendation.get("final_selection_score"),
                recommendation.get("stage3_final_score"),
                recommendation.get("structured_rank_score"),
                recommendation.get("score"),
            )
            current_item = {
                "ts_code": code,
                "name": self._resolve_stock_name(code, stock, recommendation, previous_state, previous_front_state),
                "recommendation_score": recommendation_score,
                "score_change": round(recommendation_score - previous_score, 1) if previous_score is not None else None,
                "priority_score": overall_score,
                "overall_score": overall_score,
                "hit_streak_days": streaks["hit_streak_days"],
                "miss_streak_days": streaks["miss_streak_days"],
                "in_frontlist": in_frontlist,
                "llm_focus_level": self._classify_llm_focus_level(source_tag),
                "tracking_status": self._classify_tracking_status(True, source_tag),
                "source_tag": source_tag,
                "is_repeat_pick": bool(code in previous_top3_codes),
                "setup_type": getattr(stock, "setup_type", None) or (previous_state or {}).get("setup_type"),
                "risk_level": getattr(stock, "risk_level", None) or (previous_state or {}).get("risk_level") or "medium",
                "recommendation": getattr(stock, "recommendation", None) or (previous_state or {}).get("recommendation") or "monitor",
                "position_status": (previous_state or {}).get("position_status"),
                "last_frontlist_date": trade_date,
                "times_entered_frontlist": int((previous_state or {}).get("times_entered_frontlist", 0)) + (1 if entered_frontlist else 0),
                "technical_score": recommendation.get("technical_score") if recommendation.get("technical_score") is not None else getattr(stock, "technical_score", None) if getattr(stock, "technical_score", None) is not None else (previous_state or {}).get("technical_score"),
                "fundamental_score": recommendation.get("fundamental_score") if recommendation.get("fundamental_score") is not None else (previous_state or {}).get("fundamental_score"),
                "sentiment_score": recommendation.get("sentiment_score") if recommendation.get("sentiment_score") is not None else (previous_state or {}).get("sentiment_score"),
                "news_score": recommendation.get("news_score") if recommendation.get("news_score") is not None else (previous_state or {}).get("news_score"),
                "base_score": recommendation.get("base_score") if recommendation.get("base_score") is not None else (previous_state or {}).get("base_score"),
                "sentiment_adjustment": recommendation.get("sentiment_adjustment") if recommendation.get("sentiment_adjustment") is not None else (previous_state or {}).get("sentiment_adjustment"),
                "news_adjustment": recommendation.get("news_adjustment") if recommendation.get("news_adjustment") is not None else (previous_state or {}).get("news_adjustment"),
                "score_model": recommendation.get("score_model") or rerank_metadata.get(code, {}).get("score_mode") or (previous_state or {}).get("score_model"),
                "summary": recommendation.get("review_summary") or recommendation.get("summary") or recommendation.get("ai_summary") or (previous_state or {}).get("summary"),
                "overview_reason": self._build_overview_reason({
                    "recommendation_score": recommendation_score,
                    "technical_signal": recommendation.get("technical_signal") or (previous_state or {}).get("technical_signal") or (getattr(stock, "trend_status", None) if stock else None),
                    "recommendation_text": recommendation.get("review_summary") or recommendation.get("recommendation") or recommendation.get("ai_summary") or (previous_state or {}).get("recommendation_text") or "",
                    "distribution_risk_flags": current_distribution_risk_flags,
                }),
                "close": getattr(stock, "close", None) or (previous_state or {}).get("close"),
                "pct_change": getattr(stock, "pct_change", None) if stock else (previous_state or {}).get("pct_change"),
                "volume_ratio": getattr(stock, "volume_ratio", None) if stock else (previous_state or {}).get("volume_ratio"),
                "turnover_rate": getattr(stock, "turnover_rate", None) if stock else (previous_state or {}).get("turnover_rate"),
                "ma20": recommendation.get("ma20") if recommendation.get("ma20") is not None else getattr(stock, "ma20", None) if stock else (previous_state or {}).get("ma20"),
                "strategy_count": strategy_counts.get(code, recommendation.get("strategy_count", (previous_state or {}).get("strategy_count", 0))),
                "divergence_score": divergence_score,
                "strategy_consistency_label": strategy_consistency_label,
                "news_mentioned": bool(recommendation.get("news_mentioned", False)),
                "industry": recommendation.get("industry") or getattr(stock, "industry", None) or (previous_state or {}).get("industry"),
                "industry_heat_score": recommendation.get("industry_heat_score", (previous_state or {}).get("industry_heat_score")),
                "industry_flow_bias": recommendation.get("industry_flow_bias") or (previous_state or {}).get("industry_flow_bias") or "中性",
                "distribution_risk_score": current_distribution_risk_score,
                "distribution_risk_flags": current_distribution_risk_flags,
                "moneyflow_3d_value": recommendation.get("moneyflow_3d_value", (previous_state or {}).get("moneyflow_3d_value")),
                "recent_large_order_net_inflow": recommendation.get("recent_large_order_net_inflow", (previous_state or {}).get("recent_large_order_net_inflow")),
                "recent_super_large_order_net_inflow": recommendation.get("recent_super_large_order_net_inflow", (previous_state or {}).get("recent_super_large_order_net_inflow")),
                "turnover_spike_ratio": recommendation.get("turnover_spike_ratio", (previous_state or {}).get("turnover_spike_ratio")),
                "recent_runup_5d": recommendation.get("recent_runup_5d", (previous_state or {}).get("recent_runup_5d")),
                "continuation_bias_score": recommendation.get("continuation_bias_score", (previous_state or {}).get("continuation_bias_score")),
                "continuation_positive_flags": list(recommendation.get("continuation_positive_flags") or (previous_state or {}).get("continuation_positive_flags") or []),
                "continuation_negative_flags": list(recommendation.get("continuation_negative_flags") or (previous_state or {}).get("continuation_negative_flags") or []),
                "top3_risk_penalty": round(top3_risk_penalty, 2) if is_today_top else None,
                "short_term_contradiction_penalty": contradiction_penalty if is_today_top else None,
                "final_display_recommendation_score": recommendation_score,
                "late_stage_momentum_flag": bool(recommendation.get("late_stage_momentum_flag", (previous_state or {}).get("late_stage_momentum_flag", False))),
                "candidate_risk_blocked": bool(recommendation.get("candidate_risk_blocked", False)),
                "top3_extreme_risk_blocked": bool(recommendation.get("top3_extreme_risk_blocked", False)),
                "top3_extreme_risk_reason": recommendation.get("top3_extreme_risk_reason"),
                "ai_confidence": ai_confidence,
                "display_confidence": display_confidence,
                "overall_confidence": overall_confidence,
                "confidence": selected_confidence,
                "technical_signal": recommendation.get("technical_signal") or (previous_state or {}).get("technical_signal") or getattr(stock, "trend_status", None),
                "recommendation_text": recommendation.get("review_summary") or recommendation.get("recommendation") or recommendation.get("ai_summary") or (previous_state or {}).get("recommendation_text") or "",
                "entry_price": getattr(stock, "close", None) or (previous_state or {}).get("entry_price"),
                "recommend_rank": (today_top_codes.index(code) + 1) if is_today_top else None,
                "frontlist_rank": display_rank_map.get(code),
                "rerank_pool_rank": rerank_metadata.get(code, {}).get("rerank_pool_rank") or rerank_rank_map.get(code),
                "rerank_model_score": rerank_metadata.get(code, {}).get("model_score"),
                "rerank_rule_score": rerank_metadata.get(code, {}).get("rule_score"),
                "rerank_blend_score": rerank_metadata.get(code, {}).get("blend_score"),
                "rerank_rule_weight": rerank_metadata.get(code, {}).get("rule_weight"),
                "rerank_model_target": rerank_metadata.get(code, {}).get("model_target"),
                "rerank_selected_for_llm": code in rerank_top_codes,
                "selection_stage": recommendation.get("selection_stage") or ("stage3_final_top3" if is_today_top else "stage1_candidate_pool"),
                "selection_reason": recommendation.get("selection_reason"),
                "selection_reason_components": recommendation.get("selection_reason_components") or {},
                "structured_rank_score": recommendation.get("structured_rank_score"),
                "structured_rank_position": recommendation.get("structured_rank_position"),
                "final_selection_score": final_selection_score,
                "fundamental_bonus": recommendation.get("fundamental_bonus"),
                "fundamental_bonus_breakdown": recommendation.get("fundamental_bonus_breakdown") or {},
                "previous_recommendation_score": previous_score,
                "previous_overall_score": self._resolve_real_overall_score(
                    previous_state,
                    previous_front_state,
                ),
                "previous_confidence": (previous_state or {}).get("display_confidence") if (previous_state or {}).get("display_confidence") is not None else (previous_state or {}).get("overall_confidence") if (previous_state or {}).get("overall_confidence") is not None else (previous_state or {}).get("ai_confidence") if (previous_state or {}).get("ai_confidence") is not None else (previous_front_state or {}).get("display_confidence") if (previous_front_state or {}).get("display_confidence") is not None else (previous_front_state or {}).get("overall_confidence") if (previous_front_state or {}).get("overall_confidence") is not None else (previous_front_state or {}).get("ai_confidence"),
                "today_present": True,
                "absence_reason": None,
                "action_plan": self._build_action_plan(recommendation, stock, previous_state),
                "review_status": "延续" if is_today_top and code in previous_top3_codes else ("新入选" if is_today_top else "观察"),
                "yesterday_conclusion": self._resolve_yesterday_conclusion(previous_state),
                "today_verdict": "延续走强，继续列入今日Top3" if is_today_top and code in previous_top3_codes else ("新入选，进入今日重点观察" if is_today_top else "仅保留在今日候选，未进入今日Top3"),
                "miss_reason_candidates": [],
                "missing_factor_candidates": [],
            }
            if code in {"603182.SH", "688618.SH", "300692.SZ"}:
                logger.info(
                    "Step 5 pool state debug for %s: %s",
                    code,
                    {
                        "source_tag": current_item.get("source_tag"),
                        "recommend_rank": current_item.get("recommend_rank"),
                        "recommendation_score": current_item.get("recommendation_score"),
                        "overall_score": current_item.get("overall_score"),
                        "priority_score": current_item.get("priority_score"),
                        "base_score": current_item.get("base_score"),
                        "summary": current_item.get("summary"),
                        "distribution_risk_score": current_item.get("distribution_risk_score"),
                        "moneyflow_3d_value": current_item.get("moneyflow_3d_value"),
                        "recent_runup_5d": current_item.get("recent_runup_5d"),
                        "turnover_spike_ratio": current_item.get("turnover_spike_ratio"),
                    },
                )
            states_payload.append(current_item)

        return [TrackedRecommendationState(trade_date=trade_date, **item) for item in states_payload]

    @staticmethod
    def _contains_any_keyword(text: Any, keywords: List[str]) -> bool:
        normalized = str(text or "").strip()
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _shorten_overview_text(text: Any, limit: int = 160) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip("，；、。,. ") + "…"

    @classmethod
    def _build_overview_reason(cls, item: Dict[str, Any]) -> str:
        recommendation = str(item.get("recommendation_text") or item.get("recommendation") or item.get("final_decision") or "").strip()
        technical_signal = str(item.get("technical_signal") or "").strip()
        market_context = str(item.get("market_context_view") or "").strip()
        highlights = [str(text).strip() for text in (item.get("core_highlights") or []) if str(text or "").strip()]
        risk_warnings = [str(text).strip() for text in (item.get("risk_warnings") or item.get("distribution_risk_flags") or []) if str(text or "").strip()]

        score = cls._safe_float(item.get("recommendation_score", item.get("recommend_score", item.get("score"))))
        if score is None:
            score = 0.0
        if score >= 85:
            positioning = "优先跟踪"
        elif score >= 75:
            positioning = "观察为主"
        else:
            positioning = "谨慎观察"

        support = ""
        if technical_signal:
            support = technical_signal
        elif market_context:
            support = market_context
        elif highlights:
            support = highlights[0]
        elif recommendation:
            support = recommendation

        action = "等放量确认"
        if recommendation:
            if "不参与" in recommendation or "观望" in recommendation:
                action = "先不参与"
            elif "谨慎" in recommendation or "观察" in recommendation:
                action = "先看承接"
            elif "跟踪" in recommendation or "关注" in recommendation:
                action = "等确认再跟"

        weakness = risk_warnings[0] if risk_warnings else "强度仍待验证"

        parts = [positioning]
        if support:
            parts.append(cls._shorten_overview_text(support, limit=48))
        parts.append(cls._shorten_overview_text(weakness, limit=48))
        parts.append(action)
        return "；".join(part for part in parts if part)

    @classmethod
    def _resolve_yesterday_conclusion(cls, item: Optional[Dict[str, Any]]) -> str:
        source = item or {}
        return (
            str(source.get("overview_reason") or "").strip()
            or cls._build_overview_reason(source)
            or str(source.get("summary") or "").strip()
            or str(source.get("recommendation_text") or "").strip()
            or "昨日结论缺失"
        )

    @classmethod
    def _build_short_term_contradiction_penalty(cls, recommendation: Dict[str, Any]) -> float:
        if bool(recommendation.get("candidate_risk_blocked", False)):
            return 0.0
        recommendation_text = str(
            recommendation.get("recommendation_text")
            or recommendation.get("recommendation")
            or recommendation.get("ai_summary")
            or ""
        ).strip()
        action_bias = str(
            recommendation.get("action_bias")
            or ((recommendation.get("action_plan") or {}).get("action_bias"))
            or ""
        ).strip()
        technical_signal = str(recommendation.get("technical_signal") or "").strip()
        caution_keywords = ["观察", "观望", "谨慎", "暂不建议", "不建议", "不参与", "回避", "等待"]
        positive_keywords = ["多头", "走强", "突破", "强势", "共振", "放量", "启动"]
        caution_hit = cls._contains_any_keyword(recommendation_text, caution_keywords)
        action_caution_hit = cls._contains_any_keyword(action_bias, ["观察", "观望", "不参与", "回避"])
        technical_positive_hit = cls._contains_any_keyword(technical_signal, positive_keywords)

        penalty = 0.0
        if caution_hit and (technical_positive_hit or action_caution_hit):
            penalty += 3.0
        if action_caution_hit:
            penalty += 2.0
        if bool(recommendation.get("late_stage_momentum_flag", False)):
            penalty += 1.5

        distribution_risk_score = float(recommendation.get("distribution_risk_score") or 0.0)
        if distribution_risk_score >= 2.0:
            penalty += 1.5
        elif distribution_risk_score >= 1.2:
            penalty += 0.8

        recent_runup_5d = float(recommendation.get("recent_runup_5d") or 0.0)
        moneyflow_3d_value = float(recommendation.get("moneyflow_3d_value") or 0.0)
        turnover_spike_ratio = float(recommendation.get("turnover_spike_ratio") or 0.0)
        if recent_runup_5d >= 10.0 and moneyflow_3d_value < 5000:
            penalty += 1.5
        if recent_runup_5d >= 15.0 and turnover_spike_ratio >= 1.8:
            penalty += 1.0
        return round(min(penalty, TOP3_CONTRADICTION_PENALTY), 2)

    @classmethod
    def _build_top_ranking_score(
        cls,
        code: str,
        recommendation: Dict[str, Any],
        stock: Any,
        *,
        apply_divergence_penalty: bool,
    ) -> float:
        base_score = float(
            recommendation.get("weighted_score")
            or recommendation.get("recommendation_score")
            or recommendation.get("score")
            or getattr(stock, "recommendation_score", None)
            or getattr(stock, "score", None)
            or 0.0
        )
        if not apply_divergence_penalty:
            return base_score
        divergence_score = float(recommendation.get("divergence_score") or 0.0)
        contradiction_penalty = cls._build_short_term_contradiction_penalty(recommendation)
        continuation_bias_score = float(recommendation.get("continuation_bias_score") or 0.0)
        return round(
            base_score
            + continuation_bias_score
            - divergence_score * TOP3_DIVERGENCE_PENALTY_MULTIPLIER
            - contradiction_penalty,
            4,
        )

    @classmethod
    def _build_action_plan(cls, recommendation: Dict[str, Any], stock: Any, previous_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        close = getattr(stock, "close", None) or (previous_state or {}).get("close")
        entry_value = getattr(stock, "close", None) or (previous_state or {}).get("entry_price")
        signal = str(recommendation.get("technical_signal") or "").strip()
        recommendation_text = str(
            recommendation.get("recommendation_text")
            or recommendation.get("recommendation")
            or recommendation.get("ai_summary")
            or ""
        ).strip()
        risk_blocked = bool(recommendation.get("candidate_risk_blocked", False))
        weighted_score = float(
            recommendation.get("weighted_score")
            or recommendation.get("recommendation_score")
            or recommendation.get("score")
            or 0.0
        )
        if risk_blocked or cls._contains_any_keyword(recommendation_text, ["暂不参与"]):
            action_bias = "回避"
        elif cls._contains_any_keyword(recommendation_text, ["优先关注"]):
            action_bias = "关注买点"
        elif cls._contains_any_keyword(recommendation_text, ["建议跟踪"]):
            action_bias = "跟踪"
        elif weighted_score >= 60:
            action_bias = "观察"
        else:
            action_bias = "回避"
        return {
            "action_bias": action_bias,
            "entry_zone": f"{entry_value:.2f} 附近观察" if isinstance(entry_value, (int, float)) else "等待回踩或放量确认",
            "take_profit": f"{close * 1.05:.2f} 附近分批止盈" if isinstance(close, (int, float)) else "结合盘中强弱分批止盈",
            "stop_loss": f"{close * 0.97:.2f}" if isinstance(close, (int, float)) else "跌破关键支撑止损",
            "holding_horizon": "1-5个交易日",
            "invalid_condition": signal or "量价结构走弱",
        }
