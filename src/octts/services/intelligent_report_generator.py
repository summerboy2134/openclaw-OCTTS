"""Intelligent report generation system."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from octts.config import Settings
from octts.clients.llm_client import LLMClient
from octts.prompts.report_prompt import (
    build_today_screening_analysis_prompt,
    build_today_screening_format_prompt,
    build_today_screening_report_prompt,
    build_yesterday_review_report_prompt,
)
from octts.services.news_aggregator import NewsCluster
from octts.services.multi_dimensional_analyzer import MultiDimensionalAnalyzer


TECHNICAL_SIGNAL_LABELS = {
    "bullish": "趋势偏强",
    "bullish_rising": "趋势走强",
    "neutral": "震荡观察",
    "mixed": "震荡整理",
    "improving": "逐步转强",
    "bearish_improving": "弱势修复",
    "weak_bearish": "偏弱运行",
    "bearish": "明显转弱",
}

logger = logging.getLogger(__name__)

SCREENING_REPORT_MIN_MAX_TOKENS = 6000
INTERNAL_REPORT_TEXT_MARKERS = (
    "回补样本",
    "训练集构造",
    "结构化特征近似",
    "用于短线训练",
    "fallback",
    "fallback_reason",
    "backfill",
)


class ReportType(Enum):
    """报告类型"""

    MORNING = "morning"
    NOON = "noon"
    EVENING = "evening"
    WEEKLY = "weekly"
    SPECIAL = "special"


@dataclass
class ReportSection:
    """报告章节"""

    title: str
    content: str
    priority: int
    data: Optional[Dict[str, Any]] = None


@dataclass
class IntelligentReport:
    """智能报告"""

    report_id: str
    report_type: ReportType
    title: str
    generate_time: datetime
    sections: List[ReportSection]
    summary: str
    key_points: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class IntelligentReportGenerator:
    """智能报告生成器"""

    def __init__(
        self,
        settings: Settings,
        llm_client: Optional[LLMClient] = None,
        analyzer: Optional[MultiDimensionalAnalyzer] = None,
    ):
        self.settings = settings
        self.llm_client = llm_client or LLMClient(settings)
        self.analyzer = analyzer or MultiDimensionalAnalyzer(settings)

    async def generate_morning_report(
        self,
        news_clusters: List[NewsCluster],
        market_data: Dict[str, Any],
        stock_pool: List[str],
        screening_context: Optional[Dict[str, Any]] = None,
    ) -> IntelligentReport:
        logger.info("Generating morning report")

        report_blocks = await self._generate_screening_report_blocks(
            news_clusters=news_clusters,
            market_data=market_data,
            stock_pool=stock_pool,
            screening_context=screening_context or {},
        )

        sections = self._build_sections_from_blocks(report_blocks)
        summary = self._build_summary_from_blocks(report_blocks)
        key_points = self._build_key_points_from_blocks(report_blocks)
        recommendations = [
            item.get("ts_code")
            for item in (screening_context or {}).get("today_top3") or []
            if item.get("ts_code")
        ]

        metadata = {
            "news_count": sum(len(c.news_items) for c in news_clusters),
            "cluster_count": len(news_clusters),
            "stock_count": len(stock_pool),
            "screening_context": screening_context or {},
            "report_blocks": report_blocks,
        }

        return IntelligentReport(
            report_id=f"morning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.MORNING,
            title=f"今日新闻 - {datetime.now().strftime('%Y年%m月%d日')}",
            generate_time=datetime.now(),
            sections=sections,
            summary=summary,
            key_points=key_points,
            recommendations=recommendations,
            metadata=metadata,
        )

    async def _generate_screening_report_blocks(
        self,
        *,
        news_clusters: List[NewsCluster],
        market_data: Dict[str, Any],
        stock_pool: List[str],
        screening_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        serialized_news_clusters = [self._serialize_news_cluster(item) for item in news_clusters[:6]]
        if not self.settings.screening_llm_enabled:
            logger.info("Screening report LLM generation disabled, using fallback report blocks")
            return self._build_fallback_report_blocks(stock_pool, market_data, screening_context, serialized_news_clusters)
        if not screening_context:
            return self._build_fallback_report_blocks(stock_pool, market_data, screening_context, serialized_news_clusters)

        today_payload = await self._generate_today_report_blocks(
            market_data=market_data,
            news_clusters=serialized_news_clusters,
            screening_context=screening_context,
        )
        yesterday_payload = await self._generate_yesterday_review_blocks(
            news_clusters=serialized_news_clusters,
            screening_context=screening_context,
        )

        if today_payload or yesterday_payload:
            return self._merge_report_blocks_with_context(
                {
                    "focus_stocks": today_payload.get("focus_stocks") or [],
                    "comparison": today_payload.get("comparison") or {},
                    "overall_action": today_payload.get("overall_action") or {},
                    "yesterday_reviews": yesterday_payload.get("yesterday_reviews") or [],
                },
                screening_context,
                market_data,
                serialized_news_clusters,
            )
        return self._build_fallback_report_blocks(stock_pool, market_data, screening_context, serialized_news_clusters)

    async def _generate_today_report_blocks(
        self,
        *,
        market_data: Dict[str, Any],
        news_clusters: List[Dict[str, Any]],
        screening_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched_context = self._enrich_screening_context_for_report(screening_context)
        two_stage_payload = await self._generate_two_stage_today_report_blocks(
            market_data=market_data,
            news_clusters=news_clusters,
            screening_context=enriched_context,
        )
        if two_stage_payload:
            return two_stage_payload

        logger.info("Falling back to single-stage today screening report generation")
        system_prompt, user_prompt = build_today_screening_report_prompt(
            market_data=market_data,
            news_clusters=news_clusters,
            screening_context=enriched_context,
        )
        return await self._request_report_payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            log_label="today structured screening report",
        )

    def _enrich_screening_context_for_report(self, screening_context: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(screening_context or {})
        today_top3 = []
        for item in enriched.get("today_top3") or []:
            if not isinstance(item, dict):
                continue
            stock_payload = dict(item)
            stock_payload.setdefault("evidence_digest", self._build_stock_evidence_digest(stock_payload))
            today_top3.append(stock_payload)
        enriched["today_top3"] = today_top3
        enriched["comparison_candidates"] = list(today_top3)
        enriched["cross_stock_synthesis"] = self._build_cross_stock_synthesis(today_top3)
        return enriched

    async def _generate_two_stage_today_report_blocks(
        self,
        *,
        market_data: Dict[str, Any],
        news_clusters: List[Dict[str, Any]],
        screening_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        analysis_system_prompt, analysis_user_prompt = build_today_screening_analysis_prompt(
            market_data=market_data,
            news_clusters=news_clusters,
            screening_context=screening_context,
        )
        reasoning_payload = await self._request_report_payload(
            user_prompt=analysis_user_prompt,
            system_prompt=analysis_system_prompt,
            log_label="today screening reasoning draft",
            model=self.settings.llm_report_analysis_model,
        )
        if not reasoning_payload:
            return {}

        format_system_prompt, format_user_prompt = build_today_screening_format_prompt(
            market_data=market_data,
            news_clusters=news_clusters,
            screening_context=screening_context,
            reasoning_payload=reasoning_payload,
        )
        return await self._request_report_payload(
            user_prompt=format_user_prompt,
            system_prompt=format_system_prompt,
            log_label="today structured screening report formatter",
            model=self.settings.llm_report_formatter_model,
        )

    def _build_stock_evidence_digest(self, item: Dict[str, Any]) -> Dict[str, Any]:
        positive: List[str] = []
        risk: List[str] = []
        missing: List[str] = []
        operating: List[str] = []

        name = item.get("name") or item.get("ts_code") or "该股"
        score_snapshot = item.get("score_snapshot") or {}
        model_rank = score_snapshot.get("model_rank", item.get("model_rank"))
        recommendation_score = score_snapshot.get("recommendation_score", item.get("recommendation_score"))
        overall_score = score_snapshot.get("overall_score", item.get("overall_score"))
        risk_score = score_snapshot.get("risk_score", item.get("risk_score", item.get("distribution_risk_score")))
        if model_rank not in (None, ""):
            positive.append(f"模型全市场排序第{model_rank}，是进入Top3的主排序依据")
        if recommendation_score not in (None, ""):
            positive.append(f"最终推荐分为{recommendation_score}")
        if overall_score not in (None, ""):
            positive.append(f"多维综合分为{overall_score}")

        pct_change = item.get("pct_change")
        turnover_rate = item.get("turnover_rate")
        ma20 = item.get("ma20")
        price_position = item.get("price_position_20d")
        if pct_change not in (None, ""):
            positive.append(f"当日涨跌幅{float(pct_change):+.2f}%")
        else:
            missing.append("缺少当日涨跌幅，市场表现阶段只能保守判断")
        if turnover_rate not in (None, ""):
            positive.append(f"换手率{float(turnover_rate):.2f}%")
        else:
            missing.append("缺少换手率，量能承接判断不完整")
        if ma20 not in (None, ""):
            positive.append(f"MA20为{float(ma20):.2f}")
        if price_position not in (None, ""):
            positive.append(f"20日价格位置为{float(price_position):.2f}")

        fund_flow = item.get("main_fund_flow_3d")
        if fund_flow in (None, ""):
            fund_flow = item.get("moneyflow_3d_value")
        if fund_flow not in (None, ""):
            flow_value = float(fund_flow)
            if flow_value >= 0:
                positive.append(f"近3日主力资金净流入约{flow_value:.1f}")
            else:
                risk.append(f"近3日主力资金净流出约{flow_value:.1f}")
        else:
            missing.append("缺少近3日资金流，资金承接需要盘中继续确认")

        for flag in item.get("distribution_risk_flags") or []:
            text = str(flag).strip()
            if text:
                risk.append(text)
        if risk_score not in (None, ""):
            risk_score_value = float(risk_score or 0.0)
            if risk_score_value > 0:
                risk.append(f"分歧/派发风险分约{risk_score_value:.2f}，需要作为排序扣分和执行风控依据")
            else:
                operating.append("结构化末端风险分暂未触发明显异常，但仍需结合资金承接和换手变化验证")
        if item.get("top3_extreme_risk_blocked"):
            risk.append(f"触发Top3极端风险排除：{item.get('top3_extreme_risk_reason') or '原因未明'}")
        if item.get("late_stage_momentum_flag"):
            risk.append("存在末端分歧风险")

        forecast = item.get("earnings_forecast") or {}
        if isinstance(forecast, dict) and forecast:
            summary = forecast.get("summary") or forecast.get("change_reason") or forecast.get("type")
            if summary:
                positive.append(f"业绩预告线索：{str(summary)[:120]}")
        else:
            missing.append("缺少可用业绩预告，基本面催化不能编造")

        top_list_summary = item.get("top_list_summary")
        limit_status = item.get("limit_status")
        if top_list_summary:
            positive.append(f"龙虎榜线索：{top_list_summary}")
        if limit_status:
            positive.append(f"涨跌停/连板线索：{limit_status}")

        action = item.get("action_plan") or {}
        for key in ("entry_zone", "take_profit", "stop_loss", "invalid_condition"):
            value = action.get(key)
            if value not in (None, ""):
                operating.append(f"{key}: {value}")
        if not operating:
            operating.append(f"{name} 缺少明确操作位，只能给条件型观察建议")

        return {
            "positive_evidence": positive[:10],
            "risk_and_counter_evidence": risk[:8],
            "missing_data": list(dict.fromkeys(missing)),
            "operating_premises": operating[:6],
            "analysis_task": "请基于正向证据、反证和缺失数据做归纳，不要逐字段复述。",
        }

    def _build_cross_stock_synthesis(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {}

        def _pick_max(key: str) -> Optional[str]:
            candidates = [item for item in items if item.get(key) not in (None, "")]
            if not candidates:
                return None
            return max(candidates, key=lambda item: float(item.get(key) or 0.0)).get("ts_code")

        def _pick_min(key: str) -> Optional[str]:
            candidates = [item for item in items if item.get(key) not in (None, "")]
            if not candidates:
                return None
            return min(candidates, key=lambda item: float(item.get(key) or 0.0)).get("ts_code")

        return {
            "highest_model_priority": _pick_min("model_rank"),
            "strongest_trade_score": _pick_max("recommendation_score"),
            "highest_overall_quality": _pick_max("overall_score"),
            "highest_risk": _pick_max("risk_score") or _pick_max("distribution_risk_score"),
            "lowest_risk": _pick_min("risk_score") or _pick_min("distribution_risk_score"),
            "task": "请比较三只Top3的交易性、质量、风险和证据完整度，给出横向综合判断。",
        }

    async def _generate_yesterday_review_blocks(
        self,
        *,
        news_clusters: List[Dict[str, Any]],
        screening_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = build_yesterday_review_report_prompt(
            news_clusters=news_clusters[:3],
            screening_context=screening_context,
        )
        return await self._request_report_payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            log_label="yesterday review report",
        )

    async def _request_report_payload(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        log_label: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        report_max_tokens = max(self.settings.llm_max_tokens, SCREENING_REPORT_MIN_MAX_TOKENS)
        logger.info(
            "Generating %s with model=%s max_tokens=%s",
            log_label,
            model or self.settings.llm_model,
            report_max_tokens,
        )
        try:
            response = await self.llm_client.complete(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=report_max_tokens,
                model=model,
            )
            return self._parse_json_response(response)
        except Exception:
            logger.exception("Failed to generate %s", log_label)
            return {}

    def _build_sections_from_blocks(self, blocks: Dict[str, Any]) -> List[ReportSection]:
        sections = [
            ReportSection(
                title="今日 Top3 重点点评",
                content=self._format_focus_stock_section(blocks.get("focus_stocks") or []),
                priority=1,
                data={"items": blocks.get("focus_stocks") or []},
            ),
            ReportSection(
                title="昨日 Top3 今日复盘",
                content=self._format_review_section(blocks.get("yesterday_reviews") or []),
                priority=2,
                data={"items": blocks.get("yesterday_reviews") or []},
            ),
            ReportSection(
                title="今日推荐股票横向比较",
                content=self._format_comparison_section(blocks.get("comparison") or {}),
                priority=3,
                data=blocks.get("comparison") or {},
            ),
            ReportSection(
                title="今日整体操作建议 / 风险总览",
                content=self._format_overall_action_section(blocks.get("overall_action") or {}),
                priority=4,
                data=blocks.get("overall_action") or {},
            ),
        ]
        return sections

    def _build_summary_from_blocks(self, blocks: Dict[str, Any]) -> str:
        overall = blocks.get("overall_action") or {}
        market_view = str(overall.get("market_view") or "今日以结构化复盘为主。")
        headline = str(overall.get("headline") or market_view)
        return f"{headline} {market_view}".strip()

    def _build_key_points_from_blocks(self, blocks: Dict[str, Any]) -> List[str]:
        points: List[str] = []
        overall = blocks.get("overall_action") or {}
        for item in overall.get("action_items") or []:
            text = str(item).strip()
            if text:
                points.append(text)
        for stock in (blocks.get("focus_stocks") or [])[:3]:
            assessment = str(stock.get("overall_assessment") or "").strip()
            if assessment:
                points.append(f"{stock.get('ts_code', '')}：{assessment}")
        return points[:7]

    def _merge_report_blocks_with_context(
        self,
        payload: Dict[str, Any],
        screening_context: Dict[str, Any],
        market_data: Dict[str, Any],
        news_clusters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged_focus, comparison, overall_action = self._merge_today_report_blocks(
            payload,
            screening_context,
            market_data,
        )
        merged_reviews = self._merge_yesterday_review_blocks(payload, screening_context)
        theme_focus_items = self._build_theme_focus_items(news_clusters, overall_action)
        theme_view = self._split_theme_clusters(theme_focus_items, market_data, overall_action, news_clusters)
        overall_action["theme_focuses"] = theme_focus_items

        return {
            "news_clusters": news_clusters,
            "theme_view": theme_view,
            "focus_stocks": merged_focus,
            "yesterday_reviews": merged_reviews,
            "comparison": comparison,
            "overall_action": overall_action,
        }

    def _merge_today_report_blocks(
        self,
        payload: Dict[str, Any],
        screening_context: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        authoritative_today_top3 = [item for item in (screening_context.get("today_top3") or []) if item.get("ts_code")]
        focus_map = {item.get("ts_code"): item for item in authoritative_today_top3}
        merged_focus = []
        payload_focus_map = {
            item.get("ts_code"): item
            for item in (payload.get("focus_stocks") or [])
            if item.get("ts_code")
        }
        for item in authoritative_today_top3:
            merged_focus.append(self._merge_focus_stock_item(payload_focus_map.get(item.get("ts_code"), {}), focus_map.get(item.get("ts_code"))))

        comparison = dict(payload.get("comparison") or {})
        comparison_candidates = authoritative_today_top3
        today_top3 = authoritative_today_top3
        comparison.setdefault("basic_rank", self._rank_codes(comparison_candidates, "fundamental_score"))
        comparison.setdefault("technical_rank", self._rank_codes(comparison_candidates, "technical_score"))
        comparison.setdefault("risk_rank", self._rank_codes(comparison_candidates, "risk_score", reverse=False))
        comparison.setdefault("trading_rank", self._rank_codes(comparison_candidates, "recommendation_score"))
        comparison.setdefault("best_short_term", self._pick_code(today_top3))
        comparison.setdefault("most_robust", self._pick_code(comparison_candidates, key="overall_score"))
        comparison.setdefault("highest_risk", self._pick_code(comparison_candidates, key="risk_score", reverse=True))
        comparison.setdefault(
            "cross_stock_synthesis_view",
            self._build_cross_stock_synthesis_view(screening_context.get("cross_stock_synthesis") or {}, today_top3),
        )
        comparison = self._attach_comparison_names(comparison, comparison_candidates + today_top3)

        overall_action = dict(payload.get("overall_action") or {})
        overall_action.setdefault("market_view", str(market_data.get("trend") or "市场概览数据不可用"))
        overall_action.setdefault("risk_summary", str(market_data.get("sentiment") or "市场风险概览数据不可用"))
        overall_action.setdefault("action_items", ["优先围绕今日 Top3 与昨日复评对象跟踪。"])
        overall_action.setdefault("headline", "围绕系统排序做解释与执行建议")
        overall_action.setdefault("theme_focuses", overall_action.get("theme_focuses") or [])
        return merged_focus, comparison, overall_action

    @staticmethod
    def _build_cross_stock_synthesis_view(synthesis: Dict[str, Any], today_top3: List[Dict[str, Any]]) -> str:
        if not today_top3:
            return "暂无Top3横向比较数据。"
        name_map = {
            item.get("ts_code"): item.get("name") or item.get("ts_code")
            for item in today_top3
            if item.get("ts_code")
        }

        def _name(code: Any) -> str:
            return str(name_map.get(code) or code or "暂无")

        parts = [
            f"模型优先级最高的是{_name(synthesis.get('highest_model_priority'))}",
            f"交易分最强的是{_name(synthesis.get('strongest_trade_score'))}",
            f"综合质量最高的是{_name(synthesis.get('highest_overall_quality'))}",
            f"风险最高的是{_name(synthesis.get('highest_risk'))}",
            f"风险最低的是{_name(synthesis.get('lowest_risk'))}",
        ]
        return "；".join(parts) + "。"

    def _merge_yesterday_review_blocks(
        self,
        payload: Dict[str, Any],
        screening_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        review_map = {
            item.get("ts_code"): item
            for item in screening_context.get("yesterday_top3_review") or []
            if item.get("ts_code")
        }
        merged_reviews = []
        for item in payload.get("yesterday_reviews") or []:
            merged_reviews.append(self._merge_review_item(item, review_map.get(item.get("ts_code"))))

        existing_review_codes = {item.get("ts_code") for item in merged_reviews}
        for item in screening_context.get("yesterday_top3_review") or []:
            code = item.get("ts_code")
            if code and code not in existing_review_codes:
                merged_reviews.append(self._fallback_review_item(item))
        return merged_reviews

    def _merge_focus_stock_item(self, item: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not context:
            return self._fallback_focus_stock_item(item)
        merged = dict(context)
        merged.update(item)
        action_plan = dict(context.get("action_plan") or {})
        action_plan.update(item.get("action_plan") or {})
        merged["action_plan"] = {
            "action_bias": action_plan.get("action_bias") or "观察",
            "entry_zone": action_plan.get("entry_zone") or "等待更清晰触发位",
            "take_profit": action_plan.get("take_profit") or "结合盘中强弱分批处理",
            "stop_loss": action_plan.get("stop_loss") or "跌破关键支撑后止损",
            "holding_horizon": action_plan.get("holding_horizon") or "1-5个交易日",
            "invalid_condition": action_plan.get("invalid_condition") or "技术结构失真或量能承接不足",
        }
        merged["score_rationale"] = self._clean_generated_text(merged.get("score_rationale") or self._build_focus_score_rationale(merged))
        merged["fundamental_view"] = self._clean_generated_text(merged.get("fundamental_view") or self._build_focus_fundamental_view(merged))
        merged["market_context_view"] = self._clean_generated_text(merged.get("market_context_view") or self._build_focus_market_context_view(merged))
        merged["trading_context_view"] = self._clean_generated_text(merged.get("trading_context_view") or self._build_focus_trading_context_view(merged))
        merged["market_performance_view"] = self._clean_generated_text(merged.get("market_performance_view") or self._build_focus_market_performance_view(merged))
        merged["catalyst_and_capital_view"] = self._clean_generated_text(merged.get("catalyst_and_capital_view") or self._build_focus_catalyst_and_capital_view(merged))
        overall_assessment = self._clean_generated_text(merged.get("overall_assessment"))
        if self._looks_like_score_template_text(overall_assessment):
            overall_assessment = ""
        merged["overall_assessment"] = overall_assessment or self._build_focus_overall_assessment(merged)
        focus_analysis = self._clean_generated_text(merged.get("focus_analysis"))
        if self._focus_analysis_needs_fallback(focus_analysis, merged["overall_assessment"]):
            focus_analysis = ""
        merged["focus_analysis"] = focus_analysis or self._build_focus_analysis_fallback(merged)
        merged["evidence_based_view"] = self._clean_generated_text(
            merged.get("evidence_based_view") or self._build_evidence_based_view(merged)
        )
        merged["counter_evidence"] = self._clean_generated_text(
            merged.get("counter_evidence") or self._build_counter_evidence_view(merged)
        )
        merged["core_highlights"] = self._clean_text_list(merged.get("core_highlights") or [])
        if not merged["core_highlights"]:
            merged["core_highlights"] = self._build_focus_core_highlights(merged, self._build_focus_risk_note(merged))
        merged["risk_warnings"] = self._clean_text_list(merged.get("risk_warnings") or [])
        if not merged["risk_warnings"]:
            risk_note = self._build_focus_risk_note(merged)
            merged["risk_warnings"] = [risk_note] if risk_note else ["需结合盘中承接继续确认风险"]
        return merged

    def _merge_review_item(self, item: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not context:
            return self._fallback_review_item(item)
        merged = dict(context)
        merged.update(item)
        action_plan = dict(context.get("action_plan") or {})
        action_plan.update(item.get("action_plan") or {})
        merged["action_plan"] = {
            "action_bias": action_plan.get("action_bias") or "观察",
            "entry_zone": action_plan.get("entry_zone") or "等待更清晰触发位",
            "take_profit": action_plan.get("take_profit") or "结合盘中强弱分批止盈",
            "stop_loss": action_plan.get("stop_loss") or "跌破关键支撑后止损/离场",
            "holding_horizon": action_plan.get("holding_horizon") or "1-5个交易日",
            "invalid_condition": action_plan.get("invalid_condition") or "走势转弱且量能承接不足",
        }
        merged["today_verdict"] = str(merged.get("today_verdict") or context.get("today_verdict") or merged.get("review_status") or "待复评")
        merged["review_status"] = str(merged.get("review_status") or context.get("review_status") or merged.get("status") or "延续")
        merged["status"] = str(merged.get("status") or merged.get("review_status") or "延续")
        strength_change = str(merged.get("strength_change") or "").strip()
        if self._looks_like_score_template_text(strength_change):
            strength_change = ""
        merged["strength_change"] = strength_change or self._build_review_strength_change(merged)
        merged["market_context_view"] = str(merged.get("market_context_view") or self._build_review_market_context_view(merged))
        review_analysis = str(merged.get("review_analysis") or "").strip()
        if self._review_analysis_needs_fallback(review_analysis, merged.get("analysis") or merged.get("absence_reason") or ""):
            review_analysis = ""
        merged["review_analysis"] = review_analysis or self._build_review_analysis_fallback(merged)
        merged.setdefault("miss_reason_candidates", context.get("miss_reason_candidates") or [])
        merged.setdefault("missing_factor_candidates", context.get("missing_factor_candidates") or [])
        return merged

    def _fallback_focus_stock_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        risk_warnings = [str(flag).strip() for flag in (item.get("distribution_risk_flags") or []) if str(flag).strip()]
        if not risk_warnings:
            risk_note = self._build_focus_risk_note(item)
            risk_warnings = [risk_note] if risk_note else ["需结合盘中承接继续确认风险"]
        funds_text = self._build_focus_risk_note(item)
        fallback = {
            **item,
            "core_highlights": self._build_focus_core_highlights(item, funds_text),
            "risk_warnings": risk_warnings,
            "overall_assessment": str(item.get("summary") or item.get("recommendation_text") or "建议结合盘中信号跟踪。"),
            "action_plan": dict(item.get("action_plan") or {}),
        }
        fallback["score_rationale"] = self._clean_generated_text(self._build_focus_score_rationale(fallback))
        fallback["fundamental_view"] = self._clean_generated_text(self._build_focus_fundamental_view(fallback))
        fallback["market_context_view"] = self._clean_generated_text(self._build_focus_market_context_view(fallback))
        fallback["trading_context_view"] = self._clean_generated_text(self._build_focus_trading_context_view(fallback))
        fallback["market_performance_view"] = self._clean_generated_text(self._build_focus_market_performance_view(fallback))
        fallback["catalyst_and_capital_view"] = self._clean_generated_text(self._build_focus_catalyst_and_capital_view(fallback))
        fallback["overall_assessment"] = self._build_focus_overall_assessment(fallback)
        fallback["focus_analysis"] = self._clean_generated_text(self._build_focus_analysis_fallback(fallback))
        fallback["evidence_based_view"] = self._clean_generated_text(self._build_evidence_based_view(fallback))
        fallback["counter_evidence"] = self._clean_generated_text(self._build_counter_evidence_view(fallback))
        fallback["core_highlights"] = self._clean_text_list(fallback.get("core_highlights") or [])
        fallback["risk_warnings"] = self._clean_text_list(fallback.get("risk_warnings") or [])
        return fallback

    def _build_evidence_based_view(self, item: Dict[str, Any]) -> str:
        digest = item.get("evidence_digest") or {}
        if not isinstance(digest, dict):
            return ""
        positives = [str(value).strip() for value in digest.get("positive_evidence") or [] if str(value).strip()]
        premises = [str(value).strip() for value in digest.get("operating_premises") or [] if str(value).strip()]
        parts: List[str] = []
        if positives:
            parts.append("核心证据包括" + "；".join(positives[:5]) + "。")
        if premises:
            parts.append("操作前提是" + "；".join(premises[:3]) + "。")
        return "".join(parts)

    def _build_counter_evidence_view(self, item: Dict[str, Any]) -> str:
        digest = item.get("evidence_digest") or {}
        if not isinstance(digest, dict):
            return ""
        risks = [str(value).strip() for value in digest.get("risk_and_counter_evidence") or [] if str(value).strip()]
        missing = [str(value).strip() for value in digest.get("missing_data") or [] if str(value).strip()]
        parts: List[str] = []
        if risks:
            parts.append("主要反证/风险是" + "；".join(risks[:5]) + "。")
        if missing:
            parts.append("缺失信息包括" + "；".join(missing[:3]) + "。")
        return "".join(parts)

    def _build_focus_risk_note(self, item: Dict[str, Any]) -> str:
        moneyflow_value = item.get("moneyflow_3d_value")
        turnover_spike_ratio = item.get("turnover_spike_ratio")
        recent_runup_5d = item.get("recent_runup_5d")
        distribution_risk_score = item.get("distribution_risk_score")
        distribution_risk_flags = [
            str(flag).strip()
            for flag in (item.get("distribution_risk_flags") or [])
            if str(flag).strip()
        ]
        risk_parts: List[str] = []
        if distribution_risk_flags:
            risk_parts.extend(distribution_risk_flags[:2])
        if distribution_risk_score not in (None, ""):
            risk_score = float(distribution_risk_score or 0.0)
            if risk_score >= 6:
                risk_parts.append(f"分歧/派发风险偏高({risk_score:.1f})")
            elif risk_score >= 3:
                risk_parts.append(f"分歧/派发风险中等({risk_score:.1f})")
        if turnover_spike_ratio not in (None, ""):
            turnover_ratio_value = float(turnover_spike_ratio or 0.0)
            if turnover_ratio_value >= 1.05:
                risk_parts.append(f"换手较近5日均值放大{turnover_ratio_value:.2f}倍")
        if recent_runup_5d not in (None, ""):
            runup = float(recent_runup_5d or 0.0)
            if runup >= 8:
                risk_parts.append(f"近5日累计涨幅{runup:.1f}%，追高容错率下降")
            elif runup <= -6:
                risk_parts.append(f"近5日累计回撤{runup:.1f}%，仍需确认修复质量")
        if item.get("late_stage_momentum_flag"):
            risk_parts.append("存在末端分歧风险")
        if moneyflow_value not in (None, "") and float(moneyflow_value or 0.0) < 0:
            risk_parts.append(f"近3日资金净流出({float(moneyflow_value or 0.0):.0f})，承接不足")
        return "；".join(risk_parts) if risk_parts else "暂未触发明显末端风险，但仍需看资金承接能否延续"

    def _build_focus_core_highlights(self, item: Dict[str, Any], funds_text: str) -> List[str]:
        highlights: List[str] = []
        pct_change = item.get("pct_change")
        recent_runup_5d = item.get("recent_runup_5d")
        technical_signal = self._localize_technical_signal(item.get("technical_signal"))
        business_summary = self._clean_generated_text(item.get("business_summary"))
        catalyst_summary = self._clean_generated_text(item.get("catalyst_summary"))
        strategy_count = item.get("strategy_count")
        score_explanations = item.get("score_explanations") or []

        if pct_change not in (None, ""):
            pct_value = float(pct_change)
            if pct_value >= 5:
                highlights.append(f"当日涨幅{pct_value:+.2f}%，短线处于强势推进区间")
            elif pct_value >= 2:
                highlights.append(f"当日涨幅{pct_value:+.2f}%，强度仍在延续")
            elif pct_value <= -3:
                highlights.append(f"当日回撤{pct_value:+.2f}%，正在观察修复力度")

        if technical_signal and technical_signal != "待确认":
            highlights.append(f"技术结构处于{technical_signal}，是当前交易判断的重要依据")

        if recent_runup_5d not in (None, ""):
            runup_value = float(recent_runup_5d)
            if runup_value >= 10:
                highlights.append(f"近5日累计涨幅{runup_value:.1f}%，已有明显活跃度与辨识度")
            elif runup_value <= -8:
                highlights.append(f"近5日累计涨幅{runup_value:.1f}%，当前更偏修复博弈")

        if business_summary:
            summary_text = business_summary[:40].rstrip("，。； ")
            highlights.append(f"主营逻辑聚焦{summary_text}，提供了基本面识别锚点")
        elif catalyst_summary:
            catalyst_text = catalyst_summary[:40].rstrip("，。； ")
            highlights.append(f"当前催化主要围绕{catalyst_text}")

        if strategy_count not in (None, ""):
            highlights.append(f"命中策略{int(float(strategy_count or 0))}个，说明并非单一信号触发")

        for text in score_explanations:
            cleaned = self._clean_generated_text(text)
            if cleaned and "风险" not in cleaned and "修正" not in cleaned:
                highlights.append(cleaned)
                break

        if funds_text and "异常" not in funds_text:
            highlights.append(f"资金与换手侧重点在于{funds_text}")

        unique_highlights: List[str] = []
        seen = set()
        for text in highlights:
            cleaned = text.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_highlights.append(cleaned)
            if len(unique_highlights) >= 4:
                break

        return unique_highlights or ["当前进入重点跟踪名单，后续需继续观察量价与承接是否共振"]

    def _fallback_review_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        action_plan = dict(item.get("action_plan") or {})
        fallback = {
            **item,
            "status": str(item.get("status") or item.get("review_status") or "待复评"),
            "review_status": str(item.get("review_status") or item.get("status") or "待复评"),
            "yesterday_conclusion": str(item.get("yesterday_conclusion") or item.get("overview_reason") or item.get("summary") or item.get("recommendation_text") or "昨日入选 Top3"),
            "today_verdict": str(item.get("today_verdict") or item.get("review_status") or item.get("status") or "待复评"),
            "analysis": str(item.get("analysis") or item.get("absence_reason") or "今日表现需要结合缺席原因复盘。"),
            "action_plan": {
                "action_bias": action_plan.get("action_bias") or "观察",
                "entry_zone": action_plan.get("entry_zone") or "等待更清晰触发位",
                "take_profit": action_plan.get("take_profit") or "结合盘中强弱分批止盈",
                "stop_loss": action_plan.get("stop_loss") or "跌破关键支撑后止损/离场",
                "holding_horizon": action_plan.get("holding_horizon") or "1-5个交易日",
                "invalid_condition": action_plan.get("invalid_condition") or "走势转弱且量能承接不足",
            },
            "miss_reason_candidates": list(item.get("miss_reason_candidates") or []),
            "missing_factor_candidates": list(item.get("missing_factor_candidates") or []),
        }
        fallback["strength_change"] = self._build_review_strength_change(fallback)
        fallback["market_context_view"] = self._build_review_market_context_view(fallback)
        fallback["review_analysis"] = self._build_review_analysis_fallback(fallback)
        return fallback

    def _build_focus_score_rationale(self, item: Dict[str, Any]) -> str:
        recommendation_score = float(item.get("recommendation_score") or 0.0)
        overall_score = float(item.get("overall_score") or 0.0)
        base_score = item.get("base_score")
        sentiment_adjustment = item.get("sentiment_adjustment")
        news_adjustment = item.get("news_adjustment")
        parts = [f"当前排序主要由推荐分{recommendation_score:.1f}与综合分{overall_score:.1f}共同决定。"]
        if base_score not in (None, ""):
            parts.append(f"综合分主体由技术面与基本面合成主评分{float(base_score or 0.0):.1f}。")
        adjustment_parts: List[str] = []
        if sentiment_adjustment not in (None, ""):
            adjustment_parts.append(f"情绪修正{float(sentiment_adjustment or 0.0):+.1f}")
        if news_adjustment not in (None, ""):
            adjustment_parts.append(f"新闻修正{float(news_adjustment or 0.0):+.1f}")
        if adjustment_parts:
            parts.append(f"情绪与新闻仅作为辅助修正项，当前为{'、'.join(adjustment_parts)}。")

        strategy_count = item.get("strategy_count")
        strategy_consistency_label = str(item.get("strategy_consistency_label") or "").strip()
        divergence_score = item.get("divergence_score")
        if strategy_count not in (None, ""):
            strategy_count_value = int(float(strategy_count or 0))
            if strategy_count_value <= 1:
                parts.append("当前主要由单策略触发，后续更要看盘中承接是否能扩散。")
            elif strategy_consistency_label == "存在分歧":
                parts.append(
                    f"命中策略数为{strategy_count_value}个，但内部判断存在分歧，策略技术分跨度约{float(divergence_score or 0.0):.1f}分，因此Top3排序只做了轻度折减。"
                )
            elif strategy_consistency_label == "多策略一致":
                parts.append(f"命中策略数为{strategy_count_value}个，且多策略一致，说明并非单一触发。")
            else:
                parts.append(f"命中策略数为{strategy_count_value}个，说明本次排序并非单一触发。")

        industry_heat_score = item.get("industry_heat_score")
        industry_flow_bias = str(item.get("industry_flow_bias") or "").strip()
        if industry_heat_score not in (None, "") or industry_flow_bias:
            industry_heat_value = float(industry_heat_score or 0.0)
            if industry_heat_value >= 0.45:
                heat_text = f"行业层面处于顺风，带来约{industry_heat_value:+.1f}分的轻度加分"
            elif industry_heat_value <= -0.45:
                heat_text = f"行业层面处于逆风，带来约{industry_heat_value:+.1f}分的轻度压分"
            elif industry_heat_score not in (None, ""):
                heat_text = f"行业热度修正约{industry_heat_value:+.1f}分，整体仍接近中性"
            else:
                heat_text = "行业层面整体接近中性"
            if industry_flow_bias and industry_flow_bias != "中性":
                heat_text = f"{heat_text}，行业资金风格偏向“{industry_flow_bias}”"
            parts.append(f"{heat_text}。")

        distribution_risk_score = item.get("distribution_risk_score")
        if distribution_risk_score not in (None, "") and float(distribution_risk_score or 0.0) > 0:
            parts.append(f"分歧/派发风险分为{float(distribution_risk_score or 0.0):.1f}，会对最终排序形成风险折减。")
        contradiction_penalty = item.get("short_term_contradiction_penalty")
        if contradiction_penalty not in (None, "") and float(contradiction_penalty or 0.0) > 0:
            parts.append(f"短线执行层面存在结论与盘面强弱不完全一致的问题，因此Top3排序额外扣减{float(contradiction_penalty or 0.0):.1f}分。")

        score_change = item.get("score_change")
        if score_change not in (None, ""):
            parts.append(f"较上一交易日分数变化{float(score_change or 0.0):+.1f}，体现近期强弱边际变化。")

        dimension_texts: List[str] = []
        for label, key in (("技术", "technical_score"), ("基本面", "fundamental_score"), ("情绪", "sentiment_score"), ("新闻", "news_score")):
            value = item.get(key)
            if value in (None, ""):
                continue
            score = float(value or 0.0)
            if abs(score) >= 0.1:
                dimension_texts.append(f"{label}{score:.1f}")
        if dimension_texts:
            parts.append(f"维度分数参考为：{'、'.join(dimension_texts)}。其中技术面与基本面决定主方向，情绪与新闻仅做有限修正。")

        return "".join(parts)

    def _build_focus_overall_assessment(self, item: Dict[str, Any]) -> str:
        summary = self._clean_generated_text(item.get("summary") or item.get("recommendation_text"))
        if summary and not self._looks_like_score_template_text(summary):
            return summary

        recommendation_score = self._safe_float_value(item.get("recommendation_score"), 0.0)
        overall_score = self._safe_float_value(item.get("overall_score"), 0.0)
        profit_yoy = item.get("latest_profit_yoy")
        moneyflow_3d_value = item.get("moneyflow_3d_value")
        contradiction_penalty = self._safe_float_value(item.get("short_term_contradiction_penalty"), 0.0)
        pct_change = item.get("pct_change")

        quality_text = "更偏交易性驱动"
        if overall_score >= 80:
            quality_text = "综合质量与交易性都较完整"
        elif overall_score >= 65:
            quality_text = "综合质量尚可，但仍要看盘中确认"

        if recommendation_score >= overall_score + 3:
            quality_text = "短线交易性强于中线质量"
        elif overall_score >= recommendation_score + 3:
            quality_text = "质量端好于短线博弈端"

        risk_text = "更适合跟踪承接而不是直接追高"
        if contradiction_penalty > 0:
            risk_text = "执行层面存在分歧，追价容错率偏低"
        elif moneyflow_3d_value not in (None, "") and float(moneyflow_3d_value or 0.0) < 0:
            risk_text = "资金承接还不扎实，次日更容易走出高开分歧"
        elif profit_yoy not in (None, "") and float(profit_yoy or 0.0) < 0:
            risk_text = "基本面暂时不支撑纯趋势外推，更像短线题材博弈"
        elif pct_change not in (None, "") and float(pct_change or 0.0) >= 7:
            risk_text = "当日涨幅已经较大，次日重点看量价是否继续配合"

        return f"当前这只票{quality_text}，{risk_text}"

    def _localize_technical_signal(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "待确认"
        return TECHNICAL_SIGNAL_LABELS.get(text.lower(), text)

    def _clean_generated_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = self._sanitize_structured_field_leaks(text)
        if self._contains_internal_report_text(text):
            return ""
        for raw, label in TECHNICAL_SIGNAL_LABELS.items():
            text = text.replace(raw, label)
            text = text.replace(raw.upper(), label)
            text = text.replace(raw.capitalize(), label)
        return " ".join(text.split())

    def _clean_text_list(self, values: List[Any]) -> List[str]:
        cleaned_values: List[str] = []
        seen = set()
        for value in values:
            cleaned = self._clean_generated_text(value)
            if not cleaned or cleaned in seen:
                continue
            if self._is_low_signal_user_text(cleaned):
                continue
            seen.add(cleaned)
            cleaned_values.append(cleaned)
        return cleaned_values

    @staticmethod
    def _contains_internal_report_text(text: str) -> bool:
        return any(marker.lower() in str(text or "").lower() for marker in INTERNAL_REPORT_TEXT_MARKERS)

    @staticmethod
    def _is_low_signal_user_text(text: str) -> bool:
        compact = str(text or "").replace(" ", "")
        if not compact:
            return True
        if re.fullmatch(r"(风险分|risk_score|distribution_risk_score)(为|=)?0(?:\.0+)?", compact, flags=re.IGNORECASE):
            return True
        return compact in {"风险分为0.0", "风险分为0", "风险分0.0", "风险分0"}

    @staticmethod
    def _sanitize_structured_field_leaks(text: str) -> str:
        cleaned = text
        cleaned = re.sub(
            r"[，,；;。]?\s*(?:风险分|risk_score|distribution_risk_score)\s*(?:为|=)?\s*0(?:\.0+)?\s*",
            "，",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?:风险分|risk_score|distribution_risk_score)\s*(?:为|=)?\s*([-+]?[1-9]\d*(?:\.\d+)?)",
            r"分歧/派发风险分约\1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"[（(]\s*[a-z_]{3,}\s+(?:true|false|null|none|\"[^\"]*\"|'[^']*'|[-+]?\d+(?:\.\d+)?)\s*[)）]",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b[a-z_]{3,}\s+\"([^\"]+)\"",
            r"\1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b[a-z_]{3,}\s+'([^']+)'",
            r"\1",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b[a-z_]{3,}\s+(?:true|false|null|none)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+[，。；：]\s*", lambda m: m.group(0).strip(), cleaned)
        cleaned = re.sub(r"[，,；;]{2,}", "，", cleaned)
        cleaned = re.sub(r"，[。；]", "。", cleaned)
        cleaned = re.sub(r"[（(]\s*[)）]", "", cleaned)
        return cleaned.strip(" ，；。")

    def _build_focus_market_performance_view(self, item: Dict[str, Any]) -> str:
        close = item.get("close")
        pct_change = item.get("pct_change")
        high = item.get("high")
        low = item.get("low")
        open_price = item.get("open")
        amplitude = item.get("amplitude")
        recent_runup_5d = item.get("recent_runup_5d")
        turnover_rate = item.get("turnover_rate")
        parts: List[str] = []
        close_value = float(close) if close not in (None, "") else None
        pct_value = float(pct_change) if pct_change not in (None, "") else None
        open_value = float(open_price) if open_price not in (None, "") else None
        high_value = float(high) if high not in (None, "") else None
        low_value = float(low) if low not in (None, "") else None
        if close_value is not None and pct_value is not None:
            parts.append(f"今日收于{close_value:.2f}元，涨跌幅{pct_value:+.2f}%。")

        intraday_parts: List[str] = []
        if None not in (open_value, high_value, low_value, close_value):
            close_near_high = high_value > 0 and (high_value - close_value) / high_value <= 0.015
            close_near_low = close_value > 0 and (close_value - low_value) / close_value <= 0.015
            if high_value > open_value and close_value >= open_value and close_near_high:
                intraday_parts.append("日内上冲后收盘仍贴近高位，说明强势资金没有明显松动")
            elif high_value > open_value and close_value < high_value * 0.985:
                intraday_parts.append("盘中上冲后未能收在高位，说明追价后分歧开始显现")
            if low_value < open_value and close_value > open_value:
                intraday_parts.append("下探后被重新承接，说明低位仍有回补力量")
            elif close_near_low and pct_value is not None and pct_value < 0:
                intraday_parts.append("收盘更靠近日内低位，说明尾盘承接偏弱")
        if amplitude not in (None, ""):
            intraday_parts.append(f"振幅约{float(amplitude):.2f}%")
        if turnover_rate not in (None, ""):
            intraday_parts.append(f"换手率约{float(turnover_rate):.2f}%")
        if intraday_parts:
            parts.append("，".join(intraday_parts) + "。")

        stage_parts: List[str] = []
        runup_value = float(recent_runup_5d) if recent_runup_5d not in (None, "") else None
        if runup_value is not None:
            if runup_value >= 20:
                stage_parts.append(f"近5日累计涨幅约{runup_value:.1f}%，更像快速拉升后的高位博弈阶段")
            elif runup_value >= 10:
                stage_parts.append(f"近5日累计涨幅约{runup_value:.1f}%，属于启动后进入加速与分歧并存的阶段")
            elif runup_value >= 3:
                stage_parts.append(f"近5日累计涨幅约{runup_value:.1f}%，仍处于温和抬升后的验证阶段")
            elif runup_value <= -8:
                stage_parts.append(f"近5日累计涨幅约{runup_value:.1f}%，短线仍偏修复而不是主升")
        if pct_value is not None:
            if pct_value >= 7:
                stage_parts.append("当日强度已经接近情绪化拉升，后续更要看次日是否还有量价延续")
            elif pct_value >= 3:
                stage_parts.append("当日表现偏强，但更像强势推进中的确认日")
            elif pct_value <= -5:
                stage_parts.append("当日回撤较深，原有强势节奏正在接受更严苛考验")
            elif pct_value <= -2:
                stage_parts.append("短线强度有所回落，需要观察回踩后的承接是否还能恢复")
        if stage_parts:
            parts.append("，".join(stage_parts) + "。")
        return "".join(parts).strip()

    def _build_focus_catalyst_and_capital_view(self, item: Dict[str, Any]) -> str:
        catalyst_summary = self._clean_generated_text(item.get("catalyst_summary"))
        main_fund_flow_1d = item.get("main_fund_flow_1d")
        main_fund_flow_3d = item.get("main_fund_flow_3d")
        main_fund_flow_10d = item.get("main_fund_flow_10d")
        margin_balance_change_10d = item.get("margin_balance_change_10d")
        parts: List[str] = []
        if catalyst_summary:
            parts.append(catalyst_summary.rstrip("。") + "。")
        flow_parts: List[str] = []
        if main_fund_flow_1d not in (None, ""):
            flow_parts.append(f"当日主力资金{float(main_fund_flow_1d):+.0f}")
        if main_fund_flow_3d not in (None, ""):
            flow_parts.append(f"近3日主力资金{float(main_fund_flow_3d):+.0f}")
        if main_fund_flow_10d not in (None, ""):
            flow_parts.append(f"近10日主力资金{float(main_fund_flow_10d):+.0f}")
        if flow_parts:
            parts.append("资金面上，" + "，".join(flow_parts) + "。")
        if margin_balance_change_10d not in (None, ""):
            margin_value = float(margin_balance_change_10d)
            direction = "回补" if margin_value > 0 else "回落"
            parts.append(f"融资余额近10日呈{direction}{abs(margin_value):.0f}的变化，反映杠杆资金态度仍在调整。")
        return "".join(parts).strip()

    def _build_focus_fundamental_view(self, item: Dict[str, Any]) -> str:
        business_summary = str(item.get("business_summary") or "").strip()
        revenue_yoy = item.get("latest_revenue_yoy")
        profit_yoy = item.get("latest_profit_yoy")
        pe_ttm = item.get("pe_ttm")
        industry_pe_median = item.get("industry_pe_median")
        parts: List[str] = []
        if business_summary:
            parts.append(f"公司主营方面，{business_summary}")
        if revenue_yoy not in (None, "") or profit_yoy not in (None, ""):
            yoy_parts: List[str] = []
            if revenue_yoy not in (None, ""):
                yoy_parts.append(f"营收同比{float(revenue_yoy):+.1f}%")
            if profit_yoy not in (None, ""):
                yoy_parts.append(f"利润同比{float(profit_yoy):+.1f}%")
            if yoy_parts:
                parts.append("最近财务表现上，" + "，".join(yoy_parts) + "。")
        if pe_ttm not in (None, ""):
            if industry_pe_median not in (None, ""):
                parts.append(f"估值层面，当前PE约{float(pe_ttm):.1f}倍，对比行业中位数约{float(industry_pe_median):.1f}倍。")
            else:
                parts.append(f"估值层面，当前PE约{float(pe_ttm):.1f}倍。")
        if parts:
            return "".join(parts)
        score = float(item.get("fundamental_score") or 0.0)
        if score >= 80:
            quality = "公司质地与行业逻辑相对更扎实，短线表现不完全依赖情绪推动。"
        elif score >= 60:
            quality = "公司基本面没有明显硬伤，但当前更像基本面与交易性共同驱动。"
        else:
            quality = "可直接验证的基本面信息偏少，短线仍以交易节奏和催化兑现度为主。"
        return quality

    def _build_focus_market_context_view(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        industry_flow_bias = str(item.get("industry_flow_bias") or "").strip()
        industry_heat_score = item.get("industry_heat_score")
        industry_heat_value = float(industry_heat_score or 0.0) if industry_heat_score not in (None, "") else 0.0
        if industry_flow_bias and industry_flow_bias != "中性":
            direction_text = "顺风加分" if industry_heat_value >= 0 else "逆风压分"
            parts.append(f"板块资金风格偏向“{industry_flow_bias}”，对应当前排序中的{direction_text}。")
        elif industry_heat_score not in (None, "") and abs(industry_heat_value) >= 0.2:
            parts.append(f"板块层面对排序有约{industry_heat_value:+.1f}分的轻微修正，但整体仍未脱离中性环境。")
        if item.get("news_mentioned"):
            parts.append("个股同时具备新闻或主题催化，说明并非纯技术博弈。")
        if not parts:
            parts.append("当前市场环境信息有限，需更多结合板块联动和盘面承接判断原逻辑是否顺风。")
        return "".join(parts)

    def _build_focus_trading_context_view(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        technical_signal = self._localize_technical_signal(item.get("technical_signal"))
        if technical_signal and technical_signal != "待确认":
            parts.append(f"技术状态上，当前处于{technical_signal}。")
        turnover_spike_ratio = item.get("turnover_spike_ratio")
        if turnover_spike_ratio not in (None, ""):
            turnover_ratio_value = float(turnover_spike_ratio or 0.0)
            if turnover_ratio_value >= 1.05:
                parts.append(f"换手相对近5日均值放大{turnover_ratio_value:.2f}倍。")
        recent_runup_5d = item.get("recent_runup_5d")
        if recent_runup_5d not in (None, ""):
            parts.append(f"近5日累计涨幅约{float(recent_runup_5d or 0.0):.1f}%。")
        moneyflow_value = item.get("moneyflow_3d_value")
        if moneyflow_value not in (None, ""):
            parts.append(f"近3日资金净流值为{float(moneyflow_value or 0.0):.0f}。")
        contradiction_penalty = item.get("short_term_contradiction_penalty")
        if contradiction_penalty not in (None, "") and float(contradiction_penalty or 0.0) > 0:
            parts.append("虽然表面技术形态未完全走坏，但当前更像高位分歧下的观察阶段，不适合直接按短线主攻处理。")
        if not parts:
            parts.append("技术位置与节奏的结构化数据仍不充分，盘中重点看承接、回踩和量价是否匹配。")
        return "".join(parts)

    def _build_focus_analysis_fallback(self, item: Dict[str, Any]) -> str:
        action = item.get("action_plan") or {}
        market_view = self._clean_generated_text(item.get("market_performance_view") or self._build_focus_market_performance_view(item))
        catalyst_view = self._clean_generated_text(item.get("catalyst_and_capital_view") or self._build_focus_catalyst_and_capital_view(item))
        technical_view = self._clean_generated_text(item.get("trading_context_view") or self._build_focus_trading_context_view(item))
        fundamental_view = self._clean_generated_text(item.get("fundamental_view") or self._build_focus_fundamental_view(item))
        risk_note = self._clean_generated_text(self._build_focus_risk_note(item))
        overall_assessment = self._build_focus_overall_assessment(item)
        recommendation_score = item.get("recommendation_score")
        overall_score_value = item.get("overall_score")
        contradiction_penalty = float(item.get("short_term_contradiction_penalty") or 0.0)

        parts: List[str] = []
        if market_view:
            parts.append(market_view)

        ranking_reason = []
        if recommendation_score not in (None, "") and overall_score_value not in (None, ""):
            if abs(float(recommendation_score) - float(overall_score_value)) >= 0.5:
                ranking_reason.append(
                    f"它能进入今日重点名单，不只是因为综合质量不差，更因为短线交易排序分{float(recommendation_score):.1f}相对综合分{float(overall_score_value):.1f}更占优"
                )
            else:
                ranking_reason.append("它能进入今日重点名单，更多说明综合质量与短线交易性同时过关")
        if technical_view:
            ranking_reason.append(f"从交易结构看，{technical_view.rstrip('。')}")
        if ranking_reason:
            parts.append("，".join(ranking_reason) + "。")

        support_parts: List[str] = []
        if fundamental_view:
            support_parts.append(fundamental_view.rstrip('。'))
        if catalyst_view:
            support_parts.append(catalyst_view.rstrip('。'))
        if support_parts:
            parts.append("，".join(support_parts[:2]) + "。")

        if risk_note:
            if "暂未触发明显末端风险" in risk_note:
                parts.append(f"风险侧看，{risk_note}，不能把低风险分简单理解成没有波动风险。")
            else:
                parts.append(f"真正需要提防的不是普通波动，而是{risk_note}。")
        if contradiction_penalty > 0:
            parts.append("这类票当前更像有交易性、但执行难度也更高的机会，适合先看分歧是否收敛，再决定是否提高参与度。")
        parts.append(f"综合来看，{overall_assessment}。")
        parts.append(
            f"操作上先看{str(action.get('entry_zone') or '更清晰触发区').strip()}附近能否继续获得承接；若后续向上推进，再参考{str(action.get('take_profit') or '分批处理').strip()}逐步兑现，若出现{str(action.get('invalid_condition') or '技术结构失真').strip()}，就按{str(action.get('stop_loss') or '跌破关键支撑').strip()}对应的风控思路处理。"
        )
        return "".join(part.rstrip("。") + "。" for part in parts if part)

    def _looks_like_score_template_text(self, text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        compact_text = text.replace(" ", "")
        markers = [
            "综合评分",
            "主评分",
            "情绪面",
            "新闻面",
            "推荐分较昨日",
            "当前排序主要由推荐分",
            "命中策略数为",
            "综合来看，该股当前处于",
            "综合来看",
            "操作上先看",
            "技术面",
            "基本面",
            "资金面",
        ]
        if any(marker in text for marker in markers):
            return True
        score_marker_count = sum(1 for marker in ("评分", "分数", "技术面", "基本面", "资金面", "情绪面", "新闻面") if marker in text)
        if score_marker_count >= 3 and len(text) <= 120:
            return True
        if len(text) <= 80 and compact_text.count("：") >= 2 and score_marker_count >= 2:
            return True
        return False

    def _focus_analysis_needs_fallback(self, text: str, overall_assessment: str) -> bool:
        cleaned_text = self._clean_generated_text(text)
        if not cleaned_text:
            return True
        if self._contains_internal_report_text(cleaned_text):
            return True
        if self._looks_like_score_template_text(cleaned_text):
            return True
        if len(cleaned_text) < 180:
            return True
        cleaned_assessment = self._clean_generated_text(overall_assessment)
        if cleaned_assessment and cleaned_text == cleaned_assessment:
            return True
        if cleaned_assessment and len(cleaned_text) <= len(cleaned_assessment) + 12 and cleaned_assessment in cleaned_text:
            return True
        return False

    def _review_analysis_needs_fallback(self, text: str, reference_text: str) -> bool:
        cleaned_text = self._clean_generated_text(text)
        if not cleaned_text:
            return True
        if self._looks_like_score_template_text(cleaned_text):
            return True
        if len(cleaned_text) < 60:
            return True
        cleaned_reference = self._clean_generated_text(reference_text)
        if cleaned_reference and cleaned_text == cleaned_reference:
            return True
        return False

    def _build_review_strength_change(self, item: Dict[str, Any]) -> str:
        status = str(item.get("status") or item.get("review_status") or "观察").strip()
        pct_change = item.get("pct_change")
        turnover_rate = item.get("turnover_rate")
        amplitude = item.get("amplitude")
        recent_runup_5d = item.get("recent_runup_5d")
        volume_ratio = item.get("volume_ratio")
        score_change = item.get("score_change")

        parts = [f"当前状态为“{status}”"]
        if score_change not in (None, ""):
            delta = float(score_change)
            if delta >= 8:
                parts.append(f"推荐分较昨日明显抬升 {delta:+.1f}，强度在回升")
            elif delta >= 2:
                parts.append(f"推荐分较昨日小幅改善 {delta:+.1f}，但还需要继续确认")
            elif delta <= -8:
                parts.append(f"推荐分较昨日明显回落 {delta:+.1f}，强度已有明显降温")
            elif delta <= -2:
                parts.append(f"推荐分较昨日小幅回落 {delta:+.1f}，强度边际走弱")
            else:
                parts.append(f"推荐分较昨日基本持平 {delta:+.1f}，延续性一般")

        dynamics: List[str] = []
        if pct_change not in (None, ""):
            dynamics.append(f"当日涨跌幅{float(pct_change):+.2f}%")
        if turnover_rate not in (None, ""):
            dynamics.append(f"换手率约{float(turnover_rate):.2f}%")
        if volume_ratio not in (None, ""):
            ratio = float(volume_ratio)
            if ratio >= 1.3:
                dynamics.append(f"量能仍有放大，约为均量的{ratio:.2f}倍")
            elif ratio >= 0.9:
                dynamics.append(f"量能大体持平，约为均量的{ratio:.2f}倍")
            else:
                dynamics.append(f"量能明显回落，仅为均量的{ratio:.2f}倍")
        if amplitude not in (None, ""):
            dynamics.append(f"振幅约{float(amplitude):.2f}%")
        if recent_runup_5d not in (None, ""):
            dynamics.append(f"近5日累计涨跌约{float(recent_runup_5d):+.1f}%")
        if dynamics:
            parts.append("盘面上" + "，".join(dynamics))

        analysis = str(item.get("analysis") or item.get("absence_reason") or "").strip()
        if analysis and not self._looks_like_score_template_text(analysis):
            parts.append(analysis)
        elif not dynamics:
            parts.append("重点看价格强弱、换手变化和板块承接是否还支持原判断")
        return "。".join(part for part in parts if part)

    def _build_review_market_context_view(self, item: Dict[str, Any]) -> str:
        today_verdict = str(item.get("today_verdict") or item.get("status") or "").strip()
        absence_reason = str(item.get("absence_reason") or "").strip()
        if absence_reason:
            return f"市场与排序环境的反馈主要体现在：{absence_reason}"
        if today_verdict:
            return f"市场环境对原逻辑的反馈是：{today_verdict}"
        return "市场环境信息有限，重点看原先催化是否延续、板块资金是否还愿意承接。"

    def _build_review_analysis_fallback(self, item: Dict[str, Any]) -> str:
        action = item.get("action_plan") or {}
        yesterday_conclusion = str(item.get("yesterday_conclusion") or "昨日入选 Top3").strip()
        today_verdict = str(item.get("today_verdict") or item.get("status") or "待复评").strip()
        analysis = str(item.get("analysis") or item.get("absence_reason") or "今日表现需要结合缺席原因复盘。").strip()
        if self._looks_like_score_template_text(analysis):
            analysis = "今日盘面给出的反馈更多是强度回落还是延续承接，需要结合价格、量能与板块资金继续确认。"
        miss_reason_candidates = [str(candidate).strip() for candidate in (item.get("miss_reason_candidates") or []) if str(candidate).strip()]
        missing_factor_candidates = [str(candidate).strip() for candidate in (item.get("missing_factor_candidates") or []) if str(candidate).strip()]
        next_watch = "、".join((miss_reason_candidates + missing_factor_candidates)[:3]) or str(action.get("invalid_condition") or "原有技术结构是否失真").strip()

        pct_change = item.get("pct_change")
        turnover_rate = item.get("turnover_rate")
        amplitude = item.get("amplitude")
        volume_ratio = item.get("volume_ratio")
        recent_runup_5d = item.get("recent_runup_5d")
        market_parts: List[str] = []
        if pct_change not in (None, ""):
            market_parts.append(f"当日涨跌幅{float(pct_change):+.2f}%")
        if turnover_rate not in (None, ""):
            market_parts.append(f"换手率约{float(turnover_rate):.2f}%")
        if volume_ratio not in (None, ""):
            ratio = float(volume_ratio)
            if ratio >= 1:
                market_parts.append(f"量能放大至均量的{ratio:.2f}倍")
            else:
                market_parts.append(f"量能回落至均量的{ratio:.2f}倍")
        if amplitude not in (None, ""):
            market_parts.append(f"振幅约{float(amplitude):.2f}%")
        if recent_runup_5d not in (None, ""):
            market_parts.append(f"近5日累计涨跌约{float(recent_runup_5d):+.1f}%")
        market_text = "，".join(market_parts) if market_parts else "价格与量能变化仍需结合盘中承接继续确认"

        if yesterday_conclusion == "昨日结论缺失":
            yesterday_text = "昨日缺少清晰结论"
        else:
            yesterday_text = f"昨日逻辑是“{yesterday_conclusion}”"

        return "".join(
            [
                f"{yesterday_text}，今天的复盘结论是“{today_verdict}”。",
                f"从盘面反馈看，{market_text}。",
                f"结合当日表现，{analysis}",
                f"强弱变化方面，{self._build_review_strength_change(item)}。",
                f"当前处理上更适合以{str(action.get('action_bias') or '观察').strip()}应对，关注{str(action.get('entry_zone') or '更清晰触发位').strip()}，止盈参考{str(action.get('take_profit') or '分批处理').strip()}，止损参考{str(action.get('stop_loss') or '跌破关键支撑离场').strip()}。",
                f"下一步重点观察{next_watch}，若出现{str(action.get('invalid_condition') or '原先驱动失效').strip()}，就要把昨日逻辑按失效处理。",
            ]
        )

    def _build_fallback_report_blocks(
        self,
        stock_pool: List[str],
        market_data: Dict[str, Any],
        screening_context: Optional[Dict[str, Any]] = None,
        news_clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        screening_context = screening_context or {}
        news_clusters = news_clusters or []
        focus_stocks = [self._fallback_focus_stock_item(item) for item in (screening_context.get("today_top3") or [])]
        yesterday_reviews = [self._fallback_review_item(item) for item in (screening_context.get("yesterday_top3_review") or [])]
        comparison_candidates = screening_context.get("comparison_candidates") or []
        today_top3 = screening_context.get("today_top3") or []
        comparison = {
            "basic_rank": self._rank_codes(comparison_candidates, "fundamental_score"),
            "technical_rank": self._rank_codes(comparison_candidates, "technical_score"),
            "risk_rank": self._rank_codes(comparison_candidates, "risk_score", reverse=False),
            "trading_rank": self._rank_codes(comparison_candidates, "recommendation_score"),
            "best_short_term": self._pick_code(today_top3) or (stock_pool[0] if stock_pool else ""),
            "most_robust": self._pick_code(comparison_candidates, key="overall_score"),
            "highest_risk": self._pick_code(comparison_candidates, key="risk_score", reverse=True),
        }
        comparison = self._attach_comparison_names(comparison, comparison_candidates + today_top3)
        overall_action = {
            "headline": "结构化报告暂使用本地回退结果",
            "market_view": str(market_data.get("trend") or "市场概览数据不可用"),
            "risk_summary": str(market_data.get("sentiment") or "市场风险概览数据不可用"),
            "action_items": ["优先跟踪今日 Top3，昨日对象重点看是否延续。"],
        }
        theme_focus_items = self._build_theme_focus_items(news_clusters, overall_action)
        overall_action["theme_focuses"] = theme_focus_items
        return {
            "news_clusters": news_clusters,
            "theme_view": self._split_theme_clusters(theme_focus_items, market_data, overall_action, news_clusters),
            "focus_stocks": focus_stocks,
            "yesterday_reviews": yesterday_reviews,
            "comparison": comparison,
            "overall_action": overall_action,
        }

    def _format_focus_stock_section(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "当前没有满足条件的重点个股。"
        lines: List[str] = []
        for item in items:
            action = item.get("action_plan") or {}
            lines.extend(
                [
                    f"【{item.get('ts_code', '')} {item.get('name', '')}】",
                    f"推荐分/综合分/置信度：{item.get('recommendation_score', '--')} / {item.get('overall_score', '--')} / {item.get('display_confidence', item.get('overall_confidence', '--'))}",
                    f"核心亮点：{'；'.join(item.get('core_highlights') or []) or '暂无'}",
                    f"资金/换手/末端风险：{self._build_focus_risk_note(item)}",
                    f"风险提示：{'；'.join(item.get('risk_warnings') or []) or '暂无'}",
                    f"综合评价：{item.get('overall_assessment', '暂无')}",
                    f"短线建议：{action.get('action_bias', '观察')}，入场区间 {action.get('entry_zone', '待观察')}，止盈 {action.get('take_profit', '待观察')}，止损 {action.get('stop_loss', '待观察')}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _format_review_section(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "暂无昨日 Top3 今日复评数据。"
        lines: List[str] = []
        for item in items:
            lines.extend(
                [
                    f"【{item.get('ts_code', '')} {item.get('name', '')}】{item.get('status', item.get('today_verdict', '待复评'))}",
                    f"昨日结论：{item.get('yesterday_conclusion', '暂无')}",
                    f"今日判断：{item.get('today_verdict', item.get('status', '暂无'))}",
                    f"复盘说明：{item.get('analysis', '暂无')}",
                    *([f"失误候选：{'、'.join(item.get('miss_reason_candidates') or [])}"] if (item.get('miss_reason_candidates') or []) else []),
                    *([f"缺失因子：{'、'.join(item.get('missing_factor_candidates') or [])}"] if (item.get('missing_factor_candidates') or []) else []),
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _format_comparison_section(self, comparison: Dict[str, Any]) -> str:
        if not comparison:
            return "暂无横向比较结果。"
        return "\n".join(
            [
                f"基本面排序：{' > '.join(comparison.get('basic_rank') or []) or '暂无'}",
                f"技术面排序：{' > '.join(comparison.get('technical_rank') or []) or '暂无'}",
                f"风险排序：{' > '.join(comparison.get('risk_rank') or []) or '暂无'}",
                f"短线交易性排序：{' > '.join(comparison.get('trading_rank') or []) or '暂无'}",
                f"最适合短线：{comparison.get('best_short_term') or '暂无'}",
                f"最稳健：{comparison.get('most_robust') or '暂无'}",
                f"风险最高：{comparison.get('highest_risk') or '暂无'}",
                f"横向综合：{comparison.get('cross_stock_synthesis_view') or '暂无'}",
            ]
        )

    def _format_overall_action_section(self, overall: Dict[str, Any]) -> str:
        if not overall:
            return "暂无整体建议。"
        lines = [
            str(overall.get("headline") or "今日整体建议"),
            f"市场判断：{overall.get('market_view') or '暂无'}",
            f"风险总览：{overall.get('risk_summary') or '暂无'}",
        ]
        action_items = overall.get("action_items") or []
        if action_items:
            lines.append("操作建议：")
            lines.extend([f"- {item}" for item in action_items if str(item).strip()])
        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except Exception:
            pass
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(response[start : end + 1])
        except Exception:
            pass
        return {}

    def _serialize_news_cluster(self, cluster: NewsCluster) -> Dict[str, Any]:
        news_briefs: List[str] = []
        for item in list(getattr(cluster, "news_items", []) or [])[:3]:
            title = str(getattr(item, "title", "") or "").strip()
            content = str(getattr(item, "content", "") or "").strip()
            brief = title or content
            if title and content and content != title:
                brief = f"{title}：{content}"
            if brief:
                news_briefs.append(brief)
        return {
            "theme": getattr(cluster, "theme", ""),
            "importance": getattr(cluster, "importance", 0.0),
            "summary": getattr(cluster, "summary", ""),
            "key_stocks": list(getattr(cluster, "key_stocks", []) or []),
            "news_briefs": news_briefs,
        }

    def _build_theme_focus_items(
        self,
        news_clusters: List[Dict[str, Any]],
        overall_action: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in overall_action.get("theme_focuses") or []:
            if not isinstance(item, dict):
                continue
            theme = str(item.get("theme") or "").strip()
            if not theme:
                continue
            summary = str(item.get("summary") or "").strip()
            continuity_view = str(item.get("continuity_view") or "").strip()
            tier = self._normalize_theme_tier(item.get("tier"), theme, summary, continuity_view)
            related_stocks = [str(stock).strip() for stock in (item.get("related_stocks") or []) if str(stock).strip()]
            items.append(
                {
                    "theme": theme,
                    "tier": tier,
                    "summary": summary or self._build_theme_summary_from_cluster({"theme": theme}),
                    "continuity_view": continuity_view or self._build_theme_continuity_view({"theme": theme, "summary": summary}),
                    "related_stocks": related_stocks,
                    "importance": self._safe_float_value(item.get("importance"), 0.0),
                }
            )

        existing_themes = {item.get("theme") for item in items}
        for cluster in news_clusters:
            theme = str(cluster.get("theme") or "").strip()
            if not theme or theme in existing_themes:
                continue
            summary = self._build_theme_summary_from_cluster(cluster)
            items.append(
                {
                    "theme": theme,
                    "tier": self._normalize_theme_tier(None, theme, summary, str(cluster.get("summary") or "")),
                    "summary": summary,
                    "continuity_view": self._build_theme_continuity_view(cluster),
                    "related_stocks": [str(stock).strip() for stock in (cluster.get("key_stocks") or []) if str(stock).strip()],
                    "importance": self._safe_float_value(cluster.get("importance"), 0.0),
                }
            )

        items.sort(
            key=lambda item: (
                self._theme_tier_priority(item.get("tier")),
                -self._safe_float_value(item.get("importance"), 0.0),
            )
        )
        return items[:6]

    def _split_theme_clusters(
        self,
        theme_focus_items: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        overall_action: Dict[str, Any],
        news_clusters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        market_overview_cluster = self._build_theme_market_overview(market_data, overall_action, news_clusters)
        core_theme_clusters = [item for item in theme_focus_items if item.get("tier") == "主线"][:3]
        risk_theme_clusters = [item for item in theme_focus_items if item.get("tier") == "风险"][:1]

        if not core_theme_clusters:
            core_theme_clusters = theme_focus_items[: min(3, len(theme_focus_items))]
        if not risk_theme_clusters:
            risk_theme_clusters = [self._build_risk_theme_cluster(overall_action, market_data)]

        excluded_themes = {
            str(item.get("theme") or "").strip()
            for item in (core_theme_clusters + risk_theme_clusters)
            if str(item.get("theme") or "").strip()
        }
        watchlist_theme_clusters = [
            item
            for item in theme_focus_items
            if item.get("tier") in {"次主线", "观察"}
            and str(item.get("theme") or "").strip()
            and str(item.get("theme") or "").strip() not in excluded_themes
        ][:2]

        return {
            "market_overview_cluster": market_overview_cluster,
            "core_theme_clusters": core_theme_clusters,
            "watchlist_theme_clusters": watchlist_theme_clusters,
            "risk_theme_cluster": risk_theme_clusters[0] if risk_theme_clusters else self._build_risk_theme_cluster(overall_action, market_data),
        }

    def _build_theme_market_overview(
        self,
        market_data: Dict[str, Any],
        overall_action: Dict[str, Any],
        news_clusters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        top_themes = [str(item.get("theme") or "").strip() for item in news_clusters[:3] if str(item.get("theme") or "").strip()]
        related_stocks: List[str] = []
        for cluster in news_clusters[:2]:
            for stock in cluster.get("key_stocks") or []:
                stock_text = str(stock).strip()
                if stock_text and stock_text not in related_stocks:
                    related_stocks.append(stock_text)
        return {
            "theme": "市场总线",
            "tier": "总览",
            "summary": str(overall_action.get("market_view") or market_data.get("trend") or "市场概览数据不可用"),
            "continuity_view": "、".join(top_themes) if top_themes else str(market_data.get("sentiment") or overall_action.get("risk_summary") or "市场风险概览数据不可用"),
            "related_stocks": related_stocks[:5],
        }

    def _build_risk_theme_cluster(self, overall_action: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "theme": "风险扰动",
            "tier": "风险",
            "summary": str(overall_action.get("risk_summary") or market_data.get("sentiment") or "市场风险概览数据不可用"),
            "continuity_view": "市场风险数据不可用时，仅基于个股与主题输入做保守跟踪。",
            "related_stocks": [],
        }

    def _build_theme_summary_from_cluster(self, cluster: Dict[str, Any]) -> str:
        summary = str(cluster.get("summary") or "").strip()
        if summary:
            return summary
        theme = str(cluster.get("theme") or "未命名主题").strip()
        return f"{theme} 当前缺少完整摘要，建议结合盘中强弱和持续性继续跟踪。"

    def _build_theme_continuity_view(self, cluster: Dict[str, Any]) -> str:
        summary = str(cluster.get("summary") or "").strip()
        importance = self._safe_float_value(cluster.get("importance"), 0.0)
        if any(keyword in summary for keyword in ("持续", "强化", "扩散", "走强", "发酵")):
            return "主题仍有持续发酵迹象，但需要观察是否继续扩散到更多个股。"
        if any(keyword in summary for keyword in ("分化", "回落", "冲高回落", "承压", "退潮")):
            return "主题短线有分化或回落信号，持续性需要依赖新的资金承接确认。"
        if importance >= 0.75:
            return "主题重要度较高，仍处于市场重点观察区，需确认次日是否延续。"
        if importance >= 0.4:
            return "主题处于轮动观察阶段，若无新增催化更适合作为跟踪线而非追涨线。"
        return "主题暂以观察为主，只有在资金与强度同步改善时才具备进一步跟踪价值。"

    @staticmethod
    def _normalize_theme_tier(tier: Any, theme: str, summary: str, continuity_view: str) -> str:
        tier_text = str(tier or "").strip()
        if tier_text in {"主线", "次主线", "观察", "风险"}:
            return tier_text
        text = f"{theme} {summary} {continuity_view}"
        if any(keyword in text for keyword in ("风险", "扰动", "分歧", "回撤", "退潮", "承压", "降温")):
            return "风险"
        if any(keyword in text for keyword in ("主线", "核心", "龙头", "共振", "扩散", "强化")):
            return "主线"
        if any(keyword in text for keyword in ("轮动", "次主线", "活跃", "发酵")):
            return "次主线"
        return "观察"

    @staticmethod
    def _theme_tier_priority(tier: Any) -> int:
        mapping = {"主线": 0, "风险": 1, "次主线": 2, "观察": 3}
        return mapping.get(str(tier or ""), 4)

    @staticmethod
    def _safe_float_value(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _rank_codes(items: List[Dict[str, Any]], key: str, reverse: bool = True) -> List[str]:
        ranked = sorted(
            [item for item in items if item.get("ts_code")],
            key=lambda item: float(item.get(key) or 0.0),
            reverse=reverse,
        )
        return [item.get("ts_code") for item in ranked]

    @staticmethod
    def _pick_code(items: List[Dict[str, Any]], key: str = "recommendation_score", reverse: bool = True) -> str:
        ranked = IntelligentReportGenerator._rank_codes(items, key, reverse=reverse)
        return ranked[0] if ranked else ""

    @staticmethod
    def _attach_comparison_names(comparison: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
        name_map = {
            item.get("ts_code"): item.get("name")
            for item in items
            if item.get("ts_code") and item.get("name")
        }

        def format_code(code: Any) -> str:
            code_text = str(code or "").strip()
            if not code_text:
                return ""
            name = str(name_map.get(code_text) or "").strip()
            return f"{code_text}（{name}）" if name else code_text

        result = dict(comparison)
        for key in ("basic_rank", "technical_rank", "risk_rank", "trading_rank"):
            result[key] = [formatted for formatted in (format_code(code) for code in (comparison.get(key) or [])) if formatted]
        for key in ("best_short_term", "most_robust", "highest_risk"):
            result[key] = format_code(comparison.get(key)) or comparison.get(key) or ""
        return result

    def format_report_markdown(self, report: IntelligentReport) -> str:
        lines = [
            f"# {report.title}",
            f"*生成时间：{report.generate_time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## 摘要",
            report.summary,
            "",
            "## 关键要点",
        ]
        for point in report.key_points:
            lines.append(f"- {point}")
        lines.append("")
        for section in sorted(report.sections, key=lambda s: s.priority):
            lines.extend([f"## {section.title}", section.content, ""])
        return "\n".join(lines)

    def format_report_wechat(self, report: IntelligentReport) -> str:
        lines = [
            f"【{report.title}】",
            f"⏰ {report.generate_time.strftime('%H:%M')}",
            "",
            "📋 今日摘要",
            report.summary,
        ]
        for i, point in enumerate(report.key_points, 1):
            lines.append(f"{i}. {point}")
        return "\n".join(lines)
