"""Intelligent report generation system."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from octts.config import Settings
from octts.clients.llm_client import LLMClient
from octts.prompts.report_prompt import build_intelligent_screening_report_prompt
from octts.services.news_aggregator import NewsCluster
from octts.services.multi_dimensional_analyzer import MultiDimensionalAnalyzer

logger = logging.getLogger(__name__)


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
        recommendations = list(report_blocks.get("overall_action", {}).get("action_items") or [])

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
            title=f"AI智能早报 - {datetime.now().strftime('%Y年%m月%d日')}",
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
        serialized_news_clusters = [self._serialize_news_cluster(item) for item in news_clusters[:5]]
        if not screening_context:
            return self._build_fallback_report_blocks(stock_pool, market_data, screening_context, serialized_news_clusters)

        system_prompt, user_prompt = build_intelligent_screening_report_prompt(
            market_data=market_data,
            news_clusters=serialized_news_clusters,
            screening_context=screening_context,
        )
        try:
            response = await self.llm_client.complete(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=self.settings.llm_max_tokens,
            )
            payload = self._parse_json_response(response)
            if payload:
                return self._merge_report_blocks_with_context(payload, screening_context, market_data, serialized_news_clusters)
        except Exception:
            logger.exception("Failed to generate structured intelligent screening report")
        return self._build_fallback_report_blocks(stock_pool, market_data, screening_context, serialized_news_clusters)

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
                title="重点股票横向比较",
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
        focus_map = {item.get("ts_code"): item for item in screening_context.get("comparison_candidates") or [] if item.get("ts_code")}
        review_map = {item.get("ts_code"): item for item in screening_context.get("yesterday_top3_review") or [] if item.get("ts_code")}

        merged_focus = []
        for item in payload.get("focus_stocks") or []:
            merged_focus.append(self._merge_focus_stock_item(item, focus_map.get(item.get("ts_code"))))

        existing_focus_codes = {item.get("ts_code") for item in merged_focus}
        for item in screening_context.get("today_top3") or []:
            code = item.get("ts_code")
            if code and code not in existing_focus_codes:
                merged_focus.append(self._fallback_focus_stock_item(item))

        merged_reviews = []
        for item in payload.get("yesterday_reviews") or []:
            merged_reviews.append(self._merge_review_item(item, review_map.get(item.get("ts_code"))))

        existing_review_codes = {item.get("ts_code") for item in merged_reviews}
        for item in screening_context.get("yesterday_top3_review") or []:
            code = item.get("ts_code")
            if code and code not in existing_review_codes:
                merged_reviews.append(self._fallback_review_item(item))

        comparison = dict(payload.get("comparison") or {})
        comparison_candidates = screening_context.get("comparison_candidates") or []
        today_top3 = screening_context.get("today_top3") or []
        comparison.setdefault("basic_rank", self._rank_codes(comparison_candidates, "fundamental_score"))
        comparison.setdefault("technical_rank", self._rank_codes(comparison_candidates, "technical_score"))
        comparison.setdefault("risk_rank", self._rank_codes(comparison_candidates, "risk_score", reverse=False))
        comparison.setdefault("trading_rank", self._rank_codes(comparison_candidates, "recommendation_score"))
        comparison.setdefault("best_short_term", self._pick_code(today_top3))
        comparison.setdefault("most_robust", self._pick_code(comparison_candidates, key="overall_score"))
        comparison.setdefault("highest_risk", self._pick_code(comparison_candidates, key="risk_score", reverse=True))
        comparison = self._attach_comparison_names(comparison, comparison_candidates + today_top3)

        overall_action = dict(payload.get("overall_action") or {})
        overall_action.setdefault("market_view", str(market_data.get("trend") or "震荡市，保持节奏"))
        overall_action.setdefault("risk_summary", str(market_data.get("sentiment") or "注意波动与分化风险"))
        overall_action.setdefault("action_items", ["优先围绕今日 Top3 与昨日复评对象跟踪。"])
        overall_action.setdefault("headline", "围绕系统排序做解释与执行建议")

        return {
            "news_clusters": news_clusters,
            "focus_stocks": merged_focus,
            "yesterday_reviews": merged_reviews,
            "comparison": comparison,
            "overall_action": overall_action,
        }

    def _merge_focus_stock_item(self, item: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not context:
            return item
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
        merged.setdefault("miss_reason_candidates", context.get("miss_reason_candidates") or [])
        merged.setdefault("missing_factor_candidates", context.get("missing_factor_candidates") or [])
        return merged

    def _fallback_focus_stock_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **item,
            "core_highlights": [str(item.get("recommendation_text") or "进入重点跟踪名单")],
            "risk_warnings": [str(item.get("technical_signal") or "需结合盘中确认")],
            "overall_assessment": str(item.get("summary") or item.get("recommendation_text") or "建议结合盘中信号跟踪。"),
            "action_plan": dict(item.get("action_plan") or {}),
        }

    def _fallback_review_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        action_plan = dict(item.get("action_plan") or {})
        return {
            **item,
            "status": str(item.get("status") or item.get("review_status") or "待复评"),
            "review_status": str(item.get("review_status") or item.get("status") or "待复评"),
            "yesterday_conclusion": str(item.get("yesterday_conclusion") or "昨日入选 Top3"),
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
        return {
            "news_clusters": news_clusters,
            "focus_stocks": focus_stocks,
            "yesterday_reviews": yesterday_reviews,
            "comparison": comparison,
            "overall_action": {
                "headline": "结构化报告暂使用本地回退结果",
                "market_view": str(market_data.get("trend") or "震荡市，优先控制节奏"),
                "risk_summary": str(market_data.get("sentiment") or "关注分化与承接风险"),
                "action_items": ["优先跟踪今日 Top3，昨日对象重点看是否延续。"],
            },
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
                    f"失误候选：{'、'.join(item.get('miss_reason_candidates') or []) or '暂无'}",
                    f"缺失因子：{'、'.join(item.get('missing_factor_candidates') or []) or '暂无'}",
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
        return {
            "theme": getattr(cluster, "theme", ""),
            "importance": getattr(cluster, "importance", 0.0),
            "summary": getattr(cluster, "summary", ""),
            "key_stocks": list(getattr(cluster, "key_stocks", []) or []),
        }

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
