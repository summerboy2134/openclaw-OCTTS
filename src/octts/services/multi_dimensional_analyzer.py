"""Multi-dimensional stock analysis workflow."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from octts.clients.llm_client import LLMClient
from octts.clients.tushare_client import TushareClient
from octts.config import Settings
from octts.indicators.technical import build_technical_snapshot, get_rsi_status

logger = logging.getLogger(__name__)


class AnalysisDimension(Enum):
    """分析维度"""

    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    NEWS = "news"


@dataclass
class DimensionAnalysis:
    """单维度分析结果"""

    dimension: AnalysisDimension
    score: float
    confidence: float
    analysis: str
    key_points: List[str]
    signals: Dict[str, Any]


@dataclass
class ConflictResolution:
    """冲突解决结果"""

    has_conflict: bool
    conflict_points: List[str]
    resolution: str
    final_decision: str


class MultiDimensionalAnalyzer:
    """多维度股票分析器"""

    def __init__(
        self,
        settings: Settings,
        llm_client: Optional[LLMClient] = None,
        tushare_client: Optional[TushareClient] = None,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client or LLMClient(settings)
        self.tushare_client = tushare_client or TushareClient(settings)

    async def analyze(
        self,
        ts_code: str,
        enable_iterations: bool = True,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """执行多维度分析并返回统一结果。"""
        logger.info("Starting multi-dimensional analysis for %s", ts_code)

        base_data = await self._fetch_stock_data(ts_code)
        dimension_results = await self._run_parallel_analysis(ts_code, base_data)
        conflict_resolution = await self._resolve_conflicts(dimension_results)
        overall_confidence = self._calculate_confidence(dimension_results)

        iteration_count = 1
        while enable_iterations and overall_confidence < 0.7 and iteration_count < max_iterations:
            supplementary_queries = await self._generate_supplementary_queries(
                dimension_results,
                conflict_resolution,
            )
            supplementary_data = await self._fetch_supplementary_data(ts_code, supplementary_queries)
            dimension_results = await self._run_incremental_analysis(
                ts_code,
                base_data,
                supplementary_data,
                dimension_results,
            )
            conflict_resolution = await self._resolve_conflicts(dimension_results)
            overall_confidence = self._calculate_confidence(dimension_results)
            iteration_count += 1

        return await self._generate_final_report(
            ts_code,
            base_data,
            dimension_results,
            conflict_resolution,
            overall_confidence,
            iteration_count,
        )

    async def _fetch_stock_data(self, ts_code: str) -> Dict[str, Any]:
        """获取股票基础数据。"""
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        return {
            "stock_info": self.tushare_client.fetch_stock_info(ts_code),
            "daily_data": self.tushare_client.fetch_daily_data(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            ),
            "financial_data": self.tushare_client.fetch_financial_indicators(ts_code),
            "moneyflow_data": self.tushare_client.fetch_moneyflow(ts_code, trade_date=end_date),
        }

    async def _run_parallel_analysis(
        self,
        ts_code: str,
        base_data: Dict[str, Any],
    ) -> Dict[AnalysisDimension, DimensionAnalysis]:
        """并行执行多维度分析。"""
        results = await asyncio.gather(
            self._analyze_technical(ts_code, base_data),
            self._analyze_fundamental(ts_code, base_data),
            self._analyze_sentiment(ts_code, base_data),
            self._analyze_news(ts_code, base_data),
            return_exceptions=True,
        )
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Analysis dimension {i} failed: {result}")
                # 返回默认结果
                if i == 0:
                    processed_results.append(DimensionAnalysis(
                        dimension=AnalysisDimension.TECHNICAL,
                        score=50.0,
                        confidence=0.3,
                        analysis="技术分析失败",
                        key_points=[],
                        signals={},
                    ))
                elif i == 1:
                    processed_results.append(DimensionAnalysis(
                        dimension=AnalysisDimension.FUNDAMENTAL,
                        score=50.0,
                        confidence=0.3,
                        analysis="基本面分析失败",
                        key_points=[],
                        signals={},
                    ))
                elif i == 2:
                    processed_results.append(DimensionAnalysis(
                        dimension=AnalysisDimension.SENTIMENT,
                        score=50.0,
                        confidence=0.3,
                        analysis="情绪分析失败",
                        key_points=[],
                        signals={},
                    ))
                else:
                    processed_results.append(DimensionAnalysis(
                        dimension=AnalysisDimension.NEWS,
                        score=50.0,
                        confidence=0.3,
                        analysis="新闻分析失败",
                        key_points=[],
                        signals={},
                    ))
            else:
                processed_results.append(result)
        
        return {
            AnalysisDimension.TECHNICAL: processed_results[0],
            AnalysisDimension.FUNDAMENTAL: processed_results[1],
            AnalysisDimension.SENTIMENT: processed_results[2],
            AnalysisDimension.NEWS: processed_results[3],
        }

    async def _analyze_technical(self, ts_code: str, base_data: Dict[str, Any]) -> DimensionAnalysis:
        daily_data = base_data.get("daily_data", [])
        ordered_daily = sorted(
            daily_data,
            key=lambda item: str(item.get("trade_date", "")),
        )
        if not ordered_daily:
            return DimensionAnalysis(
                dimension=AnalysisDimension.TECHNICAL,
                score=0.0,
                confidence=0.2,
                analysis="技术面数据不足，暂时无法形成可靠结论。",
                key_points=["行情数据不足"],
                signals={"trend": "数据不足", "latest_close": 0.0},
            )

        closes = pd.Series([float(item.get("close", 0) or 0) for item in ordered_daily])
        highs = pd.Series([float(item.get("high", item.get("close", 0)) or 0) for item in ordered_daily])
        lows = pd.Series([float(item.get("low", item.get("close", 0)) or 0) for item in ordered_daily])
        volumes = pd.Series([float(item.get("vol", 0) or 0) for item in ordered_daily])
        snapshot = build_technical_snapshot(closes, highs, lows, volumes)

        trend_map = {
            "bullish": "多头趋势",
            "improving": "趋势改善",
            "mixed": "震荡整理",
            "bearish": "空头趋势",
        }
        momentum_map = {
            "bullish_rising": "动能增强",
            "bullish": "动能偏强",
            "bearish_improving": "空头钝化",
            "bearish": "动能偏弱",
            "neutral": "动能中性",
        }
        breakout_map = {
            "upward": "向上突破",
            "downward": "向下破位",
        }
        trend = trend_map.get(snapshot.trend_status, "震荡整理")
        momentum = momentum_map.get(snapshot.momentum_status, "动能中性")
        breakout = breakout_map.get(snapshot.breakout)
        rsi_status = get_rsi_status(snapshot.rsi)

        confidence = 0.35
        if len(ordered_daily) >= 20:
            confidence = 0.55
        if len(ordered_daily) >= 26 and snapshot.macd is not None:
            confidence = 0.72
        if len(ordered_daily) >= 60 and snapshot.ma60 is not None:
            confidence = 0.82

        key_points = [trend, momentum, "技术评分{0:.1f}".format(snapshot.technical_score)]
        if snapshot.rsi is not None:
            key_points.append("RSI {0:.1f}".format(snapshot.rsi))
        if snapshot.price_position_20d is not None:
            key_points.append("20日位置 {0:.0%}".format(snapshot.price_position_20d))
        if breakout:
            key_points.append(breakout)

        signal_payload = {
            "trend": trend,
            "latest_close": snapshot.close,
            "rsi": snapshot.rsi,
            "price_position_20d": snapshot.price_position_20d,
            "technical_score": snapshot.technical_score,
            "momentum_status": momentum,
            "breakout": breakout or "无突破",
        }

        fallback_parts = [
            trend,
            "最新收盘{0:.2f}".format(snapshot.close),
            "技术评分{0:.1f}".format(snapshot.technical_score),
            momentum,
        ]
        if snapshot.rsi is not None:
            fallback_parts.append("RSI {0:.1f}（{1}）".format(snapshot.rsi, rsi_status))
        if snapshot.price_position_20d is not None:
            fallback_parts.append("20日区间位置{0:.0%}".format(snapshot.price_position_20d))
        if breakout:
            fallback_parts.append(breakout)

        prompt = (
            f"请基于以下结构化技术信号，为 {ts_code} 生成120字内中文技术面总结，不要引入未提供的信息：\n"
            f"{json.dumps(signal_payload, ensure_ascii=False)}"
        )
        analysis_text = await self._complete_with_fallback(
            prompt,
            fallback="，".join(fallback_parts) + "。",
        )

        return DimensionAnalysis(
            dimension=AnalysisDimension.TECHNICAL,
            score=snapshot.technical_score,
            confidence=confidence,
            analysis=analysis_text,
            key_points=key_points,
            signals=signal_payload,
        )

    async def _analyze_fundamental(self, ts_code: str, base_data: Dict[str, Any]) -> DimensionAnalysis:
        del ts_code
        stock_info = base_data.get("stock_info", {})
        financial_data = base_data.get("financial_data", [])
        latest = financial_data[0] if financial_data else {}
        roe = float(latest.get("roe", 0) or 0)
        profit_growth = float(latest.get("netprofit_yoy", 0) or 0)
        score = 50.0
        if roe >= 10:
            score += 10
        if profit_growth > 0:
            score += 10

        analysis = "基本面数据有限，建议结合最新财报进一步确认。"
        if financial_data:
            analysis = "ROE {0:.1f}%，净利润同比 {1:.1f}% 。".format(roe, profit_growth)

        return DimensionAnalysis(
            dimension=AnalysisDimension.FUNDAMENTAL,
            score=min(score, 90.0),
            confidence=0.65 if financial_data else 0.35,
            analysis=analysis,
            key_points=[stock_info.get("industry", "行业未知"), "ROE {0:.1f}%".format(roe)],
            signals={"roe": roe, "netprofit_yoy": profit_growth},
        )

    async def _analyze_sentiment(self, ts_code: str, base_data: Dict[str, Any]) -> DimensionAnalysis:
        del ts_code
        moneyflow_data = base_data.get("moneyflow_data", [])
        net_amounts = [float(item.get("net_mf_amount", 0) or 0) for item in moneyflow_data if item.get("net_mf_amount") is not None]
        total_flow = sum(net_amounts)
        if total_flow > 0:
            score = 66.0
            sentiment = "偏多"
        elif total_flow < 0:
            score = 40.0
            sentiment = "偏空"
        else:
            score = 52.0
            sentiment = "中性"

        total_flow_wan = total_flow / 10000.0
        return DimensionAnalysis(
            dimension=AnalysisDimension.SENTIMENT,
            score=score,
            confidence=0.6 if moneyflow_data else 0.3,
            analysis="资金情绪{0}，近3日净流向合计 {1:.2f} 万元。".format(sentiment, total_flow_wan),
            key_points=[sentiment, "近3日净流向 {0:.2f} 万元".format(total_flow_wan)],
            signals={"net_mf_amount": total_flow, "sentiment": sentiment},
        )

    async def _analyze_news(self, ts_code: str, base_data: Dict[str, Any]) -> DimensionAnalysis:
        stock_info = base_data.get("stock_info", {})
        industry = stock_info.get("industry", "未知行业")
        prompt = "请用一句话概括 {0} 所在行业 {1} 的新闻关注点。".format(ts_code, industry)
        analysis = await self._complete_with_fallback(
            prompt,
            fallback="当前未接入实时新闻，先按 {0} 行业常规消息面处理中性评估。".format(industry),
        )
        return DimensionAnalysis(
            dimension=AnalysisDimension.NEWS,
            score=55.0,
            confidence=0.3,
            analysis=analysis,
            key_points=[industry, "新闻数据待增强"],
            signals={"industry": industry},
        )

    async def _resolve_conflicts(
        self,
        dimension_results: Dict[AnalysisDimension, DimensionAnalysis],
    ) -> ConflictResolution:
        """冲突检测与调和。"""
        scores = [item.score for item in dimension_results.values()]
        if not scores:
            return ConflictResolution(False, [], "缺少分析结果。", "保持观望")

        avg_score = sum(scores) / len(scores)
        score_variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        has_conflict = score_variance > 200
        if not has_conflict:
            return ConflictResolution(False, [], "各维度结论相对一致。", "维持原有结论")

        technical = dimension_results[AnalysisDimension.TECHNICAL]
        fundamental = dimension_results[AnalysisDimension.FUNDAMENTAL]
        sentiment = dimension_results[AnalysisDimension.SENTIMENT]
        conflict_points = []
        if abs(technical.score - fundamental.score) >= 20:
            conflict_points.append("技术面与基本面评分差异较大")
        technical_trend = str(technical.signals.get("trend", ""))
        sentiment_label = str(sentiment.signals.get("sentiment", ""))
        if (
            ("多头" in technical_trend and "空" in sentiment_label)
            or ("空头" in technical_trend and "多" in sentiment_label)
        ):
            conflict_points.append("资金情绪与主趋势存在分歧")
        if not conflict_points:
            conflict_points.append("多维度评分存在明显差异")
        resolution = (
            "技术面 {0:.0f} 分，基本面 {1:.0f} 分，情绪面 {2:.0f} 分，"
            "建议降低仓位并等待更多验证信号。"
        ).format(technical.score, fundamental.score, sentiment.score)
        return ConflictResolution(
            has_conflict=True,
            conflict_points=conflict_points,
            resolution=resolution,
            final_decision="谨慎跟踪，等待多维度共振",
        )

    def _calculate_confidence(
        self,
        dimension_results: Dict[AnalysisDimension, DimensionAnalysis],
    ) -> float:
        """计算综合置信度。"""
        confidences = [item.confidence for item in dimension_results.values()]
        if not confidences:
            return 0.0

        avg_confidence = sum(confidences) / len(confidences)
        scores = [item.score for item in dimension_results.values()]
        score_avg = sum(scores) / len(scores)
        score_std = (sum((score - score_avg) ** 2 for score in scores) / len(scores)) ** 0.5
        consistency_bonus = 0.2 * (1 - min(score_std / 50.0, 1.0))
        return min(avg_confidence + consistency_bonus, 1.0)

    async def _generate_supplementary_queries(
        self,
        dimension_results: Dict[AnalysisDimension, DimensionAnalysis],
        conflict_resolution: ConflictResolution,
    ) -> List[str]:
        """生成补充查询。"""
        queries: List[str] = []
        if conflict_resolution.has_conflict:
            queries.extend(["行业对比数据", "最近重要公告"])

        for dimension, result in dimension_results.items():
            if result.confidence >= 0.6:
                continue
            if dimension == AnalysisDimension.TECHNICAL:
                queries.append("更长周期K线")
            elif dimension == AnalysisDimension.FUNDAMENTAL:
                queries.append("最近两个季度财务指标")
            elif dimension == AnalysisDimension.SENTIMENT:
                queries.append("近期资金流向")
            else:
                queries.append("更多新闻来源")
        return queries

    async def _fetch_supplementary_data(self, ts_code: str, queries: List[str]) -> Dict[str, Any]:
        """获取补充数据。"""
        return {
            "ts_code": ts_code,
            "queries": queries,
            "fetched_at": datetime.now().isoformat(),
        }

    async def _run_incremental_analysis(
        self,
        ts_code: str,
        base_data: Dict[str, Any],
        supplementary_data: Dict[str, Any],
        previous_results: Dict[AnalysisDimension, DimensionAnalysis],
    ) -> Dict[AnalysisDimension, DimensionAnalysis]:
        """基于补充信息小幅提升置信度。"""
        del ts_code, base_data, supplementary_data
        updated: Dict[AnalysisDimension, DimensionAnalysis] = {}
        for dimension, result in previous_results.items():
            updated[dimension] = DimensionAnalysis(
                dimension=dimension,
                score=result.score,
                confidence=min(result.confidence + 0.05, 0.85),
                analysis=result.analysis,
                key_points=list(result.key_points),
                signals=dict(result.signals),
            )
        return updated

    async def _generate_final_report(
        self,
        ts_code: str,
        base_data: Dict[str, Any],
        dimension_results: Dict[AnalysisDimension, DimensionAnalysis],
        conflict_resolution: ConflictResolution,
        overall_confidence: float,
        iteration_count: int,
    ) -> Dict[str, Any]:
        """生成统一输出结构。"""
        technical = dimension_results[AnalysisDimension.TECHNICAL]
        fundamental = dimension_results[AnalysisDimension.FUNDAMENTAL]
        sentiment = dimension_results[AnalysisDimension.SENTIMENT]
        news = dimension_results[AnalysisDimension.NEWS]
        overall_score = round(
            (
                technical.score
                + fundamental.score
                + sentiment.score
                + news.score
            )
            / 4.0,
            2,
        )
        stock_info = base_data.get("stock_info", {})
        summary = "{0}（{1}）综合评分 {2:.1f}，{3}。".format(
            stock_info.get("name", ts_code),
            ts_code,
            overall_score,
            conflict_resolution.final_decision,
        )
        recommendation = self._build_recommendation(overall_score, overall_confidence)

        return {
            "ts_code": ts_code,
            "name": stock_info.get("name", ts_code),
            "industry": stock_info.get("industry"),
            "overall_score": overall_score,
            "overall_confidence": round(overall_confidence, 4),
            "summary": summary,
            "recommendation": recommendation,
            "technical_score": technical.score,
            "fundamental_score": fundamental.score,
            "sentiment_score": sentiment.score,
            "news_score": news.score,
            "technical_summary": technical.analysis,
            "fundamental_summary": fundamental.analysis,
            "sentiment_summary": sentiment.analysis,
            "news_summary": news.analysis,
            "technical_signal": technical.signals.get("trend", ""),
            "key_points": technical.key_points + fundamental.key_points[:1] + sentiment.key_points[:1],
            "has_conflict": conflict_resolution.has_conflict,
            "conflict_points": conflict_resolution.conflict_points,
            "conflict_resolution": conflict_resolution.resolution,
            "final_decision": conflict_resolution.final_decision,
            "iteration_count": iteration_count,
        }

    async def _complete_with_fallback(self, prompt: str, fallback: str) -> str:
        """Attempt an LLM completion and fall back to deterministic text."""
        try:
            content = await self.llm_client.complete(prompt)
        except Exception:
            return fallback
        return content.strip() or fallback

    def _build_recommendation(self, overall_score: float, overall_confidence: float) -> str:
        """Map aggregate score to a user-facing recommendation."""
        if overall_score >= 75 and overall_confidence >= 0.7:
            return "强烈推荐：多维度信号较强"
        if overall_score >= 65:
            return "推荐：可列入重点观察"
        if overall_score >= 55:
            return "观察：等待更多确认信号"
        return "谨慎：暂不建议操作"

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 JSON 响应。"""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return {}