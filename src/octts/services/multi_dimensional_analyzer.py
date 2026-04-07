"""Multi-dimensional stock analysis workflow."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

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

    BASE_TECHNICAL_WEIGHT = 0.67
    BASE_FUNDAMENTAL_WEIGHT = 0.33
    SENTIMENT_ADJUSTMENT_PIVOT = 52.0
    SENTIMENT_ADJUSTMENT_FACTOR = 0.22
    SENTIMENT_ADJUSTMENT_LOWER = -5.5
    SENTIMENT_ADJUSTMENT_UPPER = 5.5
    NEWS_ADJUSTMENT_PIVOT = 50.0
    NEWS_ADJUSTMENT_FACTOR = 0.10
    NEWS_ADJUSTMENT_LOWER = -1.5
    NEWS_ADJUSTMENT_UPPER = 3.0
    SCORE_MODEL = "base_plus_adjustment"

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
        news_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行多维度分析并返回统一结果。"""
        logger.info("Starting multi-dimensional analysis for %s", ts_code)

        base_data = await self._fetch_stock_data(ts_code)
        logger.info(
            "Analyzer base data for %s: stock_info=%s, daily_rows=%s, financial_rows=%s, moneyflow_rows=%s",
            ts_code,
            bool(base_data.get("stock_info")),
            len(base_data.get("daily_data") or []),
            len(base_data.get("financial_data") or []),
            len(base_data.get("moneyflow_data") or []),
        )
        logger.info(
            "Analyzer base data detail for %s: stock_info=%s, latest_daily=%s, latest_financial=%s, latest_moneyflow=%s",
            ts_code,
            base_data.get("stock_info"),
            (base_data.get("daily_data") or [])[-1] if (base_data.get("daily_data") or []) else None,
            (base_data.get("financial_data") or [None])[0],
            (base_data.get("moneyflow_data") or [None])[0],
        )
        if news_context:
            base_data["news_context"] = news_context
        dimension_results = await self._run_parallel_analysis(ts_code, base_data)
        logger.info(
            "Analyzer dimension results for %s: technical=(score=%s, confidence=%s), fundamental=(score=%s, confidence=%s), sentiment=(score=%s, confidence=%s), news=(score=%s, confidence=%s)",
            ts_code,
            dimension_results[AnalysisDimension.TECHNICAL].score,
            dimension_results[AnalysisDimension.TECHNICAL].confidence,
            dimension_results[AnalysisDimension.FUNDAMENTAL].score,
            dimension_results[AnalysisDimension.FUNDAMENTAL].confidence,
            dimension_results[AnalysisDimension.SENTIMENT].score,
            dimension_results[AnalysisDimension.SENTIMENT].confidence,
            dimension_results[AnalysisDimension.NEWS].score,
            dimension_results[AnalysisDimension.NEWS].confidence,
        )
        conflict_resolution = await self._resolve_conflicts(dimension_results)
        overall_confidence = self._calculate_confidence(dimension_results)

        iteration_count = 1
        while enable_iterations and overall_confidence < 0.7 and iteration_count < max_iterations:
            supplementary_queries = await self._generate_supplementary_queries(
                dimension_results,
                conflict_resolution,
            )
            logger.info(
                "Analyzer round %s supplementary queries for %s: %s",
                iteration_count + 1,
                ts_code,
                supplementary_queries,
            )
            supplementary_data = await self._fetch_supplementary_data(ts_code, supplementary_queries)
            logger.info(
                "Analyzer round %s supplementary data for %s: keys=%s, payload=%s",
                iteration_count + 1,
                ts_code,
                sorted(list(supplementary_data.keys())) if isinstance(supplementary_data, dict) else type(supplementary_data).__name__,
                supplementary_data,
            )
            dimension_results = await self._run_incremental_analysis(
                ts_code,
                base_data,
                supplementary_data,
                dimension_results,
            )
            logger.info(
                "Analyzer round %s output for %s: technical=(score=%s, confidence=%s), fundamental=(score=%s, confidence=%s), sentiment=(score=%s, confidence=%s), news=(score=%s, confidence=%s)",
                iteration_count + 1,
                ts_code,
                dimension_results[AnalysisDimension.TECHNICAL].score,
                dimension_results[AnalysisDimension.TECHNICAL].confidence,
                dimension_results[AnalysisDimension.FUNDAMENTAL].score,
                dimension_results[AnalysisDimension.FUNDAMENTAL].confidence,
                dimension_results[AnalysisDimension.SENTIMENT].score,
                dimension_results[AnalysisDimension.SENTIMENT].confidence,
                dimension_results[AnalysisDimension.NEWS].score,
                dimension_results[AnalysisDimension.NEWS].confidence,
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
        logger.info(
            "Analyzer round 1 input for %s: stock_info=%s, daily_rows=%s, financial_rows=%s, moneyflow_rows=%s, news_context_keys=%s",
            ts_code,
            base_data.get("stock_info"),
            len(base_data.get("daily_data") or []),
            len(base_data.get("financial_data") or []),
            len(base_data.get("moneyflow_data") or []),
            sorted(list((base_data.get("news_context") or {}).keys())),
        )
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

        def _to_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        roe = _to_float(latest.get("roe"))
        profit_growth = _to_float(latest.get("netprofit_yoy"))
        revenue_growth = _to_float(latest.get("op_income_yoy"))
        gross_margin = _to_float(latest.get("grossprofit_margin"))
        net_margin = _to_float(latest.get("netprofit_margin"))
        asset_turn = _to_float(latest.get("assets_turn"))

        if not financial_data:
            industry = stock_info.get("industry", "行业未知")
            return DimensionAnalysis(
                dimension=AnalysisDimension.FUNDAMENTAL,
                score=38.0,
                confidence=0.25,
                analysis="最新财务指标缺失，基本面暂按偏保守处理，需等待财报或经营数据补充确认。",
                key_points=[industry, "财务指标缺失", "基本面暂按保守分处理"],
                signals={
                    "roe": None,
                    "netprofit_yoy": None,
                    "op_income_yoy": None,
                    "grossprofit_margin": None,
                    "netprofit_margin": None,
                    "assets_turn": None,
                    "fundamental_summary": "数据缺失，保守评估",
                },
            )

        score = 52.0
        score_reasons: List[str] = []
        warning_reasons: List[str] = []

        if roe is not None:
            if roe >= 20:
                score += 12
                score_reasons.append("ROE维持高位")
            elif roe >= 15:
                score += 8
                score_reasons.append("ROE表现较强")
            elif roe >= 10:
                score += 4
                score_reasons.append("ROE处于可接受区间")
            elif roe >= 5:
                score -= 1
                warning_reasons.append("ROE偏一般")
            else:
                score -= 8
                warning_reasons.append("ROE偏弱")
        else:
            warning_reasons.append("ROE缺失")

        if profit_growth is not None:
            if profit_growth >= 50:
                score += 12
                score_reasons.append("净利润同比高增")
            elif profit_growth >= 20:
                score += 7
                score_reasons.append("净利润同比明显改善")
            elif profit_growth >= 0:
                score += 3
                score_reasons.append("净利润保持正增长")
            elif profit_growth >= -20:
                score -= 6
                warning_reasons.append("净利润同比转弱")
            elif profit_growth >= -50:
                score -= 12
                warning_reasons.append("净利润同比明显下滑")
            else:
                score -= 18
                warning_reasons.append("净利润同比大幅恶化")
        else:
            warning_reasons.append("净利润同比缺失")

        if revenue_growth is not None:
            if revenue_growth >= 30:
                score += 8
                score_reasons.append("营收增速较快")
            elif revenue_growth >= 10:
                score += 4
                score_reasons.append("营收保持增长")
            elif revenue_growth >= 0:
                score += 1
            elif revenue_growth >= -15:
                score -= 4
                warning_reasons.append("营收增长放缓")
            else:
                score -= 8
                warning_reasons.append("营收同比明显回落")
        else:
            warning_reasons.append("营收同比缺失")

        if gross_margin is not None:
            if gross_margin >= 30:
                score += 3
                score_reasons.append("毛利率较好")
            elif gross_margin < 15:
                score -= 3
                warning_reasons.append("毛利率偏低")

        if net_margin is not None:
            if net_margin >= 10:
                score += 3
                score_reasons.append("净利率较好")
            elif net_margin < 5:
                score -= 3
                warning_reasons.append("净利率偏低")

        if asset_turn is not None:
            if asset_turn >= 0.8:
                score += 2
                score_reasons.append("资产周转效率较高")
            elif asset_turn < 0.3:
                score -= 2
                warning_reasons.append("资产周转偏慢")

        industry = stock_info.get("industry", "行业未知")
        clipped_score = max(20.0, min(score, 88.0))
        confidence = 0.72
        available_metrics = [
            metric
            for metric in [roe, profit_growth, revenue_growth, gross_margin, net_margin, asset_turn]
            if metric is not None
        ]
        if len(available_metrics) <= 3:
            confidence = 0.6
        if len(available_metrics) <= 1:
            confidence = 0.45

        analysis_parts = []
        metric_parts = []
        if roe is not None:
            metric_parts.append("ROE {0:.1f}%".format(roe))
        if profit_growth is not None:
            metric_parts.append("净利润同比 {0:.1f}%".format(profit_growth))
        if revenue_growth is not None:
            metric_parts.append("营收同比 {0:.1f}%".format(revenue_growth))
        if metric_parts:
            analysis_parts.append("、".join(metric_parts) + "。")
        if score_reasons:
            analysis_parts.append("当前加分主要来自{0}。".format("、".join(score_reasons[:3])))
        if warning_reasons:
            analysis_parts.append("拖累项主要为{0}。".format("、".join(warning_reasons[:3])))
        if gross_margin is not None or net_margin is not None or asset_turn is not None:
            quality_parts = []
            if gross_margin is not None:
                quality_parts.append("毛利率 {0:.1f}%".format(gross_margin))
            if net_margin is not None:
                quality_parts.append("净利率 {0:.1f}%".format(net_margin))
            if asset_turn is not None:
                quality_parts.append("资产周转 {0:.2f}".format(asset_turn))
            analysis_parts.append("质量修正参考{0}。".format("、".join(quality_parts)))
        analysis = "".join(analysis_parts) or "最新财务指标有限，基本面暂按中性偏保守理解。"

        key_points = [industry, "基本面评分{0:.1f}".format(clipped_score)]
        if roe is not None:
            key_points.append("ROE {0:.1f}%".format(roe))
        if profit_growth is not None:
            key_points.append("净利润同比 {0:.1f}%".format(profit_growth))
        elif revenue_growth is not None:
            key_points.append("营收同比 {0:.1f}%".format(revenue_growth))
        if warning_reasons:
            key_points.append(warning_reasons[0])
        elif score_reasons:
            key_points.append(score_reasons[0])

        signals = {
            "roe": roe,
            "netprofit_yoy": profit_growth,
            "op_income_yoy": revenue_growth,
            "grossprofit_margin": gross_margin,
            "netprofit_margin": net_margin,
            "assets_turn": asset_turn,
            "fundamental_summary": "；".join(score_reasons[:2] + warning_reasons[:2]) or "基本面中性",
        }

        return DimensionAnalysis(
            dimension=AnalysisDimension.FUNDAMENTAL,
            score=clipped_score,
            confidence=confidence,
            analysis=analysis,
            key_points=key_points,
            signals=signals,
        )

    async def _analyze_sentiment(self, ts_code: str, base_data: Dict[str, Any]) -> DimensionAnalysis:
        del ts_code
        moneyflow_data = sorted(
            base_data.get("moneyflow_data", []),
            key=lambda item: str(item.get("trade_date", "")),
            reverse=True,
        )
        recent_rows = moneyflow_data[:3]
        if not recent_rows:
            return DimensionAnalysis(
                dimension=AnalysisDimension.SENTIMENT,
                score=48.0,
                confidence=0.25,
                analysis="资金流数据缺失，情绪面先按中性偏谨慎处理。",
                key_points=["资金数据不足", "情绪面暂按中性偏谨慎"],
                signals={"sentiment": "中性偏谨慎", "net_mf_amount": 0.0},
            )

        total_flow = sum(self._safe_float(item.get("net_mf_amount")) for item in recent_rows)
        latest_flow = self._safe_float(recent_rows[0].get("net_mf_amount"))
        large_order_flow = sum(
            self._safe_float(item.get("buy_lg_amount"))
            + self._safe_float(item.get("buy_elg_amount"))
            - self._safe_float(item.get("sell_lg_amount"))
            - self._safe_float(item.get("sell_elg_amount"))
            for item in recent_rows
        )

        flow_component = self._clip(total_flow / 25000.0, -16.0, 16.0)
        latest_component = self._clip(latest_flow / 18000.0, -7.0, 7.0)
        quality_component = self._clip(large_order_flow / 18000.0, -8.0, 8.0)
        score = round(self._clip(52.0 + flow_component + latest_component + quality_component, 30.0, 75.0), 2)

        if score >= 64:
            sentiment = "资金承接偏强"
        elif score >= 56:
            sentiment = "资金面偏稳"
        elif score <= 42:
            sentiment = "资金承接偏弱"
        else:
            sentiment = "资金情绪中性"

        total_flow_wan = total_flow / 10000.0
        latest_flow_wan = latest_flow / 10000.0
        large_order_flow_wan = large_order_flow / 10000.0
        key_points = [
            sentiment,
            "近3日净流向 {0:.2f} 万元".format(total_flow_wan),
            "单日边际净流向 {0:.2f} 万元".format(latest_flow_wan),
        ]
        if abs(large_order_flow_wan) >= 1:
            key_points.append("大单净额 {0:.2f} 万元".format(large_order_flow_wan))

        return DimensionAnalysis(
            dimension=AnalysisDimension.SENTIMENT,
            score=score,
            confidence=0.72 if len(recent_rows) >= 3 else 0.58,
            analysis=(
                "{0}，近3日净流向合计 {1:.2f} 万元，最近1日净流向 {2:.2f} 万元，"
                "大单与超大单合计净额 {3:.2f} 万元。"
            ).format(sentiment, total_flow_wan, latest_flow_wan, large_order_flow_wan),
            key_points=key_points,
            signals={
                "net_mf_amount": round(total_flow, 2),
                "latest_net_mf_amount": round(latest_flow, 2),
                "large_order_net_amount": round(large_order_flow, 2),
                "sentiment": sentiment,
                "sentiment_components": {
                    "three_day_flow": round(flow_component, 2),
                    "latest_flow": round(latest_component, 2),
                    "large_order_quality": round(quality_component, 2),
                },
            },
        )

    async def _analyze_news(self, ts_code: str, base_data: Dict[str, Any]) -> DimensionAnalysis:
        stock_info = base_data.get("stock_info", {})
        industry = str(stock_info.get("industry") or "未知行业")
        news_context = base_data.get("news_context") or {}
        hot_stocks = set(news_context.get("hot_stocks") or set())
        strong_theme_stocks = set(news_context.get("strong_theme_stocks") or set())
        industry_themes = news_context.get("industry_themes") or {}
        stock_themes = news_context.get("stock_themes") or {}

        matched_stock_themes = list(stock_themes.get(ts_code) or [])
        matched_industry_themes = list(industry_themes.get(industry) or [])
        direct_hit = ts_code in hot_stocks
        strong_theme_hit = ts_code in strong_theme_stocks

        score = 47.0
        if matched_industry_themes:
            score += 6.0
        if direct_hit:
            score += 10.0
        if strong_theme_hit:
            score += 4.0
        score += min(len(matched_stock_themes), 2) * 3.0
        score += min(len(matched_industry_themes), 2) * 2.0
        score = round(self._clip(score, 42.0, 72.0), 2)

        if direct_hit or matched_stock_themes:
            catalyst = "存在主题催化共振"
        elif matched_industry_themes:
            catalyst = "行业层面有主题带动"
        else:
            catalyst = "暂未看到明确主题催化"

        key_points = [industry, catalyst]
        if matched_stock_themes:
            key_points.append("个股主题：" + "、".join(matched_stock_themes[:2]))
        elif matched_industry_themes:
            key_points.append("行业主题：" + "、".join(matched_industry_themes[:2]))

        analysis = (
            "新闻面定位为主题/催化共振评估。{0}；"
            "{1}"
        ).format(
            catalyst,
            (
                "个股直接命中热点主题 {0}".format("、".join(matched_stock_themes[:2]))
                if matched_stock_themes
                else "行业相关主题为 {0}".format("、".join(matched_industry_themes[:2]))
                if matched_industry_themes
                else "当前新闻聚合中未识别到与该股直接相关的热点主题"
            ),
        )

        return DimensionAnalysis(
            dimension=AnalysisDimension.NEWS,
            score=score,
            confidence=0.62 if (direct_hit or matched_stock_themes or matched_industry_themes) else 0.38,
            analysis=analysis,
            key_points=key_points,
            signals={
                "industry": industry,
                "news_direct_hit": direct_hit,
                "news_strong_theme_hit": strong_theme_hit,
                "stock_themes": matched_stock_themes,
                "industry_themes": matched_industry_themes,
                "catalyst": catalyst,
            },
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
        base_score = round(
            technical.score * self.BASE_TECHNICAL_WEIGHT
            + fundamental.score * self.BASE_FUNDAMENTAL_WEIGHT,
            2,
        )
        sentiment_adjustment = self._calculate_adjustment(
            sentiment.score,
            pivot=self.SENTIMENT_ADJUSTMENT_PIVOT,
            factor=self.SENTIMENT_ADJUSTMENT_FACTOR,
            lower=self.SENTIMENT_ADJUSTMENT_LOWER,
            upper=self.SENTIMENT_ADJUSTMENT_UPPER,
        )
        news_adjustment = self._calculate_adjustment(
            news.score,
            pivot=self.NEWS_ADJUSTMENT_PIVOT,
            factor=self.NEWS_ADJUSTMENT_FACTOR,
            lower=self.NEWS_ADJUSTMENT_LOWER,
            upper=self.NEWS_ADJUSTMENT_UPPER,
        )
        overall_score = round(
            self._clip(base_score + sentiment_adjustment + news_adjustment, 0.0, 100.0),
            2,
        )
        stock_info = base_data.get("stock_info", {})
        summary = "{0}（{1}）综合评分 {2:.1f}，主评分 {3:.1f} 由技术面与基本面构成，情绪面 {4:+.1f}、新闻面 {5:+.1f} 作为辅助修正，{6}。".format(
            stock_info.get("name", ts_code),
            ts_code,
            overall_score,
            base_score,
            sentiment_adjustment,
            news_adjustment,
            conflict_resolution.final_decision,
        )
        result = {
            "ts_code": ts_code,
            "name": stock_info.get("name", ts_code),
            "industry": stock_info.get("industry"),
            "overall_score": overall_score,
            "base_score": base_score,
            "sentiment_adjustment": sentiment_adjustment,
            "news_adjustment": news_adjustment,
            "score_model": self.SCORE_MODEL,
            "overall_confidence": round(overall_confidence, 4),
            "summary": summary,
            "technical_score": technical.score,
            "fundamental_score": fundamental.score,
            "sentiment_score": sentiment.score,
            "news_score": news.score,
            "technical_summary": technical.analysis,
            "fundamental_summary": fundamental.analysis,
            "sentiment_summary": sentiment.analysis,
            "news_summary": news.analysis,
            "technical_signal": technical.signals.get("trend", ""),
            "sentiment_signals": sentiment.signals,
            "news_signals": news.signals,
            "key_points": technical.key_points + fundamental.key_points[:1] + sentiment.key_points[:1],
            "has_conflict": conflict_resolution.has_conflict,
            "conflict_points": conflict_resolution.conflict_points,
            "conflict_resolution": conflict_resolution.resolution,
            "final_decision": conflict_resolution.final_decision,
            "iteration_count": iteration_count,
        }
        logger.info(
            "Analyzer final report for %s: overall_score=%s, base_score=%s, overall_confidence=%s, summary=%s, technical_signal=%s",
            ts_code,
            result.get("overall_score"),
            result.get("base_score"),
            result.get("overall_confidence"),
            str(result.get("summary") or "")[:200],
            result.get("technical_signal"),
        )
        return result

    def _calculate_adjustment(
        self,
        score: float,
        *,
        pivot: float,
        factor: float,
        lower: float,
        upper: float,
    ) -> float:
        return round(self._clip((score - pivot) * factor, lower, upper), 2)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    async def _complete_with_fallback(self, prompt: str, fallback: str) -> str:
        """Attempt an LLM completion and fall back to deterministic text."""
        logger.info(
            "Analyzer LLM prompt: %s",
            prompt,
        )
        try:
            content = await self.llm_client.complete(prompt)
            logger.info(
                "Analyzer LLM response: %s",
                content,
            )
        except Exception as exc:
            logger.warning(
                "Analyzer LLM fallback triggered: error=%s, fallback=%s",
                exc,
                fallback,
            )
            return fallback
        normalized = content.strip() if isinstance(content, str) else ""
        if not normalized:
            logger.warning(
                "Analyzer LLM returned empty content, using fallback: %s",
                fallback,
            )
            return fallback
        return normalized

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 JSON 响应。"""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return {}