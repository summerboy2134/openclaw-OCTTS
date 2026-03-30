"""Enhanced stock screening with AI analysis integration."""

import html
import json
import logging
import re
from datetime import date, datetime, timedelta
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from octts.clients.email_client import EmailClient
from octts.clients.wecom_client import WeComClient
from octts.config import Settings
from octts.schemas.screener import ScreenPreset, ScreenResult, TrackedRecommendationState
from octts.services.history_store import FileHistoryStore
from octts.services.stock_screener import StockScreener
from octts.services.screening_store import ScreeningStore
from octts.services.position_store import create_position_store
from octts.services.report_exporter import ReportExporter
from octts.services.multi_dimensional_analyzer import MultiDimensionalAnalyzer
from octts.services.news_aggregator import NewsAggregator
from octts.services.intelligent_report_generator import IntelligentReportGenerator
from octts.services.intelligent_screening_job_manager import maybe_await_progress_callback
from octts.services.report_email_service import ReportEmailService
from octts.services.screening_validator import ScreeningValidator

logger = logging.getLogger(__name__)

TOP_RECOMMENDATION_LIMIT = 10
TODAY_TOP_LIMIT = 3
WINDOW_RECOMMENDATION_TAG = "今日候选"
REPEAT_CONFIDENCE_BONUS = 0.08
MAX_CONFIDENCE = 0.98
INDUSTRY_FLOW_SCORE_CAP = 3.0


class EnhancedScreeningScheduler:
    """增强的选股调度器（集成AI分析）"""

    def __init__(
        self,
        settings: Settings,
        screener: Optional[StockScreener] = None,
        store: Optional[ScreeningStore] = None,
        analyzer: Optional[MultiDimensionalAnalyzer] = None,
        news_aggregator: Optional[NewsAggregator] = None,
        report_generator: Optional[IntelligentReportGenerator] = None,
        wecom_client: Optional[WeComClient] = None,
        email_service: Optional[ReportEmailService] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.settings = settings
        self.screener = screener or StockScreener(settings)
        self.store = store or ScreeningStore(settings)
        self.analyzer = analyzer or MultiDimensionalAnalyzer(settings)
        self.news_aggregator = news_aggregator or NewsAggregator(settings)
        self.report_generator = report_generator or IntelligentReportGenerator(
            settings,
            analyzer=self.analyzer
        )
        self.validator = ScreeningValidator(settings)
        self.wecom_client = wecom_client
        if self.wecom_client is None and settings.wecom_webhook_url:
            self.wecom_client = WeComClient(settings)

        self.email_service = email_service
        self.progress_callback = progress_callback
        if self.email_service is None and settings.email_enabled:
            history_store = FileHistoryStore(settings.history_dir_path)
            self.email_service = ReportEmailService(
                settings=settings,
                history_store=history_store,
                report_exporter=ReportExporter(
                    settings=settings,
                    history_store=history_store,
                    position_store=create_position_store(settings),
                ),
                email_client=EmailClient(settings),
            )

    async def run_intelligent_screening(self) -> Dict[str, Any]:
        """
        执行智能选股（带AI分析）

        Returns:
            执行结果
        """
        logger.info("Starting intelligent stock screening")
        start_time = datetime.now()
        total_steps = 6

        await self._report_progress(
            current_step=1,
            total_steps=total_steps,
            step_name="新闻采集",
            progress_percent=5,
            message="正在采集最新市场新闻...",
        )

        # 1. 采集最新新闻
        logger.info("Step 1: Collecting news")
        logger.info("Step 1.1: Starting news collector aggregation")
        news_items = await self.news_aggregator.collect_all()
        logger.info("Step 1.1 complete: Collected %s news items", len(news_items))
        await self._report_progress(
            current_step=1,
            total_steps=total_steps,
            step_name="新闻采集",
            progress_percent=12,
            message="新闻采集完成，正在评估重要性...",
            details={"news_count": len(news_items)},
        )
        logger.info("Step 1.2: Starting news importance analysis for %s items", len(news_items))
        news_items = await self.news_aggregator.analyze_importance(news_items)
        logger.info("Step 1.2 complete")
        logger.info("Step 1.3: Starting news clustering for %s items", len(news_items))
        news_clusters = await self.news_aggregator.cluster_news(news_items)
        logger.info("Step 1.3 complete: Generated %s clusters", len(news_clusters))
        await self._report_progress(
            current_step=1,
            total_steps=total_steps,
            step_name="新闻采集",
            progress_percent=20,
            message="新闻聚类完成。",
            details={
                "news_count": len(news_items),
                "cluster_count": len(news_clusters),
            },
        )

        # 2. 执行技术选股
        logger.info("Step 2: Running technical screening")
        await self._report_progress(
            current_step=2,
            total_steps=total_steps,
            step_name="技术选股",
            progress_percent=28,
            message="正在执行技术选股策略...",
        )
        screening_results, trade_date = await self._run_screening_strategies()
        await self._report_progress(
            current_step=2,
            total_steps=total_steps,
            step_name="技术选股",
            progress_percent=38,
            message="技术选股完成，正在整理最新一轮 Top10 展示窗口...",
            details={"strategy_count": len(screening_results)},
        )

        # 3. AI深度分析展示窗口股票
        logger.info("Step 3: AI analysis for display stocks")
        candidate_limit = max(TOP_RECOMMENDATION_LIMIT, self.settings.screening_top_n)
        candidate_codes = self._get_top_stocks(screening_results, limit=candidate_limit)
        eligible_candidate_codes = self._filter_out_tracked_and_holding_codes(candidate_codes)
        analysis_target_codes = eligible_candidate_codes[:TOP_RECOMMENDATION_LIMIT]
        await self._report_progress(
            current_step=3,
            total_steps=total_steps,
            step_name="AI 深度分析",
            progress_percent=42,
            message="正在分析当前 Top10 展示窗口股票...",
            details={"total_items": len(analysis_target_codes), "completed_items": 0},
        )
        ai_analyses = await self._analyze_top_stocks(
            analysis_target_codes,
            total_steps=total_steps,
            current_step=3,
        )

        # 4. 结合新闻和技术面筛选
        logger.info("Step 4: Combining news and technical analysis")
        await self._report_progress(
            current_step=4,
            total_steps=total_steps,
            step_name="融合打分",
            progress_percent=78,
            message="正在融合新闻、技术和 AI 分析结果...",
            details={"analyzed_items": len(ai_analyses)},
        )
        market_snapshot = self.screener.client.get_or_build_screening_snapshot(trade_date)
        final_recommendations = await self._combine_analysis(
            screening_results,
            ai_analyses,
            news_clusters,
            market_snapshot=market_snapshot,
        )
        pool_states = self._build_recommendation_pool_states(
            trade_date=datetime.strptime(trade_date, "%Y%m%d").date(),
            screening_results=screening_results,
            final_recommendations=final_recommendations,
            candidate_codes=eligible_candidate_codes,
        )
        self.store.upsert_recommendation_pool_states(pool_states)

        # 5. 生成智能报告
        logger.info("Step 5: Generating intelligent report")
        await self._report_progress(
            current_step=5,
            total_steps=total_steps,
            step_name="生成报告",
            progress_percent=88,
            message="正在生成智能报告...",
            details={"final_recommendations": len(final_recommendations)},
        )
        market_data = await self._get_market_data()
        report_context = self._build_report_context(
            trade_date=datetime.strptime(trade_date, "%Y%m%d").date(),
            pool_states=[item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in pool_states],
            ai_analyses=ai_analyses,
            final_recommendations=final_recommendations,
        )
        report = await self.report_generator.generate_morning_report(
            news_clusters=news_clusters,
            market_data=market_data,
            stock_pool=[item.ts_code for item in pool_states],
            screening_context=report_context,
        )
        self._save_recommendation_run(
            trade_date=trade_date,
            screening_results=screening_results,
            ai_analyses=ai_analyses,
            final_recommendations=final_recommendations,
            report_id=getattr(report, "report_id", None),
        )
        self._save_dashboard_snapshot(
            screening_results=screening_results,
            ai_analyses=ai_analyses,
            news_clusters=news_clusters,
            report=report,
            final_recommendations=final_recommendations,
            trade_date=datetime.strptime(trade_date, "%Y%m%d").date(),
            report_context=report_context,
        )
        await self._report_progress(
            current_step=5,
            total_steps=total_steps,
            step_name="生成报告",
            progress_percent=94,
            message="智能报告已生成，正在保存结果...",
            details={"report_id": getattr(report, "report_id", "")},
        )

        # 6. 推送通知
        await self._report_progress(
            current_step=6,
            total_steps=total_steps,
            step_name="通知发送",
            progress_percent=97,
            message="正在处理通知与收尾步骤...",
            details={"notifications_enabled": bool(self.settings.screening_notify)},
        )
        if self.settings.screening_notify:
            await self._send_intelligent_notifications(report, final_recommendations)

        # 7. 验证和质量检查
        logger.info("Step 7: Validating recommendation quality")
        quality_analysis = await self.validator.analyze_recommendation_quality(final_recommendations)
        logic_check = self.validator.check_logic_consistency(final_recommendations)
        
        logger.info(f"Quality analysis: {quality_analysis}")
        logger.info(f"Logic check: {logic_check}")

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "current_step": total_steps,
            "total_steps": total_steps,
            "duration_seconds": duration,
            "news_count": len(news_items),
            "cluster_count": len(news_clusters),
            "screened_stocks": len(analysis_target_codes),
            "final_recommendations": len(final_recommendations),
            "frontlist_count": len([state for state in pool_states if state.in_frontlist]),
            "tracking_pool_count": len(pool_states),
            "today_top_count": len([state for state in pool_states if state.source_tag == "今日Top3"]),
            "continuation_count": len([state for state in pool_states if state.source_tag == "昨日延续"]),
            "report_id": report.report_id,
            "quality_score": quality_analysis.get("quality_score", 0),
            "quality_analysis": quality_analysis,
            "logic_check": logic_check,
        }

    async def _run_screening_strategies(self) -> Any:
        """运行选股策略"""
        strategies = self._get_active_strategies()
        results = {}
        trade_date = self.screener._get_latest_trade_date()
        market_snapshot = self.screener.client.get_or_build_screening_snapshot(trade_date)

        logger.info("Technical screening: %s strategies queued, shared trade_date=%s", len(strategies), trade_date)
        for index, strategy in enumerate(strategies, start=1):
            logger.info(
                "Technical screening strategy %s/%s start: %s (%s)",
                index,
                len(strategies),
                strategy.name,
                strategy.id,
            )
            try:
                result = self.screener.screen(
                    strategy.criteria,
                    trade_date=trade_date,
                    market_snapshot=market_snapshot,
                )
                logger.info(
                    "Technical screening strategy %s/%s complete: %s matched %s stocks in %.2fs",
                    index,
                    len(strategies),
                    strategy.id,
                    result.total_count,
                    result.execution_time,
                )
                results[strategy.id] = result
                await self.store.save_screening_result(strategy.id, result)
                logger.info(
                    "Technical screening strategy %s/%s saved: %s",
                    index,
                    len(strategies),
                    strategy.id,
                )
            except Exception as e:
                logger.error(
                    "Technical screening strategy %s/%s failed: %s (%s)",
                    index,
                    len(strategies),
                    strategy.name,
                    e,
                )

        logger.info("Technical screening complete: %s successful strategies", len(results))
        return results, trade_date

    async def _analyze_top_stocks(
        self,
        stock_codes: List[str],
        *,
        total_steps: int,
        current_step: int,
    ) -> Dict[str, Dict[str, Any]]:
        """并行分析TOP股票"""
        analyses = {}
        total_items = len(stock_codes)

        if not stock_codes:
            await self._report_progress(
                current_step=current_step,
                total_steps=total_steps,
                step_name="AI 深度分析",
                progress_percent=76,
                message="本轮无 Top10 展示股票可供分析。",
                details={"total_items": 0, "completed_items": 0},
            )
            return analyses

        # 并行分析所有股票
        logger.info(f"Starting parallel analysis for {total_items} stocks")
        tasks = []
        for code in stock_codes:
            logger.info("Queueing AI analysis task for %s", code)
            task = self.analyzer.analyze(
                code,
                enable_iterations=True,
                max_iterations=3
            )
            tasks.append((code, task))

        # 逐个完成任务并报告进度
        completed = 0
        for index, (code, task) in enumerate(tasks, start=1):
            progress_percent = 42 + int((index - 1) * 34 / total_items)
            logger.info(f"Starting multi-dimensional analysis for {code} ({index}/{total_items})")
            await self._report_progress(
                current_step=current_step,
                total_steps=total_steps,
                step_name="AI 深度分析",
                progress_percent=progress_percent,
                message=f"正在分析前台股票 {code} ({index}/{total_items})...",
                details={
                    "current_symbol": code,
                    "completed_items": completed,
                    "total_items": total_items,
                },
            )
            try:
                analysis = await task
                analyses[code] = analysis
                logger.info(f"Successfully analyzed {code}")
            except Exception as e:
                logger.error(f"Failed to analyze {code}: {e}", exc_info=True)

            completed += 1
            progress_percent = 42 + int(completed * 34 / total_items)
            await self._report_progress(
                current_step=current_step,
                total_steps=total_steps,
                step_name="AI 深度分析",
                progress_percent=progress_percent,
                message=f"Top10 展示窗口分析进度：{completed}/{total_items}",
                details={
                    "current_symbol": code,
                    "completed_items": completed,
                    "total_items": total_items,
                },
            )

        return analyses

    async def _report_progress(
        self,
        *,
        current_step: int,
        total_steps: int,
        step_name: str,
        progress_percent: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        await maybe_await_progress_callback(
            self.progress_callback,
            {
                "status": "running",
                "current_step": current_step,
                "total_steps": total_steps,
                "step_name": step_name,
                "progress_percent": progress_percent,
                "message": message,
                "details": details or {},
            },
        )

    async def _combine_analysis(
        self,
        screening_results: Dict[str, ScreenResult],
        ai_analyses: Dict[str, Dict[str, Any]],
        news_clusters: List[Any],
        *,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        结合多维度分析生成最终推荐

        Returns:
            {股票代码: {推荐信息}}
        """
        final_recommendations = {}
        stock_map = self._build_screened_stock_map(screening_results)
        all_market_stock_map = self._build_all_market_stock_map(market_snapshot)
        industry_adjustments = self._build_industry_flow_adjustments(
            stock_map,
            all_market_stock_map=all_market_stock_map,
        )

        # 从新闻中提取热点股票
        news_hot_stocks = self._extract_news_hot_stocks(news_clusters)

        # 遍历AI分析结果
        for code, analysis in ai_analyses.items():
            confidence = analysis.get("overall_confidence", 0)
            score = analysis.get("overall_score", 50)

            # 计算综合得分
            # 基础分 = AI分析分数
            final_score = score

            # 新闻仅作为催化解释与主题归因的辅助项
            if code in news_hot_stocks:
                final_score += 3

            # 如果在多个策略中出现，加分
            appearance_count = sum(
                1 for result in screening_results.values()
                if result and any(s.ts_code == code for s in result.stocks)
            )
            final_score += appearance_count * 5

            industry_adjustment = industry_adjustments.get(code, {})
            industry_heat_score = float(industry_adjustment.get("industry_heat_score") or 0.0)
            adjusted_final_score = final_score + industry_heat_score
            weighted_score = adjusted_final_score * confidence

            if weighted_score >= 55:  # 适度放宽综合得分阈值
                final_recommendations[code] = {
                    "score": weighted_score,
                    "overall_score": score,
                    "final_score": final_score,
                    "adjusted_final_score": adjusted_final_score,
                    "weighted_score": weighted_score,
                    "ai_confidence": confidence,
                    "ai_summary": analysis.get("summary", ""),
                    "technical_signal": analysis.get("technical_signal", ""),
                    "technical_score": analysis.get("technical_score"),
                    "fundamental_score": analysis.get("fundamental_score"),
                    "sentiment_score": analysis.get("sentiment_score"),
                    "news_score": analysis.get("news_score"),
                    "news_mentioned": code in news_hot_stocks,
                    "strategy_count": appearance_count,
                    "industry": industry_adjustment.get("industry"),
                    "industry_heat_score": industry_heat_score,
                    "industry_flow_bias": industry_adjustment.get("industry_flow_bias", "中性"),
                    "industry_positive_ratio": industry_adjustment.get("industry_positive_ratio"),
                    "industry_3d_net_inflow": industry_adjustment.get("industry_3d_net_inflow"),
                    "industry_flow_value": industry_adjustment.get("industry_flow_value"),
                    "recommendation": self._generate_recommendation(
                        weighted_score,
                        analysis
                    )
                }

        # 按分数排序
        sorted_recommendations = dict(
            sorted(
                final_recommendations.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
        )

        return sorted_recommendations

    @staticmethod
    def _build_screened_stock_map(screening_results: Dict[str, ScreenResult]) -> Dict[str, Any]:
        stock_map: Dict[str, Any] = {}
        for result in screening_results.values():
            if not result:
                continue
            for stock in result.stocks:
                stock_map.setdefault(stock.ts_code, stock)
        return stock_map

    @staticmethod
    def _build_all_market_stock_map(market_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(market_snapshot, dict):
            return {}
        stocks = market_snapshot.get("stocks")
        if not isinstance(stocks, list):
            return {}
        stock_map: Dict[str, Dict[str, Any]] = {}
        for stock in stocks:
            if not isinstance(stock, dict):
                continue
            ts_code = str(stock.get("ts_code") or "").strip()
            if not ts_code:
                continue
            stock_map.setdefault(ts_code, stock)
        return stock_map

    def _build_industry_flow_adjustments(
        self,
        stock_map: Dict[str, Any],
        *,
        all_market_stock_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        candidate_industries = set()
        for stock in stock_map.values():
            industry = self._extract_industry_name(stock)
            if industry:
                candidate_industries.add(industry)

        if not candidate_industries:
            return {}

        industry_totals = self._build_industry_flow_totals(
            candidate_industries,
            all_market_stock_map=all_market_stock_map or {},
        )

        adjustments: Dict[str, Dict[str, Any]] = {}
        for code, stock in stock_map.items():
            industry = self._extract_industry_name(stock)
            if not industry:
                continue
            metrics = industry_totals.get(industry)
            if not metrics:
                continue
            adjustments[code] = {
                "industry": industry,
                "industry_3d_net_inflow": metrics["industry_3d_net_inflow"],
                "industry_heat_score": metrics["industry_heat_score"],
                "industry_flow_bias": metrics["industry_flow_bias"],
                "industry_positive_ratio": metrics["industry_positive_ratio"],
                "industry_flow_value": metrics["industry_3d_net_inflow"],
            }
        return adjustments

    def _build_industry_flow_totals(
        self,
        industries: set[str],
        *,
        all_market_stock_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        industry_groups: Dict[str, List[str]] = {industry: [] for industry in industries}
        for ts_code, stock in all_market_stock_map.items():
            industry = self._extract_industry_name(stock)
            if industry not in industry_groups:
                continue
            industry_groups[industry].append(ts_code)

        metrics_map: Dict[str, Dict[str, Any]] = {}
        for industry, ts_codes in industry_groups.items():
            if not ts_codes:
                continue
            stock_flows = [self._build_stock_moneyflow_summary(ts_code) for ts_code in ts_codes]
            valid_flows = [item for item in stock_flows if item is not None]
            if not valid_flows:
                continue
            total_inflow = sum(item["recent_3d_net_inflow"] for item in valid_flows)
            positive_ratio = sum(item["positive_flag"] for item in valid_flows) / len(valid_flows)
            normalized_flow = 0.0
            if total_inflow > 0:
                normalized_flow = 1.0
            elif total_inflow < 0:
                normalized_flow = -1.0
            raw_score = (positive_ratio - 0.5) * 4 + normalized_flow
            industry_heat_score = max(-INDUSTRY_FLOW_SCORE_CAP, min(INDUSTRY_FLOW_SCORE_CAP, round(raw_score, 2)))
            metrics_map[industry] = {
                "industry_3d_net_inflow": round(total_inflow, 2),
                "industry_positive_ratio": round(positive_ratio, 4),
                "industry_flow_bias": self._describe_industry_flow_bias(industry_heat_score),
                "industry_heat_score": industry_heat_score,
            }
        return metrics_map

    def _build_stock_moneyflow_summary(self, ts_code: str) -> Optional[Dict[str, float]]:
        recent_3d_net_inflow = self._fetch_recent_moneyflow_total(ts_code)
        return {
            "recent_3d_net_inflow": recent_3d_net_inflow,
            "positive_flag": 1.0 if recent_3d_net_inflow > 0 else 0.0,
        }

    def _fetch_recent_moneyflow_total(self, ts_code: str) -> float:
        rows = self.screener.client.fetch_moneyflow(ts_code)
        if not rows:
            return 0.0
        recent_rows = sorted(rows, key=lambda item: str(item.get("trade_date") or ""), reverse=True)[:3]
        total = 0.0
        for item in recent_rows:
            value = item.get("net_mf_amount")
            try:
                total += float(value or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _extract_industry_name(stock: Any) -> str:
        if isinstance(stock, dict):
            return str(stock.get("industry") or "").strip()
        return str(getattr(stock, "industry", "") or "").strip()

    @staticmethod
    def _describe_industry_flow_bias(industry_heat_score: float) -> str:
        if industry_heat_score >= 1.0:
            return "偏强"
        if industry_heat_score <= -1.0:
            return "偏弱"
        return "中性"

    def _extract_news_hot_stocks(self, news_clusters: List[Any]) -> set[str]:
        """Extract normalized stock codes from high-importance news clusters."""
        news_hot_stocks = set()
        for cluster in news_clusters:
            if cluster.importance <= 0.6:
                continue

            for value in getattr(cluster, "key_stocks", []):
                normalized = self._normalize_stock_code(value)
                if normalized:
                    news_hot_stocks.add(normalized)

            for item in getattr(cluster, "news_items", []):
                for value in getattr(item, "related_stocks", []) or []:
                    normalized = self._normalize_stock_code(value)
                    if normalized:
                        news_hot_stocks.add(normalized)

        return news_hot_stocks

    @staticmethod
    def _normalize_stock_code(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None

        candidate = value.strip().upper()
        if re.fullmatch(r"\d{6}\.(SH|SZ)", candidate):
            return candidate
        if re.fullmatch(r"\d{6}", candidate):
            suffix = "SH" if candidate.startswith("6") else "SZ"
            return f"{candidate}.{suffix}"
        return None

    def _generate_recommendation(
        self,
        score: float,
        analysis: Dict[str, Any]
    ) -> str:
        """生成操作建议"""
        if score >= 80:
            return "强烈推荐：多维度共振，建议重点关注"
        elif score >= 70:
            return "推荐：技术面良好，可适当关注"
        elif score >= 60:
            return "观察：有一定机会，建议跟踪"
        else:
            return "谨慎：暂不建议操作"

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
                if stock.ts_code not in stock_scores:
                    stock_scores[stock.ts_code] = {
                        "count": 0,
                        "total_score": 0,
                        "best_rank": float('inf'),
                        "technical_score": stock.technical_score or 0,
                        "pct_change": stock.pct_change or 0,
                        "volume_ratio": stock.volume_ratio or 0,
                        "rsi": stock.rsi,
                    }

                stock_scores[stock.ts_code]["count"] += 1
                stock_scores[stock.ts_code]["total_score"] += (
                    100 - i * 2  # 排名越靠前分数越高
                )
                stock_scores[stock.ts_code]["best_rank"] = min(
                    stock_scores[stock.ts_code]["best_rank"],
                    i
                )

        # 第二层：用多维度规则再筛选
        # 优先级：出现次数 > 技术评分 > 成交量 > 涨幅
        filtered_stocks = []
        for code, scores in stock_scores.items():
            # 基础筛选：至少出现在一个策略中
            if scores["count"] < 1:
                continue
            
            # 技术评分要求
            if scores["technical_score"] < 45:
                continue
            
            # 成交量要求（放量）
            if scores["volume_ratio"] < 1.0:
                continue
            
            # RSI 过热过冷过滤（可选）
            if scores["rsi"] is not None:
                if scores["rsi"] > 85 or scores["rsi"] < 15:
                    continue
            
            filtered_stocks.append((code, scores))

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
            f"Two-layer screening: {len(stock_scores)} candidates → "
            f"{len(filtered_stocks)} qualified → {min(len(sorted_stocks), limit)} selected"
        )

        return [code for code, _ in sorted_stocks[:limit]]

    async def _get_market_data(self) -> Dict[str, Any]:
        """获取市场数据"""
        # TODO: 实现市场数据获取
        return {
            "indices": {
                "上证指数": {"value": 3000.00, "change": 0.5},
                "深证成指": {"value": 10000.00, "change": 0.8},
                "创业板指": {"value": 2000.00, "change": 1.2},
            },
            "rise_count": 2000,
            "fall_count": 1500,
            "limit_up": 50,
            "limit_down": 10,
            "total_amount": 8000,
            "trend": "震荡上行",
            "volume_trend": "温和放量",
            "sentiment": "偏多",
        }

    async def _send_intelligent_notifications(
        self,
        report: Any,
        recommendations: Dict[str, Dict[str, Any]]
    ) -> None:
        """发送智能通知"""
        # 企业微信通知
        if self.wecom_client and self.settings.wecom_webhook_url:
            try:
                # 格式化推荐列表
                top_recommendations = list(recommendations.items())[:5]
                rec_text = "\n".join([
                    f"{i+1}. {code}: {info['recommendation']} (得分:{info['score']:.1f})"
                    for i, (code, info) in enumerate(top_recommendations)
                ])

                message = f"""## 🤖 AI智能选股报告

{report.summary}

### 📈 今日推荐（TOP5）
{rec_text}

### 🎯 关键要点
{chr(10).join(f'• {p}' for p in report.key_points[:3])}

### 💡 操作建议
{chr(10).join(f'• {r}' for r in report.recommendations[:3])}

> 报告时间：{report.generate_time.strftime('%Y-%m-%d %H:%M')}
"""
                self.wecom_client.send_markdown(message)
                logger.info("Sent intelligent screening report to WeChat")

            except Exception as e:
                logger.error(f"Failed to send WeChat notification: {e}")

        # 邮件通知
        if self.email_service and self.settings.email_enabled:
            try:
                html_content = self._generate_html_report(report, recommendations)
                await self.email_service.send_intelligent_screening_email(
                    subject=f"AI智能选股报告 - {datetime.now().strftime('%Y-%m-%d')}",
                    html_content=html_content,
                    body="OCTTS 智能选股摘要见邮件正文，离线报告包见附件。",
                )
                logger.info("Sent intelligent screening report via email")

            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")

    def _generate_html_report(
        self,
        report: Any,
        recommendations: Dict[str, Dict[str, Any]]
    ) -> str:
        """生成HTML报告"""
        del recommendations
        markdown = self.report_generator.format_report_markdown(report)
        return "<html><body><pre style=\"white-space: pre-wrap; font-family: sans-serif;\">{0}</pre></body></html>".format(
            html.escape(markdown)
        )

    def _save_recommendation_run(
        self,
        *,
        trade_date: str,
        screening_results: Dict[str, ScreenResult],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
        report_id: Optional[str],
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
        sorted_states = sorted(pool_states, key=lambda item: item.get("recommendation_score", 0.0), reverse=True)
        for rank, state in enumerate(sorted_states, start=1):
            code = state.get("ts_code")
            stock = stock_map.get(code)
            analysis = ai_analyses.get(code, {})
            info = final_recommendations.get(code, {})
            source_tag = state.get("source_tag") or WINDOW_RECOMMENDATION_TAG
            items.append({
                "ts_code": code,
                "name": state.get("name") or getattr(stock, "name", "") or analysis.get("name") or "",
                "recommend_rank": rank,
                "recommend_score": state.get("recommendation_score", info.get("score")),
                "priority_score": state.get("priority_score"),
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
            })
        return items

    def _save_dashboard_snapshot(
        self,
        *,
        screening_results: Dict[str, ScreenResult],
        ai_analyses: Dict[str, Dict[str, Any]],
        news_clusters: List[Any],
        report: Any,
        final_recommendations: Dict[str, Dict[str, Any]],
        trade_date: date,
        report_context: Dict[str, Any],
    ) -> None:
        """Persist the latest intelligent screening snapshot for the dashboard."""
        stock_name_map = {}
        total_stocks = 0
        for result in screening_results.values():
            if not result:
                continue
            total_stocks += len(result.stocks)
            for stock in result.stocks:
                stock_name_map.setdefault(stock.ts_code, stock.name)

        pool_states = self.store.list_recommendation_pool(trade_date=trade_date)
        display_states = sorted(pool_states, key=lambda item: item.get("recommendation_score", 0.0), reverse=True)[:TOP_RECOMMENDATION_LIMIT]
        today_top_states = [item for item in display_states if item.get("source_tag") == "今日Top3"]
        continuation_states = [item for item in display_states if item.get("source_tag") == "昨日延续"]

        snapshot = {
            "generated_at": datetime.now().isoformat(),
            "screening_results": {
                "strategy_count": len(screening_results),
                "total_stocks": total_stocks,
                "final_recommendations": len(display_states),
                "frontlist_count": len(display_states),
                "shadow_count": 0,
                "candidate_count": len(pool_states),
                "today_top_count": len(today_top_states),
                "continuation_count": len(continuation_states),
            },
            "recommendation_pool": {
                "frontlist": display_states,
                "shadow": [],
                "shadow_symbols": [],
                "today_top": today_top_states,
                "yesterday_continuations": continuation_states,
            },
            "ai_analyses": self._build_dashboard_ai_payload(
                ai_analyses,
                final_recommendations,
                stock_name_map,
                pool_states,
            ),
            "news_clusters": [
                {
                    "cluster_id": getattr(cluster, "cluster_id", ""),
                    "theme": getattr(cluster, "theme", ""),
                    "importance": getattr(cluster, "importance", 0.0),
                    "summary": getattr(cluster, "summary", ""),
                    "key_stocks": list(getattr(cluster, "key_stocks", []) or []),
                    "news_items": [
                        {
                            "title": getattr(item, "title", ""),
                            "source": getattr(getattr(item, "source", None), "value", ""),
                            "publish_time": getattr(getattr(item, "publish_time", None), "isoformat", lambda: None)(),
                        }
                        for item in getattr(cluster, "news_items", []) or []
                    ],
                }
                for cluster in news_clusters
            ],
            "intelligent_report": {
                "report_id": getattr(report, "report_id", ""),
                "title": getattr(report, "title", "AI智能报告"),
                "summary": getattr(report, "summary", ""),
                "sections": [
                    {
                        "title": getattr(section, "title", ""),
                        "content": getattr(section, "content", ""),
                        "priority": getattr(section, "priority", 0),
                        "data": getattr(section, "data", None),
                    }
                    for section in getattr(report, "sections", []) or []
                ],
                "recommendations": list(getattr(report, "recommendations", []) or []),
                "key_points": list(getattr(report, "key_points", []) or []),
                "blocks": (getattr(report, "metadata", {}) or {}).get("report_blocks", {}),
            },
            "report_context": report_context,
        }

        snapshot_dir = Path(self.settings.history_dir_path) / "intelligent_screening"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        latest_path = snapshot_dir / "latest.json"
        dated_path = snapshot_dir / f"{trade_date.strftime('%Y%m%d')}.json"
        for path in (latest_path, dated_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_dashboard_ai_payload(
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
        stock_name_map: Dict[str, str],
        pool_states: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        payload = {}
        state_map = {item.get("ts_code"): item for item in pool_states}
        merged_codes = list(dict.fromkeys(list(state_map.keys()) + list(ai_analyses.keys())))
        for code in merged_codes:
            analysis = ai_analyses.get(code, {})
            merged = dict(analysis)
            recommendation_meta = final_recommendations.get(code, {})
            state = state_map.get(code, {})
            merged["name"] = (
                stock_name_map.get(code)
                or state.get("name")
                or merged.get("name")
                or recommendation_meta.get("name")
                or ""
            )
            overall_score = merged.get("overall_score", merged.get("score", 0))
            recommendation_score = state.get(
                "recommendation_score",
                recommendation_meta.get("weighted_score", recommendation_meta.get("score", overall_score)),
            )
            if recommendation_meta:
                merged["recommendation"] = recommendation_meta.get("recommendation", merged.get("recommendation", ""))
                merged["news_mentioned"] = state.get("news_mentioned", recommendation_meta.get("news_mentioned", False))
                merged["strategy_count"] = state.get("strategy_count", recommendation_meta.get("strategy_count", 0))
            merged["overall_score"] = overall_score
            merged["priority_score"] = state.get("priority_score", recommendation_meta.get("priority_score", overall_score))
            merged["recommendation_score"] = recommendation_score
            merged["hit_streak_days"] = state.get("hit_streak_days", 0)
            merged["miss_streak_days"] = state.get("miss_streak_days", 0)
            merged["industry"] = state.get("industry", recommendation_meta.get("industry"))
            merged["industry_heat_score"] = state.get("industry_heat_score", recommendation_meta.get("industry_heat_score"))
            merged["industry_flow_bias"] = state.get("industry_flow_bias", recommendation_meta.get("industry_flow_bias"))
            merged["score_change"] = state.get("score_change")
            merged["previous_recommendation_score"] = state.get("previous_recommendation_score")
            merged["tracking_status"] = state.get("tracking_status")
            merged["in_frontlist"] = state.get("in_frontlist", False)
            merged["llm_focus_level"] = state.get("llm_focus_level")
            merged["source_tag"] = state.get("source_tag") or WINDOW_RECOMMENDATION_TAG
            merged["is_repeat_pick"] = bool(state.get("is_repeat_pick", False))
            merged["score_components"] = {
                "strategy_count": state.get("strategy_count", recommendation_meta.get("strategy_count", 0)),
                "news_mentioned": bool(state.get("news_mentioned", recommendation_meta.get("news_mentioned", False))),
                "confidence": state.get("display_confidence", state.get("ai_confidence", merged.get("overall_confidence", merged.get("confidence")))),
            }
            display_confidence = state.get(
                "display_confidence",
                state.get("ai_confidence", merged.get("overall_confidence", merged.get("confidence"))),
            )
            merged["confidence"] = display_confidence
            merged["ai_confidence"] = display_confidence
            merged["overall_confidence"] = merged.get("overall_confidence", display_confidence)
            payload[code] = merged
        return payload

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
    def _resolve_stock_name(
        code: str,
        stock: Any,
        recommendation: Dict[str, Any],
        current_state: Optional[Dict[str, Any]],
        historical_state: Optional[Dict[str, Any]],
    ) -> str:
        return (
            getattr(stock, "name", None)
            or (current_state or {}).get("name")
            or recommendation.get("name")
            or (historical_state or {}).get("name")
            or code
        )

    def _build_recommendation_pool_states(
        self,
        *,
        trade_date: date,
        screening_results: Dict[str, ScreenResult],
        final_recommendations: Dict[str, Dict[str, Any]],
        candidate_codes: List[str],
    ) -> List[TrackedRecommendationState]:
        previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(trade_date)
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

        display_codes = candidate_codes[:TOP_RECOMMENDATION_LIMIT]
        ranked_display_codes = sorted(
            display_codes,
            key=lambda code: float(
                (final_recommendations.get(code, {}) or {}).get("weighted_score")
                or (final_recommendations.get(code, {}) or {}).get("recommendation_score")
                or (final_recommendations.get(code, {}) or {}).get("score")
                or getattr(stock_map.get(code), "recommendation_score", None)
                or getattr(stock_map.get(code), "score", None)
                or 0.0
            ),
            reverse=True,
        )
        today_top_codes = ranked_display_codes[:TODAY_TOP_LIMIT]
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
        previous_top3_codes = [
            item.get("ts_code")
            for item in sorted(
                [item for item in previous_states.values() if item.get("source_tag") == "今日Top3" and item.get("ts_code")],
                key=lambda item: float(item.get("recommendation_score") or 0.0),
                reverse=True,
            )[:TODAY_TOP_LIMIT]
        ]
        display_code_set = set(display_codes)
        continuation_codes = [code for code in previous_top3_codes if code]
        merged_display_codes = display_codes + [code for code in continuation_codes if code not in display_code_set]
        states_payload: List[Dict[str, Any]] = []
        for rank, code in enumerate(merged_display_codes, start=1):
            stock = stock_map.get(code)
            recommendation = final_recommendations.get(code, {})
            previous_state = previous_states.get(code)
            previous_front_state = previous_frontlist_map.get(code)
            is_today_top = code in today_top_codes
            is_continuation = code in continuation_codes and not is_today_top
            source_tag = "今日Top3" if is_today_top else ("昨日延续" if is_continuation else WINDOW_RECOMMENDATION_TAG)
            overall_score = float(
                recommendation.get("overall_score")
                or recommendation.get("score")
                or getattr(stock, "score", None)
                or (previous_state or {}).get("priority_score")
                or (previous_front_state or {}).get("priority_score")
                or 0.0
            )
            recommendation_score = float(
                recommendation.get("weighted_score")
                or recommendation.get("recommendation_score")
                or recommendation.get("score")
                or getattr(stock, "recommendation_score", None)
                or (previous_state or {}).get("recommendation_score")
                or (previous_front_state or {}).get("recommendation_score")
                or overall_score
            )
            analysis_confidence = recommendation.get("overall_confidence")
            if analysis_confidence is None:
                analysis_confidence = recommendation.get("ai_confidence")
            if analysis_confidence is None and previous_state is not None:
                analysis_confidence = previous_state.get("ai_confidence")
            if analysis_confidence is None and previous_front_state is not None:
                analysis_confidence = previous_front_state.get("ai_confidence")
            if analysis_confidence is None and stock is not None:
                confidence_label = getattr(stock, "confidence", None)
                confidence_map = {"high": 0.8, "medium": 0.65, "low": 0.5}
                analysis_confidence = confidence_map.get(str(confidence_label).lower())
            display_confidence = round(float(analysis_confidence), 4) if analysis_confidence is not None else None
            ai_confidence = self._apply_repeat_pick_confidence_bonus(display_confidence, is_continuation)
            streaks = self._calculate_streaks(previous_state, True)
            in_frontlist = True
            entered_frontlist = not bool((previous_state or {}).get("in_frontlist"))
            previous_score = previous_score_map.get(code)
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
                "is_repeat_pick": is_continuation,
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
                "summary": recommendation.get("summary") or recommendation.get("ai_summary") or (previous_state or {}).get("summary"),
                "close": getattr(stock, "close", None) or (previous_state or {}).get("close"),
                "pct_change": getattr(stock, "pct_change", None) if stock else (previous_state or {}).get("pct_change"),
                "volume_ratio": getattr(stock, "volume_ratio", None) if stock else (previous_state or {}).get("volume_ratio"),
                "turnover_rate": getattr(stock, "turnover_rate", None) if stock else (previous_state or {}).get("turnover_rate"),
                "strategy_count": strategy_counts.get(code, recommendation.get("strategy_count", (previous_state or {}).get("strategy_count", 0))),
                "news_mentioned": bool(recommendation.get("news_mentioned", False)),
                "industry": recommendation.get("industry") or getattr(stock, "industry", None) or (previous_state or {}).get("industry"),
                "industry_heat_score": recommendation.get("industry_heat_score", (previous_state or {}).get("industry_heat_score")),
                "industry_flow_bias": recommendation.get("industry_flow_bias") or (previous_state or {}).get("industry_flow_bias") or "中性",
                "ai_confidence": ai_confidence,
                "display_confidence": display_confidence,
                "technical_signal": recommendation.get("technical_signal") or (previous_state or {}).get("technical_signal") or getattr(stock, "trend_status", None),
                "recommendation_text": recommendation.get("recommendation") or recommendation.get("ai_summary") or (previous_state or {}).get("recommendation_text") or "",
                "entry_price": getattr(stock, "close", None) or (previous_state or {}).get("entry_price"),
                "recommend_rank": rank,
                "previous_recommendation_score": previous_score,
                "previous_overall_score": (previous_state or {}).get("overall_score") or (previous_front_state or {}).get("priority_score"),
                "previous_confidence": (previous_state or {}).get("display_confidence") or (previous_front_state or {}).get("ai_confidence"),
                "today_present": True,
                "absence_reason": None,
                "action_plan": self._build_action_plan(recommendation, stock, previous_state),
                "review_status": "延续" if is_continuation else ("新入选" if is_today_top else "观察"),
                "yesterday_conclusion": (previous_state or {}).get("recommendation_text"),
                "today_verdict": "延续" if is_continuation else ("新入选" if is_today_top else "观察"),
                "miss_reason_candidates": [],
                "missing_factor_candidates": [],
            }
            states_payload.append(current_item)

        return [TrackedRecommendationState(trade_date=trade_date, **item) for item in states_payload]

    @staticmethod
    def _build_action_plan(recommendation: Dict[str, Any], stock: Any, previous_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        close = getattr(stock, "close", None) or (previous_state or {}).get("close")
        entry_value = getattr(stock, "close", None) or (previous_state or {}).get("entry_price")
        signal = str(recommendation.get("technical_signal") or "").strip()
        return {
            "action_bias": "观察" if not recommendation.get("recommendation") else "买入" if "买" in str(recommendation.get("recommendation")) else "观察",
            "entry_zone": f"{entry_value:.2f} 附近观察" if isinstance(entry_value, (int, float)) else "等待回踩或放量确认",
            "take_profit": f"{close * 1.05:.2f} 附近分批止盈" if isinstance(close, (int, float)) else "结合盘中强弱分批止盈",
            "stop_loss": f"{close * 0.97:.2f}" if isinstance(close, (int, float)) else "跌破关键支撑止损",
            "holding_horizon": "1-5个交易日",
            "invalid_condition": signal or "量价结构走弱",
        }

    def _build_report_context(
        self,
        *,
        trade_date: date,
        pool_states: List[Dict[str, Any]],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        state_map = {item.get("ts_code"): item for item in pool_states if item.get("ts_code")}
        today_top3 = [self._build_report_stock_payload(item, ai_analyses, final_recommendations) for item in pool_states if item.get("source_tag") == "今日Top3"]
        previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(trade_date)
        previous_states = {
            item.get("ts_code"): item
            for item in self.store.load_recommendation_pool_state(trade_date=previous_trade_date)
            if previous_trade_date and item.get("ts_code")
        }
        previous_top3 = [
            item for item in previous_states.values() if item.get("source_tag") == "今日Top3"
        ]
        previous_top3 = sorted(previous_top3, key=lambda item: int(item.get("recommend_rank") or 9999))[:TODAY_TOP_LIMIT]

        yesterday_top3_review: List[Dict[str, Any]] = []
        for previous_item in previous_top3:
            code = previous_item.get("ts_code")
            current_item = state_map.get(code)
            if current_item:
                review = self._build_report_stock_payload(current_item, ai_analyses, final_recommendations)
                review.update(
                    {
                        "yesterday_conclusion": previous_item.get("recommendation_text") or previous_item.get("summary") or "昨日入选 Top3",
                        "today_verdict": "延续" if current_item.get("source_tag") in {"今日Top3", "昨日延续", WINDOW_RECOMMENDATION_TAG} else "观察",
                        "review_status": "延续" if current_item.get("source_tag") in {"今日Top3", "昨日延续"} else "观察",
                        "miss_reason_candidates": [],
                        "missing_factor_candidates": [],
                    }
                )
                yesterday_top3_review.append(review)
                continue
            yesterday_top3_review.append(
                {
                    "ts_code": code,
                    "name": previous_item.get("name") or code,
                    "recommendation_score": None,
                    "overall_score": None,
                    "display_confidence": None,
                    "source_tag": "昨日复盘",
                    "today_present": False,
                    "absence_reason": "今日未进入候选池或展示池",
                    "yesterday_conclusion": previous_item.get("recommendation_text") or previous_item.get("summary") or "昨日入选 Top3",
                    "today_verdict": "失效",
                    "review_status": "失效",
                    "previous_recommendation_score": previous_item.get("recommendation_score"),
                    "previous_overall_score": previous_item.get("overall_score") or previous_item.get("priority_score"),
                    "previous_confidence": previous_item.get("display_confidence") or previous_item.get("ai_confidence"),
                    "strategy_count": 0,
                    "news_mentioned": False,
                    "technical_signal": None,
                    "summary": previous_item.get("summary") or previous_item.get("recommendation_text"),
                    "analysis": "今日未能延续到候选池，需要复盘失效原因。",
                    "miss_reason_candidates": ["评分因子缺口", "技术趋势破坏", "资金承接不足"],
                    "missing_factor_candidates": ["盘中承接强度", "热点持续性", "量能变化"],
                    "action_plan": {},
                }
            )

        comparison_candidates_map: Dict[str, Dict[str, Any]] = {}
        for item in today_top3 + yesterday_top3_review:
            code = item.get("ts_code")
            if code and code not in comparison_candidates_map:
                comparison_candidates_map[code] = item

        return {
            "trade_date": trade_date.isoformat(),
            "today_top3": today_top3,
            "yesterday_top3_review": yesterday_top3_review,
            "comparison_candidates": list(comparison_candidates_map.values()),
        }

    def _build_report_stock_payload(
        self,
        item: Dict[str, Any],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        code = item.get("ts_code")
        analysis = ai_analyses.get(code, {})
        recommendation = final_recommendations.get(code, {})
        return {
            "ts_code": code,
            "name": item.get("name") or analysis.get("name") or code,
            "source_tag": item.get("source_tag"),
            "recommendation_score": item.get("recommendation_score", recommendation.get("weighted_score")),
            "overall_score": item.get("overall_score", analysis.get("overall_score", recommendation.get("overall_score"))),
            "priority_score": item.get("priority_score"),
            "overall_confidence": analysis.get("overall_confidence", item.get("ai_confidence")),
            "display_confidence": item.get("display_confidence", analysis.get("confidence")),
            "strategy_count": item.get("strategy_count", recommendation.get("strategy_count", 0)),
            "news_mentioned": item.get("news_mentioned", recommendation.get("news_mentioned", False)),
            "industry": item.get("industry", recommendation.get("industry")),
            "industry_heat_score": item.get("industry_heat_score", recommendation.get("industry_heat_score")),
            "industry_flow_bias": item.get("industry_flow_bias", recommendation.get("industry_flow_bias", "中性")),
            "score_change": item.get("score_change"),
            "previous_recommendation_score": item.get("previous_recommendation_score"),
            "technical_signal": item.get("technical_signal") or analysis.get("technical_signal"),
            "technical_score": item.get("technical_score") if item.get("technical_score") is not None else analysis.get("technical_score"),
            "fundamental_score": item.get("fundamental_score") if item.get("fundamental_score") is not None else analysis.get("fundamental_score"),
            "sentiment_score": item.get("sentiment_score") if item.get("sentiment_score") is not None else analysis.get("sentiment_score"),
            "news_score": item.get("news_score") if item.get("news_score") is not None else analysis.get("news_score"),
            "summary": item.get("summary") or analysis.get("summary"),
            "recommendation_text": item.get("recommendation_text") or recommendation.get("recommendation") or analysis.get("recommendation"),
            "close": item.get("close"),
            "entry_price": item.get("entry_price"),
            "action_plan": item.get("action_plan") or self._build_action_plan(recommendation, None, item),
            "today_present": item.get("today_present", True),
            "review_status": item.get("review_status"),
            "today_verdict": item.get("today_verdict"),
            "yesterday_conclusion": item.get("yesterday_conclusion"),
            "miss_reason_candidates": list(item.get("miss_reason_candidates") or []),
            "missing_factor_candidates": list(item.get("missing_factor_candidates") or []),
            "absence_reason": item.get("absence_reason"),
            "previous_overall_score": item.get("previous_overall_score"),
            "previous_confidence": item.get("previous_confidence"),
        }

    def _get_active_strategies(self) -> List[ScreenPreset]:
        """获取活跃策略"""
        strategy_ids = self.settings.screening_strategies

        if not strategy_ids:
            return StockScreener.get_presets()

        all_presets = StockScreener.get_presets()
        return [
            preset for preset in all_presets
            if preset.id in strategy_ids
        ]