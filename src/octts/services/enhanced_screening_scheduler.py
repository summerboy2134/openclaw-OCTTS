"""Enhanced stock screening with AI analysis integration."""

import html
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from octts.clients.email_client import EmailClient
from octts.clients.wecom_client import WeComClient
from octts.config import Settings
from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.services.history_store import FileHistoryStore
from octts.services.stock_screener import StockScreener
from octts.services.screening_store import ScreeningStore
from octts.services.position_store import create_position_store
from octts.services.market_data_sync_service import MarketDataSyncService
from octts.services.report_exporter import ReportExporter
from octts.services.multi_dimensional_analyzer import MultiDimensionalAnalyzer
from octts.services.news_aggregator import NewsAggregator
from octts.services.intelligent_report_generator import IntelligentReportGenerator
from octts.services.intelligent_screening_job_manager import maybe_await_progress_callback
from octts.services.report_email_service import ReportEmailService
from octts.services.screening_validator import ScreeningValidator
from octts.services.short_term_training_data import ShortTermTrainingDataBuilder
from octts.services.regression_rerank_service import RegressionRerankResult, RegressionRerankService
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_constants import *
from octts.services.enhanced_screening_market_context import EnhancedScreeningMarketContextMixin
from octts.services.enhanced_screening_recommendations import EnhancedScreeningRecommendationsMixin
from octts.services.enhanced_screening_report_context import EnhancedScreeningReportContextMixin
from octts.services.enhanced_screening_risk import EnhancedScreeningRiskMixin
from octts.services.enhanced_screening_stage_pipeline import EnhancedScreeningStagePipelineMixin


logger = logging.getLogger(__name__)


class EnhancedScreeningScheduler(
    EnhancedScreeningMarketContextMixin,
    EnhancedScreeningRecommendationsMixin,
    EnhancedScreeningReportContextMixin,
    EnhancedScreeningRiskMixin,
    EnhancedScreeningStagePipelineMixin,
):
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
        self.news_aggregator = news_aggregator or NewsAggregator(settings, progress_callback=progress_callback)
        self.report_generator = report_generator or IntelligentReportGenerator(
            settings,
            analyzer=self.analyzer
        )
        self.validator = ScreeningValidator(settings)
        self.training_data_builder = ShortTermTrainingDataBuilder(settings, store=self.store)
        self.regression_rerank_service = RegressionRerankService(settings)
        self.market_raw_data_repo = MarketRawDataRepository(settings.database_url)
        self.market_data_sync_service = MarketDataSyncService(settings)
        self._stock_name_cache: Optional[Dict[str, str]] = None
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
        llm_enabled = self.settings.screening_llm_enabled
        news_items = []
        news_clusters = []
        self._latest_news_clusters = []

        if llm_enabled:
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
            await self._report_progress(
                current_step=1,
                total_steps=total_steps,
                step_name="新闻采集",
                progress_percent=14,
                message="正在分析新闻重要性...",
                details={"news_count": len(news_items), "phase": "importance_analysis"},
            )
            news_items = await self.news_aggregator.analyze_importance(news_items)
            logger.info("Step 1.2 complete")
            await self._report_progress(
                current_step=1,
                total_steps=total_steps,
                step_name="新闻采集",
                progress_percent=18,
                message="新闻重要性分析完成，正在进行主题聚类...",
                details={"news_count": len(news_items), "phase": "clustering"},
            )
            logger.info("Step 1.3: Starting news clustering for %s items", len(news_items))
            news_clusters = await self.news_aggregator.cluster_news(news_items)
            self._latest_news_clusters = news_clusters
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
        else:
            logger.info("Step 1 skipped: news collection and LLM analysis disabled by OCTTS_SCREENING_LLM_ENABLED")
            await self._report_progress(
                current_step=1,
                total_steps=total_steps,
                step_name="新闻采集",
                progress_percent=20,
                message="已跳过新闻与 LLM 分析，直接测试基础打分流程。",
                details={"llm_enabled": False},
            )

        # 2. 解析最新交易日并确保模型特征所需的当日原始数据入库
        logger.info("Step 2: Resolving latest trade date and ensuring local DB data")
        await self._report_progress(
            current_step=2,
            total_steps=total_steps,
            step_name="数据准备",
            progress_percent=28,
            message="正在确认最新交易日，并检查本地数据库当日行情数据...",
        )
        trade_date = self.screener._get_latest_trade_date()
        try:
            ensure_result = self.market_data_sync_service.ensure_trade_date_data(trade_date=trade_date)
            logger.info("Pre-model market data ensure: %s", ensure_result)
        except Exception:
            logger.exception("Failed to ensure local raw market data before model ranking: trade_date=%s", trade_date)
        await self._report_progress(
            current_step=2,
            total_steps=total_steps,
            step_name="数据准备",
            progress_percent=38,
            message=f"本地数据库已准备，正在直接构建模型特征并生成 Top{MODEL_RISK_REVIEW_POOL_LIMIT} 风险复核池...",
            details={"strategy_count": 0, "legacy_strategy_recall": False},
        )

        # 3. 模型先保留 Top200 收益排序优势，再在后续阶段做可执行性复核与 Top50/Top3 选择
        logger.info("Step 3: Select model Top%s risk-review pool and derive Top3 with extreme-risk veto", MODEL_RISK_REVIEW_POOL_LIMIT)
        candidate_limit = max(MODEL_RISK_REVIEW_POOL_LIMIT, MODEL_CANDIDATE_POOL_LIMIT, self.settings.screening_top_n)
        trade_day = datetime.strptime(trade_date, "%Y%m%d").date()
        rerank_result = self.regression_rerank_service.rank_market_universe(
            trade_date=trade_day,
            candidate_limit=candidate_limit,
            analysis_limit=TOP_RECOMMENDATION_LIMIT,
            exclude_bj=V2_EXCLUDE_BJ,
        )
        if len(rerank_result.candidate_codes) < MODEL_CANDIDATE_POOL_LIMIT:
            error_detail = rerank_result.error_message or "unknown"
            if rerank_result.fallback_reason:
                error_detail = f"{error_detail}; fallback_reason={rerank_result.fallback_reason}"
            raise RuntimeError(
                "Model universe ranking produced too few candidates: "
                f"candidates={len(rerank_result.candidate_codes)}, "
                f"required={MODEL_CANDIDATE_POOL_LIMIT}, "
                f"fallback_reason={rerank_result.fallback_reason}, "
                f"error={error_detail}"
            )
        snapshot_started_at = time.perf_counter()
        market_snapshot = self._build_lightweight_market_snapshot(trade_date)
        market_snapshot = self._hydrate_snapshot_daily_history_for_codes(
            market_snapshot,
            trade_date=trade_date,
            stock_codes=rerank_result.candidate_codes[:MODEL_RISK_REVIEW_POOL_LIMIT],
        )
        logger.info(
            "Step 3.1 complete: lightweight report snapshot ready in %.2fs, cache_status=%s, trade_date=%s, stocks=%s, basic=%s, cached_daily=%s",
            time.perf_counter() - snapshot_started_at,
            self._describe_snapshot_cache_status(market_snapshot, trade_date),
            trade_date,
            len(market_snapshot.get("stocks", [])) if isinstance(market_snapshot, dict) else 0,
            len(market_snapshot.get("daily_basic", {})) if isinstance(market_snapshot, dict) else 0,
            len(market_snapshot.get("daily", {})) if isinstance(market_snapshot, dict) else 0,
        )
        screening_results = self._build_model_candidate_screening_results(
            trade_date=trade_day,
            rerank_result=rerank_result,
            market_snapshot=market_snapshot,
        )
        await self.store.save_screening_result("model_top100", screening_results["model_top100"])
        baseline_candidate_codes = list(rerank_result.candidate_codes)
        candidate_codes = list(rerank_result.candidate_codes)
        eligible_candidate_codes = self._filter_out_tracked_and_holding_codes(candidate_codes)
        await self._report_progress(
            current_step=3,
            total_steps=total_steps,
            step_name=f"模型Top{MODEL_RISK_REVIEW_POOL_LIMIT}风险复核",
            progress_percent=42,
            message=f"正在基于预测模型生成 Top{MODEL_RISK_REVIEW_POOL_LIMIT} 风险复核池，并筛选 Top{MODEL_CANDIDATE_POOL_LIMIT}/Top3...",
            details={"candidate_items": len(eligible_candidate_codes)},
        )
        stage_pipeline = self._build_stage_pipeline_result(
            trade_date=trade_day,
            screening_results=screening_results,
            market_snapshot=market_snapshot,
            rerank_result=rerank_result,
            baseline_candidate_codes=baseline_candidate_codes,
        )
        eligible_candidate_codes = stage_pipeline["stage1_candidate_codes"]
        structured_analyses = stage_pipeline["structured_analyses"]
        structured_recommendations = stage_pipeline["stage2_recommendations"]
        final_recommendations = stage_pipeline["final_recommendations"]
        stage2_top20_codes = stage_pipeline["stage2_top20_codes"]
        stage3_top3_codes = stage_pipeline["stage3_top3_codes"]
        analysis_target_codes = stage_pipeline["analysis_target_codes"]
        if self.settings.intelligent_screening_overwrite_same_trade_date:
            clear_summary = self.store.clear_intelligent_screening_results_for_trade_date(trade_day)
            logger.info(
                "Same-trade-date intelligent screening overwrite enabled: trade_date=%s, cleared=%s",
                trade_day.isoformat(),
                clear_summary,
            )
        pre_llm_pool_states = self._build_recommendation_pool_states(
            trade_date=trade_day,
            screening_results=screening_results,
            final_recommendations=final_recommendations,
            candidate_codes=eligible_candidate_codes,
            rerank_metadata=rerank_result.metadata_by_code,
        )
        self.store.upsert_recommendation_pool_states(pre_llm_pool_states)
        focus_context_by_code = self._build_focus_stock_llm_contexts(
            stock_codes=analysis_target_codes,
            screening_results=screening_results,
            final_recommendations=final_recommendations,
            market_snapshot=market_snapshot,
            rerank_metadata=rerank_result.metadata_by_code,
        )
        await self._report_progress(
            current_step=3,
            total_steps=total_steps,
            step_name="AI 深度分析",
            progress_percent=56,
            message=(
                "已完成三阶段筛选，正在对阶段三 Top3 与昨日 Top3 复盘集合做重点分析..."
                if llm_enabled
                else "已完成三阶段筛选，跳过 AI 深度分析，继续基础打分流程..."
            ),
            details={"total_items": len(analysis_target_codes), "completed_items": 0, "llm_enabled": llm_enabled},
        )
        ai_analyses = await self._analyze_top_stocks(
            analysis_target_codes,
            total_steps=total_steps,
            current_step=3,
            focus_context_by_code=focus_context_by_code,
        ) if llm_enabled else {}

        # 4. 结合新闻和技术面筛选
        logger.info("Step 4: Combining news and technical analysis")
        step4_started_at = time.perf_counter()
        await self._report_progress(
            current_step=4,
            total_steps=total_steps,
            step_name="融合打分",
            progress_percent=78,
            message="正在融合新闻、技术和 Top3 AI 分析结果...",
            details={"analyzed_items": len(ai_analyses)},
        )
        combine_started_at = time.perf_counter()
        final_recommendations = self._build_structured_stage_selection_metadata(
            recommendations=dict(final_recommendations),
            rerank_metadata=rerank_result.metadata_by_code,
            stage2_top20_codes=stage2_top20_codes,
            stage3_top3_codes=stage3_top3_codes,
        )
        if llm_enabled and analysis_target_codes:
            if not ai_analyses:
                ai_analyses = self._build_placeholder_ai_analyses(
                    analysis_target_codes,
                    screening_results=screening_results,
                    structured_recommendations=final_recommendations,
                )
            llm_recommendations = await self._combine_analysis(
                screening_results,
                ai_analyses,
                news_clusters,
                market_snapshot=market_snapshot,
                trade_date=trade_date,
            )
            final_recommendations = self._merge_recommendation_payloads(
                base_recommendations=final_recommendations,
                llm_recommendations=llm_recommendations,
            )
        final_recommendations = dict(
            sorted(
                final_recommendations.items(),
                key=lambda item: item[1].get("score", 0.0),
                reverse=True,
            )
        )
        logger.info(
            "Step 4.2 complete: combine analysis finished in %.2fs, recommendations=%s, stage2_top20=%s, stage3_top3=%s",
            time.perf_counter() - combine_started_at,
            len(final_recommendations),
            len(stage2_top20_codes),
            len(stage3_top3_codes),
        )
        logger.info("Step 4 complete: total %.2fs", time.perf_counter() - step4_started_at)
        pool_states = self._build_recommendation_pool_states(
            trade_date=datetime.strptime(trade_date, "%Y%m%d").date(),
            screening_results=screening_results,
            final_recommendations=final_recommendations,
            candidate_codes=eligible_candidate_codes,
            rerank_metadata=rerank_result.metadata_by_code,
        )
        persisted_pool_states = self.store.upsert_recommendation_pool_states(pool_states)

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
        market_data = await self._get_market_data(trade_date=trade_date)
        await self._report_progress(
            current_step=5,
            total_steps=total_steps,
            step_name="生成报告",
            progress_percent=90,
            message="市场数据已准备，正在组装报告上下文...",
            details={"final_recommendations": len(final_recommendations), "phase": "build_context"},
        )
        report_context = self._build_report_context(
            trade_date=datetime.strptime(trade_date, "%Y%m%d").date(),
            pool_states=persisted_pool_states or self.store.list_recommendation_pool(
                trade_date=datetime.strptime(trade_date, "%Y%m%d").date()
            ),
            ai_analyses=ai_analyses,
            final_recommendations=final_recommendations,
        )
        report_context.update(
            {
                "rerank_fallback_reason": rerank_result.fallback_reason,
                "rerank_ranking_trade_date": rerank_result.ranking_trade_date.isoformat() if rerank_result.ranking_trade_date else None,
                "rerank_requested_trade_date": trade_day.isoformat(),
                "rerank_uses_fallback_trade_date": bool(
                    rerank_result.ranking_trade_date and rerank_result.ranking_trade_date != trade_day
                ),
            }
        )
        report_context.update(
            await self.news_aggregator.collect_focus_stock_live_context(
                today_top3=report_context.get("today_top3") or [],
                yesterday_top3_review=report_context.get("yesterday_top3_review") or [],
            )
        )
        await self._report_progress(
            current_step=5,
            total_steps=total_steps,
            step_name="生成报告",
            progress_percent=92,
            message=(
                "正在调用模型生成报告正文..."
                if llm_enabled
                else "已跳过报告模型生成，使用结构化兜底报告输出。"
            ),
            details={
                "final_recommendations": len(final_recommendations),
                "phase": "llm_report_generation" if llm_enabled else "fallback_report_generation",
                "llm_enabled": llm_enabled,
            },
        )
        report = await self.report_generator.generate_morning_report(
            news_clusters=news_clusters,
            market_data=market_data,
            stock_pool=[item.ts_code for item in pool_states],
            screening_context=report_context,
        )
        stage_backtest_payload = self._build_stage_backtest_payload(
            stage1_candidate_codes=stage_pipeline["stage1_candidate_codes"],
            stage2_top20_codes=stage_pipeline["stage2_top20_codes"],
            stage3_top3_codes=stage_pipeline["stage3_top3_codes"],
            rerank_result=rerank_result,
        )
        stage_backtest_path = self._save_stage_backtest_snapshot(
            trade_date=trade_date,
            payload=stage_backtest_payload,
        )
        self._save_recommendation_run(
            trade_date=trade_date,
            screening_results=screening_results,
            ai_analyses=ai_analyses,
            final_recommendations=final_recommendations,
            report_id=getattr(report, "report_id", None),
            rerank_metadata=rerank_result.metadata_by_code,
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
        try:
            self.training_data_builder.persist_samples_for_trade_date(
                datetime.strptime(trade_date, "%Y%m%d").date()
            )
        except Exception:
            logger.exception("Failed to persist short-term training samples for trade_date=%s", trade_date)
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
            "stage1_candidate_count": len(stage_pipeline["stage1_candidate_codes"]),
            "stage2_top20_count": len(stage_pipeline["stage2_top20_codes"]),
            "stage3_top3_count": len(stage_pipeline["stage3_top3_codes"]),
            "stage1_candidate_codes": list(stage_pipeline["stage1_candidate_codes"]),
            "stage2_top20_codes": list(stage_pipeline["stage2_top20_codes"]),
            "stage3_top3_codes": list(stage_pipeline["stage3_top3_codes"]),
            "stage_backtest_path": str(stage_backtest_path) if stage_backtest_path else None,
            "frontlist_count": len([state for state in pool_states if state.in_frontlist]),
            "tracking_pool_count": len(pool_states),
            "today_top_count": len([state for state in pool_states if state.source_tag == "今日Top3"]),
            "continuation_count": len([state for state in pool_states if state.source_tag == "昨日延续"]),
            "rerank_candidate_count": len(rerank_result.candidate_codes),
            "rerank_analysis_count": len(rerank_result.analysis_codes),
            "rerank_fallback_reason": rerank_result.fallback_reason,
            "rerank_ranking_trade_date": rerank_result.ranking_trade_date.isoformat() if rerank_result.ranking_trade_date else None,
            "rerank_requested_trade_date": trade_day.isoformat(),
            "rerank_uses_fallback_trade_date": bool(
                rerank_result.ranking_trade_date and rerank_result.ranking_trade_date != trade_day
            ),
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
        try:
            ensure_result = self.market_data_sync_service.ensure_trade_date_data(trade_date=trade_date)
            logger.info("Pre-screening market data ensure: %s", ensure_result)
        except Exception:
            logger.exception("Failed to ensure local raw market data before screening: trade_date=%s", trade_date)
        market_snapshot = self.screener.client.get_or_build_screening_snapshot(trade_date)

        logger.info("Technical screening: %s strategies queued, shared trade_date=%s", len(strategies), trade_date)
        screen_result_limit = max(V2_COARSE_POOL_LIMIT, self.settings.screening_top_n, TOP_RECOMMENDATION_LIMIT)
        for index, strategy in enumerate(strategies, start=1):
            logger.info(
                "Technical screening strategy %s/%s start: %s (%s), result_limit=%s",
                index,
                len(strategies),
                strategy.name,
                strategy.id,
                screen_result_limit,
            )
            try:
                criteria = strategy.criteria.model_copy(update={"limit": screen_result_limit, "offset": 0})
                result = self.screener.screen(
                    criteria,
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

    def _run_screening_strategies_sync_for_backfill(
        self,
        trade_date: str,
        *,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ScreenResult]:
        strategies = self._get_active_strategies()
        results: Dict[str, ScreenResult] = {}

        logger.info(
            "Backfill technical screening start: trade_date=%s, strategies=%s",
            trade_date,
            len(strategies),
        )
        screen_result_limit = max(BACKFILL_TRAINING_CANDIDATE_LIMIT, V2_COARSE_POOL_LIMIT, self.settings.screening_top_n)
        for index, strategy in enumerate(strategies, start=1):
            try:
                criteria = strategy.criteria.model_copy(update={"limit": screen_result_limit, "offset": 0})
                result = self.screener.screen(
                    criteria,
                    trade_date=trade_date,
                    market_snapshot=market_snapshot,
                )
                results[strategy.id] = result
                logger.info(
                    "Backfill technical screening strategy %s/%s complete: %s matched %s stocks",
                    index,
                    len(strategies),
                    strategy.id,
                    result.total_count,
                )
            except Exception:
                logger.exception(
                    "Backfill technical screening strategy %s/%s failed: %s",
                    index,
                    len(strategies),
                    strategy.id,
                )
        logger.info(
            "Backfill technical screening complete: trade_date=%s, strategies=%s",
            trade_date,
            len(results),
        )
        return results

    async def _analyze_top_stocks(
        self,
        stock_codes: List[str],
        *,
        total_steps: int,
        current_step: int,
        focus_context_by_code: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """仅对最终 Top3 与昨日 Top3 复盘集合做重点分析"""
        analyses = {}
        total_items = len(stock_codes)

        if not stock_codes:
            await self._report_progress(
                current_step=current_step,
                total_steps=total_steps,
                step_name="AI 深度分析",
                progress_percent=76,
                message="本轮无最终 Top3 / 昨日 Top3 复盘股票可供分析。",
                details={"total_items": 0, "completed_items": 0},
            )
            return analyses

        news_context = self._build_news_score_context(getattr(self, "_latest_news_clusters", []) or [])
        focus_context_by_code = focus_context_by_code or {}

        logger.info("Starting sequential analysis for %s focus stocks", total_items)

        completed = 0
        for index, code in enumerate(stock_codes, start=1):
            progress_percent = 42 + int((index - 1) * 34 / total_items)
            logger.info("Starting multi-dimensional analysis for %s (%s/%s)", code, index, total_items)
            await self._report_progress(
                current_step=current_step,
                total_steps=total_steps,
                step_name="AI 深度分析",
                progress_percent=progress_percent,
                message=f"正在重点分析股票 {code} ({index}/{total_items})...",
                details={
                    "current_symbol": code,
                    "completed_items": completed,
                    "total_items": total_items,
                },
            )
            try:
                analysis = await self.analyzer.analyze(
                    code,
                    enable_iterations=True,
                    max_iterations=2,
                    news_context={
                        **news_context,
                        "focus_stock_context": focus_context_by_code.get(code, {}),
                        "focus_stock_contexts": focus_context_by_code,
                    },
                )
                analyses[code] = analysis
                logger.info(
                    "AI analysis result for %s: keys=%s, overall_score=%s, overall_confidence=%s, summary=%s, recommendation=%s, technical_score=%s, fundamental_score=%s, sentiment_score=%s, news_score=%s",
                    code,
                    sorted(list(analysis.keys())) if isinstance(analysis, dict) else type(analysis).__name__,
                    analysis.get("overall_score") if isinstance(analysis, dict) else None,
                    analysis.get("overall_confidence") if isinstance(analysis, dict) else None,
                    (str(analysis.get("summary"))[:200] if isinstance(analysis, dict) and analysis.get("summary") is not None else None),
                    (str(analysis.get("recommendation"))[:200] if isinstance(analysis, dict) and analysis.get("recommendation") is not None else None),
                    analysis.get("technical_score") if isinstance(analysis, dict) else None,
                    analysis.get("fundamental_score") if isinstance(analysis, dict) else None,
                    analysis.get("sentiment_score") if isinstance(analysis, dict) else None,
                    analysis.get("news_score") if isinstance(analysis, dict) else None,
                )
                logger.info("Successfully analyzed %s", code)
            except Exception as e:
                logger.error("Failed to analyze %s: %s", code, e, exc_info=True)

            completed += 1
            progress_percent = 42 + int(completed * 34 / total_items)
            await self._report_progress(
                current_step=current_step,
                total_steps=total_steps,
                step_name="AI 深度分析",
                progress_percent=progress_percent,
                message=f"Top3 重点分析进度：{completed}/{total_items}",
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
        trade_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        结合多维度分析生成最终推荐

        Returns:
            {股票代码: {推荐信息}}
        """
        started_at = time.perf_counter()
        final_recommendations = {}
        stock_map = self._build_screened_stock_map(screening_results)
        all_market_stock_map = self._build_all_market_stock_map(market_snapshot)
        logger.info(
            "Step 4 combine start: candidate_stocks=%s, ai_analyses=%s, news_clusters=%s, all_market_stocks=%s",
            len(stock_map),
            len(ai_analyses),
            len(news_clusters),
            len(all_market_stock_map),
        )
        industry_adjustments = self._build_industry_flow_adjustments(
            stock_map,
            all_market_stock_map=all_market_stock_map,
            market_snapshot=market_snapshot,
            trade_date=trade_date,
        )
        distribution_risk_map = self._build_distribution_risk_map(
            stock_map,
            market_snapshot=market_snapshot,
            trade_date=trade_date,
        )
        logger.info(
            "Step 4 risk map built: total=%s, with_score=%s, blocked=%s, sample=%s",
            len(distribution_risk_map),
            sum(1 for item in distribution_risk_map.values() if item.get("distribution_risk_score") is not None),
            sum(1 for item in distribution_risk_map.values() if item.get("candidate_risk_blocked")),
            [
                {
                    "code": risk_code,
                    "score": risk_payload.get("distribution_risk_score"),
                    "moneyflow_3d_value": risk_payload.get("moneyflow_3d_value"),
                    "recent_runup_5d": risk_payload.get("recent_runup_5d"),
                    "turnover_spike_ratio": risk_payload.get("turnover_spike_ratio"),
                    "blocked": risk_payload.get("candidate_risk_blocked"),
                }
                for risk_code, risk_payload in list(distribution_risk_map.items())[:5]
            ],
        )
        debug_codes = ["603182.SH", "688618.SH", "300692.SZ"]
        logger.info(
            "Step 4 risk map debug: %s",
            {
                code: distribution_risk_map.get(code)
                for code in debug_codes
                if code in distribution_risk_map
            },
        )
        previous_states: Dict[str, Dict[str, Any]] = {}
        previous_top3_codes = set()
        try:
            current_trade_date = datetime.strptime((trade_date or ""), "%Y%m%d").date() if trade_date else datetime.strptime(self.screener._get_latest_trade_date(), "%Y%m%d").date()
            previous_trade_date = self.store.get_previous_recommendation_pool_trade_date(current_trade_date)
            if previous_trade_date:
                previous_states = {
                    item.get("ts_code"): item
                    for item in self.store.load_recommendation_pool_state(trade_date=previous_trade_date)
                    if item.get("ts_code")
                }
                previous_top3_codes = set(self._get_previous_top3_codes(current_trade_date))
        except Exception:
            previous_states = {}
            previous_top3_codes = set()

        # 从新闻中提取热点股票
        news_hot_stocks = self._extract_news_hot_stocks(news_clusters)
        theme_support_map = self._build_theme_support_map(
            stock_map,
            news_clusters=news_clusters,
            news_hot_stocks=news_hot_stocks,
            industry_adjustments=industry_adjustments,
            distribution_risk_map=distribution_risk_map,
            screening_results=screening_results,
        )

        # 遍历AI分析结果
        for code, analysis in ai_analyses.items():
            stock_item = stock_map.get(code)
            confidence = analysis.get("overall_confidence", 0)
            score = analysis.get("overall_score", 50)
            if code in {"002269.SZ", "301226.SZ", "600222.SH"}:
                logger.info(
                    "Step 4 combine candidate %s: has_real_ai=%s, analysis_keys=%s, overall_score=%s, base_score=%s, technical_score=%s, distribution_risk=%s",
                    code,
                    self._has_real_ai_overall_score(analysis),
                    sorted(list(analysis.keys())) if isinstance(analysis, dict) else [],
                    analysis.get("overall_score") if isinstance(analysis, dict) else None,
                    analysis.get("base_score") if isinstance(analysis, dict) else None,
                    analysis.get("technical_score") if isinstance(analysis, dict) else None,
                    distribution_risk_map.get(code),
                )

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
            distribution_risk = distribution_risk_map.get(code, {})
            theme_support = theme_support_map.get(code, {})
            distribution_risk_score = float(distribution_risk.get("distribution_risk_score") or 0.0)
            extra_risk_penalty = 0.0
            if distribution_risk.get("latest_weakening_flag"):
                extra_risk_penalty += 2.0
            if distribution_risk.get("high_level_pullback_flag"):
                extra_risk_penalty += 2.2
            if distribution_risk.get("theme_support_absent_flag"):
                extra_risk_penalty += 1.2
            if theme_support.get("unsupported_high_position_flag"):
                extra_risk_penalty += UNSUPPORTED_HIGH_POSITION_EXTRA_PENALTY
            if theme_support.get("leader_turnover_justified_flag"):
                extra_risk_penalty -= JUSTIFIED_TURNOVER_RELIEF
            ranking_risk_penalty = round(max(0.0, distribution_risk_score * TOP10_RISK_PENALTY_MULTIPLIER + extra_risk_penalty), 2)
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
            fundamental_bonus = self._build_light_fundamental_bonus(analysis)
            adjusted_final_score = final_score + industry_heat_score + fundamental_bonus["total_bonus"] - ranking_risk_penalty
            weighted_score = (adjusted_final_score + continuation_bias_score) * confidence
            threshold_reason_flags: List[str] = []
            score_gate_pass = (weighted_score >= 55 or adjusted_final_score >= 60 or final_score >= 65)
            if distribution_risk_score >= TOP10_MAX_DISTRIBUTION_RISK_SCORE:
                threshold_reason_flags.append("distribution_risk_score_too_high")
            if bool(theme_support.get("unsupported_high_position_flag")):
                threshold_reason_flags.append("unsupported_high_position")
            if not score_gate_pass:
                threshold_reason_flags.append("score_gate_not_met")
            if weighted_score < 55:
                threshold_reason_flags.append("weighted_score_below_55")
            if adjusted_final_score < 60:
                threshold_reason_flags.append("adjusted_final_score_below_60")
            if final_score < 65:
                threshold_reason_flags.append("final_score_below_65")
            passes_threshold = (
                distribution_risk_score < TOP10_MAX_DISTRIBUTION_RISK_SCORE
                and not bool(theme_support.get("unsupported_high_position_flag"))
                and score_gate_pass
            )

            if passes_threshold:  # 避免低置信度将中高分标的全部压没
                final_recommendations[code] = {
                    "score": weighted_score,
                    "overall_score": score,
                    "final_score": final_score,
                    "adjusted_final_score": adjusted_final_score,
                    "weighted_score": weighted_score,
                    "ranking_risk_penalty": ranking_risk_penalty,
                    "fundamental_bonus": fundamental_bonus["total_bonus"],
                    "fundamental_bonus_breakdown": fundamental_bonus["breakdown"],
                    "continuation_bias_score": continuation_bias_score,
                    "continuation_positive_flags": continuation_positive_flags,
                    "continuation_negative_flags": continuation_negative_flags,
                    "ai_confidence": confidence,
                    "ai_summary": analysis.get("summary", ""),
                    "technical_signal": analysis.get("technical_signal", ""),
                    "technical_score": analysis.get("technical_score"),
                    "fundamental_score": analysis.get("fundamental_score"),
                    "sentiment_score": analysis.get("sentiment_score"),
                    "news_score": analysis.get("news_score"),
                    "base_score": analysis.get("base_score"),
                    "sentiment_adjustment": analysis.get("sentiment_adjustment"),
                    "news_adjustment": analysis.get("news_adjustment"),
                    "score_model": analysis.get("score_model"),
                    "close": getattr(stock_item, "close", None),
                    "pct_change": getattr(stock_item, "pct_change", None),
                    "volume_ratio": getattr(stock_item, "volume_ratio", None),
                    "turnover_rate": getattr(stock_item, "turnover_rate", None),
                    "ma20": getattr(stock_item, "ma20", None),
                    "price_position_20d": getattr(stock_item, "price_position_20d", None),
                    "news_mentioned": code in news_hot_stocks,
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
                    "moneyflow_3d_value": distribution_risk.get("moneyflow_3d_value"),
                    "turnover_spike_ratio": distribution_risk.get("turnover_spike_ratio"),
                    "recent_runup_5d": distribution_risk.get("recent_runup_5d"),
                    "late_stage_momentum_flag": bool(distribution_risk.get("late_stage_momentum_flag", False)),
                    "latest_weakening_flag": bool(distribution_risk.get("latest_weakening_flag", False)),
                    "high_level_pullback_flag": bool(distribution_risk.get("high_level_pullback_flag", False)),
                    "theme_support_absent_flag": bool(distribution_risk.get("theme_support_absent_flag", False)),
                    "candidate_risk_blocked": bool(distribution_risk.get("candidate_risk_blocked", False)),
                    "action_bias": None,
                    "recommendation_text": self._generate_recommendation(
                        weighted_score,
                        analysis,
                        distribution_risk=distribution_risk,
                    ),
                    "recommendation": self._generate_recommendation(
                        weighted_score,
                        analysis,
                        distribution_risk=distribution_risk,
                    )
                }
                if code in {"603182.SH", "688618.SH", "300692.SZ"}:
                    logger.info(
                        "Step 4 final recommendation debug for %s: %s",
                        code,
                        {
                            "overall_score": final_recommendations[code].get("overall_score"),
                            "base_score": final_recommendations[code].get("base_score"),
                            "summary": final_recommendations[code].get("ai_summary"),
                            "distribution_risk_score": final_recommendations[code].get("distribution_risk_score"),
                            "moneyflow_3d_value": final_recommendations[code].get("moneyflow_3d_value"),
                            "recent_runup_5d": final_recommendations[code].get("recent_runup_5d"),
                            "turnover_spike_ratio": final_recommendations[code].get("turnover_spike_ratio"),
                            "candidate_risk_blocked": final_recommendations[code].get("candidate_risk_blocked"),
                            "weighted_score": final_recommendations[code].get("weighted_score"),
                        },
                    )

        # 按分数排序
        sorted_recommendations = dict(
            sorted(
                final_recommendations.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
        )
        logger.info(
            "Step 4 combine complete: recommendations=%s, ai_analyses=%s, industry_adjustments=%s, duration=%.2fs",
            len(sorted_recommendations),
            len(ai_analyses),
            len(industry_adjustments),
            time.perf_counter() - started_at,
        )

        return sorted_recommendations

    async def _get_market_data(self, *, trade_date: Optional[str] = None) -> Dict[str, Any]:
        """获取市场总览数据，优先使用本地 Tushare 原始行情缓存。"""
        target_trade_date = str(trade_date or self.screener._get_latest_trade_date()).replace("-", "")
        fallback_payload = {
            "indices": {},
            "rise_count": None,
            "fall_count": None,
            "flat_count": None,
            "limit_up": None,
            "limit_down": None,
            "total_amount": None,
            "trend": "市场概览数据暂不可用，报告以个股筛选与新闻上下文为主",
            "volume_trend": None,
            "sentiment": "暂无实时市场情绪数据",
            "data_available": False,
        }
        try:
            daily_summary = self.market_raw_data_repo.get_market_daily_summary(trade_date=target_trade_date)
            if int(daily_summary.get("pct_count") or 0) <= 0:
                return fallback_payload
            limit_summary = self.market_raw_data_repo.get_market_limit_summary(trade_date=target_trade_date)
            rise_count = int(daily_summary.get("rise_count") or 0)
            fall_count = int(daily_summary.get("fall_count") or 0)
            flat_count = int(daily_summary.get("flat_count") or 0)
            pct_count = int(daily_summary.get("pct_count") or 0)
            avg_pct_chg = self._safe_float(daily_summary.get("avg_pct_chg")) or 0.0
            limit_up = (
                int(limit_summary.get("limit_up") or 0)
                if int(limit_summary.get("total_count") or 0) > 0
                else int(daily_summary.get("limit_up_estimate") or 0)
            )
            limit_down = (
                int(limit_summary.get("limit_down") or 0)
                if int(limit_summary.get("total_count") or 0) > 0
                else int(daily_summary.get("limit_down_estimate") or 0)
            )
            total_amount = self._safe_float(daily_summary.get("total_amount"))
            total_amount_yi = (total_amount / 100000.0) if total_amount is not None else None
            up_ratio = rise_count / pct_count if pct_count else 0.0
            if up_ratio >= 0.58 and avg_pct_chg >= 0:
                trend_label = "市场整体偏强"
            elif up_ratio <= 0.42 and avg_pct_chg <= 0:
                trend_label = "市场整体偏弱"
            else:
                trend_label = "市场整体分化"
            amount_text = f"，全市场成交额约{total_amount_yi:.0f}亿元" if total_amount_yi is not None else ""
            trend = (
                f"{trend_label}：上涨{rise_count}家、下跌{fall_count}家、平盘{flat_count}家，"
                f"平均涨跌幅{avg_pct_chg:+.2f}%{amount_text}。"
            )
            sentiment = (
                f"涨停{limit_up}家、跌停{limit_down}家，"
                f"上涨占比{up_ratio * 100:.1f}%，短线情绪{'偏活跃' if limit_up > limit_down and up_ratio >= 0.5 else '偏谨慎' if up_ratio < 0.45 else '分化'}。"
            )
            return {
                "indices": {},
                "rise_count": rise_count,
                "fall_count": fall_count,
                "flat_count": flat_count,
                "limit_up": limit_up,
                "limit_down": limit_down,
                "total_amount": total_amount,
                "total_amount_yi": round(total_amount_yi, 2) if total_amount_yi is not None else None,
                "avg_pct_chg": round(avg_pct_chg, 4),
                "up_ratio": round(up_ratio, 4),
                "trend": trend,
                "volume_trend": amount_text.lstrip("，") if amount_text else None,
                "sentiment": sentiment,
                "data_available": True,
                "trade_date": target_trade_date,
            }
        except Exception:
            logger.exception("Failed to build market overview from local market data: trade_date=%s", target_trade_date)
            return fallback_payload

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
