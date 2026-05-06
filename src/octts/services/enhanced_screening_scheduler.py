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

logger = logging.getLogger(__name__)

TOP_RECOMMENDATION_LIMIT = 10
TODAY_TOP_LIMIT = 3
MODEL_CANDIDATE_POOL_LIMIT = 100
STAGE2_TOP20_LIMIT = 20
WINDOW_RECOMMENDATION_TAG = "今日候选"
LLM_REVIEW_CANDIDATE_LIMIT = 3
BACKFILL_TRAINING_CANDIDATE_LIMIT = 50
V2_COARSE_POOL_LIMIT = 200
V2_RULE_WEIGHT = 0.1
V2_EXCLUDE_BJ = True
LIGHT_FUNDAMENTAL_BONUS_MIN = -3.0
LIGHT_FUNDAMENTAL_BONUS_MAX = 5.0
LIGHT_FUNDAMENTAL_FORECAST_BONUS_CAP = 1.5
REPEAT_CONFIDENCE_BONUS = 0.08
MAX_CONFIDENCE = 0.98
INDUSTRY_FLOW_SCORE_CAP = 3.0
DISTRIBUTION_RISK_BLOCK_SCORE = 3.5
TOP3_MAX_DISTRIBUTION_RISK_SCORE = 2.1
TOP10_MAX_DISTRIBUTION_RISK_SCORE = 3.2
TOP3_RISK_PENALTY_MULTIPLIER = 12.0
TOP10_RISK_PENALTY_MULTIPLIER = 8.5
TOP3_DIVERGENCE_PENALTY_MULTIPLIER = 0.25
TOP3_CONTRADICTION_PENALTY = 8.0
CONTINUATION_BIAS_MAX_ABS = 6.0
REPEAT_PICK_CONTINUATION_BONUS = 1.5
DISTRIBUTION_TURNOVER_SPIKE_HIGH = 1.6
DISTRIBUTION_TURNOVER_SPIKE_VERY_HIGH = 2.1
DISTRIBUTION_VOLUME_RATIO_HIGH = 2.5
DISTRIBUTION_RECENT_RUNUP_HIGH = 10.0
DISTRIBUTION_PRICE_POSITION_HIGH = 0.88
THEME_SUPPORT_SCORE_STRONG = 3.4
THEME_SUPPORT_SCORE_MEDIUM = 2.0
UNSUPPORTED_HIGH_POSITION_EXTRA_PENALTY = 3.8
JUSTIFIED_TURNOVER_RELIEF = 1.8


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
            message="本地数据库已准备，正在直接构建模型特征并生成 Top100...",
            details={"strategy_count": 0, "legacy_strategy_recall": False},
        )

        # 3. 模型选出 Top100，Top3 只做极端风险硬排除，其余字段用于报告评估
        logger.info("Step 3: Select model Top100 and derive Top3 with extreme-risk veto only")
        candidate_limit = max(MODEL_CANDIDATE_POOL_LIMIT, self.settings.screening_top_n)
        trade_day = datetime.strptime(trade_date, "%Y%m%d").date()
        rerank_result = self.regression_rerank_service.rank_market_universe(
            trade_date=trade_day,
            candidate_limit=candidate_limit,
            analysis_limit=TOP_RECOMMENDATION_LIMIT,
            exclude_bj=V2_EXCLUDE_BJ,
        )
        if len(rerank_result.candidate_codes) < MODEL_CANDIDATE_POOL_LIMIT:
            raise RuntimeError(
                "Model universe ranking produced too few candidates: "
                f"candidates={len(rerank_result.candidate_codes)}, "
                f"required={MODEL_CANDIDATE_POOL_LIMIT}, "
                f"fallback_reason={rerank_result.fallback_reason}, "
                f"error={rerank_result.error_message or 'unknown'}"
            )
        snapshot_started_at = time.perf_counter()
        market_snapshot = self._build_lightweight_market_snapshot(trade_date)
        market_snapshot = self._hydrate_snapshot_daily_history_for_codes(
            market_snapshot,
            trade_date=trade_date,
            stock_codes=rerank_result.candidate_codes[:MODEL_CANDIDATE_POOL_LIMIT],
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
            step_name="模型Top100与极端风险筛选",
            progress_percent=42,
            message="正在基于预测模型生成 Top100，并仅用极端风险排除今日 Top3...",
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
                "moneyflow_3d_value": distribution_risk.get("moneyflow_3d_value"),
                "turnover_spike_ratio": distribution_risk.get("turnover_spike_ratio"),
                "recent_runup_5d": distribution_risk.get("recent_runup_5d"),
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
            code for code, _ in sorted(stage2_scores.items(), key=lambda item: item[1], reverse=True)
            if code in stage1_candidate_set and not bool(final_recommendations.get(code, {}).get("candidate_risk_blocked", False))
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
        daily_basic = market_snapshot.get("daily_basic") if isinstance(market_snapshot.get("daily_basic"), dict) else {}
        stock_map: Dict[str, Dict[str, Any]] = {}
        for stock in stocks:
            if not isinstance(stock, dict):
                continue
            ts_code = str(stock.get("ts_code") or "").strip()
            if not ts_code:
                continue
            merged_stock = dict(stock)
            basic = daily_basic.get(ts_code)
            if isinstance(basic, dict):
                merged_stock.update(basic)
            stock_map.setdefault(ts_code, merged_stock)
        return stock_map

    def _build_lightweight_market_snapshot(self, trade_date: str) -> Dict[str, Any]:
        stocks = self.screener.get_all_stocks()
        ts_codes = [stock.get("ts_code") for stock in stocks if isinstance(stock, dict) and stock.get("ts_code")]
        daily_basic = self.screener.client.fetch_daily_basic_batch(ts_codes=ts_codes, trade_date=trade_date)
        return {
            "snapshot_version": "lightweight_model_universe_v1",
            "snapshot_type": "lightweight_model_universe",
            "trade_date": trade_date,
            "created_at": datetime.now().isoformat(),
            "stocks": stocks,
            "daily_basic": daily_basic,
            "daily": {},
        }

    def _hydrate_snapshot_daily_history_for_codes(
        self,
        market_snapshot: Optional[Dict[str, Any]],
        *,
        trade_date: str,
        stock_codes: List[str],
    ) -> Dict[str, Any]:
        snapshot = dict(market_snapshot or {})
        daily_map = snapshot.get("daily") if isinstance(snapshot.get("daily"), dict) else {}
        hydrated_daily = dict(daily_map)
        start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=120)).strftime("%Y%m%d")
        for code in stock_codes:
            normalized = str(code or "").strip().upper()
            if not normalized or normalized in hydrated_daily:
                continue
            rows = self.market_raw_data_repo.get_daily_range(
                ts_code=normalized,
                start_date=start_date,
                end_date=trade_date,
            )
            if rows:
                hydrated_daily[normalized] = rows
        snapshot["daily"] = hydrated_daily
        logger.info(
            "Lightweight snapshot daily history hydrated from local DB: requested=%s, hydrated=%s, start_date=%s, end_date=%s",
            len(stock_codes),
            sum(1 for code in stock_codes if str(code or "").strip().upper() in hydrated_daily),
            start_date,
            trade_date,
        )
        return snapshot

    def _build_model_candidate_screening_results(
        self,
        *,
        trade_date: date,
        rerank_result: RegressionRerankResult,
        market_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, ScreenResult]:
        all_market_stock_map = self._build_all_market_stock_map(market_snapshot)
        stocks: List[StockScreenItem] = []
        for code in rerank_result.candidate_codes:
            metadata = rerank_result.metadata_by_code.get(code, {})
            snapshot = all_market_stock_map.get(code, {})
            daily_rows = self._get_snapshot_daily_rows(market_snapshot, code)
            close = self._first_defined_value(snapshot.get("close"), metadata.get("close"), 0.0)
            ma20 = self._compute_ma(daily_rows, 20)
            if ma20 is None:
                ma20 = self._estimate_ma_from_close_to_ma(close, metadata.get("close_to_ma20"))
            pct_change = self._first_defined_value(
                snapshot.get("pct_change"),
                snapshot.get("pct_chg"),
                metadata.get("pct_change"),
            )
            volume_ratio = self._first_defined_value(snapshot.get("volume_ratio"), metadata.get("volume_ratio"), 0.0)
            turnover_rate = self._first_defined_value(snapshot.get("turnover_rate"), metadata.get("turnover_rate"), 0.0)
            recommendation_score = self._first_defined_value(
                metadata.get("recommendation_score"),
                (float(metadata.get("blend_score") or 0.0) * 100.0),
                0.0,
            )
            technical_score = self._first_defined_value(
                metadata.get("technical_score"),
                (float(metadata.get("model_score_norm") or 0.0) * 100.0),
                0.0,
            )
            name = str(snapshot.get("name") or metadata.get("name") or code)
            stocks.append(
                StockScreenItem(
                    ts_code=code,
                    name=name,
                    close=float(close or 0.0),
                    pct_change=float(pct_change) if pct_change is not None else None,
                    volume_ratio=float(volume_ratio or 0.0),
                    turnover_rate=float(turnover_rate or 0.0),
                    ma20=ma20,
                    price_position_20d=self._safe_float(metadata.get("price_position_20d")),
                    technical_score=float(technical_score or 0.0),
                    recommendation_score=float(recommendation_score or 0.0),
                    score=float(recommendation_score or 0.0),
                    recommendation="monitor",
                    risk_level="medium",
                    confidence="medium",
                    market_cap=self._safe_float(metadata.get("market_cap")),
                    pe_ratio=self._safe_float(metadata.get("pe_ttm")),
                    industry=str(snapshot.get("industry") or "") or None,
                    match_reasons=["预测模型Top100"],
                )
            )

        return {
            "model_top100": ScreenResult(
                screen_id=f"model-top100-{trade_date.strftime('%Y%m%d')}-{uuid4().hex[:8]}",
                criteria=ScreenCriteria(
                    exclude_st=True,
                    exclude_bj=V2_EXCLUDE_BJ,
                    limit=max(len(stocks), 1),
                    sort_by="rerank_pool_rank",
                    sort_desc=False,
                ),
                stocks=stocks,
                total_count=len(stocks),
                execution_time=0.0,
            )
        }

    @staticmethod
    def _estimate_ma_from_close_to_ma(close: Any, close_to_ma: Any) -> Optional[float]:
        try:
            close_value = float(close)
            ratio_value = float(close_to_ma)
        except (TypeError, ValueError):
            return None
        denominator = 1.0 + ratio_value
        if close_value <= 0 or denominator == 0:
            return None
        return close_value / denominator

    def _build_focus_stock_llm_contexts(
        self,
        *,
        stock_codes: List[str],
        screening_results: Dict[str, ScreenResult],
        final_recommendations: Dict[str, Dict[str, Any]],
        market_snapshot: Optional[Dict[str, Any]],
        rerank_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        stock_map = self._build_screened_stock_map(screening_results)
        contexts: Dict[str, Dict[str, Any]] = {}
        for code in stock_codes:
            stock = stock_map.get(code)
            recommendation = final_recommendations.get(code, {})
            rerank_info = rerank_metadata.get(code, {})
            daily_rows = self._get_snapshot_daily_rows(market_snapshot, code)
            latest_daily = daily_rows[0] if daily_rows else {}
            contexts[code] = {
                "ts_code": code,
                "name": getattr(stock, "name", None) or recommendation.get("name") or code,
                "model_rank": rerank_info.get("rerank_pool_rank") or recommendation.get("rerank_pool_rank"),
                "model_score": rerank_info.get("model_score") or recommendation.get("rerank_model_score"),
                "model_blend_score": rerank_info.get("blend_score") or recommendation.get("rerank_blend_score"),
                "close": self._first_defined_value(getattr(stock, "close", None), latest_daily.get("close"), recommendation.get("close")),
                "pct_change": self._first_defined_value(getattr(stock, "pct_change", None), latest_daily.get("pct_chg"), recommendation.get("pct_change")),
                "turnover_rate": self._first_defined_value(getattr(stock, "turnover_rate", None), recommendation.get("turnover_rate")),
                "volume_ratio": self._first_defined_value(getattr(stock, "volume_ratio", None), recommendation.get("volume_ratio")),
                "ma20": self._first_defined_value(getattr(stock, "ma20", None), recommendation.get("ma20"), self._compute_ma(daily_rows, 20)),
                "price_position_20d": self._first_defined_value(getattr(stock, "price_position_20d", None), recommendation.get("price_position_20d")),
                "recommendation_score": recommendation.get("final_display_recommendation_score") or recommendation.get("weighted_score") or recommendation.get("recommendation_score") or recommendation.get("score"),
                "overall_score": recommendation.get("overall_score"),
                "risk_score": recommendation.get("distribution_risk_score"),
                "distribution_risk_score": recommendation.get("distribution_risk_score"),
                "distribution_risk_flags": list(recommendation.get("distribution_risk_flags") or []),
                "candidate_risk_blocked": bool(recommendation.get("candidate_risk_blocked", False)),
                "top3_extreme_risk_blocked": bool(recommendation.get("top3_extreme_risk_blocked", False)),
                "top3_extreme_risk_reason": recommendation.get("top3_extreme_risk_reason"),
                "moneyflow_3d_value": recommendation.get("moneyflow_3d_value"),
                "recent_large_order_net_inflow": recommendation.get("recent_large_order_net_inflow"),
                "recent_super_large_order_net_inflow": recommendation.get("recent_super_large_order_net_inflow"),
                "turnover_spike_ratio": recommendation.get("turnover_spike_ratio"),
                "recent_runup_5d": recommendation.get("recent_runup_5d"),
                "late_stage_momentum_flag": bool(recommendation.get("late_stage_momentum_flag", False)),
                "selection_stage": recommendation.get("selection_stage"),
                "selection_reason": recommendation.get("selection_reason"),
            }
        return contexts

    @staticmethod
    def _get_snapshot_daily_rows(market_snapshot: Optional[Dict[str, Any]], ts_code: str) -> List[Dict[str, Any]]:
        if not isinstance(market_snapshot, dict):
            return []
        daily_map = market_snapshot.get("daily")
        if not isinstance(daily_map, dict):
            return []
        rows = daily_map.get(ts_code) or []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    @classmethod
    def _compute_ma(cls, daily_rows: List[Dict[str, Any]], window: int) -> Optional[float]:
        closes = [
            cls._safe_float(row.get("close"))
            for row in sorted(daily_rows, key=lambda item: str(item.get("trade_date", "")))
            if isinstance(row, dict)
        ]
        closes = [value for value in closes if value is not None]
        if len(closes) < window:
            return None
        return round(sum(closes[-window:]) / window, 4)

    def _build_industry_flow_adjustments(
        self,
        stock_map: Dict[str, Any],
        *,
        all_market_stock_map: Optional[Dict[str, Dict[str, Any]]] = None,
        market_snapshot: Optional[Dict[str, Any]] = None,
        trade_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        started_at = time.perf_counter()
        cache_key = (
            str(trade_date or (market_snapshot or {}).get("trade_date") or ""),
            tuple(sorted(str(code).strip().upper() for code in stock_map.keys())),
        )
        cache: Dict[Any, Dict[str, Dict[str, Any]]] = getattr(self, "_industry_flow_adjustments_cache", {})
        cached_adjustments = cache.get(cache_key)
        if cached_adjustments is not None:
            logger.info(
                "Step 4 industry heat cache hit: candidate_stocks=%s, adjusted_stocks=%s, duration=%.2fs",
                len(stock_map),
                len(cached_adjustments),
                time.perf_counter() - started_at,
            )
            return {code: dict(payload) for code, payload in cached_adjustments.items()}

        candidate_industries = set()
        for stock in stock_map.values():
            industry = self._extract_industry_name(stock)
            if industry:
                candidate_industries.add(industry)

        logger.info(
            "Step 4 industry heat start: candidate_stocks=%s, candidate_industries=%s",
            len(stock_map),
            len(candidate_industries),
        )
        if not candidate_industries:
            return {}

        industry_totals = self._build_industry_flow_totals(
            candidate_industries,
            all_market_stock_map=all_market_stock_map or {},
            candidate_stock_map=stock_map,
            market_snapshot=market_snapshot,
            trade_date=trade_date,
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
                "industry_flow_value": metrics["industry_flow_value"],
            }
        logger.info(
            "Step 4 industry heat complete: matched_industries=%s, adjusted_stocks=%s, duration=%.2fs",
            len(industry_totals),
            len(adjustments),
            time.perf_counter() - started_at,
        )
        cache[cache_key] = {code: dict(payload) for code, payload in adjustments.items()}
        setattr(self, "_industry_flow_adjustments_cache", cache)
        return adjustments

    def _build_industry_flow_totals(
        self,
        industries: set[str],
        *,
        all_market_stock_map: Dict[str, Dict[str, Any]],
        candidate_stock_map: Optional[Dict[str, Any]] = None,
        market_snapshot: Optional[Dict[str, Any]] = None,
        trade_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        started_at = time.perf_counter()
        # IMPORTANT: after snapshot slimming, we should NOT aggregate with full-market daily rows.
        # Only aggregate within candidate stock universe (Top100 / candidates) to avoid misleading "missing" inflation.
        industry_groups: Dict[str, List[str]] = {industry: [] for industry in industries}
        if candidate_stock_map:
            for ts_code, stock in candidate_stock_map.items():
                industry = self._extract_industry_name(stock)
                if industry in industry_groups:
                    industry_groups[industry].append(str(ts_code).strip().upper())
        else:
            for ts_code, stock in all_market_stock_map.items():
                industry = self._extract_industry_name(stock)
                if industry not in industry_groups:
                    continue
                industry_groups[industry].append(ts_code)

        daily_basic_map = {}
        daily_map: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(market_snapshot, dict):
            raw_daily_basic = market_snapshot.get("daily_basic")
            if isinstance(raw_daily_basic, dict):
                daily_basic_map = raw_daily_basic
            raw_daily = market_snapshot.get("daily")
            if isinstance(raw_daily, dict):
                daily_map = raw_daily
        logger.info(
            "Step 4 industry totals mode: mode=snapshot_based, industries=%s, snapshot_basic=%s, snapshot_daily=%s, fallback_moneyflow=disabled",
            len(industry_groups),
            len(daily_basic_map),
            len(daily_map),
        )

        metrics_map: Dict[str, Dict[str, Any]] = {}
        relay_industry_map = self.market_raw_data_repo.get_industry_moneyflow_by_trade_date(
            industries=industries,
            trade_date=trade_date,
        ) if trade_date else {}
        for industry, ts_codes in industry_groups.items():
            if not ts_codes:
                continue
            industry_started_at = time.perf_counter()
            metrics = self._build_snapshot_based_industry_metrics(
                industry,
                ts_codes,
                daily_basic_map,
                daily_map,
                all_market_stock_map,
                candidate_stock_map=candidate_stock_map,
            )
            if not metrics:
                logger.info(
                    "Step 4 industry skipped: industry=%s, stocks=%s, reason=no_snapshot_metrics",
                    industry,
                    len(ts_codes),
                )
                continue
            metrics_map[industry] = metrics
            relay_industry = relay_industry_map.get(industry) or {}
            relay_net_amount = self._safe_float(relay_industry.get("net_amount"))
            relay_pct_change = self._safe_float(relay_industry.get("pct_change"))
            if relay_net_amount is not None:
                metrics_map[industry]["industry_flow_value"] = relay_net_amount
                if relay_net_amount > 0:
                    metrics_map[industry]["industry_heat_score"] = min(
                        INDUSTRY_FLOW_SCORE_CAP,
                        round(float(metrics_map[industry].get("industry_heat_score") or 0.0) + 1.1, 2),
                    )
                elif relay_net_amount < 0:
                    metrics_map[industry]["industry_heat_score"] = max(
                        -INDUSTRY_FLOW_SCORE_CAP,
                        round(float(metrics_map[industry].get("industry_heat_score") or 0.0) - 0.9, 2),
                    )
            if relay_pct_change is not None:
                if relay_pct_change > 0:
                    metrics_map[industry]["industry_heat_score"] = min(
                        INDUSTRY_FLOW_SCORE_CAP,
                        round(float(metrics_map[industry].get("industry_heat_score") or 0.0) + 0.4, 2),
                    )
                elif relay_pct_change < 0:
                    metrics_map[industry]["industry_heat_score"] = max(
                        -INDUSTRY_FLOW_SCORE_CAP,
                        round(float(metrics_map[industry].get("industry_heat_score") or 0.0) - 0.35, 2),
                    )
            metrics_map[industry]["industry_flow_bias"] = self._describe_industry_flow_bias(
                float(metrics_map[industry].get("industry_heat_score") or 0.0)
            )
            metrics_map[industry]["relay_industry_net_amount"] = relay_net_amount
            metrics_map[industry]["relay_industry_pct_change"] = relay_pct_change
            logger.info(
                "Step 4 industry aggregated: industry=%s, stocks=%s, used=%s, pct_count=%s, pct_sources=%s, positive_ratio=%.4f, heat_score=%.2f, duration=%.2fs",
                industry,
                len(ts_codes),
                metrics.get("industry_stock_count", 0),
                metrics.get("industry_pct_count", 0),
                metrics.get("industry_pct_sources", {}),
                metrics.get("industry_positive_ratio", 0.0),
                metrics.get("industry_heat_score", 0.0),
                time.perf_counter() - industry_started_at,
            )
        logger.info(
            "Step 4 industry totals complete: hit_industries=%s/%s, duration=%.2fs",
            len(metrics_map),
            len(industry_groups),
            time.perf_counter() - started_at,
        )
        return metrics_map

    def _build_snapshot_based_industry_metrics(
        self,
        industry: str,
        ts_codes: List[str],
        daily_basic_map: Dict[str, Dict[str, Any]],
        daily_map: Dict[str, List[Dict[str, Any]]],
        all_market_stock_map: Dict[str, Dict[str, Any]],
        *,
        candidate_stock_map: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        used_count = 0
        positive_count = 0
        pct_sum = 0.0
        amount_sum = 0.0
        turnover_sum = 0.0
        volume_ratio_sum = 0.0

        pct_count = 0
        pct_source_counts = {
            "daily.pct_chg": 0,
            "daily.pct_change": 0,
            "basic.pct_chg": 0,
            "basic.pct_change": 0,
            "stock.pct_change": 0,
            "missing": 0,
        }
        for ts_code in ts_codes:
            basic = daily_basic_map.get(ts_code)
            stock_snapshot = all_market_stock_map.get(ts_code) or {}
            candidate_stock = (candidate_stock_map or {}).get(ts_code) if candidate_stock_map else None
            if not isinstance(basic, dict):
                basic = {}
            if not isinstance(stock_snapshot, dict):
                stock_snapshot = {}
            if not basic and not stock_snapshot:
                continue
            used_count += 1
            daily_records = daily_map.get(ts_code) or []
            latest_daily = daily_records[0] if daily_records and isinstance(daily_records[0], dict) else {}
            pct_chg = self._safe_float(latest_daily.get("pct_chg"))
            if pct_chg is not None:
                pct_source_counts["daily.pct_chg"] += 1
            else:
                pct_chg = self._safe_float(latest_daily.get("pct_change"))
                if pct_chg is not None:
                    pct_source_counts["daily.pct_change"] += 1
                else:
                    pct_chg = self._safe_float(basic.get("pct_chg"))
                    if pct_chg is not None:
                        pct_source_counts["basic.pct_chg"] += 1
                    else:
                        pct_chg = self._safe_float(basic.get("pct_change"))
                        if pct_chg is not None:
                            pct_source_counts["basic.pct_change"] += 1
                        else:
                            pct_chg = self._safe_float(stock_snapshot.get("pct_change"))
                            if pct_chg is None and candidate_stock is not None:
                                # Candidate stocks (Top100) have pct_change even when full-market snapshot doesn't.
                                if isinstance(candidate_stock, dict):
                                    pct_chg = self._safe_float(candidate_stock.get("pct_change"))
                                else:
                                    pct_chg = self._safe_float(getattr(candidate_stock, "pct_change", None))
                            if pct_chg is not None:
                                pct_source_counts["stock.pct_change"] += 1
                            else:
                                pct_source_counts["missing"] += 1
            amount = self._safe_float(basic.get("amount"))
            if amount is None:
                amount = self._safe_float(stock_snapshot.get("amount"))
            if amount is None:
                amount = self._safe_float(latest_daily.get("amount"))
            turnover_rate = self._safe_float(basic.get("turnover_rate"))
            volume_ratio = self._safe_float(basic.get("volume_ratio"))

            if pct_chg is not None:
                pct_count += 1
                pct_sum += pct_chg
                if pct_chg > 0:
                    positive_count += 1
            if amount is not None:
                amount_sum += amount
            if turnover_rate is not None:
                turnover_sum += turnover_rate
            if volume_ratio is not None:
                volume_ratio_sum += volume_ratio

        if used_count == 0:
            return None

        positive_ratio = (positive_count / pct_count) if pct_count else 0.5
        avg_pct = (pct_sum / pct_count) if pct_count else 0.0
        avg_turnover = turnover_sum / used_count
        avg_volume_ratio = volume_ratio_sum / used_count
        amount_signal = 0.0
        if amount_sum > 0:
            amount_signal = 1.0

        positive_ratio_signal = max(-1.5, min(1.5, (positive_ratio - 0.5) * 5.0))
        avg_pct_signal = max(-1.3, min(1.3, avg_pct / 2.5))
        turnover_signal = max(-0.3, min(0.3, (avg_turnover - 3.0) / 8.0))
        volume_ratio_signal = max(-0.25, min(0.25, (avg_volume_ratio - 1.0) / 3.0))
        amount_signal = amount_signal * 0.25

        raw_score = (
            positive_ratio_signal
            + avg_pct_signal
            + turnover_signal
            + volume_ratio_signal
            + amount_signal
        )
        industry_heat_score = max(-INDUSTRY_FLOW_SCORE_CAP, min(INDUSTRY_FLOW_SCORE_CAP, round(raw_score, 2)))
        flow_value = round(amount_sum, 2) if amount_sum else None
        return {
            "industry": industry,
            "industry_stock_count": used_count,
            "industry_pct_count": pct_count,
            "industry_pct_sources": pct_source_counts,
            "industry_3d_net_inflow": None,
            "industry_positive_ratio": round(positive_ratio, 4),
            "industry_flow_bias": self._describe_industry_flow_bias(industry_heat_score),
            "industry_heat_score": industry_heat_score,
            "industry_flow_value": flow_value,
        }

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_theme_support_map(
        self,
        stock_map: Dict[str, Any],
        *,
        news_clusters: List[Any],
        news_hot_stocks: set[str],
        industry_adjustments: Dict[str, Dict[str, Any]],
        distribution_risk_map: Dict[str, Dict[str, Any]],
        screening_results: Dict[str, ScreenResult],
    ) -> Dict[str, Dict[str, Any]]:
        strategy_counts: Dict[str, int] = {}
        for result in screening_results.values():
            if not result:
                continue
            for stock in result.stocks:
                strategy_counts[stock.ts_code] = strategy_counts.get(stock.ts_code, 0) + 1

        strong_cluster_hits: Dict[str, int] = {}
        for cluster in news_clusters:
            importance = float(getattr(cluster, "importance", 0.0) or 0.0)
            if importance < 0.6:
                continue
            related_codes = set()
            for value in getattr(cluster, "key_stocks", []) or []:
                normalized = self._normalize_stock_code(value)
                if normalized:
                    related_codes.add(normalized)
            for item in getattr(cluster, "news_items", []) or []:
                for value in getattr(item, "related_stocks", []) or []:
                    normalized = self._normalize_stock_code(value)
                    if normalized:
                        related_codes.add(normalized)
            for code in related_codes:
                strong_cluster_hits[code] = strong_cluster_hits.get(code, 0) + 1

        support_map: Dict[str, Dict[str, Any]] = {}
        for code in stock_map:
            industry_adjustment = industry_adjustments.get(code, {})
            distribution_risk = distribution_risk_map.get(code, {})
            strategy_count = strategy_counts.get(code, 0)
            industry_heat_score = float(industry_adjustment.get("industry_heat_score") or 0.0)
            industry_flow_bias = str(industry_adjustment.get("industry_flow_bias") or "中性")
            moneyflow_3d_value = float(distribution_risk.get("moneyflow_3d_value") or 0.0)
            score = 0.0
            sources: List[str] = []

            cluster_hit_count = strong_cluster_hits.get(code, 0)
            if cluster_hit_count > 0:
                score += min(2.2, 1.4 + 0.4 * (cluster_hit_count - 1))
                sources.append("高重要度新闻主题命中")
            elif code in news_hot_stocks:
                score += 0.9
                sources.append("新闻提及")

            if strategy_count >= 3:
                score += 1.2
                sources.append("多策略共振")
            elif strategy_count >= 2:
                score += 0.8
                sources.append("双策略共振")
            elif strategy_count == 1:
                score += 0.3

            if industry_heat_score >= 1.2:
                score += 1.0
                sources.append("行业热度偏强")
            elif industry_heat_score >= 0.45:
                score += 0.6
                sources.append("行业热度回暖")
            elif industry_heat_score <= -0.8:
                score -= 0.5

            if industry_flow_bias in {"明显偏强", "偏强"}:
                score += 0.7
                sources.append("行业资金偏强")
            elif industry_flow_bias in {"明显偏弱", "偏弱"}:
                score -= 0.5

            if moneyflow_3d_value >= 12000:
                score += 1.2
                sources.append("3日资金承接强")
            elif moneyflow_3d_value >= 5000:
                score += 0.6
                sources.append("3日资金承接尚可")
            elif moneyflow_3d_value <= 0:
                score -= 0.8
            elif moneyflow_3d_value < 3000:
                score -= 0.4

            if distribution_risk.get("latest_weakening_flag"):
                score -= 0.5
            if distribution_risk.get("high_level_pullback_flag"):
                score -= 0.9
            if distribution_risk.get("theme_support_absent_flag"):
                score -= 0.8

            score = round(score, 2)
            leader_turnover_justified_flag = (
                score >= THEME_SUPPORT_SCORE_STRONG
                and (
                    cluster_hit_count > 0
                    or strategy_count >= 2
                    or (industry_heat_score >= 1.2 and moneyflow_3d_value >= 5000)
                )
            )
            unsupported_high_position_flag = bool(
                distribution_risk.get("theme_support_absent_flag")
                and not leader_turnover_justified_flag
                and score < THEME_SUPPORT_SCORE_MEDIUM
            )
            if score >= THEME_SUPPORT_SCORE_STRONG:
                label = "strong"
            elif score >= THEME_SUPPORT_SCORE_MEDIUM:
                label = "moderate"
            else:
                label = "weak"
            support_map[code] = {
                "theme_support_score": score,
                "theme_support_label": label,
                "theme_support_sources": sources,
                "unsupported_high_position_flag": unsupported_high_position_flag,
                "leader_turnover_justified_flag": leader_turnover_justified_flag,
            }
        return support_map

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
            volume_ratio = self._safe_float(stock.get("volume_ratio")) or 0.0
            pct_change = self._safe_float(stock.get("pct_change"))
            if pct_change is None:
                pct_change = self._safe_float(stock.get("pct_chg")) or 0.0
            price_position = self._safe_float(stock.get("price_position_20d")) or 0.0
            turnover_rate = self._safe_float(stock.get("turnover_rate")) or 0.0
        else:
            ts_code = str(getattr(stock, "ts_code", "") or "").strip()
            volume_ratio = self._safe_float(getattr(stock, "volume_ratio", None)) or 0.0
            pct_change = self._safe_float(getattr(stock, "pct_change", None)) or 0.0
            price_position = self._safe_float(getattr(stock, "price_position_20d", None)) or 0.0
            turnover_rate = self._safe_float(getattr(stock, "turnover_rate", None)) or 0.0
        moneyflow_summary = self._build_stock_moneyflow_summary(ts_code, trade_date=trade_date) if ts_code else None
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
        high_turnover_active = latest_turnover >= 12.0 or turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_HIGH
        theme_support_absent_flag = (
            price_position >= DISTRIBUTION_PRICE_POSITION_HIGH
            and recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH
            and moneyflow_3d_value <= 3000
            and turnover_spike_ratio < DISTRIBUTION_TURNOVER_SPIKE_VERY_HIGH
            and latest_turnover < 18.0
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
        if theme_support_absent_flag:
            risk_score += 0.8
            risk_flags.append("高位运行但题材承接不足")

        distribution_risk_score = round(risk_score, 2)
        relay_veto = open_times_value >= 3 or (
            open_times_value >= 2 and limit_last_time.isdigit() and int(limit_last_time) >= 145000
        ) or (top_net_amount < 0 and top_net_rate is not None and top_net_rate <= -3.0)
        return {
            "distribution_risk_score": distribution_risk_score,
            "distribution_risk_flags": risk_flags,
            "moneyflow_3d_value": round(moneyflow_3d_value, 2),
            "large_order_net_inflow": round(large_order_net_inflow, 2),
            "super_large_order_net_inflow": round(super_large_order_net_inflow, 2),
            "turnover_spike_ratio": round(turnover_spike_ratio, 2),
            "recent_runup_5d": round(recent_runup_5d, 2),
            "relay_open_times": open_times_value,
            "relay_limit_last_time": limit_last_time or None,
            "relay_limit_first_time": limit_first_time or None,
            "relay_top_net_amount": round(top_net_amount, 2),
            "relay_top_net_rate": round(top_net_rate, 2) if top_net_rate is not None else None,
            "late_stage_momentum_flag": late_stage_momentum_flag,
            "latest_weakening_flag": latest_weakening,
            "high_level_pullback_flag": high_level_pullback_flag,
            "theme_support_absent_flag": theme_support_absent_flag,
            "relay_candidate_veto": relay_veto,
            "candidate_risk_blocked": distribution_risk_score >= DISTRIBUTION_RISK_BLOCK_SCORE or relay_veto,
        }

    def _build_stock_moneyflow_summary(self, ts_code: str, *, trade_date: Optional[str] = None) -> Optional[Dict[str, float]]:
        resolved_trade_date = trade_date or self.screener._get_latest_trade_date()
        summaries = self.market_raw_data_repo.get_moneyflow_summaries_by_trade_date(
            ts_codes=[ts_code],
            trade_date=resolved_trade_date,
            lookback_days=3,
        )
        summary = summaries.get(ts_code)
        if summary:
            return {
                "recent_3d_net_inflow": float(summary.get("recent_3d_net_inflow") or 0.0),
                "recent_large_order_net_inflow": float(summary.get("recent_large_order_net_inflow") or 0.0),
                "recent_super_large_order_net_inflow": float(summary.get("recent_super_large_order_net_inflow") or 0.0),
                "positive_flag": float(summary.get("positive_flag") or 0.0),
            }
        recent_3d_net_inflow = self._fetch_recent_moneyflow_total(ts_code, trade_date=resolved_trade_date)
        return {
            "recent_3d_net_inflow": recent_3d_net_inflow,
            "recent_large_order_net_inflow": 0.0,
            "recent_super_large_order_net_inflow": 0.0,
            "positive_flag": 1.0 if recent_3d_net_inflow > 0 else 0.0,
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
        if len(recent_rows) < 5:
            return 0.0
        total = 0.0
        for item in recent_rows[-5:]:
            total += self._safe_float(item.get("pct_chg")) or 0.0
        return total

    def _fetch_recent_moneyflow_total(self, ts_code: str, *, trade_date: Optional[str] = None) -> float:
        rows = self.screener.client.fetch_moneyflow(ts_code, trade_date=trade_date)
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

    def _build_company_business_summary(self, ts_code: str, fallback_industry: str) -> str:
        profile = self.screener.client.fetch_company_profile(ts_code)
        main_business = str(profile.get("main_business") or "").strip()
        business_scope = str(profile.get("business_scope") or "").strip()
        if main_business:
            return main_business[:180]
        if business_scope:
            return business_scope[:180]
        if fallback_industry:
            return f"公司主要处于{fallback_industry}方向，具体业务以公开资料披露为准。"
        return ""

    def _build_financial_yoy_summary(self, ts_code: str) -> Dict[str, Any]:
        rows = self.screener.client.fetch_financial_indicators(ts_code)
        if not rows:
            return {"latest_revenue_yoy": None, "latest_profit_yoy": None}
        latest = sorted(rows, key=lambda item: str(item.get("end_date") or ""), reverse=True)[0]
        return {
            "latest_revenue_yoy": latest.get("op_income_yoy"),
            "latest_profit_yoy": latest.get("netprofit_yoy"),
        }

    def _estimate_backfill_fundamental_score(self, ts_code: str) -> float:
        summary = self._build_financial_yoy_summary(ts_code)
        profit_yoy = self._safe_float(summary.get("latest_profit_yoy"))
        revenue_yoy = self._safe_float(summary.get("latest_revenue_yoy"))
        score = 50.0
        if profit_yoy is not None:
            score += max(-18.0, min(22.0, profit_yoy / 6.0))
        if revenue_yoy is not None:
            score += max(-10.0, min(12.0, revenue_yoy / 10.0))
        return round(max(20.0, min(90.0, score)), 2)

    def _estimate_backfill_sentiment_score(self, stock: Any) -> float:
        pct_change = self._safe_float(getattr(stock, "pct_change", None))
        volume_ratio = self._safe_float(getattr(stock, "volume_ratio", None))
        score = 50.0
        if pct_change is not None:
            score += max(-10.0, min(12.0, pct_change / 1.5))
        if volume_ratio is not None and volume_ratio > 1.0:
            score += max(0.0, min(8.0, (volume_ratio - 1.0) * 4.0))
        return round(max(35.0, min(75.0, score)), 2)

    def _build_moneyflow_windows(self, ts_code: str) -> Dict[str, Any]:
        rows = self.screener.client.fetch_moneyflow(ts_code)
        if not rows:
            return {
                "main_fund_flow_1d": None,
                "main_fund_flow_3d": None,
                "main_fund_flow_10d": None,
            }
        recent_rows = sorted(rows, key=lambda item: str(item.get("trade_date") or ""), reverse=True)

        def _sum_recent(limit: int) -> Optional[float]:
            total = 0.0
            seen = False
            for item in recent_rows[:limit]:
                value = item.get("net_mf_amount")
                try:
                    total += float(value or 0.0)
                    seen = True
                except (TypeError, ValueError):
                    continue
            return total if seen else None

        return {
            "main_fund_flow_1d": _sum_recent(1),
            "main_fund_flow_3d": _sum_recent(3),
            "main_fund_flow_10d": _sum_recent(10),
        }

    def _build_catalyst_summary(self, item: Dict[str, Any], recommendation: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        texts = [
            str(item.get("recommendation_text") or "").strip(),
            str(recommendation.get("recommendation") or "").strip(),
            str(analysis.get("summary") or "").strip(),
        ]
        for text in texts:
            if text:
                return text[:120]
        return ""

    @staticmethod
    def _extract_industry_name(stock: Any) -> str:
        if isinstance(stock, dict):
            return str(stock.get("industry") or "").strip()
        return str(getattr(stock, "industry", "") or "").strip()

    def _describe_snapshot_cache_status(self, market_snapshot: Optional[Dict[str, Any]], trade_date: str) -> str:
        if not isinstance(market_snapshot, dict):
            return "unknown"
        if market_snapshot.get("snapshot_type") == "lightweight_model_universe":
            return "lightweight"
        created_at = market_snapshot.get("created_at")
        if not isinstance(created_at, str):
            return "unknown"
        snapshot_path = self.screener.client._screening_snapshot_path(trade_date)
        if not snapshot_path.exists():
            return "unknown"
        try:
            snapshot_time = datetime.fromisoformat(created_at)
            file_time = datetime.fromtimestamp(snapshot_path.stat().st_mtime)
        except (ValueError, OSError):
            return "unknown"
        return "hit" if abs((snapshot_time - file_time).total_seconds()) < 2 else "rebuilt"

    @staticmethod
    def _describe_industry_flow_bias(industry_heat_score: float) -> str:
        if industry_heat_score >= 1.4:
            return "明显偏强"
        if industry_heat_score >= 0.45:
            return "偏强"
        if industry_heat_score <= -1.4:
            return "明显偏弱"
        if industry_heat_score <= -0.45:
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

    def _build_news_score_context(self, news_clusters: List[Any]) -> Dict[str, Any]:
        hot_stocks = self._extract_news_hot_stocks(news_clusters)
        strong_theme_stocks: set[str] = set()
        stock_themes: Dict[str, List[str]] = {}
        industry_themes: Dict[str, List[str]] = {}

        for cluster in news_clusters:
            importance = float(getattr(cluster, "importance", 0.0) or 0.0)
            if importance < 0.45:
                continue
            theme = str(getattr(cluster, "theme", "") or "").strip()
            if not theme:
                continue

            normalized_stocks = set()
            for value in getattr(cluster, "key_stocks", []) or []:
                normalized = self._normalize_stock_code(value)
                if normalized:
                    normalized_stocks.add(normalized)
            for item in getattr(cluster, "news_items", []) or []:
                for value in getattr(item, "related_stocks", []) or []:
                    normalized = self._normalize_stock_code(value)
                    if normalized:
                        normalized_stocks.add(normalized)

            if importance >= 0.75:
                strong_theme_stocks.update(normalized_stocks)

            for code in normalized_stocks:
                stock_themes.setdefault(code, [])
                if theme not in stock_themes[code]:
                    stock_themes[code].append(theme)

            for keyword in self._extract_theme_keywords(theme):
                industry_themes.setdefault(keyword, [])
                if theme not in industry_themes[keyword]:
                    industry_themes[keyword].append(theme)

        return {
            "hot_stocks": hot_stocks,
            "strong_theme_stocks": strong_theme_stocks,
            "stock_themes": stock_themes,
            "industry_themes": industry_themes,
        }

    @staticmethod
    def _extract_theme_keywords(theme: str) -> List[str]:
        parts = re.split(r"[、/\s,，；;：:（）()]+", str(theme or ""))
        keywords: List[str] = []
        for part in parts:
            normalized = part.strip()
            if len(normalized) < 2:
                continue
            if normalized not in keywords:
                keywords.append(normalized)
        return keywords

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

        persisted_pool_states = self.store.list_recommendation_pool(trade_date=trade_date)
        authoritative_report_context = self._build_report_context(
            trade_date=trade_date,
            pool_states=persisted_pool_states,
            ai_analyses=ai_analyses,
            final_recommendations=final_recommendations,
        )
        for key in ("today_top3_live_context", "yesterday_top3_live_context"):
            if report_context.get(key) is not None:
                authoritative_report_context[key] = report_context.get(key)
        authoritative_today_top_states = self._select_authoritative_today_top_states(persisted_pool_states)
        continuation_states = [item for item in persisted_pool_states if item.get("source_tag") == "昨日延续"]
        today_candidate_states = sorted(
            [
                item for item in persisted_pool_states
                if item.get("source_tag") in {"今日Top3", WINDOW_RECOMMENDATION_TAG}
            ],
            key=self._frontlist_sort_key,
        )[:TOP_RECOMMENDATION_LIMIT]
        if not today_candidate_states:
            today_candidate_states = sorted(
                [item for item in persisted_pool_states if item.get("in_frontlist")],
                key=self._frontlist_sort_key,
            )[:TOP_RECOMMENDATION_LIMIT]
        report_today_top3 = list(authoritative_report_context.get("today_top3") or [])
        report_blocks = dict((getattr(report, "metadata", {}) or {}).get("report_blocks", {}))

        snapshot = {
            "generated_at": datetime.now().isoformat(),
            "snapshot_type": "intelligent_screening",
            "screening_results": {
                "strategy_count": len(screening_results),
                "total_stocks": total_stocks,
                "final_recommendations": len(report_today_top3),
                "frontlist_count": len(today_candidate_states),
                "shadow_count": 0,
                "candidate_count": len(persisted_pool_states),
                "today_top_count": len(authoritative_today_top_states),
                "continuation_count": len(continuation_states),
            },
            "recommendation_pool": {
                "frontlist": today_candidate_states,
                "shadow": [],
                "shadow_symbols": [],
                "today_top": report_today_top3,
                "yesterday_continuations": continuation_states,
            },
            "ai_analyses": self._build_dashboard_ai_payload(
                ai_analyses,
                final_recommendations,
                stock_name_map,
                persisted_pool_states,
            ),
            "news_clusters": [
                {
                    "cluster_id": getattr(cluster, "cluster_id", ""),
                    "theme": getattr(cluster, "theme", ""),
                    "importance": getattr(cluster, "importance", 0.0),
                    "summary": getattr(cluster, "summary", ""),
                    "key_stocks": list(getattr(cluster, "key_stocks", []) or []),
                    "news_briefs": [
                        (
                            f"{getattr(item, 'title', '')}：{getattr(item, 'content', '')}"
                            if getattr(item, "title", "") and getattr(item, "content", "") and getattr(item, "content", "") != getattr(item, "title", "")
                            else getattr(item, "title", "") or getattr(item, "content", "")
                        )
                        for item in (getattr(cluster, "news_items", []) or [])[:3]
                        if (getattr(item, "title", "") or getattr(item, "content", ""))
                    ],
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
                "blocks": report_blocks,
            },
            "report_context": authoritative_report_context,
        }

        snapshot_dir = Path(self.settings.history_dir_path) / "intelligent_screening"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        latest_path = snapshot_dir / "latest.json"
        dated_path = snapshot_dir / f"{trade_date.strftime('%Y%m%d')}.json"
        for path in (latest_path, dated_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def _build_stage_pipeline_result(
        self,
        *,
        trade_date: date,
        screening_results: Dict[str, ScreenResult],
        market_snapshot: Dict[str, Any],
        rerank_result: RegressionRerankResult,
        baseline_candidate_codes: List[str],
    ) -> Dict[str, Any]:
        stage1_candidate_codes = self._filter_out_tracked_and_holding_codes(
            rerank_result.candidate_codes or baseline_candidate_codes
        )[:MODEL_CANDIDATE_POOL_LIMIT]
        structured_analyses = self._build_backfill_ai_analyses(
            stage1_candidate_codes,
            screening_results=screening_results,
            market_snapshot=market_snapshot,
        )
        stage2_recommendations = self._build_backfill_final_recommendations(
            trade_date=trade_date,
            screening_results=screening_results,
            ai_analyses=structured_analyses,
            market_snapshot=market_snapshot,
            candidate_codes=stage1_candidate_codes,
            top20_limit=STAGE2_TOP20_LIMIT,
        )
        fusion_ranked_codes = self._rank_stage1_candidates_by_fusion(
            candidate_codes=stage1_candidate_codes,
            recommendations=stage2_recommendations,
            rerank_metadata=rerank_result.metadata_by_code or {},
        )
        stage2_top20_codes = fusion_ranked_codes[:STAGE2_TOP20_LIMIT]
        stage2_recommendations = self._build_structured_stage_selection_metadata(
            recommendations=stage2_recommendations,
            rerank_metadata=rerank_result.metadata_by_code or {},
            stage2_top20_codes=stage2_top20_codes,
            stage3_top3_codes=[],
        )
        stage3_recommendations = self._apply_stage3_moneyflow_rerank(
            trade_date=trade_date,
            stage2_recommendations=stage2_recommendations,
            stage2_top20_codes=stage2_top20_codes,
        )
        for code in stage2_top20_codes:
            payload = stage3_recommendations.get(code)
            if payload and payload.get("selection_stage") == "stage3_final_top3":
                payload["selection_stage"] = "stage2_top20_pre_moneyflow"
        stock_name_map = self._build_stock_name_map(screening_results)
        stage3_top3_codes = self._select_model_top3_with_extreme_risk_veto(
            model_candidate_codes=stage2_top20_codes or stage1_candidate_codes,
            recommendations=stage3_recommendations,
            stock_name_map=stock_name_map,
        )
        analysis_target_codes = self._build_analysis_target_codes(
            trade_date=trade_date,
            candidate_codes=stage3_top3_codes,
            screening_results=screening_results,
        )
        logger.info(
            "Stage pipeline summary: trade_date=%s, stage1=%s, stage2=%s, stage3=%s, rerank_analysis=%s",
            trade_date.isoformat(),
            len(stage1_candidate_codes),
            len(stage2_top20_codes),
            len(stage3_top3_codes),
            len(analysis_target_codes),
        )
        return {
            "stage1_candidate_codes": stage1_candidate_codes,
            "stage2_top20_codes": stage2_top20_codes,
            "stage3_top3_codes": stage3_top3_codes,
            "structured_analyses": structured_analyses,
            "stage2_recommendations": stage2_recommendations,
            "final_recommendations": stage3_recommendations,
            "analysis_target_codes": analysis_target_codes,
        }

    @classmethod
    def _rank_stage1_candidates_by_fusion(
        cls,
        *,
        candidate_codes: List[str],
        recommendations: Dict[str, Dict[str, Any]],
        rerank_metadata: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        records = {
            code: recommendations.get(code)
            for code in candidate_codes
            if recommendations.get(code)
            and not bool(recommendations.get(code, {}).get("candidate_risk_blocked", False))
        }
        if not records:
            return []

        model_values: Dict[str, Optional[float]] = {}
        rank_values: Dict[str, Optional[float]] = {}
        for code, payload in records.items():
            metadata = rerank_metadata.get(code) or {}
            model_value = cls._safe_float(
                cls._first_defined_value(
                    metadata.get("blend_score"),
                    metadata.get("model_score"),
                    payload.get("rerank_blend_score"),
                    payload.get("rerank_model_score"),
                )
            )
            rerank_rank = cls._safe_float(
                cls._first_defined_value(
                    metadata.get("rerank_pool_rank"),
                    payload.get("rerank_pool_rank"),
                )
            )
            if rerank_rank is not None:
                payload["rerank_pool_rank"] = int(rerank_rank)
            model_values[code] = model_value
            rank_values[code] = rerank_rank

        model_norm = cls._normalize_descending_score_values(model_values)
        if all(value is None for value in model_norm.values()):
            model_norm = cls._rank_values_to_percentile(rank_values)
        overall_norm = cls._normalize_descending_score_values(
            {code: cls._safe_float(payload.get("overall_score")) for code, payload in records.items()}
        )

        for code, payload in records.items():
            model_score = model_norm.get(code)
            overall_score = overall_norm.get(code)
            payload["model_score_norm"] = model_score
            payload["overall_score_norm"] = overall_score
            if model_score is not None and overall_score is not None:
                payload["fusion_70_30"] = round(model_score * 0.7 + overall_score * 0.3, 6)
                payload["top3_ranking_strategy"] = "stage1_fusion_70_30"
            else:
                payload["fusion_70_30"] = None

            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "model_score_norm": model_score,
                "overall_score_norm": overall_score,
                "fusion_70_30": payload.get("fusion_70_30"),
                "top3_ranking_strategy": payload.get("top3_ranking_strategy"),
            }

        def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[Any, ...]:
            code, payload = item
            fusion_score = cls._safe_float(payload.get("fusion_70_30"))
            fallback_score = cls._safe_float(
                cls._first_defined_value(payload.get("stage3_final_score"), payload.get("score"), 0.0)
            ) or 0.0
            return (
                fusion_score is None,
                -(fusion_score if fusion_score is not None else 0.0),
                -fallback_score,
                cls._safe_float(payload.get("rerank_pool_rank")) or 999999.0,
                code,
            )

        return [code for code, _ in sorted(records.items(), key=sort_key)]

    @staticmethod
    def _normalize_descending_score_values(values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        valid_values = [value for value in values.values() if value is not None]
        if not valid_values:
            return {code: None for code in values}
        min_value = min(valid_values)
        max_value = max(valid_values)
        if max_value == min_value:
            return {code: 100.0 if values.get(code) is not None else None for code in values}
        return {
            code: round((float(value) - min_value) / (max_value - min_value) * 100.0, 6)
            if value is not None
            else None
            for code, value in values.items()
        }

    @staticmethod
    def _rank_values_to_percentile(ranks: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        valid_ranks = [rank for rank in ranks.values() if rank is not None]
        if not valid_ranks:
            return {code: None for code in ranks}
        min_rank = min(valid_ranks)
        max_rank = max(valid_ranks)
        if max_rank == min_rank:
            return {code: 100.0 if ranks.get(code) is not None else None for code in ranks}
        return {
            code: round((max_rank - float(rank)) / (max_rank - min_rank) * 100.0, 6)
            if rank is not None
            else None
            for code, rank in ranks.items()
        }

    def _select_model_top3_with_extreme_risk_veto(
        self,
        *,
        model_candidate_codes: List[str],
        recommendations: Dict[str, Dict[str, Any]],
        stock_name_map: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        selected_codes: List[str] = []
        vetoed_count = 0
        st_excluded_count = 0
        stock_name_map = stock_name_map or {}
        for rank, code in enumerate(model_candidate_codes, start=1):
            payload = recommendations.get(code)
            if not payload:
                continue
            payload["rerank_pool_rank"] = payload.get("rerank_pool_rank") or rank
            if self._is_st_stock_for_final_veto(code=code, payload=payload, stock_name_map=stock_name_map):
                payload["top3_st_excluded"] = True
                payload["top3_st_excluded_reason"] = "stock_name_contains_ST"
                payload["selection_stage"] = "model_top100_st_veto"
                payload["selection_reason_components"] = {
                    **dict(payload.get("selection_reason_components") or {}),
                    "top3_st_excluded": True,
                    "top3_st_excluded_reason": "stock_name_contains_ST",
                }
                st_excluded_count += 1
                continue
            payload["top3_st_excluded"] = False
            payload["top3_st_excluded_reason"] = None
            extreme_risk_reason = self._get_top3_extreme_risk_reason(payload)
            if extreme_risk_reason:
                payload["top3_extreme_risk_blocked"] = True
                payload["top3_extreme_risk_reason"] = extreme_risk_reason
                payload["selection_stage"] = "model_top100_extreme_risk_veto"
                vetoed_count += 1
                continue
            payload["top3_extreme_risk_blocked"] = False
            payload["top3_extreme_risk_reason"] = None
            payload["top3_st_excluded"] = False
            payload["top3_st_excluded_reason"] = None
            if len(selected_codes) < TODAY_TOP_LIMIT:
                selected_codes.append(code)

        selected_set = set(selected_codes)
        for code, payload in recommendations.items():
            if code not in selected_set:
                continue
            payload["top3_st_excluded"] = False
            payload["top3_st_excluded_reason"] = None
            payload["selection_stage"] = "stage3_final_top3"
            payload["selection_reason"] = (
                f"model_rank={payload.get('rerank_pool_rank')}; "
                f"fusion_70_30={payload.get('fusion_70_30')}; "
                "top3_extreme_risk_veto=False; "
                "top3_st_excluded=False"
            )
            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "rerank_pool_rank": payload.get("rerank_pool_rank"),
                "fusion_70_30": payload.get("fusion_70_30"),
                "top3_ranking_strategy": payload.get("top3_ranking_strategy"),
                "top3_extreme_risk_veto": False,
                "top3_st_excluded": False,
            }

        logger.info(
            "Model Top3 selected with final vetoes: selected=%s, risk_vetoed=%s, st_excluded=%s, model_pool=%s",
            selected_codes,
            vetoed_count,
            st_excluded_count,
            len(model_candidate_codes),
        )
        return selected_codes

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

    @staticmethod
    def _name_indicates_st(name: Any) -> bool:
        text = str(name or "").strip().upper().replace(" ", "")
        return bool(text) and (text.startswith("ST") or text.startswith("*ST") or "退" in text[:3])

    @staticmethod
    def _get_top3_extreme_risk_reason(payload: Dict[str, Any]) -> Optional[str]:
        if bool(payload.get("candidate_risk_blocked", False)):
            return "candidate_risk_blocked"
        if bool(payload.get("relay_candidate_veto", False)):
            return "relay_candidate_veto"
        if bool(payload.get("stage3_moneyflow_veto", False)):
            return "stage3_moneyflow_veto"
        distribution_risk_score = payload.get("distribution_risk_score")
        if distribution_risk_score is not None and float(distribution_risk_score) >= DISTRIBUTION_RISK_BLOCK_SCORE:
            return "distribution_risk_score_block"
        return None

    def _apply_stage3_moneyflow_rerank(
        self,
        *,
        trade_date: date,
        stage2_recommendations: Dict[str, Dict[str, Any]],
        stage2_top20_codes: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        final_recommendations = {code: dict(payload) for code, payload in stage2_recommendations.items()}
        if not stage2_top20_codes:
            return final_recommendations

        moneyflow_summary_map = self.market_raw_data_repo.get_moneyflow_summaries_by_trade_date(
            ts_codes=stage2_top20_codes,
            trade_date=trade_date.strftime("%Y%m%d"),
            lookback_days=3,
        )
        stage3_scored_codes: List[tuple[str, float]] = []
        for code in stage2_top20_codes:
            payload = final_recommendations.get(code)
            if not payload:
                continue
            moneyflow_summary = (
                moneyflow_summary_map.get(code)
                or self._build_stock_moneyflow_summary(code, trade_date=trade_date.strftime("%Y%m%d"))
                or {}
            )
            recent_3d_net_inflow = float(moneyflow_summary.get("recent_3d_net_inflow") or payload.get("moneyflow_3d_value") or 0.0)
            recent_large_order_net_inflow = float(moneyflow_summary.get("recent_large_order_net_inflow") or payload.get("recent_large_order_net_inflow") or 0.0)
            recent_super_large_order_net_inflow = float(moneyflow_summary.get("recent_super_large_order_net_inflow") or payload.get("recent_super_large_order_net_inflow") or 0.0)
            stage3_moneyflow_score = 0.0
            stage3_moneyflow_flags: List[str] = []
            stage3_moneyflow_risks: List[str] = []
            if recent_3d_net_inflow > 0:
                stage3_moneyflow_score += 1.0
                stage3_moneyflow_flags.append("3日净流入为正")
            elif recent_3d_net_inflow < 0:
                stage3_moneyflow_score -= 1.0
                stage3_moneyflow_risks.append("3日净流出")
            if recent_large_order_net_inflow > 0:
                stage3_moneyflow_score += 0.8
                stage3_moneyflow_flags.append("大单净流入为正")
            elif recent_large_order_net_inflow < 0:
                stage3_moneyflow_score -= 0.8
                stage3_moneyflow_risks.append("大单净流出")
            if recent_super_large_order_net_inflow > 0:
                stage3_moneyflow_score += 1.0
                stage3_moneyflow_flags.append("超大单净流入为正")
            elif recent_super_large_order_net_inflow < 0:
                stage3_moneyflow_score -= 1.0
                stage3_moneyflow_risks.append("超大单净流出")
            moneyflow_veto = bool(payload.get("unsupported_high_position_flag", False)) and (
                bool(payload.get("relay_candidate_veto", False))
                or recent_large_order_net_inflow < 0
                or recent_super_large_order_net_inflow < 0
            )
            stage3_final_score = round(float(payload.get("score") or 0.0), 4)
            payload["moneyflow_3d_value"] = round(recent_3d_net_inflow, 2)
            payload["recent_large_order_net_inflow"] = round(recent_large_order_net_inflow, 2)
            payload["recent_super_large_order_net_inflow"] = round(recent_super_large_order_net_inflow, 2)
            payload["stage3_moneyflow_score"] = round(stage3_moneyflow_score, 4)
            payload["stage3_moneyflow_flags"] = stage3_moneyflow_flags
            payload["stage3_moneyflow_risks"] = stage3_moneyflow_risks
            payload["stage3_moneyflow_veto"] = moneyflow_veto
            payload["stage3_final_score"] = stage3_final_score
            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "moneyflow_3d_value": round(recent_3d_net_inflow, 2),
                "recent_large_order_net_inflow": round(recent_large_order_net_inflow, 2),
                "recent_super_large_order_net_inflow": round(recent_super_large_order_net_inflow, 2),
                "stage3_moneyflow_score": round(stage3_moneyflow_score, 4),
                "stage3_moneyflow_veto": moneyflow_veto,
            }
            stage3_scored_codes.append((code, stage3_final_score))

        stage3_ranked_codes = [code for code, _ in sorted(stage3_scored_codes, key=lambda item: item[1], reverse=True)]
        stage3_top3_codes = stage3_ranked_codes[:TODAY_TOP_LIMIT]
        stage3_position_map = {code: index for index, code in enumerate(stage3_ranked_codes, start=1)}
        for code in stage2_top20_codes:
            payload = final_recommendations.get(code)
            if not payload:
                continue
            payload["structured_rank_score"] = payload.get("stage3_final_score", payload.get("structured_rank_score"))
            payload["structured_rank_position"] = stage3_position_map.get(code, payload.get("structured_rank_position"))
            payload["selection_stage"] = "stage3_final_top3" if code in stage3_top3_codes else "stage2_top20_pre_moneyflow"
            payload["selection_reason"] = (
                f"stage3_final_score={float(payload.get('stage3_final_score') or 0.0):.2f}; "
                f"moneyflow_score={float(payload.get('stage3_moneyflow_score') or 0.0):.2f}; "
                f"moneyflow_veto={bool(payload.get('stage3_moneyflow_veto', False))}"
            )
        return dict(sorted(final_recommendations.items(), key=lambda item: float(item[1].get("score") or 0.0), reverse=True))

    def _build_stage_backtest_payload(
        self,
        *,
        stage1_candidate_codes: List[str],
        stage2_top20_codes: List[str],
        stage3_top3_codes: List[str],
        rerank_result: RegressionRerankResult,
    ) -> Dict[str, Any]:
        return {
            "current_flow_top3_codes": list(rerank_result.analysis_codes[:TODAY_TOP_LIMIT]),
            "stage1_candidate_codes": list(stage1_candidate_codes),
            "stage2_top20_codes": list(stage2_top20_codes),
            "stage2_top3_without_moneyflow_codes": list(stage2_top20_codes[:TODAY_TOP_LIMIT]),
            "stage3_top3_codes": list(stage3_top3_codes),
            "rerank_candidate_codes": list(rerank_result.candidate_codes),
            "rerank_analysis_codes": list(rerank_result.analysis_codes),
            "rerank_fallback_reason": rerank_result.fallback_reason,
        }

    def _save_stage_backtest_snapshot(self, *, trade_date: str, payload: Dict[str, Any]) -> Optional[Path]:
        try:
            stage_dir = Path(self.settings.history_dir_path) / "stage_backtests"
            stage_dir.mkdir(parents=True, exist_ok=True)
            path = stage_dir / f"{trade_date}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return path
        except Exception:
            logger.exception("Failed to persist stage backtest snapshot: trade_date=%s", trade_date)
            return None

    @staticmethod
    def _build_stage_rank_maps(codes: List[str]) -> Dict[str, int]:
        return {code: index for index, code in enumerate(codes, start=1)}

    def _build_stock_name_map(self, screening_results: Dict[str, ScreenResult]) -> Dict[str, str]:
        stock_name_map: Dict[str, str] = {}
        for result in screening_results.values():
            if not result:
                continue
            for stock in result.stocks:
                stock_name_map.setdefault(stock.ts_code, stock.name)
        return stock_name_map

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
            selection_reason_components = {
                **dict(payload.get("selection_reason_components") or {}),
                "structured_score": round(float(payload.get("score") or 0.0), 4),
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
            payload["structured_rank_score"] = payload.get("score")
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
            overall_score = state.get("overall_score")
            if overall_score is None:
                overall_score = EnhancedScreeningScheduler._resolve_real_overall_score(
                    merged,
                    recommendation_meta,
                )
            recommendation_score = state.get(
                "recommendation_score",
                recommendation_meta.get("weighted_score", recommendation_meta.get("score", 0)),
            )
            if recommendation_meta:
                merged["recommendation"] = recommendation_meta.get("recommendation", merged.get("recommendation", ""))
                merged["news_mentioned"] = state.get("news_mentioned", recommendation_meta.get("news_mentioned", False))
                merged["strategy_count"] = state.get("strategy_count", recommendation_meta.get("strategy_count", 0))
            merged["overall_score"] = overall_score
            merged["priority_score"] = state.get("priority_score", overall_score)
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
            merged.update(EnhancedScreeningScheduler._build_unified_score_fields(merged, state, analysis, recommendation_meta))
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
        overall_score = item.get("overall_score")
        return overall_score is not None and cls._has_real_ai_overall_score(item)

    @staticmethod
    def _today_top_sort_key(item: Dict[str, Any]) -> Any:
        return (
            item.get("recommend_rank") is None,
            int(item.get("recommend_rank") or 9999),
            -float(item.get("recommendation_score") or 0.0),
            -float(item.get("overall_score") or item.get("priority_score") or 0.0),
            item.get("ts_code") or "",
        )

    @staticmethod
    def _frontlist_sort_key(item: Dict[str, Any]) -> Any:
        return (
            -float(item.get("recommendation_score") or 0.0),
            -float(item.get("final_display_recommendation_score") or 0.0),
            -float(item.get("overall_score") or item.get("priority_score") or 0.0),
            item.get("ts_code") or "",
        )

    @classmethod
    def _select_authoritative_today_top_states(
        cls,
        pool_states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked_states = sorted(
            [item for item in pool_states if item.get("ts_code")],
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
            if item.get("source_tag") == "今日Top3" and cls._has_real_ai_overall_score(item)
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

        model_candidate_codes = [
            code for code in candidate_codes[:MODEL_CANDIDATE_POOL_LIMIT]
            if code in final_recommendations
        ]
        rerank_rank_map = {
            code: index for index, code in enumerate(model_candidate_codes, start=1)
        }
        display_codes = model_candidate_codes[:TOP_RECOMMENDATION_LIMIT]
        today_top_codes = [
            code for code in model_candidate_codes
            if str((final_recommendations.get(code, {}) or {}).get("selection_stage") or "") == "stage3_final_top3"
        ]
        if not today_top_codes:
            today_top_codes = [
                code for code in model_candidate_codes
                if not self._get_top3_extreme_risk_reason(final_recommendations.get(code, {}) or {})
            ][:TODAY_TOP_LIMIT]
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
        continuation_codes = [code for code in previous_top3_codes if code]
        rerank_top_codes = set(today_top_codes)
        merged_display_codes = list(model_candidate_codes) + [
            code for code in continuation_codes
            if code not in rerank_rank_map and code in final_recommendations
        ]
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
            recommendation_score = base_recommendation_score
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
                    "distribution_risk_flags": list(recommendation.get("distribution_risk_flags") or (previous_state or {}).get("distribution_risk_flags") or []),
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
                "distribution_risk_score": recommendation.get("distribution_risk_score", (previous_state or {}).get("distribution_risk_score")),
                "distribution_risk_flags": list(recommendation.get("distribution_risk_flags") or (previous_state or {}).get("distribution_risk_flags") or []),
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
                "frontlist_rank": display_rank if code in display_code_set else None,
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

    def _build_report_context(
        self,
        *,
        trade_date: date,
        pool_states: List[Dict[str, Any]],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        state_map = {item.get("ts_code"): item for item in pool_states if item.get("ts_code")}
        normalized_pool_states = sorted(
            [item for item in pool_states if item.get("ts_code")],
            key=self._today_top_sort_key,
        )
        if not normalized_pool_states:
            normalized_pool_states = sorted(
                [item for item in self.store.list_recommendation_pool(trade_date=trade_date) if item.get("ts_code")],
                key=self._today_top_sort_key,
            )

        today_top_states = self._select_authoritative_today_top_states(normalized_pool_states)
        if not today_top_states:
            fallback_pool_states = self.store.list_recommendation_pool(trade_date=trade_date)
            if isinstance(fallback_pool_states, list):
                today_top_states = self._select_authoritative_today_top_states(fallback_pool_states)

        today_top10_states = [
            item
            for item in normalized_pool_states
            if item.get("in_frontlist")
            and item.get("source_tag") in {"今日Top3", WINDOW_RECOMMENDATION_TAG}
        ]
        today_top10_states = sorted(today_top10_states, key=self._frontlist_sort_key)
        today_top10 = [
            self._build_report_stock_payload(item, ai_analyses, final_recommendations)
            for item in today_top10_states[:TOP_RECOMMENDATION_LIMIT]
            if item.get("ts_code")
        ]
        market_event_contexts = self._build_market_event_contexts(
            trade_date=trade_date,
            stock_codes=[item.get("ts_code") for item in today_top_states if item.get("ts_code")],
        )
        today_top3 = [
            self._build_report_stock_payload(
                item,
                ai_analyses,
                final_recommendations,
                include_fresh_moneyflow=True,
                market_event_context=market_event_contexts.get(item.get("ts_code")) or {},
            )
            for item in today_top_states
            if item.get("ts_code")
        ]
        review_signal_trade_date = self._get_review_signal_trade_date(trade_date, lookback_trading_days=3)
        review_entry_trade_date = self._get_next_trade_date(review_signal_trade_date) if review_signal_trade_date else None
        review_signal_states = {}
        if review_signal_trade_date:
            review_signal_states = {
                item.get("ts_code"): item
                for item in self.store.load_recommendation_pool_state(trade_date=review_signal_trade_date)
                if item.get("ts_code")
            }
        review_top3 = self._select_authoritative_today_top_states(list(review_signal_states.values())) if review_signal_states else []

        yesterday_top3_review: List[Dict[str, Any]] = []
        for previous_item in review_top3:
            code = previous_item.get("ts_code")
            current_item = state_map.get(code)
            performance = self._build_review_performance(
                code=code,
                signal_trade_date=review_signal_trade_date,
                entry_trade_date=review_entry_trade_date,
                current_trade_date=trade_date,
            )
            current_review = self._build_report_stock_payload(current_item, ai_analyses, final_recommendations) if current_item else {}
            return_value = performance.get("review_return")
            sellable = bool(performance.get("sellable_by_weak_profit_rule"))
            if sellable:
                review_status = "可卖出"
                today_verdict = "持有满3个交易日后收益未超过3%，可按弱票规则卖出或调仓"
            elif isinstance(return_value, (int, float)):
                review_status = "继续观察"
                today_verdict = "持有满3个交易日后收益超过3%，暂不触发弱票卖出规则"
            elif current_item:
                review_status = "观察"
                today_verdict = "价格数据不足，先按当前候选状态继续观察"
            else:
                review_status = "失效"
                today_verdict = "今日未进入候选池或展示池，且价格数据不足，需人工复核"

            analysis_text = self._build_three_day_review_analysis(
                previous_item=previous_item,
                current_item=current_item,
                performance=performance,
                sellable=sellable,
            )
            current_review.update(
                {
                    "ts_code": code,
                    "name": self._resolve_stock_name(code, None, {}, previous_item, current_item),
                    "source_tag": "3日前Top3复盘",
                    "review_signal_date": review_signal_trade_date.isoformat() if review_signal_trade_date else None,
                    "review_entry_date": review_entry_trade_date.isoformat() if review_entry_trade_date else None,
                    "review_current_date": trade_date.isoformat(),
                    "review_entry_open": performance.get("entry_open"),
                    "review_current_price": performance.get("current_price"),
                    "review_return": return_value,
                    "review_return_pct": round(return_value * 100.0, 2) if isinstance(return_value, (int, float)) else None,
                    "sellable_by_weak_profit_rule": sellable,
                    "yesterday_conclusion": self._resolve_yesterday_conclusion(previous_item),
                    "today_verdict": today_verdict,
                    "review_status": review_status,
                    "status": review_status,
                    "analysis": analysis_text,
                    "review_analysis": analysis_text,
                    "strength_change": self._build_three_day_review_strength_change(performance, review_status),
                    "market_context_view": self._build_three_day_review_market_context(performance, today_verdict),
                    "miss_reason_candidates": ["3日收益未达3%", "资金占用效率偏低", "相对强度不足"] if sellable else [],
                    "missing_factor_candidates": [] if isinstance(return_value, (int, float)) else ["入场开盘价", "今日收盘价", "有效交易日数据"],
                    "action_plan": self._build_three_day_review_action_plan(performance, sellable),
                }
            )
            yesterday_top3_review.append(current_review)

        return {
            "trade_date": trade_date.isoformat(),
            "today_top3": today_top3,
            "today_top10": today_top10,
            "yesterday_top3_review": yesterday_top3_review,
            "today_top3_live_context": [],
            "yesterday_top3_live_context": [],
            "comparison_candidates": list(today_top3),
        }

    def _get_review_signal_trade_date(self, trade_date: date, *, lookback_trading_days: int = 3) -> Optional[date]:
        start_date = (trade_date - timedelta(days=45)).strftime("%Y%m%d")
        end_date = trade_date.strftime("%Y%m%d")
        trading_dates = self.market_raw_data_repo.list_trading_dates(start_date=start_date, end_date=end_date)
        parsed_dates = [datetime.strptime(value, "%Y%m%d").date() for value in trading_dates]
        if trade_date not in parsed_dates:
            parsed_dates.append(trade_date)
            parsed_dates = sorted(set(parsed_dates))
        current_index = parsed_dates.index(trade_date) if trade_date in parsed_dates else -1
        target_index = current_index - max(int(lookback_trading_days), 1)
        if target_index < 0:
            return None
        return parsed_dates[target_index]

    def _get_next_trade_date(self, trade_date: Optional[date]) -> Optional[date]:
        if trade_date is None:
            return None
        start_date = trade_date.strftime("%Y%m%d")
        end_date = (trade_date + timedelta(days=15)).strftime("%Y%m%d")
        trading_dates = self.market_raw_data_repo.list_trading_dates(start_date=start_date, end_date=end_date)
        parsed_dates = [datetime.strptime(value, "%Y%m%d").date() for value in trading_dates]
        for value in parsed_dates:
            if value > trade_date:
                return value
        return None

    def _build_review_performance(
        self,
        *,
        code: str,
        signal_trade_date: Optional[date],
        entry_trade_date: Optional[date],
        current_trade_date: date,
    ) -> Dict[str, Any]:
        if not code or signal_trade_date is None or entry_trade_date is None:
            return {}
        entry_bar = self.market_raw_data_repo.get_daily(ts_code=code, trade_date=entry_trade_date.strftime("%Y%m%d")) or {}
        current_bar = self.market_raw_data_repo.get_daily(ts_code=code, trade_date=current_trade_date.strftime("%Y%m%d")) or {}
        entry_open = self._safe_float(entry_bar.get("open"))
        current_price = self._safe_float(current_bar.get("close"))
        if current_price is None:
            current_price = self._safe_float(current_bar.get("open"))
        review_return = None
        if entry_open is not None and entry_open > 0 and current_price is not None:
            review_return = round(current_price / entry_open - 1.0, 6)
        return {
            "signal_date": signal_trade_date.isoformat(),
            "entry_date": entry_trade_date.isoformat(),
            "current_date": current_trade_date.isoformat(),
            "entry_open": entry_open,
            "current_price": current_price,
            "review_return": review_return,
            "sellable_by_weak_profit_rule": isinstance(review_return, (int, float)) and review_return <= 0.03,
        }

    @staticmethod
    def _build_three_day_review_analysis(
        *,
        previous_item: Dict[str, Any],
        current_item: Optional[Dict[str, Any]],
        performance: Dict[str, Any],
        sellable: bool,
    ) -> str:
        del current_item
        signal_date = performance.get("signal_date") or "3个交易日前"
        entry_date = performance.get("entry_date") or "次日"
        entry_open = performance.get("entry_open")
        current_price = performance.get("current_price")
        return_value = performance.get("review_return")
        previous_reason = str(previous_item.get("overview_reason") or previous_item.get("summary") or "当日入选Top3").strip()
        if isinstance(return_value, (int, float)):
            action_text = "收益未超过3%，可按弱票规则卖出或调仓" if sellable else "收益超过3%，暂不触发弱票卖出规则"
            return (
                f"复盘对象为{signal_date}生成的Top3，按{entry_date}开盘价{entry_open:.2f}作为入场基准，"
                f"当前参考价{current_price:.2f}，区间收益{return_value * 100:+.2f}%。{action_text}。"
                f"原始入选逻辑：{previous_reason}"
            )
        return f"复盘对象为{signal_date}生成的Top3，但入场开盘价或当前价格缺失，暂不能计算3日收益；原始入选逻辑：{previous_reason}"

    @staticmethod
    def _build_three_day_review_strength_change(performance: Dict[str, Any], review_status: str) -> str:
        return_value = performance.get("review_return")
        if isinstance(return_value, (int, float)):
            return f"从次日开盘基准到当前参考价的收益为{return_value * 100:+.2f}%，状态为{review_status}。"
        return "价格数据不足，强弱变化需要人工结合行情复核。"

    @staticmethod
    def _build_three_day_review_market_context(performance: Dict[str, Any], today_verdict: str) -> str:
        entry_date = performance.get("entry_date") or "次日"
        return f"本段复盘按{entry_date}开盘买入、当前价格复核的短线持仓规则执行。{today_verdict}。"

    @staticmethod
    def _build_three_day_review_action_plan(performance: Dict[str, Any], sellable: bool) -> Dict[str, Any]:
        current_price = performance.get("current_price")
        if sellable:
            action_bias = "可卖出/调仓"
            entry_zone = "不新增，优先处理存量仓位"
            take_profit = "若盘中冲高可优先减仓"
            stop_loss = f"跌破{current_price:.2f}附近弱势延续" if isinstance(current_price, (int, float)) else "跌破当日弱势结构"
        else:
            action_bias = "继续观察"
            entry_zone = "已有仓位可继续观察，不因3日弱票规则卖出"
            take_profit = "结合后续强弱分批止盈"
            stop_loss = "若收益回落到3%以内且承接转弱，再评估调仓"
        return {
            "action_bias": action_bias,
            "entry_zone": entry_zone,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "holding_horizon": "3日复盘后滚动判断",
            "invalid_condition": "3日收益未超过3%或资金承接继续转弱",
        }

    def _build_market_event_contexts(
        self,
        *,
        trade_date: date,
        stock_codes: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        codes = [str(code).strip() for code in dict.fromkeys(stock_codes) if str(code or "").strip()]
        if not codes:
            return {}
        repo = getattr(getattr(self.screener, "client", None), "_raw_data_repo", None)
        if repo is None:
            return {}
        trade_date_text = trade_date.strftime("%Y%m%d")
        try:
            top_list_map = repo.get_top_list_by_trade_date(ts_codes=codes, trade_date=trade_date_text)
            limit_list_map = repo.get_limit_list_by_trade_date(ts_codes=codes, trade_date=trade_date_text)
        except Exception as exc:
            logger.warning("Failed to load market event context for Top3 report: %s", exc)
            return {}

        contexts: Dict[str, Dict[str, Any]] = {}
        for code in codes:
            top_list_rows = list(top_list_map.get(code) or [])
            limit_list_row = limit_list_map.get(code)
            contexts[code] = {
                "top_list": top_list_rows,
                "top_list_summary": self._summarize_top_list_rows(top_list_rows),
                "limit_list": limit_list_row,
                "limit_status": self._summarize_limit_list_row(limit_list_row),
            }
        return contexts

    @staticmethod
    def _summarize_top_list_rows(rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None
        def _to_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        parts: List[str] = []
        for row in rows[:3]:
            reason = str(row.get("reason") or "龙虎榜").strip()
            net_amount = _to_float(row.get("net_amount"))
            net_rate = _to_float(row.get("net_rate"))
            text = reason
            if net_amount is not None:
                text += f"，净买入{net_amount:.1f}万"
            if net_rate is not None:
                text += f"，净买率{net_rate:.1f}%"
            parts.append(text)
        return "；".join(parts)

    @staticmethod
    def _summarize_limit_list_row(row: Optional[Dict[str, Any]]) -> Optional[str]:
        if not row:
            return None
        limit_status = str(row.get("limit") or "").strip()
        open_times = row.get("open_times")
        last_time = str(row.get("last_time") or "").strip()
        parts = [limit_status or "涨跌停异动"]
        if open_times not in (None, ""):
            try:
                parts.append(f"开板{int(float(open_times or 0))}次")
            except (TypeError, ValueError):
                pass
        if last_time:
            parts.append(f"最后封板/触及时间{last_time}")
        return "，".join(parts)

    def _build_report_stock_payload(
        self,
        item: Dict[str, Any],
        ai_analyses: Dict[str, Dict[str, Any]],
        final_recommendations: Dict[str, Dict[str, Any]],
        *,
        include_fresh_moneyflow: bool = False,
        market_event_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        code = item.get("ts_code")
        analysis = ai_analyses.get(code, {})
        recommendation = final_recommendations.get(code, {})
        industry_name = item.get("industry") or recommendation.get("industry") or ""
        financial_summary = self._build_financial_yoy_summary(code)
        moneyflow_windows = self._build_moneyflow_windows(code) if include_fresh_moneyflow else {}
        recommendation_score = self._first_defined_value(
            item.get("final_display_recommendation_score"),
            item.get("recommendation_score"),
            recommendation.get("final_display_recommendation_score"),
            recommendation.get("weighted_score"),
            recommendation.get("recommendation_score"),
            recommendation.get("score"),
        )
        display_confidence = self._first_defined_value(
            item.get("display_confidence"),
            analysis.get("confidence"),
            analysis.get("overall_confidence"),
            item.get("overall_confidence"),
            item.get("ai_confidence"),
        )
        overall_confidence = self._first_defined_value(
            item.get("overall_confidence"),
            analysis.get("overall_confidence"),
            item.get("ai_confidence"),
            display_confidence,
        )
        final_display_recommendation_score = self._first_defined_value(
            item.get("final_display_recommendation_score"),
            recommendation.get("final_display_recommendation_score"),
            item.get("recommendation_score"),
            recommendation.get("weighted_score"),
            recommendation.get("recommendation_score"),
            recommendation.get("score"),
        )
        market_event_context = market_event_context or {}
        payload = {
            "ts_code": code,
            "name": item.get("name") or analysis.get("name") or code,
            "source_tag": item.get("source_tag"),
            "recommendation_score": recommendation_score,
            "overall_score": item.get("overall_score") if item.get("overall_score") is not None else self._resolve_real_overall_score(analysis, recommendation),
            "priority_score": item.get("priority_score"),
            "overall_confidence": overall_confidence,
            "display_confidence": display_confidence,
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "pct_change": item.get("pct_change", item.get("change_pct")),
            "amplitude": item.get("amplitude"),
            "volume_ratio": item.get("volume_ratio", recommendation.get("volume_ratio")),
            "turnover_rate": item.get("turnover_rate"),
            "ma20": item.get("ma20", recommendation.get("ma20")),
            "amount": item.get("amount"),
            "strategy_count": item.get("strategy_count", recommendation.get("strategy_count", 0)),
            "divergence_score": item.get("divergence_score", recommendation.get("divergence_score")),
            "strategy_consistency_label": item.get("strategy_consistency_label", recommendation.get("strategy_consistency_label")),
            "news_mentioned": item.get("news_mentioned", recommendation.get("news_mentioned", False)),
            "industry": item.get("industry", recommendation.get("industry")),
            "business_summary": item.get("business_summary") or analysis.get("business_summary") or recommendation.get("business_summary") or self._build_company_business_summary(code, industry_name),
            "latest_revenue_yoy": item.get("latest_revenue_yoy") or analysis.get("latest_revenue_yoy") or recommendation.get("latest_revenue_yoy") or financial_summary.get("latest_revenue_yoy"),
            "latest_profit_yoy": item.get("latest_profit_yoy") or analysis.get("latest_profit_yoy") or recommendation.get("latest_profit_yoy") or financial_summary.get("latest_profit_yoy"),
            "pe_ttm": item.get("pe_ttm") or analysis.get("pe_ttm") or recommendation.get("pe_ttm"),
            "industry_pe_median": item.get("industry_pe_median") or analysis.get("industry_pe_median") or recommendation.get("industry_pe_median"),
            "catalyst_summary": item.get("catalyst_summary") or analysis.get("catalyst_summary") or recommendation.get("catalyst_summary") or self._build_catalyst_summary(item, recommendation, analysis),
            "earnings_forecast": self._build_earnings_forecast_payload(analysis),
            "top_list": market_event_context.get("top_list") or [],
            "top_list_summary": market_event_context.get("top_list_summary"),
            "limit_list": market_event_context.get("limit_list"),
            "limit_status": market_event_context.get("limit_status"),
            "main_fund_flow_1d": item.get("main_fund_flow_1d") or analysis.get("main_fund_flow_1d") or recommendation.get("main_fund_flow_1d") or moneyflow_windows.get("main_fund_flow_1d"),
            "main_fund_flow_3d": item.get("main_fund_flow_3d") or analysis.get("main_fund_flow_3d") or recommendation.get("main_fund_flow_3d") or moneyflow_windows.get("main_fund_flow_3d"),
            "main_fund_flow_10d": item.get("main_fund_flow_10d") or analysis.get("main_fund_flow_10d") or recommendation.get("main_fund_flow_10d") or moneyflow_windows.get("main_fund_flow_10d"),
            "margin_balance_change_10d": item.get("margin_balance_change_10d") or analysis.get("margin_balance_change_10d") or recommendation.get("margin_balance_change_10d"),
            "industry_heat_score": item.get("industry_heat_score", recommendation.get("industry_heat_score")),
            "industry_flow_bias": item.get("industry_flow_bias", recommendation.get("industry_flow_bias", "中性")),
            "distribution_risk_score": item.get("distribution_risk_score", recommendation.get("distribution_risk_score")),
            "distribution_risk_flags": list(item.get("distribution_risk_flags") or recommendation.get("distribution_risk_flags") or []),
            "moneyflow_3d_value": item.get("moneyflow_3d_value", recommendation.get("moneyflow_3d_value")),
            "recent_large_order_net_inflow": item.get("recent_large_order_net_inflow", recommendation.get("recent_large_order_net_inflow")),
            "recent_super_large_order_net_inflow": item.get("recent_super_large_order_net_inflow", recommendation.get("recent_super_large_order_net_inflow")),
            "turnover_spike_ratio": item.get("turnover_spike_ratio", recommendation.get("turnover_spike_ratio")),
            "recent_runup_5d": item.get("recent_runup_5d", recommendation.get("recent_runup_5d")),
            "continuation_bias_score": item.get("continuation_bias_score", recommendation.get("continuation_bias_score")),
            "continuation_positive_flags": list(item.get("continuation_positive_flags") or recommendation.get("continuation_positive_flags") or []),
            "continuation_negative_flags": list(item.get("continuation_negative_flags") or recommendation.get("continuation_negative_flags") or []),
            "top3_risk_penalty": item.get("top3_risk_penalty", recommendation.get("top3_risk_penalty")),
            "short_term_contradiction_penalty": item.get("short_term_contradiction_penalty", recommendation.get("short_term_contradiction_penalty", self._build_short_term_contradiction_penalty(recommendation))),
            "final_display_recommendation_score": final_display_recommendation_score,
            "late_stage_momentum_flag": bool(item.get("late_stage_momentum_flag", recommendation.get("late_stage_momentum_flag", False))),
            "candidate_risk_blocked": bool(item.get("candidate_risk_blocked", recommendation.get("candidate_risk_blocked", False))),
            "top3_extreme_risk_blocked": bool(item.get("top3_extreme_risk_blocked", recommendation.get("top3_extreme_risk_blocked", False))),
            "top3_extreme_risk_reason": item.get("top3_extreme_risk_reason", recommendation.get("top3_extreme_risk_reason")),
            "short_term_contradiction_penalty": item.get("short_term_contradiction_penalty", recommendation.get("short_term_contradiction_penalty", self._build_short_term_contradiction_penalty(recommendation))),
            "latest_weakening_flag": bool(item.get("latest_weakening_flag", recommendation.get("latest_weakening_flag", False))),
            "high_level_pullback_flag": bool(item.get("high_level_pullback_flag", recommendation.get("high_level_pullback_flag", False))),
            "theme_support_absent_flag": bool(item.get("theme_support_absent_flag", recommendation.get("theme_support_absent_flag", False))),
            "score_change": item.get("score_change"),
            "previous_recommendation_score": item.get("previous_recommendation_score"),
            "technical_signal": item.get("technical_signal") or analysis.get("technical_signal"),
            "technical_score": item.get("technical_score") if item.get("technical_score") is not None else analysis.get("technical_score"),
            "fundamental_score": item.get("fundamental_score") if item.get("fundamental_score") is not None else analysis.get("fundamental_score"),
            "sentiment_score": item.get("sentiment_score") if item.get("sentiment_score") is not None else analysis.get("sentiment_score"),
            "news_score": item.get("news_score") if item.get("news_score") is not None else analysis.get("news_score"),
            "base_score": item.get("base_score") if item.get("base_score") is not None else analysis.get("base_score"),
            "sentiment_adjustment": item.get("sentiment_adjustment") if item.get("sentiment_adjustment") is not None else analysis.get("sentiment_adjustment"),
            "news_adjustment": item.get("news_adjustment") if item.get("news_adjustment") is not None else analysis.get("news_adjustment"),
            "score_model": item.get("score_model") or analysis.get("score_model"),
            "summary": item.get("summary") or analysis.get("summary"),
            "overview_reason": item.get("overview_reason") or self._build_overview_reason({
                **recommendation,
                **analysis,
                **item,
                "distribution_risk_flags": list(item.get("continuation_negative_flags") or item.get("distribution_risk_flags") or recommendation.get("distribution_risk_flags") or []),
            }),
            "recommendation_text": item.get("recommendation_text") or recommendation.get("recommendation") or analysis.get("recommendation"),
            "analysis": item.get("analysis") or analysis.get("summary") or recommendation.get("ai_summary") or item.get("summary"),
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
        payload.update(self._build_unified_score_fields(payload, item, analysis, recommendation))
        return payload

    @classmethod
    def _build_unified_score_fields(
        cls,
        payload: Dict[str, Any],
        item: Dict[str, Any],
        analysis: Dict[str, Any],
        recommendation: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_rank = cls._first_defined_value(
            payload.get("model_rank"),
            item.get("model_rank"),
            item.get("rerank_pool_rank"),
            analysis.get("model_rank"),
            recommendation.get("model_rank"),
            recommendation.get("rerank_pool_rank"),
        )
        model_score = cls._first_defined_value(
            payload.get("model_score"),
            item.get("model_score"),
            item.get("rerank_model_score"),
            analysis.get("model_score"),
            recommendation.get("model_score"),
            recommendation.get("rerank_model_score"),
        )
        model_blend_score = cls._first_defined_value(
            payload.get("model_blend_score"),
            item.get("model_blend_score"),
            item.get("rerank_blend_score"),
            analysis.get("model_blend_score"),
            recommendation.get("model_blend_score"),
            recommendation.get("rerank_blend_score"),
            recommendation.get("blend_score"),
        )
        recommendation_score = cls._first_defined_value(
            payload.get("recommendation_score"),
            item.get("final_display_recommendation_score"),
            item.get("recommendation_score"),
            recommendation.get("final_display_recommendation_score"),
            recommendation.get("weighted_score"),
            recommendation.get("recommendation_score"),
            recommendation.get("score"),
        )
        overall_score = cls._first_defined_value(payload.get("overall_score"), item.get("overall_score"), analysis.get("overall_score"))
        risk_score = cls._first_defined_value(
            payload.get("risk_score"),
            item.get("distribution_risk_score"),
            recommendation.get("distribution_risk_score"),
            analysis.get("distribution_risk_score"),
            analysis.get("risk_score"),
        )
        score_snapshot = {
            "model_rank": model_rank,
            "model_score": model_score,
            "model_blend_score": model_blend_score,
            "recommendation_score": recommendation_score,
            "overall_score": overall_score,
            "risk_score": risk_score,
        }
        fusion_70_30 = cls._first_defined_value(
            item.get("fusion_70_30"),
            recommendation.get("fusion_70_30"),
            (item.get("selection_reason_components") or {}).get("fusion_70_30"),
            (recommendation.get("selection_reason_components") or {}).get("fusion_70_30"),
        )
        return {
            "model_rank": model_rank,
            "model_score": model_score,
            "model_blend_score": model_blend_score,
            "recommendation_score": recommendation_score,
            "overall_score": overall_score,
            "risk_score": risk_score,
            "fusion_70_30": fusion_70_30,
            "score_snapshot": score_snapshot,
        }

    @staticmethod
    def _build_earnings_forecast_payload(analysis: Dict[str, Any]) -> Dict[str, Any]:
        signals = analysis.get("fundamental_signals") or {}
        if not isinstance(signals, dict):
            return {}
        payload = {
            "type": signals.get("forecast_type"),
            "p_change_min": signals.get("forecast_p_change_min"),
            "p_change_max": signals.get("forecast_p_change_max"),
            "summary": signals.get("forecast_summary"),
            "change_reason": signals.get("forecast_change_reason"),
        }
        return {key: value for key, value in payload.items() if value not in (None, "", [])}

    @staticmethod
    def _first_defined_value(*values: Any) -> Any:
        for value in values:
            if value not in (None, ""):
                return value
        return None

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