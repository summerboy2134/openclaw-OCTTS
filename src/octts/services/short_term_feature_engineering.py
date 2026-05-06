from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from octts.config import Settings
from octts.schemas.screener import ScreenResult
from octts.services.enhanced_screening_scheduler import (
    BACKFILL_TRAINING_CANDIDATE_LIMIT,
    EnhancedScreeningScheduler,
    LLM_REVIEW_CANDIDATE_LIMIT,
    TOP_RECOMMENDATION_LIMIT,
)
from octts.services.screening_store import ScreeningStore
from octts.services.stock_screener import StockScreener

logger = logging.getLogger(__name__)


class ShortTermFeatureEngineer:
    def __init__(
        self,
        settings: Settings,
        *,
        screener: Optional[StockScreener] = None,
        store: Optional[ScreeningStore] = None,
        scheduler: Optional[EnhancedScreeningScheduler] = None,
    ) -> None:
        self.settings = settings
        self.screener = screener or StockScreener(settings)
        self.store = store or ScreeningStore(settings)
        self.scheduler = scheduler or EnhancedScreeningScheduler(
            settings=settings,
            screener=self.screener,
            store=self.store,
        )

    def list_trade_dates(self, *, months_back: int = 6, end_date: Optional[date] = None) -> List[date]:
        anchor_date = end_date or datetime.now().date()
        start_date = anchor_date - timedelta(days=max(months_back, 1) * 31)
        trading_dates = self.screener.client.fetch_trading_dates(
            start_date=start_date.strftime("%Y%m%d"),
            end_date=anchor_date.strftime("%Y%m%d"),
        )
        results: List[date] = []
        for value in trading_dates:
            try:
                results.append(datetime.strptime(value, "%Y%m%d").date())
            except ValueError:
                continue
        return results

    def build_trade_date_pool_states(
        self,
        trade_date: date,
        *,
        candidate_limit: int = BACKFILL_TRAINING_CANDIDATE_LIMIT,
    ) -> Dict[str, Any]:
        trade_date_text = trade_date.strftime("%Y%m%d")
        market_snapshot = self.screener.client.get_or_build_screening_snapshot(trade_date_text)
        screening_results = self.scheduler._run_screening_strategies_sync_for_backfill(
            trade_date_text,
            market_snapshot=market_snapshot,
        )
        candidate_codes = self.scheduler._get_top_stocks(
            screening_results,
            limit=max(candidate_limit, self.settings.screening_top_n),
        )
        eligible_candidate_codes = self.scheduler._filter_out_tracked_and_holding_codes(candidate_codes)
        stage_pipeline = self.scheduler._build_stage_pipeline_result(
            trade_date=trade_date,
            screening_results=screening_results,
            market_snapshot=market_snapshot,
            rerank_result=self.scheduler.regression_rerank_service.rank_candidates(
                screening_results,
                trade_date=trade_date,
                coarse_limit=BACKFILL_TRAINING_CANDIDATE_LIMIT,
                analysis_limit=TOP_RECOMMENDATION_LIMIT,
                exclude_bj=True,
                rule_weight=0.3,
            ),
            baseline_candidate_codes=eligible_candidate_codes,
        )
        analysis_target_codes = stage_pipeline["analysis_target_codes"]
        ai_analyses = stage_pipeline["structured_analyses"]
        final_recommendations = stage_pipeline["final_recommendations"]
        pool_states = self.scheduler._build_recommendation_pool_states(
            trade_date=trade_date,
            screening_results=screening_results,
            final_recommendations=final_recommendations,
            candidate_codes=stage_pipeline["stage1_candidate_codes"],
        )
        logger.info(
            "Backfill feature engineering complete: trade_date=%s, strategies=%s, candidates=%s, eligible=%s, ai_targets=%s, recommendations=%s, pool_states=%s",
            trade_date.isoformat(),
            len(screening_results),
            len(candidate_codes),
            len(eligible_candidate_codes),
            len(analysis_target_codes),
            len(final_recommendations),
            len(pool_states),
        )
        return {
            "trade_date": trade_date,
            "screening_results": screening_results,
            "market_snapshot": market_snapshot,
            "candidate_codes": candidate_codes,
            "eligible_candidate_codes": eligible_candidate_codes,
            "analysis_target_codes": analysis_target_codes,
            "ai_analyses": ai_analyses,
            "final_recommendations": final_recommendations,
            "pool_states": pool_states,
        }


def _result_to_code_map(result: Optional[ScreenResult]) -> Dict[str, Any]:
    if not result:
        return {}
    return {stock.ts_code: stock for stock in result.stocks}
