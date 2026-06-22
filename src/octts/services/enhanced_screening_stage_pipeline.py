"""Shared helpers for enhanced screening scheduler mixins."""

import html
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.services.stock_screener import StockScreener
from octts.services.regression_rerank_service import RegressionRerankResult
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_constants import *

logger = logging.getLogger(__name__)


class EnhancedScreeningStagePipelineMixin:
    def _build_stage_pipeline_result(
        self,
        *,
        trade_date: date,
        screening_results: Dict[str, ScreenResult],
        market_snapshot: Dict[str, Any],
        rerank_result: RegressionRerankResult,
        baseline_candidate_codes: List[str],
        fusion_model_weight: float = 0.7,
        fusion_overall_weight: float = 0.3,
        fusion_risk_penalty_scale: float = 1.0,
        stage2_moneyflow_backfill_callback: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        stage1_candidate_codes = self._filter_out_tracked_and_holding_codes(
            rerank_result.candidate_codes or baseline_candidate_codes
        )[:MODEL_RISK_REVIEW_POOL_LIMIT]
        stage1_moneyflow_backfill = self._backfill_stage2_moneyflow_for_codes(
            stage1_candidate_codes,
            trade_date=trade_date,
        )
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
            model_weight=fusion_model_weight,
            overall_weight=fusion_overall_weight,
            risk_penalty_scale=fusion_risk_penalty_scale,
        )
        stock_name_map = self._build_stock_name_map(screening_results)
        st_excluded_from_stage2 = 0
        stage2_ranked_codes: List[str] = []
        for code in fusion_ranked_codes:
            payload = stage2_recommendations.get(code) or {}
            if self._is_st_stock_for_final_veto(code=code, payload=payload, stock_name_map=stock_name_map):
                if payload:
                    payload["top3_st_excluded"] = True
                    payload["top3_st_excluded_reason"] = "stock_name_contains_ST"
                    payload["selection_stage"] = "stage2_top50_st_veto"
                    payload["selection_reason_components"] = {
                        **dict(payload.get("selection_reason_components") or {}),
                        "top3_st_excluded": True,
                        "top3_st_excluded_reason": "stock_name_contains_ST",
                    }
                st_excluded_from_stage2 += 1
                continue
            stage2_ranked_codes.append(code)
        stage2_top20_codes = stage2_ranked_codes[:STAGE2_TOP20_LIMIT]
        logger.info(
            "Stage2 ST filter complete: trade_date=%s, ranked=%s, st_excluded=%s, stage2=%s",
            trade_date.isoformat(),
            len(fusion_ranked_codes),
            st_excluded_from_stage2,
            len(stage2_top20_codes),
        )
        stage2_recommendations = self._build_structured_stage_selection_metadata(
            recommendations=stage2_recommendations,
            rerank_metadata=rerank_result.metadata_by_code or {},
            stage2_top20_codes=stage2_top20_codes,
            stage3_top3_codes=[],
        )
        stage2_moneyflow_backfill = None
        if stage2_moneyflow_backfill_callback is not None:
            stage2_moneyflow_backfill = stage2_moneyflow_backfill_callback(stage2_top20_codes)
        else:
            stage2_moneyflow_backfill = self._backfill_stage2_moneyflow_for_codes(
                stage2_top20_codes,
                trade_date=trade_date,
            )
        stage3_recommendations = self._apply_stage3_moneyflow_rerank(
            trade_date=trade_date,
            stage2_recommendations=stage2_recommendations,
            stage2_top20_codes=stage2_top20_codes,
        )
        stage3_recommendations = self._apply_stage3_close_auction_rerank(
            trade_date=trade_date,
            stage3_recommendations=stage3_recommendations,
            stage2_top20_codes=stage2_top20_codes,
        )
        for code in stage2_top20_codes:
            payload = stage3_recommendations.get(code)
            if payload and payload.get("selection_stage") == "stage3_final_top3":
                payload["selection_stage"] = "stage2_top20_pre_moneyflow"
        stage3_ranked_codes = self._rank_stage3_candidates_by_score(
            candidate_codes=stage2_top20_codes or stage1_candidate_codes,
            recommendations=stage3_recommendations,
        )
        stage3_top3_codes = self._select_model_top3_with_extreme_risk_veto(
            model_candidate_codes=stage3_ranked_codes,
            recommendations=stage3_recommendations,
            stock_name_map=stock_name_map,
        )
        stage3_top3_veto_diagnostics = self._build_stage3_top3_veto_diagnostics(
            ranked_codes=stage3_ranked_codes,
            selected_codes=stage3_top3_codes,
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
            "stage3_top3_veto_diagnostics": stage3_top3_veto_diagnostics,
            "analysis_target_codes": analysis_target_codes,
            "stage1_moneyflow_backfill": stage1_moneyflow_backfill,
            "stage2_moneyflow_backfill": stage2_moneyflow_backfill,
            "fusion_parameters": {
                "model_weight": round(float(fusion_model_weight), 6),
                "overall_weight": round(float(fusion_overall_weight), 6),
                "risk_penalty_scale": round(float(fusion_risk_penalty_scale), 6),
            },
        }

    @classmethod
    def _rank_stage1_candidates_by_fusion(
        cls,
        *,
        candidate_codes: List[str],
        recommendations: Dict[str, Dict[str, Any]],
        rerank_metadata: Dict[str, Dict[str, Any]],
        model_weight: float = 0.7,
        overall_weight: float = 0.3,
        risk_penalty_scale: float = 1.0,
    ) -> List[str]:
        records = {
            code: recommendations.get(code)
            for code in candidate_codes
            if recommendations.get(code)
        }
        if not records:
            return []

        model_values: Dict[str, Optional[float]] = {}
        rank_values: Dict[str, Optional[float]] = {}
        for code, payload in records.items():
            metadata = rerank_metadata.get(code) or {}
            for metadata_key in (
                "price_position_20d",
                "recent_runup_5d",
                "turnover_spike_ratio",
                "weak_market_flag",
                "high_position_flag",
                "high_position_acceleration_flag",
                "weak_market_high_position_flag",
                "market_return_1d",
                "market_return_3d",
                "market_up_ratio_1d",
                "market_up_ratio_3d_avg",
                "market_up_days_5d",
            ):
                if payload.get(metadata_key) is None and metadata.get(metadata_key) is not None:
                    payload[metadata_key] = metadata.get(metadata_key)
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
        model_weight = max(0.0, float(model_weight))
        overall_weight = max(0.0, float(overall_weight))
        weight_total = model_weight + overall_weight
        if weight_total <= 0:
            model_weight = 0.7
            overall_weight = 0.3
            weight_total = 1.0
        normalized_model_weight = model_weight / weight_total
        normalized_overall_weight = overall_weight / weight_total
        risk_penalty_scale = max(0.0, float(risk_penalty_scale))

        for code, payload in records.items():
            model_score = model_norm.get(code)
            overall_score = overall_norm.get(code)
            payload["model_score_norm"] = model_score
            payload["overall_score_norm"] = overall_score
            if model_score is not None and overall_score is not None:
                fusion_score = round(
                    model_score * normalized_model_weight + overall_score * normalized_overall_weight,
                    6,
                )
                risk_penalty, risk_flags = cls._build_stage1_fusion_risk_penalty(payload)
                scaled_risk_penalty = round(risk_penalty * risk_penalty_scale, 6)
                risk_adjusted_fusion_score = round(max(0.0, fusion_score - scaled_risk_penalty), 6)
                payload["fusion_70_30"] = fusion_score
                payload["fusion_score"] = fusion_score
                payload["fusion_model_weight"] = round(normalized_model_weight, 6)
                payload["fusion_overall_weight"] = round(normalized_overall_weight, 6)
                payload["fusion_risk_penalty_scale"] = round(risk_penalty_scale, 6)
                payload["stage1_fusion_raw_risk_penalty"] = risk_penalty
                payload["stage1_fusion_risk_penalty"] = scaled_risk_penalty
                payload["stage1_fusion_scaled_risk_penalty"] = scaled_risk_penalty
                payload["stage1_fusion_risk_flags"] = risk_flags
                payload["risk_adjusted_fusion_score"] = risk_adjusted_fusion_score
                payload["top3_ranking_strategy"] = (
                    "stage1_risk_adjusted_fusion_"
                    f"{normalized_model_weight:.2f}_{normalized_overall_weight:.2f}_risk_{risk_penalty_scale:.2f}"
                )
            else:
                payload["fusion_70_30"] = None
                payload["fusion_score"] = None
                payload["fusion_model_weight"] = round(normalized_model_weight, 6)
                payload["fusion_overall_weight"] = round(normalized_overall_weight, 6)
                payload["fusion_risk_penalty_scale"] = round(risk_penalty_scale, 6)
                payload["stage1_fusion_raw_risk_penalty"] = None
                payload["stage1_fusion_risk_penalty"] = None
                payload["stage1_fusion_scaled_risk_penalty"] = None
                payload["stage1_fusion_risk_flags"] = []
                payload["risk_adjusted_fusion_score"] = None

            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "model_score_norm": model_score,
                "overall_score_norm": overall_score,
                "fusion_70_30": payload.get("fusion_70_30"),
                "fusion_score": payload.get("fusion_score"),
                "fusion_model_weight": payload.get("fusion_model_weight"),
                "fusion_overall_weight": payload.get("fusion_overall_weight"),
                "fusion_risk_penalty_scale": payload.get("fusion_risk_penalty_scale"),
                "stage1_fusion_raw_risk_penalty": payload.get("stage1_fusion_raw_risk_penalty"),
                "stage1_fusion_risk_penalty": payload.get("stage1_fusion_risk_penalty"),
                "stage1_fusion_scaled_risk_penalty": payload.get("stage1_fusion_scaled_risk_penalty"),
                "stage1_fusion_risk_flags": payload.get("stage1_fusion_risk_flags"),
                "risk_adjusted_fusion_score": payload.get("risk_adjusted_fusion_score"),
                "top3_ranking_strategy": payload.get("top3_ranking_strategy"),
            }

        def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[Any, ...]:
            code, payload = item
            risk_adjusted_fusion_score = cls._safe_float(
                cls._first_defined_value(payload.get("risk_adjusted_fusion_score"), payload.get("fusion_70_30"))
            )
            raw_fusion_score = cls._safe_float(payload.get("fusion_70_30"))
            fallback_score = cls._safe_float(
                cls._first_defined_value(payload.get("stage3_final_score"), payload.get("score"), 0.0)
            ) or 0.0
            return (
                bool(payload.get("candidate_risk_blocked", False)),
                bool(payload.get("relay_candidate_veto", False)),
                risk_adjusted_fusion_score is None,
                -(risk_adjusted_fusion_score if risk_adjusted_fusion_score is not None else 0.0),
                -(raw_fusion_score if raw_fusion_score is not None else 0.0),
                cls._safe_float(payload.get("rerank_pool_rank")) or 999999.0,
                -fallback_score,
                code,
            )

        return [code for code, _ in sorted(records.items(), key=sort_key)]

    @classmethod
    def _build_stage1_fusion_risk_penalty(cls, payload: Dict[str, Any]) -> Tuple[float, List[str]]:
        distribution_risk_score = cls._safe_float(payload.get("distribution_risk_score")) or 0.0
        price_position = cls._safe_float(payload.get("price_position_20d")) or 0.0
        recent_runup_5d = cls._safe_float(payload.get("recent_runup_5d")) or 0.0
        turnover_spike_ratio = cls._safe_float(payload.get("turnover_spike_ratio")) or 0.0
        moneyflow_3d_value = cls._safe_float(payload.get("moneyflow_3d_value")) or 0.0
        penalty = min(distribution_risk_score * 8.0, 28.0)
        flags: List[str] = []
        if distribution_risk_score > 0:
            flags.append("distribution_risk_score")
        if price_position >= DISTRIBUTION_PRICE_POSITION_HIGH:
            penalty += 8.0
            flags.append("high_price_position")
        if recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH:
            penalty += 6.0
            flags.append("recent_runup_high")
        if turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_HIGH:
            penalty += 4.0
            flags.append("turnover_spike")
        if bool(payload.get("late_stage_momentum_flag", False)):
            penalty += 12.0
            flags.append("late_stage_momentum")
        if bool(payload.get("candidate_risk_blocked", False)):
            penalty += 12.0
            flags.append("candidate_risk_blocked")
        if bool(payload.get("relay_candidate_veto", False)):
            penalty += 16.0
            flags.append("relay_candidate_veto")
        if bool(payload.get("high_level_pullback_flag", False)):
            penalty += 10.0
            flags.append("high_level_pullback")
        if bool(payload.get("unsupported_high_position_flag", False)) or bool(payload.get("theme_support_absent_flag", False)):
            penalty += 6.0
            flags.append("unsupported_high_position")
        if price_position >= DISTRIBUTION_PRICE_POSITION_HIGH and moneyflow_3d_value <= 0:
            penalty += 8.0
            flags.append("high_position_weak_moneyflow")
        return round(max(0.0, penalty), 6), flags

    @classmethod
    def _normalize_descending_score_values(cls, values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
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

    @classmethod
    def _rank_values_to_percentile(cls, ranks: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
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

    def _build_stage3_top3_veto_diagnostics(
        self,
        *,
        ranked_codes: List[str],
        selected_codes: List[str],
        recommendations: Dict[str, Dict[str, Any]],
        stock_name_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        selected_set = set(selected_codes)
        stock_name_map = stock_name_map or {}
        diagnostics: List[Dict[str, Any]] = []
        for rank, code in enumerate(ranked_codes, start=1):
            payload = recommendations.get(code) or {}
            quality_floor_reason = payload.get("top3_quality_floor_reason") or self._get_top3_quality_floor_reason(payload)
            extreme_risk_reason = payload.get("top3_extreme_risk_reason") or self._get_top3_extreme_risk_reason(payload)
            st_excluded = bool(payload.get("top3_st_excluded", False)) or self._is_st_stock_for_final_veto(
                code=code,
                payload=payload,
                stock_name_map=stock_name_map,
            )
            diagnostics.append({
                "rank": rank,
                "ts_code": code,
                "selected": code in selected_set,
                "stage3_final_score": payload.get("stage3_final_score"),
                "structured_rank_position": payload.get("structured_rank_position"),
                "rerank_pool_rank": payload.get("rerank_pool_rank"),
                "selection_stage": payload.get("selection_stage"),
                "quality_floor_reason": quality_floor_reason,
                "extreme_risk_reason": extreme_risk_reason,
                "st_excluded": st_excluded,
                "price_position_20d": payload.get("price_position_20d"),
                "recent_runup_5d": payload.get("recent_runup_5d"),
                "pct_change": payload.get("pct_change"),
                "moneyflow_3d_value": payload.get("moneyflow_3d_value"),
                "recent_large_order_net_inflow": payload.get("recent_large_order_net_inflow"),
                "recent_super_large_order_net_inflow": payload.get("recent_super_large_order_net_inflow"),
                "super_large_order_net_inflow_negative_days_3d": payload.get("super_large_order_net_inflow_negative_days_3d"),
                "risk_adjusted_fusion_score": payload.get("risk_adjusted_fusion_score"),
                "overall_score_norm": payload.get("overall_score_norm"),
                "stage3_moneyflow_score": payload.get("stage3_moneyflow_score"),
                "stage3_close_auction_score": payload.get("stage3_close_auction_score"),
                "stage3_moneyflow_veto": payload.get("stage3_moneyflow_veto"),
                "stage3_close_auction_veto": payload.get("stage3_close_auction_veto"),
                "stage3_close_auction_veto_softened": payload.get("stage3_close_auction_veto_softened"),
                "high_level_failed_trend_flag": payload.get("high_level_failed_trend_flag"),
                "high_level_failed_trend_signal": payload.get("high_level_failed_trend_signal"),
            })
        return diagnostics

    @staticmethod
    def _get_top3_final_moneyflow_veto_reason(payload: Dict[str, Any]) -> Optional[str]:
        def safe_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        pct_change = safe_float(payload.get("pct_change"))
        moneyflow_3d_value = safe_float(payload.get("moneyflow_3d_value"))
        recent_large_order_net_inflow = safe_float(payload.get("recent_large_order_net_inflow"))
        risk_adjusted_fusion_score = safe_float(payload.get("risk_adjusted_fusion_score"))
        stage3_moneyflow_score = safe_float(payload.get("stage3_moneyflow_score"))

        if (
            pct_change is not None
            and pct_change >= TOP3_LIMIT_UP_MONEYFLOW_DIVERGENCE_PCT_CHANGE_MIN
            and moneyflow_3d_value is not None
            and moneyflow_3d_value < 0
            and recent_large_order_net_inflow is not None
            and recent_large_order_net_inflow < 0
        ):
            return "limit_up_moneyflow_divergence_top3_veto"

        if (
            risk_adjusted_fusion_score is not None
            and risk_adjusted_fusion_score < TOP3_LOW_QUALITY_NEGATIVE_MONEYFLOW_FUSION_MAX
            and stage3_moneyflow_score is not None
            and stage3_moneyflow_score < 0
        ):
            return "low_quality_negative_moneyflow_top3_veto"

        if bool(payload.get("top3_quality_floor_degraded", False)) and stage3_moneyflow_score is not None and stage3_moneyflow_score < 0:
            return "quality_floor_degraded_negative_moneyflow_top3_veto"

        return None

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
        limit_up_continuation_count = 0
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
            quality_floor_reason = self._get_top3_quality_floor_reason(payload)
            if quality_floor_reason:
                payload["top3_quality_floor_blocked"] = False
                payload["top3_quality_floor_degraded"] = True
                payload["top3_quality_floor_reason"] = quality_floor_reason
                payload["selection_stage"] = "model_top100_quality_floor_degraded"
                payload["selection_reason_components"] = {
                    **dict(payload.get("selection_reason_components") or {}),
                    "top3_quality_floor_blocked": False,
                    "top3_quality_floor_degraded": True,
                    "top3_quality_floor_reason": quality_floor_reason,
                    "top3_quality_floor_sort_penalty": TOP3_QUALITY_FLOOR_SORT_PENALTY,
                    "top3_quality_floor_fusion_min": TOP3_QUALITY_FLOOR_FUSION_MIN,
                    "top3_quality_floor_overall_min": TOP3_QUALITY_FLOOR_OVERALL_MIN,
                    "risk_adjusted_fusion_score": payload.get("risk_adjusted_fusion_score"),
                    "overall_score_norm": payload.get("overall_score_norm"),
                    "stage3_moneyflow_score": payload.get("stage3_moneyflow_score"),
                    "stage3_close_auction_score": payload.get("stage3_close_auction_score"),
                }
            else:
                payload["top3_quality_floor_blocked"] = False
                payload["top3_quality_floor_degraded"] = False
                payload["top3_quality_floor_reason"] = None
            extreme_risk_reason = self._get_top3_extreme_risk_reason(payload)
            final_veto_reason = self._get_top3_final_moneyflow_veto_reason(payload)
            if final_veto_reason and not extreme_risk_reason:
                extreme_risk_reason = final_veto_reason
            limit_up_continuation_allowed = False
            if (
                extreme_risk_reason == "near_limit_up_pct_change"
                and limit_up_continuation_count < TOP3_LIMIT_UP_CONTINUATION_MAX_COUNT
                and self._is_limit_up_continuation_top3_eligible(payload)
            ):
                limit_up_continuation_allowed = True
                extreme_risk_reason = None
            if extreme_risk_reason:
                payload["top3_extreme_risk_blocked"] = True
                payload["top3_extreme_risk_reason"] = extreme_risk_reason
                payload["top3_distribution_risk_veto"] = extreme_risk_reason in {
                    "distribution_risk_score_top3_cap",
                    "late_stage_momentum_top3_veto",
                    "high_level_pullback_top3_veto",
                    "unsupported_high_position_weak_moneyflow_top3_veto",
                    "weak_market_high_position_top3_veto",
                    "high_position_super_large_outflow_top3_veto",
                    "rebound_distribution_super_large_outflow_top3_veto",
                    "deep_drawdown_rebound_top3_veto",
                    "limit_up_moneyflow_divergence_top3_veto",
                    "low_quality_negative_moneyflow_top3_veto",
                    "quality_floor_degraded_negative_moneyflow_top3_veto",
                }
                payload["weak_market_high_position_top3_veto"] = (
                    extreme_risk_reason == "weak_market_high_position_top3_veto"
                )
                payload["late_stage_risk_veto_reason"] = extreme_risk_reason
                payload["selection_stage"] = "model_top100_extreme_risk_veto"
                payload["selection_reason_components"] = {
                    **dict(payload.get("selection_reason_components") or {}),
                    "top3_distribution_risk_veto": payload.get("top3_distribution_risk_veto"),
                    "near_limit_up_top3_veto": extreme_risk_reason == "near_limit_up_pct_change",
                    "weak_market_high_position_top3_veto": payload.get("weak_market_high_position_top3_veto"),
                    "late_stage_risk_veto_reason": extreme_risk_reason,
                    "price_position_20d": payload.get("price_position_20d"),
                    "recent_runup_5d": payload.get("recent_runup_5d"),
                    "pct_change": payload.get("pct_change"),
                    "super_large_order_net_inflow_negative_days_3d": payload.get("super_large_order_net_inflow_negative_days_3d"),
                    "top3_high_position_event_risk_threshold": TOP3_HIGH_POSITION_EVENT_RISK_THRESHOLD,
                    "top3_high_position_event_runup_near_threshold": TOP3_HIGH_POSITION_EVENT_RUNUP_NEAR_THRESHOLD,
                    "top3_high_position_event_pct_change_min": TOP3_HIGH_POSITION_EVENT_PCT_CHANGE_MIN,
                    "top3_super_large_outflow_negative_days_min": TOP3_SUPER_LARGE_OUTFLOW_NEGATIVE_DAYS_MIN,
                    "recent_super_large_order_net_inflow": payload.get("recent_super_large_order_net_inflow"),
                    "recent_large_order_net_inflow": payload.get("recent_large_order_net_inflow"),
                    "moneyflow_3d_value": payload.get("moneyflow_3d_value"),
                    "stage3_moneyflow_score": payload.get("stage3_moneyflow_score"),
                    "risk_adjusted_fusion_score": payload.get("risk_adjusted_fusion_score"),
                    "top3_limit_up_moneyflow_divergence_pct_change_min": TOP3_LIMIT_UP_MONEYFLOW_DIVERGENCE_PCT_CHANGE_MIN,
                    "top3_low_quality_negative_moneyflow_fusion_max": TOP3_LOW_QUALITY_NEGATIVE_MONEYFLOW_FUSION_MAX,
                    "top3_rebound_distribution_runup_min": TOP3_REBOUND_DISTRIBUTION_RUNUP_MIN,
                    "top3_rebound_distribution_pct_change_min": TOP3_REBOUND_DISTRIBUTION_PCT_CHANGE_MIN,
                    "limit_up_continuation_allowed": False,
                    "limit_up_continuation_reject_reason": self._get_limit_up_continuation_reject_reason(payload)
                    if extreme_risk_reason == "near_limit_up_pct_change"
                    else None,
                    "deep_drawdown_rebound_flag": payload.get("deep_drawdown_rebound_flag"),
                    "deep_drawdown_rebound_signal": payload.get("deep_drawdown_rebound_signal"),
                }
                vetoed_count += 1
                continue
            payload["top3_extreme_risk_blocked"] = False
            payload["top3_extreme_risk_reason"] = None
            payload["top3_quality_floor_blocked"] = False
            payload["top3_quality_floor_degraded"] = bool(quality_floor_reason)
            payload["top3_quality_floor_reason"] = quality_floor_reason
            payload["top3_distribution_risk_veto"] = False
            payload["near_limit_up_top3_veto"] = False
            payload["limit_up_continuation_allowed"] = limit_up_continuation_allowed
            payload["limit_up_continuation_reject_reason"] = None
            payload["weak_market_high_position_top3_veto"] = False
            payload["late_stage_risk_veto_reason"] = None
            payload["top3_st_excluded"] = False
            payload["top3_st_excluded_reason"] = None
            if len(selected_codes) < TODAY_TOP_LIMIT:
                selected_codes.append(code)
                if limit_up_continuation_allowed:
                    limit_up_continuation_count += 1

        selected_set = set(selected_codes)
        for code, payload in recommendations.items():
            if code not in selected_set:
                continue
            payload["top3_st_excluded"] = False
            payload["top3_st_excluded_reason"] = None
            payload["selection_stage"] = "stage3_final_top3"
            selected_quality_floor_reason = self._get_top3_quality_floor_reason(payload)
            payload["selection_reason"] = (
                f"model_rank={payload.get('rerank_pool_rank')}; "
                f"fusion_70_30={payload.get('fusion_70_30')}; "
                f"stage3_final_score={payload.get('stage3_final_score')}; "
                f"close_auction_score={payload.get('stage3_close_auction_score')}; "
                "top3_extreme_risk_veto=False; "
                f"top3_quality_floor_degraded={bool(selected_quality_floor_reason)}; "
                "weak_market_high_position_top3_veto=False; "
                "top3_st_excluded=False"
            )
            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "rerank_pool_rank": payload.get("rerank_pool_rank"),
                "fusion_70_30": payload.get("fusion_70_30"),
                "stage3_final_score": payload.get("stage3_final_score"),
                "stage3_close_auction_score": payload.get("stage3_close_auction_score"),
                "stage3_close_auction_veto": payload.get("stage3_close_auction_veto"),
                "stage3_close_auction_veto_softened": payload.get("stage3_close_auction_veto_softened"),
                "high_level_failed_trend_flag": payload.get("high_level_failed_trend_flag"),
                "high_level_failed_trend_signal": payload.get("high_level_failed_trend_signal"),
                "top3_ranking_strategy": payload.get("top3_ranking_strategy"),
                "top3_extreme_risk_veto": False,
                "top3_quality_floor_veto": False,
                "top3_quality_floor_blocked": False,
                "top3_quality_floor_degraded": bool(selected_quality_floor_reason),
                "top3_quality_floor_reason": selected_quality_floor_reason,
                "price_position_20d": payload.get("price_position_20d"),
                "recent_runup_5d": payload.get("recent_runup_5d"),
                "pct_change": payload.get("pct_change"),
                "deep_drawdown_rebound_flag": payload.get("deep_drawdown_rebound_flag"),
                "deep_drawdown_rebound_signal": payload.get("deep_drawdown_rebound_signal"),
                "super_large_order_net_inflow_negative_days_3d": payload.get("super_large_order_net_inflow_negative_days_3d"),
                "top3_distribution_risk_veto": False,
                "near_limit_up_top3_veto": False,
                "limit_up_continuation_allowed": bool(payload.get("limit_up_continuation_allowed", False)),
                "limit_up_continuation_max_count": TOP3_LIMIT_UP_CONTINUATION_MAX_COUNT,
                "weak_market_high_position_top3_veto": False,
                "late_stage_risk_veto_reason": None,
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

    @classmethod
    def _is_limit_up_continuation_top3_eligible(cls, payload: Dict[str, Any]) -> bool:
        return cls._get_limit_up_continuation_reject_reason(payload) is None

    @classmethod
    def _get_limit_up_continuation_reject_reason(cls, payload: Dict[str, Any]) -> Optional[str]:
        pct_change = cls._safe_float(payload.get("pct_change"))
        if pct_change is None:
            pct_change = cls._safe_float(payload.get("pct_chg")) or 0.0
        if pct_change < TOP3_NEAR_LIMIT_UP_PCT_CHANGE_MIN:
            return "not_near_limit_up"
        rerank_rank = cls._safe_float(payload.get("rerank_pool_rank"))
        if rerank_rank is not None and rerank_rank > TOP3_LIMIT_UP_CONTINUATION_RERANK_MAX:
            return "rerank_rank_too_low"
        distribution_risk_score = cls._safe_float(payload.get("distribution_risk_score")) or 0.0
        if distribution_risk_score > TOP3_LIMIT_UP_CONTINUATION_RISK_MAX:
            return "distribution_risk_score_too_high"
        if bool(payload.get("relay_candidate_veto", False)):
            return "relay_candidate_veto"
        if bool(payload.get("stage3_moneyflow_veto", False)):
            return "stage3_moneyflow_veto"
        if bool(payload.get("stage3_close_auction_veto", False)) and not bool(
            payload.get("stage3_close_auction_veto_softened", False)
        ):
            return "stage3_close_auction_veto"
        if bool(payload.get("one_word_limit_flag", False)):
            return "one_word_limit_unfillable"
        open_times = cls._safe_float(payload.get("relay_open_times"))
        if open_times is not None and open_times > TOP3_LIMIT_UP_CONTINUATION_OPEN_TIMES_MAX:
            return "relay_open_times_too_many"
        last_time = str(payload.get("relay_limit_last_time") or "").strip()
        if last_time.isdigit() and int(last_time) >= TOP3_LIMIT_UP_CONTINUATION_LATE_SEAL_TIME_MIN:
            return "late_limit_seal"
        moneyflow_3d = cls._safe_float(payload.get("moneyflow_3d_value")) or 0.0
        large_order = cls._safe_float(payload.get("recent_large_order_net_inflow")) or 0.0
        super_large_order = cls._safe_float(payload.get("recent_super_large_order_net_inflow")) or 0.0
        if moneyflow_3d <= 0:
            return "moneyflow_3d_not_positive"
        if large_order <= 0 and super_large_order <= 0:
            return "large_order_not_positive"
        if int(cls._safe_float(payload.get("super_large_order_net_inflow_negative_days_3d")) or 0) >= TOP3_SUPER_LARGE_OUTFLOW_NEGATIVE_DAYS_MIN:
            return "super_large_outflow_negative_days"
        if bool(payload.get("late_stage_momentum_flag", False)):
            return "late_stage_momentum"
        if bool(payload.get("high_level_pullback_flag", False)):
            return "high_level_pullback"
        if bool(payload.get("deep_drawdown_rebound_flag", False)):
            return "deep_drawdown_rebound"
        if bool(payload.get("unsupported_high_position_flag", False)) and moneyflow_3d < 5000:
            return "unsupported_high_position_weak_moneyflow"
        return None

    def _backfill_stage2_moneyflow_for_codes(
        self,
        ts_codes: List[str],
        *,
        trade_date: date,
    ) -> Dict[str, Any]:
        trade_date_text = trade_date.strftime("%Y%m%d")
        target_codes = [str(code or "").strip().upper() for code in dict.fromkeys(ts_codes) if str(code or "").strip()]
        if not target_codes:
            return {"candidate_codes": 0, "pending_codes": 0, "fetched_codes": 0, "inserted_rows": 0, "missing_codes": []}

        existing = self.market_raw_data_repo.get_moneyflow_summaries_by_trade_date(
            ts_codes=target_codes,
            trade_date=trade_date_text,
            lookback_days=3,
        )
        pending_codes = [
            code
            for code in target_codes
            if not existing.get(code) or bool(existing.get(code, {}).get("stale_for_trade_date", False))
        ]
        fetched_codes = 0
        fetched_rows = 0
        inserted_rows = 0
        missing_codes: List[str] = []
        for code in pending_codes:
            try:
                rows = self.screener.client.fetch_moneyflow(code, trade_date=trade_date_text)
            except Exception as exc:
                logger.warning("Stage2 moneyflow backfill failed: trade_date=%s, ts_code=%s, error=%s", trade_date_text, code, exc)
                rows = []
            fetched_rows += len(rows)
            if rows:
                fetched_codes += 1
                try:
                    inserted_rows += self.market_raw_data_repo.save_moneyflow(rows)
                except Exception as exc:
                    logger.warning("Stage2 moneyflow save failed: trade_date=%s, ts_code=%s, error=%s", trade_date_text, code, exc)
            else:
                missing_codes.append(code)
        result = {
            "candidate_codes": len(target_codes),
            "pending_codes": len(pending_codes),
            "fetched_codes": fetched_codes,
            "fetched_rows": fetched_rows,
            "inserted_rows": inserted_rows,
            "missing_codes": missing_codes,
        }
        logger.info("Stage2 moneyflow backfill complete: trade_date=%s, result=%s", trade_date_text, result)
        return result

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
        moneyflow_10d_summary_map = self.market_raw_data_repo.get_moneyflow_summaries_by_trade_date(
            ts_codes=stage2_top20_codes,
            trade_date=trade_date.strftime("%Y%m%d"),
            lookback_days=10,
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
            moneyflow_10d_summary = moneyflow_10d_summary_map.get(code) or {}
            moneyflow_missing = not bool(moneyflow_summary) or bool(moneyflow_summary.get("moneyflow_data_missing", False))
            moneyflow_stale = bool(moneyflow_summary.get("stale_for_trade_date", False)) or bool(
                moneyflow_summary.get("moneyflow_data_stale", False)
            )
            recent_3d_net_inflow = float(moneyflow_summary.get("recent_3d_net_inflow") or payload.get("moneyflow_3d_value") or 0.0)
            recent_large_order_net_inflow = float(moneyflow_summary.get("recent_large_order_net_inflow") or payload.get("recent_large_order_net_inflow") or 0.0)
            recent_super_large_order_net_inflow = float(moneyflow_summary.get("recent_super_large_order_net_inflow") or payload.get("recent_super_large_order_net_inflow") or 0.0)
            recent_10d_net_inflow = float(moneyflow_10d_summary.get("recent_3d_net_inflow") or payload.get("moneyflow_10d_value") or 0.0)
            recent_10d_large_order_net_inflow = float(moneyflow_10d_summary.get("recent_large_order_net_inflow") or payload.get("large_order_net_inflow_10d") or 0.0)
            recent_10d_super_large_order_net_inflow = float(moneyflow_10d_summary.get("recent_super_large_order_net_inflow") or payload.get("super_large_order_net_inflow_10d") or 0.0)
            moneyflow_10d_rows = int(moneyflow_10d_summary.get("rows") or payload.get("moneyflow_10d_rows") or 0)
            super_large_order_net_inflow_negative_days_3d = int(
                moneyflow_summary.get("super_large_order_net_inflow_negative_days_3d")
                or payload.get("super_large_order_net_inflow_negative_days_3d")
                or 0
            )
            stage3_moneyflow_score = 0.0
            stage3_moneyflow_flags: List[str] = []
            stage3_moneyflow_risks: List[str] = []
            if moneyflow_missing:
                stage3_moneyflow_score -= 1.5
                stage3_moneyflow_risks.append("近3日资金流缺失")
            elif moneyflow_stale:
                stage3_moneyflow_score -= 1.0
                stage3_moneyflow_risks.append("资金流数据滞后")
            elif recent_3d_net_inflow > 0:
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
                or moneyflow_missing
                or moneyflow_stale
            )
            stage3_moneyflow_adjustment = round(stage3_moneyflow_score * 1.2, 4)
            stage3_risk_penalty, stage3_risk_flags = self._build_stage3_late_risk_penalty(
                payload,
                recent_3d_net_inflow=recent_3d_net_inflow,
                recent_large_order_net_inflow=recent_large_order_net_inflow,
                recent_super_large_order_net_inflow=recent_super_large_order_net_inflow,
            )
            stage3_final_score = round(
                max(
                    0.0,
                    float(payload.get("score") or 0.0)
                    + stage3_moneyflow_adjustment
                    - stage3_risk_penalty,
                ),
                4,
            )
            payload["moneyflow_3d_value"] = round(recent_3d_net_inflow, 2)
            payload["recent_large_order_net_inflow"] = round(recent_large_order_net_inflow, 2)
            payload["recent_super_large_order_net_inflow"] = round(recent_super_large_order_net_inflow, 2)
            payload["moneyflow_10d_value"] = round(recent_10d_net_inflow, 2)
            payload["large_order_net_inflow_10d"] = round(recent_10d_large_order_net_inflow, 2)
            payload["super_large_order_net_inflow_10d"] = round(recent_10d_super_large_order_net_inflow, 2)
            payload["moneyflow_10d_rows"] = moneyflow_10d_rows
            payload["super_large_order_net_inflow_negative_days_3d"] = super_large_order_net_inflow_negative_days_3d
            payload["moneyflow_data_missing"] = moneyflow_missing
            payload["moneyflow_data_stale"] = moneyflow_stale
            payload["stage3_moneyflow_score"] = round(stage3_moneyflow_score, 4)
            payload["stage3_moneyflow_adjustment"] = stage3_moneyflow_adjustment
            payload["stage3_late_risk_penalty"] = stage3_risk_penalty
            payload["stage3_late_risk_flags"] = stage3_risk_flags
            payload["stage3_moneyflow_flags"] = stage3_moneyflow_flags
            payload["stage3_moneyflow_risks"] = stage3_moneyflow_risks
            payload["stage3_moneyflow_veto"] = moneyflow_veto
            payload["stage3_final_score"] = stage3_final_score
            quality_floor_penalty = self._get_top3_quality_floor_penalty(payload)
            distribution_risk_cap_penalty = self._get_top3_distribution_risk_cap_penalty(payload)
            final_selection_score = self._get_effective_stage3_selection_score(payload)
            payload["top3_quality_floor_sort_penalty"] = quality_floor_penalty
            payload["top3_distribution_risk_cap_penalty"] = distribution_risk_cap_penalty
            payload["final_selection_score"] = final_selection_score
            payload["structured_rank_score"] = final_selection_score
            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "moneyflow_3d_value": round(recent_3d_net_inflow, 2),
                "recent_large_order_net_inflow": round(recent_large_order_net_inflow, 2),
                "recent_super_large_order_net_inflow": round(recent_super_large_order_net_inflow, 2),
                "moneyflow_10d_value": round(recent_10d_net_inflow, 2),
                "large_order_net_inflow_10d": round(recent_10d_large_order_net_inflow, 2),
                "super_large_order_net_inflow_10d": round(recent_10d_super_large_order_net_inflow, 2),
                "moneyflow_10d_rows": moneyflow_10d_rows,
                "super_large_order_net_inflow_negative_days_3d": super_large_order_net_inflow_negative_days_3d,
                "moneyflow_data_missing": moneyflow_missing,
                "moneyflow_data_stale": moneyflow_stale,
                "stage3_moneyflow_score": round(stage3_moneyflow_score, 4),
                "stage3_moneyflow_adjustment": stage3_moneyflow_adjustment,
                "stage3_late_risk_penalty": stage3_risk_penalty,
                "stage3_late_risk_flags": stage3_risk_flags,
                "stage3_moneyflow_veto": moneyflow_veto,
                "top3_quality_floor_sort_penalty": quality_floor_penalty,
                "top3_distribution_risk_cap_penalty": distribution_risk_cap_penalty,
                "final_selection_score": final_selection_score,
            }
            stage3_scored_codes.append((code, final_selection_score))

        stage3_ranked_codes = [code for code, _ in sorted(stage3_scored_codes, key=lambda item: item[1], reverse=True)]
        stage3_top3_codes = stage3_ranked_codes[:TODAY_TOP_LIMIT]
        stage3_position_map = {code: index for index, code in enumerate(stage3_ranked_codes, start=1)}
        for code in stage2_top20_codes:
            payload = final_recommendations.get(code)
            if not payload:
                continue
            final_selection_score = self._get_effective_stage3_selection_score(payload)
            payload["structured_rank_score"] = final_selection_score
            payload["final_selection_score"] = final_selection_score
            payload["structured_rank_position"] = stage3_position_map.get(code, payload.get("structured_rank_position"))
            payload["selection_stage"] = "stage3_final_top3" if code in stage3_top3_codes else "stage2_top20_pre_moneyflow"
            payload["selection_reason"] = (
                f"stage3_final_score={float(payload.get('stage3_final_score') or 0.0):.2f}; "
                f"final_selection_score={float(payload.get('final_selection_score') or 0.0):.2f}; "
                f"moneyflow_score={float(payload.get('stage3_moneyflow_score') or 0.0):.2f}; "
                f"late_risk_penalty={float(payload.get('stage3_late_risk_penalty') or 0.0):.2f}; "
                f"quality_floor_penalty={float(payload.get('top3_quality_floor_sort_penalty') or 0.0):.2f}; "
                f"distribution_cap_penalty={float(payload.get('top3_distribution_risk_cap_penalty') or 0.0):.2f}; "
                f"moneyflow_veto={bool(payload.get('stage3_moneyflow_veto', False))}"
            )
        return dict(sorted(final_recommendations.items(), key=lambda item: float(item[1].get("score") or 0.0), reverse=True))

    @classmethod
    def _build_stage3_late_risk_penalty(
        cls,
        payload: Dict[str, Any],
        *,
        recent_3d_net_inflow: float,
        recent_large_order_net_inflow: float,
        recent_super_large_order_net_inflow: float,
    ) -> Tuple[float, List[str]]:
        distribution_risk_score = cls._safe_float(payload.get("distribution_risk_score")) or 0.0
        price_position = cls._safe_float(payload.get("price_position_20d")) or 0.0
        recent_runup_5d = cls._safe_float(payload.get("recent_runup_5d")) or 0.0
        turnover_spike_ratio = cls._safe_float(payload.get("turnover_spike_ratio")) or 0.0
        penalty = min(distribution_risk_score * 1.5, 5.5)
        flags: List[str] = []
        if distribution_risk_score > 0:
            flags.append("distribution_risk_score")
        weak_moneyflow = (
            recent_3d_net_inflow <= 0
            or recent_large_order_net_inflow < 0
            or recent_super_large_order_net_inflow < 0
        )
        high_position = price_position >= DISTRIBUTION_PRICE_POSITION_HIGH
        high_runup = recent_runup_5d >= DISTRIBUTION_RECENT_RUNUP_HIGH
        if high_position and high_runup:
            penalty += 2.0
            flags.append("high_position_runup")
        if high_position and weak_moneyflow:
            penalty += 2.5
            flags.append("high_position_weak_moneyflow")
            if recent_runup_5d <= -10.0:
                penalty += 2.5
                flags.append("high_position_drawdown_severe_weak_moneyflow")
            elif recent_runup_5d <= -8.0:
                penalty += 2.0
                flags.append("high_position_drawdown_high_weak_moneyflow")
            elif recent_runup_5d <= -5.0:
                penalty += 1.5
                flags.append("high_position_drawdown_moderate_weak_moneyflow")
        if turnover_spike_ratio >= DISTRIBUTION_TURNOVER_SPIKE_HIGH and weak_moneyflow:
            penalty += 1.5
            flags.append("turnover_spike_weak_moneyflow")
        if bool(payload.get("late_stage_momentum_flag", False)):
            penalty += 2.5
            flags.append("late_stage_momentum")
        if bool(payload.get("high_level_pullback_flag", False)):
            penalty += 2.0
            flags.append("high_level_pullback")
        if bool(payload.get("unsupported_high_position_flag", False)) or bool(payload.get("theme_support_absent_flag", False)):
            penalty += 1.5
            flags.append("unsupported_high_position")
        return round(max(0.0, penalty), 4), flags

    def _apply_stage3_close_auction_rerank(
        self,
        *,
        trade_date: date,
        stage3_recommendations: Dict[str, Dict[str, Any]],
        stage2_top20_codes: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        final_recommendations = {code: dict(payload) for code, payload in stage3_recommendations.items()}
        if not stage2_top20_codes:
            return final_recommendations

        auction_rows: Dict[str, Dict[str, Any]] = {}
        fetch_close_auction_batch = getattr(getattr(self.screener, "client", None), "fetch_close_auction_batch", None)
        if callable(fetch_close_auction_batch):
            try:
                auction_rows = fetch_close_auction_batch(
                    ts_codes=stage2_top20_codes,
                    trade_date=trade_date.strftime("%Y%m%d"),
                )
            except Exception as exc:
                logger.warning("Close auction fetch failed: trade_date=%s, error=%s", trade_date.isoformat(), exc)

        for code in stage2_top20_codes:
            payload = final_recommendations.get(code)
            if not payload:
                continue
            signal = self._build_close_auction_signal(payload, auction_rows.get(code) or {})
            base_score = float(payload.get("stage3_final_score") or payload.get("score") or 0.0)
            auction_score = float(signal.get("stage3_close_auction_score") or 0.0)
            payload.update(signal)
            payload["stage3_final_score"] = round(base_score + auction_score, 4)
            quality_floor_penalty = self._get_top3_quality_floor_penalty(payload)
            distribution_risk_cap_penalty = self._get_top3_distribution_risk_cap_penalty(payload)
            payload["top3_quality_floor_sort_penalty"] = quality_floor_penalty
            payload["top3_distribution_risk_cap_penalty"] = distribution_risk_cap_penalty
            payload["final_selection_score"] = self._get_effective_stage3_selection_score(payload)
            payload["structured_rank_score"] = payload["final_selection_score"]
            payload["selection_reason_components"] = {
                **dict(payload.get("selection_reason_components") or {}),
                "stage3_close_auction_score": payload.get("stage3_close_auction_score"),
                "stage3_close_auction_flags": payload.get("stage3_close_auction_flags"),
                "stage3_close_auction_risks": payload.get("stage3_close_auction_risks"),
                "stage3_close_auction_veto": payload.get("stage3_close_auction_veto"),
                "close_auction_price_deviation_pct": payload.get("close_auction_price_deviation_pct"),
                "close_auction_amount_ratio": payload.get("close_auction_amount_ratio"),
                "top3_quality_floor_sort_penalty": quality_floor_penalty,
                "top3_distribution_risk_cap_penalty": distribution_risk_cap_penalty,
                "final_selection_score": payload["final_selection_score"],
            }
            payload["selection_reason"] = (
                f"stage3_final_score={float(payload.get('stage3_final_score') or 0.0):.2f}; "
                f"final_selection_score={float(payload.get('final_selection_score') or 0.0):.2f}; "
                f"moneyflow_score={float(payload.get('stage3_moneyflow_score') or 0.0):.2f}; "
                f"close_auction_score={auction_score:.2f}; "
                f"distribution_cap_penalty={float(payload.get('top3_distribution_risk_cap_penalty') or 0.0):.2f}; "
                f"moneyflow_veto={bool(payload.get('stage3_moneyflow_veto', False))}; "
                f"close_auction_veto={bool(payload.get('stage3_close_auction_veto', False))}"
            )
        return final_recommendations

    @classmethod
    def _rank_stage3_candidates_by_score(
        cls,
        *,
        candidate_codes: List[str],
        recommendations: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        ranked_items: List[Tuple[str, Dict[str, Any]]] = [
            (code, recommendations[code])
            for code in candidate_codes
            if code in recommendations
        ]
        return [
            code
            for code, _ in sorted(
                ranked_items,
                key=lambda item: (
                    -cls._get_effective_stage3_selection_score(item[1]),
                    int(cls._safe_float(item[1].get("structured_rank_position")) or 999999),
                    int(cls._safe_float(item[1].get("rerank_pool_rank")) or 999999),
                    item[0],
                ),
            )
        ]

    @classmethod
    def _get_top3_quality_floor_penalty(cls, payload: Dict[str, Any]) -> float:
        if not cls._get_top3_quality_floor_reason(payload):
            return 0.0
        risk_adjusted_fusion = cls._safe_float(payload.get("risk_adjusted_fusion_score"))
        if risk_adjusted_fusion is None:
            risk_adjusted_fusion = cls._safe_float(payload.get("fusion_70_30")) or 0.0
        overall_score = cls._safe_float(payload.get("overall_score_norm")) or 0.0
        fusion_deficit = max(0.0, TOP3_QUALITY_FLOOR_FUSION_MIN - float(risk_adjusted_fusion or 0.0))
        overall_deficit = max(0.0, TOP3_QUALITY_FLOOR_OVERALL_MIN - float(overall_score or 0.0))
        penalty = (
            TOP3_QUALITY_FLOOR_SORT_PENALTY
            + fusion_deficit * TOP3_QUALITY_FLOOR_FUSION_DEFICIT_PENALTY_MULTIPLIER
            + overall_deficit * TOP3_QUALITY_FLOOR_OVERALL_DEFICIT_PENALTY_MULTIPLIER
        )
        return round(min(TOP3_QUALITY_FLOOR_MAX_SORT_PENALTY, penalty), 4)

    @classmethod
    def _get_top3_distribution_risk_cap_penalty(cls, payload: Dict[str, Any]) -> float:
        distribution_risk_score = cls._safe_float(payload.get("distribution_risk_score")) or 0.0
        if distribution_risk_score < TOP3_MAX_DISTRIBUTION_RISK_SCORE:
            return 0.0
        penalty = (
            TOP3_DISTRIBUTION_RISK_CAP_SORT_PENALTY
            + (distribution_risk_score - TOP3_MAX_DISTRIBUTION_RISK_SCORE)
            * TOP3_DISTRIBUTION_RISK_CAP_EXCESS_MULTIPLIER
        )
        return round(min(TOP3_DISTRIBUTION_RISK_CAP_MAX_SORT_PENALTY, penalty), 4)

    @classmethod
    def _get_effective_stage3_selection_score(cls, payload: Dict[str, Any]) -> float:
        base_score = cls._safe_float(payload.get("stage3_final_score"))
        if base_score is None:
            persisted_score = cls._safe_float(payload.get("final_selection_score"))
            if persisted_score is not None:
                return round(max(0.0, float(persisted_score)), 4)
            base_score = cls._safe_float(payload.get("structured_rank_score"))
        if base_score is None:
            base_score = cls._safe_float(payload.get("score")) or 0.0
        total_penalty = (
            cls._get_top3_quality_floor_penalty(payload)
            + cls._get_top3_distribution_risk_cap_penalty(payload)
        )
        return round(max(0.0, float(base_score) - total_penalty), 4)

    @classmethod
    def _get_top3_final_moneyflow_veto_reason(cls, payload: Dict[str, Any]) -> Optional[str]:
        pct_change = cls._safe_float(payload.get("pct_change")) or 0.0
        moneyflow_3d = cls._safe_float(payload.get("moneyflow_3d_value")) or 0.0
        moneyflow_10d = cls._safe_float(payload.get("moneyflow_10d_value")) or 0.0
        large_order_3d = cls._safe_float(payload.get("recent_large_order_net_inflow")) or 0.0
        stage3_moneyflow_score = cls._safe_float(payload.get("stage3_moneyflow_score")) or 0.0
        risk_adjusted_fusion = cls._safe_float(payload.get("risk_adjusted_fusion_score"))
        if risk_adjusted_fusion is None:
            risk_adjusted_fusion = cls._safe_float(payload.get("fusion_70_30")) or 0.0
        quality_floor_degraded = bool(payload.get("top3_quality_floor_degraded", False)) or bool(
            cls._get_top3_quality_floor_reason(payload)
        )

        if (
            pct_change >= TOP3_LIMIT_UP_MONEYFLOW_DIVERGENCE_PCT_CHANGE_MIN
            and moneyflow_3d < 0
            and large_order_3d < 0
        ):
            return "limit_up_moneyflow_divergence_top3_veto"
        if risk_adjusted_fusion < TOP3_LOW_QUALITY_NEGATIVE_MONEYFLOW_FUSION_MAX and stage3_moneyflow_score < 0:
            return "low_quality_negative_moneyflow_top3_veto"
        if quality_floor_degraded and stage3_moneyflow_score < 0:
            return "quality_floor_degraded_negative_moneyflow_top3_veto"
        if moneyflow_10d < 0 and moneyflow_3d < 0 and stage3_moneyflow_score < 0:
            if risk_adjusted_fusion < TOP3_MONEYFLOW_DOUBLE_NEGATIVE_FUSION_MAX:
                return "moneyflow_10d_3d_double_negative_low_quality_top3_veto"
            if pct_change >= TOP3_MONEYFLOW_DOUBLE_NEGATIVE_PCT_CHANGE_MIN:
                return "moneyflow_10d_3d_double_negative_rising_top3_veto"
        return None

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
            "rerank_ranking_trade_date": rerank_result.ranking_trade_date.isoformat() if rerank_result.ranking_trade_date else None,
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

    @classmethod
    def _build_stage_rank_maps(cls, codes: List[str]) -> Dict[str, int]:
        return {code: index for index, code in enumerate(codes, start=1)}

    def _build_stock_name_map(self, screening_results: Dict[str, ScreenResult]) -> Dict[str, str]:
        stock_name_map: Dict[str, str] = {}
        for result in screening_results.values():
            if not result:
                continue
            for stock in result.stocks:
                stock_name_map.setdefault(stock.ts_code, stock.name)
        return stock_name_map
