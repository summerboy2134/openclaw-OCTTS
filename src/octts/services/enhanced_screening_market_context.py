"""Mixin helpers for enhanced screening scheduler."""

import html
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.services.stock_screener import StockScreener
from octts.services.position_store import create_position_store
from octts.services.regression_rerank_service import RegressionRerankResult
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_constants import *

logger = logging.getLogger(__name__)


class EnhancedScreeningMarketContextMixin:
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
                trade_dates = [
                    str(row.get("trade_date") or "").strip()
                    for row in rows
                    if isinstance(row, dict) and row.get("trade_date")
                ]
                daily_basic_by_code = self.market_raw_data_repo.get_daily_basic_by_trade_dates(
                    ts_codes=[normalized],
                    trading_dates=trade_dates,
                )
                basic_by_date = daily_basic_by_code.get(normalized, {})
                if basic_by_date:
                    rows = [
                        {
                            **row,
                            **{
                                key: value
                                for key, value in (basic_by_date.get(str(row.get("trade_date") or "")) or {}).items()
                                if key not in {"ts_code", "trade_date"} and value is not None
                            },
                        }
                        for row in rows
                    ]
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
