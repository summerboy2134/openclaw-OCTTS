from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from octts.clients.tushare_client import TushareClient
from octts.config import Settings
from octts.schemas.training import SHORT_TERM_FEATURE_SCHEMA_VERSION, ShortTermTrainingSample
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.recommendation_tracker import RecommendationTracker
from octts.services.screening_store import ScreeningStore

logger = logging.getLogger(__name__)


class ShortTermTrainingDataBuilder:
    def __init__(
        self,
        settings: Settings,
        *,
        store: Optional[ScreeningStore] = None,
        tushare_client: Optional[TushareClient] = None,
    ) -> None:
        self.settings = settings
        self.store = store or ScreeningStore(settings)
        self._tushare_client = tushare_client
        self._future_trading_dates_cache: Dict[str, List[str]] = {}
        self._screening_snapshot_cache: Dict[str, Dict[str, Any]] = {}
        self._market_raw_repo = MarketRawDataRepository(settings.database_url)

    @property
    def tushare_client(self) -> TushareClient:
        if self._tushare_client is None:
            self._tushare_client = TushareClient(self.settings)
        return self._tushare_client

    def build_samples_for_trade_date(self, trade_date: date) -> List[ShortTermTrainingSample]:
        pool_states = self.store.load_recommendation_pool_state(trade_date=trade_date)
        run_items = self.store.list_recommendation_run_items(trade_date=trade_date)
        item_map = {
            item.get("ts_code"): item
            for item in run_items
            if isinstance(item, dict) and item.get("ts_code")
        }
        label_payloads = self._build_label_payloads_for_trade_date(trade_date, pool_states, item_map)

        samples: List[ShortTermTrainingSample] = []
        for state in pool_states:
            if not isinstance(state, dict):
                continue
            ts_code = str(state.get("ts_code") or "").strip()
            if not ts_code:
                continue
            item = item_map.get(ts_code, {})
            label_payload = label_payloads.get(ts_code, {})
            return_1d = label_payload.get("return_1d")
            sample = ShortTermTrainingSample(
                feature_schema_version=SHORT_TERM_FEATURE_SCHEMA_VERSION,
                trade_date=trade_date,
                ts_code=ts_code,
                name=state.get("name"),
                source_tag=state.get("source_tag"),
                in_frontlist=bool(state.get("in_frontlist", False)),
                recommend_rank=state.get("recommend_rank"),
                strategy_count=int(state.get("strategy_count") or 0),
                is_repeat_pick=bool(state.get("is_repeat_pick", False)),
                news_mentioned=bool(state.get("news_mentioned", False)),
                technical_signal=state.get("technical_signal"),
                entry_price=label_payload.get("entry_price", item.get("entry_price", state.get("entry_price"))),
                close=state.get("close"),
                pct_change=state.get("pct_change"),
                volume_ratio=state.get("volume_ratio"),
                turnover_rate=state.get("turnover_rate"),
                recommendation_score=state.get("recommendation_score"),
                overall_score=state.get("overall_score"),
                technical_score=state.get("technical_score"),
                fundamental_score=state.get("fundamental_score"),
                sentiment_score=state.get("sentiment_score"),
                news_score=state.get("news_score"),
                base_score=state.get("base_score"),
                sentiment_adjustment=state.get("sentiment_adjustment"),
                news_adjustment=state.get("news_adjustment"),
                industry=state.get("industry"),
                industry_heat_score=state.get("industry_heat_score"),
                industry_flow_bias=state.get("industry_flow_bias"),
                distribution_risk_score=state.get("distribution_risk_score"),
                distribution_risk_flags=list(state.get("distribution_risk_flags") or []),
                moneyflow_3d_value=state.get("moneyflow_3d_value"),
                turnover_spike_ratio=state.get("turnover_spike_ratio"),
                recent_runup_5d=state.get("recent_runup_5d"),
                continuation_bias_score=state.get("continuation_bias_score"),
                continuation_positive_flags=list(state.get("continuation_positive_flags") or []),
                continuation_negative_flags=list(state.get("continuation_negative_flags") or []),
                top3_risk_penalty=state.get("top3_risk_penalty"),
                short_term_contradiction_penalty=state.get("short_term_contradiction_penalty"),
                late_stage_momentum_flag=bool(state.get("late_stage_momentum_flag", False)),
                candidate_risk_blocked=bool(state.get("candidate_risk_blocked", False)),
                previous_recommendation_score=state.get("previous_recommendation_score"),
                previous_overall_score=state.get("previous_overall_score"),
                score_change=state.get("score_change"),
                action_plan=dict(state.get("action_plan") or {}),
                return_1d=return_1d,
                return_3d=label_payload.get("return_3d"),
                return_5d=label_payload.get("return_5d"),
                return_10d=label_payload.get("return_10d"),
                max_drawdown_10d=label_payload.get("max_drawdown_10d"),
                benchmark_return_5d=label_payload.get("benchmark_return_5d"),
                vs_benchmark_5d=label_payload.get("vs_benchmark_5d"),
                label_up_1d=(bool(return_1d > 0) if return_1d is not None else None),
            )
            samples.append(sample)
        logger.info(
            "Short-term training samples built: trade_date=%s, pool_states=%s, run_items=%s, samples=%s",
            trade_date.isoformat(),
            len(pool_states),
            len(run_items),
            len(samples),
        )
        return samples

    def persist_samples_for_trade_date(self, trade_date: date) -> List[Dict[str, Any]]:
        samples = self.build_samples_for_trade_date(trade_date)
        if not samples:
            return []
        persisted = self.store.upsert_short_term_training_samples(samples)
        logger.info(
            "Short-term training samples persisted: trade_date=%s, samples=%s",
            trade_date.isoformat(),
            len(persisted),
        )
        return persisted

    def backfill_samples(
        self,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        trade_dates: Optional[List[date]] = None,
    ) -> Dict[str, Any]:
        if trade_dates is None:
            history = self.store.list_recommendation_history(limit=2000)
            available_dates = []
            for item in history:
                trade_date_text = str(item.get("trade_date") or "").strip()
                try:
                    parsed = datetime.strptime(trade_date_text, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if start_date and parsed < start_date:
                    continue
                if end_date and parsed > end_date:
                    continue
                available_dates.append(parsed)
            unique_dates = sorted(set(available_dates))
        else:
            unique_dates = sorted(set(trade_dates))

        total_samples = 0
        processed_days = 0
        for trade_day in unique_dates:
            persisted = self.persist_samples_for_trade_date(trade_day)
            processed_days += 1
            total_samples += len(persisted)
        summary = {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "processed_days": processed_days,
            "sample_count": total_samples,
            "schema_version": SHORT_TERM_FEATURE_SCHEMA_VERSION,
        }
        logger.info("Short-term sample backfill complete: %s", summary)
        return summary

    def _build_label_payloads_for_trade_date(
        self,
        trade_date: date,
        pool_states: List[Dict[str, Any]],
        item_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        trade_date_text = trade_date.isoformat()
        future_trade_dates = self._fetch_future_trading_dates(trade_date_text)
        target_dates = RecommendationTracker._resolve_target_dates(trade_date_text, future_trade_dates)

        benchmark_entry = self._get_local_close(
            ts_code=RecommendationTracker.DEFAULT_BENCHMARK_CODE,
            trade_date_text=trade_date_text,
        )
        benchmark_target_5d = None
        if 5 in target_dates:
            benchmark_target_5d = self._get_local_close(
                ts_code=RecommendationTracker.DEFAULT_BENCHMARK_CODE,
                trade_date_text=target_dates[5],
            )

        payloads: Dict[str, Dict[str, Any]] = {}
        for state in pool_states:
            ts_code = str(state.get("ts_code") or "").strip()
            if not ts_code:
                continue
            existing_item = item_map.get(ts_code, {})
            entry_price = existing_item.get("entry_price", state.get("entry_price", state.get("close")))
            entry_price_value = self._safe_float(entry_price)
            payload = {
                "entry_price": entry_price_value,
                "return_1d": existing_item.get("return_1d"),
                "return_3d": existing_item.get("return_3d"),
                "return_5d": existing_item.get("return_5d"),
                "return_10d": existing_item.get("return_10d"),
                "max_drawdown_10d": existing_item.get("max_drawdown_10d"),
                "benchmark_return_5d": existing_item.get("benchmark_return_5d"),
                "vs_benchmark_5d": existing_item.get("vs_benchmark_5d"),
            }
            if entry_price_value not in (None, 0):
                future_prices: List[float] = []
                for offset, target_date in target_dates.items():
                    target_close = self._get_local_close(ts_code=ts_code, trade_date_text=target_date)
                    if target_close is None:
                        snapshot = self._get_cached_screening_snapshot(target_date.replace("-", ""))
                        target_close = self._extract_close_from_snapshot(snapshot, ts_code)
                    if target_close is None:
                        continue
                    payload[f"return_{offset}d"] = (target_close - entry_price_value) / entry_price_value
                    future_prices.append(target_close)
                if future_prices:
                    payload["max_drawdown_10d"] = RecommendationTracker._compute_max_drawdown(entry_price_value, future_prices)
                if benchmark_entry not in (None, 0) and benchmark_target_5d is not None:
                    benchmark_return_5d = (benchmark_target_5d - benchmark_entry) / benchmark_entry
                    payload["benchmark_return_5d"] = benchmark_return_5d
                    if payload.get("return_5d") is not None:
                        payload["vs_benchmark_5d"] = payload["return_5d"] - benchmark_return_5d
            payloads[ts_code] = payload
        return payloads

    def _fetch_future_trading_dates(self, trade_date_text: str) -> List[str]:
        cached_dates = self._future_trading_dates_cache.get(trade_date_text)
        if cached_dates is not None:
            return list(cached_dates)

        start_compact = trade_date_text.replace("-", "")
        end_date = (datetime.strptime(trade_date_text, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y%m%d")
        local_dates = self._market_raw_repo.list_trading_dates(start_date=start_compact, end_date=end_date)
        normalized_dates = [
            normalized
            for normalized in (
                RecommendationTracker._normalize_trade_date(value)
                for value in local_dates
            )
            if normalized
        ]
        if not normalized_dates:
            dates = self.tushare_client.fetch_trading_dates(
                start_date=start_compact,
                end_date=end_date,
            )
            normalized_dates = [
                normalized
                for normalized in (
                    RecommendationTracker._normalize_trade_date(value)
                    for value in dates
                )
                if normalized
            ]
        self._future_trading_dates_cache[trade_date_text] = normalized_dates
        return list(normalized_dates)

    def _get_cached_screening_snapshot(self, trade_date_text: str) -> Dict[str, Any]:
        cached_snapshot = self._screening_snapshot_cache.get(trade_date_text)
        if cached_snapshot is not None:
            return cached_snapshot
        snapshot = self.tushare_client.get_or_build_screening_snapshot(trade_date_text)
        self._screening_snapshot_cache[trade_date_text] = snapshot
        return snapshot

    def _get_local_close(self, *, ts_code: str, trade_date_text: str) -> Optional[float]:
        daily_row = self._market_raw_repo.get_daily(ts_code=ts_code, trade_date=trade_date_text.replace("-", ""))
        if isinstance(daily_row, dict):
            close_value = self._safe_float(daily_row.get("close"))
            if close_value is not None:
                return close_value
        daily_basic_row = self._market_raw_repo.get_daily_basic(ts_code=ts_code, trade_date=trade_date_text.replace("-", ""))
        if isinstance(daily_basic_row, dict):
            return self._safe_float(daily_basic_row.get("close"))
        return None

    @staticmethod
    def _extract_close_from_snapshot(snapshot: Any, ts_code: str) -> Optional[float]:
        if not isinstance(snapshot, dict):
            return None
        daily_basic = snapshot.get("daily_basic")
        if isinstance(daily_basic, dict):
            basic = daily_basic.get(ts_code)
            if isinstance(basic, dict):
                close = basic.get("close")
                close_value = ShortTermTrainingDataBuilder._safe_float(close)
                if close_value is not None:
                    return close_value
        daily = snapshot.get("daily")
        if isinstance(daily, dict):
            rows = daily.get(ts_code) or []
            if rows and isinstance(rows[0], dict):
                return ShortTermTrainingDataBuilder._safe_float(rows[0].get("close"))
        return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
