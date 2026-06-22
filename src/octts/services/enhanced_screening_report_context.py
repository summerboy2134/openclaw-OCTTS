"""Shared helpers for enhanced screening scheduler mixins."""

import html
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.services.stock_screener import StockScreener
from octts.services.regression_rerank_service import RegressionRerankResult
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_constants import *

logger = logging.getLogger(__name__)


class EnhancedScreeningReportContextMixin:
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

    @classmethod
    def _build_dashboard_ai_payload(cls, 
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
                overall_score = cls._resolve_real_overall_score(
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
            merged.update(cls._build_unified_score_fields(merged, state, analysis, recommendation_meta))
            payload[code] = merged
        return payload

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

    @classmethod
    def _build_three_day_review_strength_change(cls, performance: Dict[str, Any], review_status: str) -> str:
        return_value = performance.get("review_return")
        if isinstance(return_value, (int, float)):
            return f"从次日开盘基准到当前参考价的收益为{return_value * 100:+.2f}%，状态为{review_status}。"
        return "价格数据不足，强弱变化需要人工结合行情复核。"

    @classmethod
    def _build_three_day_review_market_context(cls, performance: Dict[str, Any], today_verdict: str) -> str:
        entry_date = performance.get("entry_date") or "次日"
        return f"本段复盘按{entry_date}开盘买入、当前价格复核的短线持仓规则执行。{today_verdict}。"

    @classmethod
    def _build_three_day_review_action_plan(cls, performance: Dict[str, Any], sellable: bool) -> Dict[str, Any]:
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
        if not isinstance(top_list_map, dict):
            top_list_map = {}
        if not isinstance(limit_list_map, dict):
            limit_list_map = {}

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

    @classmethod
    def _summarize_top_list_rows(cls, rows: List[Dict[str, Any]]) -> Optional[str]:
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

    @classmethod
    def _summarize_limit_list_row(cls, row: Optional[Dict[str, Any]]) -> Optional[str]:
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
            "close_auction_price": item.get("close_auction_price", recommendation.get("close_auction_price")),
            "close_auction_amount": item.get("close_auction_amount", recommendation.get("close_auction_amount")),
            "close_auction_vwap": item.get("close_auction_vwap", recommendation.get("close_auction_vwap")),
            "close_auction_price_deviation_pct": item.get("close_auction_price_deviation_pct", recommendation.get("close_auction_price_deviation_pct")),
            "close_auction_amount_ratio": item.get("close_auction_amount_ratio", recommendation.get("close_auction_amount_ratio")),
            "stage3_close_auction_score": item.get("stage3_close_auction_score", recommendation.get("stage3_close_auction_score")),
            "stage3_close_auction_flags": list(item.get("stage3_close_auction_flags") or recommendation.get("stage3_close_auction_flags") or []),
            "stage3_close_auction_risks": list(item.get("stage3_close_auction_risks") or recommendation.get("stage3_close_auction_risks") or []),
            "stage3_close_auction_veto": bool(item.get("stage3_close_auction_veto", recommendation.get("stage3_close_auction_veto", False))),
            "stage3_close_auction_missing": bool(item.get("stage3_close_auction_missing", recommendation.get("stage3_close_auction_missing", False))),
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
        final_selection_score = cls._first_defined_value(
            payload.get("final_selection_score"),
            item.get("final_selection_score"),
            recommendation.get("final_selection_score"),
            payload.get("stage3_final_score"),
            item.get("stage3_final_score"),
            recommendation.get("stage3_final_score"),
            payload.get("structured_rank_score"),
            item.get("structured_rank_score"),
            recommendation.get("structured_rank_score"),
        )
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
            "final_selection_score": final_selection_score,
            "fusion_70_30": fusion_70_30,
            "score_snapshot": score_snapshot,
        }

    @classmethod
    def _build_earnings_forecast_payload(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
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
