from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from octts.config import Settings
from octts.services.screening_store import ScreeningStore

logger = logging.getLogger(__name__)


class DailyAnalysisScreeningContextProvider:
    """Build compact intelligent-screening context for regular stock analysis."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._store: Optional[ScreeningStore] = None

    def build_for_symbol(self, ts_code: str) -> Dict[str, Any]:
        normalized_code = ts_code.strip().upper()
        if not normalized_code:
            return {}

        snapshot_context = self._build_from_latest_snapshot(normalized_code)
        pool_context = self._build_from_latest_pool(normalized_code)
        if not snapshot_context and not pool_context:
            return {
                "data_available": False,
                "message": "最新智能选股上下文不可用；本次仅基于个股行情、财务和历史观点分析。",
            }

        merged = {
            "data_available": True,
            "ts_code": normalized_code,
            "source": "intelligent_screening",
        }
        merged.update(snapshot_context)
        if pool_context:
            merged["latest_pool_state"] = pool_context.get("stock_state")
            merged["latest_pool_trade_date"] = pool_context.get("trade_date")
            merged["latest_model_top3"] = pool_context.get("model_top3")
            if "stock_state" in pool_context:
                merged["stock_context"] = pool_context["stock_state"]
        return _drop_empty_values(merged)

    def _build_from_latest_snapshot(self, ts_code: str) -> Dict[str, Any]:
        payload = self._read_latest_snapshot()
        if not payload:
            return {}

        report_context = payload.get("report_context") if isinstance(payload.get("report_context"), dict) else {}
        recommendation_pool = payload.get("recommendation_pool") if isinstance(payload.get("recommendation_pool"), dict) else {}
        ai_analyses = payload.get("ai_analyses") if isinstance(payload.get("ai_analyses"), dict) else {}

        candidates: List[Dict[str, Any]] = []
        for key in ("today_top3", "today_top10", "yesterday_top3_review"):
            candidates.extend(_normalize_items(report_context.get(key)))
        for key in ("today_top", "frontlist", "yesterday_continuations"):
            candidates.extend(_normalize_items(recommendation_pool.get(key)))

        stock_context = _pick_item_by_code(candidates, ts_code)
        ai_context = ai_analyses.get(ts_code) if isinstance(ai_analyses.get(ts_code), dict) else {}
        if not stock_context and ai_context:
            stock_context = {"ts_code": ts_code, **ai_context}

        top3 = [_compact_screening_item(item) for item in _normalize_items(report_context.get("today_top3"))[:3]]
        result = {
            "snapshot_generated_at": payload.get("generated_at"),
            "snapshot_data_source": payload.get("data_source") or "snapshot",
            "stock_context": _compact_screening_item(stock_context) if stock_context else None,
            "ai_context": _compact_analysis(ai_context),
            "latest_report_top3": top3,
        }
        return _drop_empty_values(result)

    def _build_from_latest_pool(self, ts_code: str) -> Dict[str, Any]:
        try:
            states = self._get_store().list_recommendation_pool(limit=120)
        except Exception:
            logger.exception("Failed to load latest recommendation pool for daily analysis context")
            return {}
        if not states:
            return {}

        stock_state = _pick_item_by_code(states, ts_code)
        top3 = [
            _compact_screening_item(item)
            for item in states
            if str(item.get("source_tag") or "") == "今日Top3" or item.get("recommend_rank") in {1, 2, 3}
        ][:3]
        if not top3:
            top3 = [_compact_screening_item(item) for item in states[:3]]
        trade_date = states[0].get("trade_date")
        return _drop_empty_values(
            {
                "trade_date": trade_date,
                "stock_state": _compact_screening_item(stock_state) if stock_state else None,
                "model_top3": top3,
            }
        )

    def _read_latest_snapshot(self) -> Dict[str, Any]:
        snapshot_path = Path(self.settings.history_dir_path) / "intelligent_screening" / "latest.json"
        if not snapshot_path.exists():
            return {}
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read intelligent screening latest snapshot: %s", snapshot_path)
            return {}
        if not isinstance(payload, dict) or payload.get("snapshot_type") != "intelligent_screening":
            return {}
        return payload

    def _get_store(self) -> ScreeningStore:
        if self._store is None:
            self._store = ScreeningStore(self.settings)
        return self._store


def _normalize_items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _pick_item_by_code(items: List[Dict[str, Any]], ts_code: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if str(item.get("ts_code") or "").strip().upper() == ts_code:
            return item
    return None


def _compact_analysis(item: Dict[str, Any]) -> Dict[str, Any]:
    return _pick_keys(
        item,
        [
            "recommendation",
            "summary",
            "technical_summary",
            "technical_signal",
            "key_points",
            "final_decision",
            "conflict_points",
            "overall_confidence",
            "confidence",
        ],
    )


def _compact_screening_item(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return _pick_keys(
        item,
        [
            "ts_code",
            "name",
            "trade_date",
            "source_tag",
            "tracking_status",
            "recommend_rank",
            "rerank_pool_rank",
            "recommendation_score",
            "overall_score",
            "priority_score",
            "rerank_model_score",
            "rerank_blend_score",
            "risk_level",
            "distribution_risk_score",
            "distribution_risk_flags",
            "candidate_risk_blocked",
            "top3_extreme_risk_blocked",
            "top3_extreme_risk_reason",
            "close",
            "pct_change",
            "volume_ratio",
            "turnover_rate",
            "ma20",
            "moneyflow_3d_value",
            "recent_large_order_net_inflow",
            "recent_super_large_order_net_inflow",
            "turnover_spike_ratio",
            "recent_runup_5d",
            "industry",
            "industry_heat_score",
            "industry_flow_bias",
            "selection_reason",
            "selection_reason_components",
            "recommendation_text",
            "action_plan",
            "score_change",
            "today_present",
            "absence_reason",
        ],
    )


def _pick_keys(item: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return _drop_empty_values({key: item.get(key) for key in keys})


def _drop_empty_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }
