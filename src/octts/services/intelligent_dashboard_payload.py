from __future__ import annotations

from typing import Any, Dict, Optional

from octts.config import Settings


def load_intelligent_dashboard_payload(settings: Settings, trade_date: Optional[str] = None) -> Dict[str, Any]:
    from octts.api_legacy import _load_intelligent_dashboard_payload

    return _load_intelligent_dashboard_payload(settings, trade_date=trade_date)


def build_stock_intelligent_insight(ts_code: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from octts.api_legacy import _build_stock_intelligent_insight

    return _build_stock_intelligent_insight(ts_code, payload)


def build_intelligent_overview_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    from octts.api_legacy import _build_intelligent_overview_payload

    return _build_intelligent_overview_payload(payload)


def build_recommendation_dashboard_payload(settings: Settings) -> Dict[str, Any]:
    from octts.api_legacy import _build_recommendation_dashboard_payload

    return _build_recommendation_dashboard_payload(settings)


def load_recommendation_summary(settings: Settings) -> Dict[str, Any]:
    from octts.api_legacy import _load_recommendation_summary

    return _load_recommendation_summary(settings)


def build_recommendation_methodology_payload(settings: Settings) -> Dict[str, Any]:
    from octts.api_legacy import _build_recommendation_methodology_payload

    return _build_recommendation_methodology_payload(settings)
