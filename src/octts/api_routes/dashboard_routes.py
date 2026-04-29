from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from octts.api_legacy import (
    _build_history_store,
    _normalize_ts_code,
)
from octts.config import get_settings
from octts.services.intelligent_dashboard_payload import (
    build_intelligent_overview_payload,
    build_stock_intelligent_insight,
    load_intelligent_dashboard_payload,
)
from octts.services.position_store import create_position_store
from octts.services.report_payload import build_openclaw_status, build_validation_summary, serialize_record
from octts.ui.dashboard import render_dashboard_html, render_stock_detail_html


def register_dashboard_routes(app: FastAPI) -> None:
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(render_dashboard_html())

    @app.get("/dashboard/data")
    def dashboard_data() -> Dict[str, object]:
        settings = get_settings()
        history_store = _build_history_store(settings)
        position_store = create_position_store(settings)
        latest_records = history_store.list_latest()
        cards = [serialize_record(record, history_store, position_store, history_limit=8) for record in latest_records]
        validation_summary = build_validation_summary(latest_records)
        intelligent_payload = load_intelligent_dashboard_payload(settings)

        return {
            "generated_at": latest_records[0].generated_at if latest_records else None,
            "cards": cards,
            "validation_summary": validation_summary,
            "default_stock_pool": settings.stock_pool,
            "openclaw_status": build_openclaw_status(settings),
            "intelligent_screening": build_intelligent_overview_payload(intelligent_payload),
        }

    @app.get("/stocks/{ts_code}", response_class=HTMLResponse)
    def stock_detail_page(ts_code: str) -> HTMLResponse:
        return HTMLResponse(render_stock_detail_html(ts_code))

    @app.get("/stocks/{ts_code}/data")
    def stock_detail_data(ts_code: str) -> Dict[str, object]:
        normalized = _normalize_ts_code(ts_code)
        settings = get_settings()
        history_store = _build_history_store(settings)
        position_store = create_position_store(settings)
        records = history_store.list_records(normalized, limit=settings.history_limit_per_symbol)
        if not records:
            raise HTTPException(status_code=404, detail=f"No history found for {normalized}")

        latest = records[-1]
        intelligent_payload = load_intelligent_dashboard_payload(settings)
        return {
            "generated_at": latest.generated_at,
            "symbol": serialize_record(latest, history_store, position_store, history_limit=settings.history_limit_per_symbol),
            "validation_summary": build_validation_summary(records),
            "openclaw_status": build_openclaw_status(settings),
            "position_status": position_store.get_status(normalized),
            "default_stock_pool": settings.stock_pool,
            "intelligent_screening_insight": build_stock_intelligent_insight(normalized, intelligent_payload),
        }
