from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from octts.api_legacy import (
    _build_history_store,
    _load_memory_keys,
    _normalize_ts_code,
    _persist_stock_pool_update,
)
from octts.config import get_settings
from octts.schemas.api import AnalysisActionResponse, PositionStatusRequest, StockPoolItemRequest
from octts.services.memory_store import create_memory_store
from octts.services.position_store import create_position_store
from octts.services.report_payload import build_openclaw_status


def register_portfolio_routes(app: FastAPI) -> None:
    @app.put("/positions/{ts_code}")
    def update_position_status(ts_code: str, request: PositionStatusRequest) -> dict[str, object]:
        normalized = _normalize_ts_code(ts_code)
        settings = get_settings()
        position_store = create_position_store(settings)
        position_store.set_status(normalized, request.position_status)
        stock_pool = list(settings.stock_pool)
        if request.position_status == "holding":
            stock_pool = _persist_stock_pool_update(ts_code=normalized, action="add")
        return {"ts_code": normalized, "position_status": request.position_status, "stock_pool": stock_pool}

    @app.get("/openclaw/status")
    def openclaw_status() -> dict[str, object]:
        return build_openclaw_status(get_settings())

    @app.get("/stock-pool")
    def stock_pool() -> dict[str, object]:
        settings = get_settings()
        return {"stock_pool": settings.stock_pool}

    @app.post("/stock-pool")
    def add_stock_pool_item(request: StockPoolItemRequest) -> dict[str, object]:
        ts_code = _normalize_ts_code(request.ts_code)
        updated_pool = _persist_stock_pool_update(ts_code=ts_code, action="add")
        return {"stock_pool": updated_pool, "ts_code": ts_code}

    @app.delete("/stock-pool/{ts_code}")
    def remove_stock_pool_item(ts_code: str) -> dict[str, object]:
        normalized = _normalize_ts_code(ts_code)
        updated_pool = _persist_stock_pool_update(ts_code=normalized, action="remove")
        return {"stock_pool": updated_pool, "ts_code": normalized}

    @app.delete("/analysis-data", response_model=AnalysisActionResponse)
    def clear_all_analysis_data() -> AnalysisActionResponse:
        settings = get_settings()
        history_store = _build_history_store(settings)
        memory_store = create_memory_store(settings)
        tracked_symbols = [record.report.ts_code for record in history_store.list_latest()]
        history_records = sum(len(history_store.list_records(ts_code)) for ts_code in tracked_symbols)
        memory_items = len(_load_memory_keys(memory_store))
        history_store.clear()
        memory_store.clear()
        return AnalysisActionResponse(
            cleared_all=True,
            removed_records=history_records,
            removed_memory_items=memory_items,
        )

    @app.delete("/analysis-data/{ts_code}", response_model=AnalysisActionResponse)
    def clear_symbol_analysis_data(ts_code: str) -> AnalysisActionResponse:
        normalized = _normalize_ts_code(ts_code)
        settings = get_settings()
        history_store = _build_history_store(settings)
        memory_store = create_memory_store(settings)
        removed_records = len(history_store.list_records(normalized))
        removed_memory_items = 1 if memory_store.get(normalized) else 0
        history_store.delete_symbol(normalized)
        memory_store.delete(normalized)
        return AnalysisActionResponse(
            cleared_symbols=[normalized],
            removed_records=removed_records,
            removed_memory_items=removed_memory_items,
        )

    @app.delete("/analysis-data/{ts_code}/records", response_model=AnalysisActionResponse)
    def delete_symbol_analysis_record(
        ts_code: str,
        generated_at: str = Query(..., description="ISO 8601 timestamp of the record to delete."),
    ) -> AnalysisActionResponse:
        normalized = _normalize_ts_code(ts_code)
        settings = get_settings()
        history_store = _build_history_store(settings)
        memory_store = create_memory_store(settings)

        try:
            removed_records = history_store.delete_record(normalized, generated_at)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if removed_records == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis record found for {normalized} at {generated_at}",
            )

        remaining_records = len(history_store.list_records(normalized))
        removed_memory_items = 0
        updated_memory = False
        latest_record = history_store.get_latest_record(normalized)
        if latest_record is None:
            removed_memory_items = 1 if memory_store.get(normalized) else 0
            memory_store.delete(normalized)
        else:
            memory_store.set(latest_record.report.memory)
            updated_memory = True

        return AnalysisActionResponse(
            cleared_symbols=[normalized],
            removed_records=removed_records,
            removed_memory_items=removed_memory_items,
            removed_generated_at=generated_at,
            remaining_records=remaining_records,
            updated_memory=updated_memory,
        )
