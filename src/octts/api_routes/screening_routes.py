from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from octts.api_legacy import _get_intelligent_screening_job_manager
from octts.config import get_settings
from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult
from octts.services.enhanced_screening_scheduler import EnhancedScreeningScheduler
from octts.services.execution_confirmation_service import ExecutionConfirmationService
from octts.services.intelligent_dashboard_payload import build_recommendation_dashboard_payload
from octts.services.screening_store import ScreeningStore
from octts.services.stock_screener import StockScreener
from octts.services.stock_screening_scheduler import create_screening_scheduler
from octts.ui.intelligent_screening_dashboard import render_intelligent_screening_dashboard


def register_screening_routes(app: FastAPI) -> None:
    @app.post("/screen/technical", response_model=ScreenResult)
    def screen_technical(criteria: ScreenCriteria) -> ScreenResult:
        try:
            screener = StockScreener()
            return screener.screen(criteria)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Screening failed: {exc}") from exc

    @app.get("/screen/presets", response_model=List[ScreenPreset])
    def get_screen_presets() -> List[ScreenPreset]:
        return StockScreener.get_presets()

    @app.get("/screen/results/{screen_id}", response_model=ScreenResult)
    def get_screen_results(screen_id: str) -> ScreenResult:
        result = StockScreener.get_screen_result(screen_id)
        if result is None:
            result = ScreeningStore(get_settings()).get_screening_result(screen_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No screen result found for {screen_id}")
        return result

    @app.get("/market/stocks")
    def get_all_stocks() -> Dict[str, object]:
        try:
            screener = StockScreener()
            stocks = screener.get_all_stocks()
            return {
                "total_count": len(stocks),
                "stocks": stocks,
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to get stock list: {exc}") from exc

    @app.post("/screen/run-daily")
    async def run_daily_screening() -> Dict[str, Any]:
        try:
            scheduler = create_screening_scheduler(get_settings())
            result = await scheduler.run_daily_screening()
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Screening failed: {exc}") from exc

    @app.get("/intelligent-screening")
    def intelligent_screening_dashboard(tab: str = "overview") -> HTMLResponse:
        dashboard_payload = build_recommendation_dashboard_payload(get_settings())
        html_content = render_intelligent_screening_dashboard(
            screening_results=dashboard_payload["screening_results"],
            recommendation_pool=dashboard_payload.get("recommendation_pool") or {},
            ai_analyses=dashboard_payload["ai_analyses"],
            news_clusters=dashboard_payload["news_clusters"],
            intelligent_report=dashboard_payload["intelligent_report"],
            recommendation_summary=dashboard_payload.get("recommendation_summary") or {},
            recommendation_methodology=dashboard_payload.get("recommendation_methodology") or {},
            report_context=dashboard_payload.get("report_context") or {},
            generated_at=dashboard_payload.get("generated_at"),
            active_tab=tab,
        )
        return HTMLResponse(content=html_content)

    @app.post("/screen/intelligent", status_code=202)
    async def run_intelligent_screening(request: Request) -> Dict[str, Any]:
        return await create_intelligent_screening_job(request)

    @app.post("/screen/intelligent/candidates", status_code=202)
    async def run_candidate_screening(request: Request) -> Dict[str, Any]:
        """Run post-close candidate generation workflow."""
        return await create_intelligent_screening_job(request)

    @app.post("/screen/intelligent/execution-confirmation")
    async def run_execution_confirmation(
        source_trade_date: Optional[str] = None,
        force: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            service = ExecutionConfirmationService(get_settings())
            return await service.run_pre_open_confirmation(source_trade_date=source_trade_date, force=force)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Execution confirmation failed: {exc}") from exc

    @app.post("/screen/intelligent/jobs", status_code=202)
    async def create_intelligent_screening_job(request: Request) -> Dict[str, Any]:
        manager = _get_intelligent_screening_job_manager(request)
        settings = get_settings()
        main_loop = __import__("asyncio").get_running_loop()

        async def runner(progress_callback):
            def thread_safe_progress_callback(payload: Dict[str, Any]) -> None:
                future = __import__("asyncio").run_coroutine_threadsafe(progress_callback(payload), main_loop)
                try:
                    future.result()
                except __import__("concurrent.futures").futures.CancelledError:
                    return

            def run_scheduler() -> Dict[str, Any]:
                scheduler = EnhancedScreeningScheduler(settings, progress_callback=thread_safe_progress_callback)
                return __import__("asyncio").run(scheduler.run_intelligent_screening())

            return await __import__("asyncio").to_thread(run_scheduler)

        try:
            payload = await manager.start_job(runner)
            job = dict(payload["job"])
            job["created"] = payload["created"]
            return job
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to create intelligent screening job: {exc}") from exc

    @app.get("/screen/intelligent/jobs/active")
    async def get_active_intelligent_screening_job(request: Request) -> Dict[str, Any]:
        manager = _get_intelligent_screening_job_manager(request)
        payload = await manager.get_active_job()
        if payload is None:
            return {"job": None}
        return {"job": payload}

    @app.get("/screen/intelligent/jobs/{job_id}")
    async def get_intelligent_screening_job(job_id: str, request: Request) -> Dict[str, Any]:
        manager = _get_intelligent_screening_job_manager(request)
        payload = await manager.get_job(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Intelligent screening job not found: {job_id}")
        return payload

    @app.get("/screen/history")
    def get_screening_history(
        strategy_id: Optional[str] = None,
        days: int = Query(default=30, ge=1, le=365),
    ) -> Dict[str, Any]:
        try:
            store = ScreeningStore(get_settings())
            if strategy_id:
                history = store.get_screening_history(strategy_id, days)
            else:
                history = store.get_all_screening_history(days)
            return {
                "total_count": len(history),
                "history": history,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get history: {exc}") from exc

    @app.get("/stock/{ts_code}/performance")
    def get_stock_performance(
        ts_code: str,
        days: int = Query(default=30, ge=1, le=365),
    ) -> Dict[str, Any]:
        try:
            store = ScreeningStore(get_settings())
            return store.get_stock_performance(ts_code, days=days)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to get stock performance: {exc}") from exc
