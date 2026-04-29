from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from octts.api_legacy import (
    _build_backtest_engine,
    _build_pipeline,
    _build_recommendation_dashboard_payload,
    _build_report_email_service,
    _build_report_exporter_with_settings,
    _build_screening_scheduler,
    _configure_logging,
    _load_recommendation_summary,
    get_active_intelligent_screening_job,
    get_intelligent_screening_job,
)
from octts.api_routes import (
    register_analysis_routes,
    register_backtest_routes,
    register_base_routes,
    register_dashboard_routes,
    register_portfolio_routes,
    register_screening_routes,
)
from octts.config import get_settings
from octts.services.intelligent_dashboard_payload import (
    build_intelligent_overview_payload as _build_intelligent_overview_payload,
    build_recommendation_methodology_payload as _build_recommendation_methodology_payload,
    build_stock_intelligent_insight as _build_stock_intelligent_insight,
    load_intelligent_dashboard_payload as _load_intelligent_dashboard_payload,
)
from octts.services.automation_scheduler import create_automation_scheduler
from octts.services.intelligent_screening_job_manager import IntelligentScreeningJobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.intelligent_screening_job_manager = IntelligentScreeningJobManager(settings.history_dir_path)
    scheduler = create_automation_scheduler(
        settings=settings,
        pipeline_factory=_build_pipeline,
        report_email_service_factory=_build_report_email_service,
        screening_scheduler_factory=_build_screening_scheduler,
    )
    app.state.automation_scheduler = scheduler
    if scheduler:
        scheduler.start()
    try:
        yield
    finally:
        job_manager = getattr(app.state, "intelligent_screening_job_manager", None)
        if job_manager is not None:
            await job_manager.shutdown()
        if scheduler:
            scheduler.shutdown(wait=False)


_configure_logging()

app = FastAPI(title="OCTTS", version="0.1.0", lifespan=lifespan)

register_base_routes(app)
register_analysis_routes(app)
register_dashboard_routes(app)
register_portfolio_routes(app)
register_screening_routes(app)
register_backtest_routes(app)
