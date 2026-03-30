from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from octts.clients.email_client import EmailClient
from octts.clients.llm_client import LLMClient
from octts.clients.tushare_client import TushareClient
from octts.clients.wecom_client import WeComClient
from octts.config import Settings, get_settings
from octts.schemas.backtest import BacktestRequest, BacktestResult
from octts.schemas.report import AnalysisRequest, AnalysisResult, PositionStatus
from octts.services.analysis_pipeline import AnalysisPipeline
from octts.services.automation_scheduler import build_automation_slots, create_automation_scheduler
from octts.services.backtest_engine import BacktestEngine
from octts.services.history_store import FileHistoryStore
from octts.services.memory_store import create_memory_store
from octts.services.position_store import create_position_store
from octts.services.report_email_service import ReportEmailService
from octts.services.report_exporter import ReportExporter
from octts.ui.dashboard import render_dashboard_html, render_stock_detail_html
from octts.ui.intelligent_screening_dashboard import render_intelligent_screening_dashboard
from octts.schemas.screener import ScreenCriteria, ScreenResult, ScreenPreset
from octts.services.stock_screener import StockScreener
from octts.services.stock_screening_scheduler import create_screening_scheduler
from octts.services.enhanced_screening_scheduler import EnhancedScreeningScheduler
from octts.services.intelligent_screening_job_manager import IntelligentScreeningJobManager
from octts.services.recommendation_tracker import RecommendationTracker
from octts.services.screening_store import ScreeningStore


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


def _configure_logging() -> None:
    settings = get_settings()
    log_dir = Path(settings.history_dir_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    for handler in root_logger.handlers:
        if getattr(handler, "baseFilename", None) == str(log_path):
            return

    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


_configure_logging()


app = FastAPI(title="OCTTS", version="0.1.0", lifespan=lifespan)


class StockPoolItemRequest(BaseModel):
    ts_code: str


class PositionStatusRequest(BaseModel):
    position_status: PositionStatus


class AnalysisActionResponse(BaseModel):
    cleared_symbols: List[str] = Field(default_factory=list)
    cleared_all: bool = False
    removed_records: int = 0
    removed_memory_items: int = 0
    removed_generated_at: Optional[str] = None
    remaining_records: int = 0
    updated_memory: bool = False


class LightweightBacktestRequest(BaseModel):
    start_date: str
    end_date: str
    holding_days: int = Field(default=5, ge=1, le=60)
    top_n: int = Field(default=10, ge=1, le=50)
    commission_rate: float = Field(default=0.0003, ge=0)
    slippage_rate: float = Field(default=0.0005, ge=0)
    strategies: List[str] = Field(default_factory=list)
    stock_pool: List[str] = Field(default_factory=list)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalysisRequest) -> AnalysisResult:
    try:
        pipeline = _build_pipeline()
        return pipeline.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}") from exc


@app.post("/backtest", response_model=BacktestResult)
def backtest(request: BacktestRequest) -> BacktestResult:
    try:
        engine = _build_backtest_engine()
        return engine.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backtest failed: {exc}") from exc


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(render_dashboard_html())


@app.get("/dashboard/data")
def dashboard_data() -> Dict[str, object]:
    settings = get_settings()
    history_store = _build_history_store(settings)
    position_store = create_position_store(settings)
    latest_records = history_store.list_latest()
    cards = [_serialize_record(record, history_store, position_store, history_limit=8) for record in latest_records]
    validation_summary = _build_validation_summary(latest_records)
    intelligent_payload = _load_intelligent_dashboard_payload(settings)

    return {
        "generated_at": latest_records[0].generated_at if latest_records else None,
        "cards": cards,
        "validation_summary": validation_summary,
        "default_stock_pool": settings.stock_pool,
        "openclaw_status": _build_openclaw_status(settings),
        "intelligent_screening": _build_intelligent_overview_payload(intelligent_payload),
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
    intelligent_payload = _load_intelligent_dashboard_payload(settings)
    return {
        "generated_at": latest.generated_at,
        "symbol": _serialize_record(latest, history_store, position_store, history_limit=settings.history_limit_per_symbol),
        "validation_summary": _build_validation_summary(records),
        "openclaw_status": _build_openclaw_status(settings),
        "position_status": position_store.get_status(normalized),
        "default_stock_pool": settings.stock_pool,
        "intelligent_screening_insight": _build_stock_intelligent_insight(normalized, intelligent_payload),
    }


@app.put("/positions/{ts_code}")
def update_position_status(ts_code: str, request: PositionStatusRequest) -> Dict[str, object]:
    normalized = _normalize_ts_code(ts_code)
    settings = get_settings()
    position_store = create_position_store(settings)
    position_store.set_status(normalized, request.position_status)
    stock_pool = list(settings.stock_pool)
    if request.position_status == "holding":
        stock_pool = _persist_stock_pool_update(ts_code=normalized, action="add")
    return {"ts_code": normalized, "position_status": request.position_status, "stock_pool": stock_pool}


@app.get("/openclaw/status")
def openclaw_status() -> Dict[str, object]:
    return _build_openclaw_status(get_settings())


@app.get("/stock-pool")
def stock_pool() -> Dict[str, object]:
    settings = get_settings()
    return {"stock_pool": settings.stock_pool}


@app.post("/stock-pool")
def add_stock_pool_item(request: StockPoolItemRequest) -> Dict[str, object]:
    ts_code = _normalize_ts_code(request.ts_code)
    updated_pool = _persist_stock_pool_update(ts_code=ts_code, action="add")
    return {"stock_pool": updated_pool, "ts_code": ts_code}


@app.delete("/stock-pool/{ts_code}")
def remove_stock_pool_item(ts_code: str) -> Dict[str, object]:
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


def _build_pipeline() -> AnalysisPipeline:
    settings = get_settings()
    tushare_client = TushareClient(settings)
    llm_client = LLMClient(settings)
    wecom_client = None
    if settings.wecom_webhook_url:
        wecom_client = WeComClient(settings)

    return AnalysisPipeline(
        settings=settings,
        tushare_client=tushare_client,
        llm_client=llm_client,
        memory_store=create_memory_store(settings),
        history_store=_build_history_store(settings),
        position_store=create_position_store(settings),
        wecom_client=wecom_client,
    )


def _build_backtest_engine() -> BacktestEngine:
    settings = get_settings()
    tushare_client = TushareClient(settings)
    wecom_client = WeComClient(settings) if settings.wecom_webhook_url else None
    pipeline = AnalysisPipeline(
        settings=settings,
        tushare_client=tushare_client,
        llm_client=LLMClient(settings),
        memory_store=create_memory_store(settings),
        history_store=_build_history_store(settings),
        position_store=create_position_store(settings),
        wecom_client=wecom_client,
    )
    return BacktestEngine(
        pipeline=pipeline,
        market_data_client=tushare_client,
    )


def _build_report_exporter() -> ReportExporter:
    settings = get_settings()
    return _build_report_exporter_with_settings(settings)


def _build_report_exporter_with_settings(settings: Settings) -> ReportExporter:
    return ReportExporter(
        settings=settings,
        history_store=_build_history_store(settings),
        position_store=create_position_store(settings),
    )


def _build_report_email_service() -> ReportEmailService:
    settings = get_settings()
    return ReportEmailService(
        settings=settings,
        history_store=_build_history_store(settings),
        report_exporter=_build_report_exporter(),
        email_client=EmailClient(settings),
    )


def _build_screening_scheduler():
    """构建智能选股调度器"""
    return EnhancedScreeningScheduler(get_settings())


def _build_history_store(settings: Settings) -> FileHistoryStore:
    return FileHistoryStore(
        directory_path=settings.history_dir_path,
        limit_per_symbol=settings.history_limit_per_symbol,
    )


def _serialize_record(record, history_store: FileHistoryStore, position_store, history_limit: int) -> Dict[str, object]:
    history = history_store.list_records(record.report.ts_code, limit=history_limit)
    return {
        "ts_code": record.report.ts_code,
        "generated_at": record.generated_at,
        "phase": record.report.phase,
        "name": record.snapshot.name,
        "trend_judgement": record.report.trend_judgement,
        "trend_breakdown": record.report.trend_breakdown.model_dump(mode="json"),
        "summary_markdown": record.report.summary_markdown,
        "previous_view_status": record.report.previous_view_status,
        "operation_advice": record.report.operation_advice,
        "decision": record.report.decision.model_dump(mode="json"),
        "prediction_windows": [item.model_dump(mode="json") for item in record.report.prediction_windows],
        "validation": record.validation.model_dump(mode="json"),
        "snapshot": record.snapshot.model_dump(mode="json"),
        "memory": record.report.memory.model_dump(mode="json"),
        "position_status": position_store.get_status(record.report.ts_code),
        "default_stock_pool": get_settings().stock_pool,
        "history": [item.model_dump(mode="json") for item in history],
    }


def _build_validation_summary(records) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        status = record.validation.status
        summary[status] = summary.get(status, 0) + 1
    return summary


def _build_openclaw_status(settings: Settings) -> Dict[str, object]:
    automation_enabled = settings.automation_enabled
    return {
        "mode": "built_in_scheduler" if automation_enabled else "external_orchestration",
        "gateway_url": settings.openclaw_gateway_url,
        "agent_id": settings.openclaw_agent_id,
        "hooks_enabled": settings.openclaw_hooks_enabled,
        "connected": bool(settings.openclaw_gateway_url) or automation_enabled,
        "automation_enabled": automation_enabled,
        "automation_notify": settings.automation_notify,
        "automation_timezone": settings.automation_timezone,
        "automation_slots": build_automation_slots(settings),
    }


def _persist_stock_pool_update(*, ts_code: str, action: str) -> List[str]:
    settings = get_settings()
    stock_pool = list(settings.stock_pool)

    if action == "add":
        if ts_code not in stock_pool:
            stock_pool.append(ts_code)
    elif action == "remove":
        stock_pool = [item for item in stock_pool if item != ts_code]
    else:
        raise ValueError(f"Unsupported stock pool action: {action}")

    _write_stock_pool_to_env(stock_pool)
    _clear_settings_cache()
    return stock_pool


def _normalize_ts_code(ts_code: str) -> str:
    normalized = ts_code.strip().upper()
    if not re.fullmatch(r"\d{6}\.(SH|SZ)", normalized):
        raise HTTPException(status_code=400, detail="ts_code must look like 600000.SH or 000001.SZ")
    return normalized


def _write_stock_pool_to_env(stock_pool: List[str]) -> None:
    env_path = _env_file_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    target_line = f"OCTTS_STOCK_POOL={','.join(stock_pool)}"

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated_lines: List[str] = []
    replaced = False

    for line in lines:
        if line.startswith("OCTTS_STOCK_POOL="):
            if not replaced:
                updated_lines.append(target_line)
                replaced = True
            continue
        updated_lines.append(line)

    if not replaced:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(target_line)

    env_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")


def _env_file_path() -> Path:
    env_file = Settings.model_config.get("env_file", ".env")
    if isinstance(env_file, (list, tuple)):
        env_file = env_file[0]
    return Path(env_file)


def _clear_settings_cache() -> None:
    cache_clear = getattr(get_settings, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def _load_memory_keys(memory_store) -> List[str]:
    if hasattr(memory_store, "_load"):
        payload = memory_store._load()
        if isinstance(payload, dict):
            return [key for key in payload.keys() if isinstance(key, str)]
    if hasattr(memory_store, "_client") and hasattr(memory_store, "_prefix"):
        keys = memory_store._client.keys(f"{memory_store._prefix}:*")
        return [key.rsplit(":", maxsplit=1)[-1] for key in keys]
    return []


# 股票筛选相关端点
@app.post("/screen/technical", response_model=ScreenResult)
def screen_technical(criteria: ScreenCriteria) -> ScreenResult:
    """
    执行技术指标筛选

    Args:
        criteria: 筛选条件

    Returns:
        筛选结果
    """
    try:
        screener = StockScreener()
        return screener.screen(criteria)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Screening failed: {exc}") from exc


@app.get("/screen/presets", response_model=List[ScreenPreset])
def get_screen_presets() -> List[ScreenPreset]:
    """获取预设筛选策略"""
    return StockScreener.get_presets()


@app.get("/screen/results/{screen_id}", response_model=ScreenResult)
def get_screen_results(screen_id: str) -> ScreenResult:
    """
    获取筛选结果

    Args:
        screen_id: 筛选ID

    Returns:
        筛选结果
    """
    result = StockScreener.get_screen_result(screen_id)
    if result is None:
        from octts.services.screening_store import ScreeningStore

        result = ScreeningStore(get_settings()).get_screening_result(screen_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No screen result found for {screen_id}")
    return result


@app.get("/market/stocks")
def get_all_stocks() -> Dict[str, object]:
    """
    获取所有可交易股票列表

    Returns:
        股票列表
    """
    try:
        screener = StockScreener()
        stocks = screener.get_all_stocks()
        return {
            "total_count": len(stocks),
            "stocks": stocks
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to get stock list: {exc}") from exc


@app.post("/screen/run-daily")
async def run_daily_screening() -> Dict[str, Any]:
    """
    手动触发每日选股任务

    Returns:
        执行结果
    """
    try:
        scheduler = create_screening_scheduler(get_settings())
        result = await scheduler.run_daily_screening()
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Screening failed: {exc}") from exc


@app.get("/intelligent-screening")
def intelligent_screening_dashboard(tab: str = "overview") -> HTMLResponse:
    """
    智能选股仪表板页面

    Returns:
        HTML页面
    """
    dashboard_payload = _build_recommendation_dashboard_payload(get_settings())

    html_content = render_intelligent_screening_dashboard(
        screening_results=dashboard_payload["screening_results"],
        recommendation_pool=dashboard_payload.get("recommendation_pool") or {},
        ai_analyses=dashboard_payload["ai_analyses"],
        news_clusters=dashboard_payload["news_clusters"],
        intelligent_report=dashboard_payload["intelligent_report"],
        recommendation_summary=dashboard_payload.get("recommendation_summary") or {},
        recommendation_methodology=dashboard_payload.get("recommendation_methodology") or {},
        generated_at=dashboard_payload.get("generated_at"),
        active_tab=tab,
    )

    return HTMLResponse(content=html_content)


@app.post("/screen/intelligent", status_code=202)
async def run_intelligent_screening(request: Request) -> Dict[str, Any]:
    """
    运行智能选股（带AI分析）

    Returns:
        任务状态
    """
    return await create_intelligent_screening_job(request)


@app.post("/screen/intelligent/jobs", status_code=202)
async def create_intelligent_screening_job(request: Request) -> Dict[str, Any]:
    """
    创建智能选股后台任务

    Returns:
        任务状态
    """
    manager = _get_intelligent_screening_job_manager(request)
    settings = get_settings()

    async def runner(progress_callback):
        scheduler = EnhancedScreeningScheduler(settings, progress_callback=progress_callback)
        return await scheduler.run_intelligent_screening()

    try:
        payload = await manager.start_job(runner)
        job = dict(payload["job"])
        job["created"] = payload["created"]
        return job
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create intelligent screening job: {exc}") from exc


@app.get("/screen/intelligent/jobs/active")
async def get_active_intelligent_screening_job(request: Request) -> Dict[str, Any]:
    """查询当前运行中的智能选股后台任务。"""
    manager = _get_intelligent_screening_job_manager(request)
    payload = await manager.get_active_job()
    if payload is None:
        return {"job": None}
    return {"job": payload}


@app.get("/screen/intelligent/jobs/{job_id}")
async def get_intelligent_screening_job(job_id: str, request: Request) -> Dict[str, Any]:
    """
    查询智能选股后台任务状态

    Returns:
        任务状态
    """
    manager = _get_intelligent_screening_job_manager(request)
    payload = await manager.get_job(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Intelligent screening job not found: {job_id}")
    return payload


def _get_intelligent_screening_job_manager(request: Request) -> IntelligentScreeningJobManager:
    manager = getattr(request.app.state, "intelligent_screening_job_manager", None)
    if manager is None:
        manager = IntelligentScreeningJobManager(get_settings().history_dir_path)
        request.app.state.intelligent_screening_job_manager = manager
    return manager


def _build_recommendation_dashboard_payload(settings: Settings) -> Dict[str, Any]:
    payload = _load_intelligent_dashboard_payload(settings)
    payload["recommendation_summary"] = _load_recommendation_summary(settings)
    payload["recommendation_methodology"] = _build_recommendation_methodology_payload(settings)
    return payload


def _load_recommendation_summary(settings: Settings) -> Dict[str, Any]:
    if not settings.use_database:
        return {}
    try:
        from octts.services.screening_store import ScreeningStore

        return ScreeningStore(settings).get_recommendation_summary(lookback_days=30)
    except Exception:
        return {}



def _build_recommendation_methodology_payload(settings: Settings) -> Dict[str, Any]:
    scheduler = EnhancedScreeningScheduler(settings)
    strategies = scheduler._get_active_strategies()
    strategy_items = []
    for strategy in strategies:
        if isinstance(strategy, ScreenPreset):
            strategy_items.append(
                {
                    "id": strategy.id,
                    "name": strategy.name,
                    "description": strategy.description,
                }
            )

    return {
        "strategy_count": len(strategy_items),
        "strategies": strategy_items,
        "candidate_selection": [
            "先汇总所有启用策略的候选股票，优先保留多策略同时命中的标的。",
            "默认先过滤 ST 名称标的与近年连续亏损风险较高的标的。",
            "候选股需满足技术评分不低于 45。",
            "候选股需满足成交量比不低于 1.0，优先考虑放量标的。",
            "若 RSI 高于 85 或低于 15，则视为过热/过冷，先过滤。",
            "候选池按优先级收敛为持续跟踪池 Top10，其中前台 Top5 作为默认展示名单。",
        ],
        "ai_analysis": [
            "默认只对前台 Top3 与高关注股票补充执行 AI 分析，shadow 仅保留规则跟踪，不调用 LLM。",
            "分析页面会同步展示技术面、基本面、市场情绪、新闻舆情四个维度的结果。",
            "AI 还会给出 overall_confidence 作为最终推荐分数的置信度权重。",
        ],
        "score_formula": [
            "基础分 = AI 综合分数 overall_score。",
            "若股票出现在高重要性新闻热点中，额外加 3 分。",
            "每多命中 1 个策略，额外加 5 分。",
            "再叠加小幅行业近 3 日资金氛围修正，基于所属行业近 3 日净流入与净流入占比做温和加减分。",
            "最终分数 = (AI 综合分数 + 新闻加分 + 多策略加分 + 行业近 3 日资金氛围修正) × AI 置信度。",
            "最终分数达到 55 分才会进入最终推荐池。",
        ],
        "recommendation_levels": [
            {"label": "强烈推荐", "rule": "最终分数 ≥ 80", "description": "多维度共振，建议重点关注"},
            {"label": "推荐", "rule": "70 ≤ 最终分数 < 80", "description": "技术面良好，可适当关注"},
            {"label": "观察", "rule": "60 ≤ 最终分数 < 70", "description": "有一定机会，建议跟踪"},
            {"label": "谨慎", "rule": "最终分数 < 60", "description": "暂不建议操作"},
        ],
        "tracking_metrics": [
            "入场价格统一使用推荐日收盘价。",
            "自动回填 T+1 / T+3 / T+5 / T+10 收益。",
            "10 日最大回撤按推荐日收盘价为基准计算。",
            "5 日胜率定义为 return_5d > 0。",
            "默认基准为沪深300（000300.SH），用于计算 5 日超额收益。",
        ],
    }


def _load_intelligent_dashboard_payload(settings: Settings, trade_date: Optional[str] = None) -> Dict[str, Any]:
    """Load the intelligent screening payload for the dashboard."""
    file_name = f"{trade_date}.json" if trade_date else "latest.json"
    snapshot_path = Path(settings.history_dir_path) / "intelligent_screening" / file_name
    if snapshot_path.exists():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "generated_at": payload.get("generated_at"),
                "screening_results": payload.get("screening_results", {}),
                "recommendation_pool": payload.get("recommendation_pool", {}),
                "ai_analyses": payload.get("ai_analyses", {}),
                "news_clusters": payload.get("news_clusters", []),
                "intelligent_report": payload.get("intelligent_report"),
                "recommendation_summary": payload.get("recommendation_summary", {}),
                "recommendation_methodology": payload.get("recommendation_methodology", {}),
                "report_context": payload.get("report_context", {}),
                "data_source": "snapshot",
            }
        except Exception:
            pass

    from octts.services.screening_store import ScreeningStore

    store = ScreeningStore(settings)
    try:
        latest_results = store.get_latest_results()
    except Exception:
        latest_results = {}
    ai_analyses: Dict[str, Dict[str, Any]] = {}
    total_stocks = 0
    for strategy_result in latest_results.values():
        stocks = strategy_result.get("stocks", []) if isinstance(strategy_result, dict) else []
        total_stocks += len(stocks)
        for stock in stocks:
            if not isinstance(stock, dict):
                continue
            code = stock.get("ts_code")
            if not code:
                continue
            current_score = float(stock.get("recommendation_score") or stock.get("score") or stock.get("technical_score") or 0.0)
            existing_score = float(ai_analyses.get(code, {}).get("recommendation_score", -1))
            if current_score <= existing_score:
                continue
            ai_analyses[code] = {
                "name": stock.get("name", ""),
                "recommendation_score": current_score,
                "overall_score": float(stock.get("score") or stock.get("technical_score") or current_score),
                "overall_confidence": 0.6,
                "recommendation": "基于最新选股结果，建议结合实时行情进一步确认。",
                "technical_score": float(stock.get("technical_score") or stock.get("score") or 0.0),
                "fundamental_score": float(stock.get("market_cap") or 0.0),
                "sentiment_score": 0.0,
                "news_score": 0.0,
                "summary": "当前页面展示的是最近一次筛选结果生成的候选股概览。",
                "technical_summary": "匹配原因：" + "；".join(stock.get("match_reasons", [])[:4]),
                "technical_signal": stock.get("trend_status") or "待进一步确认",
                "key_points": stock.get("match_reasons", [])[:6],
                "final_decision": "优先跟踪高分标的，等待下一次智能分析刷新。",
                "has_conflict": False,
                "conflict_points": [],
            }

    frontlist = sorted(
        (
            {
                "ts_code": code,
                "name": analysis.get("name", ""),
                "priority_score": float(analysis.get("priority_score") or analysis.get("overall_score") or 0.0),
                "recommendation_score": float(analysis.get("recommendation_score") or analysis.get("overall_score") or 0.0),
                "in_frontlist": True,
                "tracking_status": "active" if index < 3 else "candidate",
                "llm_focus_level": "high" if index < 3 else "medium",
                "hit_streak_days": 0,
                "miss_streak_days": 0,
                "source_tag": "今日Top3" if index < 3 else "今日候选",
                "is_repeat_pick": False,
                "recommendation_text": analysis.get("recommendation", ""),
                "ai_confidence": analysis.get("overall_confidence") or analysis.get("confidence") or 0.6,
            }
            for index, (code, analysis) in enumerate(sorted(ai_analyses.items(), key=lambda item: item[1].get("recommendation_score", 0), reverse=True)[:10])
        ),
        key=lambda item: item["recommendation_score"],
        reverse=True,
    )
    recommendation_pool = {
        "frontlist": frontlist,
        "shadow": [],
        "shadow_symbols": [],
        "today_top": [item for item in frontlist if item.get("source_tag") == "今日Top3"],
        "yesterday_continuations": [item for item in frontlist if item.get("source_tag") == "昨日延续"],
    }
    return {
        "screening_results": {
            "strategy_count": len(latest_results) if latest_results else None,
            "total_stocks": total_stocks if latest_results else None,
            "final_recommendations": len(frontlist),
            "frontlist_count": len(frontlist),
            "shadow_count": 0,
            "candidate_count": len(ai_analyses),
            "today_top_count": len(recommendation_pool["today_top"]),
            "continuation_count": len(recommendation_pool["yesterday_continuations"]),
        },
        "recommendation_pool": recommendation_pool,
        "ai_analyses": ai_analyses,
        "news_clusters": [],
        "intelligent_report": {
            "title": "智能选股页面",
            "summary": "当前展示最近一次选股结果汇总。运行一次智能选股后，页面会展示完整的新闻热点、AI多维分析和智能报告。",
            "sections": [
                {
                    "title": "当前状态",
                    "content": "页面已切换为真实数据源，不再使用固定 mock 数据。",
                }
            ],
            "blocks": {},
        },
        "report_context": {},
        "data_source": "fallback",
    }


def _build_stock_intelligent_insight(ts_code: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    report_context = payload.get("report_context", {}) or {}
    report_blocks = ((payload.get("intelligent_report") or {}).get("blocks") or {})
    focus_items = {item.get("ts_code"): item for item in report_blocks.get("focus_stocks") or [] if isinstance(item, dict) and item.get("ts_code")}
    review_items = {item.get("ts_code"): item for item in report_blocks.get("yesterday_reviews") or [] if isinstance(item, dict) and item.get("ts_code")}
    context_items = {}
    for key in ("today_top3", "yesterday_top3_review", "comparison_candidates"):
        for item in report_context.get(key) or []:
            if isinstance(item, dict) and item.get("ts_code"):
                context_items[item["ts_code"]] = item
    recommendation_pool = payload.get("recommendation_pool") or {}
    frontlist = recommendation_pool.get("frontlist") or []
    today_top = recommendation_pool.get("today_top") or []
    yesterday_continuations = recommendation_pool.get("yesterday_continuations") or []
    pool_item = next((item for item in frontlist if isinstance(item, dict) and item.get("ts_code") == ts_code), {})
    today_top_item = next((item for item in today_top if isinstance(item, dict) and item.get("ts_code") == ts_code), {})
    continuation_item = next((item for item in yesterday_continuations if isinstance(item, dict) and item.get("ts_code") == ts_code), {})
    focus_item = focus_items.get(ts_code, {})
    review_item = review_items.get(ts_code, {})
    context_item = context_items.get(ts_code, {})
    action = {}
    for source in (
        focus_item.get("action_plan"),
        review_item.get("action_plan"),
        context_item.get("action_plan"),
        today_top_item.get("action_plan"),
        continuation_item.get("action_plan"),
        pool_item.get("action_plan"),
    ):
        if isinstance(source, dict):
            action.update({key: value for key, value in source.items() if value not in (None, "")})
    return {
        "ts_code": ts_code,
        "in_today_top3": pool_item.get("source_tag") == "今日Top3" or context_item.get("source_tag") == "今日Top3" or bool(today_top_item),
        "in_yesterday_review": bool(review_item or context_item.get("review_status") or continuation_item),
        "source_tag": pool_item.get("source_tag") or continuation_item.get("source_tag") or today_top_item.get("source_tag") or context_item.get("source_tag"),
        "recommendation_score": context_item.get("recommendation_score") or pool_item.get("recommendation_score") or continuation_item.get("recommendation_score") or today_top_item.get("recommendation_score"),
        "overall_score": context_item.get("overall_score") or context_item.get("priority_score") or pool_item.get("priority_score") or continuation_item.get("priority_score") or today_top_item.get("priority_score"),
        "confidence": context_item.get("display_confidence") or context_item.get("overall_confidence") or pool_item.get("ai_confidence") or continuation_item.get("ai_confidence") or today_top_item.get("ai_confidence"),
        "core_highlights": focus_item.get("core_highlights") or [],
        "risk_warnings": focus_item.get("risk_warnings") or review_item.get("risk_warnings") or [],
        "overall_assessment": focus_item.get("overall_assessment") or review_item.get("overall_assessment") or context_item.get("summary") or context_item.get("recommendation_text") or continuation_item.get("summary") or continuation_item.get("recommendation_text") or today_top_item.get("summary") or today_top_item.get("recommendation_text"),
        "technical_signal": focus_item.get("technical_signal") or review_item.get("technical_signal") or context_item.get("technical_signal") or pool_item.get("technical_signal") or continuation_item.get("technical_signal") or today_top_item.get("technical_signal"),
        "recommendation_text": context_item.get("recommendation_text") or pool_item.get("recommendation_text") or continuation_item.get("recommendation_text") or today_top_item.get("recommendation_text"),
        "action_plan": {
            "action_bias": action.get("action_bias"),
            "entry_zone": action.get("entry_zone"),
            "take_profit": action.get("take_profit"),
            "stop_loss": action.get("stop_loss"),
            "holding_horizon": action.get("holding_horizon"),
            "invalid_condition": action.get("invalid_condition"),
        },
        "yesterday_vs_today": {
            "previous_recommendation_score": context_item.get("previous_recommendation_score") or review_item.get("previous_recommendation_score"),
            "previous_overall_score": context_item.get("previous_overall_score") or review_item.get("previous_overall_score"),
            "previous_confidence": context_item.get("previous_confidence") or review_item.get("previous_confidence"),
            "today_verdict": review_item.get("today_verdict") or context_item.get("today_verdict") or continuation_item.get("today_verdict"),
            "review_status": review_item.get("review_status") or review_item.get("status") or context_item.get("review_status") or continuation_item.get("review_status") or continuation_item.get("status"),
        },
    }


def _build_intelligent_overview_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ai_analyses = payload.get("ai_analyses", {}) or {}
    recommendation_pool = payload.get("recommendation_pool", {}) or {}
    frontlist = recommendation_pool.get("frontlist") or []
    if frontlist:
        sorted_recommendations = sorted(
            (
                {
                    "ts_code": item.get("ts_code"),
                    "name": item.get("name", ""),
                    "score": float(((ai_analyses.get(item.get("ts_code"), {}) or {}).get("overall_score") or item.get("priority_score") or 0.0)),
                    "overall_score": float(((ai_analyses.get(item.get("ts_code"), {}) or {}).get("overall_score") or item.get("priority_score") or 0.0)),
                    "priority_score": float(item.get("priority_score") or (ai_analyses.get(item.get("ts_code"), {}) or {}).get("priority_score") or 0.0),
                    "recommendation_score": float(((ai_analyses.get(item.get("ts_code"), {}) or {}).get("recommendation_score") or item.get("recommendation_score") or item.get("score") or 0.0)),
                    "confidence": float((ai_analyses.get(item.get("ts_code"), {}) or {}).get("confidence") or (ai_analyses.get(item.get("ts_code"), {}) or {}).get("overall_confidence") or item.get("ai_confidence") or 0.0),
                    "technical_signal": item.get("technical_signal") or (ai_analyses.get(item.get("ts_code"), {}) or {}).get("technical_signal") or "信号待确认",
                    "recommendation": item.get("recommendation_text") or (ai_analyses.get(item.get("ts_code"), {}) or {}).get("final_decision") or (ai_analyses.get(item.get("ts_code"), {}) or {}).get("recommendation") or "建议继续观察",
                    "summary": (ai_analyses.get(item.get("ts_code"), {}) or {}).get("summary") or (ai_analyses.get(item.get("ts_code"), {}) or {}).get("technical_summary") or "",
                    "hit_streak_days": int(item.get("hit_streak_days") or 0),
                    "miss_streak_days": int(item.get("miss_streak_days") or 0),
                    "tracking_status": item.get("tracking_status") or "active",
                    "llm_focus_level": item.get("llm_focus_level") or "medium",
                    "source_tag": item.get("source_tag") or "今日Top3",
                    "is_repeat_pick": bool(item.get("is_repeat_pick", False)),
                }
                for item in frontlist
                if isinstance(item, dict)
            ),
            key=lambda item: item["recommendation_score"],
            reverse=True,
        )[:10]
    else:
        sorted_recommendations = sorted(
            (
                {
                    "ts_code": code,
                    "name": analysis.get("name", ""),
                    "score": float(analysis.get("overall_score") or analysis.get("priority_score") or 0.0),
                    "overall_score": float(analysis.get("overall_score") or analysis.get("priority_score") or 0.0),
                    "priority_score": float(analysis.get("priority_score") or analysis.get("overall_score") or 0.0),
                    "recommendation_score": float(analysis.get("recommendation_score") or analysis.get("score") or analysis.get("overall_score") or 0.0),
                    "confidence": float(analysis.get("confidence") or analysis.get("overall_confidence") or 0.0),
                    "technical_signal": analysis.get("technical_signal") or "信号待确认",
                    "recommendation": analysis.get("final_decision") or analysis.get("recommendation") or "建议继续观察",
                    "summary": analysis.get("summary") or analysis.get("technical_summary") or "",
                    "hit_streak_days": int(analysis.get("hit_streak_days") or 0),
                    "miss_streak_days": int(analysis.get("miss_streak_days") or 0),
                    "tracking_status": analysis.get("tracking_status") or "active",
                    "llm_focus_level": analysis.get("llm_focus_level") or "medium",
                    "source_tag": analysis.get("source_tag") or "今日Top3",
                    "is_repeat_pick": bool(analysis.get("is_repeat_pick", False)),
                }
                for code, analysis in ai_analyses.items()
                if isinstance(analysis, dict)
            ),
            key=lambda item: item["recommendation_score"],
            reverse=True,
        )[:10]
    report = payload.get("intelligent_report") or {}
    screening_results = payload.get("screening_results") or {}
    recommendation_summary = payload.get("recommendation_summary") or {}
    stats = recommendation_summary.get("stats") or {}
    return {
        "generated_at": payload.get("generated_at"),
        "strategy_count": screening_results.get("strategy_count", 0),
        "total_stocks": screening_results.get("total_stocks", 0),
        "final_recommendations": screening_results.get("final_recommendations", len(sorted_recommendations)),
        "frontlist_count": screening_results.get("frontlist_count", len(sorted_recommendations)),
        "shadow_count": 0,
        "candidate_count": screening_results.get("candidate_count", len(sorted_recommendations)),
        "today_top_count": screening_results.get("today_top_count", len([item for item in sorted_recommendations if item.get("source_tag") == "今日Top3"])),
        "continuation_count": screening_results.get("continuation_count", len([item for item in sorted_recommendations if item.get("source_tag") == "昨日延续"])),
        "news_cluster_count": len(payload.get("news_clusters", []) or []),
        "top_recommendations": sorted_recommendations[:10],
        "report_title": report.get("title") or "智能选股报告",
        "report_summary": report.get("summary") or "运行一次智能选股后，这里会显示最新摘要。",
        "tracked_count": stats.get("window_count", len(sorted_recommendations)),
        "win_rate_5d": stats.get("win_rate_5d", 0.0),
        "shadow_symbols": [],
    }


@app.get("/screen/history")
def get_screening_history(
    strategy_id: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365)
) -> Dict[str, Any]:
    """
    获取筛选历史记录

    Args:
        strategy_id: 策略ID（可选）
        days: 查询天数

    Returns:
        历史记录
    """
    try:
        from octts.services.screening_store import ScreeningStore

        store = ScreeningStore(get_settings())
        if strategy_id:
            history = store.get_screening_history(strategy_id, days)
        else:
            history = store.get_all_screening_history(days)

        return {
            "total_count": len(history),
            "history": history
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {exc}") from exc


@app.get("/stock/{ts_code}/performance")
def get_stock_performance(
    ts_code: str,
    days: int = Query(default=30, ge=1, le=365)
) -> Dict[str, Any]:
    """
    获取股票在选股系统中的历史表现

    Args:
        ts_code: 股票代码
        days: 查询天数

    Returns:
        股票表现数据
    """
    try:
        from octts.services.screening_store import ScreeningStore

        store = ScreeningStore(get_settings())
        performance = store.get_stock_performance(ts_code, days=days)

        return performance
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get stock performance: {exc}") from exc


@app.get("/backtest")
def backtest_page() -> HTMLResponse:
    """策略回测页面"""
    from octts.ui.backtest_page import render_backtest_page
    return HTMLResponse(content=render_backtest_page())


@app.post("/api/backtest")
async def run_backtest(request: LightweightBacktestRequest) -> Dict[str, Any]:
    """
    运行策略回测

    Args:
        request: 包含回测参数的请求
            - start_date: 开始日期
            - end_date: 结束日期
            - holding_days: 持有天数
            - top_n: 每次选股数量
            - strategies: 策略ID列表

    Returns:
        回测结果
    """
    try:
        from octts.services.lightweight_backtester import LightweightBacktester

        backtester = LightweightBacktester(get_settings())

        # 获取策略
        all_presets = StockScreener.get_presets()
        selected_strategies = [
            preset for preset in all_presets
            if preset.id in request.strategies
        ]

        if not selected_strategies:
            raise HTTPException(status_code=400, detail="No valid strategies selected")

        # 运行回测
        results = backtester.compare_strategies(
            strategies=selected_strategies,
            start_date=request.start_date,
            end_date=request.end_date,
            holding_days=request.holding_days,
            top_n=request.top_n,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            stock_pool=request.stock_pool,
        )

        # 生成摘要
        best_strategy = max(results.items(), key=lambda x: x[1].total_return)
        stock_scope = (
            f"限定股票池：{', '.join(request.stock_pool)}"
            if request.stock_pool
            else "全市场中命中当前策略条件的股票"
        )
        summary = {
            "period": f"{request.start_date} - {request.end_date}",
            "best_strategy": best_strategy[0],
            "best_total_return": best_strategy[1].total_return,
            "best_max_drawdown": best_strategy[1].max_drawdown,
            "best_sharpe_ratio": best_strategy[1].sharpe_ratio,
            "commission_rate": request.commission_rate,
            "slippage_rate": request.slippage_rate,
            "stock_scope": stock_scope,
            "selected_stock_pool": request.stock_pool,
            "recommendation": (
                f"{stock_scope}下，建议关注{best_strategy[0]}策略，"
                f"总收益{best_strategy[1].total_return:.1f}%，"
                f"胜率{best_strategy[1].win_rate:.1%}"
            )
        }

        # 转换结果格式
        formatted_results = {}
        for name, result in results.items():
            equity = 1.0
            peak = 1.0
            equity_curve = []
            sorted_records = sorted(result.detail_records, key=lambda item: item.get("entry_date", ""))
            for record in sorted_records:
                record_return = float(record.get("return_pct", 0.0)) / 100.0
                equity *= (1 + record_return)
                peak = max(peak, equity)
                drawdown = 0.0 if peak <= 0 else (equity - peak) / peak
                equity_curve.append({
                    "trade_date": record.get("entry_date") or record.get("signal_date") or "",
                    "value": round(equity, 4),
                    "drawdown": round(drawdown * 100, 2),
                })

            formatted_results[name] = {
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "avg_return": result.avg_return,
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "detail_records": sorted_records,
                "equity_curve": equity_curve,
            }

        return {
            "results": formatted_results,
            "summary": summary
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
