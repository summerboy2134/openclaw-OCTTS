from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel

from octts.clients.email_client import EmailClient
from octts.clients.llm_client import LLMClient
from octts.clients.tushare_client import TushareClient
from octts.clients.wecom_client import WeComClient
from octts.config import Settings, get_settings
from octts.schemas.backtest import BacktestRequest, BacktestResult
from octts.schemas.report import AnalysisRequest, AnalysisResult
from octts.services.analysis_pipeline import AnalysisPipeline
from octts.services.automation_scheduler import build_automation_slots, create_automation_scheduler
from octts.services.backtest_engine import BacktestEngine
from octts.services.history_store import FileHistoryStore
from octts.services.memory_store import create_memory_store
from octts.services.report_email_service import ReportEmailService
from octts.services.report_exporter import ReportExporter
from octts.ui.dashboard import render_dashboard_html, render_stock_detail_html


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = create_automation_scheduler(
        settings=get_settings(),
        pipeline_factory=_build_pipeline,
        report_email_service_factory=_build_report_email_service,
    )
    app.state.automation_scheduler = scheduler
    if scheduler:
        scheduler.start()
    try:
        yield
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)


app = FastAPI(title="OCTTS", version="0.1.0", lifespan=lifespan)


class StockPoolItemRequest(BaseModel):
    ts_code: str


class AnalysisActionResponse(BaseModel):
    cleared_symbols: list[str] = []
    cleared_all: bool = False
    removed_records: int = 0
    removed_memory_items: int = 0
    removed_generated_at: str | None = None
    remaining_records: int = 0
    updated_memory: bool = False


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> dict[str, str]:
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
def dashboard_data() -> dict[str, object]:
    settings = get_settings()
    history_store = _build_history_store(settings)
    latest_records = history_store.list_latest()
    cards = [_serialize_record(record, history_store, history_limit=8) for record in latest_records]
    validation_summary = _build_validation_summary(latest_records)

    return {
        "generated_at": latest_records[0].generated_at if latest_records else None,
        "cards": cards,
        "validation_summary": validation_summary,
        "default_stock_pool": settings.stock_pool,
        "openclaw_status": _build_openclaw_status(settings),
    }


@app.get("/stocks/{ts_code}", response_class=HTMLResponse)
def stock_detail_page(ts_code: str) -> HTMLResponse:
    return HTMLResponse(render_stock_detail_html(ts_code))


@app.get("/stocks/{ts_code}/data")
def stock_detail_data(ts_code: str) -> dict[str, object]:
    settings = get_settings()
    history_store = _build_history_store(settings)
    records = history_store.list_records(ts_code, limit=settings.history_limit_per_symbol)
    if not records:
        raise HTTPException(status_code=404, detail=f"No history found for {ts_code}")

    latest = records[-1]
    return {
        "generated_at": latest.generated_at,
        "symbol": _serialize_record(latest, history_store, history_limit=settings.history_limit_per_symbol),
        "validation_summary": _build_validation_summary(records),
        "openclaw_status": _build_openclaw_status(settings),
    }


@app.get("/openclaw/status")
def openclaw_status() -> dict[str, object]:
    return _build_openclaw_status(get_settings())


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
        wecom_client=wecom_client,
    )
    return BacktestEngine(
        pipeline=pipeline,
        market_data_client=tushare_client,
    )


def _build_report_exporter() -> ReportExporter:
    settings = get_settings()
    return ReportExporter(
        settings=settings,
        history_store=_build_history_store(settings),
    )


def _build_report_email_service() -> ReportEmailService:
    settings = get_settings()
    return ReportEmailService(
        settings=settings,
        history_store=_build_history_store(settings),
        report_exporter=_build_report_exporter(),
        email_client=EmailClient(settings),
    )


def _build_history_store(settings: Settings) -> FileHistoryStore:
    return FileHistoryStore(
        directory_path=settings.history_dir_path,
        limit_per_symbol=settings.history_limit_per_symbol,
    )


def _serialize_record(record, history_store: FileHistoryStore, history_limit: int) -> dict[str, object]:
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
        "history": [item.model_dump(mode="json") for item in history],
    }


def _build_validation_summary(records) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        status = record.validation.status
        summary[status] = summary.get(status, 0) + 1
    return summary


def _build_openclaw_status(settings: Settings) -> dict[str, object]:
    automation_enabled = settings.automation_enabled
    status_note = (
        "当前已启用内置定时分析，服务会按配置时间自动扫描默认股票池。"
        if automation_enabled
        else "当前保持外部编排模式。"
    ) + "如需真实联动状态，可在后续接入网关健康检查或 job 列表接口。"
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
        "status_note": status_note,
    }


def _persist_stock_pool_update(*, ts_code: str, action: str) -> list[str]:
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


def _write_stock_pool_to_env(stock_pool: list[str]) -> None:
    env_path = _env_file_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    target_line = f"OCTTS_STOCK_POOL={','.join(stock_pool)}"

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated_lines: list[str] = []
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
    get_settings.cache_clear()


def _load_memory_keys(memory_store) -> list[str]:
    if hasattr(memory_store, "_load"):
        payload = memory_store._load()
        if isinstance(payload, dict):
            return [key for key in payload.keys() if isinstance(key, str)]
    if hasattr(memory_store, "_client") and hasattr(memory_store, "_prefix"):
        keys = memory_store._client.keys(f"{memory_store._prefix}:*")
        return [key.rsplit(":", maxsplit=1)[-1] for key in keys]
    return []
