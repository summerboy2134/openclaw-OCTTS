from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from zipfile import ZIP_DEFLATED, ZipFile
import json

from octts.config import Settings
from octts.schemas.report import HistoricalAnalysisRecord
from octts.services.automation_scheduler import build_automation_slots
from octts.services.history_store import FileHistoryStore
from octts.services.position_store import FilePositionStore
from octts.ui.dashboard import render_dashboard_html, render_stock_detail_html
from octts.ui.intelligent_screening_dashboard import render_intelligent_screening_dashboard


class ReportExporter:
    def __init__(self, *, settings: Settings, history_store: FileHistoryStore, position_store: FilePositionStore) -> None:
        self._settings = settings
        self._history_store = history_store
        self._position_store = position_store

    def build_dashboard_payload(self, ts_codes: Optional[list[str]] = None) -> dict[str, object]:
        latest_records = self._filter_latest_records(self._history_store.list_latest(), ts_codes)
        cards = [
            _serialize_record(
                record,
                self._history_store,
                self._position_store,
                history_limit=8,
            )
            for record in latest_records
        ]
        return {
            "generated_at": latest_records[0].generated_at.isoformat() if latest_records else None,
            "cards": cards,
            "validation_summary": _build_validation_summary(latest_records),
            "default_stock_pool": self._settings.stock_pool,
            "openclaw_status": _build_openclaw_status(self._settings),
        }

    def build_stock_detail_payload(self, ts_code: str) -> dict[str, object]:
        records = self._history_store.list_records(ts_code, limit=self._settings.history_limit_per_symbol)
        if not records:
            raise ValueError(f"No history found for {ts_code}")

        latest = records[-1]
        from octts.api import _build_stock_intelligent_insight

        intelligent_payload = _load_intelligent_dashboard_payload(self._settings)
        return {
            "generated_at": latest.generated_at.isoformat(),
            "symbol": _serialize_record(
                latest,
                self._history_store,
                self._position_store,
                history_limit=self._settings.history_limit_per_symbol,
            ),
            "validation_summary": _build_validation_summary(records),
            "openclaw_status": _build_openclaw_status(self._settings),
            "position_status": self._position_store.get_status(ts_code),
            "intelligent_screening_insight": _build_stock_intelligent_insight(ts_code, intelligent_payload),
        }

    def export_latest_report_zip(self, ts_codes: Optional[list[str]] = None) -> tuple[str, bytes]:
        latest_records = self._filter_latest_records(self._history_store.list_latest(), ts_codes)
        if not latest_records:
            raise ValueError("No analysis history available for export.")

        dashboard_payload = self.build_dashboard_payload(ts_codes)
        latest_generated_at = latest_records[0].generated_at.strftime("%Y%m%d-%H%M%S")
        archive_name = f"octts-report-{latest_generated_at}.zip"
        buffer = BytesIO()

        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                "index.html",
                render_dashboard_html(
                    dashboard_payload,
                    stock_detail_href_prefix="./stocks/",
                    stock_detail_href_suffix=".html",
                    interactive=False,
                ),
            )
            for record in latest_records:
                archive.writestr(
                    f"stocks/{record.report.ts_code}.html",
                    render_stock_detail_html(
                        record.report.ts_code,
                        self.build_stock_detail_payload(record.report.ts_code),
                        back_href="../index.html",
                        interactive=False,
                    ),
                )

        return archive_name, buffer.getvalue()

    def build_intelligent_screening_payload(self, trade_date: Optional[str] = None) -> dict[str, Any]:
        payload = _load_intelligent_dashboard_payload(self._settings, trade_date=trade_date)
        payload["recommendation_summary"] = _load_recommendation_summary(self._settings)
        payload["recommendation_methodology"] = _build_recommendation_methodology_payload(self._settings)
        return payload

    def export_latest_intelligent_screening_zip(self) -> tuple[str, bytes]:
        payload = self.build_intelligent_screening_payload()
        generated_at = str(payload.get("generated_at") or "latest").replace(":", "").replace("T", "-")
        archive_name = f"octts-intelligent-screening-{generated_at}.zip"
        buffer = BytesIO()

        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                "index.html",
                render_intelligent_screening_dashboard(
                    screening_results=payload.get("screening_results") or {},
                    recommendation_pool=payload.get("recommendation_pool") or {},
                    ai_analyses=payload.get("ai_analyses") or {},
                    news_clusters=payload.get("news_clusters") or [],
                    intelligent_report=payload.get("intelligent_report") or {},
                    recommendation_summary=payload.get("recommendation_summary") or {},
                    recommendation_methodology=payload.get("recommendation_methodology") or {},
                    generated_at=payload.get("generated_at"),
                    dashboard_href="./dashboard.html",
                    backtest_href=None,
                    refresh_href="./index.html",
                    jobs_api_base="",
                    autorun_enabled=False,
                    stock_detail_href_prefix="./stocks/",
                    stock_detail_href_suffix=".html",
                ),
            )
            archive.writestr(
                "dashboard.html",
                render_dashboard_html(
                    self.build_dashboard_payload(),
                    stock_detail_href_prefix="./stocks/",
                    stock_detail_href_suffix=".html",
                    interactive=False,
                ),
            )
            for record in self._history_store.list_latest():
                archive.writestr(
                    f"stocks/{record.report.ts_code}.html",
                    render_stock_detail_html(
                        record.report.ts_code,
                        self.build_stock_detail_payload(record.report.ts_code),
                        back_href="../dashboard.html",
                        interactive=False,
                    ),
                )

        return archive_name, buffer.getvalue()

    def _filter_latest_records(
        self, records: list[HistoricalAnalysisRecord], ts_codes: Optional[list[str]]
    ) -> list[HistoricalAnalysisRecord]:
        if not ts_codes:
            return records
        allowed_codes = {item.strip().upper() for item in ts_codes if item and item.strip()}
        if not allowed_codes:
            return []
        return [record for record in records if record.report.ts_code.upper() in allowed_codes]


def _serialize_record(
    record: HistoricalAnalysisRecord,
    history_store: FileHistoryStore,
    position_store: FilePositionStore,
    *,
    history_limit: int,
) -> dict[str, object]:
    history = history_store.list_records(record.report.ts_code, limit=history_limit)
    return {
        "ts_code": record.report.ts_code,
        "generated_at": record.generated_at.isoformat(),
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
        "history": [item.model_dump(mode="json") for item in history],
    }


def _build_validation_summary(records: list[HistoricalAnalysisRecord]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        status = record.validation.status
        summary[status] = summary.get(status, 0) + 1
    return summary


def _build_openclaw_status(settings: Settings) -> dict[str, object]:
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


def _load_intelligent_dashboard_payload(settings: Settings, trade_date: Optional[str] = None) -> dict[str, Any]:
    file_name = f"{trade_date}.json" if trade_date else "latest.json"
    snapshot_path = Path(settings.history_dir_path) / "intelligent_screening" / file_name
    if snapshot_path.exists():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return {
                "generated_at": payload.get("generated_at"),
                "screening_results": payload.get("screening_results", {}),
                "recommendation_pool": payload.get("recommendation_pool", {}),
                "ai_analyses": payload.get("ai_analyses", {}),
                "news_clusters": payload.get("news_clusters", []),
                "intelligent_report": payload.get("intelligent_report"),
                "report_context": payload.get("report_context", {}),
            }
        except Exception:
            pass
    return {
        "generated_at": None,
        "screening_results": {},
        "recommendation_pool": {},
        "ai_analyses": {},
        "news_clusters": [],
        "intelligent_report": {},
        "report_context": {},
    }


def _load_recommendation_summary(settings: Settings) -> dict[str, Any]:
    if not settings.use_database:
        return {}
    try:
        from octts.services.recommendation_tracker import RecommendationTracker
        from octts.services.screening_store import ScreeningStore

        RecommendationTracker(settings).update_recommendation_performance(lookback_days=15)
        return ScreeningStore(settings).get_recommendation_summary(lookback_days=30)
    except Exception:
        return {}


def _build_recommendation_methodology_payload(settings: Settings) -> dict[str, Any]:
    from octts.services.enhanced_screening_scheduler import EnhancedScreeningScheduler
    from octts.schemas.screener import ScreenPreset

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
