from __future__ import annotations

from io import BytesIO
from typing import Optional
from zipfile import ZIP_DEFLATED, ZipFile

from octts.config import Settings
from octts.schemas.report import HistoricalAnalysisRecord
from octts.services.automation_scheduler import build_automation_slots
from octts.services.history_store import FileHistoryStore
from octts.ui.dashboard import render_dashboard_html, render_stock_detail_html


class ReportExporter:
    def __init__(self, *, settings: Settings, history_store: FileHistoryStore) -> None:
        self._settings = settings
        self._history_store = history_store

    def build_dashboard_payload(self, ts_codes: Optional[list[str]] = None) -> dict[str, object]:
        latest_records = self._filter_latest_records(self._history_store.list_latest(), ts_codes)
        cards = [
            _serialize_record(
                record,
                self._history_store,
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
        return {
            "generated_at": latest.generated_at.isoformat(),
            "symbol": _serialize_record(
                latest,
                self._history_store,
                history_limit=self._settings.history_limit_per_symbol,
            ),
            "validation_summary": _build_validation_summary(records),
            "openclaw_status": _build_openclaw_status(self._settings),
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
