from __future__ import annotations

from typing import Dict, List

from octts.config import Settings
from octts.schemas.report import HistoricalAnalysisRecord
from octts.services.automation_scheduler import build_automation_slots
from octts.services.history_store import FileHistoryStore
from octts.services.position_store import FilePositionStore


def serialize_record(
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


def build_validation_summary(records: List[HistoricalAnalysisRecord]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        status = record.validation.status
        summary[status] = summary.get(status, 0) + 1
    return summary


def build_openclaw_status(settings: Settings) -> Dict[str, object]:
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
