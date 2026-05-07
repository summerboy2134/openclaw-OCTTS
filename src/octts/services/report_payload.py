from __future__ import annotations

from typing import Dict, List, Optional

from octts.config import Settings
from octts.schemas.report import HistoricalAnalysisRecord, PriceSnapshot
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
        "analysis_context": build_analysis_context(record.snapshot),
        "memory": record.report.memory.model_dump(mode="json"),
        "position_status": position_store.get_status(record.report.ts_code),
        "history": [item.model_dump(mode="json") for item in history],
    }


def build_analysis_context(snapshot: PriceSnapshot) -> dict[str, object]:
    bars = _build_daily_bars(snapshot)
    lows = [_safe_float(item.get("low")) for item in bars[-5:]]
    highs = [_safe_float(item.get("high")) for item in bars[-5:]]
    closes = [_safe_float(item.get("close")) for item in bars]
    valid_closes = [value for value in closes if value is not None]
    ma20 = sum(valid_closes[-20:]) / 20 if len(valid_closes) >= 20 else None
    close = _safe_float(snapshot.close)

    return {
        "technical": {
            "ma20": ma20,
            "distance_to_ma20_pct": _calculate_distance_pct(close, ma20),
            "vol_ratio": _safe_float(snapshot.vol_ratio),
            "turnover_rate": _safe_float(snapshot.turnover_rate),
            "recent_5d_low": _min_optional(lows),
            "recent_5d_high": _max_optional(highs),
        },
        "moneyflow": dict(snapshot.moneyflow_summary or {}),
    }


def _build_daily_bars(snapshot: PriceSnapshot) -> list[dict[str, object]]:
    bars_by_date: dict[str, dict[str, object]] = {}
    for raw_bar in snapshot.daily_summary:
        if not isinstance(raw_bar, dict):
            continue
        bar = _normalize_daily_bar(raw_bar)
        trade_date = bar.get("trade_date")
        if trade_date:
            bars_by_date[str(trade_date)] = bar

    current_bar = _normalize_daily_bar(
        {
            "trade_date": snapshot.trade_date,
            "open": snapshot.open,
            "high": snapshot.high,
            "low": snapshot.low,
            "close": snapshot.close,
            "pct_chg": snapshot.pct_chg,
            "amount": snapshot.amount,
            "turnover_rate": snapshot.turnover_rate,
            "vol_ratio": snapshot.vol_ratio,
        }
    )
    if current_bar.get("trade_date"):
        bars_by_date[str(current_bar["trade_date"])] = current_bar
    return sorted(bars_by_date.values(), key=lambda item: str(item.get("trade_date") or ""))


def _normalize_daily_bar(raw_bar: dict[str, object]) -> dict[str, object]:
    return {
        "trade_date": _normalize_trade_date(raw_bar.get("trade_date")),
        "open": _safe_float(raw_bar.get("open")),
        "high": _safe_float(raw_bar.get("high")),
        "low": _safe_float(raw_bar.get("low")),
        "close": _safe_float(raw_bar.get("close")),
        "pct_chg": _safe_float(raw_bar.get("pct_chg")),
        "amount": _safe_float(raw_bar.get("amount")),
        "turnover_rate": _safe_float(raw_bar.get("turnover_rate")),
        "vol_ratio": _safe_float(raw_bar.get("vol_ratio")),
    }


def _normalize_trade_date(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if len(text) == 8 and text.isdigit() else None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_optional(values: List[Optional[float]]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    return min(present_values) if present_values else None


def _max_optional(values: List[Optional[float]]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    return max(present_values) if present_values else None


def _calculate_distance_pct(close: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if close is None or baseline in (None, 0):
        return None
    return (close - baseline) / baseline * 100


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
