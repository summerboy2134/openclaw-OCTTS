from __future__ import annotations

import json
from pathlib import Path

from octts.schemas.report import (
    DecisionValidation,
    HistoricalAnalysisRecord,
    PriceSnapshot,
    TradingDecision,
    ValidationUpdate,
)

TERMINAL_STATUSES = {"take_profit_hit", "stop_loss_hit", "expired", "no_signal"}


class FileHistoryStore:
    def __init__(self, directory_path: str, limit_per_symbol: int = 30) -> None:
        self._directory = self._normalize_directory_path(directory_path)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._limit_per_symbol = limit_per_symbol
        self._migrate_legacy_payload_if_needed(directory_path)

    def append(self, record: HistoricalAnalysisRecord) -> None:
        existing = [HistoricalAnalysisRecord.model_validate(item) for item in self._load_symbol(record.report.ts_code)]
        replacement_index = None
        for index in range(len(existing) - 1, -1, -1):
            if _same_analysis_slot(existing[index], record):
                replacement_index = index
                break

        if replacement_index is None:
            existing.append(record)
        else:
            existing[replacement_index] = record

        self._save_symbol(
            record.report.ts_code,
            [item.model_dump(mode="json") for item in existing[-self._limit_per_symbol :]],
        )

    def list_records(self, ts_code: str, limit: int | None = None) -> list[HistoricalAnalysisRecord]:
        records = self._load_symbol(ts_code)
        if limit is not None:
            records = records[-limit:]
        return [HistoricalAnalysisRecord.model_validate(item) for item in records]

    def list_latest(self) -> list[HistoricalAnalysisRecord]:
        latest: list[HistoricalAnalysisRecord] = []
        for path in self._directory.glob("*.json"):
            records = self._load_path(path)
            if records:
                latest.append(HistoricalAnalysisRecord.model_validate(records[-1]))
        latest.sort(key=lambda item: item.generated_at, reverse=True)
        return latest

    def refresh_validations(self, *, ts_code: str, snapshot: PriceSnapshot) -> list[ValidationUpdate]:
        records = [HistoricalAnalysisRecord.model_validate(item) for item in self._load_symbol(ts_code)]
        updates: list[ValidationUpdate] = []

        for record in records:
            previous_status = record.validation.status
            if previous_status in TERMINAL_STATUSES:
                continue
            new_validation = evaluate_decision_validation(
                decision=record.report.decision,
                current_snapshot=snapshot,
                previous_validation=record.validation,
                generated_at=record.generated_at,
            )
            if new_validation.status != previous_status or new_validation.note != record.validation.note:
                updates.append(
                    ValidationUpdate(
                        record_id=record.record_id,
                        ts_code=ts_code,
                        previous_status=previous_status,
                        current_status=new_validation.status,
                        note=new_validation.note,
                        checked_at=new_validation.checked_at,
                    )
                )
                record.validation = new_validation

        self._save_symbol(ts_code, [record.model_dump(mode="json") for record in records])
        return updates

    def delete_symbol(self, ts_code: str) -> None:
        path = self._symbol_path(ts_code)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        for path in self._directory.glob("*.json"):
            path.unlink()

    def _load_symbol(self, ts_code: str) -> list[dict[str, object]]:
        return self._load_path(self._symbol_path(ts_code))

    def _load_path(self, path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        payload = json.loads(content)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            legacy_records = payload.get(path.stem)
            if isinstance(legacy_records, list):
                return legacy_records
        return []

    def _save_symbol(self, ts_code: str, payload: list[dict[str, object]]) -> None:
        self._symbol_path(ts_code).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _symbol_path(self, ts_code: str) -> Path:
        safe_code = ts_code.replace("/", "_").replace("\\", "_")
        return self._directory / f"{safe_code}.json"

    def _normalize_directory_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.suffix == ".json":
            return path.with_suffix("")
        return path

    def _migrate_legacy_payload_if_needed(self, raw_path: str) -> None:
        legacy_path = Path(raw_path)
        if legacy_path.suffix != ".json" or not legacy_path.exists():
            return
        content = legacy_path.read_text(encoding="utf-8").strip()
        if not content:
            return
        payload = json.loads(content)
        if not isinstance(payload, dict):
            return
        for ts_code, records in payload.items():
            if not isinstance(ts_code, str) or not isinstance(records, list):
                continue
            self._save_symbol(ts_code, records[-self._limit_per_symbol :])


def evaluate_decision_validation(
    *,
    decision: TradingDecision,
    current_snapshot: PriceSnapshot,
    previous_validation: DecisionValidation | None,
    generated_at,
) -> DecisionValidation:
    close = current_snapshot.close
    high = current_snapshot.high or close
    low = current_snapshot.low or close
    stop_loss = decision.stop_loss
    first_target = decision.take_profit[0] if decision.take_profit else None
    entry_triggered = _entry_triggered(decision, current_snapshot)

    if decision.signal == "avoid":
        return DecisionValidation(
            status="no_signal",
            note="该建议为规避信号，不参与命中率统计。",
            current_close=close,
            current_high=high,
            current_low=low,
        )

    if _is_expired(decision.holding_horizon, generated_at, current_snapshot.trade_date):
        return DecisionValidation(
            status="expired",
            note="已超过建议持有周期，标记为过期。",
            entry_triggered=entry_triggered,
            current_close=close,
            current_high=high,
            current_low=low,
        )

    is_bearish = decision.signal in {"reduce", "sell"}
    if stop_loss is not None:
        if not is_bearish and low is not None and low <= stop_loss:
            return DecisionValidation(
                status="stop_loss_hit",
                note=f"价格触及止损位 {stop_loss}，建议失效。",
                entry_triggered=entry_triggered,
                stop_loss_hit=True,
                current_close=close,
                current_high=high,
                current_low=low,
            )
        if is_bearish and high is not None and high >= stop_loss:
            return DecisionValidation(
                status="stop_loss_hit",
                note=f"反向波动触及止损位 {stop_loss}，看空建议失效。",
                entry_triggered=entry_triggered,
                stop_loss_hit=True,
                current_close=close,
                current_high=high,
                current_low=low,
            )

    if first_target is not None:
        if not is_bearish and high is not None and high >= first_target:
            return DecisionValidation(
                status="take_profit_hit",
                note=f"价格触及第一止盈位 {first_target}。",
                entry_triggered=entry_triggered,
                target_hit_level=first_target,
                current_close=close,
                current_high=high,
                current_low=low,
            )
        if is_bearish and low is not None and low <= first_target:
            return DecisionValidation(
                status="take_profit_hit",
                note=f"价格达到空头目标位 {first_target}。",
                entry_triggered=entry_triggered,
                target_hit_level=first_target,
                current_close=close,
                current_high=high,
                current_low=low,
            )

    if entry_triggered or (previous_validation and previous_validation.entry_triggered):
        return DecisionValidation(
            status="entered",
            note="价格进入建议区间，继续跟踪目标位与止损位。",
            entry_triggered=True,
            current_close=close,
            current_high=high,
            current_low=low,
        )

    return DecisionValidation(
        status="watching_entry",
        note="价格尚未进入建议区间，继续等待。",
        entry_triggered=False,
        current_close=close,
        current_high=high,
        current_low=low,
    )


def build_initial_validation(*, decision: TradingDecision, snapshot: PriceSnapshot) -> DecisionValidation:
    if decision.signal == "avoid":
        return DecisionValidation(
            status="no_signal",
            note="该建议为规避信号，不参与命中率统计。",
            current_close=snapshot.close,
            current_high=snapshot.high,
            current_low=snapshot.low,
        )
    if _entry_triggered(decision, snapshot):
        return DecisionValidation(
            status="entered",
            note="建议生成时价格已处于入场区间，后续继续跟踪止盈止损。",
            entry_triggered=True,
            current_close=snapshot.close,
            current_high=snapshot.high,
            current_low=snapshot.low,
        )
    return DecisionValidation(
        status="watching_entry",
        note="建议已生成，等待价格进入入场区间。",
        entry_triggered=False,
        current_close=snapshot.close,
        current_high=snapshot.high,
        current_low=snapshot.low,
    )


def _entry_triggered(decision: TradingDecision, snapshot: PriceSnapshot) -> bool:
    zone = decision.entry_zone
    if zone is None:
        return False

    close = snapshot.close
    high = snapshot.high or close
    low = snapshot.low or close
    lower = zone.low if zone.low is not None else zone.high
    upper = zone.high if zone.high is not None else zone.low

    if lower is None and upper is None:
        return False
    if lower is None:
        lower = upper
    if upper is None:
        upper = lower
    if low is None or high is None:
        return close is not None and lower <= close <= upper
    return low <= upper and high >= lower


def _is_expired(holding_horizon: str, generated_at, trade_date: str | None) -> bool:
    if not trade_date:
        return False

    generated_day = generated_at.strftime("%Y%m%d")
    if holding_horizon == "intraday":
        return trade_date > generated_day
    if holding_horizon == "swing":
        return _day_diff(generated_day, trade_date) > 5
    return _day_diff(generated_day, trade_date) > 20


def _day_diff(start_day: str, end_day: str) -> int:
    from datetime import datetime

    start = datetime.strptime(start_day, "%Y%m%d")
    end = datetime.strptime(end_day, "%Y%m%d")
    return (end - start).days


def _same_analysis_slot(existing: HistoricalAnalysisRecord, incoming: HistoricalAnalysisRecord) -> bool:
    return (
        existing.report.ts_code == incoming.report.ts_code
        and existing.report.phase == incoming.report.phase
        and _record_identity_day(existing) == _record_identity_day(incoming)
    )


def _record_identity_day(record: HistoricalAnalysisRecord) -> str:
    if record.snapshot.trade_date:
        return record.snapshot.trade_date
    return record.generated_at.strftime("%Y%m%d")
