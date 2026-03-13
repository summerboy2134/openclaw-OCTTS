from datetime import UTC, datetime

from octts.schemas.report import (
    DecisionValidation,
    HistoricalAnalysisRecord,
    MemorySummary,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
)
from octts.services.history_store import FileHistoryStore, build_initial_validation


def _build_record() -> HistoricalAnalysisRecord:
    snapshot = PriceSnapshot(
        ts_code="600000.SH",
        trade_date="20260309",
        close=10.2,
        high=10.3,
        low=10.0,
    )
    report = StructuredAnalysis(
        ts_code="600000.SH",
        phase="review",
        trend_judgement="等待向上突破",
        previous_view_status="initial",
        operation_advice="靠近支撑时分批关注",
        risk_warning=["若跌破 9.8 需止损"],
        observation_points=["关注 10.5 压力位"],
        summary_markdown="等待突破。",
        decision=TradingDecision(
            signal="buy",
            rationale="支撑有效且量能温和修复。",
            entry_zone=PriceZone(low=10.0, high=10.2),
            stop_loss=9.8,
            take_profit=[10.5],
            invalidation_condition="放量跌破 9.8",
            holding_horizon="swing",
            confidence_score=0.72,
            risk_reward_ratio=1.8,
            evidence=["支撑位有效"],
        ),
        memory=MemorySummary(
            ts_code="600000.SH",
            phase="review",
            trend_bias="bullish",
            capital_flow_view="主力资金小幅回流",
            confidence_score=0.72,
            summary="等待突破确认",
        ),
    )
    return HistoricalAnalysisRecord(
        record_id="r1",
        request_id="req1",
        generated_at=datetime(2026, 3, 9, tzinfo=UTC),
        snapshot=snapshot,
        report=report,
        validation=build_initial_validation(decision=report.decision, snapshot=snapshot),
    )


def test_history_store_updates_validation_to_take_profit(tmp_path) -> None:
    store = FileHistoryStore(str(tmp_path / "history.json"))
    record = _build_record()
    store.append(record)

    updates = store.refresh_validations(
        ts_code="600000.SH",
        snapshot=PriceSnapshot(
            ts_code="600000.SH",
            trade_date="20260310",
            close=10.45,
            high=10.6,
            low=10.1,
        ),
    )

    latest = store.list_records("600000.SH")[-1]
    assert updates
    assert latest.validation.status == "take_profit_hit"


def test_history_store_initial_validation_marks_entry_state() -> None:
    record = _build_record()
    assert record.validation.status == "entered"


def test_history_store_writes_one_file_per_symbol(tmp_path) -> None:
    store = FileHistoryStore(str(tmp_path / "history"))
    store.append(_build_record())

    symbol_file = tmp_path / "history" / "600000.SH.json"
    assert symbol_file.exists()
    assert store.list_records("600000.SH")


def test_history_store_overwrites_same_day_same_phase(tmp_path) -> None:
    store = FileHistoryStore(str(tmp_path / "history"))
    first = _build_record()
    second = _build_record()
    second.record_id = "r2"
    second.report.trend_judgement = "已切换为更强的向上趋势"

    store.append(first)
    store.append(second)

    records = store.list_records("600000.SH")
    assert len(records) == 1
    assert records[0].record_id == "r2"
    assert records[0].report.trend_judgement == "已切换为更强的向上趋势"


def test_history_store_delete_symbol_and_clear(tmp_path) -> None:
    store = FileHistoryStore(str(tmp_path / "history"))
    first = _build_record()
    second = _build_record()
    second.report.ts_code = "000001.SZ"
    second.snapshot.ts_code = "000001.SZ"
    second.report.memory.ts_code = "000001.SZ"

    store.append(first)
    store.append(second)
    store.delete_symbol("600000.SH")

    assert store.list_records("600000.SH") == []
    assert store.list_records("000001.SZ")

    store.clear()
    assert store.list_latest() == []
