from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from zipfile import ZipFile

from octts.config import Settings
from octts.schemas.report import (
    DecisionValidation,
    HistoricalAnalysisRecord,
    MemorySummary,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
)
from octts.services.history_store import FileHistoryStore
from octts.services.position_store import FilePositionStore
from octts.services.report_email_service import ReportEmailService
from octts.services.report_exporter import ReportExporter

UTC = timezone.utc


def _build_record(ts_code: str, generated_at: datetime) -> HistoricalAnalysisRecord:
    snapshot = PriceSnapshot(
        ts_code=ts_code,
        name=f"Name-{ts_code}",
        trade_date="20260318",
        close=10.2,
        high=10.5,
        low=9.9,
    )
    report = StructuredAnalysis(
        ts_code=ts_code,
        phase="review",
        trend_judgement=f"{ts_code} 趋势观察",
        previous_view_status="initial",
        operation_advice="观察关键位",
        risk_warning=["量能不足"],
        observation_points=["关注压力位"],
        summary_markdown=f"{ts_code} summary",
        decision=TradingDecision(
            signal="buy",
            rationale="形态仍可跟踪。",
            entry_zone=PriceZone(low=10.0, high=10.3),
            stop_loss=9.8,
            take_profit=[10.8],
            invalidation_condition="跌破止损位",
            holding_horizon="swing",
            confidence_score=0.7,
            risk_reward_ratio=1.6,
            evidence=["结构未破坏"],
        ),
        memory=MemorySummary(
            ts_code=ts_code,
            generated_at=generated_at,
            phase="review",
            trend_bias="bullish",
            capital_flow_view="资金中性偏强",
            confidence_score=0.7,
            summary="继续观察",
        ),
    )
    return HistoricalAnalysisRecord(
        record_id=f"record-{ts_code}",
        request_id="req-1",
        generated_at=generated_at,
        snapshot=snapshot,
        report=report,
        validation=DecisionValidation(status="watching_entry", note="等待触发入场。"),
    )


class CapturingEmailClient:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def send_message(self, *, subject, body, recipients, attachments) -> None:
        self.payload = {
            "subject": subject,
            "body": body,
            "recipients": recipients,
            "attachments": attachments,
        }


def test_send_latest_report_email_only_includes_default_stock_pool(tmp_path) -> None:
    settings = Settings(
        OCTTS_STOCK_POOL="600000.SH",
        OCTTS_EMAIL_RECIPIENTS="test@example.com",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    history_store = FileHistoryStore(str(tmp_path / "history"))
    position_store = FilePositionStore(str(tmp_path / "positions.json"))
    now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
    history_store.append(_build_record("600000.SH", now))
    history_store.append(_build_record("000001.SZ", now - timedelta(minutes=1)))
    exporter = ReportExporter(settings=settings, history_store=history_store, position_store=position_store)
    email_client = CapturingEmailClient()

    ReportEmailService(
        settings=settings,
        history_store=history_store,
        report_exporter=exporter,
        email_client=email_client,
    ).send_latest_report_email()

    assert email_client.payload is not None
    assert "股票数量: 1" in str(email_client.payload["body"])
    assert "600000.SH" in str(email_client.payload["body"])
    assert "000001.SZ" not in str(email_client.payload["body"])

    archive_name, archive_bytes, mime_type = email_client.payload["attachments"][0]
    assert archive_name.endswith(".zip")
    assert mime_type == "application/zip"
    with ZipFile(BytesIO(archive_bytes)) as archive:
        names = sorted(archive.namelist())
    assert names == ["index.html", "stocks/600000.SH.html"]


def test_export_latest_report_zip_filters_requested_stock_pool(tmp_path) -> None:
    settings = Settings(
        OCTTS_STOCK_POOL="600000.SH,000001.SZ",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    history_store = FileHistoryStore(str(tmp_path / "history"))
    position_store = FilePositionStore(str(tmp_path / "positions.json"))
    now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
    history_store.append(_build_record("600000.SH", now))
    history_store.append(_build_record("000001.SZ", now - timedelta(minutes=1)))

    archive_name, archive_bytes = ReportExporter(
        settings=settings,
        history_store=history_store,
        position_store=position_store,
    ).export_latest_report_zip(["000001.SZ"])

    assert archive_name.endswith(".zip")
    with ZipFile(BytesIO(archive_bytes)) as archive:
        names = sorted(archive.namelist())
    assert names == ["index.html", "stocks/000001.SZ.html"]
