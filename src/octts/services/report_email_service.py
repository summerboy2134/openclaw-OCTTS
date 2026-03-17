from __future__ import annotations

from octts.clients.email_client import EmailClient
from octts.config import Settings
from octts.services.analysis_pipeline import format_reports_as_markdown
from octts.services.history_store import FileHistoryStore
from octts.services.report_exporter import ReportExporter


class ReportEmailService:
    def __init__(
        self,
        *,
        settings: Settings,
        history_store: FileHistoryStore,
        report_exporter: ReportExporter,
        email_client: EmailClient,
    ) -> None:
        self._settings = settings
        self._history_store = history_store
        self._report_exporter = report_exporter
        self._email_client = email_client

    def send_latest_report_email(self) -> None:
        latest_records = self._history_store.list_latest()
        if not latest_records:
            raise ValueError("No analysis history available for email delivery.")
        if not self._settings.email_recipients:
            raise ValueError("OCTTS_EMAIL_RECIPIENTS is required when email is enabled.")

        generated_at = latest_records[0].generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        reports = [record.report for record in latest_records]
        subject = f"{self._settings.email_subject_prefix} 自动报告 {latest_records[0].generated_at.strftime('%Y-%m-%d %H:%M')}"
        archive_name, archive_bytes = self._report_exporter.export_latest_report_zip()
        summary = format_reports_as_markdown(reports)
        body = (
            f"OCTTS 最新离线报告已生成。\n"
            f"生成时间: {generated_at}\n"
            f"股票数量: {len(reports)}\n\n"
            f"{summary}\n"
        )
        self._email_client.send_message(
            subject=subject,
            body=body,
            recipients=self._settings.email_recipients,
            attachments=[(archive_name, archive_bytes, "application/zip")],
        )
