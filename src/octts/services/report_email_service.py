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
        default_stock_pool = self._settings.stock_pool
        allowed_codes = {item.strip().upper() for item in default_stock_pool if item and item.strip()}
        latest_records = [
            record for record in self._history_store.list_latest() if record.report.ts_code.upper() in allowed_codes
        ]
        if not latest_records:
            raise ValueError("No analysis history available for email delivery.")
        if not self._settings.email_recipients:
            raise ValueError("OCTTS_EMAIL_RECIPIENTS is required when email is enabled.")

        generated_at = latest_records[0].generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        reports = [record.report for record in latest_records]
        subject = f"{self._settings.email_subject_prefix} 自动报告 {latest_records[0].generated_at.strftime('%Y-%m-%d %H:%M')}"
        archive_name, archive_bytes = self._report_exporter.export_latest_report_zip(default_stock_pool)
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

    async def send_screening_report(self, report: dict) -> None:
        """
        发送选股报告邮件

        Args:
            report: 选股报告数据
        """
        if not self._settings.email_recipients:
            raise ValueError("OCTTS_EMAIL_RECIPIENTS is required when email is enabled.")

        from octts.services.screening_report import ScreeningReportService
        report_service = ScreeningReportService(self._settings)

        # 生成HTML格式的报告
        html_content = report_service.format_html_report(report)

        subject = f"{self._settings.email_subject_prefix} 选股报告 {report['report_date']}"

        body = (
            f"OCTTS 每日选股报告\n"
            f"生成时间: {report['report_time']}\n"
            f"运行策略: {report['strategy_count']}个\n"
            f"选出股票: {report['total_stocks']}只（去重后{report['unique_stocks']}只）\n\n"
            f"详细报告见邮件正文。"
        )

        # 发送邮件
        self._email_client.send_message(
            subject=subject,
            body=body,
            recipients=self._settings.email_recipients,
            attachments=None,
            html_content=html_content,
        )

    async def send_email(
        self,
        *,
        subject: str,
        html_content: str,
        body: str = "",
    ) -> None:
        """Send a generic HTML email through the configured SMTP client."""
        if not self._settings.email_recipients:
            raise ValueError("OCTTS_EMAIL_RECIPIENTS is required when email is enabled.")

        plain_body = body or "请查看 HTML 正文。"
        self._email_client.send_message(
            subject=subject,
            body=plain_body,
            recipients=self._settings.email_recipients,
            attachments=None,
            html_content=html_content,
        )

    async def send_intelligent_screening_email(self, *, subject: str, html_content: str, body: str = "") -> None:
        if not self._settings.email_recipients:
            raise ValueError("OCTTS_EMAIL_RECIPIENTS is required when email is enabled.")

        archive_name, archive_bytes = self._report_exporter.export_latest_intelligent_screening_zip()
        self._email_client.send_message(
            subject=subject,
            body=body or "请查看 HTML 正文与离线智能选股报告附件。",
            recipients=self._settings.email_recipients,
            attachments=[(archive_name, archive_bytes, "application/zip")],
            html_content=html_content,
        )
