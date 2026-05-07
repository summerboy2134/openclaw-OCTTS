from __future__ import annotations

from typing import Any, Dict, List

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
        recommended_top3 = self._load_intelligent_recommended_top3()
        combined_codes = self._merge_stock_codes(default_stock_pool, [item.get("ts_code") for item in recommended_top3])
        allowed_codes = {item.strip().upper() for item in combined_codes if item and item.strip()}
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
        attachments = []
        archive_name, archive_bytes = self._report_exporter.export_combined_latest_report_zip(combined_codes)
        attachments.append((archive_name, archive_bytes, "application/zip"))
        summary = format_reports_as_markdown(reports)
        top3_summary = self._format_recommended_top3_summary(recommended_top3)
        body = (
            f"OCTTS 最新离线报告已生成。\n"
            f"生成时间: {generated_at}\n"
            f"默认股票池数量: {len(default_stock_pool)}\n"
            f"智能推荐股票数量: {len(recommended_top3)}\n"
            f"本次邮件股票数量: {len(reports)}\n\n"
            f"{top3_summary}\n"
            f"{summary}\n"
        )
        self._email_client.send_message(
            subject=subject,
            body=body,
            recipients=self._settings.email_recipients,
            attachments=attachments,
        )

    def _load_intelligent_recommended_top3(self) -> List[Dict[str, Any]]:
        try:
            payload = self._report_exporter.build_intelligent_screening_payload()
        except Exception:
            return []

        report_context = payload.get("report_context") if isinstance(payload.get("report_context"), dict) else {}
        recommendation_pool = payload.get("recommendation_pool") if isinstance(payload.get("recommendation_pool"), dict) else {}
        candidates = report_context.get("today_top3") if isinstance(report_context.get("today_top3"), list) else []
        if not candidates:
            candidates = recommendation_pool.get("today_top") if isinstance(recommendation_pool.get("today_top"), list) else []
        if not candidates:
            candidates = recommendation_pool.get("frontlist") if isinstance(recommendation_pool.get("frontlist"), list) else []

        normalized: List[Dict[str, Any]] = []
        seen_codes = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            code = str(item.get("ts_code") or "").strip().upper()
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            normalized.append(item)
            if len(normalized) >= 3:
                break
        return normalized

    @staticmethod
    def _merge_stock_codes(default_stock_pool: List[str], recommended_codes: List[Any]) -> List[str]:
        merged: List[str] = []
        seen_codes = set()
        for code in list(default_stock_pool or []) + [str(item or "") for item in recommended_codes or []]:
            normalized = str(code or "").strip().upper()
            if not normalized or normalized in seen_codes:
                continue
            seen_codes.add(normalized)
            merged.append(normalized)
        return merged

    @staticmethod
    def _format_recommended_top3_summary(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "智能选股推荐Top3: 暂无最新数据\n"

        lines = ["智能选股推荐Top3:"]
        for index, item in enumerate(items, start=1):
            code = str(item.get("ts_code") or "--").strip()
            name = str(item.get("name") or code).strip()
            score = item.get("recommendation_score", item.get("overall_score", item.get("priority_score")))
            score_text = ReportEmailService._format_score(score)
            reason = str(
                item.get("selection_reason")
                or item.get("recommendation_text")
                or item.get("overview_reason")
                or item.get("summary")
                or ""
            ).strip()
            if len(reason) > 120:
                reason = reason[:117] + "..."
            suffix = f" - {reason}" if reason else ""
            lines.append(f"{index}. {code} {name} 推荐分: {score_text}{suffix}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_score(value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "--"

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
