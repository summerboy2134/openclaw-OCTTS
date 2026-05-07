from __future__ import annotations

from io import BytesIO
from typing import Any, Optional
from zipfile import ZIP_DEFLATED, ZipFile

from octts.config import Settings
from octts.schemas.report import HistoricalAnalysisRecord
from octts.services.history_store import FileHistoryStore
from octts.services.intelligent_dashboard_payload import (
    build_recommendation_methodology_payload,
    build_stock_intelligent_insight,
    load_intelligent_dashboard_payload,
    load_recommendation_summary,
)
from octts.services.position_store import FilePositionStore
from octts.services.report_payload import build_openclaw_status, build_validation_summary, serialize_record
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
            serialize_record(
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
            "validation_summary": build_validation_summary(latest_records),
            "default_stock_pool": self._settings.stock_pool,
            "openclaw_status": build_openclaw_status(self._settings),
        }

    def build_stock_detail_payload(self, ts_code: str) -> dict[str, object]:
        records = self._history_store.list_records(ts_code, limit=self._settings.history_limit_per_symbol)
        if not records:
            raise ValueError(f"No history found for {ts_code}")

        latest = records[-1]
        intelligent_payload = load_intelligent_dashboard_payload(self._settings)
        return {
            "generated_at": latest.generated_at.isoformat(),
            "symbol": serialize_record(
                latest,
                self._history_store,
                self._position_store,
                history_limit=self._settings.history_limit_per_symbol,
            ),
            "validation_summary": build_validation_summary(records),
            "openclaw_status": build_openclaw_status(self._settings),
            "position_status": self._position_store.get_status(ts_code),
            "intelligent_screening_insight": build_stock_intelligent_insight(ts_code, intelligent_payload),
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
            self._write_dashboard_archive_files(
                archive=archive,
                latest_records=latest_records,
                dashboard_payload=dashboard_payload,
                index_path="index.html",
                stock_dir="stocks",
                stock_back_href="../index.html",
            )

        return archive_name, buffer.getvalue()

    def export_combined_latest_report_zip(self, ts_codes: Optional[list[str]] = None) -> tuple[str, bytes]:
        latest_records = self._filter_latest_records(self._history_store.list_latest(), ts_codes)
        if not latest_records:
            raise ValueError("No analysis history available for export.")

        dashboard_payload = self.build_dashboard_payload(ts_codes)
        intelligent_payload = self.build_intelligent_screening_payload()
        latest_generated_at = latest_records[0].generated_at.strftime("%Y%m%d-%H%M%S")
        archive_name = f"octts-combined-report-{latest_generated_at}.zip"
        buffer = BytesIO()

        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            self._write_dashboard_archive_files(
                archive=archive,
                latest_records=latest_records,
                dashboard_payload=dashboard_payload,
                index_path="index.html",
                stock_dir="stocks",
                stock_back_href="../index.html",
                intelligent_screening_href="./intelligent-screening/index.html",
            )
            self._write_intelligent_screening_archive_files(
                archive=archive,
                payload=intelligent_payload,
                latest_records=latest_records,
                base_dir="intelligent-screening",
                dashboard_href="../index.html",
                include_dashboard_page=False,
                stock_dir="stocks",
            )

        return archive_name, buffer.getvalue()

    def build_intelligent_screening_payload(self, trade_date: Optional[str] = None) -> dict[str, Any]:
        payload = load_intelligent_dashboard_payload(self._settings, trade_date=trade_date)
        payload["recommendation_summary"] = load_recommendation_summary(self._settings)
        payload["recommendation_methodology"] = build_recommendation_methodology_payload(self._settings)
        return payload

    def export_latest_intelligent_screening_zip(self) -> tuple[str, bytes]:
        payload = self.build_intelligent_screening_payload()
        generated_at = str(payload.get("generated_at") or "latest").replace(":", "").replace("T", "-")
        archive_name = f"octts-intelligent-screening-{generated_at}.zip"
        buffer = BytesIO()

        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            self._write_intelligent_screening_archive_files(
                archive=archive,
                payload=payload,
                latest_records=self._history_store.list_latest(),
                base_dir="",
                dashboard_href="./dashboard.html",
                include_dashboard_page=True,
                stock_dir="stocks",
            )

        return archive_name, buffer.getvalue()

    def _write_dashboard_archive_files(
        self,
        *,
        archive: ZipFile,
        latest_records: list[HistoricalAnalysisRecord],
        dashboard_payload: dict[str, object],
        index_path: str,
        stock_dir: str,
        stock_back_href: str,
        intelligent_screening_href: Optional[str] = None,
    ) -> None:
        archive.writestr(
            index_path,
            render_dashboard_html(
                dashboard_payload,
                stock_detail_href_prefix=f"./{stock_dir}/",
                stock_detail_href_suffix=".html",
                interactive=False,
                intelligent_screening_href=intelligent_screening_href,
            ),
        )
        for record in latest_records:
            archive.writestr(
                f"{stock_dir}/{record.report.ts_code}.html",
                render_stock_detail_html(
                    record.report.ts_code,
                    self.build_stock_detail_payload(record.report.ts_code),
                    back_href=stock_back_href,
                    interactive=False,
                ),
            )

    def _write_intelligent_screening_archive_files(
        self,
        *,
        archive: ZipFile,
        payload: dict[str, Any],
        latest_records: list[HistoricalAnalysisRecord],
        base_dir: str,
        dashboard_href: str,
        include_dashboard_page: bool,
        stock_dir: str,
    ) -> None:
        base_prefix = f"{base_dir}/" if base_dir else ""
        tab_href_map = {
            "overview": "./index.html",
            "report": "./news.html",
            "focus": "./focus.html",
        }
        for active_tab, filename in (("overview", "index.html"), ("report", "news.html"), ("focus", "focus.html")):
            archive.writestr(
                f"{base_prefix}{filename}",
                render_intelligent_screening_dashboard(
                    screening_results=payload.get("screening_results") or {},
                    recommendation_pool=payload.get("recommendation_pool") or {},
                    ai_analyses=payload.get("ai_analyses") or {},
                    news_clusters=payload.get("news_clusters") or [],
                    intelligent_report=payload.get("intelligent_report") or {},
                    recommendation_summary=payload.get("recommendation_summary") or {},
                    recommendation_methodology=payload.get("recommendation_methodology") or {},
                    report_context=payload.get("report_context") or {},
                    generated_at=payload.get("generated_at"),
                    dashboard_href=dashboard_href,
                    backtest_href=None,
                    refresh_href="./index.html",
                    jobs_api_base="",
                    autorun_enabled=False,
                    active_tab=active_tab,
                    tab_href_map=tab_href_map,
                ),
            )

        if include_dashboard_page:
            archive.writestr(
                f"{base_prefix}dashboard.html",
                render_dashboard_html(
                    self.build_dashboard_payload(),
                    stock_detail_href_prefix=f"./{stock_dir}/",
                    stock_detail_href_suffix=".html",
                    interactive=False,
                ),
            )
            for record in latest_records:
                archive.writestr(
                    f"{base_prefix}{stock_dir}/{record.report.ts_code}.html",
                    render_stock_detail_html(
                        record.report.ts_code,
                        self.build_stock_detail_payload(record.report.ts_code),
                        back_href="../dashboard.html",
                        interactive=False,
                    ),
                )

    def _filter_latest_records(
        self, records: list[HistoricalAnalysisRecord], ts_codes: Optional[list[str]]
    ) -> list[HistoricalAnalysisRecord]:
        if not ts_codes:
            return records
        allowed_codes = {item.strip().upper() for item in ts_codes if item and item.strip()}
        if not allowed_codes:
            return []
        return [record for record in records if record.report.ts_code.upper() in allowed_codes]


