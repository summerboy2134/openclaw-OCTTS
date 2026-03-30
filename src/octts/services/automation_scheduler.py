from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from octts.config import Settings
from octts.schemas.report import AnalysisPhase, AnalysisRequest
from octts.services.analysis_pipeline import AnalysisPipeline

if TYPE_CHECKING:
    from octts.services.report_email_service import ReportEmailService

logger = logging.getLogger(__name__)


def create_automation_scheduler(
    *,
    settings: Settings,
    pipeline_factory: Callable[[], AnalysisPipeline],
    report_email_service_factory: Optional[Callable[[], "ReportEmailService"]] = None,
    screening_scheduler_factory: Optional[Callable[[], Any]] = None,
) -> Optional[BackgroundScheduler]:
    if not settings.automation_enabled and not settings.email_enabled and not settings.screening_enabled:
        return None

    timezone = ZoneInfo(settings.automation_timezone)
    scheduler = BackgroundScheduler(timezone=timezone)

    # 原有的分析任务
    if settings.automation_enabled:
        for slot in build_automation_slots(settings):
            hour, minute = _parse_hour_minute(slot["time"])
            scheduler.add_job(
                _run_scheduled_analysis,
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=hour,
                    minute=minute,
                    timezone=timezone,
                ),
                id=f"octts-{slot['phase']}",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=900,
                kwargs={
                    "phase": slot["phase"],
                    "notify": settings.automation_notify,
                    "pipeline_factory": pipeline_factory,
                },
            )

    # 添加选股任务
    if settings.screening_enabled and screening_scheduler_factory:
        hour, minute = _parse_hour_minute(settings.screening_time)
        scheduler.add_job(
            _run_scheduled_screening,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=hour,
                minute=minute,
                timezone=timezone,
            ),
            id="octts-screening",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=900,
            kwargs={
                "screening_scheduler_factory": screening_scheduler_factory,
            },
        )

    if settings.email_enabled:
        if not settings.email_send_time:
            raise ValueError("OCTTS_EMAIL_SEND_TIME is required when email is enabled.")
        if report_email_service_factory is None:
            raise ValueError("report_email_service_factory is required when email is enabled.")
        hour, minute = _parse_hour_minute(settings.email_send_time)
        scheduler.add_job(
            _run_scheduled_email,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=hour,
                minute=minute,
                timezone=timezone,
            ),
            id="octts-email-report",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=900,
            kwargs={
                "report_email_service_factory": report_email_service_factory,
            },
        )
    return scheduler


def build_automation_slots(settings: Settings) -> list[dict[str, str]]:
    slots = [
        {"phase": "morning", "time": settings.automation_morning_time, "label": "早盘分析"},
        {"phase": "afternoon", "time": settings.automation_afternoon_time, "label": "尾盘分析"},
        {"phase": "review", "time": settings.automation_review_time, "label": "复盘总结"},
    ]
    enabled_phases = set(settings.automation_phases)
    return [slot for slot in slots if slot["phase"] in enabled_phases]


def _parse_hour_minute(raw_value: str) -> tuple[int, int]:
    hour_text, minute_text = raw_value.split(":", maxsplit=1)
    hour = int(hour_text)
    minute = int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid automation time: {raw_value}")
    return hour, minute


def _run_scheduled_analysis(
    *,
    phase: AnalysisPhase,
    notify: bool,
    pipeline_factory: Callable[[], AnalysisPipeline],
) -> None:
    try:
        pipeline_factory().run(
            AnalysisRequest(
                phase=phase,
                notify=notify,
            )
        )
        logger.info("Scheduled OCTTS analysis completed", extra={"phase": phase})
    except Exception:
        logger.exception("Scheduled OCTTS analysis failed", extra={"phase": phase})


def _run_scheduled_email(
    *,
    report_email_service_factory: Callable[[], "ReportEmailService"],
) -> None:
    try:
        report_email_service_factory().send_latest_report_email()
        logger.info("Scheduled OCTTS report email completed")
    except Exception:
        logger.exception("Scheduled OCTTS report email failed")


def _run_scheduled_screening(
    *,
    screening_scheduler_factory: Callable[[], Any],
) -> None:
    """运行定时选股任务"""
    try:
        import asyncio
        scheduler = screening_scheduler_factory()
        run_method = getattr(scheduler, "run_intelligent_screening", None)
        if run_method is None:
            run_method = scheduler.run_daily_screening
        # 如果是异步方法，需要在事件循环中运行
        if asyncio.iscoroutinefunction(run_method):
            asyncio.run(run_method())
        else:
            run_method()
        logger.info("Scheduled stock screening completed")
    except Exception:
        logger.exception("Scheduled stock screening failed")
