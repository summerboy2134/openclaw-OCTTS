from __future__ import annotations

import logging
from collections.abc import Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from octts.config import Settings
from octts.schemas.report import AnalysisPhase, AnalysisRequest
from octts.services.analysis_pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


def create_automation_scheduler(
    *,
    settings: Settings,
    pipeline_factory: Callable[[], AnalysisPipeline],
) -> BackgroundScheduler | None:
    if not settings.automation_enabled:
        return None

    timezone = ZoneInfo(settings.automation_timezone)
    scheduler = BackgroundScheduler(timezone=timezone)
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
