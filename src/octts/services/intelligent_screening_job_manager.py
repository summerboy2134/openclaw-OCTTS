"""Background job tracking for intelligent screening."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], Optional[Awaitable[None]]]
JobRunner = Callable[[ProgressCallback], Awaitable[Dict[str, Any]]]


@dataclass
class IntelligentScreeningJob:
    job_id: str
    status: str = "queued"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_percent: int = 0
    current_step: int = 0
    total_steps: int = 0
    step_name: str = ""
    message: str = "任务已创建，等待执行。"
    details: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "step_name": self.step_name,
            "message": self.message,
            "details": dict(self.details),
            "result": self.result,
            "error": self.error,
            "is_active": self.status in {"queued", "running"},
        }


class IntelligentScreeningJobManager:
    """Manage a single active intelligent screening background task."""

    _ACTIVE_STATUSES = {"queued", "running"}

    def __init__(self, history_dir_path: str, retention_seconds: int = 3600):
        self._jobs: Dict[str, IntelligentScreeningJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running_job_id: Optional[str] = None
        self._retention = timedelta(seconds=retention_seconds)
        self._lock = asyncio.Lock()
        self._snapshot_dir = Path(history_dir_path) / "intelligent_screening_jobs"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._active_snapshot_path = self._snapshot_dir / "active.json"

    async def start_job(self, runner: JobRunner) -> Dict[str, Any]:
        async with self._lock:
            self._cleanup_finished_jobs_locked()
            active_snapshot = self._read_active_snapshot()
            if self._running_job_id:
                existing_job = self._jobs.get(self._running_job_id)
                if existing_job and existing_job.status in self._ACTIVE_STATUSES:
                    return {"created": False, "job": existing_job.to_dict()}
                self._running_job_id = None

            if active_snapshot is not None:
                snapshot_job_id = str(active_snapshot.get("job_id") or "")
                snapshot_job = self._jobs.get(snapshot_job_id) if snapshot_job_id else None
                if snapshot_job and snapshot_job.status in self._ACTIVE_STATUSES:
                    self._running_job_id = snapshot_job_id
                    return {"created": False, "job": snapshot_job.to_dict()}
                self._mark_snapshot_job_stale_locked(active_snapshot)

            job = IntelligentScreeningJob(job_id=uuid.uuid4().hex)
            self._jobs[job.job_id] = job
            self._running_job_id = job.job_id
            self._persist_job_snapshot_locked(job)
            self._tasks[job.job_id] = asyncio.create_task(self._execute_job(job.job_id, runner))
            return {"created": True, "job": job.to_dict()}

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            self._cleanup_finished_jobs_locked()
            job = self._jobs.get(job_id)
            if job:
                return job.to_dict()

        return self._read_job_snapshot(job_id)

    async def get_active_job(self) -> Optional[Dict[str, Any]]:
        async with self._lock:
            self._cleanup_finished_jobs_locked()
            if self._running_job_id:
                job = self._jobs.get(self._running_job_id)
                if job is not None and job.status in self._ACTIVE_STATUSES:
                    return job.to_dict()

        payload = self._read_active_snapshot()
        if payload and payload.get("status") in self._ACTIVE_STATUSES:
            return payload
        return None

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
            self._running_job_id = None

        for task in tasks:
            if not task.done():
                task.cancel()

        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Unhandled error while shutting down intelligent screening task")

    async def _execute_job(self, job_id: str, runner: JobRunner) -> None:
        await self._update_job(
            job_id,
            status="running",
            started_at=datetime.utcnow(),
            progress_percent=1,
            message="后台任务已启动。",
        )
        try:
            result = await runner(lambda payload: self.update_progress(job_id, payload))
            final_message = self._build_success_message(result)
            await self._update_job(
                job_id,
                status="succeeded",
                finished_at=datetime.utcnow(),
                progress_percent=100,
                current_step=max(int(result.get("current_step", 0) or 0), 0),
                total_steps=max(int(result.get("total_steps", 0) or 0), 0),
                message=final_message,
                result=result,
                details={},
                clear_running=True,
            )
        except asyncio.CancelledError:
            await self._update_job(
                job_id,
                status="failed",
                finished_at=datetime.utcnow(),
                message="后台任务已取消。",
                error="任务已取消",
                clear_running=True,
            )
            raise
        except Exception as exc:
            logger.exception("Intelligent screening background task failed")
            await self._update_job(
                job_id,
                status="failed",
                finished_at=datetime.utcnow(),
                message="智能选股后台任务执行失败。",
                error=str(exc),
                clear_running=True,
            )

    async def update_progress(self, job_id: str, payload: Dict[str, Any]) -> None:
        updates = dict(payload)
        if "progress_percent" in updates:
            try:
                updates["progress_percent"] = max(0, min(100, int(updates["progress_percent"])))
            except (TypeError, ValueError):
                updates["progress_percent"] = 0
        await self._update_job(job_id, **updates)

    async def _update_job(self, job_id: str, clear_running: bool = False, **updates: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            for key, value in updates.items():
                if key == "details" and isinstance(value, dict):
                    job.details = dict(value)
                    continue
                setattr(job, key, value)

            if clear_running and self._running_job_id == job_id:
                self._running_job_id = None

            self._persist_job_snapshot_locked(job)

    def _cleanup_finished_jobs_locked(self) -> None:
        now = datetime.utcnow()
        expired_job_ids = []
        for job_id, job in self._jobs.items():
            if job.status in {"queued", "running"}:
                continue
            finished_at = job.finished_at or job.created_at
            if now - finished_at > self._retention:
                expired_job_ids.append(job_id)

        for job_id in expired_job_ids:
            self._jobs.pop(job_id, None)
            task = self._tasks.pop(job_id, None)
            if task and not task.done():
                task.cancel()
            self._remove_job_snapshot(job_id)

    def _persist_job_snapshot_locked(self, job: IntelligentScreeningJob) -> None:
        payload = job.to_dict()
        job_snapshot_path = self._snapshot_dir / f"{job.job_id}.json"
        self._write_snapshot(job_snapshot_path, payload)
        if job.status in {"queued", "running"}:
            self._write_snapshot(self._active_snapshot_path, payload)
            return
        if self._active_snapshot_path.exists():
            active_payload = self._read_snapshot(self._active_snapshot_path)
            if active_payload and active_payload.get("job_id") == job.job_id:
                try:
                    self._active_snapshot_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Failed to remove intelligent screening active snapshot: %s", self._active_snapshot_path)

    def _read_active_snapshot(self) -> Optional[Dict[str, Any]]:
        payload = self._read_snapshot(self._active_snapshot_path)
        if payload and payload.get("status") in self._ACTIVE_STATUSES:
            return payload

        latest_active_payload = None
        latest_created_at = ""
        for snapshot_path in sorted(self._snapshot_dir.glob("*.json"), reverse=True):
            if snapshot_path == self._active_snapshot_path:
                continue
            candidate = self._read_snapshot(snapshot_path)
            if not candidate or candidate.get("status") not in self._ACTIVE_STATUSES:
                continue
            created_at = str(candidate.get("created_at") or "")
            if created_at >= latest_created_at:
                latest_created_at = created_at
                latest_active_payload = candidate

        return latest_active_payload

    def _read_job_snapshot(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._read_snapshot(self._snapshot_dir / f"{job_id}.json")

    def _mark_snapshot_job_stale_locked(self, payload: Dict[str, Any]) -> None:
        job_id = str(payload.get("job_id") or "")
        if not job_id:
            return
        stale_payload = dict(payload)
        stale_payload["status"] = "failed"
        stale_payload["finished_at"] = datetime.utcnow().isoformat()
        stale_payload["error"] = stale_payload.get("error") or "任务状态已失效"
        stale_payload["message"] = "检测到陈旧任务快照，已标记为失效。"
        stale_payload["is_active"] = False
        self._write_snapshot(self._snapshot_dir / f"{job_id}.json", stale_payload)
        if self._active_snapshot_path.exists():
            try:
                self._active_snapshot_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Failed to remove stale intelligent screening active snapshot: %s", self._active_snapshot_path)

    def _remove_job_snapshot(self, job_id: str) -> None:
        snapshot_path = self._snapshot_dir / f"{job_id}.json"
        try:
            snapshot_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove intelligent screening job snapshot: %s", snapshot_path)

    @staticmethod
    def _read_snapshot(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read intelligent screening job snapshot: %s", path)
            return None

    @staticmethod
    def _write_snapshot(path: Path, payload: Dict[str, Any]) -> None:
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            logger.warning("Failed to write intelligent screening job snapshot: %s", path)

    @staticmethod
    def _build_success_message(result: Dict[str, Any]) -> str:
        frontlist_count = int(result.get("frontlist_count", result.get("final_recommendations", 0)) or 0)
        tracking_pool_count = int(result.get("tracking_pool_count", 0) or 0)
        return "智能选股完成：前台推荐 {0} 只，跟踪池 {1} 只。".format(frontlist_count, tracking_pool_count)


async def maybe_await_progress_callback(
    callback: Optional[Callable[[Dict[str, Any]], Any]],
    payload: Dict[str, Any],
) -> None:
    """Invoke sync or async progress callbacks safely."""
    if callback is None:
        return

    result = callback(payload)
    if inspect.isawaitable(result):
        await result
