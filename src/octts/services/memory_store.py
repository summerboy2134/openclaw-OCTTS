from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from octts.config import Settings
from octts.schemas.report import MemorySummary


class MemoryStore(Protocol):
    def get(self, ts_code: str) -> MemorySummary | None:
        ...

    def set(self, summary: MemorySummary) -> None:
        ...

    def delete(self, ts_code: str) -> None:
        ...

    def clear(self) -> None:
        ...


class RedisMemoryStore:
    def __init__(self, redis_url: str, prefix: str = "octts:memory") -> None:
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError("redis is not installed.") from exc
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix

    def get(self, ts_code: str) -> MemorySummary | None:
        payload = self._client.get(self._key(ts_code))
        if not payload:
            return None
        return MemorySummary.model_validate_json(payload)

    def set(self, summary: MemorySummary) -> None:
        self._client.set(self._key(summary.ts_code), summary.model_dump_json())

    def delete(self, ts_code: str) -> None:
        self._client.delete(self._key(ts_code))

    def clear(self) -> None:
        keys = self._client.keys(f"{self._prefix}:*")
        if keys:
            self._client.delete(*keys)

    def _key(self, ts_code: str) -> str:
        return f"{self._prefix}:{ts_code}"


class FileMemoryStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, ts_code: str) -> MemorySummary | None:
        payload = self._load()
        record = payload.get(ts_code)
        if not record:
            return None
        return MemorySummary.model_validate(record)

    def set(self, summary: MemorySummary) -> None:
        payload = self._load()
        payload[summary.ts_code] = summary.model_dump(mode="json")
        self._save(payload)

    def delete(self, ts_code: str) -> None:
        payload = self._load()
        payload.pop(ts_code, None)
        self._save(payload)

    def clear(self) -> None:
        self._save({})

    def _load(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        content = self._path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        return json.loads(content)

    def _save(self, payload: dict[str, object]) -> None:
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def create_memory_store(settings: Settings) -> MemoryStore:
    if settings.memory_backend == "redis" and settings.redis_url:
        return RedisMemoryStore(settings.redis_url)
    return FileMemoryStore(settings.memory_file_path)
