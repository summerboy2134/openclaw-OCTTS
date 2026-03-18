from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from octts.config import Settings
from octts.schemas.report import PositionStatus


class FilePositionStore:
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def get_status(self, ts_code: str) -> Optional[PositionStatus]:
        payload = self._load()
        value = payload.get(_normalize_ts_code(ts_code))
        if value in {"holding", "watching"}:
            return value
        return None

    def set_status(self, ts_code: str, status: PositionStatus) -> None:
        payload = self._load()
        payload[_normalize_ts_code(ts_code)] = status
        self._save(payload)

    def delete_status(self, ts_code: str) -> None:
        payload = self._load()
        payload.pop(_normalize_ts_code(ts_code), None)
        self._save(payload)

    def clear(self) -> None:
        self._save({})

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        content = self._path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        payload = json.loads(content)
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items()}

    def _save(self, payload: dict[str, str]) -> None:
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_position_store(settings: Settings) -> FilePositionStore:
    return FilePositionStore(settings.position_file_path)


def _normalize_ts_code(ts_code: str) -> str:
    return ts_code.strip().upper()
