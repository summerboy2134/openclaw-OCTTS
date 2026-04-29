from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from octts.config import Settings


def configure_tool_logging(settings: Settings, tool_name: str) -> logging.Logger:
    log_dir = Path(settings.history_dir_path).parent / "logs" / "tools"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tool_name}.log"

    logger = logging.getLogger(f"octts.tools.{tool_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def print_json(payload: Dict[str, Any], *, output_file: Optional[str] = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
