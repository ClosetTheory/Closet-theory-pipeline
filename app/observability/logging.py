"""Structured JSON logging for Image Ingestion Pipeline."""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured extra fields if present
        if hasattr(record, "structured_data"):
            log_obj.update(record.structured_data)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def log_stage_event(
    pipeline_run_id: str,
    garment_id: str,
    stage: str,
    status: str,
    attempt: int = 1,
    duration_ms: Optional[float] = None,
    model: Optional[str] = None,
    model_version: Optional[str] = None,
    input_hash: Optional[str] = None,
    output_hash: Optional[str] = None,
    error: Optional[str] = None,
):
    """Logs structured stage event conforming to PRD Section 22."""
    extra = {
        "structured_data": {
            "pipeline_run_id": pipeline_run_id,
            "garment_id": garment_id,
            "stage": stage,
            "status": status,
            "attempt": attempt,
            "duration_ms": duration_ms,
            "model": model,
            "model_version": model_version,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "error": error,
        }
    }
    level = logging.ERROR if status in ("FAILED",) else logging.INFO
    logger.log(
        level,
        f"Stage {stage} [{status}] for garment {garment_id} (attempt {attempt})",
        extra=extra,
    )
