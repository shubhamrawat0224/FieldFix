#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
REQUEST_LOG_PATH = LOG_DIR / "request_audit.jsonl"


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event_name: str, status: str, correlation_id: str | None = None, **context: Any) -> None:
    _ensure_log_dir()
    payload: dict[str, Any] = {
        "ts": time.time(),
        "event": event_name,
        "status": status,
        "correlation_id": correlation_id or new_correlation_id(),
    }
    for key, value in context.items():
        if value is not None:
            payload[key] = value

    with REQUEST_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


@contextmanager
def stage_timer(event_name: str, correlation_id: str | None = None, **context: Any) -> Iterator[str]:
    cid = correlation_id or new_correlation_id()
    started = time.perf_counter()
    try:
        yield cid
        status = "ok"
    except Exception as exc:  # pragma: no cover - debugging path
        status = "error"
        context["error"] = str(exc)
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        context["duration_ms"] = duration_ms
        log_event(event_name, status, correlation_id=cid, **context)


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "openai/gpt-5-mini": {"input": 0.00000025, "output": 0.0000010},
        "openai/gpt-4.1-mini": {"input": 0.0000004, "output": 0.0000016},
        "openai/gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
        "openai/gpt-4o": {"input": 0.000005, "output": 0.000015},
        "openai/text-embedding-3-small": {"input": 0.00000002, "output": 0.0},
    }
    pricing = prices.get(model_name, {"input": 0.0000003, "output": 0.0000012})
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
