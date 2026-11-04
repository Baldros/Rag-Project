"""Record of one session activation, readable while it is still running.

Activation can take minutes on this machine and has no deadline, so the user
must be able to see which stage it is in. Each attempt owns one JSONL file. The
controller and the runtime both append to it, one complete line per event, and
any client may read it without talking to the runtime.
"""

from __future__ import annotations

import json
import re
import secrets
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from kb.config import STORE_DIR

STARTUP_DIR = STORE_DIR / "startup"
KEEP_ATTEMPTS = 20
SCHEMA_VERSION = 1
MAX_TRACEBACK_CHARS = 4000

STAGES = {
    "ingestion": (
        "process_start", "validate_config", "database", "vector_index",
        "parser_import", "parser_load", "chunker_import", "tokenizer_load",
        "embedder_load", "checks", "listener_bind",
    ),
    "interaction": (
        "process_start", "validate_config", "database", "vector_index",
        "models_load", "listener_bind",
    ),
}

_VALID_ID = re.compile(r"^[0-9a-f]{8,64}$")


def new_id() -> str:
    return secrets.token_hex(8)


def _attempt_path(startup_id: str, suffix: str) -> Path:
    # The id reaches this module through argv and session.json.
    if not _VALID_ID.match(startup_id):
        raise ValueError(f"Identificador de ativação inválido: {startup_id!r}")
    return STARTUP_DIR / f"{startup_id}{suffix}"


def events_path(startup_id: str) -> Path:
    return _attempt_path(startup_id, ".jsonl")


class StartupLog:
    """Append-only writer. Failing to write never fails the activation."""

    def __init__(self, startup_id: str | None, component: str) -> None:
        self.startup_id = startup_id
        self.component = component

    def emit(self, event: str, **fields: Any) -> None:
        if not self.startup_id:
            return
        record = {
            **fields,
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "component": self.component,
        }
        try:
            path = events_path(self.startup_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, ValueError):
            pass

    def warning(self, check: str, message: str) -> None:
        self.emit("warning", check=check, message=message)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.emit("stage.started", stage=name)
        try:
            yield
        except BaseException as exc:
            self.emit(
                "stage.failed", stage=name, error_type=type(exc).__name__, error=str(exc)[:1000],
                # format_exc captures frames, never locals or the environment.
                traceback=traceback.format_exc()[-MAX_TRACEBACK_CHARS:],
            )
            raise
        self.emit("stage.finished", stage=name)


def read_events(startup_id: str) -> list[dict[str, Any]]:
    events = []
    try:
        lines = events_path(startup_id).read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return events
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # The other process may still be writing the last line.
    return events


def failure_summary(startup_id: str) -> str | None:
    """The failing stage of this attempt, never an error left by an older one."""
    for event in reversed(read_events(startup_id)):
        if event.get("event") == "stage.failed":
            return f"Etapa {event.get('stage')}: {event.get('error_type')}: {event.get('error')}"
    return None


def last_stage(startup_id: str) -> str | None:
    for event in reversed(read_events(startup_id)):
        if event.get("event", "").startswith("stage."):
            return event.get("stage")
    return None


def request_cancel(startup_id: str) -> None:
    path = _attempt_path(startup_id, ".cancel")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def cancel_requested(startup_id: str) -> bool:
    return _attempt_path(startup_id, ".cancel").exists()


def clear_cancel(startup_id: str) -> None:
    _attempt_path(startup_id, ".cancel").unlink(missing_ok=True)


def prune(keep: int = KEEP_ATTEMPTS) -> None:
    """Keep the most recent attempts, which are enough to calibrate and diagnose."""
    try:
        files = sorted(STARTUP_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files[keep:]:
            path.unlink(missing_ok=True)
            path.with_suffix(".cancel").unlink(missing_ok=True)
    except OSError:
        pass
