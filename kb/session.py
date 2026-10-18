"""Manual, single-instance lifecycle for the local RAG runtime.

The CLI is the public interface. A private authenticated pipe carries commands
to the one process that owns loaded models; no HTTP endpoint is exposed.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Iterator

from kb.config import PROJECT_ROOT, STORE_DIR

STATE_PATH = STORE_DIR / "session.json"
LOCK_PATH = STORE_DIR / "session.lock"
LOG_PATH = STORE_DIR / "session.log"
MODES = {"interaction", "ingestion"}


class SessionError(RuntimeError):
    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@contextmanager
def _locked() -> Iterator[None]:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as file:
        file.seek(0)
        if file.read(1) == b"":
            file.seek(0)
            file.write(b"0")
            file.flush()
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)


def _read_state() -> dict[str, Any] | None:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_state(state: dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _request(state: dict[str, Any], action: str, **payload: Any) -> dict[str, Any]:
    with Client(
        state["address"],
        family="AF_PIPE" if os.name == "nt" else "AF_UNIX",
        authkey=bytes.fromhex(state["authkey"]),
    ) as connection:
        connection.send({"action": action, "payload": payload})
        reply = connection.recv()
    if reply.get("nonce") != state["nonce"]:
        raise SessionError("SESSION_STALE", "Identidade da sessão não confere")
    if not reply.get("ok"):
        error = reply.get("error", {})
        raise SessionError(
            error.get("code", "RUNTIME_ERROR"),
            error.get("message", "Falha na sessão"),
            error.get("hint"),
        )
    return reply.get("data", {})


def _probe(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None
    try:
        result = _request(state, "ping")
        # Windows venv python.exe is a launcher that starts the real interpreter
        # as a child. The authenticated nonce identifies the runtime; its PID
        # legitimately differs from Popen.pid on that platform.
        return result
    except (OSError, EOFError, KeyError, ValueError, SessionError):
        return None


def status() -> dict[str, Any]:
    state = _read_state()
    live = _probe(state)
    return {"active": bool(live), **(live or {})}


def call(mode: str, action: str, **payload: Any) -> dict[str, Any]:
    state = _read_state()
    live = _probe(state)
    if not live or not state:
        raise SessionError(
            "SESSION_OFF", "RAG desligado", f"Execute `kb on --mode {mode}`"
        )
    if live["mode"] != mode:
        raise SessionError(
            "WRONG_MODE",
            f"Sessão em modo {live['mode']}; esta operação requer {mode}",
            f"Execute `kb switch --mode {mode}`",
        )
    return _request(state, action, **payload)


def _address() -> str:
    suffix = secrets.token_hex(8)
    if os.name == "nt":
        return rf"\\.\pipe\rag-project-{suffix}"
    return str(STORE_DIR / f"session-{suffix}.sock")


def _start_locked(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise SessionError("INVALID_MODE", f"Modo inválido: {mode}")
    old = _read_state()
    live = _probe(old)
    if live:
        if live["mode"] == mode:
            return live
        raise SessionError(
            "WRONG_MODE",
            f"Sessão já está em {live['mode']}",
            f"Execute `kb switch --mode {mode}`",
        )
    if old:
        STATE_PATH.unlink(missing_ok=True)

    state = {
        "mode": mode,
        "address": _address(),
        "authkey": secrets.token_hex(32),
        "nonce": secrets.token_hex(16),
    }
    command = [
        sys.executable, "-m", "kb.runtime", "--mode", mode,
        "--address", state["address"], "--nonce", state["nonce"],
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=flags,
            start_new_session=os.name != "nt",
            env={**os.environ, "KB_SESSION_AUTHKEY": state["authkey"]},
        )
    state["pid"] = process.pid
    _write_state(state)
    deadline = time.monotonic() + 300
    try:
        while time.monotonic() < deadline:
            live = _probe(state)
            if live:
                return live
            if process.poll() is not None:
                detail = _last_log_error()
                raise SessionError(
                    "START_FAILED", f"Falha ao iniciar o modo {mode}",
                    detail or f"Consulte {LOG_PATH}",
                )
            time.sleep(0.5)
        raise SessionError("START_TIMEOUT", f"Tempo esgotado ao iniciar {mode}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if _read_state() == state:
            STATE_PATH.unlink(missing_ok=True)
        raise


def _last_log_error() -> str | None:
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        return next((line for line in reversed(lines) if line.startswith(
            ("RuntimeError:", "ValueError:", "ImportError:", "ModuleNotFoundError:")
        )), None)
    except OSError:
        return None


def start(mode: str) -> dict[str, Any]:
    with _locked():
        return _start_locked(mode)


def _decision(state: dict[str, Any], interactive: bool) -> bool:
    """True means cancel. A completed job dismisses a pending prompt."""
    if not interactive or not sys.stdin.isatty():
        return False
    sys.stderr.write("Há um job ativo. [Enter] esperar / [c] cancelar: ")
    sys.stderr.flush()
    if os.name == "nt":
        import msvcrt

        while True:
            current = _probe(state)
            if not current or not current.get("job_id"):
                sys.stderr.write("\nJob concluído; continuando.\n")
                return False
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                sys.stderr.write(key + "\n")
                return key == "c"
            time.sleep(0.2)
    answer = input().strip().lower()
    return answer in {"c", "cancelar", "cancel"}


@contextmanager
def _process_target(pid: int):
    """Hold a Windows process handle so a reused PID cannot target another app."""
    if os.name != "nt":
        yield None
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x0001, False, pid)
    if not handle:
        raise SessionError("PROCESS_GONE", "O processo da sessão já terminou")
    try:
        yield handle
    finally:
        kernel.CloseHandle(handle)


def _await_exit(pid: int, handle, timeout: float = 60) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        wait = ctypes.windll.kernel32.WaitForSingleObject
        wait.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        wait.restype = wintypes.DWORD
        return wait(handle, int(timeout * 1000)) == 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def _stop_locked(cancel: bool, interactive: bool, fail_if_busy: bool = False) -> dict[str, Any]:
    state = _read_state()
    live = _probe(state)
    if not state or not live:
        STATE_PATH.unlink(missing_ok=True)
        return {"active": False, "already_off": True}
    job_id = live.get("job_id")
    if job_id and not cancel:
        if fail_if_busy:
            raise SessionError("BUSY", f"Job {job_id} em andamento",
                               f"Consulte `kb jobs show {job_id}` ou cancele explicitamente")
        cancel = _decision(state, interactive)
    latest = _probe(state)
    if latest is None:
        STATE_PATH.unlink(missing_ok=True)
        return {"active": False, "already_off": True}
    job_id = latest.get("job_id")
    runtime_pid = int(latest["pid"])
    model_unload_failed = False
    with _process_target(runtime_pid) as handle:
        if cancel and job_id:
            # The authenticated pipe proved the identity before opening the handle.
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                terminate = ctypes.windll.kernel32.TerminateProcess
                terminate.argtypes = [wintypes.HANDLE, wintypes.UINT]
                terminate(handle, 1)
            else:
                os.kill(runtime_pid, signal.SIGTERM)
            if not _await_exit(runtime_pid, handle):
                raise SessionError("STOP_TIMEOUT", "Não consegui encerrar a sessão")
            from kb.jobs import update_job

            update_job(job_id, status="cancelled", finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            if latest.get("pass_name") == "enrich":
                from kb import llm

                model_unload_failed = not llm.unload(force=True)
        else:
            while job_id:
                time.sleep(0.5)
                latest = _probe(state)
                if not latest:
                    break
                job_id = latest.get("job_id")
            if _probe(state):
                _request(state, "shutdown")
            if not _await_exit(runtime_pid, handle):
                raise SessionError("STOP_TIMEOUT", "Sessão ainda está liberando recursos")
    STATE_PATH.unlink(missing_ok=True)
    result = {"active": False, "cancelled_job": job_id if cancel else None}
    if model_unload_failed:
        result["warning"] = "Não consegui confirmar a descarga do modelo de geração"
    return result


def stop(cancel: bool = False, interactive: bool = False,
         fail_if_busy: bool = False) -> dict[str, Any]:
    with _locked():
        return _stop_locked(cancel, interactive, fail_if_busy)


def switch(mode: str, cancel: bool = False, interactive: bool = False,
           fail_if_busy: bool = False) -> dict[str, Any]:
    with _locked():
        current = _probe(_read_state())
        if current and current["mode"] == mode:
            return current
        if current:
            _stop_locked(cancel, interactive, fail_if_busy)
        return _start_locked(mode)
