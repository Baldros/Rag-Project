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

from kb import startup_log
from kb.config import PROJECT_ROOT, STORE_DIR
from kb.startup_log import StartupLog

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
        # Reading the file would fail on Windows while another client holds the byte lock.
        if os.fstat(file.fileno()).st_size == 0:
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


def _starting(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """An activation still in progress: spawned, not yet answering, process alive."""
    if not state or "owner_started" in state or not state.get("startup_id"):
        return None
    from kb import processes

    if not processes.alive(state.get("pid"), state.get("spawn_started")):
        return None
    return {"mode": state.get("mode"), "startup_id": state["startup_id"]}


def status() -> dict[str, Any]:
    state = _read_state()
    live = _probe(state)
    if live:
        return {"active": True, **live}
    starting = _starting(state)
    return {"active": False, **({"starting": starting} if starting else {})}


def call(mode: str, action: str, **payload: Any) -> dict[str, Any]:
    # Admissions and mode transitions share a lock across CLI clients. Waiting
    # for an ingestion job happens after this request has released the lock.
    with _locked():
        return _call_locked(mode, action, **payload)


def _call_locked(mode: str, action: str, **payload: Any) -> dict[str, Any]:
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


def _runtime_command(state: dict[str, Any], mode: str) -> list[str]:
    return [
        sys.executable, "-m", "kb.runtime", "--mode", mode, "--address", state["address"],
        "--nonce", state["nonce"], "--startup-id", state["startup_id"],
    ]


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
        if _starting(old):
            raise SessionError("SESSION_STARTING", f"Já há uma ativação de {old.get('mode')} em andamento",
                               "Aguarde terminar ou desligue para cancelar")
        _require_dead(old)
        STATE_PATH.unlink(missing_ok=True)

    from kb import processes

    startup_id = startup_log.new_id()
    startup_log.prune()
    attempt = StartupLog(startup_id, "session")
    state = {
        "mode": mode,
        "address": _address(),
        "authkey": secrets.token_hex(32),
        "nonce": secrets.token_hex(16),
        "startup_id": startup_id,
    }
    command = _runtime_command(state, mode)
    attempt.emit("plan", mode=mode, stages=list(startup_log.STAGES[mode]))
    attempt.emit("stage.started", stage="process_start")
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
            # Unbuffered so a client following the log sees the activation as it happens.
            env={**os.environ, "KB_SESSION_AUTHKEY": state["authkey"], "PYTHONUNBUFFERED": "1"},
        )
    state["pid"] = process.pid
    state["spawn_started"] = processes.identity(process.pid)
    _write_state(state)
    # No deadline: loading is slow on this machine and that is expected. It ends
    # when the runtime answers, when it dies, or when the user cancels with `off`.
    try:
        while True:
            live = _probe(state)
            if live:
                state["pid"] = live["pid"]
                state["owner_started"] = processes.identity(live["pid"])
                _write_state(state)
                return live
            if process.poll() is not None:
                detail = startup_log.failure_summary(startup_id) or _last_log_error()
                attempt.emit("failed", exit_code=process.returncode, detail=detail)
                raise SessionError(
                    "START_FAILED", f"Falha ao iniciar o modo {mode}",
                    detail or f"Consulte {LOG_PATH}",
                )
            if startup_log.cancel_requested(startup_id):
                attempt.emit("cancelled")
                raise SessionError("START_CANCELLED", f"Ativação do modo {mode} cancelada")
            time.sleep(0.5)
    except BaseException:
        startup_log.clear_cancel(startup_id)
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
            ("RuntimeError:", "ValueError:", "ImportError:", "ModuleNotFoundError:",
             "kb.session.SessionError:")
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
    from kb import processes

    while time.monotonic() < deadline:
        if not processes.alive(pid, None):
            return True
        time.sleep(0.1)
    return False


def _terminate(pid: int, handle) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        terminate = ctypes.windll.kernel32.TerminateProcess
        terminate.argtypes = [wintypes.HANDLE, wintypes.UINT]
        terminate(handle, 1)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Runtime may exit itself immediately after acknowledging.


def _stop_abandoned_start(state: dict[str, Any]) -> dict[str, Any]:
    """Holding the lock proves the controller is gone, so nobody else can stop it."""
    pid = int(state["pid"])
    try:
        with _process_target(pid) as handle:
            _terminate(pid, handle)
            if not _await_exit(pid, handle):
                raise SessionError("STOP_TIMEOUT", "Não consegui encerrar a ativação")
    except SessionError as exc:
        if exc.code != "PROCESS_GONE":
            raise
    StartupLog(state["startup_id"], "session").emit("cancelled")
    startup_log.clear_cancel(state["startup_id"])
    STATE_PATH.unlink(missing_ok=True)
    return {"active": False, "cancelled_start": True}


def _require_dead(state: dict[str, Any]) -> None:
    from kb import processes

    if processes.alive(state.get("pid"), state.get("owner_started")):
        raise SessionError("SESSION_UNREACHABLE", "Processo existe, mas a sessão não respondeu",
                           "Não é seguro iniciar outra sessão ou descartar a existente")


def _stop_locked(cancel: bool, interactive: bool, fail_if_busy: bool = False,
                 expected_job_id: str | None = None) -> dict[str, Any]:
    state = _read_state()
    live = _probe(state)
    if not state or not live:
        if not expected_job_id and _starting(state):
            return _stop_abandoned_start(state)
        if state:
            _require_dead(state)
        if expected_job_id:
            raise SessionError("JOB_NOT_ACTIVE", f"Job {expected_job_id} não está ativo")
        STATE_PATH.unlink(missing_ok=True)
        return {"active": False, "already_off": True}
    job_id = live.get("job_id")
    if expected_job_id and job_id != expected_job_id:
        raise SessionError("JOB_NOT_ACTIVE", f"Job {expected_job_id} não está ativo")
    selected_job_id = job_id
    if job_id and not cancel:
        if fail_if_busy:
            raise SessionError("BUSY", f"Job {job_id} em andamento",
                               f"Consulte `kb jobs show {job_id}` ou cancele explicitamente")
        cancel = _decision(state, interactive)
    latest = _probe(state)
    if latest is None:
        _require_dead(state)
        STATE_PATH.unlink(missing_ok=True)
        return {"active": False, "already_off": True}
    job_id = latest.get("job_id")
    if (expected_job_id and job_id != expected_job_id) or (cancel and job_id and job_id != selected_job_id):
        raise SessionError("JOB_NOT_ACTIVE", f"Job {expected_job_id or selected_job_id} não está ativo")
    runtime_pid = int(latest["pid"])
    model_unload_failed = False
    with _process_target(runtime_pid) as handle:
        if cancel and job_id:
            from kb import processes
            from kb.jobs import finish_job, get_job

            cancel_pid, cancel_started = processes.current()
            # Atomically fence admissions and persist intent before terminating.
            _request(state, "cancel", job_id=job_id, cancel_pid=cancel_pid,
                     cancel_owner_started=cancel_started)
            # The authenticated pipe proved the identity before opening the handle.
            _terminate(runtime_pid, handle)
            if not _await_exit(runtime_pid, handle):
                raise SessionError("STOP_TIMEOUT", "Não consegui encerrar a sessão")
            # Read the persisted phase after exit: it may have changed after ping.
            cancelled = get_job(job_id)
            if cancelled and cancelled.get("pass_name") == "enrich":
                from kb import llm

                try:
                    model_unload_failed = not llm.unload(force=True)
                except Exception:
                    model_unload_failed = True
            finish_job(job_id, "cancelled",
                       cleanup_status="failed" if model_unload_failed else "done",
                       cleanup_error="Descarga do modelo de geração não confirmada" if model_unload_failed else None)
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
         fail_if_busy: bool = False, expected_job_id: str | None = None) -> dict[str, Any]:
    # An activation holds the lock for as long as it takes, so ask it to stop first.
    state = _read_state()
    if not expected_job_id and _starting(state):
        startup_log.request_cancel(state["startup_id"])
    with _locked():
        return _stop_locked(cancel, interactive, fail_if_busy, expected_job_id)


def switch(mode: str, cancel: bool = False, interactive: bool = False,
           fail_if_busy: bool = False) -> dict[str, Any]:
    with _locked():
        current = _probe(_read_state())
        if current and current["mode"] == mode:
            return current
        if current:
            _stop_locked(cancel, interactive, fail_if_busy)
        return _start_locked(mode)
