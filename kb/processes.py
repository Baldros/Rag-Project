"""Process birth identities, without importing models or contacting the runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def identity(pid: int) -> str | None:
    """Return an OS birth token, None for a dead process, or raise if unknown.

    A permission error is not evidence of death. Windows and Linux are the
    supported runtime platforms; fail closed on platforms without a birth token.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no such PID
                return None
            raise ctypes.WinError(error)
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value != 259:  # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"{boot}:{fields[19]}"
        except FileNotFoundError:
            return None
    raise OSError(f"Process birth identity unsupported on {sys.platform}")


def alive(pid: int | None, started: str | None) -> bool:
    if not pid:
        return False
    try:
        current = identity(int(pid))
        # Legacy records lack a birth token: only recover on proven death.
        return current is not None and (started is None or current == started)
    except OSError:
        return True


def current() -> tuple[int, str]:
    pid = os.getpid()
    started = identity(pid)
    if started is None:
        raise RuntimeError("Cannot establish runtime process identity")
    return pid, started
