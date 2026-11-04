"""Activation of a real runtime process, with synthetic stages and no model loads."""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kb import processes, session, startup_log


STUB = r'''
import os, time
from pathlib import Path
from kb import runtime

root = Path(os.environ["TEST_STARTUP_ROOT"])
behaviour = os.environ["TEST_STARTUP_BEHAVIOUR"]

def prepare(self):
    (root / "runtime.pid").write_text(str(os.getpid()))
    if behaviour == "broken":
        with self.startup.stage("parser_import"):
            raise ModuleNotFoundError("No module named 'docling'")
    if behaviour == "slow":
        with self.startup.stage("parser_import"):
            time.sleep(60)
    self.startup.warning("ollama", "synthetic warning")
    with self.startup.stage("parser_load"):
        pass

runtime.Runtime._prepare_ingestion = prepare
runtime.Runtime.close = lambda self: None
runtime.main()
'''


class ActivationTests(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.store = self.root / "store"
        self.store.mkdir()
        self.environ = {"KB_STORE_DIR": str(self.store), "TEST_STARTUP_ROOT": str(self.root)}
        stack.enter_context(patch.dict(os.environ, self.environ))
        stack.enter_context(patch.object(session, "STORE_DIR", self.store))
        stack.enter_context(patch.object(session, "STATE_PATH", self.store / "session.json"))
        stack.enter_context(patch.object(session, "LOCK_PATH", self.store / "session.lock"))
        stack.enter_context(patch.object(session, "LOG_PATH", self.store / "session.log"))
        stack.enter_context(patch.object(startup_log, "STARTUP_DIR", self.store / "startup"))
        stack.enter_context(patch.object(session, "_runtime_command", side_effect=lambda state, mode: [
            sys.executable, "-c", STUB, "--mode", mode, "--address", state["address"],
            "--nonce", state["nonce"], "--startup-id", state["startup_id"],
        ]))

    def behave(self, behaviour):
        patcher = patch.dict(os.environ, {"TEST_STARTUP_BEHAVIOUR": behaviour})
        patcher.start()
        self.addCleanup(patcher.stop)

    def attempt_events(self):
        files = list((self.store / "startup").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        return startup_log.read_events(files[0].stem)

    def runtime_pid(self):
        return int((self.root / "runtime.pid").read_text())

    def wait_until(self, condition, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.05)
        self.fail("Condition not reached in time")

    def assert_gone(self, pid):
        self.wait_until(lambda: not processes.alive(pid, None))

    def test_activation_records_stages_warnings_and_readiness(self):
        self.behave("ready")
        live = session.start("ingestion")
        self.addCleanup(session.stop)
        self.assertTrue(live["ready"])
        self.assertTrue(session.status()["active"])
        events = self.attempt_events()
        self.assertEqual(events[0]["event"], "plan")
        self.assertEqual(events[0]["stages"], list(startup_log.STAGES["ingestion"]))
        finished = [e["stage"] for e in events if e["event"] == "stage.finished"]
        self.assertEqual(finished, ["process_start", "validate_config", "database", "parser_load", "listener_bind"])
        self.assertEqual([e["check"] for e in events if e["event"] == "warning"], ["ollama"])
        self.assertEqual(events[-1]["event"], "ready")

    def test_broken_installation_fails_in_its_own_stage(self):
        self.behave("broken")
        with self.assertRaises(session.SessionError) as error:
            session.start("ingestion")
        self.assertEqual(error.exception.code, "START_FAILED")
        self.assertIn("parser_import", error.exception.hint)
        self.assertIn("No module named 'docling'", error.exception.hint)
        self.assertFalse(session.STATE_PATH.exists())
        failed = [e for e in self.attempt_events() if e["event"] == "stage.failed"]
        self.assertEqual(failed[0]["error_type"], "ModuleNotFoundError")
        self.assertIn("Traceback", failed[0]["traceback"])

    def test_old_failure_is_never_attributed_to_a_new_attempt(self):
        self.behave("ready")
        startup_log.StartupLog("00000000aaaaaaaa", "session").emit(
            "stage.failed", stage="parser_import", error_type="Old", error="old")
        self.assertIsNone(startup_log.failure_summary("11111111bbbbbbbb"))

    def test_status_reports_activation_in_progress(self):
        self.behave("slow")
        errors = []

        def start():
            try:
                session.start("ingestion")
            except session.SessionError as exc:
                errors.append(exc)

        thread = threading.Thread(target=start)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(session.stop)
        self.wait_until(lambda: session.status().get("starting"))
        status = session.status()
        self.assertFalse(status["active"])
        self.assertEqual(status["starting"]["mode"], "ingestion")
        self.assertTrue(startup_log.events_path(status["starting"]["startup_id"]).exists())

    def test_off_cancels_an_activation_and_leaves_no_process(self):
        self.behave("slow")
        errors = []

        def start():
            try:
                session.start("ingestion")
            except session.SessionError as exc:
                errors.append(exc)

        thread = threading.Thread(target=start)
        thread.start()
        self.addCleanup(thread.join)
        self.wait_until((self.root / "runtime.pid").exists)
        pid = self.runtime_pid()
        result = session.stop()
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors[0].code, "START_CANCELLED")
        self.assertFalse(result["active"])
        self.assertFalse(session.STATE_PATH.exists())
        self.assert_gone(pid)
        self.assertEqual(self.attempt_events()[-1]["event"], "cancelled")
        self.assertEqual(list((self.store / "startup").glob("*.cancel")), [])

    def test_off_stops_an_activation_whose_controller_died(self):
        self.behave("slow")
        state = {"mode": "ingestion", "address": session._address(), "authkey": "00" * 32,
                 "nonce": "11" * 16, "startup_id": startup_log.new_id()}
        process = subprocess.Popen(
            session._runtime_command(state, "ingestion"), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**os.environ, "KB_SESSION_AUTHKEY": state["authkey"]},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.addCleanup(process.wait)
        state.update(pid=process.pid, spawn_started=processes.identity(process.pid))
        session._write_state(state)
        self.wait_until((self.root / "runtime.pid").exists)
        pid = self.runtime_pid()

        with self.assertRaises(session.SessionError) as refused:
            session.start("ingestion")
        self.assertEqual(refused.exception.code, "SESSION_STARTING")

        self.assertEqual(session.stop(), {"active": False, "cancelled_start": True})
        self.assertFalse(session.STATE_PATH.exists())
        self.assert_gone(pid)


class StartupLogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(startup_log, "STARTUP_DIR", Path(directory.name) / "startup")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_identifier_cannot_escape_the_directory(self):
        with self.assertRaises(ValueError):
            startup_log.events_path("..\\..\\session")

    def test_incomplete_last_line_is_ignored(self):
        startup_log.StartupLog("aaaaaaaa", "session").emit("plan", mode="ingestion", stages=[])
        with startup_log.events_path("aaaaaaaa").open("a", encoding="utf-8") as file:
            file.write('{"event": "stage.st')
        self.assertEqual([e["event"] for e in startup_log.read_events("aaaaaaaa")], ["plan"])

    def test_writing_never_raises_and_null_log_is_silent(self):
        startup_log.StartupLog(None, "runtime").emit("ready")
        startup_log.StartupLog("not valid", "runtime").emit("ready")
        self.assertFalse(startup_log.STARTUP_DIR.exists())

    def test_prune_keeps_most_recent_attempts(self):
        for index in range(5):
            attempt = f"{index:08x}"
            startup_log.StartupLog(attempt, "session").emit("plan", mode="ingestion", stages=[])
            os.utime(startup_log.events_path(attempt), (index + 1, index + 1))
        startup_log.request_cancel("00000000")
        startup_log.prune(keep=2)
        self.assertEqual(sorted(path.stem for path in startup_log.STARTUP_DIR.glob("*.jsonl")),
                         ["00000003", "00000004"])
        self.assertFalse(startup_log.cancel_requested("00000000"))

    def test_events_are_single_json_lines(self):
        log = startup_log.StartupLog("bbbbbbbb", "runtime")
        with self.assertRaises(ValueError):
            with log.stage("database"):
                raise ValueError("linha\nquebrada")
        lines = startup_log.events_path("bbbbbbbb").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["event"] for line in lines], ["stage.started", "stage.failed"])
