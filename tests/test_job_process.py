"""Actual runtime/pipe lifecycle with a synthetic ingestion and no model loads."""

import contextlib
import os
import secrets
import subprocess
import sys
import tempfile
import time
import unittest
from multiprocessing.connection import Client
from pathlib import Path
from unittest.mock import patch

from kb import db, jobs, llm, processes, session


STUB = r'''
import os, time
from pathlib import Path
from kb import runtime
from kb.ingest import pipeline

root = Path(os.environ["TEST_JOB_ROOT"])
runtime.Runtime._prepare_ingestion = staticmethod(lambda: None)
runtime.Runtime._cleanup_job = lambda self: []
runtime.Runtime.close = lambda self: None

def ingest(paths, on_checkpoint, on_progress, **kwargs):
    result = {"indexed": [{"doc_id": "synthetic"}], "skipped": [], "failed": []}
    on_checkpoint(result)
    on_progress("enrich", 0, 1, "synthetic enrichment")
    (root / "started").touch()
    deadline = time.monotonic() + 20
    while not (root / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Test did not release synthetic ingestion")
        time.sleep(0.02)
    return result

pipeline.ingest_paths = ingest
runtime.main()
'''


class RuntimeProcessTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        store = self.root / "store"
        store.mkdir()
        self.state = {"address": (rf"\\.\pipe\rag-test-{secrets.token_hex(8)}" if os.name == "nt"
                                  else str(self.root / "runtime.sock")),
                      "nonce": secrets.token_hex(16), "authkey": secrets.token_hex(32)}
        log = self.stack.enter_context((self.root / "runtime.log").open("w+", encoding="utf-8"))
        self.process = subprocess.Popen(
            [sys.executable, "-c", STUB, "--mode", "ingestion", "--address", self.state["address"],
             "--nonce", self.state["nonce"]], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            env={**os.environ, "KB_STORE_DIR": str(store), "KB_INGEST_ROOTS": str(self.root),
                 "KB_SESSION_AUTHKEY": self.state["authkey"], "TEST_JOB_ROOT": str(self.root)},
        )
        self.stack.callback(self.stop_stub)
        self.wait_for(lambda: session._probe(self.state))
        self.state.update(session._probe(self.state))
        self.state["owner_started"] = processes.identity(self.state["pid"])
        from kb.config import DB_PATH
        self.conn = self.stack.enter_context(contextlib.closing(db.init_db(store / DB_PATH.name)))
        self.stack.enter_context(patch.object(jobs, "get_conn", return_value=self.conn))
        self.stack.enter_context(patch.object(session, "STORE_DIR", store))
        self.stack.enter_context(patch.object(session, "STATE_PATH", store / "session.json"))
        self.stack.enter_context(patch.object(session, "LOCK_PATH", store / "session.lock"))
        session._write_state(self.state)
        self.pdf = self.root / "synthetic.pdf"
        self.pdf.write_bytes(b"synthetic fixture; never parsed")

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            if self.process.poll() is not None:
                self.fail((self.root / "runtime.log").read_text(encoding="utf-8"))
            time.sleep(0.02)
        self.fail("Synthetic runtime did not reach expected state")

    def stop_stub(self):
        if self.process.poll() is not None:
            return
        (self.root / "release").touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            live = session._probe(self.state)
            if not live:
                break
            if not live.get("job_id"):
                session._request(self.state, "shutdown")
                break
            time.sleep(0.02)
        self.process.wait(timeout=10)

    def start_job(self):
        accepted = session._request(self.state, "ingest", paths=[str(self.pdf)])
        self.wait_for(lambda: (self.root / "started").exists())
        return accepted["job_id"]

    def test_job_survives_returned_call_and_client_disconnect(self):
        job_id = self.start_job()
        # The initiating connection has closed; the synthetic job is still active.
        self.assertEqual(jobs.get_job(job_id)["status"], "running")
        (self.root / "release").touch()
        self.wait_for(lambda: jobs.get_job(job_id)["status"] == "done")
        self.assertEqual(jobs.get_job(job_id)["cleanup_status"], "done")
        self.assertIsNone(session._probe(self.state)["job_id"])

    def test_disconnection_before_reply_does_not_kill_an_accepted_job(self):
        with Client(self.state["address"], family="AF_PIPE" if os.name == "nt" else "AF_UNIX",
                    authkey=bytes.fromhex(self.state["authkey"])) as connection:
            connection.send({"action": "ingest", "payload": {"paths": [str(self.pdf)]}})
        self.wait_for(lambda: (self.root / "started").exists())
        job_id = jobs.list_jobs()[0]["job_id"]
        (self.root / "release").touch()
        self.wait_for(lambda: jobs.get_job(job_id)["status"] == "done")
        self.assertTrue(session._probe(self.state))

    def test_real_cancel_waits_for_exit_and_external_cleanup_before_terminal(self):
        job_id = self.start_job()

        def unload(force=False):
            self.assertTrue(force)
            self.assertFalse(processes.alive(self.state["pid"], self.state["owner_started"]))
            job = jobs.get_job(job_id)
            self.assertEqual(job["status"], "cancelling")
            self.assertIsNone(job["finished_at"])
            self.assertEqual(job["result"]["indexed"][0]["doc_id"], "synthetic")
            return True

        with patch.object(llm, "unload", side_effect=unload) as cleanup:
            result = session.stop(cancel=True, expected_job_id=job_id)
        cleanup.assert_called_once()
        self.process.wait(timeout=5)
        self.assertEqual(result["cancelled_job"], job_id)
        before = jobs.get_job(job_id)
        self.assertEqual(before["status"], "cancelled")
        self.assertEqual(before["cleanup_status"], "done")
        with self.assertRaises(session.SessionError):
            session.stop(cancel=True, expected_job_id=job_id)
        self.assertEqual(jobs.get_job(job_id), before)


if __name__ == "__main__":
    unittest.main()
