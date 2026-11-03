"""Durable job boundaries, using temporary SQLite databases and fake models."""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kb import cli, db, health, jobs, llm, models, processes, runtime, session
from kb.ingest import parse, pipeline, segment


class JobFixture(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.path = self.root / "jobs.sqlite3"
        self.conn = self.stack.enter_context(contextlib.closing(db.init_db(self.path)))
        self.stack.enter_context(patch.object(jobs, "get_conn", return_value=self.conn))

    def job(self):
        return jobs.create_job("ingest", "synthetic.pdf", conn=self.conn)

    def orphan(self, status="queued"):
        job_id = self.job()
        self.conn.execute("UPDATE jobs SET status=?, pid=NULL, owner_started=NULL WHERE job_id=?",
                          (status, job_id))
        return job_id


class JobTests(JobFixture):
    def test_queued_is_owned_before_worker_starts(self):
        job = jobs.get_job(self.job())
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["pid"], os.getpid())
        self.assertEqual(job["owner_started"], processes.identity(os.getpid()))
        self.assertIsNone(job["finished_at"])

    def test_all_orphan_states_recover_durably_once_and_views_agree(self):
        for state in jobs.ACTIVE:
            with self.subTest(state=state):
                job_id = self.orphan(state)
                saved = {"indexed": [{"doc_id": "saved"}], "pending": [{"phase": "enrich"}]}
                self.conn.execute("UPDATE jobs SET result_json=? WHERE job_id=?", (json.dumps(saved), job_id))
                listed = next(j for j in jobs.list_jobs() if j["job_id"] == job_id)
                shown = jobs.get_job(job_id)
                with patch.object(session, "status", return_value={"active": False}), \
                     patch.object(health, "_gpu_info", return_value={}), \
                     patch.object(health, "_dir_size_mb", return_value=0):
                    recent = next(j for j in health.status(conn=self.conn)["recent_jobs"] if j["job_id"] == job_id)
                self.assertEqual([listed["status"], shown["status"], recent["status"]], ["failed"] * 3)
                self.assertTrue(shown["finished_at"])
                self.assertEqual(shown["result"], saved)
                self.assertEqual(shown["cleanup_status"], "unconfirmed")
                self.assertFalse(any(j["job_id"] == job_id for j in jobs.active_jobs()))
                self.assertEqual(jobs.get_job(job_id), shown)
                with contextlib.closing(db.init_db(self.path)) as reopened:
                    self.assertEqual(jobs.get_job(job_id, reopened), shown)

    def test_reused_pid_is_not_the_job_owner(self):
        job_id = self.job()
        self.conn.execute("UPDATE jobs SET owner_started='old-process' WHERE job_id=?", (job_id,))
        self.assertEqual(jobs.get_job(job_id)["status"], "failed")

    def test_permission_error_does_not_prove_death(self):
        job_id = self.job()
        with patch.object(processes, "identity", side_effect=PermissionError("denied")):
            self.assertEqual(jobs.get_job(job_id)["status"], "queued")

    def test_actual_process_exit_recovers_queued_and_running_without_resuming(self):
        for state in ("queued", "running", "finalizing"):
            with self.subTest(state=state):
                script = (
                    "import os, sys; from kb import db, jobs; "
                    "conn=db.init_db(sys.argv[1]); "
                    "job=jobs.create_job('ingest', 'synthetic', conn=conn); "
                    "jobs.update_job(job, conn=conn, status=sys.argv[2], result={'saved': True}); "
                    "print(job, flush=True); os._exit(9)"
                )
                result = subprocess.run([sys.executable, "-c", script, str(self.path), state],
                                        capture_output=True, text=True, timeout=20,
                                        env={**os.environ, "KB_STORE_DIR": str(self.root)},
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.assertEqual(result.returncode, 9, result.stderr)
                job = jobs.get_job(result.stdout.strip())
                self.assertEqual(job["status"], "failed")
                self.assertEqual(job["result"], {"saved": True})

    def test_cancellation_controller_owns_cleanup_until_it_finishes_or_dies(self):
        job_id = self.job()
        pid, started = processes.current()
        self.assertTrue(jobs.request_cancel(job_id, pid, started))
        self.conn.execute("UPDATE jobs SET pid=NULL WHERE job_id=?", (job_id,))
        self.assertEqual(jobs.get_job(job_id)["status"], "cancelling")
        self.conn.execute("UPDATE jobs SET cancel_owner_started='dead-controller' WHERE job_id=?", (job_id,))
        recovered = jobs.get_job(job_id)
        self.assertEqual(recovered["status"], "cancelled")
        self.assertEqual(recovered["cleanup_status"], "unconfirmed")

    def test_terminal_outcome_cannot_be_published_twice_or_overwritten(self):
        for status in jobs.TERMINAL:
            with self.subTest(status=status):
                job_id = self.job()
                self.assertTrue(jobs.finish_job(job_id, status, result={"original": 1}, cleanup_status="done"))
                before = jobs.get_job(job_id)
                self.assertFalse(jobs.finish_job(job_id, "failed", result={"late": 1}))
                self.assertFalse(jobs.update_job(job_id, status="running", result={"late": 1}))
                self.assertFalse(jobs.request_cancel(job_id, *processes.current()))
                self.assertEqual(jobs.get_job(job_id), before)

    def test_runtime_startup_reconciles_old_jobs(self):
        job_id = self.orphan()
        with patch("kb.providers.validate_config"), \
             patch.object(runtime, "init_db", side_effect=lambda: db.init_db(self.path)), \
             patch.object(runtime.Runtime, "_prepare_ingestion"):
            runtime.Runtime("ingestion", "test").prepare()
        self.assertEqual(jobs.get_job(job_id)["status"], "failed")


class WorkerTests(JobFixture):
    def worker(self):
        worker = runtime.Runtime("ingestion", "test")
        worker.job_id = self.job()
        return worker

    def test_result_is_saved_but_job_stays_busy_until_cleanup_returns(self):
        worker = self.worker()
        job_id = worker.job_id
        result = {"indexed": [{"doc_id": "doc"}], "pending": [{"phase": "enrich"}]}

        def cleanup():
            job = jobs.get_job(job_id)
            self.assertEqual(job["result"], result)
            self.assertEqual(job["status"], "finalizing")
            self.assertIsNone(job["finished_at"])
            with self.assertRaises(session.SessionError) as error:
                worker.handle({"action": "reindex"})
            self.assertEqual(error.exception.code, "BUSY")
            return []

        with patch.object(pipeline, "ingest_paths", return_value=result), \
             patch.object(worker, "_cleanup_job", side_effect=cleanup), \
             patch.object(worker, "_prepare_ingestion") as reload_parser:
            worker._run_job(job_id, "ingest", {"paths": []})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["cleanup_status"], "done")
        self.assertTrue(job["finished_at"])
        self.assertIsNone(worker.job_id)
        reload_parser.assert_not_called()

    def test_cleanup_failure_preserves_result_and_is_terminal(self):
        worker = self.worker()
        job_id = worker.job_id
        result = {"indexed": [{"doc_id": "doc"}]}
        with patch.object(pipeline, "ingest_paths", return_value=result), \
             patch.object(worker, "_cleanup_job", return_value=["LLM unload failed"]):
            worker._run_job(job_id, "ingest", {"paths": []})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["result"], result)
        self.assertEqual(job["cleanup_status"], "failed")
        self.assertIn("LLM", job["cleanup_error"])

    def test_exception_keeps_last_checkpoint_and_cleans_all_resources(self):
        worker = self.worker()
        job_id = worker.job_id

        def fail(**kwargs):
            kwargs["on_checkpoint"]({"prepared": [{"doc_id": "saved"}]})
            raise RuntimeError("generation failed")

        with patch.object(pipeline, "ingest_paths", side_effect=lambda paths, **kw: fail(**kw)), \
             patch.object(worker, "_cleanup_job", return_value=[]) as cleanup:
            worker._run_job(job_id, "ingest", {"paths": []})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["result"], {"prepared": [{"doc_id": "saved"}]})
        self.assertIn("generation failed", job["error"])
        cleanup.assert_called_once()

    def test_thread_construction_and_start_failure_do_not_leave_queued_jobs(self):
        worker = runtime.Runtime("ingestion", "test")
        for factory in (Mock(side_effect=RuntimeError("no thread")), Mock(return_value=Mock(start=Mock(side_effect=RuntimeError("cannot start"))))):
            with patch.object(runtime.threading, "Thread", factory), self.assertRaises(RuntimeError):
                worker.handle({"action": "reindex"})
            self.assertIsNone(worker.job_id)
            self.assertFalse(jobs.active_jobs())
            self.assertTrue(all(j["status"] == "failed" for j in jobs.list_jobs()))

    def test_cleanup_attempts_every_resource_even_if_one_raises(self):
        with patch.object(parse, "release_converter", side_effect=RuntimeError("parser")), \
             patch.object(segment, "release_chunker") as chunker, \
             patch.object(models, "unload_all") as model_cleanup, \
             patch.object(llm, "unload", return_value=False) as generation:
            errors = runtime.Runtime("ingestion", "test")._cleanup_job()
        self.assertEqual(len(errors), 2)
        chunker.assert_called_once()
        model_cleanup.assert_called_once()
        generation.assert_called_once()

    def test_cancel_is_persisted_before_runtime_stops_admitting_work(self):
        worker = self.worker()
        job_id = worker.job_id
        pid, started = processes.current()
        worker.handle({"action": "cancel", "payload": {
            "job_id": job_id, "cancel_pid": pid, "cancel_owner_started": started,
        }})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "cancelling")
        self.assertTrue(job["cancel_requested_at"])
        self.assertIsNone(job["finished_at"])
        with self.assertRaises(session.SessionError) as error:
            worker.handle({"action": "reindex"})
        self.assertEqual(error.exception.code, "SESSION_STOPPING")


    def test_cancel_prevents_starting_generation_after_phase_was_fenced(self):
        worker = self.worker()
        job_id = worker.job_id
        generate = Mock()

        def ingest(paths, on_progress, **kwargs):
            on_progress("parse", 0, 1, "fixture")
            pid, started = processes.current()
            worker.handle({"action": "cancel", "payload": {
                "job_id": job_id, "cancel_pid": pid, "cancel_owner_started": started,
            }})
            on_progress("enrich", 0, 1, "must not start")
            generate()

        with patch.object(pipeline, "ingest_paths", side_effect=ingest), \
             patch.object(worker, "_cleanup_job", return_value=[]):
            worker._run_job(job_id, "ingest", {"paths": []})
        generate.assert_not_called()
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "cancelling")
        self.assertEqual(job["pass_name"], "parse")
        self.assertIsNone(job["finished_at"])

    def test_first_running_write_failure_is_finalized_instead_of_stranding_queued(self):
        worker = self.worker()
        job_id = worker.job_id
        update = jobs.update_job
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("first write failed")
            return update(*args, **kwargs)

        with patch.object(jobs, "update_job", side_effect=fail_once), \
             patch.object(worker, "_cleanup_job", return_value=[]):
            worker._run_job(job_id, "ingest", {"paths": []})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("first write failed", job["error"])
        self.assertIsNone(worker.job_id)

    def test_failed_reindex_retains_completed_batch_count(self):
        from kb.ingest import embed
        worker = self.worker()
        job_id = worker.job_id

        def embed_batches(doc_ids, on_progress):
            on_progress(12, 20, "saved first batch")
            raise RuntimeError("second batch failed")

        with patch.object(worker, "_release_ingestion"), \
             patch.object(worker, "_cleanup_job", return_value=[]), \
             patch.object(embed, "embed_pending", side_effect=embed_batches):
            worker._run_job(job_id, "reindex", {})
        job = jobs.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["result"], {"invalidated": 0, "embedded": 12})


class SharedSessionTests(unittest.TestCase):
    def test_cancel_cli_passes_exact_id_into_locked_transition(self):
        with patch.object(session, "stop", return_value={}) as stop:
            cli._run(cli.build_parser().parse_args(["jobs", "cancel", "job-a"]))
        stop.assert_called_once_with(cancel=True, expected_job_id="job-a")

    def test_changed_job_during_cancel_is_not_killed(self):
        state = {"pid": 10}
        live = {"pid": 10, "mode": "ingestion", "job_id": "job-a"}
        for expected in (None, "job-a"):
            with patch.object(session, "_read_state", return_value=state), \
                 patch.object(session, "_probe", side_effect=[live, {**live, "job_id": "job-b"}]), \
                 patch.object(session, "_process_target") as target, \
                 self.assertRaises(session.SessionError) as error:
                session._stop_locked(True, False, expected_job_id=expected)
            self.assertEqual(error.exception.code, "JOB_NOT_ACTIVE")
            target.assert_not_called()

    def test_unreachable_live_session_is_not_replaced_or_deleted(self):
        state = {"pid": os.getpid(), "owner_started": processes.identity(os.getpid())}
        for operation in (lambda: session._start_locked("interaction"), lambda: session._stop_locked(False, False)):
            with patch.object(session, "_read_state", return_value=state), \
                 patch.object(session, "_probe", return_value=None), \
                 patch.object(type(session.STATE_PATH), "unlink") as unlink, \
                 patch.object(session.subprocess, "Popen") as spawn, \
                 self.assertRaises(session.SessionError) as error:
                operation()
            self.assertEqual(error.exception.code, "SESSION_UNREACHABLE")
            unlink.assert_not_called()
            spawn.assert_not_called()

    def test_admission_holds_the_same_lock_as_mode_transitions(self):
        held = []

        @contextlib.contextmanager
        def locked():
            held.append(True)
            try:
                yield
            finally:
                held.pop()

        def request(*args, **kwargs):
            self.assertTrue(held)
            return {"job_id": "accepted"}

        with patch.object(session, "_locked", locked), \
             patch.object(session, "_read_state", return_value={"mode": "ingestion"}), \
             patch.object(session, "_probe", return_value={"mode": "ingestion"}), \
             patch.object(session, "_request", side_effect=request):
            self.assertEqual(session.call("ingestion", "ingest")["job_id"], "accepted")
        self.assertEqual(held, [])

    def test_unused_llm_does_not_trigger_external_unload_and_failure_can_retry(self):
        adapter = Mock()
        adapter.generate.side_effect = RuntimeError("request failed after load")
        adapter.unload.side_effect = [False, True]
        with patch.object(llm, "_adapter", adapter), patch.object(llm, "_needs_unload", False):
            self.assertTrue(llm.unload())
            adapter.unload.assert_not_called()
            with self.assertRaises(RuntimeError):
                llm.generate("test")
            self.assertFalse(llm.unload())
            self.assertTrue(llm.unload())
            self.assertTrue(llm.unload())
            self.assertEqual(adapter.unload.call_count, 2)


if __name__ == "__main__":
    unittest.main()
