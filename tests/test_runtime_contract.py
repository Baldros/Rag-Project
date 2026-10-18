"""Fast contract tests; no model or GPU is loaded."""

from __future__ import annotations

import contextlib
import io
import json
import os
import secrets
import sys
import threading
import time
import unittest
from unittest.mock import patch

from kb import cli, config, providers, runtime, session
from kb.models import ModelNotLoaded, ModelSlot


class ModelContractTests(unittest.TestCase):
    def test_model_never_loads_from_get(self):
        loaded = []
        slot = ModelSlot("fake", lambda: loaded.append(1) or object())
        with self.assertRaises(ModelNotLoaded):
            slot.get()
        self.assertEqual(loaded, [])
        slot.load()
        slot.get()
        self.assertEqual(loaded, [1])

    def test_unknown_provider_has_no_fallback(self):
        with patch.object(config, "EMBED_PROVIDER", "unknown"):
            with self.assertRaisesRegex(ValueError, "unknown"):
                providers.embedding_adapter()
        with patch.object(config, "LLM_PROVIDER", "unknown"):
            with self.assertRaisesRegex(ValueError, "unknown"):
                providers.generation_adapter()


class CliContractTests(unittest.TestCase):
    def test_search_off_returns_structured_error_without_importing_torch(self):
        output = io.StringIO()
        with patch.object(session, "_read_state", return_value=None), \
             contextlib.redirect_stdout(output):
            code = cli.main(["search", "teste"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "SESSION_OFF")
        self.assertNotIn("torch", sys.modules)

    def test_models_are_reported_without_loading_adapters(self):
        output = io.StringIO()
        with patch.object(session, "status", return_value={"active": False}), \
             contextlib.redirect_stdout(output):
            code = cli.main(["models"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["data"]["generation"]["provider"], "ollama")
        self.assertNotIn("torch", sys.modules)


class PipeContractTests(unittest.TestCase):
    def test_windows_venv_launcher_pid_can_differ_from_runtime_pid(self):
        state = {"pid": 111, "nonce": "test"}
        with patch.object(session, "_request", return_value={"pid": 222, "mode": "interaction"}):
            self.assertEqual(session._probe(state)["pid"], 222)

    def test_private_pipe_identifies_session_and_shuts_down(self):
        token = secrets.token_hex(32)
        nonce = secrets.token_hex(16)
        address = session._address()
        state = {"authkey": token, "address": address, "nonce": nonce,
                 "mode": "interaction", "pid": os.getpid()}
        errors = []

        def server():
            try:
                with patch.object(runtime.Runtime, "prepare"), \
                     patch.object(runtime.Runtime, "close"), \
                     patch.object(sys, "argv", ["kb.runtime", "--mode", "interaction",
                                                "--address", address, "--nonce", nonce]), \
                     patch.dict(os.environ, {"KB_SESSION_AUTHKEY": token}):
                    runtime.main()
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        for _ in range(100):
            if session._probe(state):
                break
            time.sleep(0.05)
        else:
            self.fail(f"Pipe did not start: {errors}")
        self.assertEqual(session._request(state, "shutdown"), {"active": False})
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


class RuntimeModeTests(unittest.TestCase):
    def test_ingestion_mode_rejects_search_without_loading_models(self):
        worker = runtime.Runtime("ingestion", "nonce")
        with self.assertRaises(session.SessionError) as error:
            worker.handle({"action": "search", "payload": {"query": "x"}})
        self.assertEqual(error.exception.code, "WRONG_MODE")
        self.assertNotIn("torch", sys.modules)

    def test_shutdown_refuses_active_job(self):
        worker = runtime.Runtime("ingestion", "nonce")
        worker.job_id = "existing-job"
        with self.assertRaises(session.SessionError) as error:
            worker.handle({"action": "shutdown", "payload": {}})
        self.assertEqual(error.exception.code, "BUSY")


class TransitionTests(unittest.TestCase):
    def test_finished_job_during_prompt_is_not_cancelled(self):
        state = {"pid": 10, "nonce": "n", "address": "unused", "authkey": "00"}
        with patch.object(session, "_read_state", return_value=state), \
             patch.object(session, "_probe", side_effect=[
                 {"pid": 10, "mode": "ingestion", "job_id": "job"},
                 {"pid": 10, "mode": "ingestion", "job_id": None},
                 {"pid": 10, "mode": "ingestion", "job_id": None},
                 None,
             ]), \
             patch.object(session, "_decision", return_value=True), \
             patch.object(session, "_request", return_value={"active": False}) as request, \
             patch.object(session, "_process_target", return_value=contextlib.nullcontext(None)), \
             patch.object(session, "_await_exit", return_value=True), \
             patch.object(type(session.STATE_PATH), "unlink"), \
             patch.object(session.os, "kill") as kill:
            result = session._stop_locked(cancel=False, interactive=True)
        kill.assert_not_called()
        request.assert_called_once_with(state, "shutdown")
        self.assertIsNone(result["cancelled_job"])


class IndexIdentityTests(unittest.TestCase):
    def test_different_embedding_requires_full_reindex(self):
        from kb import vectors

        class FakeIndex:
            def embedding_identity(self):
                return "different-provider:model"

        with patch.object(vectors, "_index", FakeIndex()):
            with self.assertRaisesRegex(RuntimeError, "reindex --reset"):
                vectors.get_index()


if __name__ == "__main__":
    unittest.main()
