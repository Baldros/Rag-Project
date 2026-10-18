"""CLI behaviors consumed by external MCP clients; no models are loaded."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import unittest
from unittest.mock import patch

from kb import cli, collection, db, jobs, session


def invoke(*args: str) -> tuple[int, dict]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            code = cli.main(list(args))
        except SystemExit as exc:
            code = int(exc.code)
    return code, json.loads(output.getvalue())


class CatalogPaginationTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(db.SCHEMA)
        for index in range(3):
            self.conn.execute(
                "INSERT INTO collections(collection_id, name, created_at) VALUES (?, ?, ?)",
                (f"c{index}", f"Collection {index}", "2026-01-01"),
            )
            self.conn.execute(
                "INSERT INTO documents(doc_id, filename, source_path, created_at)"
                " VALUES (?, ?, ?, ?)",
                (f"d{index}", f"document-{index}.pdf", f"/docs/{index}", "2026-01-01"),
            )
            self.conn.execute(
                "INSERT INTO jobs(job_id, kind, status, created_at) VALUES (?, ?, ?, ?)",
                (f"j{index}", "ingest", "done", f"2026-01-0{index + 1}"),
            )

    def tearDown(self):
        self.conn.close()

    def test_unbounded_catalog_remains_compatible_and_pages_are_stable(self):
        self.assertEqual(len(collection.list_all(self.conn)), 3)
        self.assertEqual(len(collection.documents_in(conn=self.conn)), 3)
        self.assertEqual([row["collection_id"] for row in
                          collection.list_all(self.conn, limit=1, offset=1)], ["c1"])
        self.assertEqual([row["doc_id"] for row in
                          collection.documents_in(conn=self.conn, limit=1, offset=1)], ["d1"])
        self.assertEqual([row["job_id"] for row in
                          jobs.list_jobs(1, self.conn, offset=1)], ["j1"])

    def test_cli_list_flags_and_invalid_page_size(self):
        with patch.object(collection, "get_conn", return_value=self.conn):
            code, result = invoke("collections", "list", "--limit", "1", "--offset", "2")
        self.assertEqual(code, 0)
        self.assertEqual(result["data"][0]["collection_id"], "c2")
        code, result = invoke("documents", "list", "--limit", "0")
        self.assertEqual(code, 2)
        self.assertEqual(result["error"]["code"], "USAGE_ERROR")


class BridgeErrorTests(unittest.TestCase):
    def test_argument_boundary_keeps_positional_values_as_data(self):
        parser = cli.build_parser()
        search = parser.parse_args(["search", "--collection=Physics A", "--", "--query"])
        self.assertEqual(search.collection, ["Physics A"])
        self.assertEqual(search.query, "--query")
        ingest = parser.parse_args(["ingest", "--background", "--", "E:/My PDFs/book.pdf"])
        self.assertEqual(ingest.paths, ["E:/My PDFs/book.pdf"])
        create = parser.parse_args(["collections", "create", "--name=Book A", "--", "--id"])
        self.assertEqual(create.collection, "--id")
        add = parser.parse_args(["collections", "add-docs", "--", "c", "--doc"])
        self.assertEqual(add.doc_ids, ["--doc"])

    def test_missing_metadata_is_a_cli_error(self):
        with patch("kb.retrieval.get_outline", return_value={"error": "missing"}):
            code, result = invoke("outline", "absent")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")

        with patch.object(collection, "get", return_value=None):
            code, result = invoke("collections", "delete", "absent")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")

        with patch.object(collection, "delete_document", return_value={"ok": False}):
            code, result = invoke("documents", "delete", "absent")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")

        with patch("kb.retrieval.fetch_pages", return_value={"filename": None}):
            code, result = invoke("fetch", "--doc-id", "absent", "--page-start", "1")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")

    def test_nonblocking_shutdown_preserves_running_job(self):
        state = {"nonce": "test"}
        with patch.object(session, "_read_state", return_value=state), \
             patch.object(session, "_probe", return_value={"job_id": "running-job"}), \
             patch.object(session, "_request") as request:
            with self.assertRaises(session.SessionError) as raised:
                session._stop_locked(cancel=False, interactive=False, fail_if_busy=True)
        self.assertEqual(raised.exception.code, "BUSY")
        request.assert_not_called()

    def test_cli_no_wait_passes_busy_policy(self):
        with patch.object(session, "stop", side_effect=session.SessionError(
            "BUSY", "Job running", "Check job")) as stop:
            code, result = invoke("off", "--no-wait")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"], {
            "code": "BUSY", "message": "Job running", "hint": "Check job",
        })
        stop.assert_called_once_with(cancel=False, interactive=False, fail_if_busy=True)


if __name__ == "__main__":
    unittest.main()
