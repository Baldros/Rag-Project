"""CLI behaviors consumed by external MCP clients; no models are loaded."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
import unittest
from unittest.mock import patch

from kb import cli, collection, config, db, jobs, session
from kb.retrieval import fetch_pages
from kb.retrieval.expand import expand_to_sections
from kb.retrieval.search import Hit, fts_query, lexical_search


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


class PageSignalTests(unittest.TestCase):
    """Listagem cheia é indistinguível do fim da lista sem page.has_more."""

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

    def listing(self, *args: str) -> tuple[int, dict]:
        with patch.object(collection, "get_conn", return_value=self.conn),              patch.object(jobs, "get_conn", return_value=self.conn):
            return invoke(*args)

    def test_surplus_row_becomes_has_more(self):
        for command in (("collections", "list"), ("documents", "list"), ("jobs", "list")):
            code, result = self.listing(*command, "--limit", "2")
            self.assertEqual(code, 0, command)
            self.assertEqual(len(result["data"]), 2, command)
            self.assertEqual(result["page"],
                             {"limit": 2, "offset": 0, "has_more": True}, command)

            code, result = self.listing(*command, "--limit", "2", "--offset", "2")
            self.assertEqual(len(result["data"]), 1, command)
            self.assertEqual(result["page"],
                             {"limit": 2, "offset": 2, "has_more": False}, command)

    def test_unbounded_listing_keeps_data_a_list_and_reports_no_more(self):
        code, result = self.listing("collections", "list")
        self.assertEqual(code, 0)
        self.assertEqual(len(result["data"]), 3)
        self.assertEqual(result["data"][0]["collection_id"], "c0")
        self.assertEqual(result["page"], {"limit": None, "offset": 0, "has_more": False})

    def test_human_output_says_where_to_continue(self):
        output = io.StringIO()
        with patch.object(collection, "get_conn", return_value=self.conn),              contextlib.redirect_stdout(output):
            cli.main(["collections", "list", "--limit", "2", "-H"])
        self.assertIn("--offset 2", output.getvalue())


class IncompleteReadTests(unittest.TestCase):
    """Corte de texto é informado em campo, não só no meio do texto."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(db.SCHEMA)
        self.conn.execute(
            "INSERT INTO documents(doc_id, filename, source_path, created_at)"
            " VALUES (?, ?, ?, ?)",
            ("d0", "livro.pdf", "/docs/livro", "2026-01-01"),
        )

    def tearDown(self):
        self.conn.close()

    def add_chunk(self, ord_: int, page: int, size: int) -> None:
        self.conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, ord, page_start, page_end,"
            " heading_path, text, embed_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"k{ord_}", "d0", ord_, page, page, f"Capítulo {page}",
             "x" * size, "x" * size),
        )

    def test_long_range_is_cut_at_a_page_boundary_with_the_next_page(self):
        for ord_, page in enumerate([1, 2, 3, 4]):
            self.add_chunk(ord_, page, 1000)

        first = fetch_pages("d0", 1, 4, max_chars=1500, conn=self.conn)
        self.assertEqual(first["n_chunks"], 2)
        self.assertTrue(first["truncated"])
        self.assertEqual(first["next_page_start"], 3)

        rest = fetch_pages("d0", first["next_page_start"], 4, max_chars=1500,
                           conn=self.conn)
        self.assertEqual(rest["n_chunks"], 2)
        self.assertFalse(rest["truncated"])
        self.assertIsNone(rest["next_page_start"])

    def test_a_single_page_is_never_split_so_reading_always_advances(self):
        for ord_ in range(3):
            self.add_chunk(ord_, 1, 1000)

        result = fetch_pages("d0", 1, 1, max_chars=500, conn=self.conn)
        self.assertEqual(result["n_chunks"], 3)
        self.assertFalse(result["truncated"])
        self.assertIsNone(result["next_page_start"])

    def test_short_range_reports_no_truncation(self):
        self.add_chunk(0, 1, 100)
        result = fetch_pages("d0", 1, conn=self.conn)
        self.assertFalse(result["truncated"])
        self.assertIsNone(result["next_page_start"])

    def test_cut_section_is_flagged_for_a_follow_up_fetch(self):
        self.conn.execute(
            "INSERT INTO sections(section_id, doc_id, ord, heading, heading_path,"
            " page_start, page_end, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s0", "d0", 0, "Campo elétrico", "Campo elétrico", 1, 2, "y" * 400),
        )
        hit = Hit(chunk_id="k0", doc_id="d0", section_id="s0", text="y" * 20,
                  heading_path="Campo elétrico", page_start=1, page_end=1,
                  content_type="text", filename="livro.pdf")

        cut = expand_to_sections([hit], max_chars=100, conn=self.conn)[0].to_dict()
        self.assertTrue(cut["truncated"])
        self.assertEqual(cut["section_id"], "s0")

        whole = expand_to_sections([hit], max_chars=4000, conn=self.conn)[0].to_dict()
        self.assertFalse(whole["truncated"])
        self.assertEqual(len(whole["text"]), 400)


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


class ScopeContractTests(unittest.TestCase):
    """A collection que não existe é erro; a vazia é escopo válido."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(db.SCHEMA)
        self.conn.execute(
            "INSERT INTO collections(collection_id, name, created_at) VALUES (?, ?, ?)",
            ("fisica-3", "Física 3", "2026-01-01"),
        )
        self.conn.execute(
            "INSERT INTO collections(collection_id, name, created_at) VALUES (?, ?, ?)",
            ("vazia", "Sem documentos", "2026-01-01"),
        )
        self.conn.execute(
            "INSERT INTO documents(doc_id, filename, source_path, created_at)"
            " VALUES (?, ?, ?, ?)",
            ("d0", "livro.pdf", "/docs/livro", "2026-01-01"),
        )
        self.conn.execute(
            "INSERT INTO document_collections(doc_id, collection_id, added_at)"
            " VALUES (?, ?, ?)",
            ("d0", "fisica-3", "2026-01-01"),
        )

    def tearDown(self):
        self.conn.close()

    def test_name_instead_of_id_is_not_found_but_empty_collection_is_a_scope(self):
        self.assertIsNone(collection.resolve_scope(conn=self.conn))
        self.assertEqual(collection.resolve_scope(["fisica-3"], conn=self.conn), ["d0"])
        self.assertEqual(collection.resolve_scope(["vazia"], conn=self.conn), [])

        for absent in (["Física 3"], ["fisica3"], ["fisica-3", "fisica3"]):
            with self.assertRaises(session.SessionError) as raised:
                collection.resolve_scope(absent, conn=self.conn)
            self.assertEqual(raised.exception.code, "NOT_FOUND")

    def test_document_listing_refuses_an_unknown_collection(self):
        self.assertEqual(collection.documents_in("vazia", conn=self.conn), [])
        with self.assertRaises(session.SessionError) as raised:
            collection.documents_in("fisica3", conn=self.conn)
        self.assertEqual(raised.exception.code, "NOT_FOUND")

        with patch.object(collection, "get_conn", return_value=self.conn):
            code, result = invoke("documents", "list", "--collection", "fisica3")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")

    def test_writing_to_an_unknown_collection_is_refused(self):
        with patch.object(collection, "get_conn", return_value=self.conn):
            code, result = invoke("collections", "add-docs", "fisica3", "d0")
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")


class ArgumentValidationTests(unittest.TestCase):
    """A CLI é a interface pública: valor inválido falha antes de virar slice."""

    def test_sizes_and_pages_must_be_positive(self):
        for args in (
            ("search", "x", "--top-k", "-1"),
            ("search", "x", "--top-k", "0"),
            ("outline", "x", "--max-depth", "0"),
            ("fetch", "--doc-id", "x", "--page-start", "0"),
            ("fetch", "--doc-id", "x", "--page-start", "1", "--page-end", "-2"),
        ):
            code, result = invoke(*args)
            self.assertEqual(code, 2, args)
            self.assertEqual(result["error"]["code"], "USAGE_ERROR", args)

    def test_search_defaults_follow_the_configured_top_k(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["search", "x"]).top_k, config.FINAL_TOP_K)

    def test_every_argument_is_documented(self):
        parser = cli.build_parser()
        missing = []

        def walk(node, path):
            for action in node._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for name, child in action.choices.items():
                        walk(child, f"{path} {name}")
                elif action.help is None:
                    missing.append(f"{path} {action.dest}")

        walk(parser, "kb")
        self.assertEqual(missing, [])


class LexicalQueryTests(unittest.TestCase):
    """The BM25 channel exists for the exact technical term the vectors miss."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(db.SCHEMA)
        self.conn.execute(
            "INSERT INTO documents(doc_id, filename, source_path, created_at)"
            " VALUES (?, ?, ?, ?)",
            ("d0", "quimica.pdf", "/docs/quimica", "2026-01-01"),
        )
        self.conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, ord, text, embed_text)"
            " VALUES (?, ?, ?, ?, ?)",
            ("k0", "d0", 0, "O pH mede a acidez da solução.", "O pH mede a acidez."),
        )
        self.conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, ord, text, embed_text)"
            " VALUES (?, ?, ?, ?, ?)",
            ("k1", "d0", 1, "A lei de Ohm vale em CA e em CC.", "A lei de Ohm."),
        )

    def tearDown(self):
        self.conn.close()

    def test_acronyms_and_symbols_survive_the_short_word_cutoff(self):
        self.assertEqual(fts_query("pH da solução tampão"),
                         '"ph" OR "solucao" OR "tampao"')
        self.assertEqual(fts_query("lei de Ohm em CA e CC"),
                         '"lei" OR "ohm" OR "ca" OR "cc"')
        self.assertEqual(fts_query("raios UV e luz IR"),
                         '"raios" OR "uv" OR "luz" OR "ir"')
        self.assertEqual(fts_query("5G vs 4G"), '"5g" OR "4g"')
        # Palavra curta em minúscula continua fora, e o termo repetido entra uma vez.
        self.assertEqual(fts_query("de da e em"), "")
        self.assertEqual(fts_query("campo e campo"), '"campo"')

    def test_short_term_finds_the_indexed_chunk(self):
        self.assertEqual(
            [chunk_id for chunk_id, _ in lexical_search("pH", 10, None, conn=self.conn)],
            ["k0"],
        )
        self.assertEqual(
            [chunk_id for chunk_id, _ in lexical_search("CC", 10, None, conn=self.conn)],
            ["k1"],
        )
        # Sem termo nenhum o canal léxico não roda e a busca segue só vetorial.
        self.assertEqual(lexical_search("ok", 10, None, conn=self.conn), [])


if __name__ == "__main__":
    unittest.main()
