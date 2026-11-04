"""Regression boundaries for publication, recovery and extraction reporting."""
import contextlib
import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kb import cli, config, db, health, jobs, quality, session
from kb.ingest import embed, parse, pipeline
from kb.ingest.segment import ChunkRecord, SectionRecord
from kb.retrieval import fetch_section, fetch_pages
from kb.retrieval.search import lexical_search


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class FakeDocument:
    def __init__(self, pages, formula=False):
        self.pages = dict.fromkeys(pages, {})
        self.formula = formula

    def iterate_items(self):
        if self.formula:
            yield SimpleNamespace(label=SimpleNamespace(value="formula"), text="",
                                  self_ref="#/texts/0", prov=[SimpleNamespace(page_no=min(self.pages))]), 0

    def save_as_json(self, path):
        path.write_text(json.dumps({"schema_name": "DoclingDocument", "pages": self.pages,
                                    "formula": self.formula}), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(map(int, data["pages"]), data["formula"])


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(folder)
        self.pdf = self.root / "Inferência Estatística.pdf"
        self.pdf.write_bytes(b"pdf-content")
        self.stack.enter_context(patch.object(parse, "ARTIFACTS_DIR", self.root / "artifacts"))
        self.stack.enter_context(patch.object(parse, "PAGE_BLOCK_SIZE", 1))
        self.stack.enter_context(patch.object(parse, "get_pdf_page_count", return_value=2))
        self.stack.enter_context(patch.object(parse, "force_gc"))
        self.converter = Mock()
        self.converter.convert.side_effect = self.convert
        self.stack.enter_context(patch.object(parse, "get_converter", return_value=self.converter))
        self.stack.enter_context(patch.dict("sys.modules", {
            "docling.datamodel.base_models": SimpleNamespace(DocumentStream=SimpleNamespace, ConversionStatus=Status),
            "docling_core.types.doc": SimpleNamespace(DoclingDocument=FakeDocument),
        }))

    def convert(self, source, page_range, **kwargs):
        self.assertEqual(source.stream.getvalue(), b"pdf-content")
        self.assertEqual(source.name, self.pdf.name)
        return SimpleNamespace(status=Status.SUCCESS, errors=[], document=FakeDocument(range(page_range[0], page_range[1] + 1)))

    def test_unicode_stream_and_complete_generation_do_not_publish_implicitly(self):
        manifest = parse.parse_document(self.pdf, "doc")
        self.assertTrue(parse.validate_manifest("doc", manifest))
        self.assertEqual(manifest["quality"]["status"], "complete")
        self.assertFalse(parse.manifest_path("doc").exists())
        self.assertEqual(self.converter.convert.call_count, 2)

    def test_failed_status_with_document_is_not_success_and_retry_reuses_good_block(self):
        def fail_second(source, page_range, **kwargs):
            result = self.convert(source, page_range)
            if page_range[0] == 2:
                result.status = Status.FAILURE
            return result
        self.converter.convert.side_effect = fail_second
        with self.assertRaises(parse.ParseError) as raised:
            parse.parse_document(self.pdf, "doc")
        self.assertFalse(raised.exception.manifest["complete"])
        self.converter.convert.side_effect = self.convert
        self.converter.convert.reset_mock()
        result = parse.parse_document(self.pdf, "doc")
        self.assertTrue(result["complete"])
        self.assertEqual(self.converter.convert.call_count, 1)
        self.assertEqual(self.converter.convert.call_args.kwargs["page_range"], (2, 2))

    def test_missing_page_is_fatal_even_with_success_status(self):
        self.converter.convert.return_value = SimpleNamespace(status=Status.SUCCESS, errors=[], document=FakeDocument([]))
        self.converter.convert.side_effect = None
        with self.assertRaises(parse.ParseError):
            parse.parse_document(self.pdf, "doc")

    def test_corrupt_cached_block_is_reconverted(self):
        result = parse.parse_document(self.pdf, "doc")
        path = parse._path("doc", result["blocks"][0]["file"])
        path.write_text("{}")
        self.assertFalse(parse.validate_manifest("doc", result))
        self.converter.convert.reset_mock()
        self.assertTrue(parse.parse_document(self.pdf, "doc")["complete"])
        self.assertEqual(self.converter.convert.call_count, 1)

    def test_formula_retry_is_bounded_and_cached_layout_is_reused(self):
        def formulas(source, page_range, **kwargs):
            result = self.convert(source, page_range)
            result.document.formula = True
            return result
        self.converter.convert.side_effect = formulas
        with patch.object(parse, "_enrich_existing", return_value=["inference failed"]) as retry:
            manifest = parse.parse_document(self.pdf, "doc")
            self.assertEqual(retry.call_count, 2)  # one per block, never an unbounded retry
            self.assertEqual(manifest["quality"]["status"], "partial")
            self.assertEqual(manifest["quality"]["formula_missing"], 2)
            self.converter.convert.reset_mock()
            retry.reset_mock()
            parse.parse_document(self.pdf, "doc")
            self.converter.convert.assert_not_called()
            self.assertEqual(retry.call_count, 2)

    def test_segment_settings_do_not_invalidate_parse(self):
        before_parse, before_segment = config.parse_version(), config.segment_version()
        with patch.object(config, "CHUNK_MAX_TOKENS", config.CHUNK_MAX_TOKENS + 1):
            self.assertEqual(config.parse_version(), before_parse)
            self.assertNotEqual(config.segment_version(), before_segment)

    def test_unbalanced_formula_is_reported_even_when_nonempty(self):
        item = SimpleNamespace(text=r"\frac{x}{y")
        self.assertIn("desequilibradas", parse._formula_problem(item))
        item.text = r"\left\{ x \right\} + \frac{x}{y}"
        self.assertIsNone(parse._formula_problem(item))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        db._migrate(self.conn)
        self.addCleanup(self.conn.close)
        pipeline.register_document("doc", Path("old.pdf"), 2, self.conn)

    def store(self, text="Estat´ ısticas suficientes", count=1):
        section = SectionRecord("s", "doc", None, 0, 1, "heading", "heading", 1, 2, [text])
        chunks = [ChunkRecord(f"c{i}", "doc", "s", i, 1, 2, 10, "heading", text, text, "text") for i in range(count)]
        pipeline.persist_segments("doc", [section], chunks, conn=self.conn)

    def test_normalization_is_lexical_only_and_fts_update_delete_stays_consistent(self):
        self.store()
        row = self.conn.execute("SELECT text, search_text FROM chunks").fetchone()
        self.assertEqual(row["text"], "Estat´ ısticas suficientes")
        self.assertEqual(row["search_text"], "Estatísticas suficientes")
        self.assertTrue(lexical_search("estatísticaS", 5, None, self.conn))
        self.conn.execute("UPDATE chunks SET text = 'Introdu¸ c˜ ao' WHERE chunk_id = 'c0'")
        self.assertFalse(lexical_search("estatísticas", 5, None, self.conn))
        self.assertTrue(lexical_search("introdução", 5, None, self.conn))
        self.conn.execute("DELETE FROM chunks")
        self.assertFalse(lexical_search("introdução", 5, None, self.conn))

    def test_formula_and_code_are_not_normalized(self):
        text = r"$xˆ i + c¸$ `Estat´ ısticas` $$a˜ o$$"
        self.assertEqual(quality.normalize_lexical(text), text)
        self.assertEqual(quality.normalize_lexical("Mˆ onica e Introdu¸ c˜ ao"), "Mônica e Introdução")
        self.assertEqual(quality.normalize_lexical("INTRODUC ¸ ˜ AO"), "INTRODUÇÃO")
        self.assertEqual(quality.normalize_lexical("uma ´ arvore e v´ arias folhas"), "uma árvore e várias folhas")

    def test_nested_publication_rolls_back_chunks_and_flags_together(self):
        self.store("original")
        with self.assertRaises(RuntimeError):
            with db.transaction(self.conn):
                self.store("replacement", 2)
                raise RuntimeError("association failed")
        self.assertEqual(self.conn.execute("SELECT text FROM chunks").fetchall()[0][0], "original")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)

    def test_failed_parse_keeps_indexed_document_and_summary(self):
        self.store("original")
        self.conn.execute("UPDATE documents SET status='indexed', summary='valid', vector_reset_pending=0")
        with patch.object(pipeline, "expand_paths", return_value=[Path("new.pdf")]), \
             patch.object(pipeline, "file_sha256", return_value="doc"), \
             patch.object(pipeline, "get_pdf_page_count", return_value=2), \
             patch.object(pipeline, "parse_document", side_effect=RuntimeError("bad PDF")):
            result = pipeline.ingest_paths(["new.pdf"], conn=self.conn)
        self.assertEqual(len(result["failed"]), 1)
        row = self.conn.execute("SELECT * FROM documents").fetchone()
        self.assertEqual((row["status"], row["summary"], row["source_path"]), ("indexed", "valid", "old.pdf"))
        self.assertEqual(self.conn.execute("SELECT text FROM chunks").fetchone()[0], "original")

    def test_vector_cleanup_is_retried_before_embedding(self):
        self.store(count=2)
        index = Mock()
        index.delete_docs.side_effect = RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            embed.cleanup_pending_vectors(self.conn, index, ["doc"])
        self.assertEqual(self.conn.execute("SELECT vector_reset_pending FROM documents").fetchone()[0], 1)
        index.delete_docs.side_effect = None
        order = []
        index.delete_docs.side_effect = lambda ids: order.append("delete")
        index.upsert.side_effect = lambda **kwargs: order.append("upsert")
        with patch.object(embed, "get_index", return_value=index), patch.object(embed, "EMBEDDER"), \
             patch.object(embed, "embed_texts", return_value=[[1.0], [2.0]]), patch.object(embed, "unload_all"):
            self.assertEqual(embed.embed_pending(["doc"], conn=self.conn), 2)
        self.assertEqual(order, ["delete", "upsert"])
        self.assertEqual(self.conn.execute("SELECT status FROM documents").fetchone()[0], "indexed")

    def test_embedding_failure_never_reports_the_prepared_document_as_indexed(self):
        section = SectionRecord("s", "doc", None, 0, 1, "heading", "heading", 1, 2, ["new"])
        chunk = ChunkRecord("c", "doc", "s", 0, 1, 2, 10, "heading", "new", "new", "text")
        manifest = {"manifest_file": "generation/manifest.json", "quality": {"status": "complete"},
                    "filename": "new.pdf", "source_path": "new.pdf", "n_pages": 2}
        with patch.object(pipeline, "expand_paths", return_value=[Path("new.pdf")]), \
             patch.object(pipeline, "file_sha256", return_value="doc"), \
             patch.object(pipeline, "get_pdf_page_count", return_value=2), \
             patch.object(pipeline, "parse_document", return_value=manifest), \
             patch.object(pipeline, "segment_document", return_value=([section], [chunk])), \
             patch.object(pipeline, "export_markdown"), patch.object(pipeline, "activate_manifest"), \
             patch.object(pipeline, "embed_pending", side_effect=RuntimeError("GPU failed")):
            result = pipeline.ingest_paths(["new.pdf"], conn=self.conn, enrich=False)
        self.assertEqual(result["indexed"], [])
        self.assertEqual(result["failed"][0]["phase"], "embed")
        self.assertEqual(jobs.result_status(result), "failed")
        self.assertEqual(self.conn.execute("SELECT vector_reset_pending FROM documents").fetchone()[0], 1)

    def test_cli_partial_job_stops_waiting_and_points_to_details(self):
        with patch.object(jobs, "get_job", return_value={"status": "partial"}):
            with self.assertRaises(session.SessionError) as raised:
                cli._wait_job("job123")
        self.assertEqual(raised.exception.code, "JOB_PARTIAL")
        self.assertIn("job123", raised.exception.hint)

    def test_existing_summary_does_not_require_an_available_llm(self):
        from kb import enrich, llm
        self.store()
        self.conn.execute("UPDATE documents SET summary='already summarized'")
        with patch.object(llm, "is_available", return_value=False) as available:
            result = enrich.run_pass(["doc"], sections=False, conn=self.conn)
        available.assert_not_called()
        self.assertEqual(result, {"documents": 0, "sections": 0, "failed": 0, "skipped": None})

    def test_equal_vector_counts_do_not_hide_wrong_ids(self):
        self.store()
        self.conn.execute("UPDATE chunks SET embedded=1")
        with patch("kb.vectors.get_index", return_value=SimpleNamespace(ids=lambda: {"orphan"})):
            report = health.consistency(self.conn)
        self.assertFalse(report["ok"])
        self.assertEqual((report["missing_count"], report["orphan_count"], report["drift"]), (1, 1, 0))

    def test_quality_reaches_fetch_and_jobs_retain_full_results(self):
        self.store()
        details = {"status": "partial", "issues": [{"pages": [2], "message": "missing"}], "warnings": ["gap"]}
        self.conn.execute("UPDATE documents SET quality_json=?", (json.dumps(details),))
        self.assertEqual(fetch_section("s", self.conn)["quality"]["status"], "partial")
        self.assertEqual(fetch_pages("doc", 2, conn=self.conn)["quality"]["status"], "partial")
        result = {"indexed": [{"doc_id": "doc", "quality": details}], "failed": [], "skipped": [], "long": "x" * 5000}
        job = jobs.create_job("ingest", "source", conn=self.conn)
        jobs.update_job(job, conn=self.conn, status=jobs.result_status(result), result=result)
        actual = jobs.get_job(job, self.conn)
        self.assertEqual(actual["status"], "partial")
        self.assertEqual(actual["result"], result)
        self.assertEqual(jobs.result_status({"indexed": [], "skipped": [{}], "failed": [{}]}), "partial")
        self.assertEqual(jobs.result_status({"indexed": [], "skipped": [], "failed": [{}]}), "failed")


    def test_partial_reingestion_without_force_retries_and_keeps_prior_data_on_parse_failure(self):
        self.store("published text")
        self.conn.execute("UPDATE documents SET status='indexed', summary='saved', quality_json=?, vector_reset_pending=0",
                          (json.dumps({"status": "partial", "issues": [{"pages": [2]}]}),))
        self.conn.execute("UPDATE chunks SET embedded=1")
        self.assertFalse(pipeline.is_indexed("doc", self.conn))
        with patch.object(pipeline, "expand_paths", return_value=[Path("same.pdf")]), \
             patch.object(pipeline, "file_sha256", return_value="doc"), \
             patch.object(pipeline, "get_pdf_page_count", return_value=2), \
             patch.object(pipeline, "parse_document", side_effect=RuntimeError("retry failed")) as parse_pdf:
            result = pipeline.ingest_paths(["same.pdf"], conn=self.conn)
        self.assertFalse(parse_pdf.call_args.kwargs["force"])
        self.assertEqual(result["failed"][0]["phase"], "parse")
        self.assertEqual(result["pending"][0]["phase"], "extraction")
        self.assertEqual(self.conn.execute("SELECT summary FROM documents").fetchone()[0], "saved")
        self.assertEqual(tuple(self.conn.execute("SELECT text, embedded FROM chunks").fetchone()),
                         ("published text", 1))

    def test_unfilled_summary_is_partial_even_without_reported_generation_error(self):
        from kb import enrich
        self.store()
        self.conn.execute("UPDATE documents SET status='indexed', vector_reset_pending=0, quality_json=?",
                          (json.dumps({"status": "complete"}),))
        self.conn.execute("UPDATE chunks SET embedded=1")
        job_id = jobs.create_job("ingest", "same.pdf", conn=self.conn)
        jobs.update_job(job_id, status="running", conn=self.conn)

        def checkpoint(snapshot):
            jobs.update_job(job_id, result=snapshot, conn=self.conn)

        def enrichment(**kwargs):
            self.assertEqual(self.conn.execute("SELECT status FROM documents").fetchone()[0], "indexed")
            job = jobs.get_job(job_id, self.conn)
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["result"]["pending"][0]["phase"], "enrich")
            return {"documents": 0, "sections": 0, "failed": 0}

        with patch.object(pipeline, "expand_paths", return_value=[Path("same.pdf")]), \
             patch.object(pipeline, "file_sha256", return_value="doc"), \
             patch.object(pipeline, "is_indexed", return_value=True), \
             patch.object(pipeline, "embed_pending", return_value=0), \
             patch.object(enrich, "run_pass", side_effect=enrichment):
            result = pipeline.ingest_paths(["same.pdf"], conn=self.conn, on_checkpoint=checkpoint)
        self.assertEqual(jobs.result_status(result), "partial")
        self.assertEqual(result["pending_inputs"], [])
        self.assertTrue(result["pending"][0]["document_summary"])


class MigrationTests(unittest.TestCase):
    def test_real_v1_migration_preserves_sources_summaries_vectors_and_collections(self):
        with contextlib.closing(db.connect(":memory:")) as conn:
            conn.executescript((Path(__file__).parent / "schema_v1.sql").read_text(encoding="utf-8"))
            conn.execute("PRAGMA user_version = 1")
            conn.execute("INSERT INTO documents(doc_id, filename, source_path, created_at, summary, status) VALUES ('d', 'a.pdf', 'original', 'now', 'summary', 'indexed')")
            conn.execute("INSERT INTO collections VALUES ('c', 'Collection', NULL, 'now')")
            conn.execute("INSERT INTO document_collections VALUES ('d', 'c', 'now')")
            conn.execute("INSERT INTO chunks(chunk_id, doc_id, ord, text, embed_text, embedded) VALUES ('k', 'd', 0, 'Estat´ ısticas', 'Estat´ ısticas', 1)")
            db._migrate(conn)
            self.assertEqual(db._schema_version(conn), db.SCHEMA_VERSION)
            self.assertEqual(tuple(conn.execute("SELECT text, search_text, embedded FROM chunks").fetchone()),
                             ("Estat´ ısticas", "Estatísticas", 1))
            self.assertEqual(tuple(conn.execute("SELECT source_path, summary, status, quality_json FROM documents").fetchone()),
                             ("original", "summary", "indexed", None))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM document_collections").fetchone()[0], 1)
            self.assertTrue(lexical_search("estatísticas", 5, None, conn))


if __name__ == "__main__":
    unittest.main()
