"""Manual LaTeX corrections: what they reach, what they keep and what they survive."""
import contextlib
import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kb import cli, db, enrich, runtime
from kb.ingest import embed, parse, pipeline
from kb.ingest.segment import ChunkRecord, SectionRecord
from kb.retrieval.search import lexical_search
from kb.utils import file_sha256

BROKEN = r"\begin{array} {"
FIXED = r"\begin{array}{rcl} x & = & 0 \end{array}"
BOX = [100.0, 300.0, 300.0, 250.0]


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class FakeDocument:
    def __init__(self, pages, items=()):
        self.pages = dict.fromkeys(pages, {})
        self.items = list(items)

    def iterate_items(self):
        for item in self.items:
            yield item, 0

    def export_to_markdown(self):
        return "\n".join(f"$${item.text}$$" for item in self.items)

    def save_as_json(self, path):
        path.write_text(json.dumps({
            "schema_name": "DoclingDocument", "pages": self.pages,
            "items": [{"ref": i.self_ref, "text": i.text, "page": i.prov[0].page_no,
                       "box": [i.prov[0].bbox.l, i.prov[0].bbox.t, i.prov[0].bbox.r, i.prov[0].bbox.b]}
                      for i in self.items],
        }), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(map(int, data["pages"]), [_formula(i["ref"], i["text"], i["page"], i["box"]) for i in data["items"]])


def _formula(ref, text, page, box=BOX):
    l, t, r, b = box
    return SimpleNamespace(label=SimpleNamespace(value="formula"), self_ref=ref, text=text,
                           prov=[SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=l, t=t, r=r, b=b))])


class CorrectionTests(unittest.TestCase):
    """Two one-page blocks; page 2 holds the flagged formula and page 1 a good one."""

    filler_chunks = 0

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(folder)
        self.pdf = self.root / "livro.pdf"
        self.pdf.write_bytes(b"pdf-content")
        self.broken = BROKEN
        self.box = BOX
        self.stack.enter_context(patch.object(parse, "ARTIFACTS_DIR", self.root / "artifacts"))
        self.stack.enter_context(patch.object(parse, "PAGE_BLOCK_SIZE", 1))
        self.stack.enter_context(patch.object(parse, "get_pdf_page_count", return_value=2))
        self.stack.enter_context(patch.object(parse, "force_gc"))
        self.stack.enter_context(patch.object(parse, "_enrich_existing", return_value=[]))
        converter = Mock()
        converter.convert.side_effect = self.convert
        self.converter = converter
        self.stack.enter_context(patch.object(parse, "get_converter", return_value=converter))
        self.stack.enter_context(patch.dict("sys.modules", {
            "docling.datamodel.base_models": SimpleNamespace(DocumentStream=SimpleNamespace, ConversionStatus=Status),
            "docling_core.types.doc": SimpleNamespace(DoclingDocument=FakeDocument),
        }))
        self.stack.enter_context(patch.object(pipeline, "segment_document", side_effect=self.segment))
        self.index = Mock()
        self.stack.enter_context(patch.object(embed, "get_index", return_value=self.index))
        self.stack.enter_context(patch.object(embed, "EMBEDDER"))
        self.stack.enter_context(patch.object(embed, "embed_texts", side_effect=lambda texts, **_: [[0.0]] * len(texts)))
        self.conn = db.connect(":memory:")
        db._migrate(self.conn)
        self.addCleanup(self.conn.close)
        self.publish()

    def convert(self, source, page_range, **kwargs):
        page = page_range[0]
        items = [_formula("#/texts/0", "x = 1" if page == 1 else self.broken, page, self.box if page == 2 else BOX)]
        return SimpleNamespace(status=Status.SUCCESS, errors=[], document=FakeDocument([page], items))

    def segment(self, doc_id, manifest):
        texts = [f"preâmbulo {i}" for i in range(self.filler_chunks)]
        texts += [f"$${doc.items[0].text}$$" for _, _, doc in parse.iter_blocks(doc_id, manifest)]
        section = SectionRecord("s", doc_id, None, 0, 1, "h", "h", 1, 2, list(texts))
        chunks = [ChunkRecord(f"c{i}", doc_id, "s", i, 1, 2, 10, "h", text, text, "text") for i, text in enumerate(texts)]
        return [section], chunks

    def publish(self):
        manifest = parse.parse_document(self.pdf, "doc")
        pipeline.register_document("doc", self.pdf, 2, self.conn)
        sections, chunks = self.segment("doc", manifest)
        pipeline.persist_segments("doc", sections, chunks, conn=self.conn, manifest=manifest)
        parse.activate_manifest("doc", manifest)
        embed.embed_pending(doc_ids=["doc"], conn=self.conn, unload_when_done=False)
        self.conn.execute("UPDATE documents SET summary = 'Resumo antigo', key_topics = '[\"a\"]' WHERE doc_id = 'doc'")
        self.manifest = manifest

    def apply(self, text=FIXED, expected=BROKEN, ref="#/texts/0", page=2):
        return pipeline.apply_formula_corrections(
            "doc", [{"ref": ref, "page": page, "text": text, "expected": expected}], conn=self.conn)

    def current_manifest(self):
        row = self.conn.execute("SELECT artifact_manifest FROM documents WHERE doc_id = 'doc'").fetchone()
        return parse.read_manifest("doc", row["artifact_manifest"])

    def doc(self):
        return self.conn.execute("SELECT * FROM documents WHERE doc_id = 'doc'").fetchone()

    def chunk_texts(self):
        return [row[0] for row in self.conn.execute("SELECT text FROM chunks ORDER BY ord")]

    def test_starting_point_is_flagged(self):
        self.assertEqual(self.manifest["quality"]["status"], "partial")
        self.assertEqual(self.manifest["quality"]["formula_flagged"], 1)

    def test_correction_reaches_every_copy_and_clears_the_flag(self):
        result = self.apply()
        self.assertEqual(len(result["applied"]), 1)
        self.assertEqual(result["rejected"], [])
        self.assertIn(f"$${FIXED}$$", self.chunk_texts())
        section = self.conn.execute("SELECT text FROM sections").fetchone()[0]
        self.assertIn(FIXED, section)
        self.assertNotIn(BROKEN + "$$", section)
        self.assertTrue(lexical_search("rcl", 5, None, self.conn))
        manifest = self.current_manifest()
        markdown = parse._path("doc", str(Path(manifest["manifest_file"]).parent / "document.md"))
        self.assertIn(FIXED, markdown.read_text(encoding="utf-8"))
        quality = json.loads(self.doc()["quality_json"])
        self.assertEqual((quality["status"], quality["formula_flagged"], quality["formula_corrected"]), ("complete", 0, 1))
        self.assertEqual(parse.read_manifest("doc")["manifest_file"], manifest["manifest_file"])
        self.assertEqual(parse.read_manifest("doc", "candidate.json")["manifest_file"], manifest["manifest_file"])

    def test_vectors_are_rebuilt_for_the_document(self):
        self.index.reset_mock()
        result = self.apply()
        self.index.delete_docs.assert_called_once_with(["doc"])
        self.assertEqual(result["n_embedded"], 2)
        self.assertEqual(self.doc()["status"], "indexed")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 0").fetchone()[0], 0)

    def test_block_file_keeps_the_models_output_and_its_hash(self):
        block = self.manifest["blocks"][1]
        before = file_sha256(parse._path("doc", block["file"]))
        self.apply()
        manifest = self.current_manifest()
        self.assertTrue(parse.validate_manifest("doc", manifest))
        self.assertEqual(manifest["blocks"][1]["file"], block["file"])
        self.assertEqual(file_sha256(parse._path("doc", block["file"])), before)
        correction = manifest["corrections"][0]
        self.assertEqual((correction["original"], correction["text"], correction["page"]), (BROKEN, FIXED, 2))
        self.assertTrue(parse.is_parsed("doc", manifest=manifest))

    def test_summary_is_kept_and_reported_as_affected_inside_the_sample(self):
        result = self.apply()
        self.assertTrue(result["summary_affected"])
        self.assertEqual((self.doc()["summary"], self.doc()["key_topics"]), ("Resumo antigo", '["a"]'))

    def test_new_conversion_keeps_the_correction(self):
        self.apply()
        previous = self.current_manifest()
        for force in (False, True):
            with self.subTest(force=force):
                manifest = parse.parse_document(self.pdf, "doc", force=force, previous=previous)
                self.assertEqual(manifest["quality"]["formula_flagged"], 0)
                self.assertEqual([c["text"] for c in manifest["corrections"]], [FIXED])
                texts = [doc.items[0].text for _, _, doc in parse.iter_blocks("doc", manifest)]
                self.assertEqual(texts, ["x = 1", FIXED])

    def test_formula_that_moved_is_not_overwritten_and_the_text_is_kept(self):
        self.apply()
        previous = self.current_manifest()
        self.box = [100.0, 700.0, 300.0, 650.0]
        manifest = parse.parse_document(self.pdf, "doc", force=True, previous=previous)
        texts = [doc.items[0].text for _, _, doc in parse.iter_blocks("doc", manifest)]
        self.assertEqual(texts[1], BROKEN)
        orphan = [i for i in manifest["quality"]["issues"] if i["code"] == "FORMULA_CORRECTION_ORPHANED"]
        self.assertEqual(len(orphan), 1)
        self.assertIn(FIXED, orphan[0]["message"])
        self.assertEqual([c["text"] for c in manifest["corrections"]], [FIXED])

    def test_retrying_a_block_with_an_orphan_does_not_break(self):
        self.apply()
        previous = self.current_manifest()
        self.box = [100.0, 700.0, 300.0, 650.0]
        orphaned = parse.parse_document(self.pdf, "doc", force=True, previous=previous)
        parse.parse_document(self.pdf, "doc", previous=orphaned)

    def test_changed_formula_is_rejected_without_touching_anything(self):
        before = self.doc()["artifact_manifest"]
        result = self.apply(expected="outro latex")
        self.assertEqual(result["applied"], [])
        self.assertEqual(len(result["rejected"]), 1)
        self.assertIn("mudou", result["rejected"][0]["reason"])
        self.assertEqual(self.doc()["artifact_manifest"], before)
        self.assertEqual(runtime._job_status("formulas", result), "partial")

    def test_unknown_formula_and_empty_text_are_rejected(self):
        for change in ({"ref": "#/texts/99", "text": FIXED, "expected": BROKEN}, {"page": 99, "text": FIXED, "expected": BROKEN},
                       {"text": "   ", "expected": BROKEN}):
            with self.subTest(change=change):
                result = self.apply(**change)
                self.assertEqual(len(result["rejected"]), 1)
                self.assertEqual(result["applied"], [])

    def test_applying_again_is_not_a_conflict_and_resumes_vectors(self):
        self.apply()
        generation = self.doc()["artifact_manifest"]
        self.conn.execute("UPDATE chunks SET embedded = 0 WHERE doc_id = 'doc'")
        self.conn.execute("UPDATE documents SET status = 'parsed' WHERE doc_id = 'doc'")
        result = self.apply()
        self.assertEqual(result["rejected"], [])
        self.assertEqual(len(result["applied"]), 1)
        self.assertEqual(result["n_embedded"], 2)
        self.assertEqual(self.doc()["artifact_manifest"], generation)
        self.assertEqual(self.doc()["status"], "indexed")
        self.assertEqual(runtime._job_status("formulas", result), "done")

    def test_second_correction_compares_with_the_first(self):
        self.apply()
        self.assertEqual(len(self.apply(text=FIXED + " ", expected=BROKEN)["rejected"]), 1)
        result = self.apply(text=r"\alpha", expected=FIXED)
        self.assertEqual(len(result["applied"]), 1)
        correction = self.current_manifest()["corrections"][0]
        self.assertEqual((correction["text"], correction["original"]), (r"\alpha", BROKEN))

    def test_restoring_the_original_brings_the_flag_back(self):
        self.apply()
        result = self.apply(text=None, expected=FIXED)
        self.assertEqual(len(result["applied"]), 1)
        manifest = self.current_manifest()
        self.assertEqual(manifest["corrections"], [])
        self.assertEqual(manifest["quality"]["formula_flagged"], 1)
        self.assertIn(f"$${BROKEN}$$", self.chunk_texts())
        self.assertEqual(self.doc()["summary"], "Resumo antigo")

    def test_restoring_without_a_correction_is_rejected(self):
        self.assertEqual(len(self.apply(text=None)["rejected"]), 1)

    def test_failure_before_publishing_leaves_the_old_state_whole(self):
        before = (self.doc()["artifact_manifest"], self.doc()["quality_json"], self.chunk_texts())
        with patch.object(pipeline, "segment_document", side_effect=RuntimeError("queda")):
            with self.assertRaises(RuntimeError):
                self.apply()
        self.assertEqual((self.doc()["artifact_manifest"], self.doc()["quality_json"], self.chunk_texts()), before)
        self.assertEqual(self.doc()["status"], "indexed")
        self.assertEqual(self.apply()["rejected"], [])

    def test_document_without_artifacts_is_refused(self):
        self.conn.execute("UPDATE documents SET artifact_manifest = NULL WHERE doc_id = 'doc'")
        with self.assertRaisesRegex(ValueError, "sem artefatos"):
            self.apply()

    def test_conflict_and_a_good_change_in_one_batch(self):
        result = pipeline.apply_formula_corrections("doc", [
            {"ref": "#/texts/0", "page": 1, "text": r"\beta", "expected": "x = 1"},
            {"ref": "#/texts/0", "page": 2, "text": FIXED, "expected": "outro"},
        ], conn=self.conn)
        self.assertEqual([c["page"] for c in result["applied"]], [1])
        self.assertEqual([c["page"] for c in result["rejected"]], [2])
        self.assertEqual(json.loads(self.doc()["quality_json"])["formula_flagged"], 1)


class OutsideSampleTests(CorrectionTests):
    """The same behaviour when the formula is past what the summary reads."""

    filler_chunks = 8

    def test_summary_is_kept_and_reported_as_affected_inside_the_sample(self):
        result = self.apply()
        self.assertFalse(result["summary_affected"])
        self.assertEqual(self.doc()["summary"], "Resumo antigo")

    test_vectors_are_rebuilt_for_the_document = None
    test_applying_again_is_not_a_conflict_and_resumes_vectors = None


class RegenerateSummaryTests(unittest.TestCase):
    def setUp(self):
        self.conn = db.connect(":memory:")
        db._migrate(self.conn)
        self.addCleanup(self.conn.close)
        pipeline.register_document("doc", Path("livro.pdf"), 1, self.conn)
        section = SectionRecord("s", "doc", None, 0, 1, "h", "h", 1, 1, ["texto"])
        chunk = ChunkRecord("c", "doc", "s", 0, 1, 1, 1, "h", "texto", "texto", "text")
        pipeline.persist_segments("doc", [section], [chunk], conn=self.conn)
        self.conn.execute("UPDATE documents SET status = 'indexed', summary = 'Antigo' WHERE doc_id = 'doc'")
        self.llm = Mock()
        self.llm.is_available.return_value = True
        self.llm.unload.return_value = True
        patcher = patch.dict("sys.modules", {"kb.llm": self.llm})
        patcher.start()
        self.addCleanup(patcher.stop)
        attr = patch("kb.llm", self.llm, create=True)
        attr.start()
        self.addCleanup(attr.stop)

    def summary(self):
        return self.conn.execute("SELECT summary FROM documents").fetchone()[0]

    def test_replaces_the_summary_when_a_new_one_is_generated(self):
        self.llm.generate.return_value = "Novo\nTÓPICOS: a; b"
        result = enrich.run_pass(doc_ids=["doc"], regenerate=True, conn=self.conn)
        self.assertEqual(result["documents"], 1)
        self.assertEqual(self.summary(), "Novo")
        self.assertEqual(runtime._job_status("summary", {"enrichment": result}), "done")
        self.llm.unload.assert_called()

    def test_keeps_the_summary_when_the_model_fails(self):
        self.llm.generate.side_effect = RuntimeError("sem memória")
        result = enrich.run_pass(doc_ids=["doc"], regenerate=True, conn=self.conn)
        self.assertEqual((result["documents"], result["failed"]), (0, 1))
        self.assertEqual(self.summary(), "Antigo")
        self.assertEqual(runtime._job_status("summary", {"enrichment": result}), "failed")

    def test_keeps_the_summary_when_the_model_is_unavailable(self):
        self.llm.is_available.return_value = False
        result = enrich.run_pass(doc_ids=["doc"], regenerate=True, conn=self.conn)
        self.assertEqual(self.summary(), "Antigo")
        self.assertTrue(result["skipped"])
        self.assertEqual(runtime._job_status("summary", {"enrichment": result}), "failed")

    def test_without_regenerate_a_summary_is_left_alone(self):
        self.assertEqual(enrich.run_pass(doc_ids=["doc"], conn=self.conn)["documents"], 0)
        self.llm.generate.assert_not_called()

    def test_regenerate_requires_documents(self):
        with self.assertRaises(ValueError):
            enrich.run_pass(regenerate=True, conn=self.conn)


class ContractTests(unittest.TestCase):
    def test_commands_parse(self):
        parser = cli.build_parser()
        args = parser.parse_args(["formulas", "apply", "--doc-id", "d", "--file", "c.json", "--background"])
        self.assertEqual((args.command, args.action, args.doc_id, args.file, args.background),
                         ("formulas", "apply", "d", "c.json", True))
        args = parser.parse_args(["summarize", "--doc-id", "d"])
        self.assertEqual((args.command, args.doc_id, args.background), ("summarize", "d", False))

    def test_changes_file_keeps_latex_intact_and_refuses_bad_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.json"
            latex = r"\begin{array}{c} \frac{a}{b} \& 100\% \text{“x”} \end{array}"
            path.write_text(json.dumps([{"ref": "#/texts/1", "page": 3, "text": latex}, {"ref": "#/texts/2", "page": 3, "text": None}]),
                            encoding="utf-8")
            self.assertEqual(cli._read_formula_changes(str(path))[0]["text"], latex)
            for bad in ([], {}, [{"ref": 1, "page": 3, "text": "x"}], [{"ref": "r", "page": "3", "text": "x"}], [{"ref": "r", "page": 3, "text": 1}]):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        cli._read_formula_changes(str(path))

    def test_runtime_accepts_the_jobs_only_in_ingestion(self):
        with patch("kb.jobs.create_job", return_value="job"), patch.object(runtime, "_job_target", return_value="livro.pdf"), \
                patch("threading.Thread"):
            worker = runtime.Runtime("ingestion", "nonce")
            for action in ("formulas", "summary"):
                worker.job_id = None
                self.assertEqual(worker.handle({"action": action, "payload": {"doc_id": "d", "changes": []}})["job_id"], "job")
        worker = runtime.Runtime("interaction", "nonce")
        with self.assertRaises(runtime.SessionError) as caught:
            worker.handle({"action": "formulas", "payload": {"doc_id": "d", "changes": []}})
        self.assertEqual(caught.exception.code, "WRONG_MODE")

    def test_job_target_needs_a_known_document(self):
        conn = db.connect(":memory:")
        db._migrate(conn)
        self.addCleanup(conn.close)
        with patch("kb.db.get_conn", return_value=conn):
            with self.assertRaisesRegex(ValueError, "não encontrado"):
                runtime._job_target("formulas", {"doc_id": "d"})
            pipeline.register_document("d", Path("livro.pdf"), 1, conn)
            self.assertEqual(runtime._job_target("summary", {"doc_id": "d"}), "livro.pdf")


if __name__ == "__main__":
    unittest.main()
