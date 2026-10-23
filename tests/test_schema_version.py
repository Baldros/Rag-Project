"""Schema version is read before it is written; a newer database is refused."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from kb import cli, db, session


class SchemaVersionTests(unittest.TestCase):
    def setUp(self):
        self.store = tempfile.TemporaryDirectory()
        self.path = Path(self.store.name) / "kb.sqlite3"

    def tearDown(self):
        self.store.cleanup()

    def version(self) -> int:
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def stamp(self, version: int) -> None:
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            conn.execute(f"PRAGMA user_version = {version}")

    def test_new_database_is_stamped_with_the_current_version(self):
        with contextlib.closing(db.init_db(self.path)) as conn:
            self.assertEqual(db._schema_version(conn), db.SCHEMA_VERSION)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM documents")
                             .fetchone()[0], 0)
        self.assertEqual(self.version(), db.SCHEMA_VERSION)

    def test_reopening_keeps_the_version_and_the_schema(self):
        db.init_db(self.path).close()
        with contextlib.closing(db.init_db(self.path)) as conn:
            self.assertEqual(db._schema_version(conn), db.SCHEMA_VERSION)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks")
                             .fetchone()[0], 0)

    def test_newer_database_is_refused_instead_of_downgraded(self):
        db.init_db(self.path).close()
        self.stamp(db.SCHEMA_VERSION + 1)

        with self.assertRaises(session.SessionError) as raised:
            db.init_db(self.path)

        self.assertEqual(raised.exception.code, "SCHEMA_TOO_NEW")
        self.assertEqual(self.version(), db.SCHEMA_VERSION + 1)

    def test_refusal_reaches_the_cli_envelope(self):
        db.init_db(self.path).close()
        self.stamp(db.SCHEMA_VERSION + 1)
        output = io.StringIO()

        with patch.object(db, "DB_PATH", self.path), \
             patch.object(db, "STORE_DIR", self.path.parent), \
             contextlib.redirect_stdout(output):
            code = cli.main(["collections", "list"])

        self.assertEqual(code, 1)
        error = json.loads(output.getvalue())["error"]
        self.assertEqual(error["code"], "SCHEMA_TOO_NEW")
        self.assertIn("hint", error)

    def test_schema_only_database_is_migrated_without_running_upgrades(self):
        # O fixture dos outros testes cria o esquema sem estampar a versão.
        with contextlib.closing(sqlite3.connect(self.path)) as conn:
            conn.executescript(db.SCHEMA)
        self.assertEqual(self.version(), 0)

        with patch.object(db, "_upgrade") as upgrade:
            db.init_db(self.path).close()

        upgrade.assert_not_called()
        self.assertEqual(self.version(), db.SCHEMA_VERSION)

    def test_newer_version_committed_by_another_connection_is_refused(self):
        db.init_db(self.path).close()
        waiting = threading.Event()
        published = threading.Event()
        connect = db.connect
        schema_version = db._schema_version

        def traced_connect(path=None):
            conn = connect(path)
            conn.set_trace_callback(
                lambda sql: waiting.set() if sql == "BEGIN IMMEDIATE" else None)
            return conn

        def read_version(conn):
            version = schema_version(conn)
            if not conn.in_transaction:
                # Reproduz a leitura antiga sem lock: outro processo publica
                # a nova versão depois desta leitura e antes do restante do init.
                waiting.set()
                if not published.wait(5):
                    raise TimeoutError("A nova versão não foi publicada")
            return version

        def open_database():
            db.init_db(self.path).close()

        with contextlib.closing(connect(self.path)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
            with patch.object(db, "connect", side_effect=traced_connect), \
                 patch.object(db, "_schema_version", side_effect=read_version), \
                 ThreadPoolExecutor(max_workers=1) as pool:
                opening = pool.submit(open_database)
                try:
                    self.assertTrue(waiting.wait(5), "A abertura não chegou à migração")
                finally:
                    writer.commit()
                    published.set()
                with self.assertRaises(session.SessionError) as raised:
                    opening.result(timeout=5)
        self.assertEqual(raised.exception.code, "SCHEMA_TOO_NEW")
        self.assertEqual(self.version(), db.SCHEMA_VERSION + 1)

    def test_concurrent_openers_apply_an_upgrade_once_and_preserve_data(self):
        with contextlib.closing(db.init_db(self.path)) as conn:
            conn.execute("INSERT INTO collections VALUES ('c', 'Name', NULL, '2026-01-01')")

        def upgrade(conn, version):
            self.assertEqual(version, 1)
            conn.execute("ALTER TABLE collections ADD COLUMN migrated INTEGER DEFAULT 1")

        def open_database(_):
            db.init_db(self.path).close()

        with patch.object(db, "SCHEMA_VERSION", 2), \
             patch.object(db, "_upgrade", side_effect=upgrade) as upgraded, \
             ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(open_database, range(2)))

        upgraded.assert_called_once()
        self.assertEqual(self.version(), 2)
        with contextlib.closing(db.connect(self.path)) as conn:
            self.assertEqual(tuple(conn.execute(
                "SELECT collection_id, name, migrated FROM collections").fetchone()),
                ("c", "Name", 1))

    def test_failed_or_interrupted_upgrade_rolls_back_schema_and_version(self):
        db.init_db(self.path).close()

        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                def upgrade(conn, version):
                    conn.execute("ALTER TABLE collections ADD COLUMN unfinished TEXT")
                    raise error("Interrupted migration")

                with patch.object(db, "SCHEMA_VERSION", 2), \
                     patch.object(db, "_upgrade", side_effect=upgrade):
                    with self.assertRaises(error):
                        db.init_db(self.path)

                self.assertEqual(self.version(), 1)
                with contextlib.closing(db.connect(self.path)) as conn:
                    columns = [row["name"] for row in conn.execute(
                        "PRAGMA table_info(collections)")]
                self.assertNotIn("unfinished", columns)

    def test_failed_bootstrap_leaves_no_partial_schema(self):
        with patch.object(db, "SCHEMA", db.SCHEMA + "\nINVALID SQL;\n"):
            with self.assertRaises(sqlite3.OperationalError):
                db.init_db(self.path)

        self.assertEqual(self.version(), 0)
        with contextlib.closing(db.connect(self.path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM sqlite_master")
                             .fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
