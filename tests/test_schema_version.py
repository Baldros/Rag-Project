"""Schema version is read before it is written; a newer database is refused."""

from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
