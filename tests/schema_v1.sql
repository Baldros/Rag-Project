-- Recortes de conhecimento do usuário (o "notebook").
CREATE TABLE IF NOT EXISTS collections (
    collection_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,          -- sha256 do arquivo
    filename      TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    n_pages       INTEGER,
    n_sections    INTEGER,
    n_chunks      INTEGER,
    title         TEXT,
    summary       TEXT,                      -- preenchido pelo agente
    key_topics    TEXT,                      -- JSON list, preenchido pelo agente
    enriched_at   TEXT,
    parse_version TEXT,
    embed_model   TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    created_at    TEXT NOT NULL,
    indexed_at    TEXT
);

CREATE TABLE IF NOT EXISTS document_collections (
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    collection_id TEXT NOT NULL REFERENCES collections(collection_id) ON DELETE CASCADE,
    added_at      TEXT NOT NULL,
    PRIMARY KEY (doc_id, collection_id)
);

-- A unidade devolvida ao agente: recupera-se pelo chunk, entrega-se a seção.
CREATE TABLE IF NOT EXISTS sections (
    section_id   TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_id    TEXT,
    ord          INTEGER NOT NULL,
    level        INTEGER,
    heading      TEXT,
    heading_path TEXT,
    page_start   INTEGER,
    page_end     INTEGER,
    n_chars      INTEGER,
    text         TEXT,
    summary      TEXT,                       -- preguiçoso, sob demanda
    enriched_at  TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id   TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
    ord          INTEGER NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    n_tokens     INTEGER,
    heading_path TEXT,
    text         TEXT NOT NULL,
    embed_text   TEXT NOT NULL,              -- heading_path + text: o que foi embedado
    content_type TEXT NOT NULL DEFAULT 'text',
    embedded     INTEGER NOT NULL DEFAULT 0  -- 0/1: já está no índice vetorial
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,             -- queued|running|done|failed|cancelled
    target        TEXT,
    collection_id TEXT,
    pass_name     TEXT,                      -- parse|embed
    progress      INTEGER NOT NULL DEFAULT 0,
    total         INTEGER NOT NULL DEFAULT 0,
    message       TEXT,
    error         TEXT,
    pid           INTEGER,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section  ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_pending  ON chunks(embedded) WHERE embedded = 0;
CREATE INDEX IF NOT EXISTS idx_sections_doc    ON sections(doc_id, ord);
CREATE INDEX IF NOT EXISTS idx_sections_parent ON sections(parent_id);
CREATE INDEX IF NOT EXISTS idx_doccol_col      ON document_collections(collection_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_docs_status     ON documents(status);

-- BM25 sobre o mesmo texto, sem duplicá-lo: external content aponta para `chunks`.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    heading_path,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading_path)
    VALUES (new.rowid, new.text, new.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
    VALUES ('delete', old.rowid, old.text, old.heading_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE OF text, heading_path ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
    VALUES ('delete', old.rowid, old.text, old.heading_path);
    INSERT INTO chunks_fts(rowid, text, heading_path)
    VALUES (new.rowid, new.text, new.heading_path);
END;
