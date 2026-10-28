"""SQLite storage for documents, chunks and the question log."""
import sqlite3
import threading

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    page INTEGER,
    heading TEXT,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    answered INTEGER NOT NULL,
    top_score REAL,
    feedback INTEGER,               -- 1 = helpful, -1 = not helpful
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Store:
    def __init__(self, path: str | None = None):
        self.conn = sqlite3.connect(path or config.DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    # documents --------------------------------------------------------
    def add_document(self, name: str, chunks, embeddings=None) -> dict:
        with self.lock, self.conn:
            cur = self.conn.execute("INSERT INTO documents (name, n_chunks) VALUES (?, ?)", (name, len(chunks)))
            doc_id = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO chunks (doc_id, idx, page, heading, text, embedding) VALUES (?,?,?,?,?,?)",
                [(doc_id, i, c.page, c.heading, c.text, embeddings[i] if embeddings else None)
                 for i, c in enumerate(chunks)],
            )
        return self.get_document(doc_id)

    def get_document(self, doc_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def find_document(self, name: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM documents WHERE name=?", (name,)).fetchone()
        return dict(r) if r else None

    def list_documents(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM documents ORDER BY id")]

    def delete_document(self, doc_id: int) -> bool:
        with self.lock, self.conn:
            return self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0

    def all_chunks(self) -> list[dict]:
        q = ("SELECT c.id, c.doc_id, c.idx, c.page, c.heading, c.text, c.embedding, d.name AS doc_name "
             "FROM chunks c JOIN documents d ON d.id=c.doc_id ORDER BY c.id")
        return [dict(r) for r in self.conn.execute(q)]

    # question log -----------------------------------------------------
    def log_question(self, session_id, question, answer, answered, top_score) -> int:
        with self.lock, self.conn:
            return self.conn.execute(
                "INSERT INTO questions (session_id, question, answer, answered, top_score) VALUES (?,?,?,?,?)",
                (session_id, question, answer, int(answered), top_score),
            ).lastrowid

    def set_feedback(self, qid: int, value: int) -> bool:
        with self.lock, self.conn:
            return self.conn.execute("UPDATE questions SET feedback=? WHERE id=?", (value, qid)).rowcount > 0

    def recent_history(self, session_id: str, n: int = 3) -> list[dict]:
        rows = self.conn.execute(
            "SELECT question, answer FROM questions WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, n))
        return [dict(r) for r in rows][::-1]

    def gaps(self, limit: int = 50) -> list[dict]:
        """Questions the bot could not answer or students marked unhelpful."""
        q = ("SELECT id, question, answer, answered, feedback, created_at FROM questions "
             "WHERE answered=0 OR feedback=-1 ORDER BY id DESC LIMIT ?")
        return [dict(r) for r in self.conn.execute(q, (limit,))]

    def stats(self) -> dict:
        one = lambda q: self.conn.execute(q).fetchone()[0]  # noqa: E731
        return {
            "documents": one("SELECT COUNT(*) FROM documents"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "questions": one("SELECT COUNT(*) FROM questions"),
            "answered_pct": round(100 * (one("SELECT AVG(answered) FROM questions") or 0)),
            "helpful": one("SELECT COUNT(*) FROM questions WHERE feedback=1"),
            "not_helpful": one("SELECT COUNT(*) FROM questions WHERE feedback=-1"),
        }
