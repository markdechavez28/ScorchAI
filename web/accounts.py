"""SQLite-backed user accounts, sessions, and chat history for the web UI.

Lightweight by design (stdlib sqlite3/hashlib/secrets only, no ORM) --
appropriate for a local demo, not a hardened production auth system.
"""
import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarout.config import APP_DB_PATH

PBKDF2_ITERATIONS = 200_000


@contextmanager
def _connect():
    APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        _migrate_chat_messages(conn)


def _migrate_chat_messages(conn: sqlite3.Connection):
    """Add conversation_id to chat_messages and backfill pre-existing rows
    (from before conversations existed) into one "Imported chat" per user."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)")}
    if "conversation_id" not in columns:
        conn.execute("ALTER TABLE chat_messages ADD COLUMN conversation_id INTEGER")

    orphan_user_ids = [
        row["user_id"]
        for row in conn.execute(
            "SELECT DISTINCT user_id FROM chat_messages WHERE conversation_id IS NULL"
        )
    ]
    for user_id in orphan_user_ids:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at) VALUES (?, ?, ?)",
            (user_id, "Imported chat", datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "UPDATE chat_messages SET conversation_id = ? WHERE user_id = ? AND conversation_id IS NULL",
            (cur.lastrowid, user_id),
        )


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()


def create_user(username: str, password: str) -> int:
    salt_hex = secrets.token_hex(16)
    password_hash = _hash_password(password, salt_hex)
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, password_salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, salt_hex, datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' is already taken")
        return cur.lastrowid


def authenticate(username: str, password: str) -> int | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, password_hash, password_salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return None
    if _hash_password(password, row["password_salt"]) != row["password_hash"]:
        return None
    return row["id"]


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, datetime.now(timezone.utc).isoformat()),
        )
    return token


def get_user_id_for_token(token: str | None) -> int | None:
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id FROM sessions WHERE token = ?", (token,)
        ).fetchone()
    return row["user_id"] if row else None


def delete_session(token: str | None):
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def get_username(user_id: int) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["username"] if row else None


def make_title(text: str, max_len: int = 48) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def create_conversation(user_id: int, title: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at) VALUES (?, ?, ?)",
            (user_id, title, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def list_conversations(user_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]


def get_conversation_messages(conversation_id: int, user_id: int) -> list[dict]:
    with _connect() as conn:
        owner = conn.execute(
            "SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if owner is None or owner["user_id"] != user_id:
            return []
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def delete_conversation(conversation_id: int, user_id: int):
    with _connect() as conn:
        owner = conn.execute(
            "SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if owner is None or owner["user_id"] != user_id:
            return
        conn.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def add_message(conversation_id: int, user_id: int, role: str, content: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (conversation_id, user_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
