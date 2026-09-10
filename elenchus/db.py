"""Database layer: schema, connection, and the append-only event log."""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS binaries (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    arch        TEXT,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS functions (
    id          INTEGER PRIMARY KEY,
    binary_id   INTEGER NOT NULL REFERENCES binaries(id),
    address     INTEGER NOT NULL,
    size        INTEGER,
    raw_name    TEXT,
    is_external INTEGER NOT NULL DEFAULT 0,
    is_thunk    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (binary_id, address)
);

CREATE TABLE IF NOT EXISTS strings (
    id        INTEGER PRIMARY KEY,
    binary_id INTEGER NOT NULL REFERENCES binaries(id),
    address   INTEGER NOT NULL,
    value     TEXT NOT NULL,
    encoding  TEXT,
    length    INTEGER,
    UNIQUE (binary_id, address)
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    binary_id  INTEGER NOT NULL REFERENCES binaries(id),
    seq        INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    UNIQUE (binary_id, seq)
);

CREATE TABLE IF NOT EXISTS event_links (
    event_id    INTEGER NOT NULL REFERENCES events(id),
    entity_kind TEXT NOT NULL CHECK (entity_kind IN ('function', 'string')),
    entity_id   INTEGER NOT NULL,
    role        TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_kind, entity_id, role)
);

CREATE INDEX IF NOT EXISTS idx_events_type  ON events (binary_id, type);
CREATE INDEX IF NOT EXISTS idx_links_entity ON event_links (entity_kind, entity_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, applying settings SQLite does not enable by default."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def add_event(conn, binary_id, event_type, payload, links=()):
    """Append one event and its entity links. Returns the new event id.

    links: iterable of (entity_kind, entity_id, role)
    """
    with conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM events WHERE binary_id = ?",
            (binary_id,),
        ).fetchone()
        seq = row["next"]

        cur = conn.execute(
            "INSERT INTO events (binary_id, seq, created_at, type, payload) "
            "VALUES (?, ?, datetime('now'), ?, ?)",
            (binary_id, seq, event_type, json.dumps(payload)),
        )
        event_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO event_links (event_id, entity_kind, entity_id, role) "
            "VALUES (?, ?, ?, ?)",
            [(event_id, kind, eid, role) for kind, eid, role in links],
        )

    return event_id


def add_events(conn, binary_id, records):
    """Append many events in a single transaction.

    records: iterable of (event_type, payload, links)
    Returns the number of events written.
    """
    count = 0
    with conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS last FROM events WHERE binary_id = ?",
            (binary_id,),
        ).fetchone()
        seq = row["last"]

        for event_type, payload, links in records:
            seq += 1
            cur = conn.execute(
                "INSERT INTO events (binary_id, seq, created_at, type, payload) "
                "VALUES (?, ?, datetime('now'), ?, ?)",
                (binary_id, seq, event_type, json.dumps(payload)),
            )
            event_id = cur.lastrowid

            conn.executemany(
                "INSERT INTO event_links (event_id, entity_kind, entity_id, role) "
                "VALUES (?, ?, ?, ?)",
                [(event_id, kind, eid, role) for kind, eid, role in links],
            )
            count += 1

    return count