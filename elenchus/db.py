"""Database layer: schema, connection, runs, and the append-only event log."""

import json
import sqlite3
import subprocess
from functools import cache
from pathlib import Path

from elenchus import migrations
from elenchus.entities import entity_kind_check


def schema() -> str:
    """Return the full schema DDL, with entity kinds injected from the registry."""
    return f"""
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    tool         TEXT,
    tool_version TEXT,
    code_version TEXT NOT NULL,
    params       TEXT NOT NULL,
    seed         INTEGER,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    status       TEXT NOT NULL CHECK (status IN ('running', 'ok', 'failed'))
);

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
    run_id     INTEGER NOT NULL REFERENCES runs(id),
    seq        INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    UNIQUE (binary_id, seq)
);

CREATE TABLE IF NOT EXISTS event_links (
    event_id    INTEGER NOT NULL REFERENCES events(id),
    entity_kind TEXT NOT NULL {entity_kind_check()},
    entity_id   INTEGER NOT NULL,
    role        TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_kind, entity_id, role)
);

CREATE INDEX IF NOT EXISTS idx_events_type  ON events (binary_id, type);
CREATE INDEX IF NOT EXISTS idx_events_run   ON events (run_id);
CREATE INDEX IF NOT EXISTS idx_links_entity ON event_links (entity_kind, entity_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, create or migrate it, and return the connection.

    A database created now is stamped as current and skips the migrations
    entirely: schema() already produces the shape they would build.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    fresh = migrations.is_fresh(conn)
    conn.executescript(schema())

    if fresh:
        migrations.stamp(conn, migrations.SCHEMA_VERSION)
    else:
        migrations.migrate(conn)

    return conn


@cache
def code_version() -> str:
    """Return the git commit this code came from, marked dirty if edited.

    Git runs against the directory holding this file, not the caller's
    working directory: running elenchus from inside another checkout must
    not record that project's commit.

    Returns 'unknown' when there is no git to ask - an archive download,
    or a machine without git installed. That is honest; there is no commit
    to point at.
    """
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain")
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def start_run(conn, kind, tool=None, tool_version=None, params=None, seed=None):
    """Open a run and return its id. The run stays 'running' until finished.

    kind: what the run does - 'extract', 'agent', 'train', 'eval'
    params: anything needed to repeat the run, stored as JSON
    """
    payload = json.dumps(params or {})

    with conn:
        cur = conn.execute(
            "INSERT INTO runs "
            "(kind, tool, tool_version, code_version, params, seed, "
            " started_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 'running')",
            (kind, tool, tool_version, code_version(), payload, seed),
        )

    return cur.lastrowid


def finish_run(conn, run_id, status):
    """Close a run, recording how it ended: 'ok' or 'failed'.

    A run left as 'running' means the process died without closing it, so
    its events are suspect. That distinction is worth keeping.
    """
    with conn:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = datetime('now') WHERE id = ?",
            (status, run_id),
        )


def add_event(conn, binary_id, run_id, event_type, payload, links=()):
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
            "INSERT INTO events (binary_id, run_id, seq, created_at, type, payload) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?)",
            (binary_id, run_id, seq, event_type, json.dumps(payload)),
        )
        event_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO event_links (event_id, entity_kind, entity_id, role) "
            "VALUES (?, ?, ?, ?)",
            [(event_id, kind, eid, role) for kind, eid, role in links],
        )

    return event_id


def add_events(conn, binary_id, run_id, records):
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
                "INSERT INTO events (binary_id, run_id, seq, created_at, type, payload) "
                "VALUES (?, ?, ?, datetime('now'), ?, ?)",
                (binary_id, run_id, seq, event_type, json.dumps(payload)),
            )
            event_id = cur.lastrowid

            conn.executemany(
                "INSERT INTO event_links (event_id, entity_kind, entity_id, role) "
                "VALUES (?, ?, ?, ?)",
                [(event_id, kind, eid, role) for kind, eid, role in links],
            )
            count += 1

    return count