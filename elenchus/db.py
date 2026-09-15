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
    library     TEXT,
    is_external INTEGER NOT NULL DEFAULT 0,
    is_thunk    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (binary_id, address)
);

CREATE TABLE IF NOT EXISTS basic_blocks (
    id          INTEGER PRIMARY KEY,
    binary_id   INTEGER NOT NULL REFERENCES binaries(id),
    function_id INTEGER NOT NULL REFERENCES functions(id),
    address     INTEGER NOT NULL,
    size        INTEGER NOT NULL,
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

-- A binary that belongs to the corpus carries extra identity: which package
-- and build it came from, and which twin it pairs with. A plain scan of some
-- unknown .exe has no row here; only compiled corpus members do.
CREATE TABLE IF NOT EXISTS corpus_binaries (
    binary_id INTEGER PRIMARY KEY REFERENCES binaries(id),
    package   TEXT NOT NULL,
    version   TEXT,
    compiler  TEXT NOT NULL,
    opt_level TEXT NOT NULL,
    stripped  INTEGER NOT NULL,
    twin_id   INTEGER REFERENCES binaries(id)
);

-- Ground truth: what the compiler recorded, before stripping. This is truth,
-- not an observation - the standard the agent's claims are measured against -
-- so it lives in its own table rather than mixing with Ghidra's guesses in
-- functions. Attached to the stripped binary by address; function_id is null
-- when a ground-truth function has no stripped counterpart (inlined away).
CREATE TABLE IF NOT EXISTS ground_truth (
    id          INTEGER PRIMARY KEY,
    binary_id   INTEGER NOT NULL REFERENCES binaries(id),
    function_id INTEGER REFERENCES functions(id),
    address     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    return_type TEXT,
    param_types TEXT,
    decl_file   TEXT,
    decl_line   INTEGER,
    UNIQUE (binary_id, address)
);

-- The instruction stream of one function, stored raw: Ghidra's own operand
-- text, addresses already made relative. Classification into the encoder's
-- vocabulary happens in encoder/normalise.py, not here, so the scheme can
-- change without paying for another Ghidra pass. One row per function,
-- replaced by a later scan rather than appended to - the events record who
-- saw what and when.
CREATE TABLE IF NOT EXISTS function_code (
    function_id    INTEGER PRIMARY KEY REFERENCES functions(id),
    binary_id      INTEGER NOT NULL REFERENCES binaries(id),
    n_instructions INTEGER NOT NULL,
    code_size      INTEGER NOT NULL,
    byte_hash      TEXT NOT NULL,
    listing        TEXT NOT NULL
);

-- Which split each package belongs to. Recorded rather than recomputed so
-- that a training run months later can be told exactly which packages the
-- model was allowed to see, and so the assignment can be audited.
CREATE TABLE IF NOT EXISTS dataset_split (
    package TEXT PRIMARY KEY,
    split   TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
);

-- Levels of a corpus package known to be missing, and why. A decision, not a
-- defect: a scan that cannot succeed is recorded so checks stop warning about
-- it and builds stop retrying it, while the gap stays visible in every report.
CREATE TABLE IF NOT EXISTS corpus_gaps (
    package     TEXT NOT NULL,
    opt_level   TEXT NOT NULL CHECK (opt_level IN ('O0', 'O1', 'O2', 'O3')),
    reason      TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (package, opt_level)
);

-- What a method scored, when, on which corpus. A projection: the numbers
-- can be recomputed. The circumstance - which commit, which dataset - cannot.
CREATE TABLE IF NOT EXISTS measurements (
    id         INTEGER PRIMARY KEY,
    run_id     INTEGER NOT NULL REFERENCES runs(id),
    method     TEXT NOT NULL,
    split      TEXT NOT NULL,
    query_opt  TEXT NOT NULL,
    pool_opt   TEXT NOT NULL,
    n_queries  INTEGER NOT NULL,
    n_pool     INTEGER NOT NULL,
    metrics    TEXT NOT NULL,
    created_at TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_events_type   ON events (binary_id, type);
CREATE INDEX IF NOT EXISTS idx_events_run    ON events (run_id);
CREATE INDEX IF NOT EXISTS idx_links_entity  ON event_links (entity_kind, entity_id);
CREATE INDEX IF NOT EXISTS idx_blocks_func   ON basic_blocks (function_id);
CREATE INDEX IF NOT EXISTS idx_gt_function   ON ground_truth (function_id);
CREATE INDEX IF NOT EXISTS idx_code_hash     ON function_code (byte_hash);
CREATE INDEX IF NOT EXISTS idx_code_binary   ON function_code (binary_id);
CREATE INDEX IF NOT EXISTS idx_meas_split    ON measurements (split, method);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database, create or migrate it, and return the connection.

    A database created now is stamped as current and skips the migrations
    entirely: schema() already produces the shape they would build.

    An existing one is migrated *before* schema() runs, and the order is not
    cosmetic. schema() describes today's tables, so running it first on an
    old database creates tomorrow's shape behind the migrations' backs - and
    a migration that adds a column then finds it already there and fails.
    That is not hypothetical: a database from before the corpus tables
    existed could not be opened at all, because migration 6 tried to add a
    column schema() had just created.

    schema() still runs afterwards, where it is a no-op for everything the
    migrations built and a safety net for indexes.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    if migrations.is_fresh(conn):
        conn.executescript(schema())
        migrations.stamp(conn, migrations.SCHEMA_VERSION)
    else:
        migrations.migrate(conn)
        conn.executescript(schema())

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


def set_run_tool_version(conn, run_id, version):
    """Record the tool version, often known only once the tool has started."""
    with conn:
        conn.execute(
            "UPDATE runs SET tool_version = ? WHERE id = ?", (version, run_id)
        )


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
