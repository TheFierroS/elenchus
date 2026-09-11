"""Schema migrations: bring an existing database up to the current schema.

A fresh database is created directly from schema() and stamped as current,
so it never runs a migration. Only databases that predate a schema change
walk the chain.

Each migration is a (version, description, function) tuple. Migrations are
append-only: once released, a migration is never edited or removed, because
someone may still hold a database that has not passed through it yet.
"""

SCHEMA_VERSION = 2


def _migration_001_runs(conn):
    """Introduce the runs table and attach every existing event to a run.

    Events predating this migration have no run of their own, so they are
    attached to a single synthetic 'legacy' run rather than being discarded.
    The run records what little is known: it came from Ghidra extraction at
    an unrecorded code version.

    SQLite cannot add a NOT NULL column to a populated table, and cannot
    alter a column afterwards, so events is rebuilt rather than altered.
    """
    legacy_id = conn.execute(
        "INSERT INTO runs "
        "(kind, tool, code_version, params, started_at, ended_at, status) "
        "VALUES ('extract', 'ghidra', 'unknown', '{}', "
        "        datetime('now'), datetime('now'), 'ok')"
    ).lastrowid

    conn.execute("""
        CREATE TABLE events_new (
            id         INTEGER PRIMARY KEY,
            binary_id  INTEGER NOT NULL REFERENCES binaries(id),
            run_id     INTEGER NOT NULL REFERENCES runs(id),
            seq        INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            type       TEXT NOT NULL,
            payload    TEXT NOT NULL,
            UNIQUE (binary_id, seq)
        )
    """)

    conn.execute(
        "INSERT INTO events_new "
        "(id, binary_id, run_id, seq, created_at, type, payload) "
        "SELECT id, binary_id, ?, seq, created_at, type, payload FROM events",
        (legacy_id,),
    )

    conn.execute("DROP TABLE events")
    conn.execute("ALTER TABLE events_new RENAME TO events")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_type ON events (binary_id, type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id)"
    )


def _migration_002_function_library(conn):
    """Add functions.library, holding the source DLL for imported functions.

    Nullable, so a plain ALTER suffices - no rebuild. Internal functions
    leave it null; only imports carry a library name.
    """
    conn.execute("ALTER TABLE functions ADD COLUMN library TEXT")


MIGRATIONS = [
    (1, "runs table, events.run_id", _migration_001_runs),
    (2, "functions.library for imports", _migration_002_function_library),
]


def current_version(conn) -> int:
    """Return the schema version recorded in the database."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def stamp(conn, version: int) -> None:
    """Record a schema version. PRAGMA does not accept parameters."""
    conn.execute(f"PRAGMA user_version = {int(version)}")


def is_fresh(conn) -> bool:
    """True if the database has no tables yet, i.e. it is being created now."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
    ).fetchone()
    return row[0] == 0


def migrate(conn) -> list[int]:
    """Apply every migration the database has not seen. Returns those applied.

    Foreign keys are disabled for the duration: rebuilding a table means
    dropping one that other rows may reference, which a live foreign key
    would reject even though the end state is consistent.
    """
    version = current_version(conn)
    applied = []

    pending = [m for m in MIGRATIONS if m[0] > version]
    if not pending:
        return applied

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for number, _description, function in pending:
            with conn:
                function(conn)
                stamp(conn, number)
            applied.append(number)
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    broken = conn.execute("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise RuntimeError(f"migration left {len(broken)} broken references")

    return applied
