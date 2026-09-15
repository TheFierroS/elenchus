"""Schema migrations: bring an existing database up to the current schema.

A fresh database is created directly from schema() and stamped as current,
so it never runs a migration. Only databases that predate a schema change
walk the chain.

Each migration is a (version, description, function) tuple. Migrations are
append-only: once released, a migration is never edited or removed, because
someone may still hold a database that has not passed through it yet.
"""

SCHEMA_VERSION = 9


def has_column(conn, table, column):
    """True if table already has this column.

    Migrations run against databases that have been through who knows what -
    a restore, a manual fix, an older version of this code that ordered its
    steps differently. Asking first costs one query and turns a crash into a
    no-op.
    """
    return any(
        row[1] == column
        for row in conn.execute(f"PRAGMA table_info({table})")
    )


def _migration_001_runs(conn):
    """Introduce the runs table and attach every existing event to a run.

    Events predating this migration have no run of their own, so they are
    attached to a single synthetic 'legacy' run rather than being discarded.
    The run records what little is known: it came from Ghidra extraction at
    an unrecorded code version.

    SQLite cannot add a NOT NULL column to a populated table, and cannot
    alter a column afterwards, so events is rebuilt rather than altered.

    The runs table is created here rather than left to schema(). A migration
    that assumes another step ran first is a migration that works until the
    order changes - and the order did change, once, to fix a different bug.
    Each one builds what it needs.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY,
            kind         TEXT NOT NULL,
            tool         TEXT,
            tool_version TEXT,
            code_version TEXT,
            params       TEXT NOT NULL,
            seed         INTEGER,
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            status       TEXT NOT NULL
                         CHECK (status IN ('running', 'ok', 'failed'))
        )
    """)

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
    if not has_column(conn, "functions", "library"):
        conn.execute("ALTER TABLE functions ADD COLUMN library TEXT")


def _migration_003_basic_blocks(conn):
    """Add the basic_blocks table for control-flow structure.

    A brand new table, so CREATE is enough - nothing to migrate into it.
    Existing binaries gain blocks the next time they are scanned.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS basic_blocks (
            id          INTEGER PRIMARY KEY,
            binary_id   INTEGER NOT NULL REFERENCES binaries(id),
            function_id INTEGER NOT NULL REFERENCES functions(id),
            address     INTEGER NOT NULL,
            size        INTEGER NOT NULL,
            UNIQUE (binary_id, address)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_blocks_func ON basic_blocks (function_id)"
    )


def _migration_004_corpus(conn):
    """Add corpus_binaries and ground_truth for the self-labelling corpus.

    Both are new tables, so CREATE suffices. corpus_binaries records which
    package and build a binary came from and its twin; ground_truth holds the
    compiler's truth, attached to the stripped binary by address.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus_binaries (
            binary_id INTEGER PRIMARY KEY REFERENCES binaries(id),
            package   TEXT NOT NULL,
            version   TEXT,
            compiler  TEXT NOT NULL,
            opt_level TEXT NOT NULL,
            stripped  INTEGER NOT NULL,
            twin_id   INTEGER REFERENCES binaries(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ground_truth (
            id          INTEGER PRIMARY KEY,
            binary_id   INTEGER NOT NULL REFERENCES binaries(id),
            function_id INTEGER REFERENCES functions(id),
            address     INTEGER NOT NULL,
            name        TEXT NOT NULL,
            return_type TEXT,
            param_types TEXT,
            decl_line   INTEGER,
            UNIQUE (binary_id, address)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_gt_function ON ground_truth (function_id)"
    )


def _migration_005_function_code(conn):
    """Add function_code, holding each function's instruction listing.

    A new table, so CREATE is enough. Binaries already scanned gain their
    listings the next time they are scanned - `elenchus extract-code` does
    exactly that without recompiling or re-extracting anything else.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS function_code (
            function_id    INTEGER PRIMARY KEY REFERENCES functions(id),
            binary_id      INTEGER NOT NULL REFERENCES binaries(id),
            n_instructions INTEGER NOT NULL,
            code_size      INTEGER NOT NULL,
            byte_hash      TEXT NOT NULL,
            listing        TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_code_hash ON function_code (byte_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_code_binary ON function_code (binary_id)"
    )


def _migration_006_decl_file(conn):
    """Add ground_truth.decl_file: the source file a function came from.

    Nullable, so a plain ALTER suffices. Existing rows stay null until
    `elenchus refresh-truth` reads the debug twins again - the DWARF is still
    on disk, so nothing has to be recompiled or rescanned to fill it.

    This column is what separates code we compiled from runtime the toolchain
    linked in. Without it the two are told apart by keeping a list of C
    runtime function names, which is wrong the moment the toolchain changes.
    """
    if not has_column(conn, "ground_truth", "decl_file"):
        conn.execute("ALTER TABLE ground_truth ADD COLUMN decl_file TEXT")


def _migration_007_dataset_split(conn):
    """Add dataset_split, recording which packages a model may learn from.

    The assignment is deterministic, so it could be recomputed - but a
    training run has to be able to say which packages it was shown, months
    later, after the corpus has grown and the same function would recompute
    differently. Writing it down is what makes that answerable.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_split (
            package TEXT PRIMARY KEY,
            split   TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
        )
    """)


def _migration_008_measurements(conn):
    """Add measurements: what each method scored, against which run.

    Evaluation results cannot be events - an event belongs to a binary, and a
    measurement belongs to a dataset. They hang off runs instead, which
    already carry the commit hash and the timestamp that make a number
    comparable to another number.
    """
    conn.execute("""
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
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_meas_split "
        "ON measurements (split, method)"
    )


def _migration_009_corpus_gaps(conn):
    """Add corpus_gaps: levels known to be missing, with the reason.

    A level that cannot be scanned - quickjs -O1 exhausts Ghidra's heap however
    much it is given - otherwise leaves a permanent warning in every check. A
    warning that is always there teaches everyone to stop reading warnings, so
    the next real one goes unseen. Recording the gap turns it from a warning
    into a stated, dated decision, and lets builds stop retrying it.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus_gaps (
            package     TEXT NOT NULL,
            opt_level   TEXT NOT NULL
                        CHECK (opt_level IN ('O0', 'O1', 'O2', 'O3')),
            reason      TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (package, opt_level)
        )
    """)


MIGRATIONS = [
    (1, "runs table, events.run_id", _migration_001_runs),
    (2, "functions.library for imports", _migration_002_function_library),
    (3, "basic_blocks table", _migration_003_basic_blocks),
    (4, "corpus_binaries and ground_truth", _migration_004_corpus),
    (5, "function_code instruction listings", _migration_005_function_code),
    (6, "ground_truth.decl_file for provenance", _migration_006_decl_file),
    (7, "dataset_split table", _migration_007_dataset_split),
    (8, "measurements table", _migration_008_measurements),
    (9, "corpus_gaps table", _migration_009_corpus_gaps),
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
