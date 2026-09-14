"""Tests for the migration chain.

These were missing for eight migrations, and the reason they were missing is
worth stating: every test creates a fresh database, a fresh database is built
straight from schema() and stamped current, and so no test ever ran a
migration. The code path that touches real users' data was the one path
nothing exercised.

The first run of this file found a real defect. connect() ran schema()
before migrate(), so an old database got today's tables created underneath
it, and migration 6 then failed adding a column that had just appeared. Any
database from before the corpus tables existed could not be opened at all.

Two properties are checked for every starting version. The data survives -
rows that were there before are there after. And the shape converges - a
database migrated up to current is indistinguishable from one created
current, because two ways of building the same schema drift apart silently
and nothing else would notice.
"""

import sqlite3

import pytest

from elenchus.db import connect, schema
from elenchus.migrations import MIGRATIONS, SCHEMA_VERSION, has_column, stamp

# The schema as it stood before any migration existed. Written out by hand
# rather than generated: a historical shape has to stay fixed even as
# schema() moves on, or the test quietly stops testing what it was written
# for. Note what is absent - no runs, no library column, no basic blocks,
# and events without a run of their own.
V0_SCHEMA = """
CREATE TABLE binaries (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
    arch TEXT, imported_at TEXT NOT NULL
);
CREATE TABLE functions (
    id INTEGER PRIMARY KEY, binary_id INTEGER NOT NULL, address INTEGER NOT NULL,
    size INTEGER, raw_name TEXT, is_external INTEGER DEFAULT 0,
    is_thunk INTEGER DEFAULT 0, UNIQUE (binary_id, address)
);
CREATE TABLE strings (
    id INTEGER PRIMARY KEY, binary_id INTEGER NOT NULL, address INTEGER NOT NULL,
    value TEXT NOT NULL, encoding TEXT, length INTEGER, UNIQUE (binary_id, address)
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY, binary_id INTEGER NOT NULL, seq INTEGER NOT NULL,
    created_at TEXT NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL,
    UNIQUE (binary_id, seq)
);
CREATE TABLE event_links (
    event_id INTEGER NOT NULL, entity_kind TEXT NOT NULL,
    entity_id INTEGER NOT NULL, role TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_kind, entity_id, role)
);
"""


def old_database(path, version):
    """Build a database as it looked at the given version, with data in it.

    The intermediate shapes come from running the migrations themselves
    rather than from more hand-written schemas. That keeps the fixtures
    honest - a version-4 database is by definition whatever migration 4
    produces - and it means adding a migration does not mean writing another
    historical schema by hand.

    The data goes in before any of it runs, so migrations that rebuild a
    table have something to lose.
    """
    conn = sqlite3.connect(path)
    conn.executescript(V0_SCHEMA)

    conn.execute(
        "INSERT INTO binaries (id, path, sha256, arch, imported_at) "
        "VALUES (1, '/tmp/old.dll', 'abc', 'x86:LE:64:default', "
        "        '2026-01-01 00:00:00')"
    )
    conn.execute(
        "INSERT INTO functions (id, binary_id, address, size, raw_name) "
        "VALUES (1, 1, 4096, 32, 'deflate_stored')"
    )
    conn.execute(
        "INSERT INTO events (binary_id, seq, created_at, type, payload) "
        "VALUES (1, 1, '2026-01-01', 'observation.function', '{}')"
    )
    conn.execute(
        "INSERT INTO event_links (event_id, entity_kind, entity_id, role) "
        "VALUES (1, 'function', 1, 'subject')"
    )
    stamp(conn, 0)

    for number, _description, function in MIGRATIONS:
        if number > version:
            break
        function(conn)
        stamp(conn, number)

    conn.commit()
    conn.close()


def shape_of(conn):
    """Return the schema as {table: {column: type}}, plus the index names.

    Structure, not text. SQLite keeps the original CREATE statement of every
    table, so a table that no migration ever rebuilt still carries the
    wording it was born with - and a column added by ALTER sits at the end
    rather than where schema() puts it. Comparing the text would fail on
    every database that was ever migrated, which is every real one.

    What has to match is what queries depend on: the same tables, holding
    the same columns, of the same types, with the same indexes.
    """
    shape = {}
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    ]

    for table in tables:
        shape[table] = {
            row["name"]: row["type"].upper()
            for row in conn.execute(f"PRAGMA table_info({table})")
        }

    shape["__indexes__"] = {
        row["name"]: ""
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }

    return shape


VERSIONS = [entry[0] - 1 for entry in MIGRATIONS]


def test_the_chain_is_numbered_without_gaps():
    """Migrations are append-only and applied in order; a gap breaks that."""
    numbers = [entry[0] for entry in MIGRATIONS]
    assert numbers == list(range(1, len(numbers) + 1))
    assert numbers[-1] == SCHEMA_VERSION


def test_every_migration_has_a_description():
    for number, description, function in MIGRATIONS:
        assert description, number
        assert function.__doc__, number


@pytest.mark.parametrize("version", VERSIONS)
def test_a_database_at_any_version_opens(tmp_path, version):
    """The defect this file was written for: version 3 could not be opened."""
    path = tmp_path / f"v{version}.db"
    old_database(str(path), version)

    conn = connect(str(path))

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


@pytest.mark.parametrize("version", VERSIONS)
def test_migrating_preserves_the_data(tmp_path, version):
    path = tmp_path / f"v{version}.db"
    old_database(str(path), version)

    conn = connect(str(path))

    assert conn.execute(
        "SELECT raw_name FROM functions WHERE id = 1"
    ).fetchone()["raw_name"] == "deflate_stored"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM events"
    ).fetchone()["n"] == 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM event_links"
    ).fetchone()["n"] == 1


@pytest.mark.parametrize("version", VERSIONS)
def test_a_migrated_database_matches_a_fresh_one(tmp_path, version):
    """Two ways of building one schema must not drift apart.

    A migration that creates a table slightly differently from schema() -
    a missing index, a constraint spelled another way - produces a database
    that works until the day it does not, and nothing else would notice.
    """
    old_path = tmp_path / f"v{version}.db"
    old_database(str(old_path), version)
    migrated = connect(str(old_path))

    fresh = connect(str(tmp_path / f"fresh{version}.db"))

    assert shape_of(migrated) == shape_of(fresh)


def test_an_already_current_database_is_left_alone(tmp_path):
    path = tmp_path / "current.db"
    first = connect(str(path))
    with first:
        first.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES ('/tmp/x', 'x', 'x86:LE:64:default', datetime('now'))"
        )
    first.close()

    second = connect(str(path))
    assert second.execute(
        "SELECT COUNT(*) AS n FROM binaries"
    ).fetchone()["n"] == 1
    assert second.execute(
        "PRAGMA user_version"
    ).fetchone()[0] == SCHEMA_VERSION


def test_migrating_twice_changes_nothing(tmp_path):
    """Opening a database repeatedly must be safe; it happens constantly."""
    path = tmp_path / "twice.db"
    old_database(str(path), 3)

    connect(str(path)).close()
    before = shape_of(connect(str(path)))

    connect(str(path)).close()
    after = shape_of(connect(str(path)))

    assert before == after


def test_column_check_answers_honestly(tmp_path):
    conn = connect(str(tmp_path / "cols.db"))
    assert has_column(conn, "ground_truth", "decl_file")
    assert not has_column(conn, "ground_truth", "no_such_column")


def test_a_schema_with_no_version_is_treated_as_the_oldest(tmp_path):
    """user_version defaults to 0, which is where the chain starts."""
    path = tmp_path / "unversioned.db"
    old_database(str(path), 0)

    conn = sqlite3.connect(str(path))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()

    conn = connect(str(path))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_the_shipped_schema_creates_every_table_the_code_uses(tmp_path):
    """A table added to a migration but forgotten in schema() would only
    break for new users, who are the least likely to report it."""
    conn = connect(str(tmp_path / "complete.db"))
    present = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    for table in (
        "binaries", "functions", "basic_blocks", "strings", "events",
        "event_links", "runs", "corpus_binaries", "ground_truth",
        "function_code", "dataset_split", "measurements",
    ):
        assert table in present, table

    assert "CREATE TABLE" in schema()
