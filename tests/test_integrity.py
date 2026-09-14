"""Tests for the integrity checks added after a day of interrupted runs.

Every check here exists because of something that happened or nearly did: a
process killed mid-scan, three corpus builds started by accident on one
database, a split left behind by a smaller corpus. The tests build each of
those situations and confirm the check sees it - and, just as important, that
a healthy database stays silent.
"""

import sys
import types

from elenchus.check import (
    CHECKS,
    WARNINGS,
    check_code_orphans,
    check_ground_truth_binding,
    check_measurement_runs,
    check_split_packages,
    check_twin_links,
    check_unfinished_runs,
    close_stale_runs,
)

sys.modules.setdefault("pyghidra", types.ModuleType("pyghidra"))

from elenchus.corpus.cli import active_runs  # noqa: E402


def without_foreign_keys(conn):
    """Build a state SQLite would normally refuse.

    Foreign keys are on, so a row pointing at a missing parent cannot be
    written by this code - and that is the first line of defence. The checks
    are the second: a database restored from a dump, copied by a tool that
    did not enforce them, or reshaped by a future migration can still arrive
    inconsistent, and then the constraint is not there to catch it.

    Turning them off here is how that arrival is simulated.
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


def new_binary(conn, path, sha):
    with conn:
        return conn.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (path, sha),
        ).lastrowid


def new_function(conn, binary_id, address=0x1000):
    with conn:
        return conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 16, 'FUN')",
            (binary_id, address),
        ).lastrowid


def new_run(conn, status="ok", kind="extract", age_minutes=0):
    with conn:
        return conn.execute(
            "INSERT INTO runs (kind, code_version, params, started_at, status) "
            "VALUES (?, 'abc', '{}', datetime('now', ?), ?)",
            (kind, f"-{age_minutes} minutes", status),
        ).lastrowid


# ------------------------------------------------------------- clean slate


def test_an_empty_database_passes_everything(db):
    for label, check in CHECKS:
        assert check(db) == [], label


def test_foreign_keys_are_enforced(db):
    """The first line of defence, which the checks sit behind.

    If this ever stops being true, several of the checks below stop being
    redundant and start being the only thing standing there.
    """
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# -------------------------------------------------------- unfinished runs


def test_a_killed_process_leaves_a_visible_trace(db):
    new_run(db, status="ok")
    new_run(db, status="running")

    found = check_unfinished_runs(db)
    assert len(found) == 1


def test_an_unfinished_run_is_a_warning_not_a_defect():
    """A dead process is worth seeing; it is not a broken database."""
    assert "unfinished runs" in WARNINGS


def test_stale_runs_close_but_recent_ones_are_left_alone(db):
    old = new_run(db, status="running", age_minutes=180)
    fresh = new_run(db, status="running", age_minutes=2)

    assert close_stale_runs(db, older_than_minutes=60) == 1

    rows = dict(db.execute("SELECT id, status FROM runs").fetchall())
    assert rows[old] == "failed"      # what actually happened to it
    assert rows[fresh] == "running"   # may still be working


# ------------------------------------------------------------ code orphans


def test_a_listing_attached_to_another_binary_is_caught(db):
    first = new_binary(db, "/tmp/a.dll", "a")
    second = new_binary(db, "/tmp/b.dll", "b")
    function_id = new_function(db, first)

    with db:
        db.execute(
            "INSERT INTO function_code "
            "(function_id, binary_id, n_instructions, code_size, byte_hash, "
            " listing) VALUES (?, ?, 1, 1, 'h', 'x')",
            (function_id, second),
        )

    assert check_code_orphans(db) == [(function_id, second)]


def test_a_listing_for_a_missing_function_is_caught(db):
    binary_id = new_binary(db, "/tmp/a.dll", "a")
    without_foreign_keys(db)
    with db:
        db.execute(
            "INSERT INTO function_code "
            "(function_id, binary_id, n_instructions, code_size, byte_hash, "
            " listing) VALUES (999, ?, 1, 1, 'h', 'x')",
            (binary_id,),
        )

    assert check_code_orphans(db) == [(999, binary_id)]


# ----------------------------------------------------- ground truth binding


def test_truth_matched_into_the_wrong_binary_is_caught(db):
    """The silent matching failure: a full match rate over wrong answers."""
    first = new_binary(db, "/tmp/a.dll", "a")
    second = new_binary(db, "/tmp/b.dll", "b")
    stranger = new_function(db, second, 0x2000)

    with db:
        db.execute(
            "INSERT INTO ground_truth "
            "(binary_id, function_id, address, name, return_type, "
            " param_types, decl_line) "
            "VALUES (?, ?, 0x1000, 'deflate', 'int', '[]', 1)",
            (first, stranger),
        )

    found = check_ground_truth_binding(db)
    assert len(found) == 1
    assert found[0][1] == "deflate"


def test_unmatched_ground_truth_is_not_a_violation(db):
    """function_id is null for inlined functions, which is expected."""
    binary_id = new_binary(db, "/tmp/a.dll", "a")
    with db:
        db.execute(
            "INSERT INTO ground_truth "
            "(binary_id, function_id, address, name, return_type, "
            " param_types, decl_line) "
            "VALUES (?, NULL, 0x1000, 'inlined', 'int', '[]', 1)",
            (binary_id,),
        )

    assert check_ground_truth_binding(db) == []


# -------------------------------------------------------------- twin links


def corpus_row(conn, binary_id, package, opt, stripped, twin=None):
    with conn:
        conn.execute(
            "INSERT INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped, twin_id) "
            "VALUES (?, ?, 'gcc', ?, ?, ?)",
            (binary_id, package, opt, 1 if stripped else 0, twin),
        )


def test_a_healthy_twin_pair_passes(db):
    debug = new_binary(db, "/tmp/d.dll", "d")
    strip = new_binary(db, "/tmp/s.dll", "s")
    corpus_row(db, debug, "zlib", "O0", stripped=False, twin=strip)
    corpus_row(db, strip, "zlib", "O0", stripped=True, twin=debug)

    assert check_twin_links(db) == []


def test_a_twin_pointing_at_the_same_side_is_caught(db):
    """Reading ground truth from a stripped file would find nothing."""
    first = new_binary(db, "/tmp/s1.dll", "s1")
    second = new_binary(db, "/tmp/s2.dll", "s2")
    corpus_row(db, first, "zlib", "O0", stripped=True, twin=second)
    corpus_row(db, second, "zlib", "O0", stripped=True, twin=first)

    assert len(check_twin_links(db)) == 2


def test_a_twin_pointing_at_nothing_is_caught(db):
    binary_id = new_binary(db, "/tmp/s.dll", "s")
    without_foreign_keys(db)
    corpus_row(db, binary_id, "zlib", "O0", stripped=True, twin=4242)

    found = check_twin_links(db)
    assert len(found) == 1
    assert found[0][3] == "twin missing"


# --------------------------------------------------------- split and runs


def test_a_split_naming_an_absent_package_is_caught(db):
    binary_id = new_binary(db, "/tmp/s.dll", "s")
    corpus_row(db, binary_id, "zlib", "O0", stripped=True)

    with db:
        db.execute(
            "INSERT INTO dataset_split (package, split) VALUES ('zlib', 'train')"
        )
        db.execute(
            "INSERT INTO dataset_split (package, split) VALUES ('gone', 'test')"
        )

    assert check_split_packages(db) == ["gone"]


def test_a_measurement_without_its_run_is_caught(db):
    without_foreign_keys(db)
    with db:
        db.execute(
            "INSERT INTO measurements "
            "(run_id, method, split, query_opt, pool_opt, n_queries, n_pool, "
            " metrics, created_at) "
            "VALUES (999, 'bm25', 'test', 'O0', 'O3', 1, 1, '{}', "
            "        datetime('now'))"
        )

    assert check_measurement_runs(db) == [(1, "bm25")]


# ------------------------------------------------------- concurrent builds


def test_a_recent_open_run_blocks_a_second_build(db):
    """Three corpus builds once ran at once on one database."""
    new_run(db, status="running", kind="extract", age_minutes=2)
    assert len(active_runs(db)) == 1


def test_the_wreckage_of_an_old_crash_does_not_block(db):
    """A run open since yesterday is a dead process, not a competitor."""
    new_run(db, status="running", kind="extract", age_minutes=600)
    assert active_runs(db) == []


def test_a_finished_run_does_not_block(db):
    new_run(db, status="ok", kind="extract", age_minutes=1)
    new_run(db, status="failed", kind="extract", age_minutes=1)
    assert active_runs(db) == []
