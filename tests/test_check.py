"""Tests for the consistency checks: they pass clean data and catch broken."""

from elenchus.check import check_orphan_events, check_seq_gaps
from elenchus.db import add_events, finish_run, start_run


def _binary(db):
    """Insert a throwaway binary and return its id."""
    cur = db.execute(
        "INSERT INTO binaries (path, sha256, arch, imported_at) "
        "VALUES ('x', 'h', 'a', datetime('now'))"
    )
    return cur.lastrowid


def test_clean_data_has_no_orphans(db):
    """Events under a finished run are not orphaned."""
    binary_id = _binary(db)
    run_id = start_run(db, "extract")
    add_events(db, binary_id, run_id, [("observation.test", {}, [])])
    finish_run(db, run_id, "ok")

    assert check_orphan_events(db) == []


def test_running_run_orphans_its_events(db):
    """Events under a run still 'running' are flagged."""
    binary_id = _binary(db)
    run_id = start_run(db, "extract")  # left running, never finished
    add_events(db, binary_id, run_id, [("observation.test", {}, [])])

    assert len(check_orphan_events(db)) == 1


def test_contiguous_seq_has_no_gaps(db):
    """A clean 1..N sequence reports no gaps."""
    binary_id = _binary(db)
    run_id = start_run(db, "extract")
    add_events(
        db,
        binary_id,
        run_id,
        [("observation.test", {}, []) for _ in range(5)],
    )
    finish_run(db, run_id, "ok")

    assert check_seq_gaps(db) == []


def test_seq_gap_is_detected(db):
    """A hole in the sequence is caught."""
    binary_id = _binary(db)
    run_id = start_run(db, "extract")
    add_events(
        db,
        binary_id,
        run_id,
        [("observation.test", {}, []) for _ in range(5)],
    )
    finish_run(db, run_id, "ok")

    # Punch a hole: delete seq 3, leaving 1, 2, 4, 5.
    db.execute("DELETE FROM events WHERE binary_id = ? AND seq = 3", (binary_id,))
    db.commit()

    gaps = check_seq_gaps(db)
    assert len(gaps) == 1
    assert gaps[0][0] == binary_id
