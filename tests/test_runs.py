"""Tests for run lifecycle and provenance."""

from elenchus.db import finish_run, start_run


def test_run_starts_running(db):
    """A new run is open, with a start time and no end time."""
    run_id = start_run(db, "extract", tool="ghidra")

    row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "running"
    assert row["started_at"] is not None
    assert row["ended_at"] is None


def test_finish_records_end(db):
    """Finishing a run sets its status and stamps the end time."""
    run_id = start_run(db, "extract")
    finish_run(db, run_id, "ok")

    row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "ok"
    assert row["ended_at"] is not None


def test_params_round_trip(db):
    """Params go in as a dict and come back as the same JSON."""
    import json

    run_id = start_run(db, "extract", params={"binary": "x.exe", "n": 4})
    row = db.execute("SELECT params FROM runs WHERE id = ?", (run_id,)).fetchone()

    assert json.loads(row["params"]) == {"binary": "x.exe", "n": 4}


def test_status_constraint_rejects_garbage(db):
    """The CHECK constraint refuses a status outside the allowed set."""
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO runs (kind, code_version, params, started_at, status) "
            "VALUES ('extract', 'x', '{}', datetime('now'), 'banana')"
        )
