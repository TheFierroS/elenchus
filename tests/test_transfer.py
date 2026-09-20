"""Moving training to another machine and bringing its record back.

elenchus/transfer.py exports the tables a training run reads - about a
tenth of this database, the rest being its audit trail - and imports the
runs that came back. These tests build a small database, export it, write
a run into the copy as a remote machine would, and bring it home.
"""

import argparse
import sqlite3

import pytest

from elenchus.db import connect, finish_run, schema, start_run
from elenchus.transfer import EXPORTED, export_training, import_runs


def corpus(conn, package="zlib", split="train"):
    """One package: a binary, a function, its code, ground truth, a split."""
    binary = conn.execute(
        "INSERT INTO binaries (path, sha256, arch, imported_at) "
        "VALUES (?, ?, 'x86-64', '2026-09-20') RETURNING id",
        (f"/out/{package}.dll", f"sha-{package}"),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO corpus_binaries (binary_id, package, version, compiler, "
        "opt_level, stripped) VALUES (?, ?, '1', 'gcc', 'O0', 1)",
        (binary, package),
    )
    function = conn.execute(
        "INSERT INTO functions (binary_id, address, size, is_external) "
        "VALUES (?, 4096, 32, 0) RETURNING id", (binary,),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO function_code (function_id, binary_id, n_instructions, "
        "code_size, byte_hash, listing) VALUES (?, ?, 20, 32, 'h', 'MOV EAX')",
        (function, binary),
    )
    conn.execute(
        "INSERT INTO ground_truth (binary_id, function_id, address, name, "
        "decl_file) VALUES (?, ?, 4096, 'f', ?)",
        (binary, function, f"/src/{package}/f.c"),
    )
    conn.execute(
        "INSERT INTO dataset_split (package, split) VALUES (?, ?)",
        (package, split),
    )
    conn.commit()
    return binary, function


def noise(conn, binary, run_id, count=3):
    """Audit-trail rows, which are the bulk of the real database."""
    for seq in range(count):
        conn.execute(
            "INSERT INTO events (binary_id, run_id, seq, created_at, type, "
            "payload) VALUES (?, ?, ?, '2026-09-20', 'scanned', '{}')",
            (binary, run_id, seq),
        )
    conn.commit()


@pytest.fixture
def source(db):
    run = start_run(db, "scan", params={})
    binary, function = corpus(db, "zlib", "train")
    corpus(db, "lz4", "val")
    noise(db, binary, run, count=5)
    finish_run(db, run, "ok")
    return db


# ------------------------------------------------------------------ export


def test_the_export_carries_what_training_reads(source, tmp_path):
    counts = export_training(source, tmp_path / "train.db")

    assert counts["function_code"] == 2
    assert counts["ground_truth"] == 2
    assert counts["dataset_split"] == 2
    assert set(counts) == set(EXPORTED)


def test_the_export_leaves_the_audit_trail_behind(source, tmp_path):
    """Events are 84% of the real database and training opens none of them."""
    out = tmp_path / "train.db"
    export_training(source, out)

    copy = connect(out)
    assert copy.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert copy.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert copy.execute("SELECT COUNT(*) FROM event_links").fetchone()[0] == 0


def test_the_export_keeps_the_row_ids(source, tmp_path):
    """A remote run that refers to function 7 must mean this function 7."""
    out = tmp_path / "train.db"
    export_training(source, out)

    here = source.execute("SELECT id, address FROM functions ORDER BY id").fetchall()
    there = connect(out).execute(
        "SELECT id, address FROM functions ORDER BY id").fetchall()
    assert [tuple(r) for r in here] == [tuple(r) for r in there]


def test_the_export_refuses_to_overwrite(source, tmp_path):
    out = tmp_path / "train.db"
    out.write_text("not a database")
    with pytest.raises(FileExistsError):
        export_training(source, out)


def test_the_exported_database_can_be_read_by_the_dataset_code(source, tmp_path):
    from elenchus.corpus.dataset import candidate_rows, load_splits

    out = tmp_path / "train.db"
    export_training(source, out)
    copy = connect(out)

    assert load_splits(copy) == {"zlib": "train", "lz4": "val"}
    assert len(candidate_rows(copy)) == 2


# ------------------------------------------------------------------ import


def remote_run(path, kind="train", started="2026-09-21 03:00:00"):
    conn = connect(path)
    run = conn.execute(
        "INSERT INTO runs (kind, code_version, params, seed, started_at, status) "
        "VALUES (?, 'abc1234', '{}', 0, ?, 'ok') RETURNING id", (kind, started),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO measurements (run_id, method, split, query_opt, pool_opt, "
        "n_queries, n_pool, metrics, created_at) "
        "VALUES (?, 'encoder', 'val', 'O0', 'O3', 10, 10, '{}', ?)",
        (run, started),
    )
    conn.commit()
    return run


def test_a_remote_run_comes_back_renumbered(source, tmp_path):
    out = tmp_path / "train.db"
    export_training(source, out)
    remote_id = remote_run(out)

    before = source.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    found = import_runs(source, out, apply=True)

    assert [r[0] for r in found["runs"]] == [remote_id]
    new_id = found["runs"][0][1]
    assert new_id > source.execute(
        "SELECT MAX(id) FROM runs WHERE id != ?", (new_id,)).fetchone()[0]
    assert source.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == before + 1


def test_its_measurements_follow_it_to_the_new_id(source, tmp_path):
    out = tmp_path / "train.db"
    export_training(source, out)
    remote_run(out)

    found = import_runs(source, out, apply=True)
    new_id = found["runs"][0][1]

    rows = source.execute(
        "SELECT run_id, method FROM measurements").fetchall()
    assert [(r["run_id"], r["method"]) for r in rows] == [(new_id, "encoder")]


def test_reporting_changes_nothing(source, tmp_path):
    out = tmp_path / "train.db"
    export_training(source, out)
    remote_run(out)

    before = source.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    found = import_runs(source, out)

    assert len(found["runs"]) == 1
    assert source.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == before


def test_importing_twice_brings_nothing_the_second_time(source, tmp_path):
    out = tmp_path / "train.db"
    export_training(source, out)
    remote_run(out)

    import_runs(source, out, apply=True)
    again = import_runs(source, out, apply=True)

    assert again["runs"] == []
    assert again["already_here"] == 1


def test_two_remote_runs_keep_their_order(source, tmp_path):
    out = tmp_path / "train.db"
    export_training(source, out)
    remote_run(out, started="2026-09-21 03:00:00")
    remote_run(out, started="2026-09-21 05:00:00")

    found = import_runs(source, out, apply=True)

    assert [r[3] for r in found["runs"]] == ["2026-09-21 03:00:00",
                                             "2026-09-21 05:00:00"]
    assert found["runs"][0][1] + 1 == found["runs"][1][1]


def test_a_database_that_is_not_there_is_reported_not_raised(source, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_runs(source, tmp_path / "nowhere.db")


def test_a_file_that_is_not_a_database_is_an_error(source, tmp_path):
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"not sqlite at all, not even close")
    with pytest.raises(sqlite3.Error):
        import_runs(source, broken)


def test_the_schema_the_export_writes_is_the_current_one(tmp_path):
    """A stale copy of the DDL here would show up as a missing column
    months later, on a rented machine, at three in the morning."""
    out = tmp_path / "fresh.db"
    conn = connect(out)
    conn.executescript(schema())
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(EXPORTED) <= tables


def test_every_command_takes_the_arguments_the_cli_hands_it():
    """main() calls args.func(args), one argument. A command written as
    func(conn, args) parses fine, registers fine, and fails only when
    someone runs it - which is how export-training first failed."""
    import inspect

    from elenchus import cli

    parser = cli.build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    commands = {name: sub.get_default("func")
                for action in actions for name, sub in action.choices.items()}

    assert commands, "no subcommands found"
    for name, func in commands.items():
        assert func is not None, f"{name} has no func"
        positional = [
            p for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        assert len(positional) == 1, f"{name}: {func.__name__}{inspect.signature(func)}"
