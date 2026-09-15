"""Tests for freezing the split once a model has learnt from it.

Moving a package from train to test after training puts studied functions
into the exam. Nothing breaks; the numbers just stop meaning anything. So
the reassignment that would do it is refused.
"""

import types

import pytest

from elenchus import cli
from elenchus.corpus.dataset import load_splits, store_splits
from elenchus.db import finish_run, start_run


def add_function(conn, package, name, op):
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x', datetime('now'))", (f"/{package}", package))
        binary_id = conn.execute(
            "SELECT id FROM binaries WHERE sha256 = ?", (package,)).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO corpus_binaries (binary_id, package, compiler, "
            "opt_level, stripped) VALUES (?, ?, 'gcc', 'O0', 1)", (binary_id, package))
        count = conn.execute(
            "SELECT COUNT(*) FROM functions WHERE binary_id = ?", (binary_id,)
        ).fetchone()[0]
        fid = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, 'F')", (binary_id, 0x1000 + count)).lastrowid
        conn.execute(
            "INSERT INTO ground_truth (binary_id, function_id, address, name, "
            "return_type, param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, fid, 0x1000 + count, name, f"src/{package}.c"))
        listing = "\n".join(f"{i:x}\t{op}\tRAX RBX\t" for i in range(10 + count))
        conn.execute(
            "INSERT INTO function_code VALUES (?, ?, ?, 32, ?, ?)",
            (fid, binary_id, 10 + count, f"h{fid}", listing))


@pytest.fixture
def corpus(db, monkeypatch):
    for package, op, n in (("big", "xor", 6), ("mid", "imul", 3), ("small", "shl", 2)):
        for i in range(n):
            add_function(db, package, f"{package}{i}", op)
    monkeypatch.setattr(cli, "connect", lambda _p: db)
    monkeypatch.setattr(cli, "load_manifest", lambda _p: [])
    return db


def run(assign=True, force=False):
    return cli.cmd_dataset(types.SimpleNamespace(db=":memory:", assign=assign,
                                                 force=force, manifest=None))


def test_assigning_before_any_training_is_allowed(corpus):
    store_splits(corpus, {"big": "test", "mid": "train", "small": "val"})
    assert run() == 0
    assert load_splits(corpus)["big"] == "train"


def test_reassigning_after_training_is_refused(corpus, capsys):
    store_splits(corpus, {"big": "test", "mid": "train", "small": "val"})
    finish_run(corpus, start_run(corpus, "train", params={}), "ok")

    assert run() == 1
    assert load_splits(corpus) == {"big": "test", "mid": "train", "small": "val"}
    assert "refused" in capsys.readouterr().out


def test_a_failed_training_run_still_locks_the_split(corpus):
    """It may have written a checkpoint; its data exposure happened regardless."""
    store_splits(corpus, {"big": "test", "mid": "train", "small": "val"})
    finish_run(corpus, start_run(corpus, "train", params={}), "failed")
    assert run() == 1


def test_force_overrides_the_lock(corpus):
    store_splits(corpus, {"big": "test", "mid": "train", "small": "val"})
    finish_run(corpus, start_run(corpus, "train", params={}), "ok")

    assert run(force=True) == 0
    assert load_splits(corpus)["big"] == "train"


def test_reassigning_to_the_same_split_after_training_is_fine(corpus):
    run()
    finish_run(corpus, start_run(corpus, "train", params={}), "ok")
    assert run() == 0


def test_other_runs_do_not_lock_the_split(corpus):
    store_splits(corpus, {"big": "test", "mid": "train", "small": "val"})
    finish_run(corpus, start_run(corpus, "evaluate", params={}), "ok")
    assert run() == 0


def test_the_command_splits_by_the_domains_in_the_manifest(corpus, monkeypatch, capsys):
    manifest = [types.SimpleNamespace(name=n, domain=d)
                for n, d in (("big", "a"), ("mid", "b"), ("small", "c"))]
    monkeypatch.setattr(cli, "load_manifest", lambda _p: manifest)
    seen = {}
    real = cli.assign_splits

    def spy(sizes, shares=None, domains=None):
        seen["domains"] = domains
        return real(sizes, domains=domains)

    monkeypatch.setattr(cli, "assign_splits", spy)

    assert run() == 0
    assert seen["domains"] == {"big": "a", "mid": "b", "small": "c"}
    # One package per domain cannot cover three splits, and the report says so.
    assert "missing a split: a, b, c" in capsys.readouterr().out


def test_a_package_missing_from_the_manifest_is_named(corpus, capsys):
    run()
    assert "no domain in the manifest for big, mid, small" in capsys.readouterr().out
