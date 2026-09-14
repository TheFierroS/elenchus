"""Tests for re-reading ground truth without Ghidra.

Ground truth comes from DWARF, and DWARF needs no disassembler. That is what
makes it possible to improve the reader - to start extracting a source file,
say - and bring a corpus up to date in seconds rather than recompiling it
over hours. The tests here hold that property in place: the right twin is
read, a bad one does not end the pass, and the run is recorded either way.
"""

import sys
import types

import pytest

sys.modules.setdefault("pyghidra", types.ModuleType("pyghidra"))

from elenchus.corpus.refresh import cmd_refresh_truth, corpus_pairs  # noqa: E402


def twin_pair(conn, package, opt="O0"):
    """Register a debug and a stripped binary, linked to each other."""
    with conn:
        debug = conn.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (f"/tmp/{package}_{opt}.dll", f"{package}{opt}d"),
        ).lastrowid
        strip = conn.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (f"/tmp/{package}_{opt}_stripped.dll", f"{package}{opt}s"),
        ).lastrowid

        conn.execute(
            "INSERT INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped, twin_id) "
            "VALUES (?, ?, 'gcc', ?, 0, ?)",
            (debug, package, opt, strip),
        )
        conn.execute(
            "INSERT INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped, twin_id) "
            "VALUES (?, ?, 'gcc', ?, 1, ?)",
            (strip, package, opt, debug),
        )
    return debug, strip


def args_for(path):
    return types.SimpleNamespace(db=path)


def test_the_debug_half_is_the_one_read(db):
    """Reading the stripped twin would find no DWARF and no ground truth."""
    debug, strip = twin_pair(db, "zlib")

    pairs = corpus_pairs(db)
    assert len(pairs) == 1
    assert pairs[0]["stripped_id"] == strip
    assert pairs[0]["debug_path"].endswith("zlib_O0.dll")


def test_a_binary_without_a_twin_is_not_offered(db):
    with db:
        binary_id = db.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES ('/tmp/lonely.dll', 'x', 'x86:LE:64:default', "
            "        datetime('now'))"
        ).lastrowid
        db.execute(
            "INSERT INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped) "
            "VALUES (?, 'lonely', 'gcc', 'O0', 1)",
            (binary_id,),
        )

    assert corpus_pairs(db) == []


def test_pairs_come_back_in_a_stable_order(db):
    for package in ("zlib", "lua"):
        for opt in ("O1", "O0"):
            twin_pair(db, package, opt)

    order = [(row["package"], row["opt_level"]) for row in corpus_pairs(db)]
    assert order == [("lua", "O0"), ("lua", "O1"),
                     ("zlib", "O0"), ("zlib", "O1")]


def test_an_empty_corpus_says_so_rather_than_pretending(db, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "elenchus.corpus.refresh.connect", lambda _path: db
    )
    assert cmd_refresh_truth(args_for(str(tmp_path / "x.db"))) == 1


def test_one_unreadable_binary_does_not_end_the_pass(db, monkeypatch, capsys,
                                                     tmp_path):
    """The same rule as the corpus build: lose a binary, not the run."""
    twin_pair(db, "zlib", "O0")
    twin_pair(db, "lua", "O0")

    seen = []

    def fake_store(conn, stripped_id, debug_path):
        seen.append(debug_path)
        if "lua" in debug_path:
            raise OSError("file is not a PE")
        return 10, 10

    monkeypatch.setattr(
        "elenchus.corpus.refresh.connect", lambda _path: db
    )
    monkeypatch.setattr(
        "elenchus.corpus.refresh.store_ground_truth", fake_store
    )

    assert cmd_refresh_truth(args_for(str(tmp_path / "x.db"))) == 0
    assert len(seen) == 2          # both were attempted

    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "10/10 matched" in out


def test_the_pass_is_recorded_as_a_run(db, monkeypatch, tmp_path):
    """A rewrite of ground truth is work, and work leaves a run behind."""
    twin_pair(db, "zlib", "O0")

    monkeypatch.setattr(
        "elenchus.corpus.refresh.connect", lambda _path: db
    )
    monkeypatch.setattr(
        "elenchus.corpus.refresh.store_ground_truth",
        lambda conn, stripped_id, path: (5, 5),
    )

    cmd_refresh_truth(args_for(str(tmp_path / "x.db")))

    run = db.execute(
        "SELECT kind, tool, status, params FROM runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert run["kind"] == "extract"
    assert run["tool"] == "dwarf"
    assert run["status"] == "ok"
    assert "ground-truth-refresh" in run["params"]


@pytest.mark.parametrize("failure", [OSError("gone"), ValueError("bad")])
def test_expected_failures_are_caught_not_raised(db, monkeypatch, tmp_path,
                                                 failure):
    twin_pair(db, "zlib", "O0")

    monkeypatch.setattr(
        "elenchus.corpus.refresh.connect", lambda _path: db
    )
    monkeypatch.setattr(
        "elenchus.corpus.refresh.store_ground_truth",
        lambda *_args: (_ for _ in ()).throw(failure),
    )

    assert cmd_refresh_truth(args_for(str(tmp_path / "x.db"))) == 0
