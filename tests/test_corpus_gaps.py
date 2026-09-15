"""Tests for known corpus gaps.

A gap record changes what checks say and what builds do, so it must never be
able to say something false. The tests below try to record untrue gaps,
leave a gap standing after the level is stored, and hide a real missing level
behind a recorded one - and check that each is refused or reported.
"""

import types

import pytest

from elenchus.check import check_corpus_completeness, check_stale_gaps
from elenchus.corpus.gaps import (
    GapError,
    cmd_corpus_gap,
    known_gaps,
    record_gap,
    remove_gap,
)

LEVELS = ("O0", "O1", "O2", "O3")


def store_level(db, package, level, sides=(False, True)):
    for stripped in sides:
        with db:
            binary_id = db.execute(
                "INSERT INTO binaries (path, sha256, arch, imported_at) "
                "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
                (f"/tmp/{package}_{level}_{stripped}.dll",
                 f"{package}{level}{stripped}"),
            ).lastrowid
            db.execute(
                "INSERT INTO corpus_binaries "
                "(binary_id, package, compiler, opt_level, stripped) "
                "VALUES (?, ?, 'gcc', ?, ?)",
                (binary_id, package, level, int(stripped)),
            )


@pytest.fixture
def quickjs(db):
    """The real case: three levels stored, -O1 lost to a heap error."""
    for level in ("O0", "O2", "O3"):
        store_level(db, "quickjs", level)
    return db


# ---------------------------------------------------------------- recording


def test_a_missing_level_can_be_recorded_with_its_reason(quickjs):
    record_gap(quickjs, "quickjs", "O1", "Ghidra heap exhausted at 4G and 6G")
    assert known_gaps(quickjs) == {
        "quickjs": {"O1": "Ghidra heap exhausted at 4G and 6G"}
    }


def test_a_stored_level_cannot_be_called_missing(quickjs):
    with pytest.raises(GapError, match="is stored"):
        record_gap(quickjs, "quickjs", "O2", "untrue")


def test_a_level_with_only_one_twin_counts_as_missing(db):
    store_level(db, "half", "O0", sides=(True,))
    store_level(db, "half", "O1")
    record_gap(db, "half", "O0", "debug twin lost")
    assert "O0" in known_gaps(db)["half"]


def test_a_package_not_in_the_corpus_has_no_gap_to_record(db):
    with pytest.raises(GapError, match="nothing in the corpus"):
        record_gap(db, "never-built", "O1", "reason")


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_a_gap_without_a_reason_is_refused(quickjs, reason):
    with pytest.raises(GapError, match="reason"):
        record_gap(quickjs, "quickjs", "O1", reason)


def test_an_unknown_level_is_refused(quickjs):
    with pytest.raises(GapError, match="unknown level"):
        record_gap(quickjs, "quickjs", "Os", "reason")


def test_recording_twice_does_not_overwrite_the_first_reason(quickjs):
    record_gap(quickjs, "quickjs", "O1", "first")
    with pytest.raises(GapError, match="already recorded"):
        record_gap(quickjs, "quickjs", "O1", "second")
    assert known_gaps(quickjs)["quickjs"]["O1"] == "first"


def test_a_gap_can_be_removed(quickjs):
    record_gap(quickjs, "quickjs", "O1", "reason")
    remove_gap(quickjs, "quickjs", "O1")
    assert known_gaps(quickjs) == {}


def test_removing_a_gap_that_is_not_recorded_is_refused(quickjs):
    with pytest.raises(GapError, match="not recorded"):
        remove_gap(quickjs, "quickjs", "O1")


def test_the_database_itself_rejects_an_unknown_level(quickjs):
    """The CHECK constraint holds even for a writer that skips record_gap."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        with quickjs:
            quickjs.execute(
                "INSERT INTO corpus_gaps VALUES ('quickjs', 'O9', 'x', datetime('now'))"
            )


# ---------------------------------------------------------------- checks


def test_a_missing_level_is_warned_about_by_name(quickjs):
    assert check_corpus_completeness(quickjs) == [("quickjs", ("O1",))]


def test_a_recorded_gap_is_not_warned_about(quickjs):
    record_gap(quickjs, "quickjs", "O1", "reason")
    assert check_corpus_completeness(quickjs) == []


def test_a_recorded_gap_cannot_hide_another_missing_level(db):
    """Recording -O1 must not quietly cover a later loss of -O3."""
    store_level(db, "pkg", "O0")
    store_level(db, "pkg", "O2")
    record_gap(db, "pkg", "O1", "reason")

    assert check_corpus_completeness(db) == [("pkg", ("O3",))]


def test_a_level_registered_twice_does_not_make_up_for_a_missing_one(db):
    for level in ("O0", "O1", "O2"):
        store_level(db, "pkg", level)
    store_level(db, "pkg-dup", "O0")
    db.execute("UPDATE corpus_binaries SET package = 'pkg' WHERE package = 'pkg-dup'")
    db.commit()

    assert check_corpus_completeness(db) == [("pkg", ("O3",))]


def test_a_gap_left_behind_after_the_level_is_stored_is_stale(quickjs):
    record_gap(quickjs, "quickjs", "O1", "heap")
    store_level(quickjs, "quickjs", "O1")

    assert check_stale_gaps(quickjs) == [("quickjs", "O1")]
    assert check_corpus_completeness(quickjs) == []


def test_no_gaps_nothing_stale(quickjs):
    assert check_stale_gaps(quickjs) == []


# ---------------------------------------------------------------- commands


def run_gap(monkeypatch, db, **kwargs):
    from elenchus.corpus import gaps as module

    monkeypatch.setattr("elenchus.db.connect", lambda _p: db)
    defaults = {"db": ":memory:", "package": None, "level": None, "reason": None}
    return module.cmd_corpus_gap(types.SimpleNamespace(**(defaults | kwargs)))


def test_the_command_records_and_lists(monkeypatch, quickjs, capsys):
    code = run_gap(monkeypatch, quickjs, action="add", package="quickjs",
                   level="O1", reason="heap exhausted")
    out = capsys.readouterr().out
    assert code == 0
    assert "recorded: quickjs -O1" in out and "heap exhausted" in out


def test_the_command_reports_a_refusal_and_fails(monkeypatch, quickjs, capsys):
    code = run_gap(monkeypatch, quickjs, action="add", package="quickjs",
                   level="O2", reason="untrue")
    assert code == 1
    assert "refused" in capsys.readouterr().out
    assert known_gaps(quickjs) == {}


def test_check_names_known_gaps_and_stays_clean(monkeypatch, quickjs, capsys):
    from elenchus import cli

    record_gap(quickjs, "quickjs", "O1", "Ghidra heap exhausted")
    monkeypatch.setattr(cli, "connect", lambda _p: quickjs)

    code = cli.cmd_check(types.SimpleNamespace(
        db=":memory:", close_stale=False, stale_minutes=60))
    out = capsys.readouterr().out

    assert code == 0
    assert "corpus completeness    ok" in out
    assert "known gaps" in out and "quickjs -O1: Ghidra heap exhausted" in out
    assert "warnings only" not in out


def test_cmd_corpus_gap_is_wired_into_the_cli():
    from elenchus import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        ["corpus-gap", "add", "quickjs", "O1", "--reason", "heap"])
    assert args.func is cmd_corpus_gap
    assert (args.package, args.level, args.reason) == ("quickjs", "O1", "heap")
