"""Tests for corpus storage: membership and ground-truth matching.

These run without Ghidra. store_ground_truth reads DWARF from the committed
fixture DLL and matches it against functions we insert by hand, so the
matching logic is tested without needing a real stripped scan.
"""

import json
from pathlib import Path

from elenchus.corpus.dwarf import ground_truth
from elenchus.corpus.store import (
    link_twins,
    register_corpus_binary,
    store_ground_truth,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.dll"


def _binary(db, sha):
    cur = db.execute(
        "INSERT INTO binaries (path, sha256, arch, imported_at) "
        "VALUES ('x', ?, 'a', datetime('now'))",
        (sha,),
    )
    return cur.lastrowid


def test_register_corpus_binary(db):
    """A corpus member records its package and build."""
    bid = _binary(db, "sha-a")
    register_corpus_binary(db, bid, "zlib", "gcc", "O2", stripped=True, version="1.3.1")

    row = db.execute(
        "SELECT * FROM corpus_binaries WHERE binary_id = ?", (bid,)
    ).fetchone()
    assert row["package"] == "zlib"
    assert row["opt_level"] == "O2"
    assert row["stripped"] == 1


def test_link_twins_points_both_ways(db):
    """Debug and stripped twins reference each other."""
    debug = _binary(db, "sha-debug")
    strip = _binary(db, "sha-strip")
    register_corpus_binary(db, debug, "zlib", "gcc", "O2", stripped=False)
    register_corpus_binary(db, strip, "zlib", "gcc", "O2", stripped=True)
    link_twins(db, debug, strip)

    d = db.execute(
        "SELECT twin_id FROM corpus_binaries WHERE binary_id = ?", (debug,)
    ).fetchone()
    s = db.execute(
        "SELECT twin_id FROM corpus_binaries WHERE binary_id = ?", (strip,)
    ).fetchone()
    assert d["twin_id"] == strip
    assert s["twin_id"] == debug


def test_ground_truth_matches_by_address(db):
    """Ground truth attaches to a function at the same address."""
    bid = _binary(db, "sha-strip2")

    # Insert stripped functions at the fixture's real addresses, so matching
    # by address succeeds without a Ghidra scan.
    truth = list(ground_truth(FIXTURE))
    assert truth, "fixture should carry DWARF"

    target = next(f for f in truth if f.name == "add")
    db.execute(
        "INSERT INTO functions (binary_id, address, raw_name, is_external) "
        "VALUES (?, ?, 'FUN_x', 0)",
        (bid, target.address),
    )
    db.commit()

    total, matched = store_ground_truth(db, bid, FIXTURE)
    assert total == len(truth)
    assert matched >= 1

    row = db.execute(
        "SELECT gt.name, gt.return_type, gt.param_types "
        "FROM ground_truth gt "
        "JOIN functions f ON f.id = gt.function_id "
        "WHERE gt.binary_id = ? AND gt.name = 'add'",
        (bid,),
    ).fetchone()
    assert row["name"] == "add"
    assert row["return_type"] == "int"
    assert json.loads(row["param_types"]) == ["int", "int"]


def test_unmatched_ground_truth_is_still_stored(db):
    """A ground-truth function with no stripped match is stored, id null."""
    bid = _binary(db, "sha-strip3")
    # No functions inserted, so nothing can match.
    total, matched = store_ground_truth(db, bid, FIXTURE)

    assert total > 0
    assert matched == 0

    nulls = db.execute(
        "SELECT COUNT(*) n FROM ground_truth "
        "WHERE binary_id = ? AND function_id IS NULL",
        (bid,),
    ).fetchone()
    assert nulls["n"] == total
