"""R0 - the measurement of whether a false claim can be caught.

The protocol's own correctness matters as much as the verifier's: a sample
that accidentally contains true pairs would report refutation power that is
really false-refutation rate, and the number would look terrible for the
wrong reason.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("unicorn")

from elenchus.corpus.dwarf import ground_truth  # noqa: E402
from elenchus.db import connect  # noqa: E402
from elenchus.emulation.r0 import (  # noqa: E402
    R0Report,
    run_r0,
    sample_false_pairs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"


@pytest.fixture
def corpus(tmp_path):
    """A tiny corpus of the fixture binaries, enough to sample pairs from."""
    conn = connect(":memory:")
    ids = {}
    for level in ("O0", "O3"):
        path = str((FIXTURES / f"cases_{level}.dll").resolve())
        ids[level] = conn.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86-64', 'now') RETURNING id",
            (path, "sha-" + level)).fetchone()["id"]
        conn.execute(
            "INSERT INTO corpus_binaries (binary_id, package, version, "
            "compiler, opt_level, stripped) VALUES (?, 'demo', '1', 'gcc', ?, 1)",
            (ids[level], level))
    conn.execute("INSERT INTO dataset_split (package, split) VALUES "
                 "('demo','train')")
    wanted = {"add3", "mix64", "sum", "low_byte", "is_odd", "count_nonzero"}
    for level in ("O0", "O3"):
        for f in ground_truth(FIXTURES / f"cases_{level}.dll"):
            if f.name in wanted:
                conn.execute(
                    "INSERT INTO ground_truth (binary_id, address, name, "
                    "decl_file, abi) VALUES (?, ?, ?, 'cases.c', ?)",
                    (ids[level], f.address, f.name, json.dumps(f.abi)))
    conn.commit()
    return conn


# ------------------------------------------------------ the sample itself


def test_every_sampled_pair_is_actually_false(corpus):
    """The whole measurement rests on this: a true pair in the sample would
    turn a correct refusal to refute into a reported miss."""
    pairs = sample_false_pairs(corpus, count=40, seed=0)
    assert pairs
    for q, k in pairs:
        assert q["name"] != k["name"]


def test_the_sample_is_reproducible(corpus):
    """A measurement that moves on its own cannot be compared across
    changes, which is the only thing it is for."""
    first = [(q["name"], k["name"]) for q, k in
             sample_false_pairs(corpus, count=20, seed=1)]
    again = [(q["name"], k["name"]) for q, k in
             sample_false_pairs(corpus, count=20, seed=1)]
    assert first == again
    other = [(q["name"], k["name"]) for q, k in
             sample_false_pairs(corpus, count=20, seed=2)]
    assert first != other


def test_both_sides_come_from_one_package(corpus):
    """Two functions from one library is the confusion a model would make;
    two from unrelated worlds would be too easy and flatter the number."""
    for q, k in sample_false_pairs(corpus, count=20, seed=0):
        assert q["package"] == k["package"]


def test_the_sample_stops_rather_than_spinning(corpus):
    """Asking for more pairs than the corpus can make must end."""
    pairs = sample_false_pairs(corpus, count=10_000, seed=0)
    assert isinstance(pairs, list)


# ------------------------------------------------------------ the report


def test_the_report_counts_what_it_is_asked_to():
    report = R0Report()
    report.add("refuted")
    report.add("survived")
    report.add("survived", forced_only=True)
    report.add("inconclusive")
    assert (report.total, report.refuted, report.survived,
            report.inconclusive) == (4, 1, 2, 1)
    assert report.forced_survivals == 1


def test_a_forced_survival_is_counted_apart():
    """A survival resting only on forced paths is weaker evidence and is not
    folded in with a feasible one, here as everywhere."""
    report = R0Report()
    report.add("survived")
    report.add("survived", forced_only=True)
    assert report.survived == 2 and report.forced_survivals == 1


# -------------------------------------------------------------- end to end


def test_r0_runs_and_refutes_at_least_something(corpus):
    """On fixtures that genuinely differ - add3 against mix64 and the rest -
    the verifier must catch some of them, or it is not a verifier."""
    report = run_r0(corpus, count=25, seed=0)
    assert report.total > 0
    assert report.refuted + report.survived + report.inconclusive == report.total
    assert report.refuted > 0


# ---------------------------------------------------- what is not judged


def test_a_pair_that_cannot_be_opened_is_counted_not_dropped(corpus):
    """A binary that cannot be opened leaves its pairs unjudged, and the
    report says how many: a run that judged fewer pairs than it was asked
    to must not look like one that judged them all."""
    corpus.execute("UPDATE binaries SET path = '/nonexistent/cases_O3.dll' "
                   "WHERE sha256 = 'sha-O3'")
    report = run_r0(corpus, count=10, seed=0)
    assert report.total == 0
    assert sum(report.unjudged.values()) == 10
    assert all(reason.startswith("load failed") for reason in report.unjudged)


def test_a_failure_inside_the_comparison_is_not_swallowed(monkeypatch):
    """Only opening the binaries is guarded. An error in the comparison is
    the verifier's own, and _close raises on purpose when Unicorn can no
    longer be closed (F46): swallowed, it would leave R0 reporting rates
    over whatever pairs happened not to fail."""
    from elenchus.emulation import r0

    def broken(*args, **kwargs):
        raise RuntimeError("the comparison failed")

    monkeypatch.setattr(r0, "compare", broken)
    path = str((FIXTURES / "cases_O0.dll").resolve())
    abi = json.dumps({"params": [], "return": {"kind": "void", "size": 0}})
    with pytest.raises(RuntimeError, match="the comparison failed"):
        r0.judge_false_pair((path, 0, abi, path, 0, 5_000, "demo/a vs b"))


def test_what_was_not_judged_stays_out_of_the_rates():
    report = R0Report()
    report.add("refuted")
    report.unjudged["load failed: FileNotFoundError"] += 1
    assert (report.total, report.refuted) == (1, 1)
