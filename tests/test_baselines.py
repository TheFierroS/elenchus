"""Tests for the retrieval task, its metrics, and the baselines.

A wrong metric is worse than a missing one: it sets the bar the encoder is
judged against, and nothing downstream would look wrong. So the arithmetic
is pinned against cases whose answers can be worked out by hand, and the
baselines are checked for the one property that makes them meaningful -
that they do better than chance on a task where the answer is findable.
"""

import pytest

from elenchus.evaluation.baselines import (
    BM25Mnemonics,
    ImportJaccard,
    RandomBaseline,
    StructuralBaseline,
    all_baselines,
)
from elenchus.evaluation.metrics import evaluate, format_table, rank_of
from elenchus.evaluation.task import Sample, equivalence_classes


def sample(name, opcodes, imports=(), blocks=1, package="p", opt="O0"):
    """Build a Sample from an opcode list and a set of imported calls."""
    lines = []
    for index, opcode in enumerate(opcodes):
        target = ""
        if index < len(imports):
            target = f"IMPORT:KERNEL32.dll!{list(imports)[index]}"
        lines.append(f"{index:x}\t{opcode}\tRAX\t{target}")

    return Sample(
        function_id=abs(hash((name, opt))) % 10**6,
        key=(package, "f.c", name),
        name=name,
        package=package,
        opt_level=opt,
        listing="\n".join(lines),
        n_instructions=len(opcodes),
        n_blocks=blocks,
    )


# ---------------------------------------------------------------- metrics


def test_rank_is_one_when_the_answer_wins_outright():
    assert rank_of([0.9, 0.1, 0.2], 0) == 1


def test_rank_counts_only_strictly_better_candidates():
    assert rank_of([0.1, 0.5, 0.9], 0) == 3


def test_ties_are_scored_by_their_average_position():
    """Four equal scores are a non-answer and must not round to first place."""
    assert rank_of([1.0, 1.0, 1.0, 1.0], 0) == 2.5
    # One candidate clearly ahead, then a three-way tie for second.
    assert rank_of([2.0, 1.0, 1.0, 1.0], 1) == 3


def test_any_of_several_correct_answers_counts():
    """Identical functions cannot be told apart, so either one answers."""
    # Two acceptable answers tied at the top: picking one is certain to be
    # right, so this is rank 1, not a coin flip scored as 1.5.
    assert rank_of([1.0, 1.0, 0.0], {0, 1}) == 1

    # Two acceptable answers inside a four-way tie: shuffling and taking the
    # first acceptable one lands at 5/3 on average.
    assert rank_of([1.0, 1.0, 1.0, 1.0], {0, 1}) == pytest.approx(5 / 3)

    # An acceptable answer behind one clearly better wrong candidate.
    assert rank_of([5.0, 1.0, 1.0], {1, 2}) == pytest.approx(1 + 3 / 3)


def test_a_wrong_tie_still_costs():
    """Equivalence must not launder a baseline that cannot discriminate."""
    # One correct answer buried in a ten-way tie scores as badly as before.
    assert rank_of([1.0] * 10, {0}) == 5.5


class Fixed:
    """A scorer returning scores decided by the test."""

    name = "fixed"

    def __init__(self, table):
        self.table = table

    def prepare(self, pool):
        self.pool = pool

    def scores(self, query):
        return self.table[query.name]


def test_metrics_are_computed_as_defined():
    queries = [sample("a", ["mov"]), sample("b", ["mov"]), sample("c", ["mov"])]
    pool = [sample("a", ["mov"]), sample("b", ["mov"]), sample("c", ["mov"])]
    gold = [0, 1, 2]

    scorer = Fixed({
        "a": [9.0, 1.0, 0.0],   # answer first
        "b": [9.0, 5.0, 0.0],   # answer second
        "c": [9.0, 5.0, 1.0],   # answer third
    })

    metrics = evaluate(scorer, queries, pool, gold, cutoffs=(1, 2))

    assert metrics["queries"] == 3
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@2"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert metrics["median_rank"] == 2


def test_packages_are_summarised_alone_and_averaged_with_equal_weight():
    """Two queries from a big package, one from a small one: the query mean
    lets the big package count twice, the package mean does not."""
    queries = [sample("a", ["mov"], package="big"), sample("b", ["mov"], package="big"),
               sample("c", ["mov"], package="small")]
    pool = [sample(n, ["mov"]) for n in ("a", "b", "c", "d")]
    scorer = Fixed({
        "a": [9.0, 1.0, 1.0, 0.0],   # rank 1
        "b": [9.0, 5.0, 1.0, 0.0],   # rank 2
        "c": [9.0, 8.0, 1.0, 5.0],   # rank 4
    })
    metrics = evaluate(scorer, queries, pool, [0, 1, 2], cutoffs=(1, 2))

    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)
    assert metrics["per_package"]["big"] == pytest.approx(
        {"queries": 2, "recall@1": 1 / 2, "recall@2": 1.0, "mrr": 3 / 4,
         "median_rank": 2})
    assert metrics["per_package"]["small"]["mrr"] == pytest.approx(1 / 4)
    assert metrics["package_mean"] == pytest.approx(
        {"packages": 2, "recall@1": 1 / 4, "recall@2": 1 / 2, "mrr": (3 / 4 + 1 / 4) / 2})
    assert list(metrics["per_package"]) == ["big", "small"]


def test_queries_without_a_package_get_no_package_summary():
    import types

    queries = [types.SimpleNamespace(name="a")]
    metrics = evaluate(Fixed({"a": [1.0, 0.0]}), queries, [None, None], [0])
    assert "per_package" not in metrics and "package_mean" not in metrics
    assert metrics["mrr"] == 1.0


def test_the_package_table_lists_big_packages_first_and_ends_with_both_means():
    from elenchus.evaluation.metrics import format_packages

    def result(mrr, packages):
        return {"mrr": mrr, "recall@10": 0.0,
                "package_mean": {"mrr": sum(v for _, v in packages.values()) / 2,
                                 "recall@10": 0.0},
                "per_package": {p: {"queries": q, "mrr": v, "recall@10": 0.0}
                                for p, (q, v) in packages.items()}}

    lines = format_packages({
        "strong": result(0.9, {"small": (3, 0.8), "big": (40, 0.95)}),
        "weak": result(0.1, {"big": (40, 0.1)}),
    })
    assert lines[0].strip() == "mrr per package"
    assert lines[1].split()[-2:] == ["weak", "strong"]
    assert lines[2].split()[0] == "big" and lines[3].split()[0] == "small"
    assert lines[3].split()[-2:] == ["-", "0.800"]
    assert lines[-2].split()[-2:] == ["0.100", "0.900"]
    assert "mean over packages" in lines[-1] and lines[-1].split()[-1] == "0.875"
    assert format_packages({"bare": {"mrr": 0.5}}) == []


def test_an_empty_task_reports_nothing_rather_than_dividing_by_zero():
    assert evaluate(Fixed({}), [], [], []) == {"queries": 0}


def test_table_puts_the_strongest_last():
    lines = format_table({
        "weak": {"queries": 1, "pool": 1, "mrr": 0.1},
        "strong": {"queries": 1, "pool": 1, "mrr": 0.9},
    })
    assert "strong" in lines[-1]


def test_identical_functions_form_one_answer():
    """The sanity check that caught this: -O0 against -O0 should be perfect.

    It was not, because mpc contains dozens of functions compiling to the
    same code, and the metric was calling every choice but one wrong. The
    fix is to group them; without it the ceiling of every later measurement
    sits wherever the corpus happens to have duplicates.
    """
    pool = [
        sample("wrapper_a", ["push", "call", "pop", "ret"]),
        sample("wrapper_b", ["push", "call", "pop", "ret"]),
        sample("real_work", ["xor", "shl", "rol", "add", "mul"]),
    ]

    classes = equivalence_classes(pool)
    assert classes[0] == {0, 1}
    assert classes[1] == {0, 1}
    assert classes[2] == {2}


# -------------------------------------------------------------- baselines


def test_random_is_reproducible_and_unaware():
    pool = [sample(f"f{i}", ["mov"]) for i in range(5)]

    first, second = RandomBaseline(seed=7), RandomBaseline(seed=7)
    first.prepare(pool)
    second.prepare(pool)

    query = sample("f0", ["mov"])
    assert first.scores(query) == second.scores(query)
    assert RandomBaseline(seed=8).scores is not None


def test_structural_prefers_the_similar_size():
    pool = [
        sample("tiny", ["mov"] * 4, blocks=1),
        sample("huge", ["mov"] * 400, blocks=40),
    ]
    scorer = StructuralBaseline()
    scorer.prepare(pool)

    scores = scorer.scores(sample("query", ["mov"] * 380, blocks=38))
    assert scores[1] > scores[0]


def test_import_overlap_is_jaccard():
    pool = [
        sample("same", ["call"] * 3, imports=["malloc", "free", "memcpy"]),
        sample("other", ["call"] * 3, imports=["socket", "bind", "listen"]),
    ]
    scorer = ImportJaccard()
    scorer.prepare(pool)

    query = sample("q", ["call"] * 3, imports=["malloc", "free", "memcpy"])
    scores = scorer.scores(query)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0.0


def test_a_function_with_no_imports_claims_no_similarity():
    """Otherwise everything without imports would match everything else."""
    pool = [sample("x", ["mov"]), sample("y", ["mov"])]
    scorer = ImportJaccard()
    scorer.prepare(pool)
    assert scorer.scores(sample("q", ["mov"])) == [0.0, 0.0]


def test_bm25_ranks_the_matching_opcode_sequence_first():
    pool = [
        sample("other", ["push", "pop", "ret"]),
        sample("target", ["xor", "shl", "rol", "xor", "shl"]),
        sample("noise", ["add", "add", "add"]),
    ]
    scorer = BM25Mnemonics()
    scorer.prepare(pool)

    scores = scorer.scores(sample("q", ["xor", "shl", "rol", "xor", "shl"]))
    assert scores[1] == max(scores)


def test_bm25_discounts_windows_everything_shares():
    """A sequence present in every function tells you nothing about which."""
    common = ["push", "mov", "pop"]
    pool = [sample(f"f{i}", common) for i in range(20)]
    pool.append(sample("rare", common + ["aesenc", "aesenc", "aesenc"]))

    scorer = BM25Mnemonics()
    scorer.prepare(pool)
    scores = scorer.scores(sample("q", common + ["aesenc", "aesenc", "aesenc"]))

    assert scores[-1] == max(scores)
    assert scores[-1] > 2 * max(scores[:-1])


def test_short_functions_do_not_break_the_n_gram_window():
    pool = [sample("one", ["ret"]), sample("two", ["mov", "ret"])]
    scorer = BM25Mnemonics(n=3)
    scorer.prepare(pool)
    assert len(scorer.scores(sample("q", ["ret"]))) == 2


def test_every_baseline_beats_chance_on_a_findable_task():
    """The property that makes a bar a bar.

    Ten functions, each with a distinctive opcode sequence, an import set,
    and a size. Every baseline except random should find the answer more
    often than one in ten - and random should not.
    """
    families = [
        (["xor", "shl", "rol"], ["CryptEncrypt"]),
        (["add", "imul", "sub"], ["malloc"]),
        (["cmp", "jz", "mov"], ["strcmp"]),
        (["movss", "mulss", "addss"], ["powf"]),
        (["push", "call", "pop"], ["CreateFileW"]),
    ]

    pool, queries, gold = [], [], []
    for index, (opcodes, imports) in enumerate(families):
        size = 5 * (index + 1)
        pool.append(sample(f"f{index}", opcodes * size, imports, blocks=index + 1))
        queries.append(
            sample(f"f{index}", opcodes * size, imports, blocks=index + 1, opt="O3")
        )
        gold.append(index)

    for scorer in all_baselines():
        metrics = evaluate(scorer, queries, pool, gold)
        if scorer.name == "random":
            continue
        assert metrics["recall@1"] > 0.5, f"{scorer.name} found nothing"


def test_bm25_normalises_for_length_by_default():
    """Chosen by measurement on val, not by convention.

    The usual default is 0.75. On cross-optimisation retrieval, where the
    query side is long and the pool side short, full normalisation is worth
    a third again in mrr - and the opposite was expected before it was
    measured, which is why the number is pinned here rather than left to a
    library default that would drift.
    """
    assert BM25Mnemonics().b == 1.0


def test_a_short_exact_match_beats_a_long_catch_all():
    """What the penalty is for, in one case.

    A long pool entry containing a bit of everything answers every query a
    little. A short entry that is genuinely the same function answers one
    query exactly. The first should not win, and at b=1 it does not.

    A single synthetic case cannot settle the parameter - the evidence for
    that is the sweep over val, where mrr rises monotonically from 0.037 at
    b=0 to 0.123 at b=1. This only keeps the behaviour legible.
    """
    opcodes = ["xor", "shl", "rol"]
    pool = [
        sample("catch_all", (opcodes + ["mov", "add", "sub"]) * 30),
        sample("target", opcodes * 3),
    ]

    scorer = BM25Mnemonics()
    scorer.prepare(pool)
    scores = scorer.scores(sample("q", opcodes * 3, opt="O0"))

    assert scores[1] > scores[0]
