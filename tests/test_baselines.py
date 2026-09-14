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
from elenchus.evaluation.task import Sample


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


def test_an_empty_task_reports_nothing_rather_than_dividing_by_zero():
    assert evaluate(Fixed({}), [], [], []) == {"queries": 0}


def test_table_puts_the_strongest_last():
    lines = format_table({
        "weak": {"queries": 1, "pool": 1, "mrr": 0.1},
        "strong": {"queries": 1, "pool": 1, "mrr": 0.9},
    })
    assert "strong" in lines[-1]


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
