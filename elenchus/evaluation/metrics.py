"""Score a ranking: recall@k and mean reciprocal rank.

Three numbers, because each hides something the others show.

recall@1 asks whether the first guess was right. It is the strictest and the
easiest to understand, and it throws away the difference between a near miss
and a wild one.

recall@10 asks whether the answer was anywhere in the top ten. This is the
number that matters most for Elenchus in particular: the encoder is a
proposer, not a judge. Ten candidates that emulation can test one by one is
a perfectly good result. One confident wrong answer is not.

MRR asks how far down the answer was, giving 1 for first place, a half for
second, a fifth for fifth. It is the one that notices a system quietly
getting better while recall@1 sits still.

Ties are scored by average rank, not by best case. A baseline that gives
four hundred functions the same score has not found anything, and rounding
that in its favour would let a bad baseline look like a good one - which
would then set too high a bar for the encoder, or too low, depending on
which way it broke.
"""

from collections import OrderedDict


def rank_of(scores, target, tolerance=1e-12):
    """Return the 1-based rank of target under scores, averaging ties.

    Higher scores rank first. When several candidates tie with the target,
    all of them get the middle of the band they occupy, so a baseline
    guessing between ten equals scores as tenth-ish, not first.
    """
    value = scores[target]

    better = 0
    tied = 0
    for score in scores:
        if score > value + tolerance:
            better += 1
        elif abs(score - value) <= tolerance:
            tied += 1

    return better + (tied + 1) / 2


def evaluate(scorer, queries, pool, gold, cutoffs=(1, 10)):
    """Run one scorer over the whole task and return its metrics.

    scorer.prepare(pool) is called once, then scorer.scores(query) per query,
    returning one number per pool entry. Splitting it that way lets an index
    (BM25's, or later the encoder's embedded pool) be built a single time.
    """
    scorer.prepare(pool)

    ranks = [
        rank_of(scorer.scores(query), answer)
        for query, answer in zip(queries, gold)
    ]

    if not ranks:
        return {"queries": 0}

    result = OrderedDict(queries=len(ranks), pool=len(pool))
    for k in cutoffs:
        result[f"recall@{k}"] = sum(1 for r in ranks if r <= k) / len(ranks)
    result["mrr"] = sum(1 / r for r in ranks) / len(ranks)
    result["median_rank"] = sorted(ranks)[len(ranks) // 2]

    return result


def format_table(results):
    """Render {name: metrics} as a table, strongest by MRR last.

    Ordered deliberately: the bottom row is what the encoder has to beat, and
    it should be the last thing read.
    """
    if not results:
        return ["no results"]

    columns = ["recall@1", "recall@10", "mrr", "median_rank"]
    lines = [
        f"  {'baseline':<24} {'queries':>7} {'pool':>6} "
        + " ".join(f"{c:>11}" for c in columns)
    ]

    order = sorted(results.items(), key=lambda item: item[1].get("mrr", 0))
    for name, metrics in order:
        cells = []
        for column in columns:
            value = metrics.get(column, 0)
            cells.append(
                f"{value:>11.3f}" if column != "median_rank"
                else f"{value:>11.1f}"
            )
        lines.append(
            f"  {name:<24} {metrics.get('queries', 0):>7} "
            f"{metrics.get('pool', 0):>6} " + " ".join(cells)
        )

    return lines
