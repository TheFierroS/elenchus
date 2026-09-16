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

An answer can also be more than one function. Libraries contain functions
that compile to identical code - a parser combinator library is half made of
them - and no method can be asked to pick between things that are the same.
So a query is scored against the whole set of functions indistinguishable
from its answer, and any of them counts.

Every number is also given per package, and averaged over packages with
each package weighing the same. The query mean lets big packages speak for
the rest: 65% of val identities come from three packages (stb, libuv,
libsodium), and test's largest package alone holds 25%. A model chosen by
the query mean may be the one best at those three. The package mean is
reported beside it, not instead of it, so that whether the two ever
disagree is a measurement rather than a worry.
"""

from collections import OrderedDict, defaultdict
from statistics import fmean


def rank_of(scores, targets, tolerance=1e-12):
    """Return the 1-based rank of the best acceptable answer.

    targets may be a single index or a set of them, all equally correct.

    Ties are resolved as an expectation rather than a best case. Among a band
    of n equally scored candidates of which g are acceptable, shuffling and
    taking the first acceptable one lands at position (n + 1) / (g + 1) on
    average - which is 1 when the band holds only correct answers, and the
    familiar (n + 1) / 2 when it holds exactly one.
    """
    if isinstance(targets, int):
        targets = (targets,)
    targets = set(targets)

    best = max(scores[index] for index in targets)

    better = 0
    tied_wrong = 0
    tied_right = 0
    for index, score in enumerate(scores):
        if score > best + tolerance:
            if index not in targets:
                better += 1
        elif abs(score - best) <= tolerance:
            if index in targets:
                tied_right += 1
            else:
                tied_wrong += 1

    band = tied_right + tied_wrong
    return better + (band + 1) / (tied_right + 1)


def evaluate(scorer, queries, pool, gold, cutoffs=(1, 10)):
    """Run one scorer over the whole task and return its metrics.

    scorer.prepare(pool) is called once, then scorer.scores(query) per query,
    returning one number per pool entry. Splitting it that way lets an index
    (BM25's, or later the encoder's embedded pool) be built a single time.
    A scorer that can also work on all queries at once (the encoder batches
    their forward passes) offers prepare_queries, called after prepare.

    Queries that carry a package (every Sample does) are also summarised per
    package, under per_package, and averaged over packages, under
    package_mean.
    """
    scorer.prepare(pool)
    if hasattr(scorer, "prepare_queries"):
        scorer.prepare_queries(queries)

    ranks = [
        rank_of(scorer.scores(query), answers)
        for query, answers in zip(queries, gold)
    ]

    if not ranks:
        return {"queries": 0}
    result = OrderedDict(queries=len(ranks), pool=len(pool))
    result.update(summarise(ranks, cutoffs))

    packages = [getattr(query, "package", None) for query in queries]
    if all(package is not None for package in packages):
        grouped = defaultdict(list)
        for package, rank in zip(packages, ranks):
            grouped[package].append(rank)
        per_package = {package: {"queries": len(group), **summarise(group, cutoffs)}
                       for package, group in sorted(grouped.items())}
        averaged = [f"recall@{k}" for k in cutoffs] + ["mrr"]
        result["package_mean"] = {"packages": len(per_package),
                                  **{key: fmean(m[key] for m in per_package.values())
                                     for key in averaged}}
        result["per_package"] = per_package
    return result


def summarise(ranks, cutoffs=(1, 10)):
    """recall@k, MRR and median rank of a non-empty list of ranks."""
    result = OrderedDict()
    for k in cutoffs:
        result[f"recall@{k}"] = sum(1 for r in ranks if r <= k) / len(ranks)
    result["mrr"] = sum(1 / r for r in ranks) / len(ranks)
    result["median_rank"] = sorted(ranks)[len(ranks) // 2]
    return result


def format_packages(results, metric="mrr"):
    """Render one metric per package, a column per method, as a table.

    Packages are listed largest first, with their query count; the last two
    rows are the mean over queries and the mean over packages, which is where
    the two can be compared. Methods run in the same order as format_table.
    """
    measured = {name: m for name, m in results.items() if "per_package" in m}
    if not measured:
        return []
    order = sorted(measured, key=lambda name: measured[name].get("mrr", 0))
    sizes = {}
    for metrics in measured.values():
        for package, row in metrics["per_package"].items():
            sizes[package] = max(sizes.get(package, 0), row["queries"])
    widths = [max(9, len(name)) for name in order]

    def line(label, cells):
        return f"  {label:<22}" + " ".join(f"{c:>{w}}" for c, w in zip(cells, widths))

    lines = [f"  {metric} per package", line("package (queries)", order)]
    for package in sorted(sizes, key=lambda p: (-sizes[p], p)):
        cells = [f"{measured[n]['per_package'][package][metric]:.3f}"
                 if package in measured[n]["per_package"] else "-" for n in order]
        lines.append(line(f"{package} ({sizes[package]})", cells))
    lines.append(line("mean over queries", [f"{measured[n][metric]:.3f}" for n in order]))
    lines.append(line("mean over packages",
                      [f"{measured[n]['package_mean'][metric]:.3f}" for n in order]))
    return lines


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
