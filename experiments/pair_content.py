"""One-off, read-only: how many contrastive positive pairs are identical code?

PairSampler picks two distinct optimisation levels of a function, then one
row at each. It never compares their content. When -O2 and -O3 compile to
the same code (common), such a pair is free: "these two are the same" is
true at a glance, the loss teaches nothing, and a batch slot is spent.

Two numbers, because they have different remedies:
  identical pairs    share of the pairs the real sampler draws whose two views
                     are the same normalised code - fixable by drawing again
  no useful pair     functions whose every view is one code - no draw can help

Counted twice, as a check on the reading of the sampler: exactly, as the
expectation over the sampler's choices, and empirically, by running
PairSampler itself for a few epochs. The two should agree.

Run from the repo root:  python experiments/pair_content.py
"""

import os
from collections import Counter, defaultdict
from itertools import permutations

from elenchus.corpus.dataset import candidate_rows
from elenchus.db import connect
from elenchus.encoder.data import PairSampler, load_examples
from elenchus.encoder.vocab import Vocab

EPOCHS = 3


def expected(views):
    """Expected identical share per level pair, and overall, over the sampler.

    views: {identity: {level: [example, ...]}}. For each identity the sampler
    takes an ordered pair of distinct levels uniformly, then one row of each
    uniformly, so P(identical) is exact to compute.
    """
    per_pair = defaultdict(lambda: [0.0, 0.0])   # level pair -> [weight, identical]
    total = identical = 0.0
    for levels in views.values():
        ordered = list(permutations(sorted(levels), 2))
        for first, second in ordered:
            a, b = levels[first], levels[second]
            same = sum(x.content == y.content for x in a for y in b) / (len(a) * len(b))
            weight = 1 / len(ordered)
            key = tuple(sorted((first, second)))
            per_pair[key][0] += weight
            per_pair[key][1] += weight * same
            total += weight
            identical += weight * same
    return identical / total, {k: (w, s / w) for k, (w, s) in sorted(per_pair.items())}


def empirical(sampler, epochs=EPOCHS):
    drawn = Counter()
    same = Counter()
    for epoch in range(epochs):
        for batch in sampler.batches(epoch):
            for first, second, a, b in zip(batch.anchor_levels, batch.positive_levels,
                                           batch.anchor_content, batch.positive_content):
                key = tuple(sorted((first, second)))
                drawn[key] += 1
                same[key] += a == b
    return drawn, same


def report(conn, vocab, batch_size=64):
    examples = load_examples(conn, "train", candidate_rows(conn))
    sampler = PairSampler(examples, vocab, batch_size=batch_size)
    views = sampler.views

    one_code = sum(1 for levels in views.values()
                   if len({e.content for rows in levels.values() for e in rows}) == 1)
    overall, per_pair = expected(views)
    drawn, same = empirical(sampler)
    return {"pairable": len(views), "one_code": one_code, "expected": overall,
            "per_pair": per_pair, "drawn": drawn, "same": same}


def show(r):
    print(f"pairable train functions      : {r['pairable']:,}")
    print(f"  every view the same code    : {r['one_code']:,} "
          f"({r['one_code'] / r['pairable']:.1%})  <- no draw can help these")
    total_drawn = sum(r["drawn"].values())
    total_same = sum(r["same"].values())
    print(f"identical pairs, expected     : {r['expected']:.1%}")
    if total_drawn:
        print(f"identical pairs, drawn        : {total_same / total_drawn:.1%}  "
              f"({total_same:,} of {total_drawn:,} over {EPOCHS} epochs)")
    print(f"\n{'level pair':10} {'share of draws':>15} {'identical exp.':>15} "
          f"{'identical drawn':>16}")
    weight_total = sum(w for w, _ in r["per_pair"].values())
    for key, (weight, share) in r["per_pair"].items():
        drawn = r["drawn"][key]
        seen = r["same"][key] / drawn if drawn else 0.0
        print(f"{'-'.join(key):10} {weight / weight_total:>15.1%} {share:>15.1%} "
              f"{seen:>16.1%}")


def main():
    conn = connect(os.environ["ELENCHUS_DB"])
    show(report(conn, Vocab.load("models/vocab.json")))


if __name__ == "__main__":
    main()
