"""Choosing which branch to force, and in both versions the same one.

Forcing a branch is only useful if the two builds are forced at branches that
*correspond*. We cannot know which conditional jump in -O0 is which in -O3 -
that is the whole difficulty - but PEM's answer is that we do not have to:
rank each version's branches by how close its comparison came to going the
other way, and pick at the same rank in each. Optimisations remove, duplicate
and move predicates but do not invent them, so the extremes of that ranking
are the part most likely to survive; they measure it holding over 80% of the
time (docs/related-work.md).

The ranking and the sampling are arithmetic over recorded branches, so they
live here and are tested without an emulator. Nothing in this module decides
a verdict: a forced path is infeasible and can only ever corroborate.
"""

from __future__ import annotations

import random

# PEM samples the ranked list from a Beta with both shape parameters small,
# which is U-shaped: the branches whose comparison was closest and furthest
# from flipping are favoured and the middle is rare. Their value, kept until
# measured here (F28's lesson - a borrowed idea is worth having, a borrowed
# parameter has to be measured).
BETA_SHAPE = 0.03


def ranked(predicates) -> list:
    """The branches worth forcing, ordered by selectivity, least first.

    A branch whose compare could not be read has no selectivity and cannot be
    ranked, so it is left out: including it would put an unrankable branch at
    a position the other version's list does not mean.
    """
    return sorted((p for p in predicates if p.selectivity is not None),
                  key=lambda p: (p.selectivity, p.count))


def percentiles(count: int, seed: int) -> list[float]:
    """`count` positions in [0, 1], U-shaped and reproducible.

    The same seed gives the same positions, so both versions are asked for the
    same ranks and a verdict stays replayable.
    """
    rng = random.Random(f"elenchus-flip-{seed}")
    return [rng.betavariate(BETA_SHAPE, BETA_SHAPE) for _ in range(count)]


def at_percentile(order: list, position: float):
    """The branch at `position` through a ranked list, or None if empty."""
    if not order:
        return None
    index = int(round(position * (len(order) - 1)))
    return order[max(0, min(index, len(order) - 1))]


def flip_directive(predicate) -> dict | None:
    """What `run(force=...)` needs to send this branch the other way.

    Keyed by instruction count, because the same jump inside a loop is a
    different instance each time round and only one of them is meant.
    """
    if predicate is None:
        return None
    return {predicate.count: predicate.branch.other(predicate.taken)}


def corresponding_flips(q_predicates, k_predicates, attempts: int, seed: int):
    """Pairs of directives, one per version, at matching ranks.

    Yields (q_directive, k_directive) for each attempt. A rank that exists in
    one version and not the other yields nothing for that attempt: forcing
    only one side would compare a forced path against an ordinary one, which
    says nothing about either.

    Duplicate pairs are skipped - the U-shape lands on the extremes often, and
    re-running an attempt already made buys nothing.
    """
    q_order = ranked(q_predicates)
    k_order = ranked(k_predicates)
    seen = set()
    for position in percentiles(attempts, seed):
        q_choice = at_percentile(q_order, position)
        k_choice = at_percentile(k_order, position)
        if q_choice is None or k_choice is None:
            continue
        key = (q_choice.count, k_choice.count)
        if key in seen:
            continue
        seen.add(key)
        yield flip_directive(q_choice), flip_directive(k_choice)
