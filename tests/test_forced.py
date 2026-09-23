"""Choosing which branch to force.

Pure arithmetic over recorded branches, so it is checked on hand-built ones
where the right answer is known. What matters most here is that both versions
are asked for the *same rank*: forcing different branches in the two builds
would compare two unrelated paths and mean nothing.
"""

from elenchus.emulation.branches import Branch, PredicateInstance
from elenchus.emulation.forced import (
    at_percentile,
    corresponding_flips,
    flip_directive,
    percentiles,
    ranked,
)


def predicate(count, selectivity, taken=True, target=0x2000, fall=0x1000):
    return PredicateInstance(count, Branch(0x500, 2, target, fall), taken,
                             selectivity)


# ----------------------------------------------------------- ranking


def test_branches_are_ranked_by_how_close_the_compare_came():
    order = ranked([predicate(1, 500), predicate(2, 3), predicate(3, 90)])
    assert [p.selectivity for p in order] == [3, 90, 500]


def test_an_unrankable_branch_is_left_out():
    """A compare that could not be read has no place in the ranking: it would
    sit at a position the other version's list does not mean."""
    order = ranked([predicate(1, 7), predicate(2, None), predicate(3, 9)])
    assert [p.count for p in order] == [1, 3]


def test_ties_are_broken_by_order_of_appearance():
    """Two branches equally close must still rank the same way every run, or
    the choice stops being replayable."""
    order = ranked([predicate(9, 4), predicate(2, 4)])
    assert [p.count for p in order] == [2, 9]


# --------------------------------------------------------- sampling


def test_the_positions_are_reproducible():
    """A verdict has to be replayable, so the same seed gives the same
    choices."""
    assert percentiles(8, 5) == percentiles(8, 5)
    assert percentiles(8, 5) != percentiles(8, 6)


def test_the_sampling_favours_both_extremes():
    """PEM's U-shape: the branches whose comparison came closest and furthest
    from flipping are the ones whose ranking survives optimisation, so they
    are where the budget goes."""
    positions = percentiles(400, 1)
    middle = sum(1 for p in positions if 0.25 < p < 0.75)
    assert middle < len(positions) * 0.2        # the middle is rare


def test_a_position_maps_onto_the_list():
    order = ranked([predicate(i, i * 10) for i in range(1, 6)])
    assert at_percentile(order, 0.0).count == 1
    assert at_percentile(order, 1.0).count == 5
    assert at_percentile([], 0.5) is None


# ------------------------------------------------------- directives


def test_a_directive_sends_the_branch_the_other_way():
    taken = predicate(11, 5, taken=True, target=0x2000, fall=0x1000)
    assert flip_directive(taken) == {11: 0x1000}
    not_taken = predicate(11, 5, taken=False, target=0x2000, fall=0x1000)
    assert flip_directive(not_taken) == {11: 0x2000}


def test_a_directive_names_the_instance_not_the_address():
    """The same jump inside a loop is a different instance each time round,
    and only one of them is meant."""
    assert set(flip_directive(predicate(42, 5))) == {42}


# ------------------------------------------- the two sides together


def test_both_versions_are_forced_at_the_same_rank():
    """The whole point: -O0's third-closest branch is compared against -O3's
    third-closest, because which is which cannot be known any other way."""
    q = [predicate(1, 5), predicate(2, 50), predicate(3, 500)]
    k = [predicate(10, 7), predicate(20, 70), predicate(30, 700)]
    pairs = list(corresponding_flips(q, k, attempts=40, seed=1))
    assert pairs
    q_ranks = {1: 0, 2: 1, 3: 2}
    k_ranks = {10: 0, 20: 1, 30: 2}
    for q_directive, k_directive in pairs:
        q_count = next(iter(q_directive))
        k_count = next(iter(k_directive))
        assert q_ranks[q_count] == k_ranks[k_count]


def test_nothing_is_forced_when_one_side_has_no_branches():
    """Forcing one side only would compare a forced path against an ordinary
    one, which says nothing about either."""
    assert list(corresponding_flips([predicate(1, 5)], [], 10, 1)) == []
    assert list(corresponding_flips([], [predicate(1, 5)], 10, 1)) == []


def test_the_same_pair_is_not_tried_twice():
    """The U-shape lands on the extremes often; re-running an attempt already
    made buys nothing."""
    q = [predicate(1, 5), predicate(2, 500)]
    k = [predicate(10, 7), predicate(20, 700)]
    pairs = list(corresponding_flips(q, k, attempts=100, seed=1))
    keys = [(next(iter(a)), next(iter(b))) for a, b in pairs]
    assert len(keys) == len(set(keys))
    assert len(keys) <= 2                       # only two ranks exist
