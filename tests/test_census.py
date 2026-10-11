"""The census reads where a wild address came from - exactly, or not at all.

Fault anatomy (docs/verifier.md, the census, observation 8) decides whether
repairing inputs is worth building, so its count must never be inflated. Every
test here either shows a pointer read out of our own fill being recognised, or
shows something that is not one being refused.
"""

import random
from dataclasses import dataclass

from elenchus.emulation.census import (
    BEYOND,
    CODE_FROM_FILL,
    GOLDEN_INVERSE,
    INPUT_FILL,
    MASK64,
    NULL,
    OTHER,
    REACH,
    STACK_GARBAGE,
    UNINITIALISED_FILL,
    aggregate,
    census_lines,
    classify_wild,
    fill_origin,
    summarise,
    trace,
)
from elenchus.emulation.compare import (
    SEED_A,
    SEED_B,
    VARIANT_A,
    VARIANT_B,
    Verdict,
)
from elenchus.emulation.harness import (
    ARENA_BASE,
    GAMMA,
    GOLDEN,
    INPUT_REGION_BASE,
    PAGE,
    Outcome,
    Status,
    _fill,
)
from elenchus.emulation.inputs import BUFFER_STRIDE

# A page well inside the input region, and one in the arena.
INPUT_PAGE = INPUT_REGION_BASE + 37 * PAGE
ARENA_PAGE = ARENA_BASE + 5 * PAGE


def word_at(page_base, within, constant):
    """The 8-byte word the harness's own fill puts at page_base + within."""
    page = _fill(page_base, constant)
    return int.from_bytes(page[within:within + 8], "little")


# ------------------------------------------------------- the arithmetic


def test_the_inverse_is_an_inverse():
    assert (GOLDEN * GOLDEN_INVERSE) & MASK64 == 1


def test_a_word_of_the_input_fill_is_recognised_with_its_offset():
    """Built by _fill itself, so this is the harness's fill, not a copy of
    the formula that could drift from it."""
    value = word_at(INPUT_PAGE, 0x238, VARIANT_B)
    assert fill_origin(value, (VARIANT_B,)) == (
        VARIANT_B, INPUT_PAGE % GAMMA + 0x238, 0)


def test_a_field_access_past_the_pointer_is_recognised_both_ways():
    """p->next lands at the pointer plus a constant; the census must see
    through that, out to the edge of its reach and no further."""
    value = word_at(INPUT_PAGE, 0x40, VARIANT_A)
    for displacement in (8, 0x1F8, REACH, -REACH):
        origin = fill_origin(value + displacement, (VARIANT_A,))
        assert origin is not None
        assert origin[2] == displacement


def test_beyond_the_reach_nothing_is_claimed():
    value = word_at(INPUT_PAGE, 0x40, VARIANT_A)
    assert fill_origin(value + REACH + 1, (VARIANT_A,)) is None


def test_the_fill_constant_says_which_memory_it_came_from():
    """The arena and wild pages are filled from the run's seed, the input
    buffers from the variant. A word from one is not mistaken for the other."""
    value = word_at(ARENA_PAGE, 0x10, SEED_A)
    assert fill_origin(value, (VARIANT_A,)) is None
    origin = fill_origin(value, (VARIANT_A, SEED_A))
    assert origin is not None
    assert origin[0] == SEED_A


# ------------------------------------------------------- what it refuses


def test_random_addresses_are_not_mistaken_for_the_fill():
    """The count must be a lower bound. If this failed, fault anatomy would
    credit our fill with faults it did not cause."""
    rng = random.Random(20261010)
    constants = (VARIANT_A, VARIANT_B, SEED_A, SEED_B)
    for _ in range(64):
        assert fill_origin(rng.getrandbits(64), constants) is None


def test_an_unaligned_read_is_not_recognised():
    """A pointer read across two words is not one word of the fill. It is
    left as "other" - an undercount, never an overcount."""
    page = _fill(INPUT_PAGE, VARIANT_A)
    straddling = int.from_bytes(page[0x44:0x4C], "little")
    assert fill_origin(straddling, (VARIANT_A,)) is None


# ------------------------------------------------------- classification


def test_each_origin_is_told_apart():
    data = word_at(INPUT_PAGE, 0x100, VARIANT_B)
    garbage = word_at(ARENA_PAGE, 0x100, SEED_B)
    assert classify_wild("null", 0x18, SEED_B, VARIANT_B) == NULL
    assert classify_wild("read", 0x18, SEED_B, VARIANT_B) == NULL
    assert classify_wild("read", data + 8, SEED_B, VARIANT_B) == INPUT_FILL
    assert classify_wild("write", garbage, SEED_B, VARIANT_B) \
        == UNINITIALISED_FILL
    assert classify_wild("fetch", data, SEED_B, VARIANT_B) == CODE_FROM_FILL
    assert classify_wild("read", 0x7123_4567_89AB_CDEF, SEED_B, VARIANT_B) \
        == OTHER


def test_only_the_runs_own_constants_are_tried():
    """A run under SEED_A cannot have read a word SEED_B made."""
    garbage = word_at(ARENA_PAGE, 0x100, SEED_B)
    assert classify_wild("read", garbage, SEED_A, VARIANT_A) == OTHER




# ------------------------------------------------------- what a run shows


def run_trace(status=Status.COMPLETED, calls=(), seed=SEED_A,
              variant=VARIANT_A, read_invented=False, first_wild=None,
              allocations=(), writes=(), ret=1, overflow=False):
    out = Outcome(status, ret_int=ret, write_ranges=tuple(writes),
                  calls=tuple(calls), calls_overflow=overflow,
                  first_wild=first_wild, allocations=tuple(allocations),
                  read_invented=read_invented)
    return trace(out, seed, variant)


def under(variant, q, k, q_seed_b=None, k_seed_b=None):
    """One input fill's four runs, Q's and K's seeds agreeing unless told."""
    return (run_trace(seed=SEED_A, variant=variant, **q),
            run_trace(seed=SEED_B, variant=variant, **(q_seed_b or q)),
            run_trace(seed=SEED_A, variant=variant, **k),
            run_trace(seed=SEED_B, variant=variant, **(k_seed_b or k)))


def both_fills(q, k):
    return (under(VARIANT_A, q, k), under(VARIANT_B, q, k))


def census_of(*inputs, ret_kind="int"):
    return summarise("inconclusive", False, [], list(inputs), ret_kind)


def test_a_trace_reads_which_buffers_and_how_much_arena():
    t = run_trace(writes=[(INPUT_REGION_BASE + 16, 4),
                          (INPUT_REGION_BASE + BUFFER_STRIDE + 8, 8),
                          (ARENA_BASE, 24)])
    assert t.buffers == {0, 1}
    assert t.arena_bytes == 24


def test_a_run_that_stopped_shows_no_return():
    assert run_trace(ret=0).null_return is True
    assert run_trace(status=Status.FAULT, ret=0).null_return is None


# ------------------------------------------------------- the gates


def test_a_clean_difference_is_counted():
    found = census_of(both_fills({"calls": ("memcpy",)},
                                 {"calls": ("strlen",)}))
    assert found.comparable["calls"] and found.differs["calls"]
    assert found.call_names == ("memcpy", "strlen")


def test_a_difference_under_one_fill_only_is_ours():
    """F24 applied to a new observation: it came from the bytes we invented."""
    one = (under(VARIANT_A, {"calls": ("memcpy",)}, {"calls": ("strlen",)}),
           under(VARIANT_B, {"calls": ("memcpy",)}, {"calls": ("memcpy",)}))
    found = census_of(one)
    assert found.comparable["calls"]
    assert not found.differs["calls"]


def test_an_observation_the_seeds_disagree_on_is_not_compared():
    """F22: what changes with the seed came from memory nobody wrote."""
    unstable = (under(VARIANT_A, {"calls": ("memcpy",)},
                      {"calls": ("strlen",)}, q_seed_b={"calls": ()}),
                under(VARIANT_B, {"calls": ("memcpy",)},
                      {"calls": ("strlen",)}))
    found = census_of(unstable)
    assert not found.comparable["calls"]
    assert not found.differs["calls"]


def test_a_run_that_read_invented_memory_is_not_compared():
    """F32: what follows from our fill is ours."""
    found = census_of(both_fills(
        {"calls": ("memcpy",), "read_invented": True},
        {"calls": ("strlen",)}))
    assert not found.comparable["calls"]


def test_a_run_that_stopped_compares_nothing():
    found = census_of(both_fills({"calls": ("memcpy",),
                                  "status": Status.FAULT},
                                 {"calls": ("strlen",)}))
    assert not found.comparable["calls"]


def test_an_overflowing_call_list_is_not_compared():
    """A list cut at 256 would read as complete; it is left out instead."""
    found = census_of(both_fills({"calls": ("memcpy",), "overflow": True},
                                 {"calls": ("strlen",)}))
    assert not found.comparable["calls"]


# ------------------------------------------------------- each observation


def test_the_allow_list_projection_ignores_what_a_compiler_can_add():
    """memcpy against memset differs in the calls, not on the allow-list;
    a stream write on one side only differs on both."""
    lists = census_of(both_fills(
        {"calls": ("memcpy", "EnterCriticalSection")},
        {"calls": ("memset", "EnterCriticalSection")}))
    assert lists.differs["calls"]
    assert not lists.differs["calls on the allow-list"]
    stream = census_of(both_fills({"calls": ("fputs",)}, {"calls": ()}))
    assert stream.differs["calls on the allow-list"]


def test_writing_different_buffers_is_seen():
    """The difference _judge cannot see: no byte written by both."""
    found = census_of(both_fills(
        {"writes": [(INPUT_REGION_BASE, 4)]},
        {"writes": [(INPUT_REGION_BASE + BUFFER_STRIDE, 4)]}))
    assert found.differs["buffers written"]


def test_a_null_return_is_observed_only_for_a_pointer():
    found = census_of(both_fills({"ret": 0}, {"ret": 7}),
                      ret_kind="pointer")
    assert found.differs["null return"]
    plain = census_of(both_fills({"ret": 0}, {"ret": 7}))
    assert "null return" not in plain.comparable


def test_ending_differently_is_a_status_split():
    found = census_of(both_fills({}, {"status": Status.FAULT}))
    assert found.differs["status split"]
    assert not found.comparable["calls"]


# ------------------------------------------------------- mass and anatomy


@dataclass
class Judged:
    verdict: Verdict
    compared_bytes: int
    saw_return: bool


def test_a_survivals_mass_is_its_strongest_input():
    inputs = [Judged(Verdict.SURVIVED, 3, False),
              Judged(Verdict.SURVIVED, 800, True),
              Judged(Verdict.INCONCLUSIVE, 0, False)]
    found = summarise("survived", False, inputs, [], "int")
    assert found.survival_mass == 800
    assert found.survival_saw_return


def test_a_forced_survival_has_no_mass():
    """Forced paths are another tier; their agreements are weighed apart."""
    inputs = [Judged(Verdict.SURVIVED, 800, True)]
    assert summarise("survived", True, inputs, [], "int").survival_mass \
        is None


def test_the_first_wild_touch_is_classified():
    pointer = word_at(INPUT_PAGE, 0x100, VARIANT_A)
    found = census_of(both_fills(
        {"first_wild": ("read", pointer + 8), "read_invented": True},
        {}))
    assert found.wild_origin == INPUT_FILL
    assert found.wild_status == "completed"


def test_every_first_tier_run_is_counted():
    found = census_of(both_fills({}, {}), both_fills({}, {}))
    assert len(found.instructions) == 2 * 2 * 4


# ------------------------------------------------------- adding up


def test_a_ceiling_counts_only_pairs_not_already_refuted():
    """A difference on a pair the verifier already refutes adds nothing."""
    calls = both_fills({"calls": ("a",)}, {"calls": ("b",)})
    open_pair = census_of(calls)
    refuted = summarise("refuted", False, [], [calls], "int")
    summary = aggregate([(None, open_pair), (None, refuted)], 5_000)
    row = summary["observations"]["calls"]
    assert row["differs"] == 2
    assert row["differs_unrefuted"] == 1
    assert any("calls" in line for line in census_lines(summary, "R0"))


# ------------------------------------------------------- what "other" was


def test_an_uninitialised_local_is_told_apart():
    """The stack's garbage does not invert, but it is deterministic and can
    be listed. A pointer read out of it is an uninitialised local."""
    from elenchus.emulation.harness import _stack_garbage
    block = _stack_garbage(SEED_A)
    local = int.from_bytes(block[0x1F00:0x1F08], "little")
    assert classify_wild("read", local + 16, SEED_A, VARIANT_A) \
        == STACK_GARBAGE
    assert classify_wild("fetch", local, SEED_A, VARIANT_A) == CODE_FROM_FILL
    assert classify_wild("read", local, SEED_B, VARIANT_A) == OTHER


def test_an_address_just_past_a_region_is_told_apart():
    """A real base plus a garbage length or index: the integer inputs, not
    the pointers, sent the function there."""
    from elenchus.emulation.harness import ARENA_END, INPUT_REGION_END
    assert classify_wild("read", INPUT_REGION_END + 0x1000,
                         SEED_A, VARIANT_A) == BEYOND
    assert classify_wild("write", INPUT_REGION_BASE - 4,
                         SEED_A, VARIANT_A) == BEYOND
    assert classify_wild("read", ARENA_END + 0xFFFF_FFFF * 8,
                         SEED_A, VARIANT_A) == BEYOND


def test_a_faulting_pair_is_read_by_the_run_that_faulted():
    """A completed run touched wild memory first; another run faulted. The
    fault is what the anatomy is about."""
    early = word_at(INPUT_PAGE, 0x100, VARIANT_A)
    found = census_of(
        both_fills({"first_wild": ("read", early), "read_invented": True}, {}),
        both_fills({"status": Status.FAULT, "first_wild": ("null", 0)}, {}))
    assert found.wild_origin == NULL
    assert found.wild_status == "fault"


def test_which_observations_differed_is_kept_for_the_union():
    """Ceilings overlap; the union needs to know which differed together."""
    found = census_of(both_fills({"calls": ("a",)}, {"calls": ("b",)}))
    assert found.differing == ("calls",)
    summary = aggregate([(None, found)], 5_000)
    assert summary["differing"] == [["inconclusive", ["calls"], ""]]
