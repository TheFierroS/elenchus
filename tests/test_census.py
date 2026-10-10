"""The census reads where a wild address came from - exactly, or not at all.

Fault anatomy (docs/verifier.md, the census, observation 8) decides whether
repairing inputs is worth building, so its count must never be inflated. Every
test here either shows a pointer read out of our own fill being recognised, or
shows something that is not one being refused.
"""

import random

from elenchus.emulation.census import (
    CODE_FROM_FILL,
    GOLDEN_INVERSE,
    INPUT_FILL,
    MASK64,
    NULL,
    OTHER,
    REACH,
    UNINITIALISED_FILL,
    classify_wild,
    fill_origin,
)
from elenchus.emulation.compare import SEED_A, SEED_B, VARIANT_A, VARIANT_B
from elenchus.emulation.harness import (
    ARENA_BASE,
    GAMMA,
    GOLDEN,
    INPUT_REGION_BASE,
    PAGE,
    _fill,
)

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
