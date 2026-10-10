"""The observation census: what else the verifier could compare, measured first.

docs/verifier.md, "What else could be observed: the census". The verifier
compares two things - the masked return and the bytes both versions wrote -
and this module measures, without changing any verdict, how much a third would
be worth and what it would cost. Nothing here decides anything. It holds no
emulator state and imports no emulator, so it is tested on its own, like abi,
inputs, branches, forced and chain.

This first part reads one thing exactly: where a wild address came from.

A function run on invented data often dereferences a pointer it read out of
that data, and lands somewhere the harness never handed out. The fill that put
those bytes there is not random. A word of it is

    offset * GOLDEN + c    (mod 2**64)

where `offset` is the word's byte position in the fill block and `c` is the
input variant inside the argument buffers, the run's seed everywhere else
(harness._fill_block, harness._fill_seed_for). GOLDEN is odd, so it has an
inverse modulo 2**64, and a value can be tested exactly for having been one of
those words: multiply back, and see whether a word offset inside the block
comes out.

That is what Tinbergen gets from a backward taint analysis - which input bytes
were a pointer (docs/related-work.md, 5.1) - read here from the arithmetic of
our own fill instead. The census's eighth observation, fault anatomy, is built
on it, and it decides whether repairing inputs is worth building.
"""

from __future__ import annotations

from elenchus.emulation.harness import GAMMA, GOLDEN, NULL_GUARD

MASK64 = (1 << 64) - 1
GOLDEN_INVERSE = pow(GOLDEN, -1, 1 << 64)
WORD = 8

# How far past the pointer it read a function may land. A field access adds a
# small constant before dereferencing - p->next at +8, a member a few hundred
# bytes into a context - so the address that faults is the pointer plus that.
# An index scaled by garbage lands much further and is left as "other", which
# makes every count built on this a lower bound, never an overcount.
REACH = 1024

# Where a run's first wild access came from.
NULL = "null"
INPUT_FILL = "data pointer from the input fill"
UNINITIALISED_FILL = "data pointer from uninitialised fill"
CODE_FROM_FILL = "code pointer from fill"
OTHER = "other"
ORIGINS = (NULL, INPUT_FILL, UNINITIALISED_FILL, CODE_FROM_FILL, OTHER)


def fill_origin(address: int, constants, reach: int = REACH):
    """How `address` was reached from a word of our fill, or None.

    Returns (constant, offset, displacement): the fill constant that made the
    word, the word's byte offset in the fill block, and how far past the word's
    value the address lies. Nearer displacements are tried first.

    A random 64-bit value passes the test by accident with probability about
    (GAMMA / 8) / 2**64 for each displacement and constant tried - under one in
    a billion over the whole reach - so a match is not a coincidence.

    A pointer read from an unaligned position spans two words and is not of
    this form, so it is not recognised; nor is one read from the stack, whose
    garbage comes from random.Random rather than the arithmetic fill. Both
    leave the count a lower bound.
    """
    address &= MASK64
    for distance in range(reach + 1):
        for displacement in (distance, -distance) if distance else (0,):
            pointer = (address - displacement) & MASK64
            for constant in constants:
                offset = ((pointer - constant) * GOLDEN_INVERSE) & MASK64
                if offset < GAMMA and offset % WORD == 0:
                    return constant, offset, displacement
    return None


def classify_wild(kind: str, address: int, seed: int, variant: int) -> str:
    """Where a run's first wild access came from, as one of ORIGINS.

    `kind` is how the harness touched it: "read", "write", "fetch", or "null"
    for a dereference of the guard page. `seed` and `variant` are the run's
    own, the only two constants its fill could have used.

    A fetch is code: the function called or jumped through a pointer it read
    out of our fill - a callback in a context, an operations table. Handing it
    a buffer cannot repair that, so it is kept apart from data pointers, which
    a buffer can.
    """
    if kind == "null" or address < NULL_GUARD:
        return NULL
    origin = fill_origin(address, (variant, seed))
    if origin is None:
        return OTHER
    if kind == "fetch":
        return CODE_FROM_FILL
    return INPUT_FILL if origin[0] == variant else UNINITIALISED_FILL
