"""Build input vectors for a function from its resolved signature.

The verifier runs a function on chosen inputs; those inputs have to make
sense for the function's parameters, or it faults before it computes
anything. A pointer parameter given the value 1 dereferences address 1 and
crashes - which is most of why a first V0 pass, with fixed integer inputs,
left so many pairs faulting rather than judged.

This reads the resolved signature (dwarf.py's abi: each parameter's kind,
width and sign) and builds vectors that fit it:

- an integer parameter gets boundary values (0, 1, -1, its min and max) and
  a few seeded random values, so overflow-sensitive code is exercised at its
  edges without every input triggering undefined behaviour;
- a pointer parameter gets the address of a real buffer, far apart from the
  others so two pointers never alias by accident, and the harness maps and
  fills that buffer on first touch (input memory, not garbage);
- a float or double gets ordinary values and exact small integers.

The vectors are positional, matching the placement the harness loads them
by. The generator is deterministic given a seed, which is recorded with a
verdict so it can be replayed, and it runs no code, so it is tested on its
own.
"""

from __future__ import annotations

import random

from elenchus.emulation.abi import aggregate_in_register, returns_through_hidden_pointer

# Buffers for pointer arguments live here, far from the image, the stack, the
# arena and the traps, each argument's buffer its own well-separated slot so
# two pointers passed together never overlap.
BUFFER_BASE = 0x0000_2000_0000_0000
BUFFER_STRIDE = 0x0000_0000_0010_0000     # 1 MiB apart


def _int_values(size, signed, rng, extra):
    """Boundary and random values for an integer of `size` bytes.

    The boundaries are where optimised code most often differs from
    unoptimised: zero, one, minus one, and the type's limits. `extra` seeded
    random values fill in the middle. Everything is masked to the width, so a
    value is a valid bit pattern for the register the harness will load.
    """
    mask = (1 << (size * 8)) - 1
    values = [0, 1, mask]                     # 0, 1, and all-ones (-1 if signed)
    if signed:
        high = 1 << (size * 8 - 1)
        values += [high, high - 1]            # INT_MIN, INT_MAX
    else:
        values += [mask >> 1]                 # near the top of an unsigned range
    for _ in range(extra):
        values.append(rng.getrandbits(size * 8))
    # de-duplicate, keep order, mask to width
    seen = []
    for v in values:
        v &= mask
        if v not in seen:
            seen.append(v)
    return seen


def _float_bits(size, rng, extra):
    """Bit patterns for a float (4) or double (8): ordinary values and exact
    small integers, packed to the width the harness places."""
    import struct
    samples = [0.0, 1.0, -1.0, 0.5, 2.0, 3.5]
    for _ in range(extra):
        samples.append(rng.uniform(-1e6, 1e6))
    out = []
    for x in samples:
        if size == 4:
            out.append(struct.unpack("<I", struct.pack("<f", x))[0])
        else:
            out.append(struct.unpack("<Q", struct.pack("<d", x))[0])
    return out


def input_vectors(abi, count=6, seed=0):
    """Return up to `count` input vectors for a resolved signature.

    Each vector is a list of integers, positional, one per parameter, ready
    for the harness: an integer or a float's packed bits, or a pointer's
    buffer address. A pointer always gets the same buffer address across
    vectors - the harness fills it on first touch, so its contents are the
    address-derived input pattern, the same for both versions of a claim.

    Integer parameters vary across the vectors (each takes the next of its
    boundary/random values, cycling); pointer and the buffer layout are
    fixed, so the vectors differ where it can refute and stay stable where a
    difference would only be garbage.
    """
    if abi is None:
        return []
    params = abi.get("params", [])
    rng = random.Random(seed)

    # Per-parameter candidate values, and a fixed buffer address for pointers.
    per_param = []
    buffer_index = 0

    if returns_through_hidden_pointer(abi.get("return")):
        # An aggregate too large for a register comes back written into space
        # the caller provides, whose address leads the arguments (abi.py).
        # It gets a buffer of its own, before the declared parameters.
        per_param.append(("fixed", [BUFFER_BASE]))
        buffer_index = 1

    for p in params:
        kind, size = p["kind"], p["size"]
        if kind in ("struct", "union") and not aggregate_in_register(size):
            # Passed by reference: the argument is the address of a copy, and
            # the buffer's fill pattern is that copy's contents.
            address = BUFFER_BASE + buffer_index * BUFFER_STRIDE
            buffer_index += 1
            per_param.append(("fixed", [address]))
        elif kind in ("struct", "union"):
            # Small enough to ride in a register: its bytes are the value, so
            # integer patterns of that width do as well as anything.
            per_param.append(("cycle", _int_values(size or 8, False, rng, extra=3)))
        elif kind == "pointer":
            address = BUFFER_BASE + buffer_index * BUFFER_STRIDE
            buffer_index += 1
            per_param.append(("fixed", [address]))
        elif kind == "float":
            per_param.append(("cycle", _float_bits(size, rng, extra=3)))
        elif kind == "int":
            signed = bool(p.get("signed"))
            per_param.append(("cycle", _int_values(size or 8, signed, rng, extra=3)))
        else:
            # A kind the placement would have declined; the caller does not
            # reach here for those, but stay safe with a single zero.
            per_param.append(("fixed", [0]))

    if not params:
        return [[]]                            # a no-argument function: one call

    vectors = []
    for i in range(count):
        vector = []
        for mode, values in per_param:
            if mode == "fixed":
                vector.append(values[0])
            else:
                vector.append(values[i % len(values)])
        if vector not in vectors:
            vectors.append(vector)
    return vectors
