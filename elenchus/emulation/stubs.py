"""Stubs for the block-memory family: memcpy, memmove, memset, memcmp.

A stub stands in for an imported function the emulator has no code for
(docs/verifier.md). This family is the first, because a call to memcpy or
memset is one of the commonest reasons a function would otherwise be
inconclusive, and because the four share one small set of memory operations
that is worth writing and testing once.

Each stub is held to the three rules of the design:

1. It implements the C standard behaviour of the function, nothing more. It
   moves or compares bytes; it never decides the result looks wrong.
2. It is tested on its own against that behaviour, corners included, before
   it is used.
3. It is deterministic: the same arguments and memory give the same effect.

A stub takes a Machine (harness.Machine, or anything with the same read,
write, arg and set_return): it reads its arguments from the argument
registers, works on emulated memory, and sets its return value. That keeps a
stub testable on a plain in-memory Machine with no emulator, which is how the
tests here drive them.

Win64 return conventions this family uses:
- memcpy, memmove, memset return the destination pointer (their first
  argument) in RAX.
- memcmp returns an int: negative, zero or positive. C only fixes the sign,
  so this returns -1, 0 or 1, and the sign is what a caller may branch on.
"""

from __future__ import annotations


def memcpy(m) -> None:
    """void *memcpy(void *dest, const void *src, size_t n).

    Copies n bytes; the regions must not overlap (that is memmove's job).
    n == 0 copies nothing. Returns dest.
    """
    dest, src, n = m.arg(0), m.arg(1), m.arg(2)
    if n:
        m.write(dest, m.read(src, n))
    m.set_return(dest)


def memmove(m) -> None:
    """void *memmove(void *dest, const void *src, size_t n).

    Like memcpy but safe when the regions overlap: it reads the whole source
    before writing, so a forward-overlapping copy does not clobber bytes it
    has not read yet. Reading all n bytes first gives that for free. Returns
    dest.
    """
    dest, src, n = m.arg(0), m.arg(1), m.arg(2)
    if n:
        m.write(dest, m.read(src, n))       # read fully, then write: overlap-safe
    m.set_return(dest)


def memset(m) -> None:
    """void *memset(void *dest, int c, size_t n).

    Writes the low byte of c, n times. c is an int argument but only its
    least significant byte is used, per the standard. Returns dest.
    """
    dest, c, n = m.arg(0), m.arg(1), m.arg(2)
    if n:
        m.write(dest, bytes([c & 0xFF]) * n)
    m.set_return(dest)


def memcmp(m) -> None:
    """int memcmp(const void *a, const void *b, size_t n).

    Compares n bytes and returns the sign of the difference of the first
    unequal pair, as unsigned chars; 0 if all n are equal. C fixes only the
    sign, so this returns -1, 0 or 1. n == 0 compares equal.
    """
    a, b, n = m.arg(0), m.arg(1), m.arg(2)
    if n == 0:
        m.set_return(0)
        return
    left = m.read(a, n)
    right = m.read(b, n)
    for x, y in zip(left, right):
        if x != y:
            m.set_return(1 if x > y else _neg1())
            return
    m.set_return(0)


def _neg1() -> int:
    """-1 as a 64-bit value, since a Machine stores an unsigned register.

    memcmp returns int, so the caller reads it masked to 32 bits; -1 as
    0xFFFFFFFFFFFFFFFF masks to 0xFFFFFFFF, which is -1 as a 32-bit int.
    """
    return (1 << 64) - 1


# The family, by the import names it serves. The resolver in the verifier
# will look a call's name up here; a name not present is not this family's.
BLOCK_MEMORY = {
    "memcpy": memcpy,
    "memmove": memmove,
    "memset": memset,
    "memcmp": memcmp,
    # GCC lowers some calls to __builtin_ or _-prefixed spellings; they are
    # the same functions.
    "__builtin_memcpy": memcpy,
    "__builtin_memmove": memmove,
    "__builtin_memset": memset,
    "__builtin_memcmp": memcmp,
}
