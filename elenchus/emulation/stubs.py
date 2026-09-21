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

from elenchus.emulation.harness import StubDeclined


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


# --- allocation: malloc, calloc, realloc, free, over a deterministic arena --

# A block's header, kept in a Python dict on the Machine rather than in
# emulated memory, so a program cannot corrupt the allocator's own bookkeeping
# and a freed block can be recognised. Addresses come from one region, far
# from the image, the stack and the traps.
def malloc(m) -> None:
    """void *malloc(size_t size). Returns a fresh block, or the arena declines."""
    size = m.arg(0)
    m.set_return(m.arena.allocate(size))


def calloc(m) -> None:
    """void *calloc(size_t n, size_t size). Zeroed; n*size overflow returns 0."""
    n, size = m.arg(0), m.arg(1)
    total = n * size
    if total >> 64:                          # would overflow size_t
        m.set_return(0)
        return
    address = m.arena.allocate(total)
    if total:
        m.write(address, b"\x00" * total)
    m.set_return(address)


def realloc(m) -> None:
    """void *realloc(void *p, size_t size).

    realloc(NULL, size) is malloc; realloc(p, 0) frees p and returns a fresh
    minimal block. Otherwise a new block is allocated, the smaller of the two
    sizes is copied, and the old block freed - moving every time, which is
    always allowed and keeps the arena a simple bump allocator.
    """
    old, size = m.arg(0), m.arg(1)
    if old == 0:
        m.set_return(m.arena.allocate(size))
        return
    old_size = m.arena.size_of(old)          # declines a bad pointer
    new = m.arena.allocate(size)
    keep = min(old_size, size)
    if keep:
        m.write(new, m.read(old, keep))
    m.arena.free(old)
    m.set_return(new)


def free(m) -> None:
    """void free(void *p). NULL is a no-op; a bad pointer declines."""
    m.arena.free(m.arg(0))
    m.set_return(0)


ALLOCATION = {
    "malloc": malloc,
    "calloc": calloc,
    "realloc": realloc,
    "free": free,
}


# --- C string: strlen, strcmp, strchr, ... over a NUL-terminated read -------

# The one operation the family shares: read a NUL-terminated byte string from
# emulated memory, without knowing its length in advance. A cap stops a
# missing terminator - a caller that passed a non-string, or memory the fill
# pattern never zeroed - from reading forever; over the cap the input is
# declined rather than guessed at.
CSTRING_CAP = 1 << 20            # 1 MiB: longer than any real C string here


def _cstring(m, address):
    """The bytes before the first NUL at `address` (the NUL not included)."""
    out = bytearray()
    while len(out) < CSTRING_CAP:
        byte = m.read(address + len(out), 1)[0]
        if byte == 0:
            return bytes(out)
        out.append(byte)
    raise StubDeclined("string longer than the cap, or unterminated")


def strlen(m) -> None:
    """size_t strlen(const char *s). Bytes before the NUL."""
    m.set_return(len(_cstring(m, m.arg(0))))


def strcmp(m) -> None:
    """int strcmp(const char *a, const char *b). Sign of the first difference,
    as unsigned chars; the NUL terminators take part, so a prefix is less."""
    a, b = _cstring(m, m.arg(0)), _cstring(m, m.arg(1))
    m.set_return(_sign(a, b))


def strncmp(m) -> None:
    """int strncmp(const char *a, const char *b, size_t n). strcmp over at
    most n bytes; comparison also stops at a NUL in either string."""
    n = m.arg(2)
    a = _cstring_capped(m, m.arg(0), n)
    b = _cstring_capped(m, m.arg(1), n)
    m.set_return(_sign(a, b))


def strchr(m) -> None:
    """char *strchr(const char *s, int c). Address of the first c, or NULL;
    c == 0 finds the terminator, which strchr includes."""
    base = m.arg(0)
    target = m.arg(1) & 0xFF
    s = _cstring(m, base)
    if target == 0:
        m.set_return(base + len(s))              # the NUL itself
        return
    index = s.find(target)
    m.set_return(base + index if index >= 0 else 0)


def strrchr(m) -> None:
    """char *strrchr(const char *s, int c). The last c, or NULL."""
    base = m.arg(0)
    target = m.arg(1) & 0xFF
    s = _cstring(m, base)
    if target == 0:
        m.set_return(base + len(s))
        return
    index = s.rfind(target)
    m.set_return(base + index if index >= 0 else 0)


def strstr(m) -> None:
    """char *strstr(const char *hay, const char *needle). First occurrence of
    needle, or NULL; an empty needle matches at the start."""
    base = m.arg(0)
    hay = _cstring(m, base)
    needle = _cstring(m, m.arg(1))
    index = hay.find(needle)
    m.set_return(base + index if index >= 0 else 0)


def strcpy(m) -> None:
    """char *strcpy(char *d, const char *s). Copy s and its NUL to d; return d."""
    dest, src = m.arg(0), m.arg(1)
    s = _cstring(m, src)
    m.write(dest, s + b"\x00")
    m.set_return(dest)


def strncpy(m) -> None:
    """char *strncpy(char *d, const char *s, size_t n). Copy up to n bytes of
    s; if s is shorter, pad the rest with NUL to exactly n bytes. Return d."""
    dest, src, n = m.arg(0), m.arg(1), m.arg(2)
    s = _cstring_capped(m, src, n)
    padded = (s + b"\x00" * n)[:n]
    if n:
        m.write(dest, padded)
    m.set_return(dest)


def strcat(m) -> None:
    """char *strcat(char *d, const char *s). Append s and a NUL at d's
    terminator; return d."""
    dest, src = m.arg(0), m.arg(1)
    end = dest + len(_cstring(m, dest))
    s = _cstring(m, src)
    m.write(end, s + b"\x00")
    m.set_return(dest)


def strdup(m) -> None:
    """char *strdup(const char *s). A malloc'd copy of s, NUL included, or
    NULL if the arena declines."""
    s = _cstring(m, m.arg(0))
    address = m.arena.allocate(len(s) + 1)
    m.write(address, s + b"\x00")
    m.set_return(address)


def _cstring_capped(m, address, limit):
    """The bytes before the first NUL at `address`, but at most `limit`."""
    out = bytearray()
    while len(out) < limit:
        byte = m.read(address + len(out), 1)[0]
        if byte == 0:
            break
        out.append(byte)
    return bytes(out)


def _sign(a: bytes, b: bytes) -> int:
    """The strcmp sign of two byte strings, as -1, 0 or 1. The comparison is
    over the strings with their terminators, so a prefix compares less."""
    a_term, b_term = a + b"\x00", b + b"\x00"
    for x, y in zip(a_term, b_term):
        if x != y:
            return 1 if x > y else (1 << 64) - 1
    return 0


CSTRING = {
    "strlen": strlen,
    "strcmp": strcmp,
    "strncmp": strncmp,
    "strchr": strchr,
    "strrchr": strrchr,
    "strstr": strstr,
    "strcpy": strcpy,
    "strncpy": strncpy,
    "strcat": strcat,
    "strdup": strdup,
    "__builtin_strlen": strlen,
    "__builtin_strcpy": strcpy,
}


# Every stub family, merged into one table. A new family is added here and
# nowhere else: the resolver, the harness and compare all read this, so
# coverage grows in one place. A later family must not silently reuse a name
# an earlier one claimed, so the merge checks for a collision.
def _merge(*families):
    merged = {}
    for family in families:
        for name, stub in family.items():
            if name in merged and merged[name] is not stub:
                raise ValueError(f"two stubs claim the name {name!r}")
            merged[name] = stub
    return merged


STUBS = _merge(BLOCK_MEMORY, ALLOCATION, CSTRING)


def resolver(name):
    """Return the stub for an import name, or None if no family serves it.

    This is the resolver the harness and compare use: a name it does not know
    makes the call inconclusive, never a guess. The whole verifier shares one
    resolver so a claim is judged against the same stubs however it is run.
    """
    return STUBS.get(name)
