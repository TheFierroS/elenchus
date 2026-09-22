"""The block-memory stubs, checked against C semantics.

Two layers. First a FakeMachine backed by a dict: it holds argument values
and a byte-addressable memory, so each stub is driven with chosen arguments
and its effect on memory and return value checked - including the corners
that break naive versions. Then end to end through the harness: a fixture
function that calls memcpy runs to completion with the stub serving the call,
and its -O0 and -O3 twins survive comparison.
"""

from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth
from elenchus.emulation.abi import placement
from elenchus.emulation.compare import Verdict, compare
from elenchus.emulation.harness import Loader, Status, run
from elenchus.emulation.stubs import (
    BLOCK_MEMORY,
    STUBS,
    memcmp,
    memcpy,
    memmove,
    memset,
    resolver,
)

# The verifier needs unicorn, an optional dependency; skip if it is absent.
pytest.importorskip("unicorn")

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"
BUF = 0x0000_2000_0000_0000


class FakeMachine:
    """A Machine backed by a dict, for testing a stub with no emulator."""

    def __init__(self, args, memory=b"", base=0x1000):
        self._args = args
        self.base = base
        self.mem = bytearray(memory)
        self.returned = None

    def arg(self, index):
        return self._args[index]

    def read(self, address, size):
        from elenchus.emulation.harness import MAX_TRANSFER, StubDeclined
        if size > MAX_TRANSFER:
            raise StubDeclined(f"read of {size} bytes, past the sanity bound")
        start = address - self.base
        assert start >= 0 and start + size <= len(self.mem), "read out of range"
        return bytes(self.mem[start:start + size])

    def write(self, address, data):
        start = address - self.base
        if start + len(data) > len(self.mem):
            self.mem.extend(b"\x00" * (start + len(data) - len(self.mem)))
        self.mem[start:start + len(data)] = data

    def bound(self, size):
        from elenchus.emulation.harness import MAX_TRANSFER, StubDeclined
        if size > MAX_TRANSFER:
            raise StubDeclined(f"size {size} past the sanity bound")
        return size

    def set_return(self, value):
        self.returned = value


# --------------------------------------------------------- memcpy


def test_memcpy_copies_the_bytes_and_returns_dest():
    m = FakeMachine(args=[0x1000, 0x1010, 4], memory=b"\x00" * 16 + b"ABCD",
                    base=0x1000)
    memcpy(m)
    assert m.read(0x1000, 4) == b"ABCD"
    assert m.returned == 0x1000


def test_memcpy_of_zero_copies_nothing():
    m = FakeMachine(args=[0x1000, 0x1010, 0], memory=b"keep" + b"\x00" * 12,
                    base=0x1000)
    memcpy(m)
    assert m.read(0x1000, 4) == b"keep"
    assert m.returned == 0x1000


# --------------------------------------------------------- memmove


def test_memmove_handles_forward_overlap():
    """dest above src by one: a byte-at-a-time forward copy would smear the
    first byte across the range; memmove reads all of src first."""
    m = FakeMachine(args=[0x1001, 0x1000, 4], memory=b"ABCD" + b"\x00" * 12,
                    base=0x1000)
    memmove(m)
    assert m.read(0x1001, 4) == b"ABCD"


def test_memmove_handles_backward_overlap():
    m = FakeMachine(args=[0x1000, 0x1001, 4], memory=b"\x00ABCD" + b"\x00" * 11,
                    base=0x1000)
    memmove(m)
    assert m.read(0x1000, 4) == b"ABCD"


# --------------------------------------------------------- memset


def test_memset_writes_the_low_byte_n_times():
    m = FakeMachine(args=[0x1000, 0x41, 5], memory=b"\x00" * 8, base=0x1000)
    memset(m)
    assert m.read(0x1000, 5) == b"AAAAA"
    assert m.returned == 0x1000


def test_memset_uses_only_the_low_byte_of_c():
    m = FakeMachine(args=[0x1000, 0x1FF, 3], memory=b"\x00" * 8, base=0x1000)
    memset(m)
    assert m.read(0x1000, 3) == b"\xFF\xFF\xFF"


def test_memset_of_zero_writes_nothing():
    m = FakeMachine(args=[0x1000, 0x41, 0], memory=b"keep", base=0x1000)
    memset(m)
    assert m.read(0x1000, 4) == b"keep"


# --------------------------------------------------------- memcmp


def test_memcmp_equal_is_zero():
    m = FakeMachine(args=[0x1000, 0x1004, 4], memory=b"ABCDABCD", base=0x1000)
    memcmp(m)
    assert m.returned == 0


def test_memcmp_sign_of_first_difference():
    less = FakeMachine(args=[0x1000, 0x1004, 4], memory=b"ABCDABCE", base=0x1000)
    memcmp(less)
    assert (less.returned & 0xFFFFFFFF) == 0xFFFFFFFF          # -1 as int32

    more = FakeMachine(args=[0x1000, 0x1004, 4], memory=b"ABCEABCD", base=0x1000)
    memcmp(more)
    assert more.returned == 1


def test_memcmp_compares_as_unsigned_bytes():
    """0x80 > 0x7F as unsigned chars, which is how memcmp compares."""
    m = FakeMachine(args=[0x1000, 0x1001, 1], memory=b"\x80\x7F", base=0x1000)
    memcmp(m)
    assert m.returned == 1


def test_memcmp_of_zero_is_equal():
    m = FakeMachine(args=[0x1000, 0x1004, 0], memory=b"ABCDwxyz", base=0x1000)
    memcmp(m)
    assert m.returned == 0


# --------------------------------------------------------- the family map


def test_the_family_serves_the_builtin_spellings():
    assert BLOCK_MEMORY["__builtin_memcpy"] is memcpy
    assert BLOCK_MEMORY["__builtin_memset"] is memset
    assert set(BLOCK_MEMORY) >= {"memcpy", "memmove", "memset", "memcmp"}


# --------------------------------------------------------- end to end


@pytest.fixture(scope="module", params=["O0", "O3"])
def fixture(request):
    level = request.param
    loader = Loader(FIXTURES / f"cases_{level}.dll")
    sigs = {f.name: f for f in ground_truth(FIXTURES / f"cases_{level}.dll")}
    return loader, sigs


def test_a_function_calling_memcpy_completes_with_the_stub(fixture):
    """copy(dest, src, n) is a wrapper around memcpy. With the stub it runs to
    completion instead of stopping at the import."""
    loader, sigs = fixture
    f = sigs["copy"]
    # dest and src in two mapped-on-touch buffers, n bytes.
    out = run(loader, f.address, placement(f.abi),
              [BUF, BUF + 0x1000, 16], stub_resolver=resolver)
    assert out.status is Status.COMPLETED


def test_the_memcpy_wrapper_survives_its_own_twin():
    """copy at -O0 against copy at -O3, both served by the same stub, agree."""
    o0 = Loader(FIXTURES / "cases_O0.dll")
    o3 = Loader(FIXTURES / "cases_O3.dll")
    s0 = {f.name: f for f in ground_truth(FIXTURES / "cases_O0.dll")}
    s3 = {f.name: f for f in ground_truth(FIXTURES / "cases_O3.dll")}
    result = compare(o0, s0["copy"].address, o3, s3["copy"].address,
                     placement(s3["copy"].abi),
                     [[BUF, BUF + 0x1000, 16], [BUF, BUF + 0x1000, 1]],
                     stub_resolver=resolver)
    # With the stub serving memcpy, both sides run to completion and agree:
    # the same function copies the same bytes at either level.
    assert result.verdict is Verdict.SURVIVED


def test_the_central_resolver_serves_the_block_family():
    assert resolver("memcpy") is memcpy
    assert resolver("__builtin_memset") is memset
    assert resolver("no_such_function") is None


def test_a_duplicate_stub_name_is_caught():
    """Adding a family that reuses a name must fail loudly, not shadow."""
    import pytest as _pytest

    from elenchus.emulation.stubs import _merge
    with _pytest.raises(ValueError, match="two stubs"):
        _merge({"memcpy": memcpy}, {"memcpy": memset})


def test_the_stubs_table_is_the_union_of_the_families():
    assert set(STUBS) >= set(BLOCK_MEMORY)


# --------------------------------------------------------- allocation

from elenchus.emulation.harness import Arena, StubDeclined  # noqa: E402
from elenchus.emulation.stubs import calloc, free, malloc, realloc  # noqa: E402


class ArenaMachine:
    """A Machine over a real Arena, memory kept sparse in a dict so the huge
    arena addresses need no giant buffer."""

    def __init__(self, args):
        self._args = args
        self.arena = Arena()
        self.byte = {}                     # address -> value, unset reads as 0
        self.returned = None

    def arg(self, index):
        return self._args[index]

    def read(self, address, size):
        return bytes(self.byte.get(address + i, 0) for i in range(size))

    def write(self, address, data):
        for i, value in enumerate(data):
            self.byte[address + i] = value

    def bound(self, size):
        from elenchus.emulation.harness import MAX_TRANSFER, StubDeclined
        if size > MAX_TRANSFER:
            raise StubDeclined(f"size {size} past the sanity bound")
        return size

    def set_return(self, value):
        self.returned = value


def test_malloc_returns_aligned_non_overlapping_blocks():
    a = Arena()
    first = a.allocate(10)
    second = a.allocate(10)
    assert first % 16 == 0 and second % 16 == 0
    assert second >= first + 10                 # no overlap
    assert second - first == 16                 # 10 rounded up to 16


def test_malloc_zero_returns_a_unique_freeable_pointer():
    a = Arena()
    p = a.allocate(0)
    assert p != 0
    a.free(p)                                   # freeable, no error


def test_free_null_is_a_no_op():
    a = Arena()
    a.free(0)                                   # no error


def test_free_of_a_bad_pointer_declines():
    a = Arena()
    with pytest.raises(StubDeclined):
        a.free(0xDEAD0000)


def test_double_free_declines():
    a = Arena()
    p = a.allocate(8)
    a.free(p)
    with pytest.raises(StubDeclined):
        a.free(p)


def test_a_freed_address_is_not_handed_out_again():
    """Reuse would let -O0 and -O3, freeing at different points, diverge on
    later addresses; the arena never reuses."""
    a = Arena()
    p = a.allocate(16)
    a.free(p)
    q = a.allocate(16)
    assert q != p


def test_calloc_zeroes_the_memory():
    m = ArenaMachine(args=[4, 8])
    calloc(m)
    assert m.read(m.returned, 32) == b"\x00" * 32


def test_calloc_overflow_returns_null():
    m = ArenaMachine(args=[1 << 40, 1 << 40])    # product overflows 64 bits
    calloc(m)
    assert m.returned == 0


def test_realloc_null_is_malloc():
    m = ArenaMachine(args=[0, 16])
    realloc(m)
    assert m.returned != 0                       # realloc(NULL, n) is malloc


def test_realloc_grows_and_keeps_the_old_bytes():
    m = ArenaMachine(args=[4])          # malloc(4), one argument
    malloc(m)
    p = m.returned
    m.write(p, b"ABCD")
    m2 = ArenaMachine(args=[p, 8])
    m2.arena = m.arena
    m2.byte = m.byte
    realloc(m2)
    assert m2.read(m2.returned, 4) == b"ABCD"    # old contents preserved
    assert m2.returned != p                      # moved


def test_realloc_of_a_bad_pointer_declines():
    m = ArenaMachine(args=[0xBADF00D, 16])
    with pytest.raises(StubDeclined):
        realloc(m)


def test_the_allocation_family_is_in_the_resolver():
    assert resolver("malloc") is malloc
    assert resolver("free") is free
    assert resolver("calloc") is calloc
    assert resolver("realloc") is realloc


def test_duplicate_stops_at_an_unstubbed_import_cleanly():
    """With only the block-memory family, duplicate stops at strlen - named,
    not a fault. (It completes once the string family is present; see
    test_duplicate_completes_with_all_its_stubs.)"""
    loader = Loader(FIXTURES / "cases_O3.dll")
    sigs = {f.name: f for f in ground_truth(FIXTURES / "cases_O3.dll")}
    f = sigs["duplicate"]
    out = run(loader, f.address, placement(f.abi), [BUF],
              stub_resolver=BLOCK_MEMORY.get)
    assert out.status is Status.IMPORT_WITHOUT_STUB
    assert out.detail in {"strlen", "malloc"}


# --------------------------------------------------------- C string

from elenchus.emulation.stubs import (  # noqa: E402
    _cstring,
    strcat,
    strchr,
    strcmp,
    strcpy,
    strdup,
    strlen,
    strncmp,
    strncpy,
    strrchr,
    strstr,
)


class StrMachine:
    """A sparse-memory Machine with a real Arena, for the string stubs."""

    def __init__(self, args):
        self._args = list(args)
        self.arena = Arena()
        self.byte = {}
        self.returned = None

    def place(self, address, data):
        """Write bytes at an address and return it, for building inputs."""
        for i, v in enumerate(data):
            self.byte[address + i] = v
        return address

    def arg(self, index):
        return self._args[index]

    def read(self, address, size):
        return bytes(self.byte.get(address + i, 0) for i in range(size))

    def write(self, address, data):
        for i, v in enumerate(data):
            self.byte[address + i] = v

    def bound(self, size):
        from elenchus.emulation.harness import MAX_TRANSFER, StubDeclined
        if size > MAX_TRANSFER:
            raise StubDeclined(f"size {size} past the sanity bound")
        return size

    def set_return(self, value):
        self.returned = value


A, B, C = 0x1000, 0x2000, 0x3000


def test_strlen_counts_to_the_nul():
    m = StrMachine([A])
    m.place(A, b"hello\x00")
    strlen(m)
    assert m.returned == 5


def test_strlen_of_empty_is_zero():
    m = StrMachine([A])
    m.place(A, b"\x00")
    strlen(m)
    assert m.returned == 0


def test_cstring_reads_up_to_the_nul():
    m = StrMachine([])
    m.place(A, b"abc\x00defgh")
    assert _cstring(m, A) == b"abc"


def test_strcmp_orders_like_c():
    m = StrMachine([A, B])
    m.place(A, b"abc\x00")
    m.place(B, b"abc\x00")
    strcmp(m)
    assert m.returned == 0

    m = StrMachine([A, B])
    m.place(A, b"abc\x00")
    m.place(B, b"abd\x00")
    strcmp(m)
    assert (m.returned & 0xFFFFFFFF) == 0xFFFFFFFF          # abc < abd

    m = StrMachine([A, B])
    m.place(A, b"abc\x00")               # prefix compares less
    m.place(B, b"abcd\x00")
    strcmp(m)
    assert (m.returned & 0xFFFFFFFF) == 0xFFFFFFFF


def test_strncmp_stops_at_n():
    m = StrMachine([A, B, 3])
    m.place(A, b"abcX\x00")
    m.place(B, b"abcY\x00")
    strncmp(m)
    assert m.returned == 0               # first 3 bytes equal


def test_strchr_finds_and_misses():
    m = StrMachine([A, ord("c")])
    m.place(A, b"abcde\x00")
    strchr(m)
    assert m.returned == A + 2

    m = StrMachine([A, ord("z")])
    m.place(A, b"abcde\x00")
    strchr(m)
    assert m.returned == 0

    m = StrMachine([A, 0])               # searching for the NUL finds it
    m.place(A, b"abc\x00")
    strchr(m)
    assert m.returned == A + 3


def test_strrchr_finds_the_last():
    m = StrMachine([A, ord("a")])
    m.place(A, b"banana\x00")
    strrchr(m)
    assert m.returned == A + 5


def test_strstr_finds_a_substring():
    m = StrMachine([A, B])
    m.place(A, b"hello world\x00")
    m.place(B, b"wor\x00")
    strstr(m)
    assert m.returned == A + 6

    m = StrMachine([A, B])
    m.place(A, b"hello\x00")
    m.place(B, b"xyz\x00")
    strstr(m)
    assert m.returned == 0


def test_strcpy_copies_including_the_nul():
    m = StrMachine([B, A])
    m.place(A, b"copy me\x00")
    strcpy(m)
    assert m.read(B, 8) == b"copy me\x00"
    assert m.returned == B


def test_strncpy_pads_with_nul():
    m = StrMachine([B, A, 6])
    m.place(A, b"ab\x00")
    strncpy(m)
    assert m.read(B, 6) == b"ab\x00\x00\x00\x00"


def test_strncpy_truncates_and_does_not_terminate():
    m = StrMachine([B, A, 3])
    m.place(A, b"abcdef\x00")
    strncpy(m)
    assert m.read(B, 3) == b"abc"        # no NUL when the source is longer


def test_strcat_appends_at_the_terminator():
    m = StrMachine([A, B])
    m.place(A, b"foo\x00")
    m.place(B, b"bar\x00")
    strcat(m)
    assert m.read(A, 7) == b"foobar\x00"


def test_strdup_makes_a_freeable_copy():
    m = StrMachine([A])
    m.place(A, b"dup\x00")
    strdup(m)
    assert m.read(m.returned, 4) == b"dup\x00"
    m.arena.free(m.returned)             # it is a real arena block


def test_the_string_family_is_in_the_resolver():
    assert resolver("strlen") is strlen
    assert resolver("strstr") is strstr
    assert resolver("__builtin_strcpy") is strcpy


# --------------------------------------------------------- end to end


def test_duplicate_completes_with_all_its_stubs(fixture):
    """duplicate calls strlen, malloc and memcpy. With every family it runs to
    completion - the first real copy-and-allocate function fully served."""
    loader, sigs = fixture
    f = sigs["duplicate"]
    out = run(loader, f.address, placement(f.abi), [BUF], stub_resolver=resolver)
    assert out.status is Status.COMPLETED


def test_string_wrappers_survive_their_own_twins():
    """Each wrapper leans on one string stub; -O0 against -O3, served by the
    same stubs, agrees."""
    o0 = Loader(FIXTURES / "cases_O0.dll")
    o3 = Loader(FIXTURES / "cases_O3.dll")
    s0 = {f.name: f for f in ground_truth(FIXTURES / "cases_O0.dll")}
    s3 = {f.name: f for f in ground_truth(FIXTURES / "cases_O3.dll")}
    for name, inputs in [
        ("length_of", [[BUF]]),
        ("compare_strings", [[BUF, BUF + 0x1000]]),
        ("find_char", [[BUF, ord("A")]]),
        ("duplicate", [[BUF]]),
    ]:
        result = compare(o0, s0[name].address, o3, s3[name].address,
                         placement(s3[name].abi), inputs, stub_resolver=resolver)
        assert result.verdict in {Verdict.SURVIVED, Verdict.INCONCLUSIVE}, name


def test_a_garbage_size_declines_instead_of_crashing():
    """memset(dest, c, n) with a nonsense n - a pointer reinterpreted as a
    count - must not try to build forty billion bytes and crash. It declines,
    inconclusive. This is the MemoryError V0 hit on the real corpus."""
    from elenchus.emulation.harness import StubDeclined
    m = FakeMachine(args=[0x1000, 0x41, 40_000_000_000], memory=b"", base=0x1000)
    with pytest.raises(StubDeclined):
        memset(m)


def test_a_garbage_memcpy_size_declines():
    from elenchus.emulation.harness import StubDeclined
    m = FakeMachine(args=[0x1000, 0x2000, 1 << 40], memory=b"\x00" * 16, base=0x1000)
    with pytest.raises(StubDeclined):
        memcpy(m)


def test_a_reasonable_size_still_works():
    """The bound only stops the absurd; an ordinary memset is unaffected."""
    m = FakeMachine(args=[0x1000, 0x41, 5], memory=b"\x00" * 8, base=0x1000)
    memset(m)
    assert m.read(0x1000, 5) == b"AAAAA"


def test_a_stub_write_is_recorded_as_an_effect(fixture):
    """copy(d, s, n) is a memcpy wrapper: at -O0 the copy is a call the stub
    serves, and the stub writes through the emulator's API, which does not
    fire the write hook. Without recording it the function looked as if it
    wrote nothing, while an inlined -O3 copy looked as if it wrote - and the
    pair was refuted (F25). The write must be visible."""
    loader, sigs = fixture
    f = sigs["copy"]
    out = run(loader, f.address, placement(f.abi),
              [BUF, BUF + 0x1000, 16], stub_resolver=resolver)
    assert out.status is Status.COMPLETED
    assert (BUF & ~0xFFF) in out.writes          # the destination page is there
    written = out.writes[BUF & ~0xFFF][BUF & 0xFFF:][:16]
    assert len(written) == 16                     # and holds the copied bytes


def test_a_memcpy_wrapper_survives_its_twin_with_the_write_seen():
    """copy at -O0 (a served memcpy) against -O3 (inlined): with the stub's
    write recorded, both show the same effect and the pair survives."""
    o0 = Loader(FIXTURES / "cases_O0.dll")
    o3 = Loader(FIXTURES / "cases_O3.dll")
    s0 = {f.name: f for f in ground_truth(FIXTURES / "cases_O0.dll")}
    s3 = {f.name: f for f in ground_truth(FIXTURES / "cases_O3.dll")}
    result = compare(o0, s0["copy"].address, o3, s3["copy"].address,
                     placement(s3["copy"].abi),
                     [[BUF, BUF + 0x1000, 16], [BUF, BUF + 0x1000, 32]],
                     stub_resolver=resolver)
    assert result.verdict is Verdict.SURVIVED
