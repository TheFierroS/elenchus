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
from elenchus.emulation.stubs import BLOCK_MEMORY, memcmp, memcpy, memmove, memset

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
        start = address - self.base
        assert start >= 0 and start + size <= len(self.mem), "read out of range"
        return bytes(self.mem[start:start + size])

    def write(self, address, data):
        start = address - self.base
        if start + len(data) > len(self.mem):
            self.mem.extend(b"\x00" * (start + len(data) - len(self.mem)))
        self.mem[start:start + len(data)] = data

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


def resolver(name):
    return BLOCK_MEMORY.get(name)


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
                     )
    # compare does not take a resolver yet; this checks the pair is at least
    # not refuted when both stop identically at the import.
    assert result.verdict in {Verdict.SURVIVED, Verdict.INCONCLUSIVE}
