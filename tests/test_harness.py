"""The harness, run against the fixture DLLs built from cases.c.

These call real functions in real MinGW-built binaries at -O0 and -O3 and
check what the harness observed: the masked return, memory written through a
pointer, and the buckets a function that cannot be run falls into. Argument
buffers are mapped before the call, so a function reading them sees input,
not touched-garbage.
"""

import struct
from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth
from elenchus.emulation.abi import placement
from elenchus.emulation.harness import PAGE, Loader, Status, _fill, run

# The verifier needs unicorn, an optional dependency; skip if it is absent.
pytest.importorskip("unicorn")

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"
# A region for argument buffers, far from the image base and the stack.
BUF = 0x0000_2000_0000_0000


@pytest.fixture(scope="module", params=["O0", "O3"])
def level(request):
    return request.param


@pytest.fixture(scope="module")
def fixture(level):
    loader = Loader(FIXTURES / f"cases_{level}.dll")
    sigs = {f.name: f for f in ground_truth(FIXTURES / f"cases_{level}.dll")}
    return loader, sigs


def call(fixture, name, args, seed=1, budget=5_000_000):
    loader, sigs = fixture
    f = sigs[name]
    return run(loader, f.address, placement(f.abi), args,
               seed=seed, budget=budget)


def masked(outcome, fixture, name):
    _, sigs = fixture
    return outcome.ret_int & placement(sigs[name].abi).ret.mask


# ------------------------------------------------------- plain returns


def test_a_plain_integer_function_returns_the_right_value(fixture):
    out = call(fixture, "add3", [7, 5, 2])
    assert out.status is Status.COMPLETED
    assert masked(out, fixture, "add3") == 7 * 3 + 5 - 2


def test_a_wide_integer_uses_the_full_register(fixture):
    a, b = 0xDEADBEEF12345678, 0xABCD
    out = call(fixture, "mix64", [a, b])
    assert out.status is Status.COMPLETED
    expected = (a ^ (b << 17)) * 0x9E3779B97F4A7C15 & ((1 << 64) - 1)
    assert masked(out, fixture, "mix64") == expected


def test_a_narrow_return_is_masked_to_its_width(fixture):
    """low_byte returns signed char: only the low 8 bits of RAX are defined,
    and the harness must not compare the rest (docs/verifier.md, risk 1)."""
    out = call(fixture, "low_byte", [10])
    assert out.status is Status.COMPLETED
    assert masked(out, fixture, "low_byte") == ((10 * 7 + 3) & 0xFF)


# ------------------------------------------------------- pointer input




def test_a_function_fills_a_buffer_through_a_pointer(fixture):
    """fill(p, 20, 'A') writes 20 ascending bytes; both levels write the
    same, and it is a void function so nothing is returned."""
    out = call(fixture, "fill", [BUF, 20, 0x41])
    assert out.status is Status.COMPLETED
    page = out.writes[BUF & ~(PAGE - 1)]
    start = BUF & (PAGE - 1)
    assert page[start:start + 20] == bytes((0x41 + i) & 0xFF for i in range(20))


def test_two_levels_fill_a_buffer_identically(fixture):
    out = call(fixture, "fill", [BUF, 32, 0x10])
    written = out.writes[BUF & ~(PAGE - 1)][BUF & (PAGE - 1):][:32]
    assert written == bytes((0x10 + i) & 0xFF for i in range(32))


def test_a_function_reads_its_pointer_argument(fixture):
    """count_nonzero reads the buffer at BUF. The harness maps it on first
    touch with the address-only pattern, so both runs see the same bytes;
    the count is whatever that pattern holds, but it is the same each time
    and the same at both levels."""
    a = call(fixture, "count_nonzero", [BUF, 64], seed=1)
    b = call(fixture, "count_nonzero", [BUF, 64], seed=2)
    assert a.status is b.status is Status.COMPLETED
    # Argument memory does not take the seed, so the two agree.
    assert a.ret_int == b.ret_int


# ------------------------------------------------------- float returns


def test_a_double_return_comes_back_through_xmm0(fixture):
    out = call(fixture, "scale", [_d(3.0), 4])
    assert out.status is Status.COMPLETED
    assert _from_double_bits(out.ret_float_bits) == pytest.approx(3.0 * 4 + 0.5)


def test_a_float_return_is_single_precision(fixture):
    out = call(fixture, "fma3", [_f(2.0), _f(3.0), _f(1.5)])
    assert out.status is Status.COMPLETED
    assert _from_float_bits(out.ret_float_bits & 0xFFFFFFFF) == pytest.approx(7.5)


# ------------------------------------------------------- following a call


def test_a_call_within_the_binary_is_followed(fixture):
    """uses_helper calls a static helper twice; at -O0 those are real calls,
    not inlined, and the harness follows them into the binary's own code
    (tier 1). helper(x) = x*x + 1, so uses_helper(a,b) = helper(a)-helper(b)."""
    out = call(fixture, "uses_helper", [5, 3])
    assert out.status is Status.COMPLETED
    assert masked(out, fixture, "uses_helper") == (5 * 5 + 1) - (3 * 3 + 1)


# ------------------------------------------------------- the buckets


def test_an_import_without_a_stub_is_reported(fixture):
    """duplicate calls strlen and malloc, both imports; with no stub resolver
    the run stops at the first and names it, not a fault."""
    out = call(fixture, "duplicate", [BUF])
    assert out.status is Status.IMPORT_WITHOUT_STUB
    assert out.detail in {"strlen", "malloc", "memcpy"}


def test_a_nonterminating_function_exhausts_the_budget(fixture):
    """spin(5) never returns; a small budget makes it budget-exhausted, not
    a hang."""
    out = call(fixture, "spin", [5], budget=100_000)
    assert out.status is Status.BUDGET_EXHAUSTED


def test_a_null_dereference_is_a_fault_not_a_crash(fixture):
    """null_read reads through address 0; the harness reports a fault and
    keeps going rather than letting the emulator error escape."""
    out = call(fixture, "null_read", [0])
    assert out.status is Status.FAULT


# ------------------------------------------------------- determinism


def test_the_same_call_gives_the_same_outcome(fixture):
    a = call(fixture, "add3", [7, 5, 2])
    b = call(fixture, "add3", [7, 5, 2])
    assert a.ret_int == b.ret_int and a.status == b.status


def test_the_fill_pattern_is_the_address_at_seed_zero():
    """Input memory (seed 0) depends only on the address, so both runs and
    both levels see the same bytes."""
    assert _fill(0x1000, 0) == _fill(0x1000, 0)
    assert _fill(0x1000, 0) != _fill(0x2000, 0)
    assert _fill(0x1000, 1) != _fill(0x1000, 2)


# ------------------------------------------------------- float helpers


def _d(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _f(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _from_double_bits(bits):
    return struct.unpack("<d", struct.pack("<Q", bits & ((1 << 64) - 1)))[0]


def _from_float_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def test_the_page_limit_is_small_enough_to_catch_a_lost_walk():
    """A function chasing a garbage pointer maps a fresh page per step; the
    limit caps that at 2 MiB, well under what a real function touches, so the
    lost walk stops quickly as inconclusive instead of mapping 16 MiB slowly
    (F16, the cause of the first V0 pass's time)."""
    from elenchus.emulation.harness import UNMAPPED_FILL_LIMIT
    assert UNMAPPED_FILL_LIMIT <= 512


def test_a_slow_run_times_out_rather_than_hanging(fixture):
    """spin(n) never returns. With a tiny wall-clock timeout it stops as
    TIMED_OUT well before the instruction budget, so no function can hang the
    verifier no matter how it loops."""
    loader, sigs = fixture
    f = sigs["spin"]
    out = run(loader, f.address, placement(f.abi), [5],
              budget=10_000_000, timeout_us=100_000)   # 0.1s, huge budget
    assert out.status is Status.TIMED_OUT
    assert out.instructions < 10_000_000               # stopped on time, not count


def test_budget_and_timeout_are_told_apart(fixture):
    """spin with a tiny budget and a generous timeout is BUDGET_EXHAUSTED, not
    TIMED_OUT: the two are distinguished by whether the count reached the
    budget."""
    loader, sigs = fixture
    f = sigs["spin"]
    out = run(loader, f.address, placement(f.abi), [5],
              budget=20_000, timeout_us=5_000_000)      # small budget, 5s
    assert out.status is Status.BUDGET_EXHAUSTED

def test_run_releases_its_emulator(fixture):
    """Every run releases its Unicorn handle, or a long run leaks the C-side
    memory of every mapped page and is killed (F19). A fresh spy per call
    avoids any cross-test total; the leak itself is measured out of band."""
    import unicorn
    loader, sigs = fixture
    f = sigs["add3"]

    seen = []
    real = unicorn.Uc.release_handle
    try:
        unicorn.Uc.release_handle = lambda self: seen.append(1) or real(self)
        run(loader, f.address, placement(f.abi), [7, 5, 2], budget=50_000)
    finally:
        unicorn.Uc.release_handle = real
    assert seen == [1]                     # released exactly once


def test_a_faulting_run_still_releases(fixture):
    """The release is in a finally, so a faulting run frees its emulator too."""
    import unicorn
    loader, sigs = fixture

    seen = []
    real = unicorn.Uc.release_handle
    try:
        unicorn.Uc.release_handle = lambda self: seen.append(1) or real(self)
        out = run(loader, sigs["null_read"].address,
                  placement(sigs["null_read"].abi), [0], budget=50_000)
    finally:
        unicorn.Uc.release_handle = real
    assert out.status is Status.FAULT
    assert seen == [1]


def test_the_loader_does_not_retain_the_pefile_object():
    """A Loader keeps only the raw bytes it needs to map, not the pefile
    object, which for a large binary is 100+ MB; over V0's ~100 distinct
    binaries retaining every one leaked gigabytes (F20)."""
    loader = Loader(FIXTURES / "cases_O3.dll")
    assert not hasattr(loader, "pe")           # the heavy object is gone
    assert isinstance(loader._header, bytes)
    assert all(isinstance(d, bytes) for _, d in loader._sections)


def test_a_loader_still_maps_correctly_from_raw_bytes(fixture):
    """Dropping the pefile object must not change what gets mapped: add3 still
    runs and returns the right value from the saved section bytes."""
    loader, sigs = fixture
    f = sigs["add3"]
    out = run(loader, f.address, placement(f.abi), [7, 5, 2], budget=50_000)
    assert out.status is Status.COMPLETED
    assert (out.ret_int & placement(f.abi).ret.mask) == 24
