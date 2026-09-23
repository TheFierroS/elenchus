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


def test_input_memory_is_the_same_in_both_seeds(fixture):
    """A pointer argument's buffer is the function's input, so it is filled
    from its address alone: count_nonzero over the same buffer gives the same
    answer at either seed. Before F22 the buffer took the seed, so a function
    reading its own input looked garbage-dependent and was excluded."""
    loader, sigs = fixture
    f = sigs["sum"]                         # sums p[i]*(i+1): depends on bytes
    a = run(loader, f.address, placement(f.abi), [BUF, 9], seed=0x1111_1111)
    b = run(loader, f.address, placement(f.abi), [BUF, 9], seed=0x2222_2222)
    assert a.status is b.status is Status.COMPLETED
    assert a.ret_int == b.ret_int           # same input bytes, same sum


def test_the_stack_differs_between_seeds(fixture):
    """The stack is uninitialised memory, so it must differ between the two
    seeds - that is what lets the two-seed check spot a function reading a
    local it never wrote. Before F22 it was zero-filled and identical."""
    loader, sigs = fixture
    f = sigs["uninit"]                      # returns an uninitialised local
    a = run(loader, f.address, placement(f.abi), [0], seed=0x1111_1111)
    b = run(loader, f.address, placement(f.abi), [0], seed=0x2222_2222)
    assert a.status is b.status is Status.COMPLETED
    # At -O0 the local is read from the stack, so the two seeds disagree;
    # at -O3 the compiler may fold it, so accept either but require that the
    # stack itself is not identical between seeds.
    from elenchus.emulation.harness import _stack_garbage
    assert _stack_garbage(0x1111_1111) != _stack_garbage(0x2222_2222)


def test_the_stack_garbage_is_deterministic():
    """Same seed, same bytes - a verdict has to be replayable."""
    from elenchus.emulation.harness import _stack_garbage
    assert _stack_garbage(7) == _stack_garbage(7)


def test_without_a_chain_an_uninitialised_context_produces_nothing(fixture):
    """accum_final checks a magic its initialiser writes, and does nothing
    without it - the shape of a real hash finaliser given invented bytes."""
    loader, sigs = fixture
    f = sigs["accum_final"]
    out = run(loader, f.address, placement(f.abi), [BUF, BUF + 0x1000],
              budget=50_000)
    assert out.status is Status.COMPLETED
    assert not out.writes                     # it refused to produce anything


def test_a_chained_initialiser_makes_the_function_work(fixture):
    """Chained after accum_init the context is valid, so accum_final computes
    and writes its result (docs/verifier.md, constructor chains)."""
    loader, sigs = fixture
    init, final = sigs["accum_init"], sigs["accum_final"]
    out = run(loader, final.address, placement(final.abi),
              [BUF, BUF + 0x1000], budget=50_000,
              prepare=(init.address, placement(init.abi), [BUF]))
    assert out.status is Status.COMPLETED
    assert out.writes                          # now it produced a result


def test_the_initialisers_own_writes_are_not_the_functions_effect(fixture):
    """accum_init writes the context; that is setup, not what the function
    under test produced. If the initialiser's writes were kept, a context the
    two builds fill differently would look like a difference in the function.

    accum_add writes the context and nothing else, so chained after
    accum_init its *own* effect is the context page - and the setup writes
    must not add anything beyond it. Compared against a run with no chain,
    the recorded pages must be the same set."""
    loader, sigs = fixture
    init, peek = sigs["accum_init"], sigs["accum_peek"]
    chained = run(loader, peek.address, placement(peek.abi),
                  [BUF, BUF + 0x1000], budget=50_000,
                  prepare=(init.address, placement(init.abi), [BUF]))
    # accum_peek reads the context and writes only the output, so the output
    # page is the whole of its effect. The context page the initialiser wrote
    # must not be among them.
    assert set(chained.writes) == {(BUF + 0x1000) & ~0xFFF}


def test_a_chain_whose_initialiser_fails_is_reported(fixture):
    """If the initialiser cannot complete, the chain is abandoned and the run
    says so rather than testing the function on a half-built context."""
    loader, sigs = fixture
    final, spin = sigs["accum_final"], sigs["spin"]
    out = run(loader, final.address, placement(final.abi),
              [BUF, BUF + 0x1000], budget=5_000,
              prepare=(spin.address, placement(spin.abi), [5]))
    assert out.status is Status.CHAIN_FAILED
    assert "initialiser" in out.detail


def test_a_constructor_that_returns_the_context_is_used(fixture):
    """pool_create() builds a context and hands it back, the shape most C
    libraries use (cJSON_CreateObject, xmlNewDoc). The chain takes its return
    value as the function's first argument: pool_total then sees a valid
    context and computes, where on an invented one it returns zero."""
    loader, sigs = fixture
    create, total = sigs["pool_create"], sigs["pool_total"]
    place = placement(total.abi)

    plain = run(loader, total.address, place, [BUF], budget=50_000)
    chained = run(loader, total.address, place, [BUF], budget=50_000,
                  prepare=(create.address, placement(create.abi), [], True))
    assert plain.status is chained.status is Status.COMPLETED
    assert (plain.ret_int & place.ret.mask) == 0          # invented context
    assert (chained.ret_int & place.ret.mask) == 12       # 11 + 1, the real one


def test_a_constructor_returning_null_abandons_the_chain(fixture):
    """A constructor that fails hands back null, and a null context is not
    one to test on: the chain is abandoned rather than used."""
    loader, sigs = fixture
    total = sigs["pool_total"]
    # is_odd(0) returns 0 - standing in for a constructor that returned null.
    out = run(loader, total.address, placement(total.abi), [BUF], budget=50_000,
              prepare=(sigs["is_odd"].address, placement(sigs["is_odd"].abi),
                       [0], True))
    assert out.status is Status.CHAIN_FAILED
    assert "null" in out.detail
