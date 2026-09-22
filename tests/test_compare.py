"""Turning two runs into a verdict, and never refuting a true claim.

Two layers of test. The first builds Outcomes by hand and feeds them to the
judging logic, so every branch - masked return, garbage exclusion, memory
difference, a version that did not complete - is checked without an emulator.
The second runs real fixture functions end to end: the same function at -O0
and -O3 must survive (it is the same function), and two genuinely different
functions must be refuted.
"""

import struct
from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth
from elenchus.emulation.abi import placement
from elenchus.emulation.compare import (
    Verdict,
    _judge,
    _stable_return,
    _stable_writes,
    compare,
)
from elenchus.emulation.harness import Loader, Outcome, Status

# The verifier needs unicorn, an optional dependency; skip if it is absent.
pytest.importorskip("unicorn")

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"
BUF = 0x0000_2000_0000_0000


# ------------------------------------------------------- hand-built outcomes


def done(ret_int=0, ret_float_bits=0, writes=None):
    return Outcome(Status.COMPLETED, ret_int=ret_int,
                   ret_float_bits=ret_float_bits, writes=writes or {})


def place(kind="int", size=4):
    return placement({"return": {"kind": kind, "size": size, "signed": True},
                      "params": [], "variadic": False})


def test_a_stable_masked_return_is_read():
    """The upper bits of RAX differ but the low 4 are the same: the value is
    stable, because only the masked width is read."""
    a = done(ret_int=0xAAAA_AAAA_0000_0042)
    b = done(ret_int=0xBBBB_BBBB_0000_0042)
    assert _stable_return(a, b, place("int", 4)) == ("int", 0x42)


def test_a_return_that_changes_with_the_seed_is_garbage():
    """Different low bytes between the two seeds: the return depended on
    uninitialised memory, so it cannot be compared."""
    a = done(ret_int=0x0000_0001)
    b = done(ret_int=0x0000_0002)
    assert _stable_return(a, b, place("int", 4)) is None


def test_a_void_return_is_nothing_to_compare():
    assert _stable_return(done(), done(), place("void", 0)) == "void"


def test_stable_writes_keep_only_pages_both_seeds_agree_on():
    a = done(writes={0x1000: b"same", 0x2000: b"garbageA"})
    b = done(writes={0x1000: b"same", 0x2000: b"garbageB"})
    assert _stable_writes(a, b) == {0x1000: b"same"}


# ------------------------------------------------------- judging one input


def test_two_runs_that_agree_survive():
    q = (done(ret_int=7), done(ret_int=7))
    k = (done(ret_int=7), done(ret_int=7))
    assert _judge(*q, *k, place()).verdict is Verdict.SURVIVED


def test_a_stable_difference_refutes():
    q = (done(ret_int=7), done(ret_int=7))
    k = (done(ret_int=8), done(ret_int=8))
    result = _judge(*q, *k, place())
    assert result.verdict is Verdict.REFUTED
    assert "return" in result.reason


def test_a_difference_only_one_seed_shows_does_not_refute():
    """Q returns 7 then 8 - its own return is garbage-dependent - and K is a
    steady 7. The difference is not real, so this input cannot refute; with
    nothing else to compare it is inconclusive, not a refutation."""
    q = (done(ret_int=7), done(ret_int=8))
    k = (done(ret_int=7), done(ret_int=7))
    result = _judge(*q, *k, place())
    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_version_that_did_not_complete_is_inconclusive():
    q = (Outcome(Status.BUDGET_EXHAUSTED), Outcome(Status.BUDGET_EXHAUSTED))
    k = (done(ret_int=7), done(ret_int=7))
    result = _judge(*q, *k, place())
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "budget" in result.reason


def test_a_stable_memory_difference_refutes():
    q = (done(writes={0x4000: b"AAAA"}), done(writes={0x4000: b"AAAA"}))
    k = (done(writes={0x4000: b"BBBB"}), done(writes={0x4000: b"BBBB"}))
    result = _judge(*q, *k, place("void", 0))
    assert result.verdict is Verdict.REFUTED
    assert "memory" in result.reason


def test_garbage_memory_does_not_refute():
    """Each version writes a page that differs between its own seeds: garbage,
    dropped from both, so the two versions are not compared on it."""
    q = (done(ret_int=1, writes={0x4000: b"qA"}), done(ret_int=1, writes={0x4000: b"qB"}))
    k = (done(ret_int=1, writes={0x4000: b"kA"}), done(ret_int=1, writes={0x4000: b"kB"}))
    assert _judge(*q, *k, place()).verdict is Verdict.SURVIVED


def test_a_float_return_compares_by_bits():
    bits = struct.unpack("<Q", struct.pack("<d", 3.5))[0]
    q = (done(ret_float_bits=bits), done(ret_float_bits=bits))
    k = (done(ret_float_bits=bits), done(ret_float_bits=bits))
    fplace = placement({"return": {"kind": "float", "size": 8}, "params": [],
                        "variadic": False})
    assert _judge(*q, *k, fplace).verdict is Verdict.SURVIVED


# ------------------------------------------------------- end to end


def loaders_and_sigs():
    o0 = Loader(FIXTURES / "cases_O0.dll")
    o3 = Loader(FIXTURES / "cases_O3.dll")
    s0 = {f.name: f for f in ground_truth(FIXTURES / "cases_O0.dll")}
    s3 = {f.name: f for f in ground_truth(FIXTURES / "cases_O3.dll")}
    return o0, o3, s0, s3


@pytest.fixture(scope="module")
def env():
    return loaders_and_sigs()


def compare_named(env, q_name, k_name, inputs):
    o0, o3, s0, s3 = env
    return compare(o0, s0[q_name].address, o3, s3[k_name].address,
                   placement(s3[k_name].abi), inputs)


def test_the_same_function_at_two_levels_survives(env):
    """add3 at -O0 against add3 at -O3: the same function, so it must never
    be refuted - the requirement the whole design rests on."""
    result = compare_named(env, "add3", "add3", [[1, 2, 3], [7, 5, 2], [-1, 0, 100]])
    assert result.verdict is Verdict.SURVIVED


def test_a_wide_function_survives_itself(env):
    result = compare_named(env, "mix64", "mix64",
                           [[1, 2], [0xDEADBEEF, 0xABCD], [0, 0]])
    assert result.verdict is Verdict.SURVIVED


def test_a_void_writer_survives_itself(env):
    result = compare_named(env, "fill", "fill", [[BUF, 16, 0x41], [BUF, 8, 0]])
    assert result.verdict is Verdict.SURVIVED


def test_a_double_function_survives_itself(env):
    d3 = struct.unpack("<Q", struct.pack("<d", 3.0))[0]
    d2 = struct.unpack("<Q", struct.pack("<d", 2.5))[0]
    result = compare_named(env, "scale", "scale", [[d3, 4], [d2, 7]])
    assert result.verdict is Verdict.SURVIVED


def test_two_different_functions_are_refuted(env):
    """add3 against mix64 under add3's signature: different functions, so
    some input must make them disagree."""
    o0, o3, s0, s3 = env
    result = compare(o0, s0["add3"].address, o3, s3["mix64"].address,
                     placement(s3["add3"].abi),
                     [[1, 2, 3], [7, 5, 2], [100, 200, 300], [0, 0, 0]])
    assert result.verdict is Verdict.REFUTED
    assert result.refuting_input is not None


def test_a_caller_survives_itself(env):
    """uses_helper calls into the binary; following those calls, -O0 and -O3
    still agree."""
    result = compare_named(env, "uses_helper", "uses_helper",
                           [[5, 3], [10, 10], [-2, 7]])
    assert result.verdict is Verdict.SURVIVED


def test_an_unstubbed_import_is_inconclusive(env):
    """duplicate calls malloc; with no stubs both sides stop the same way, so
    the claim cannot be judged - inconclusive, not refuted."""
    result = compare_named(env, "duplicate", "duplicate", [[BUF]])
    assert result.verdict is Verdict.INCONCLUSIVE


def test_the_first_refutation_stops_the_comparison(env):
    o0, o3, s0, s3 = env
    result = compare(o0, s0["add3"].address, o3, s3["mix64"].address,
                     placement(s3["add3"].abi),
                     [[5, 5, 5], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert result.verdict is Verdict.REFUTED
    # It stopped at the first refuting input, not run all four.
    assert len(result.inputs) == result.refuting_input + 1


def test_a_write_to_a_global_does_not_refute(env):
    """record(i, v) writes into a global array, which lives at a different
    address in the -O0 and -O3 binaries. That write is the function's own
    global state, not a comparable effect (docs/verifier.md, risk 9): the
    harness excludes writes into the image, so the pair survives. This is the
    false refutation V0 found on lua_upvaluejoin, in miniature."""
    result = compare_named(env, "record", "record", [[0, 42], [3, 7], [9, 100]])
    assert result.verdict is Verdict.SURVIVED


def test_a_pointer_return_does_not_refute(env):
    """banner() returns a pointer into .rdata, at a different address in the
    -O0 and -O3 binaries. The returned pointer is not compared by value (F18),
    so the pair survives - the return sibling of the global-write fix (F15).
    This is the class of false refutation V0 found (utf8proc_version,
    opus_strerror, and six more, all pointer returns)."""
    result = compare_named(env, "banner", "banner", [[], [], []])
    assert result.verdict is Verdict.SURVIVED


def test_a_float_return_is_masked_to_its_width():
    """XMM0 is 128 bits; a 4-byte float defines only the low 32. Two runs
    whose low 32 bits agree but whose upper bits differ must not refute - the
    float analogue of masking a narrow integer return (F21). Built by hand so
    the upper bits differ exactly."""
    from elenchus.emulation.abi import placement
    from elenchus.emulation.compare import _stable_return
    from elenchus.emulation.harness import Outcome, Status

    fplace = placement({"return": {"kind": "float", "size": 4}, "params": [],
                        "variadic": False})
    low = 0x4F82A000
    a = Outcome(Status.COMPLETED, ret_float_bits=0x00000000_00000000 | low)
    b = Outcome(Status.COMPLETED, ret_float_bits=0x4F82A000_00000000 | low)
    # low 32 identical, upper differ: masked to 4 bytes they agree
    assert _stable_return(a, b, fplace) == ("float", low)


def test_a_double_return_uses_all_64_bits():
    """A double defines the low 64; masking must not throw those away."""
    from elenchus.emulation.abi import placement
    from elenchus.emulation.compare import _stable_return
    from elenchus.emulation.harness import Outcome, Status

    dplace = placement({"return": {"kind": "float", "size": 8}, "params": [],
                        "variadic": False})
    bits = 0x400921FB54442D18            # pi as a double
    a = Outcome(Status.COMPLETED, ret_float_bits=bits)
    b = Outcome(Status.COMPLETED, ret_float_bits=bits)
    assert _stable_return(a, b, dplace) == ("float", bits)

    c = Outcome(Status.COMPLETED, ret_float_bits=bits ^ 1)   # differ in a low bit
    assert _stable_return(a, c, dplace) is None              # a real difference


def test_a_written_pointer_into_the_image_does_not_refute(env):
    """publish(out, count) writes the address of a global string into the
    caller's buffer, and that address differs between the -O0 and -O3
    binaries. The written pointer value is an address, not data, so it is
    masked before comparison (F23) - the third sibling of F15 (global writes)
    and F18 (returned pointers). The count written beside it still compares.
    This is the class V0 found: rhash_ripemd160_final, FLAC cuesheet,
    lexbor style_mutation_init."""
    result = compare_named(env, "publish", "publish", [[BUF, BUF + 0x1000]])
    assert result.verdict is Verdict.SURVIVED


def test_masking_only_blanks_image_addresses():
    """The mask must not blank ordinary data that happens to be large, nor
    pointers into the input buffers, which are at the same address in both
    versions."""
    from elenchus.emulation.compare import _mask_image_pointers
    image = [(0x140000000, 0x140100000)]
    inside = (0x140005000).to_bytes(8, "little")
    outside = (0x2000000000000).to_bytes(8, "little")      # a buffer address
    data = (12345).to_bytes(8, "little")
    content = inside + outside + data
    masked = _mask_image_pointers(content, 0, image)
    assert masked[0:8] == b"\x00" * 8                      # the image pointer
    assert masked[8:16] == outside                         # buffer pointer kept
    assert masked[16:24] == data                           # plain data kept


# ------------------------------------------- the input-variant rule (F24)


def _input_result(verdict, reason="", detail=None):
    from elenchus.emulation.compare import InputResult
    return InputResult(verdict, reason, detail or {})


def test_a_difference_under_both_fills_refutes():
    """The difference is there whatever we put in the buffers, so it is the
    functions' own."""
    from elenchus.emulation.compare import _combine_variants
    both = [_input_result(Verdict.REFUTED, "return value differs"),
            _input_result(Verdict.REFUTED, "return value differs")]
    assert _combine_variants(both).verdict is Verdict.REFUTED


def test_a_difference_under_one_fill_only_is_inconclusive():
    """A function that branches on the invented buffer bytes - a hash
    finaliser given a garbage context - differs under one fill and not the
    other. That difference is ours, not the function's, so it cannot refute."""
    from elenchus.emulation.compare import _combine_variants
    mixed = [_input_result(Verdict.REFUTED, "written memory differs"),
             _input_result(Verdict.SURVIVED, "agreed on this input")]
    result = _combine_variants(mixed)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "one input fill" in result.reason


def test_agreement_under_either_fill_survives():
    from elenchus.emulation.compare import _combine_variants
    mixed = [_input_result(Verdict.INCONCLUSIVE, "Q budget exhausted"),
             _input_result(Verdict.SURVIVED, "agreed on this input")]
    assert _combine_variants(mixed).verdict is Verdict.SURVIVED


def test_two_unjudgeable_fills_stay_inconclusive():
    from elenchus.emulation.compare import _combine_variants
    neither = [_input_result(Verdict.INCONCLUSIVE, "Q import without a stub"),
               _input_result(Verdict.INCONCLUSIVE, "Q import without a stub")]
    assert _combine_variants(neither).verdict is Verdict.INCONCLUSIVE


def test_a_real_difference_still_refutes_through_both_fills(env):
    """The variant rule must not blunt the verifier: add3 against mix64 differ
    whatever fills the buffers, and are still refuted."""
    o0, o3, s0, s3 = env
    result = compare(o0, s0["add3"].address, o3, s3["mix64"].address,
                     placement(s3["add3"].abi),
                     [[1, 2, 3], [7, 5, 2], [100, 200, 300]])
    assert result.verdict is Verdict.REFUTED


def test_a_true_pair_still_survives_through_both_fills(env):
    result = compare_named(env, "sum", "sum", [[BUF, 9], [BUF, 3]])
    assert result.verdict is Verdict.SURVIVED
