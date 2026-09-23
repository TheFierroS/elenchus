"""The Win64 calling convention, checked against signatures from the fixture.

abi.placement turns a resolved signature into where each argument goes and
how the return is read. It runs no code, so these tests need no emulator:
they read the abi field dwarf.py resolved from tests/fixtures/emulation and
check the placement against what the Win64 convention says.
"""

from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth
from elenchus.emulation.abi import (
    FLOAT_REGISTERS,
    INT_REGISTERS,
    FloatArg,
    IntArg,
    Undecidable,
    placement,
)

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"


@pytest.fixture(scope="module")
def sigs():
    return {f.name: f.abi for f in ground_truth(FIXTURES / "cases_O3.dll")}


def test_the_first_four_integers_go_in_the_integer_registers(sigs):
    p = placement(sigs["six"])          # six long long arguments
    assert [a.register for a in p.args[:4]] == list(INT_REGISTERS)


def test_arguments_past_the_fourth_go_on_the_stack(sigs):
    p = placement(sigs["six"])
    fifth, sixth = p.args[4], p.args[5]
    assert fifth.register is None and fifth.stack_offset == 0
    assert sixth.register is None and sixth.stack_offset == 8
    assert p.stack_bytes == 16


def test_a_register_slot_is_used_by_position_not_by_kind(sigs):
    """mixed(int a, double b, int c, double d): the double in position two
    takes XMM1, not XMM0, because slot one was spent on the int in RCX."""
    p = placement(sigs["mixed"])
    a, b, c, d = p.args
    assert isinstance(a, IntArg) and a.register == "rcx"
    assert isinstance(b, FloatArg) and b.register == "xmm1"
    assert isinstance(c, IntArg) and c.register == "r8"
    assert isinstance(d, FloatArg) and d.register == "xmm3"


def test_floats_use_the_xmm_registers(sigs):
    p = placement(sigs["fma3"])         # three floats
    assert [a.register for a in p.args] == list(FLOAT_REGISTERS[:3])
    assert all(isinstance(a, FloatArg) for a in p.args)


def test_five_mixed_spills_the_fifth_to_the_stack(sigs):
    """five_mixed(double, int, double, int, double): four slots by position,
    then the fifth on the stack whatever its kind."""
    p = placement(sigs["five_mixed"])
    assert [type(a).__name__ for a in p.args] == [
        "FloatArg", "IntArg", "FloatArg", "IntArg", "FloatArg"]
    assert [a.register for a in p.args[:4]] == ["xmm0", "rdx", "xmm2", "r9"]
    assert p.args[4].register is None and p.args[4].stack_offset == 0


def test_a_pointer_is_an_eight_byte_integer_argument(sigs):
    p = placement(sigs["sum"])          # const int *, int
    ptr, n = p.args
    assert isinstance(ptr, IntArg) and ptr.register == "rcx" and ptr.size == 8
    assert isinstance(n, IntArg) and n.register == "rdx"


def test_the_return_mask_is_the_type_width(sigs):
    assert placement(sigs["add3"]).ret.mask == 0xFFFFFFFF        # int, 4 bytes
    assert placement(sigs["mix64"]).ret.mask == (1 << 64) - 1    # 8 bytes
    assert placement(sigs["low_byte"]).ret.mask == 0xFF          # signed char
    assert placement(sigs["low_word"]).ret.mask == 0xFFFF        # short
    assert placement(sigs["is_odd"]).ret.mask == 0xFF            # _Bool


def test_a_void_return_is_read_as_nothing(sigs):
    ret = placement(sigs["fill"]).ret
    assert ret.kind == "void" and ret.mask == 0


def test_a_float_return_comes_from_xmm0(sigs):
    assert placement(sigs["scale"]).ret.kind == "float"
    assert placement(sigs["scale"]).ret.size == 8
    assert placement(sigs["fma3"]).ret.size == 4


def test_a_pointer_return_is_its_own_kind_not_compared(sigs):
    """duplicate returns char*. A returned pointer is an address that differs
    between -O0 and -O3 (F18), so it has its own kind and is not compared by
    value - the function's effect is in the memory it wrote."""
    ret = placement(sigs["duplicate"]).ret
    assert ret.kind == "pointer"
    assert ret.mask == 0                # nothing of it is compared


# ---------------------------------------------------- what it declines


def test_a_small_struct_argument_rides_in_a_register(sigs):
    """pair_sum(struct pair) takes 8 bytes, which Win64 passes by value in one
    integer register - the struct's own bytes."""
    p = placement(sigs["pair_sum"])
    arg = p.args[0]
    assert isinstance(arg, IntArg)
    assert arg.register == "rcx" and arg.size == 8


def test_a_large_struct_argument_travels_by_reference(sigs):
    """big_sum(struct big) takes 24 bytes: too large for a register, so the
    caller passes a pointer to its copy."""
    p = placement(sigs["big_sum"])
    arg = p.args[0]
    assert isinstance(arg, IntArg)
    assert arg.register == "rcx" and arg.size == 8      # an address


def test_a_large_struct_return_uses_a_hidden_first_argument(sigs):
    """make_big(int) returns 24 bytes, which come back written into space the
    caller provides: its address goes in RCX and the declared int moves to
    RDX."""
    p = placement(sigs["make_big"])
    assert p.hidden_return
    assert len(p.args) == 2                              # hidden + the int
    assert p.args[0].register == "rcx" and p.args[0].size == 8
    assert p.args[1].register == "rdx"
    assert p.ret.kind == "pointer"                       # RAX is that address
    assert p.ret.mask == 0                               # so it is not compared


def test_an_incomplete_struct_is_still_declined():
    """A forward declaration with no size cannot be placed at all."""
    with pytest.raises(Undecidable, match="unknown size"):
        placement({"return": {"kind": "void", "size": 0},
                   "params": [{"kind": "struct", "size": 0}],
                   "variadic": False})


def test_a_variadic_function_is_declined(sigs):
    with pytest.raises(Undecidable, match="variadic"):
        placement(sigs["vsum"])


def test_long_double_is_declined(sigs):
    with pytest.raises(Undecidable, match="long double"):
        placement(sigs["wide"])


def test_no_signature_is_declined():
    with pytest.raises(Undecidable):
        placement(None)


def test_both_optimisation_levels_place_the_same():
    """The placement comes from the signature, which is the same at -O0 and
    -O3, so the harness sets up both versions identically."""
    o0 = {f.name: f.abi for f in ground_truth(FIXTURES / "cases_O0.dll")}
    o3 = {f.name: f.abi for f in ground_truth(FIXTURES / "cases_O3.dll")}
    for name in ("add3", "mixed", "six", "sum", "scale", "duplicate"):
        assert placement(o0[name]) == placement(o3[name])
