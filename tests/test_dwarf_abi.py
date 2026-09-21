"""Resolving a function's signature to what a call needs.

ground_truth used to record types by name - uint32_t, size_t, word - and a
name does not say how wide a value is, or whether a function takes "...".
The verifier passes arguments and reads return values by kind, width and
sign, so dwarf.py now resolves each type through its typedefs to a base
type. These read the emulation fixture, built from cases.c at -O0 and -O3
by the corpus toolchain.
"""

from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"


@pytest.fixture(scope="module", params=["O0", "O3"])
def functions(request):
    return {f.name: f for f in ground_truth(FIXTURES / f"cases_{request.param}.dll")}


def kind(t):
    return (t["kind"], t["size"], t["signed"])


def test_plain_integers(functions):
    abi = functions["add3"].abi
    assert kind(abi["return"]) == ("int", 4, True)
    assert [kind(p) for p in abi["params"]] == [("int", 4, True)] * 3
    assert abi["variadic"] is False


def test_a_typedef_resolves_to_its_width(functions):
    """uint64_t and uint32_t are names; their widths are in the base types."""
    abi = functions["mix64"].abi
    assert kind(abi["return"]) == ("int", 8, False)
    assert [kind(p) for p in abi["params"]] == [("int", 8, False), ("int", 4, False)]


def test_a_typedef_of_a_typedef_resolves_all_the_way(functions):
    """word is u32 is unsigned int: a name-based table would stop at word."""
    abi = functions["through_typedef"].abi
    assert kind(abi["return"]) == ("int", 4, False)
    assert kind(abi["params"][0]) == ("int", 4, False)


def test_size_t_is_eight_unsigned_bytes(functions):
    abi = functions["count_nonzero"].abi
    assert kind(abi["return"]) == ("int", 8, False)
    assert [kind(p) for p in abi["params"]] == [("pointer", 8, False), ("int", 8, False)]


def test_narrow_returns_keep_their_width(functions):
    """The verifier masks RAX to this width; wider would compare garbage."""
    assert kind(functions["low_byte"].abi["return"]) == ("int", 1, True)
    assert kind(functions["low_word"].abi["return"]) == ("int", 2, False)
    assert kind(functions["is_odd"].abi["return"]) == ("int", 1, False)


def test_void_has_no_width(functions):
    assert kind(functions["fill"].abi["return"]) == ("void", 0, None)


def test_floating_point_widths(functions):
    assert kind(functions["scale"].abi["return"]) == ("float", 8, None)
    assert [kind(p) for p in functions["fma3"].abi["params"]] == [("float", 4, None)] * 3


def test_long_double_is_sixteen_bytes(functions):
    """MinGW's long double is the x87 80-bit type in 16 bytes; Win64 passes
    it by reference, which the harness must decline rather than misplace."""
    abi = functions["wide"].abi
    assert kind(abi["return"]) == ("float", 16, None)
    assert kind(abi["params"][0]) == ("float", 16, None)


def test_an_enum_resolves_to_its_underlying_type(functions):
    assert kind(functions["colour_value"].abi["params"][0]) == ("int", 4, False)


def test_pointers_of_every_shape_are_pointers(functions):
    """A function pointer and a pointer to a const pointer are both one
    eight-byte register, whatever the readable name makes of them."""
    assert kind(functions["apply"].abi["params"][0]) == ("pointer", 8, False)
    assert kind(functions["first_char"].abi["params"][0]) == ("pointer", 8, False)
    assert kind(functions["first_char"].abi["return"]) == ("pointer", 8, False)


def test_structs_carry_their_size(functions):
    """Win64 passes an 8-byte struct in a register and a 24-byte one by
    reference, and returns a 24-byte one through a hidden first argument;
    the size is what decides it."""
    assert kind(functions["pair_sum"].abi["params"][0]) == ("struct", 8, None)
    assert kind(functions["big_sum"].abi["params"][0]) == ("struct", 24, None)
    assert kind(functions["make_big"].abi["return"]) == ("struct", 24, None)


def test_a_variadic_function_says_so(functions):
    """vsum(int n, ...) used to be recorded as vsum(int) - a function the
    verifier would have called with one argument."""
    f = functions["vsum"]
    assert f.abi["variadic"] is True
    assert len(f.abi["params"]) == 1
    assert f.signature == "int vsum(int, ...)"


def test_const_on_a_pointer_is_written_after_it(functions):
    """const char *const * used to read "const const char**"."""
    signature = functions["first_char"].signature
    assert signature == "const char* first_char(const char* const*)"


def test_const_on_the_pointee_still_reads_as_before(functions):
    assert functions["copy"].signature == "void copy(char*, const char*, size_t)"


def test_the_readable_names_are_unchanged():
    """param_types keep the names; only abi resolves them."""
    f = {g.name: g for g in ground_truth(FIXTURES / "cases_O3.dll")}["mix64"]
    assert f.param_types == ("uint64_t", "uint32_t")
    assert f.return_type == "uint64_t"


def test_both_levels_agree_on_every_signature():
    """The same source at -O0 and -O3 has one signature. The verifier takes
    the reference's; if the levels disagreed, it would matter which."""
    o0 = {f.name: f.abi for f in ground_truth(FIXTURES / "cases_O0.dll")}
    o3 = {f.name: f.abi for f in ground_truth(FIXTURES / "cases_O3.dll")}
    common = set(o0) & set(o3)
    assert len(common) > 30
    assert {n: o0[n] for n in common} == {n: o3[n] for n in common}


def test_the_resolved_signature_reaches_the_database(db):
    """What refresh-truth will write for every corpus binary."""
    import json

    from elenchus.corpus.store import store_ground_truth

    binary = db.execute(
        "INSERT INTO binaries (path, sha256, arch, imported_at) "
        "VALUES ('cases_O3_stripped.dll', 'x', 'x86-64', 'now') RETURNING id"
    ).fetchone()["id"]
    db.commit()

    store_ground_truth(db, binary, FIXTURES / "cases_O3.dll")

    rows = {r["name"]: json.loads(r["abi"]) for r in db.execute(
        "SELECT name, abi FROM ground_truth WHERE binary_id = ?", (binary,))}
    assert rows["vsum"]["variadic"] is True
    assert rows["mix64"]["params"][1] == {"kind": "int", "size": 4, "signed": False}
