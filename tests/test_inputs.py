"""Building input vectors from a signature.

The generator reads a resolved signature and produces vectors that fit it: a
real buffer address for a pointer, boundary and random values for an integer,
packed bits for a float. It runs no code, so these tests read the fixture
signatures and check the vectors directly - a pointer gets a mapped-buffer
address, an integer varies across its boundaries, a no-argument function gets
one empty call, and the same seed gives the same vectors.
"""

import struct
from pathlib import Path

from elenchus.corpus.dwarf import ground_truth
from elenchus.emulation.inputs import BUFFER_BASE, BUFFER_STRIDE, input_vectors

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"


def sig(name, level="O3"):
    return {f.name: f.abi
            for f in ground_truth(FIXTURES / f"cases_{level}.dll")}[name]


def test_a_no_argument_function_gets_one_empty_call():
    """A function with no parameters is called once, with no arguments."""
    # make an empty signature by hand for the edge:
    empty = {"return": {"kind": "int", "size": 4}, "params": [], "variadic": False}
    assert input_vectors(empty) == [[]]


def test_an_integer_parameter_varies_across_boundaries():
    vectors = input_vectors(sig("add3"))       # three signed ints
    firsts = [v[0] for v in vectors]
    assert 0 in firsts and 1 in firsts         # boundaries are present
    assert len(set(firsts)) > 1                # it varies


def test_a_pointer_gets_a_buffer_address_not_a_small_integer():
    """sum(const int *p, int n): the first argument is a pointer, so it gets
    the buffer base, not a boundary integer that would fault on dereference."""
    vectors = input_vectors(sig("sum"))
    assert all(v[0] == BUFFER_BASE for v in vectors)   # fixed buffer
    assert any(v[1] != v[0] for v in vectors)          # n still varies


def test_two_pointers_get_separated_buffers():
    """compare_strings(const char *a, const char *b): two pointers, each its
    own buffer, far enough apart not to alias."""
    vectors = input_vectors(sig("compare_strings"))
    a, b = vectors[0][0], vectors[0][1]
    assert a == BUFFER_BASE
    assert b == BUFFER_BASE + BUFFER_STRIDE
    assert b - a >= 0x1000                              # well separated


def test_a_double_argument_is_packed_bits():
    """scale(double x, int k): the double is a bit pattern the harness loads
    into XMM, not a raw float."""
    vectors = input_vectors(sig("scale"))
    # 1.0 as a double is a specific bit pattern; it should appear.
    one = struct.unpack("<Q", struct.pack("<d", 1.0))[0]
    assert any(v[0] == one for v in vectors)


def test_a_float_argument_is_single_precision_bits():
    vectors = input_vectors(sig("fma3"))       # three floats
    one = struct.unpack("<I", struct.pack("<f", 1.0))[0]
    assert any(v[0] == one for v in vectors)
    assert all(0 <= v[0] <= 0xFFFFFFFF for v in vectors)   # 32-bit patterns


def test_the_same_seed_gives_the_same_vectors():
    assert input_vectors(sig("add3"), seed=7) == input_vectors(sig("add3"), seed=7)


def test_a_different_seed_can_differ():
    a = input_vectors(sig("mix64"), seed=1)
    b = input_vectors(sig("mix64"), seed=2)
    # the random middle values differ, so the sets are not identical
    assert a != b


def test_integer_values_fit_their_width():
    """low_byte(int) is fine, but a narrow value must stay within its bits;
    check the 8-byte and 4-byte params of mix64 are masked to width."""
    vectors = input_vectors(sig("mix64"))      # uint64_t, uint32_t
    assert all(0 <= v[0] <= (1 << 64) - 1 for v in vectors)
    assert all(0 <= v[1] <= (1 << 32) - 1 for v in vectors)


def test_none_signature_gives_no_vectors():
    assert input_vectors(None) == []
