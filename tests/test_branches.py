"""Decoding branches and measuring how close a comparison came.

These are arithmetic and decoding, with no emulator, so they are checked on
hand-assembled bytes where the right answer is known by construction.
"""

import pytest

pytest.importorskip("capstone")

from elenchus.emulation.branches import (  # noqa: E402
    Compare,
    Decoder,
    selectivity,
)


def decoder(code: bytes, base: int = 0x1000):
    """A Decoder over a fixed block of bytes."""
    def read(address, size):
        offset = address - base
        if offset < 0 or offset >= len(code):
            raise ValueError("outside the block")
        return code[offset:offset + size]
    return Decoder(read)


# cmp eax, 0x20 ; jne +5 ; nop
CMP_JNE = bytes.fromhex("83f820" "7505" "90")


# ------------------------------------------------------------- branches


def test_a_conditional_jump_gives_both_destinations():
    """Forcing a branch means sending it to the place it did not go, so both
    have to be known."""
    d = decoder(CMP_JNE)
    branch = d.branch_at(0x1003, 8)
    assert branch is not None
    assert branch.target == 0x100A            # 0x1005 + 5
    assert branch.fall_through == 0x1005      # after the two-byte jne
    assert branch.other(taken=True) == 0x1005
    assert branch.other(taken=False) == 0x100A


def test_an_unconditional_jump_is_not_a_branch():
    """jmp has nothing to force: there is no other way for it to go."""
    d = decoder(bytes.fromhex("eb05"))        # jmp +5
    assert d.branch_at(0x1000, 8) is None


def test_an_indirect_jump_is_not_offered():
    """jmp rax could go anywhere; forcing it to a place we invented would be
    forcing the function somewhere it could never reach."""
    d = decoder(bytes.fromhex("ffe0"))        # jmp rax
    assert d.branch_at(0x1000, 8) is None


def test_an_ordinary_instruction_is_not_a_branch():
    d = decoder(bytes.fromhex("4801d8"))      # add rax, rbx
    assert d.branch_at(0x1000, 8) is None


def test_the_decode_is_cached():
    """A loop executes the same address thousands of times; it is decoded
    once."""
    calls = []

    def read(address, size):
        calls.append(address)
        return CMP_JNE[address - 0x1000:address - 0x1000 + size]

    d = Decoder(read)
    for _ in range(5):
        d.branch_at(0x1003, 8)
    assert len(calls) == 1


# ------------------------------------------------------------- compares


def test_a_compare_is_found_with_its_operands():
    d = decoder(CMP_JNE)
    compare = d.compare_at(0x1000, 8)
    assert compare is not None
    assert compare.left == ("reg", "eax")
    assert compare.right == ("imm", 0x20)


def test_a_test_instruction_counts_as_a_compare():
    d = decoder(bytes.fromhex("85c0"))        # test eax, eax
    assert d.compare_at(0x1000, 8) is not None


def test_an_add_is_not_a_compare():
    """add sets flags too, but the distance between its operands is not what
    the following branch turns on."""
    d = decoder(bytes.fromhex("4801d8"))      # add rax, rbx
    assert d.compare_at(0x1000, 8) is None


# ---------------------------------------------------------- selectivity


def registers(**values):
    return lambda name: values.get(name)


def test_selectivity_is_the_distance_between_the_values():
    """PEM's dynamic selectivity: how far the comparison was from going the
    other way, which is what makes a branch rankable."""
    d = decoder(CMP_JNE)
    compare = d.compare_at(0x1000, 8)
    assert selectivity(compare, registers(eax=0xAD), None) == 0xAD - 0x20


def test_a_comparison_that_barely_went_one_way_has_small_selectivity():
    d = decoder(CMP_JNE)
    compare = d.compare_at(0x1000, 8)
    assert selectivity(compare, registers(eax=0x21), None) == 1
    assert selectivity(compare, registers(eax=0x20), None) == 0


def test_values_are_read_at_the_operands_width():
    """A 32-bit compare looks at 32 bits: the rest of the register is not
    part of the comparison and must not be part of the distance."""
    d = decoder(CMP_JNE)
    compare = d.compare_at(0x1000, 8)
    assert selectivity(compare, registers(eax=0xFFFF_FFFF_0000_0021), None) == 1


def test_a_value_that_cannot_be_read_gives_no_selectivity():
    """A predicate with no selectivity is still a predicate - it just cannot
    be ranked, and ranking is all selectivity is for."""
    d = decoder(CMP_JNE)
    compare = d.compare_at(0x1000, 8)
    assert selectivity(compare, registers(), None) is None


def test_a_memory_operand_is_read_through_the_reader():
    """cmp dword ptr [rbx + 8], 0x20"""
    d = decoder(bytes.fromhex("837b0820"))
    compare = d.compare_at(0x1000, 8)
    assert compare is not None
    seen = {}

    def memory(address, size):
        seen["address"] = address
        return (0x30).to_bytes(size, "little")

    assert selectivity(compare, registers(rbx=0x2000), memory) == 0x10
    assert seen["address"] == 0x2008


def test_a_signed_comparison_measures_a_real_distance():
    """-1 against 1 is two apart, not four billion."""
    compare = Compare(0x1000, ("imm", -1), ("imm", 1), 4)
    assert selectivity(compare, registers(), None) == 2
