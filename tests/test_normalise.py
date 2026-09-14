"""Tests for the instruction normalisation scheme.

Pure text in, tokens out - no Ghidra, no database - so these run in CI and
in a second. The point of each test is that something the compiler chose
arbitrarily (a register, an offset, an address) disappears, while something
the function actually does survives.
"""

from elenchus.encoder.normalise import (
    immediate_class,
    mnemonics,
    normalise,
    normalise_line,
    operand_token,
    register_class,
    target_token,
)


def line(offset, mnemonic, operands="", target=""):
    """Build one listing line the way extract/instructions.py writes it."""
    return "\t".join((offset, mnemonic, operands, target))


def test_register_widths():
    assert register_class("rax") == "REG64"
    assert register_class("r12") == "REG64"
    assert register_class("eax") == "REG32"
    assert register_class("r12d") == "REG32"
    assert register_class("ax") == "REG16"
    assert register_class("al") == "REG8"
    assert register_class("r9b") == "REG8"
    assert register_class("xmm0") == "VEC"
    assert register_class("fs") == "SEG"
    assert register_class("0x10") is None


def test_immediate_buckets():
    assert immediate_class(0) == "IMM_ZERO"
    assert immediate_class(1) == "IMM_SMALL"
    assert immediate_class(-8) == "IMM_SMALL"
    assert immediate_class(0x1000) == "IMM_MED"
    assert immediate_class(0x401230) == "IMM_LARGE"


def test_memory_operands():
    assert operand_token("qwordptr[RBP+-0x18]") == "MEM_STACK"
    assert operand_token("[RSP+0x20]") == "MEM_STACK"
    assert operand_token("qwordptr[RIP+0x2ef1]") == "MEM_GLOBAL"
    assert operand_token("[0x404000]") == "MEM_GLOBAL"
    assert operand_token("byteptr[RAX+RCX*0x4]") == "MEM_INDEXED"
    assert operand_token("[RAX]") == "MEM_REG"
    # A field offset survives optimisation the way a stack offset does not,
    # so its presence is kept even though its value is not.
    assert operand_token("qwordptr[RAX+0x8]") == "MEM_FIELD"
    assert operand_token("[RBX+-0x10]") == "MEM_FIELD"
    assert operand_token("[RAX+0x0]") == "MEM_REG"


def test_stack_offset_is_discarded():
    """Two frames differing only in layout must normalise identically."""
    a = normalise_line(line("0", "mov", "RAX qwordptr[RBP+-0x18]"))
    b = normalise_line(line("0", "mov", "RCX qwordptr[RBP+-0x40]"))
    assert a == b == ["mov", "REG64", "MEM_STACK"]


def test_branch_direction_survives_address_does_not():
    assert target_token("LOCAL:+0x2f") == "BB_FWD"
    assert target_token("LOCAL:-0x2f") == "BB_BACK"
    assert normalise_line(line("10", "jmp", "0x40129a", "LOCAL:-0x30")) == [
        "jmp",
        "BB_BACK",
    ]


def test_import_name_is_kept():
    """The one label a stripped Windows binary cannot hide."""
    tokens = normalise_line(
        line("20", "call", "qwordptr[0x404120]", "IMPORT:KERNEL32.dll!CreateFileW")
    )
    assert tokens == ["call", "IMPORT:CreateFileW"]


def test_internal_call_loses_its_address():
    assert normalise_line(line("30", "call", "0x401230", "CALL:internal")) == [
        "call",
        "FUNC_INTERNAL",
    ]
    assert normalise_line(line("30", "call", "RAX", "CALL:indirect")) == [
        "call",
        "REG64",
        "FUNC_INDIRECT",
    ]


def test_normalise_whole_listing():
    listing = "\n".join([
        line("0", "push", "RBP"),
        line("1", "mov", "RBP RSP"),
        line("4", "cmp", "dwordptr[RBP+-0x4] 0x1000"),
        line("b", "jz", "0x401040", "LOCAL:+0x12"),
        line("d", "call", "qwordptr[0x404120]", "IMPORT:KERNEL32.dll!ReadFile"),
    ])

    assert normalise(listing) == [
        "push", "REG64",
        "mov", "REG64", "REG64",
        "cmp", "MEM_STACK", "IMM_MED",
        "jz", "BB_FWD",
        "call", "IMPORT:ReadFile",
    ]


def test_no_absolute_address_survives():
    """The property the whole scheme exists for."""
    listing = "\n".join([
        line("0", "mov", "RAX qwordptr[RIP+0x2ef1]"),
        line("7", "call", "0x401230", "CALL:internal"),
        line("c", "jmp", "0x40129a", "LOCAL:+0x8"),
    ])
    tokens = normalise(listing)
    assert not any("0x" in token for token in tokens)


def test_mnemonics_only():
    listing = "\n".join([
        line("0", "push", "RBP"),
        line("1", "ret"),
    ])
    assert mnemonics(listing) == ["push", "ret"]


def test_malformed_line_is_ignored():
    assert normalise_line("nonsense") == []
    assert normalise("\n\n") == []
