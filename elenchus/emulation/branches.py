"""Reading a function's branches, so one of them can be forced.

The remaining coverage is behind input checks a function makes on data we
invented, and the way past them is to force the check the other way rather
than to satisfy it (docs/verifier.md, two tiers; docs/related-work.md). Two
things are needed for that, and both come from decoding:

  - a conditional jump's two destinations, so one can be chosen;
  - the values the compare before it looked at, so branches can be *ranked*
    by how close the comparison was - the dynamic selectivity PEM uses to pick
    corresponding branches in two builds without knowing the correspondence.

This module does the decoding and the arithmetic. It holds no emulator state:
it is handed bytes and a way to read registers and memory, which makes it
testable on its own and keeps the harness free of capstone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Branch:
    """A conditional jump, decoded once and cached by address."""
    address: int
    size: int
    target: int                  # where it goes when taken
    fall_through: int            # where it goes when not

    def other(self, taken: bool) -> int:
        """The destination it did *not* take - what forcing it means."""
        return self.fall_through if taken else self.target


@dataclass(frozen=True)
class Compare:
    """A flag-setting compare, and where its two values come from."""
    address: int
    left: tuple                  # ("reg", id) | ("imm", value) | ("mem", ...)
    right: tuple
    size: int                    # operand width in bytes
    signed: bool = True


@dataclass(frozen=True)
class PredicateInstance:
    """One conditional jump as it actually happened, during one run.

    `count` is the instruction number within the run, which is how a forced
    run is told *which* instance to force: the same jump in a loop is a
    different predicate instance each time round.
    """
    count: int
    branch: Branch
    taken: bool
    selectivity: int | None      # |x - y| at the compare, None if unknown


def _decoder():
    """A capstone decoder with operand detail, made once."""
    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    return md


# Mnemonics that set flags in a way a following jcc reads, and whose two
# operands are the values being compared. sub and and are included because a
# compiler emits them where cmp and test would do, and their flag effect is
# the same; add and or are not, because the interesting distance there is not
# between the operands.
COMPARE_MNEMONICS = {"cmp", "test", "sub", "and"}

# A conditional jump, as opposed to jmp - capstone puts both in the jump
# group, so the unconditional one is excluded by name.
UNCONDITIONAL = {"jmp", "ljmp"}


class Decoder:
    """Decodes instructions on demand and remembers what it found.

    A run executes the same addresses over and over inside loops, so the
    decode is cached by address; only the first visit costs anything.
    """

    def __init__(self, read_code):
        """read_code(address, size) -> bytes, usually the emulator's memory."""
        self._read = read_code
        self._md = None
        self._branches: dict[int, Branch | None] = {}
        self._compares: dict[int, Compare | None] = {}

    def _decode(self, address: int, size: int):
        if self._md is None:
            self._md = _decoder()
        try:
            code = self._read(address, size)
        except Exception:                       # noqa: BLE001 - unreadable code
            return None
        for instruction in self._md.disasm(code, address):
            return instruction
        return None

    def branch_at(self, address: int, size: int) -> Branch | None:
        """The conditional jump at `address`, or None if it is not one."""
        if address in self._branches:
            return self._branches[address]
        found = None
        instruction = self._decode(address, size)
        if instruction is not None and instruction.mnemonic not in UNCONDITIONAL:
            import capstone
            if capstone.x86.X86_GRP_JUMP in instruction.groups:
                operands = instruction.operands
                # A conditional jump's destination is an immediate; an
                # indirect jump has a register or memory operand and cannot be
                # forced to a known place, so it is not a branch we offer.
                if len(operands) == 1 and operands[0].type == capstone.x86.X86_OP_IMM:
                    found = Branch(address, instruction.size,
                                   operands[0].imm, address + instruction.size)
        self._branches[address] = found
        return found

    def compare_at(self, address: int, size: int) -> Compare | None:
        """The flag-setting compare at `address`, or None."""
        if address in self._compares:
            return self._compares[address]
        found = None
        instruction = self._decode(address, size)
        if instruction is not None and instruction.mnemonic in COMPARE_MNEMONICS:
            operands = instruction.operands
            if len(operands) == 2:
                left = _operand(instruction, operands[0])
                right = _operand(instruction, operands[1])
                if left and right:
                    found = Compare(address, left, right, operands[0].size)
        self._compares[address] = found
        return found


def _operand(instruction, operand) -> tuple | None:
    """Describe an operand as something a Machine can later evaluate."""
    import capstone
    if operand.type == capstone.x86.X86_OP_IMM:
        return ("imm", operand.imm)
    if operand.type == capstone.x86.X86_OP_REG:
        return ("reg", instruction.reg_name(operand.reg))
    if operand.type == capstone.x86.X86_OP_MEM:
        mem = operand.mem
        return ("mem",
                instruction.reg_name(mem.base) if mem.base else None,
                instruction.reg_name(mem.index) if mem.index else None,
                mem.scale, mem.disp, operand.size)
    return None


def selectivity(compare: Compare, read_register, read_memory) -> int | None:
    """How far the compare was from going the other way: |x - y|.

    None when either value cannot be read - an unreadable address, a register
    the caller does not know. A predicate with no selectivity is still a
    predicate; it simply cannot be ranked, and ranking is what selectivity is
    for.

    Values are taken at the operand's width and treated as signed, because
    that is the comparison a compiler most often means and the distance is
    what matters rather than the sign.
    """
    left = _value(compare.left, compare.size, read_register, read_memory)
    right = _value(compare.right, compare.size, read_register, read_memory)
    if left is None or right is None:
        return None
    return abs(left - right)


def _value(operand: tuple, size: int, read_register, read_memory):
    kind = operand[0]
    if kind == "imm":
        return _signed(operand[1] & _mask(size), size)
    if kind == "reg":
        raw = read_register(operand[1])
        if raw is None:
            return None
        return _signed(raw & _mask(size), size)
    if kind == "mem":
        _, base, index, scale, disp, width = operand
        address = disp
        if base:
            value = read_register(base)
            if value is None:
                return None
            address += value
        if index:
            value = read_register(index)
            if value is None:
                return None
            address += value * scale
        raw = read_memory(address & 0xFFFFFFFFFFFFFFFF, width)
        if raw is None:
            return None
        return _signed(int.from_bytes(raw, "little"), width)
    return None


def _mask(size: int) -> int:
    return (1 << (size * 8)) - 1


def _signed(value: int, size: int) -> int:
    bits = size * 8
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return value
