"""Turn a raw instruction listing into the token sequence the encoder reads.

Raw disassembly is far too easy to memorise. `mov rax, [rbp-0x18]` names a
register the compiler picked arbitrarily and an offset that depends on how
many locals happened to come before it; recompile with a different
optimisation level and both change while the function does exactly the same
work. A model trained on that text learns GCC's register allocator, not what
the function means.

So the text is reduced to classes:

    mov rax,[rbp-0x18]        ->  mov REG64 MEM_STACK
    cmp eax,0x1000            ->  cmp REG32 IMM_LARGE
    jmp <forwards>            ->  jmp BB_FWD
    call <another function>   ->  call FUNC_INTERNAL
    call CreateFileW          ->  call IMPORT:CreateFileW

What survives: the mnemonic, the shape of each operand, the direction of
control flow, and the names of imported functions. That last one is kept
deliberately. On Windows an import name is not an arbitrary label the
compiler chose - it is a fixed, meaningful string that survives stripping,
and a function calling CreateFileW and ReadFile is doing file I/O whatever
its own name used to be.

What is discarded: absolute addresses, which register was chosen, exact stack
offsets, exact constants. These are the memorisation handles.

This module is pure text in, tokens out. No database, no Ghidra. That is what
makes the scheme cheap to change: normalisation can be rerun over a stored
listing in seconds, while producing that listing took Ghidra minutes.
"""

import re

_SIZE_PREFIX = re.compile(
    r"^(byte|word|dword|qword|tbyte|xmmword|ymmword|zmmword)ptr"
)
_REGISTER_NUMBERED = re.compile(r"^r(\d+)(d|w|b)?$")
_HEX = re.compile(r"^-?0x[0-9a-f]+$", re.IGNORECASE)
_DECIMAL = re.compile(r"^-?\d+$")

_REG64 = {
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "rip",
}
_REG32 = {
    "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
}
_REG16 = {
    "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
}
_REG8 = {
    "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh", "sil", "dil", "bpl", "spl",
}
_SEGMENT = {"cs", "ds", "es", "fs", "gs", "ss"}

# Targets whose destination Ghidra could not pin down. For these the operand
# is kept, because how the address is computed is the only thing left to say.
_UNRESOLVED = {"FUNC_INDIRECT", "FUNC_UNKNOWN", "TAIL_UNKNOWN"}

# Registers that address the stack frame. A memory operand built on one of
# these is a local variable or an argument, which is a different kind of thing
# from a global or a heap pointer, and worth telling apart.
_STACK = {"rbp", "rsp", "ebp", "esp"}


def register_class(name):
    """Classify a register name by width, or return None if it is not one."""
    name = name.lower()

    if name.startswith(("xmm", "ymm", "zmm")):
        return "VEC"
    if name in _REG64:
        return "REG64"
    if name in _REG32:
        return "REG32"
    if name in _REG16:
        return "REG16"
    if name in _REG8:
        return "REG8"
    if name in _SEGMENT:
        return "SEG"

    numbered = _REGISTER_NUMBERED.match(name)
    if numbered and 8 <= int(numbered.group(1)) <= 15:
        suffix = numbered.group(2)
        return {None: "REG64", "d": "REG32", "w": "REG16", "b": "REG8"}[suffix]

    return None


def _scalar_value(text):
    """Parse a hex or decimal literal, or return None if it is neither."""
    if _HEX.match(text):
        return int(text, 16)
    if _DECIMAL.match(text):
        return int(text)
    return None


def immediate_class(value):
    """Bucket a constant by magnitude.

    The exact number is a memorisation handle - a buffer size, a magic
    constant, a loop bound - but its size still says something: zero tends to
    be initialisation or a null check, a small value an index or a flag, a
    large one an address, a mask, or a table.
    """
    value = abs(value)
    if value == 0:
        return "IMM_ZERO"
    if value <= 0xFF:
        return "IMM_SMALL"
    if value <= 0xFFFF:
        return "IMM_MED"
    return "IMM_LARGE"


def memory_class(inside):
    """Classify a memory operand from what sits between the brackets.

    The distinction that matters here is what stays put across optimisation
    levels. A stack offset does not: the compiler lays out locals as it
    pleases, and -O2 may not give a variable a stack slot at all. A field
    offset does: [RAX+0x8] is the second member of whatever RAX points at,
    and the layout of a struct is fixed by the source, not the compiler. So
    the number is still dropped - it would be memorised - but the fact that
    one is present is kept, because it survives the transformation that the
    encoder is being asked to see through.
    """
    parts = [part for part in re.split(r"[+\-*]", inside) if part]
    registers = [part for part in parts if register_class(part) is not None]
    literals = [
        part for part in parts
        if register_class(part) is None and _scalar_value(part) is not None
    ]

    if any(r.lower() in _STACK for r in registers):
        return "MEM_STACK"
    if any(r.lower() == "rip" for r in registers):
        return "MEM_GLOBAL"
    if "*" in inside:
        return "MEM_INDEXED"
    if not registers:
        return "MEM_GLOBAL"
    if any(value != 0 for value in (_scalar_value(part) for part in literals)):
        return "MEM_FIELD"
    return "MEM_REG"


def operand_token(text):
    """Reduce one operand to a single class token."""
    if not text:
        return "OP_NONE"

    stripped = _SIZE_PREFIX.sub("", text)

    if stripped.startswith("[") and stripped.endswith("]"):
        return memory_class(stripped[1:-1])

    register = register_class(stripped)
    if register is not None:
        return register

    value = _scalar_value(stripped)
    if value is not None:
        return immediate_class(value)

    # A named location Ghidra resolved - a data label, a jump table, a symbol
    # that outlived stripping in some other form. Rare, and not worth its own
    # vocabulary entry.
    return "OP_OTHER"


def target_token(target):
    """Reduce a call or branch target to a token.

    An import keeps its name; everything else keeps only its direction or
    kind, because the destination address is exactly what must not be learnt.
    """
    if not target:
        return None

    kind, _, rest = target.partition(":")

    if kind == "IMPORT":
        _, _, name = rest.partition("!")
        return f"IMPORT:{name or rest}"

    if kind == "LOCAL":
        return "BB_BACK" if rest.startswith("-") else "BB_FWD"

    if kind == "CALL":
        return {
            "internal": "FUNC_INTERNAL",
            "indirect": "FUNC_INDIRECT",
        }.get(rest, "FUNC_UNKNOWN")

    if kind == "TAIL":
        return "TAIL_INTERNAL" if rest == "internal" else "TAIL_UNKNOWN"

    return "FUNC_UNKNOWN"


def normalise_line(line):
    """Turn one listing line into its tokens: a mnemonic, then its operands."""
    parts = line.split("\t")
    if len(parts) < 4:
        return []

    _offset, mnemonic, operands, target = parts[0], parts[1], parts[2], parts[3]

    resolved = target_token(target)
    tokens = [mnemonic]

    # When the destination is known, the operand only spells out the address
    # the target token already describes, so it is dropped. When it is not
    # known, the operand is the only thing that says *how* the destination is
    # computed - a register, a table, a pointer loaded from an object - and
    # that distinction is worth keeping.
    skip_operands = resolved is not None and resolved not in _UNRESOLVED

    for operand in operands.split(" "):
        if not operand or skip_operands:
            continue
        tokens.append(operand_token(operand))

    if resolved is not None:
        tokens.append(resolved)

    return tokens


def normalise(listing):
    """Turn a whole function listing into one flat token sequence."""
    tokens = []
    for line in listing.splitlines():
        if line.strip():
            tokens.extend(normalise_line(line))
    return tokens


def mnemonics(listing):
    """Return just the mnemonics, in order.

    The BM25 baseline in week 5 works on mnemonic n-grams, and so does a quick
    eyeball check of whether two functions look at all alike.
    """
    out = []
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append(parts[1])
    return out
