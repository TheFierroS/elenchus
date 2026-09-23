"""The Win64 calling convention, as pure data - no emulator here.

The verifier's experiment is defined by the reference function's signature
(docs/verifier.md): the claim "Q is K" means Q has K's parameters and return
type, so K's resolved signature - from dwarf.py's abi field - decides how
arguments are placed and how the return value is read, for both versions.

This module turns a resolved signature into a Placement: which register or
stack slot each argument occupies, and how wide (and whether masked) the
return value is. It knows the convention; it touches no memory and runs no
code, so it is tested on its own, and the harness that does touch memory is
tested against it.

Win64 (Microsoft x64), which is what MinGW targets:

- The first four arguments go in registers by position, not by counting each
  kind separately: an integer or pointer in RCX, RDX, R8, R9; a float or
  double in XMM0, XMM1, XMM2, XMM3. Position three is R8 or XMM2 depending on
  that argument's kind - the slot is used up either way.
- Arguments past the fourth go on the stack, above 32 bytes of shadow space
  the caller always reserves.
- A struct larger than 8 bytes is passed by a pointer to a copy; a struct of
  1, 2, 4 or 8 bytes is passed in one integer register. A struct returned by
  value larger than 8 bytes is written through a hidden first pointer
  argument, which shifts everything else along.
- The return value is in RAX for integers and pointers, XMM0 for float and
  double. A type narrower than 8 bytes defines only its low bytes; the rest
  of RAX is left however the code happened to leave it, so it is read masked
  to the type's width (docs/verifier.md, risk 1).

Anything the convention has a rule for but the harness cannot yet honour -
a by-value struct, a vararg, long double - is declined here, so the caller
makes the input inconclusive rather than placing an argument wrong.
"""

from dataclasses import dataclass, field

INT_REGISTERS = ("rcx", "rdx", "r8", "r9")
FLOAT_REGISTERS = ("xmm0", "xmm1", "xmm2", "xmm3")
SHADOW_SPACE = 32


class Undecidable(Exception):
    """The signature has a form the harness declines to place or read.

    Raised rather than guessed: a misplaced argument would make a true
    function disagree with itself. The caller turns it into an inconclusive
    verdict.
    """


@dataclass(frozen=True)
class IntArg:
    """An integer or pointer argument in a register or a stack slot."""
    register: str | None      # one of INT_REGISTERS, or None if on the stack
    stack_offset: int | None  # bytes above the shadow space, or None
    size: int
    signed: bool


@dataclass(frozen=True)
class FloatArg:
    """A float or double argument in an XMM register or a stack slot."""
    register: str | None      # one of FLOAT_REGISTERS, or None if on the stack
    stack_offset: int | None
    size: int


@dataclass(frozen=True)
class Return:
    """How to read the return value once the function has returned."""
    kind: str                 # "int", "float" or "void"
    size: int                 # bytes that are defined; 0 for void

    @property
    def mask(self) -> int:
        """The bits of the return register that are defined for this type.

        A char defines 8 bits of RAX, an int 32, a long 64; the rest is
        whatever was last in the register and differs between -O0 and -O3.
        """
        if self.kind in ("void", "pointer") or self.size == 0:
            return 0
        return (1 << (self.size * 8)) - 1


@dataclass(frozen=True)
class Placement:
    """Everything the harness needs to make one call and read its result."""
    args: tuple = field(default_factory=tuple)   # IntArg | FloatArg, in order
    ret: Return = field(default_factory=lambda: Return("void", 0))
    stack_bytes: int = 0                          # stack space for spilled args
    hidden_return: bool = False                   # a leading pointer to space
    #                                               for an aggregate return


# Win64 passes an aggregate of 1, 2, 4 or 8 bytes in one register, by value;
# any other size travels by reference, the caller having made a copy. A return
# follows the same sizes, and a larger one comes back through a hidden first
# pointer argument the caller supplies.
REGISTER_AGGREGATE_SIZES = (1, 2, 4, 8)


def aggregate_in_register(size: int) -> bool:
    """Whether an aggregate of `size` bytes rides in a register by value."""
    return size in REGISTER_AGGREGATE_SIZES


def returns_through_hidden_pointer(ret_abi) -> bool:
    """Whether a return needs the caller to pass space for it.

    A struct or union larger than a register comes back written into memory
    the caller provides, whose address goes in the first argument register
    and pushes every declared argument one place along.
    """
    if ret_abi is None:
        return False
    return (ret_abi.get("kind") in ("struct", "union")
            and not aggregate_in_register(ret_abi.get("size", 0)))


def _decline_if_unsupported(kind, size):
    if kind in ("struct", "union") and size == 0:
        # An incomplete type - a forward declaration with no definition here.
        # Its size decides how it travels, and without one it cannot be
        # placed.
        raise Undecidable(f"{kind} of unknown size")
    if kind == "float" and size > 8:
        # long double is MinGW's 80-bit x87 type in 16 bytes, passed by
        # reference on Win64. Declined until the harness handles it.
        raise Undecidable("long double")
    if kind in ("complex", "unknown"):
        raise Undecidable(f"unsupported type kind {kind!r}")


def placement(abi: dict) -> Placement:
    """Derive the Win64 placement for a resolved signature (dwarf.py's abi).

    Raises Undecidable for a signature the harness declines, so the caller
    never places an argument it is unsure of.
    """
    if abi is None:
        raise Undecidable("no resolved signature")
    if abi.get("variadic"):
        # The named arguments follow the convention, but a variadic callee
        # reads the rest by rules the harness does not model, so the call
        # cannot be set up faithfully. Declined whole.
        raise Undecidable("variadic function")

    args = []
    slot = 0                 # the next of the four register slots, by position
    stack_offset = 0         # bytes above the shadow space for spilled args

    ret_abi = abi.get("return") or {"kind": "void", "size": 0}
    _decline_if_unsupported(ret_abi["kind"], ret_abi["size"])
    hidden_return = returns_through_hidden_pointer(ret_abi)
    if hidden_return:
        # The caller provides space for the return value and passes its
        # address first, pushing every declared argument one place along.
        args.append(IntArg(INT_REGISTERS[0], None, 8, False))
        slot = 1

    for param in abi.get("params", []):
        kind = param["kind"]
        size = param["size"]
        _decline_if_unsupported(kind, size)
        is_float = kind == "float"
        # An aggregate of 1/2/4/8 bytes rides in a register by value; any
        # other size travels as a pointer to a copy the caller made.
        by_reference = (kind in ("struct", "union")
                        and not aggregate_in_register(size))

        if slot < 4:
            register = (FLOAT_REGISTERS if is_float else INT_REGISTERS)[slot]
            on_stack = None
            slot += 1
        else:
            register = None
            on_stack = stack_offset
            stack_offset += 8    # every spilled argument takes an 8-byte slot

        if is_float:
            args.append(FloatArg(register, on_stack, size))
        elif by_reference:
            # The argument is the address of the copy, so eight bytes.
            args.append(IntArg(register, on_stack, 8, False))
        else:
            signed = bool(param.get("signed"))
            args.append(IntArg(register, on_stack, size or 8, signed))

    if ret_abi["kind"] == "float":
        ret = Return("float", ret_abi["size"])
    elif ret_abi["kind"] == "void":
        ret = Return("void", 0)
    elif ret_abi["kind"] == "pointer":
        # A returned pointer is an address into the binary's own image or the
        # heap, at a different value in -O0 and -O3 the same way a global's
        # address is (F18, the return sibling of F15). It is never compared by
        # value; the function's effect is seen in the memory it wrote, not in
        # the address it handed back.
        ret = Return("pointer", 8)
    elif ret_abi["kind"] in ("struct", "union"):
        if hidden_return:
            # RAX comes back holding the hidden pointer the caller passed -
            # an address, not data, and the same reasoning as F18: the value
            # is not compared, the memory it points at is.
            ret = Return("pointer", 8)
        else:
            # A small aggregate rides home in RAX, its bytes the value.
            ret = Return("int", ret_abi["size"])
    else:
        # int, enum resolved to its underlying integer: read from RAX, masked
        # to the width the type defines.
        ret = Return("int", ret_abi["size"] or 8)

    return Placement(tuple(args), ret, stack_offset, hidden_return)
