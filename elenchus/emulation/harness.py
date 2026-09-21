"""Run one function of a binary in isolation and observe what it did.

This is the harness of docs/verifier.md: it maps a binary's own sections at
its image base, places arguments by the Win64 convention (abi.py), runs the
function under Unicorn until it returns to a sentinel or the instruction
budget is spent, and reports the masked return value and the memory it wrote
outside its own stack. It follows calls within the binary into their real
code (tier 1); a call that leaves the binary stops at a hook the caller
resolves - a stub, or inconclusive.

Nothing here decides a verdict. It runs a function and records what
happened; comparing two runs is the caller's job. The one rule it keeps is
determinism: the same binary, function, arguments and seed give the same
Outcome every time, so a verdict can be replayed.

Two kinds of memory are filled differently, which is the correction the
first smoke test forced (docs/verifier.md, risk 2):

- Memory reached through an argument pointer is *input*. It is filled from
  its address alone, the same in both runs, so a function that reads its
  argument sees the same bytes each time.
- The stack, and whatever an allocator stub hands out, is *uninitialised*.
  It is filled from the address and a seed, so running twice with two seeds
  reveals a result that depends on garbage.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pefile

from elenchus.emulation.abi import FLOAT_REGISTERS, Placement

PAGE = 0x1000
GOLDEN = 0x9E3779B97F4A7C15          # an odd constant, for a spread-out fill

# A private address range for the call frame and for memory the harness hands
# out, chosen far from any image base so it never overlaps a mapped binary.
STACK_TOP = 0x0000_7FF0_0000_0000
STACK_SIZE = 0x0010_0000             # 1 MiB
SENTINEL = 0x0000_0000_0000_1000     # return address; not a real code page
UNMAPPED_FILL_LIMIT = 4096           # pages the harness will map on first touch

# Imports are given trap addresses in a page of their own. A call to an
# import compiles to `call [thunk]`, an indirect call through the import
# address table; the real loader fills each slot with the function's address,
# and here each slot is filled with a distinct trap address in this page. A
# call then lands on the trap, which the code hook catches - a stub, or
# IMPORT_WITHOUT_STUB - instead of jumping into unresolved bytes.
TRAP_BASE = 0x0000_0000_0000_2000    # first trap; one import every 16 bytes
TRAP_STRIDE = 0x10

# The lowest page is left unmapped on purpose, so a null dereference faults
# rather than being filled like any other first-touched address.
NULL_GUARD = 0x1_0000


class Status(Enum):
    """Why a run ended. Only COMPLETED yields a return value to compare."""
    COMPLETED = "completed"
    UNSUPPORTED_INSTRUCTION = "unsupported instruction"
    IMPORT_WITHOUT_STUB = "import without a stub"
    BUDGET_EXHAUSTED = "budget exhausted"
    TOO_MUCH_MEMORY = "mapped too much memory"
    FAULT = "fault"


@dataclass
class Outcome:
    """What one run of one function produced."""
    status: Status
    ret_int: int = 0                 # RAX, unmasked; mask with the placement
    ret_float_bits: int = 0          # XMM0's low 64 bits, for a float return
    writes: dict = field(default_factory=dict)   # page base -> bytes it holds
    detail: str = ""                 # the import name, the faulting address...
    instructions: int = 0


def _fill(page_base: int, seed: int) -> bytes:
    """The bytes a freshly mapped page holds: a function of its address, and
    of the seed for uninitialised memory (seed 0 means input memory)."""
    out = bytearray()
    for offset in range(0, PAGE, 8):
        word = ((page_base + offset) * GOLDEN + seed) & 0xFFFFFFFFFFFFFFFF
        out += struct.pack("<Q", word)
    return bytes(out)


class Loader:
    """A binary mapped at its image base, read once and reused across runs.

    Parsing a PE and copying its sections is the same for every call into it,
    so it is done once. The import table is read here too: each thunk's
    address is mapped to the name it imports, so a call through it can be
    recognised as leaving the binary.
    """

    def __init__(self, path: str | Path):
        self.pe = pefile.PE(str(path), fast_load=False)
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.size = (self.pe.OPTIONAL_HEADER.SizeOfImage + PAGE - 1) & ~(PAGE - 1)
        # slot address -> (dll, name), and trap address -> name.
        self.slots, self.traps = self._import_traps()

    def _import_traps(self) -> tuple[dict, dict]:
        """Assign each named import a trap address and record both directions.

        slots maps the import-address-table slot to (dll, name); the slot is
        overwritten with the trap address at load time, so `call [slot]` lands
        on the trap. traps maps the trap address back to the import name, so
        the code hook can name what was called.
        """
        slots, traps = {}, {}
        if not hasattr(self.pe, "DIRECTORY_ENTRY_IMPORT"):
            return slots, traps
        index = 0
        for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode(errors="replace")
            for imp in entry.imports:
                if not imp.name:
                    continue
                name = imp.name.decode(errors="replace")
                trap = TRAP_BASE + index * TRAP_STRIDE
                slots[imp.address] = (dll, name, trap)
                traps[trap] = name
                index += 1
        return slots, traps

    def write_sections(self, uc) -> None:
        """Copy the headers and every section into an emulator's memory, then
        point every import slot at its trap.

        A section's raw data can be shorter than its virtual size (.bss holds
        zeros with no file bytes); the mapping is already zero, so copying the
        raw bytes over it leaves the rest zero. After the sections are in, the
        import slots are overwritten with trap addresses (see _import_traps).
        """
        uc.mem_write(self.base, self.pe.header)
        for section in self.pe.sections:
            data = section.get_data()
            if data:
                uc.mem_write(self.base + section.VirtualAddress, data)
        for slot, (_dll, _name, trap) in self.slots.items():
            uc.mem_write(slot, struct.pack("<Q", trap))


def _load_arguments(uc, placement: Placement, args, rsp: int) -> None:
    """Place arguments in registers and the stack per the Win64 placement.

    Integer and pointer values go in RCX/RDX/R8/R9 or into stack slots above
    the shadow space; float and double values go in XMM0-3 or the same stack
    slots. args is a sequence of ints: an integer value, a pointer's address,
    or the bit pattern of a float already packed to its width.
    """
    from unicorn.x86_const import (
        UC_X86_REG_R8,
        UC_X86_REG_R9,
        UC_X86_REG_RCX,
        UC_X86_REG_RDX,
    )

    int_regs = {"rcx": UC_X86_REG_RCX, "rdx": UC_X86_REG_RDX,
                "r8": UC_X86_REG_R8, "r9": UC_X86_REG_R9}

    for arg, value in zip(placement.args, args):
        if arg.register in int_regs:
            uc.reg_write(int_regs[arg.register], value & 0xFFFFFFFFFFFFFFFF)
        elif arg.register in FLOAT_REGISTERS:
            _write_xmm(uc, arg.register, value)
        else:
            # A spilled argument sits above the 32-byte shadow space.
            uc.mem_write(rsp + 8 + 32 + arg.stack_offset,
                         struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))


def _write_xmm(uc, name: str, low_bits: int) -> None:
    from unicorn.x86_const import (
        UC_X86_REG_XMM0,
        UC_X86_REG_XMM1,
        UC_X86_REG_XMM2,
        UC_X86_REG_XMM3,
    )
    regs = {"xmm0": UC_X86_REG_XMM0, "xmm1": UC_X86_REG_XMM1,
            "xmm2": UC_X86_REG_XMM2, "xmm3": UC_X86_REG_XMM3}
    uc.reg_write(regs[name], low_bits & ((1 << 128) - 1))


def _read_xmm0_low(uc) -> int:
    from unicorn.x86_const import UC_X86_REG_XMM0
    return uc.reg_read(UC_X86_REG_XMM0) & ((1 << 64) - 1)


class StubDeclined(Exception):
    """A stub cannot serve this call faithfully, so the input is inconclusive.

    Raised by a stub when honouring the call exactly is beyond what it
    implements - a realloc that would have to move a block the arena cannot,
    an argument it will not assume. Never a refutation: the caller turns it
    into IMPORT_WITHOUT_STUB, and the pair is inconclusive.
    """


def _fill_page(uc, page_base: int, seed: int) -> None:
    """Map one page and fill it with its address-and-seed pattern."""
    uc.mem_map(page_base, PAGE)
    uc.mem_write(page_base, _fill(page_base, seed))


def _ensure_mapped(uc, mapped: set, address: int, size: int, seed: int) -> None:
    """Map and fill every page a range [address, address+size) touches that is
    not mapped yet, so a read of it returns the same bytes a function's own
    read would. Used by a stub through the Machine."""
    first = address & ~(PAGE - 1)
    last = (address + size - 1) & ~(PAGE - 1)
    for page in range(first, last + PAGE, PAGE):
        if page not in mapped:
            _fill_page(uc, page, seed)
            mapped.add(page)


ARENA_BASE = 0x0000_3000_0000_0000
ARENA_LIMIT = 0x0000_3000_1000_0000      # 256 MiB of address space to hand out
ALIGN = 16                               # malloc returns 16-byte-aligned memory


class Arena:
    """A deterministic bump allocator shared across one function's run.

    Addresses are handed out in order from ARENA_BASE, 16-byte aligned, so
    the same sequence of allocations gives the same addresses every time and
    in both versions of a claim - which is why a pointer the allocator
    returned is never compared by value, only the memory it points at.

    Blocks are tracked here, out of emulated memory, so free can tell a live
    block from a bad pointer and realloc can find a block's size. free marks a
    block freed but does not reuse its address: reuse would let -O0 and -O3,
    which free at different points, diverge on later addresses. Address space
    is cheap; determinism is not.
    """

    def __init__(self):
        self.next = ARENA_BASE
        self.blocks = {}          # address -> {"size", "live"}

    def allocate(self, size):
        if size == 0:
            size = 1              # malloc(0) returns a unique, freeable pointer
        address = self.next
        stride = (size + ALIGN - 1) & ~(ALIGN - 1)
        if address + stride > ARENA_LIMIT:
            raise StubDeclined("arena exhausted")
        self.next += stride
        self.blocks[address] = {"size": size, "live": True}
        return address

    def free(self, address):
        if address == 0:
            return               # free(NULL) is a no-op
        block = self.blocks.get(address)
        if block is None or not block["live"]:
            raise StubDeclined("free of a bad or already-freed pointer")
        block["live"] = False

    def size_of(self, address):
        block = self.blocks.get(address)
        if block is None or not block["live"]:
            raise StubDeclined("realloc of a bad or already-freed pointer")
        return block["size"]



class Machine:
    """What a stub is handed: the emulator's memory and the call's registers.

    A stub reads its arguments from here by the Win64 convention (RCX, RDX,
    R8, R9 for the first four integers or pointers), works on emulated memory
    through read/write, and sets its return value. It never touches the
    emulator directly.

    read and write map memory on first touch with the same pattern the
    function's own accesses use, so a stub serving memcpy sees the buffer's
    address-derived bytes rather than inventing data or faulting on memory the
    function had not reached yet.
    """

    def __init__(self, uc, mapped, seed, arena):
        self._uc = uc
        self._mapped = mapped
        self._seed = seed
        self.arena = arena

    def arg(self, index: int) -> int:
        """The index-th integer or pointer argument (0-based), from its
        register. Stubs here take no more than four, so no stack reads."""
        from unicorn.x86_const import (
            UC_X86_REG_R8,
            UC_X86_REG_R9,
            UC_X86_REG_RCX,
            UC_X86_REG_RDX,
        )
        regs = (UC_X86_REG_RCX, UC_X86_REG_RDX, UC_X86_REG_R8, UC_X86_REG_R9)
        return self._uc.reg_read(regs[index]) & 0xFFFFFFFFFFFFFFFF

    def read(self, address: int, size: int) -> bytes:
        if size:
            _ensure_mapped(self._uc, self._mapped, address, size, self._seed)
        return bytes(self._uc.mem_read(address, size))

    def write(self, address: int, data: bytes) -> None:
        if data:
            _ensure_mapped(self._uc, self._mapped, address, len(data), self._seed)
        self._uc.mem_write(address, data)

    def set_return(self, value: int) -> None:
        from unicorn.x86_const import UC_X86_REG_RAX
        self._uc.reg_write(UC_X86_REG_RAX, value & 0xFFFFFFFFFFFFFFFF)


def _run_stub(uc, stub, mapped, seed, arena) -> None:
    """Run a stub in place of the call, then return to the caller.

    The call pushed a return address and jumped to the trap; the stub does
    the function's work on the Machine, and then RIP is set to the pushed
    return address and RSP stepped past it, exactly as a `ret` would, so the
    caller continues as if the real function had run and returned.
    """
    from unicorn.x86_const import UC_X86_REG_RIP, UC_X86_REG_RSP

    stub(Machine(uc, mapped, seed, arena))

    rsp = uc.reg_read(UC_X86_REG_RSP)
    return_address = int.from_bytes(uc.mem_read(rsp, 8), "little")
    uc.reg_write(UC_X86_REG_RSP, rsp + 8)
    uc.reg_write(UC_X86_REG_RIP, return_address)


def run(loader: Loader, address: int, placement: Placement, args,
        seed: int = 1, budget: int = 5_000_000, stub_resolver=None) -> Outcome:
    """Run one function and return what it produced.

    args are integers positioned by `placement`: an integer, a pointer's
    address, or a float's packed bits. Memory the function reaches that the
    harness did not map is mapped on first touch, filled by `_fill`; the
    stack takes the seed, argument-reachable memory does not (the caller maps
    argument buffers before the call so they are input, not touched-garbage).

    stub_resolver, if given, is called as resolver(name) when the function
    calls an import; it returns a stub or None. In this first version there
    are no stubs yet, so any import ends the run as IMPORT_WITHOUT_STUB.
    """
    from unicorn import (
        UC_ARCH_X86,
        UC_HOOK_CODE,
        UC_HOOK_MEM_FETCH_UNMAPPED,
        UC_HOOK_MEM_READ_UNMAPPED,
        UC_HOOK_MEM_WRITE,
        UC_HOOK_MEM_WRITE_UNMAPPED,
        UC_MODE_64,
        Uc,
        UcError,
    )
    from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RIP, UC_X86_REG_RSP

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(loader.base, loader.size)
    loader.write_sections(uc)

    uc.mem_map(STACK_TOP - STACK_SIZE, STACK_SIZE)
    uc.mem_map(SENTINEL & ~(PAGE - 1), PAGE)
    # The trap page holds no code; a call to an import lands here and is caught
    # by the code hook before it can execute. Mapping it keeps that a caught
    # import rather than a fetch fault.
    uc.mem_map(TRAP_BASE & ~(PAGE - 1), PAGE)

    # A 16-byte-aligned frame with the sentinel return address on top, and 32
    # bytes of shadow space below it as the convention requires.
    rsp = (STACK_TOP - 0x2000) & ~0xF
    rsp -= 8
    uc.mem_write(rsp, struct.pack("<Q", SENTINEL))
    uc.reg_write(UC_X86_REG_RSP, rsp)
    _load_arguments(uc, placement, args, rsp)

    state = {"status": None, "detail": ""}
    mapped: set = set()          # pages the harness filled on first touch
    written: set = set()
    arena = Arena()              # fresh per run: both versions allocate alike

    def on_unmapped(uc, access, addr, size, value, _):
        page = addr & ~(PAGE - 1)
        if addr < NULL_GUARD:
            # A null (or near-null) dereference. Left unmapped so it faults,
            # the way it would on a real machine, rather than being filled.
            state["status"] = Status.FAULT
            state["detail"] = f"null dereference at {addr:#x}"
            uc.emu_stop()
            return False
        if len(mapped) >= UNMAPPED_FILL_LIMIT:
            state["status"] = Status.TOO_MUCH_MEMORY
            uc.emu_stop()
            return False
        _fill_page(uc, page, seed)
        mapped.add(page)
        return True

    def on_write(uc, access, addr, size, value, _):
        if not (STACK_TOP - STACK_SIZE <= addr < STACK_TOP):
            written.add(addr & ~(PAGE - 1))

    def on_code(uc, addr, size, _):
        # A call to an import lands on its trap address (see Loader). Catch it
        # here and either run a stub in the emulator's place, or end the run
        # naming the import.
        name = loader.traps.get(addr)
        if name is None:
            return
        stub = stub_resolver(name) if stub_resolver else None
        if stub is None:
            state["status"] = Status.IMPORT_WITHOUT_STUB
            state["detail"] = name
            uc.emu_stop()
            return
        try:
            _run_stub(uc, stub, mapped, seed, arena)
        except StubDeclined as exc:
            state["status"] = Status.IMPORT_WITHOUT_STUB
            state["detail"] = f"{name}: {exc}"
            uc.emu_stop()

    uc.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED
                | UC_HOOK_MEM_FETCH_UNMAPPED, on_unmapped)
    uc.hook_add(UC_HOOK_MEM_WRITE, on_write)
    uc.hook_add(UC_HOOK_CODE, on_code)

    try:
        uc.emu_start(address, SENTINEL, count=budget)
    except UcError as exc:
        status = state["status"] or Status.FAULT
        return Outcome(status, detail=state["detail"] or str(exc))

    if state["status"] is not None:
        return Outcome(state["status"], detail=state["detail"])

    if uc.reg_read(UC_X86_REG_RIP) != SENTINEL:
        return Outcome(Status.BUDGET_EXHAUSTED)

    writes = {page: bytes(uc.mem_read(page, PAGE)) for page in sorted(written)}
    return Outcome(Status.COMPLETED,
                   ret_int=uc.reg_read(UC_X86_REG_RAX),
                   ret_float_bits=_read_xmm0_low(uc),
                   writes=writes)
