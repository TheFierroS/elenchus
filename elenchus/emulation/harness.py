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
from elenchus.emulation.branches import PredicateInstance, selectivity

PAGE = 0x1000
GOLDEN = 0x9E3779B97F4A7C15          # an odd constant, for a spread-out fill

# A private address range for the call frame and for memory the harness hands
# out, chosen far from any image base so it never overlaps a mapped binary.
STACK_TOP = 0x0000_7FF0_0000_0000
STACK_SIZE = 0x0010_0000             # 1 MiB
SENTINEL = 0x0000_0000_0000_1000     # return address; not a real code page
CHUNK = 16 * 4096                    # 64 KiB mapped per first touch: one
#                                      region instead of 16, which is where
#                                      Unicorn's cost actually lies.
#
# The budget is in pages, but what it has to bound is the number of *distinct
# touches* a lost walk makes, and a touch now costs CHUNK. Set so that the
# walk gets the same 512 touches it had before chunking: the first version of
# this kept the budget at 4096 pages, which with a 64-page chunk allowed only
# 64 touches, and cost 2.2 points of coverage (F27). Measured, 512 touches of
# 16 pages is 32 MiB and 83 ms, against 2 MiB and 390 ms before.
UNMAPPED_FILL_LIMIT = 128 * 16       # pages: 128 touches of CHUNK each.
#                                      Measured: 128 and 512 touches judge
#                                      exactly the same pairs, and 128 runs in
#                                      134 s where 512 takes 368 s. A walk
#                                      that has reached 128 separate places
#                                      has already lost its way (F28).
#                                      before calling it a lost walk. 2 MiB is
#                                      already far more than a real function
#                                      touches; a garbage pointer chased
#                                      through a data structure hits this fast,
#                                      and each page filled in Python is not
#                                      cheap, so a high limit was most of the
#                                      first V0 pass's time (F16).

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

# Where input buffers live. inputs.input_vectors hands a pointer argument an
# address in this region, and memory here is the function's *input*: it is
# filled from its address alone, identical in both runs and both versions, so
# a function that reads its argument gets the same bytes every time. Memory
# outside it that a run touches - a wild pointer, the stack - is
# uninitialised, and takes the seed-dependent fill so the two-seed check can
# tell a garbage-dependent result from a real one (F22).
INPUT_REGION_BASE = 0x0000_2000_0000_0000
INPUT_REGION_END = 0x0000_2000_1000_0000     # 256 MiB of buffer space

# A wall-clock ceiling per run, the last safety net. The instruction budget
# and the page limit catch the patterns we know; this catches whatever they
# do not, so no single function can ever hang the verifier. A run past it is
# TIMED_OUT - inconclusive, like the budget, never a refutation. Unicorn takes
# the timeout in microseconds.
DEFAULT_TIMEOUT_US = 2_000_000       # 2 seconds

# The largest single read or write a stub may perform. A stub takes a size or
# length from the function's arguments, and V0 feeds functions garbage, so
# that size can be nonsense - forty billion, a pointer reinterpreted as a
# count. A real memset of that size would exhaust memory; the stub must not.
# Anything past this makes the input inconclusive (StubDeclined), never a
# crash and never a wrong result. 16 MiB is far more than any real buffer a
# function touches in one call, and well past the page-map limit above.
MAX_TRANSFER = 16 * 1024 * 1024


class Status(Enum):
    """Why a run ended. Only COMPLETED yields a return value to compare."""
    COMPLETED = "completed"
    UNSUPPORTED_INSTRUCTION = "unsupported instruction"
    IMPORT_WITHOUT_STUB = "import without a stub"
    STUB_DECLINED = "stub declined the call"
    CHAIN_FAILED = "the initialiser did not complete"
    BUDGET_EXHAUSTED = "budget exhausted"
    TIMED_OUT = "timed out"
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
    # The conditional jumps this run took, when recording was asked for: what
    # the second tier ranks to choose a branch to force (docs/verifier.md).
    predicates: tuple = ()
    # Every write the run made, as (address, size, value), when asked for.
    # Off by default and only used to work out why two versions differ: the
    # pages alone say what ended up in memory, not who put it there.
    write_log: tuple = ()
    # Which bytes each store covered, as (address, size), always. A page says
    # what memory holds; this says which of it the function put there, and
    # which store put it - both of which the comparison needs (F31).
    write_ranges: tuple = ()
    # Stores that reached outside comparable memory, so only part of the
    # value they wrote is visible - and part of a value is not one.
    partial_writes: tuple = ()


# The content a page gets is a window into one bounded random block, indexed
# by the page's address modulo the block's size, after PEM's probabilistic
# memory model (docs/related-work.md). A memory model used this way has to
# hold two properties, and naming them is worth more than the code:
#
#   equivalence-preserving - two equivalent runs must see the same bytes at
#     the same addresses, or two builds of one function would diverge for a
#     reason that is ours, not theirs;
#   difference-revealing - two different addresses must see different bytes,
#     or two genuinely different functions could read the same thing
#     everywhere and look alike.
#
# A constant fill holds the first and fails the second. Generating bytes from
# the address held both but cost a Python loop per page, which was most of the
# time a lost walk spent (F16). A cached block holds both and costs a slice.
#
# GAMMA must not share a factor with the stride between input buffers, or two
# different pointer arguments would land on the same window and read identical
# bytes. 1021 is prime and 1021 pages is ~4 MiB, so buffers a megabyte apart
# (256 pages) stay distinct for the first 1021 of them.
GAMMA_PAGES = 1021
GAMMA = GAMMA_PAGES * PAGE

_FILL_BLOCKS: dict[int, bytes] = {}


def _fill_block(seed: int) -> bytes:
    """The block a fill draws from, one per seed, generated once.

    The words are arithmetic in the offset rather than random. PEM uses random
    bytes; measured on this corpus over 150 pairs, the arithmetic pattern
    judges 57.3% of pairs against 54.7% for random - 2.6 points, and the same
    zero false refutations either way (F28). A borrowed idea is worth having;
    a borrowed parameter has to be measured here.

    Why it would be better is a guess: a function that reads a length or an
    index from its own input gets a value related to where it read it, so
    walking one step forward gives a related value rather than an unrelated
    one, and the walk stays in the arena rather than scattering.
    """
    block = _FILL_BLOCKS.get(seed)
    if block is None:
        out = bytearray()
        for offset in range(0, GAMMA, 8):
            out += struct.pack("<Q", (offset * GOLDEN + seed) & 0xFFFFFFFFFFFFFFFF)
        block = bytes(out)
        _FILL_BLOCKS[seed] = block
    return block


def _fill(page_base: int, seed: int) -> bytes:
    """The bytes a freshly mapped page holds: a window into the seed's block
    chosen by the page's address (seed 0 means input memory)."""
    offset = page_base % GAMMA
    return _fill_block(seed)[offset:offset + PAGE]


class Loader:
    """A binary mapped at its image base, read once and reused across runs.

    Parsing a PE and copying its sections is the same for every call into it,
    so it is done once. The import table is read here too: each thunk's
    address is mapped to the name it imports, so a call through it can be
    recognised as leaving the binary.
    """

    def __init__(self, path: str | Path):
        pe = pefile.PE(str(path), fast_load=False)
        self.base = pe.OPTIONAL_HEADER.ImageBase
        self.size = (pe.OPTIONAL_HEADER.SizeOfImage + PAGE - 1) & ~(PAGE - 1)
        self.slots, self.traps = self._import_traps(pe)

        # Keep only what mapping needs - the header and each section's bytes at
        # its virtual address - as plain bytes, and drop the pefile object.
        # A PE parsed with fast_load=False holds the whole file and its parsed
        # structures, 100+ MB for a large binary; over V0's ~100 distinct
        # binaries, retaining every pefile object leaked gigabytes (F20). The
        # raw bytes are a few MB and are all write_sections ever reads.
        self._header = bytes(pe.header)
        self._sections = [
            (section.VirtualAddress, section.get_data())
            for section in pe.sections
            if section.get_data()
        ]
        pe.close()
        del pe

    def _import_traps(self, pe) -> tuple[dict, dict]:
        """Assign each named import a trap address and record both directions.

        slots maps the import-address-table slot to (dll, name); the slot is
        overwritten with the trap address at load time, so `call [slot]` lands
        on the trap. traps maps the trap address back to the import name, so
        the code hook can name what was called.
        """
        slots, traps = {}, {}
        if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            return slots, traps
        index = 0
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
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

        Uses the raw bytes kept at construction, not a pefile object. A
        section's raw data can be shorter than its virtual size (.bss holds
        zeros with no file bytes); the mapping is already zero, so copying the
        raw bytes over it leaves the rest zero. After the sections are in, the
        import slots are overwritten with trap addresses (see _import_traps).
        """
        uc.mem_write(self.base, self._header)
        for virtual_address, data in self._sections:
            uc.mem_write(self.base + virtual_address, data)
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


def _stack_garbage(seed: int) -> bytes:
    """The bytes a fresh stack holds, deterministic per seed.

    Generated with random.Random, which is reproducible across runs and
    platforms for a given seed, and fast enough to do per run: filling a
    megabyte word by word in Python would cost more than the emulation.
    """
    import random
    return random.Random(f"elenchus-stack-{seed}").randbytes(STACK_SIZE)


def _fill_seed_for(address: int, seed: int, input_variant: int = 0) -> int:
    """The fill seed a page at `address` takes.

    Input buffers are the function's input and must look the same in every
    run of one comparison, so they are filled from the address and the
    *input variant* - not the run seed. Running a whole comparison twice with
    two variants is how a function whose behaviour turns on the invented
    input data is recognised (F24): if it writes different places depending
    on bytes we made up, the data is meaningless to it and the claim cannot
    be judged on it.

    Everything else a run touches is uninitialised - the stack, a wild
    pointer's page - and takes the run's seed, which is what makes the
    two-seed check able to spot a result that depends on garbage.
    """
    if INPUT_REGION_BASE <= address < INPUT_REGION_END:
        return input_variant
    return seed


def _chunk_fill(base: int, size: int, seed: int) -> bytes:
    """The bytes for a whole chunk, page by page from the seed's block.

    Built by concatenating each page's window so a page holds exactly what it
    would have held had it been mapped alone: whether a page arrives on its
    own or inside a chunk must not change what the function reads.
    """
    return b"".join(_fill(base + offset, seed)
                    for offset in range(0, size, PAGE))


def _fill_page(uc, page_base: int, seed: int, input_variant: int = 0) -> int:
    """Map a chunk around a touched page and fill it. Returns pages mapped.

    Mapping one page per fault is what a lost walk does thousands of times,
    and Unicorn's cost grows faster than linearly in the number of separate
    regions: measured, 4096 single pages take 19 s where the same 16 MiB in
    64-page chunks takes 53 ms. A walk also tends to move forward, so the
    neighbours it maps are usually the ones it wants next.

    The chunk is clamped away from the null guard, and if it would overlap
    something already mapped - the image, the stack - the single page is
    mapped instead, which is rare and correct.
    """
    fill_seed = _fill_seed_for(page_base, seed, input_variant)
    start = page_base & ~(CHUNK - 1)
    if start < NULL_GUARD:
        start = page_base                      # keep the guard page unmapped
        size = PAGE
    else:
        size = CHUNK
    from unicorn import UcError
    try:
        uc.mem_map(start, size)
    except UcError:
        # Something in that range is already mapped; fall back to the page.
        start, size = page_base, PAGE
        uc.mem_map(start, size)
    uc.mem_write(start, _chunk_fill(start, size, fill_seed))
    return start, size // PAGE


def _ensure_mapped(uc, mapped: set, address: int, size: int, seed: int,
                   input_variant: int = 0) -> None:
    """Map and fill every page a range [address, address+size) touches that is
    not mapped yet, so a read of it returns the same bytes a function's own
    read would. Used by a stub through the Machine."""
    first = address & ~(PAGE - 1)
    last = (address + size - 1) & ~(PAGE - 1)
    for page in range(first, last + PAGE, PAGE):
        if page not in mapped:
            start, count = _fill_page(uc, page, seed, input_variant)
            mapped.update(start + i * PAGE for i in range(count))


ARENA_BASE = 0x0000_3000_0000_0000
ARENA_END = ARENA_BASE + 0x1000_0000        # 256 MiB, far more than it hands


def _is_handed_out(page: int) -> bool:
    """Whether a page is memory this harness gave the function.

    Only two regions are: the input buffers the generator pointed arguments
    at, and the arena an allocation stub hands out. Both sit at the same
    addresses in both versions, so what a function writes there can be
    matched between them.

    Everything else it may write to cannot. The stack is its own, and the
    image is its globals, which the two builds place differently (F15). And a
    function walking a pointer we invented writes wherever that pointer led -
    an address that differs between the versions for a reason that is ours,
    so a page there belongs to neither and comparing it refuted true claims
    (F30). A write just outside a buffer is the same thing: the function has
    left the space we gave it, on data we made up.
    """
    return (INPUT_REGION_BASE <= page < INPUT_REGION_END
            or ARENA_BASE <= page < ARENA_END)
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
        # Per-name blocks for the C runtime's own state - errno, the stdio
        # table - so a function that reads one twice in a run sees the same
        # location, and one that stores into it reads back what it stored.
        self.runtime_state = {}

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

    def __init__(self, uc, mapped, seed, arena, input_variant=0, record=None,
                 log=None):
        self._uc = uc
        self._mapped = mapped
        self._seed = seed
        self._input_variant = input_variant
        self._record = record
        self._log = log
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
        if size > MAX_TRANSFER:
            raise StubDeclined(f"read of {size} bytes, past the sanity bound")
        if size:
            _ensure_mapped(self._uc, self._mapped, address, size, self._seed,
                           self._input_variant)
        return bytes(self._uc.mem_read(address, size))

    def write(self, address: int, data: bytes) -> None:
        if len(data) > MAX_TRANSFER:
            raise StubDeclined(f"write of {len(data)} bytes, past the sanity bound")
        if data:
            _ensure_mapped(self._uc, self._mapped, address, len(data),
                           self._seed, self._input_variant)
        self._uc.mem_write(address, data)
        # A stub writes through the emulator's API, which does not fire the
        # write hook, so its effect would otherwise be invisible: a -O0 build
        # calling memcpy to fill the caller's buffer looked as if it wrote
        # nothing, while the -O3 build that inlined the copy looked as if it
        # wrote, and the pair was refuted (F25). Recording here makes a stub's
        # writes count exactly like the function's own.
        if self._record is not None and data:
            self._record(address, len(data))
        if self._log is not None and data:
            self._log.append((address, len(data), None))

    def set_return(self, value: int) -> None:
        from unicorn.x86_const import UC_X86_REG_RAX
        self._uc.reg_write(UC_X86_REG_RAX, value & 0xFFFFFFFFFFFFFFFF)

    def bound(self, size: int) -> int:
        """Return size if it is within the sanity bound, else decline.

        A stub calls this before building a buffer of `size` bytes, so a
        garbage size from the function's arguments declines the input rather
        than materialising forty billion bytes and crashing. read and write
        check the bound too, but a stub that constructs the buffer itself
        (memset building c*n) must check before constructing it.
        """
        if size > MAX_TRANSFER:
            raise StubDeclined(f"size {size} past the sanity bound")
        return size


def _run_stub(uc, stub, mapped, seed, arena, input_variant=0, record=None,
              log=None) -> None:
    """Run a stub in place of the call, then return to the caller.

    The call pushed a return address and jumped to the trap; the stub does
    the function's work on the Machine, and then RIP is set to the pushed
    return address and RSP stepped past it, exactly as a `ret` would, so the
    caller continues as if the real function had run and returned.
    """
    from unicorn.x86_const import UC_X86_REG_RIP, UC_X86_REG_RSP

    stub(Machine(uc, mapped, seed, arena, input_variant, record, log))

    rsp = uc.reg_read(UC_X86_REG_RSP)
    return_address = int.from_bytes(uc.mem_read(rsp, 8), "little")
    uc.reg_write(UC_X86_REG_RSP, rsp + 8)
    uc.reg_write(UC_X86_REG_RIP, return_address)


def run(loader: Loader, address: int, placement: Placement, args,
        seed: int = 1, budget: int = 5_000_000, stub_resolver=None,
        timeout_us: int = DEFAULT_TIMEOUT_US,
        input_variant: int = 0, prepare=None,
        record_predicates: bool = False, force=None,
        record_writes: bool = False) -> Outcome:
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
    from unicorn import UC_ARCH_X86, UC_MODE_64, Uc

    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    # The run's mutable state lives here rather than inside the body, so what
    # a recorded run saw can be attached however the body exited - a run that
    # faulted still took branches on the way, and those are worth keeping.
    state = {"status": None, "detail": "", "count": 0,
             "predicates": [],     # PredicateInstance, in the order seen
             "pending": None,      # a branch whose outcome is not known yet
             "selectivity": None,  # the last compare's distance
             "resume_at": None,    # where a forced branch is being sent
             "write_log": [],      # (address, size, value) when asked for
             "write_ranges": [],   # (address, size) of every recorded store
             "partial_writes": []}  # stores reaching outside comparable memory
    try:
        outcome = _run_in(uc, loader, address, placement, args, seed, budget,
                          stub_resolver, timeout_us, input_variant, prepare,
                          state, record_predicates, force, record_writes)
        if record_predicates:
            outcome.predicates = tuple(state["predicates"])
        if record_writes:
            outcome.write_log = tuple(state["write_log"])
        outcome.write_ranges = tuple(state["write_ranges"])
        outcome.partial_writes = tuple(state["partial_writes"])
        return outcome
    finally:
        # Unicorn holds C-side memory (every mapped page, the hooks) that is
        # not freed when the Python object is collected, and the hook closures
        # reference uc, a cycle the collector is slow to break. Over a long
        # run - V0's thousands of pairs - that leaks gigabytes and the process
        # is killed (F19). Releasing the handle here frees it at once, and the
        # emulator is one per run anyway, so nothing outlives the finally.
        try:
            uc.release_handle()
        except Exception:                        # noqa: BLE001
            pass


def _decoder_over(uc):
    """A Decoder reading code out of an emulator's memory."""
    from elenchus.emulation.branches import Decoder
    return Decoder(lambda address, size: bytes(uc.mem_read(address, size)))


def _register_reader(uc):
    """Read a register by capstone's name for it, or None if it has none."""
    from unicorn import x86_const

    def read(name):
        constant = getattr(x86_const, f"UC_X86_REG_{name.upper()}", None)
        return None if constant is None else uc.reg_read(constant)
    return read


def _safe_read(uc, address: int, width: int, seed: int = 0,
               input_variant: int = 0):
    """The bytes at `address`, whether or not they have been mapped yet.

    Selectivity is read in the code hook, before the instruction executes -
    which is while its operands still hold the values compared, but also
    before a first touch has mapped anything. A compare against memory the
    function has not read yet (`cmp [rcx], imm` as its first instruction, the
    -O3 shape of a context check) would otherwise have no selectivity at all.

    What an unmapped page will hold is already decided by its address, so it
    is computed rather than waited for. Reading does not map: the instruction
    itself will, a moment later, through the ordinary path.
    """
    try:
        return bytes(uc.mem_read(address, width))
    except Exception:                           # noqa: BLE001 - not yet mapped
        out = bytearray()
        offset = address
        while len(out) < width:
            page = offset & ~(PAGE - 1)
            within = offset - page
            chunk = _fill(page, _fill_seed_for(page, seed, input_variant))
            take = min(width - len(out), PAGE - within)
            out += chunk[within:within + take]
            offset += take
        return bytes(out)


def _watch_branch(uc, addr, size, state, decoder, force,
                  seed=0, input_variant=0) -> None:
    """Record the branch at `addr`, and force it if this instance was chosen.

    A conditional jump's outcome is not known until the next instruction
    runs, so the jump is held as pending and resolved when the next address
    arrives: it went to the target, or it fell through.

    Forcing is {instruction count: destination}, the destination taken from a
    recording run, so the caller names exactly which instance of a jump in a
    loop it means. Unicorn cannot be redirected from inside a hook reliably,
    so the run is stopped here and resumed at the destination by execute().
    """
    pending = state["pending"]
    if pending is not None:
        state["pending"] = None
        count, branch, distance = pending
        if addr in (branch.target, branch.fall_through):
            state["predicates"].append(
                PredicateInstance(count, branch, addr == branch.target,
                                  distance))

    if force and state["count"] in force:
        state["resume_at"] = force[state["count"]]
        return

    branch = decoder.branch_at(addr, size)
    if branch is not None:
        state["pending"] = (state["count"], branch, state["selectivity"])
        state["selectivity"] = None
        return

    compare = decoder.compare_at(addr, size)
    if compare is not None:
        # How close this comparison came to going the other way, read before
        # the instruction executes - which is while its operands still hold
        # the values being compared.
        state["selectivity"] = selectivity(
            compare, _register_reader(uc),
            lambda address, width: _safe_read(uc, address, width, seed,
                                              input_variant))


def _run_in(uc, loader, address, placement, args, seed, budget, stub_resolver,
            timeout_us, input_variant=0, prepare=None, state=None,
            record_predicates=False, force=None, record_writes=False):
    """The body of one run, on an already-created emulator; run() owns its
    lifetime and releases it. Split out so every return path is covered by
    run()'s finally without repeating the cleanup."""
    from unicorn import (
        UC_HOOK_CODE,
        UC_HOOK_MEM_FETCH_UNMAPPED,
        UC_HOOK_MEM_READ_UNMAPPED,
        UC_HOOK_MEM_WRITE,
        UC_HOOK_MEM_WRITE_UNMAPPED,
        UcError,
    )
    from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RIP, UC_X86_REG_RSP

    uc.mem_map(loader.base, loader.size)
    loader.write_sections(uc)

    uc.mem_map(STACK_TOP - STACK_SIZE, STACK_SIZE)
    # The stack is uninitialised memory: a function that reads a local it
    # never wrote reads whatever was there. Unicorn zero-fills a fresh
    # mapping, which is the same in both seeds, so the two-seed check could
    # not see such a read at all (F22). Filling it from the seed makes a
    # garbage-dependent result differ between the seeds, which is exactly
    # what the check excludes.
    uc.mem_write(STACK_TOP - STACK_SIZE, _stack_garbage(seed))
    uc.mem_map(SENTINEL & ~(PAGE - 1), PAGE)
    # The trap page holds no code; a call to an import lands here and is caught
    # by the code hook before it can execute. Mapping it keeps that a caught
    # import rather than a fetch fault.
    uc.mem_map(TRAP_BASE & ~(PAGE - 1), PAGE)

    def frame(call_placement, call_args):
        """Lay a fresh call frame: a 16-byte-aligned stack with the sentinel
        return address on top and 32 bytes of shadow space, arguments placed.
        Used once per call, so a chained initialiser and the function under
        test each start from a clean frame."""
        rsp = (STACK_TOP - 0x2000) & ~0xF
        rsp -= 8
        uc.mem_write(rsp, struct.pack("<Q", SENTINEL))
        uc.reg_write(UC_X86_REG_RSP, rsp)
        _load_arguments(uc, call_placement, call_args, rsp)

    # Branch watching - recording predicate instances, and forcing one - is
    # off unless asked for, so an ordinary run never touches capstone.
    watching = record_predicates or bool(force)
    decoder = _decoder_over(uc) if watching else None
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
        start, count = _fill_page(uc, page, seed, input_variant)
        mapped.update(start + i * PAGE for i in range(count))
        return True

    loader.base + loader.size

    def record_write(addr, size):
        """Record the pages a write touches, if it is an observable effect.

        The stack is the function's own; the binary's own image is its global
        and static data, which -O0 and -O3 place at different addresses with
        different contents, so a write there cannot be matched between the
        twins (docs/verifier.md, risk 9). Only writes to memory the harness
        handed out - argument buffers and the arena, at the same address in
        both versions - are the function's observable effect.

        Both the emulator's write hook and a stub's Machine.write go through
        here, so a stub's effect counts exactly like the function's own (F25).
        """
        first = addr & ~(PAGE - 1)
        last = (addr + max(size, 1) - 1) & ~(PAGE - 1)
        pages = list(range(first, last + PAGE, PAGE))
        inside = [page for page in pages if _is_handed_out(page)]
        for page in inside:
            written.add(page)
        if not inside:
            return
        state["write_ranges"].append((addr, max(size, 1)))
        if len(inside) != len(pages):
            # The store reaches into memory that is not ours, so part of the
            # value it wrote is somewhere we never look. Half of one value is
            # not a value: what we *can* see of it cannot be judged either,
            # and is recorded as such rather than compared (F31). dtoa does
            # exactly this, writing eight bytes four below a buffer.
            state["partial_writes"].append((addr, max(size, 1)))

    def on_write(uc, access, addr, size, value, _):
        if record_writes:
            state["write_log"].append((addr, size, value))
        record_write(addr, size)

    def on_code(uc, addr, size, _):
        # Runs once per instruction (the hook is already here for imports), so
        # it also counts them: how many a function took is what V0 reads to
        # set the real budget.
        state["count"] += 1
        if watching:
            _watch_branch(uc, addr, size, state, decoder, force, seed,
                          input_variant)
            if state["resume_at"] is not None:
                uc.emu_stop()
                return
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
            _run_stub(uc, stub, mapped, seed, arena, input_variant,
                      record_write,
                      state["write_log"] if record_writes else None)
        except StubDeclined as exc:
            # The stub exists but cannot serve this call faithfully - a free
            # of a pointer the arena never handed out, a size past the sanity
            # bound. Distinct from having no stub at all: one says write a
            # stub, the other says this call was beyond the one we have, and
            # mixing them made the histogram ask for stubs already written.
            state["status"] = Status.STUB_DECLINED
            state["detail"] = f"{name}: {exc}"
            uc.emu_stop()

    uc.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED
                | UC_HOOK_MEM_FETCH_UNMAPPED, on_unmapped)
    uc.hook_add(UC_HOOK_MEM_WRITE, on_write)
    uc.hook_add(UC_HOOK_CODE, on_code)

    def execute(call_address, call_placement, call_args):
        """Run one call to its return, and say how it ended.

        Returns None on a clean return, or a Status. Used for the chained
        initialiser and for the function under test, so both are subject to
        the same budget, timeout and hooks.
        """
        frame(call_placement, call_args)
        entry = call_address
        while True:
            # The budget belongs to the call, not to a segment of it: a forced
            # branch splits one call into several emu_start calls, and each
            # must not hand the function a fresh allowance.
            remaining = budget - state["count"]
            if remaining <= 0:
                return Status.BUDGET_EXHAUSTED
            try:
                uc.emu_start(entry, SENTINEL, timeout=timeout_us,
                             count=remaining)
            except UcError:
                return state["status"] or Status.FAULT
            if state["status"] is not None:
                return state["status"]
            if state["resume_at"] is not None:
                # A branch was forced: carry on from where it was sent. The
                # jump itself never executed, which is the point.
                entry = state["resume_at"]
                state["resume_at"] = None
                state["pending"] = None
                continue
            if uc.reg_read(UC_X86_REG_RIP) != SENTINEL:
                return (Status.BUDGET_EXHAUSTED if state["count"] >= budget
                        else Status.TIMED_OUT)
            return None

    call_args = args
    if prepare is not None:
        # The initialiser runs first, in this same emulator, so the context
        # the function under test receives is one the library built rather
        # than bytes we invented (docs/verifier.md, constructor chains). Each
        # version runs its own initialiser, so a context that holds a pointer
        # into its own binary stays valid.
        #
        # Two shapes: an in-place initialiser fills the buffer the function
        # will be given, and the arguments stand; a constructor returns the
        # context it built, and that return value becomes the function's first
        # argument, because the buffer we would have passed is not the context
        # it made.
        prepare_address, prepare_placement, prepare_args = prepare[:3]
        context_from_return = len(prepare) > 3 and prepare[3]
        failure = execute(prepare_address, prepare_placement, prepare_args)
        if failure is not None:
            # A half-built context is not a context: abandon the chain rather
            # than test the function on it.
            return Outcome(Status.CHAIN_FAILED,
                           detail=f"initialiser: {failure.value}"
                                  f"{': ' + state['detail'] if state['detail'] else ''}",
                           instructions=state["count"])
        if context_from_return:
            context = uc.reg_read(UC_X86_REG_RAX)
            if context == 0:
                # The constructor failed - out of memory, bad arguments - and
                # a null context is not one to test on.
                return Outcome(Status.CHAIN_FAILED,
                               detail="the constructor returned null",
                               instructions=state["count"])
            call_args = [context] + list(args[1:])
        # What the initialiser wrote is setup, not the function's effect.
        written.clear()
        state["status"] = None
        state["detail"] = ""

    failure = execute(address, placement, call_args)
    if failure is not None:
        return Outcome(failure, detail=state["detail"],
                       instructions=state["count"])

    writes = {page: bytes(uc.mem_read(page, PAGE)) for page in sorted(written)}
    return Outcome(Status.COMPLETED,
                   ret_int=uc.reg_read(UC_X86_REG_RAX),
                   ret_float_bits=_read_xmm0_low(uc),
                   instructions=state["count"],
                   writes=writes)
