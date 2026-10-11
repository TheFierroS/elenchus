"""The observation census: what else the verifier could compare, measured first.

docs/verifier.md, "What else could be observed: the census". The verifier
compares two things - the masked return and the bytes both versions wrote -
and this module measures, without changing any verdict, how much a third would
be worth and what it would cost. Nothing here decides anything. It holds no
emulator state and imports no emulator, so it is tested on its own, like abi,
inputs, branches, forced and chain.

Two things live here. The first reads one thing exactly: where a wild address
came from. The second is the counting - what the census keeps of each run, and
when a difference between two versions counts at all.

A function run on invented data often dereferences a pointer it read out of
that data, and lands somewhere the harness never handed out. The fill that put
those bytes there is not random. A word of it is

    offset * GOLDEN + c    (mod 2**64)

where `offset` is the word's byte position in the fill block and `c` is the
input variant inside the argument buffers, the run's seed everywhere else
(harness._fill_block, harness._fill_seed_for). GOLDEN is odd, so it has an
inverse modulo 2**64, and a value can be tested exactly for having been one of
those words: multiply back, and see whether a word offset inside the block
comes out.

That is what Tinbergen gets from a backward taint analysis - which input bytes
were a pointer (docs/related-work.md, 5.1) - read here from the arithmetic of
our own fill instead. The census's eighth observation, fault anatomy, is built
on it, and it decides whether repairing inputs is worth building.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from elenchus.emulation.harness import (
    ARENA_BASE,
    ARENA_END,
    GAMMA,
    GOLDEN,
    INPUT_REGION_BASE,
    INPUT_REGION_END,
    NULL_GUARD,
    STACK_SIZE,
    STACK_TOP,
    Status,
    _stack_garbage,
)
from elenchus.emulation.inputs import BUFFER_STRIDE
from elenchus.emulation.stubs import LOCKS, STREAM_OUTPUT

MASK64 = (1 << 64) - 1
GOLDEN_INVERSE = pow(GOLDEN, -1, 1 << 64)
WORD = 8

# How far past the pointer it read a function may land. A field access adds a
# small constant before dereferencing - p->next at +8, a member a few hundred
# bytes into a context - so the address that faults is the pointer plus that.
# An index scaled by garbage lands much further and is left as "other", which
# makes every count built on this a lower bound, never an overcount.
REACH = 1024

# Where a run's first wild access came from.
NULL = "null"
INPUT_FILL = "data pointer from the input fill"
UNINITIALISED_FILL = "data pointer from uninitialised fill"
STACK_GARBAGE = "data pointer from an uninitialised local"
BEYOND = "past a region of ours, by a garbage offset"
CODE_FROM_FILL = "code pointer from invented data"
OTHER = "other"
ORIGINS = (NULL, INPUT_FILL, UNINITIALISED_FILL, STACK_GARBAGE, BEYOND,
           CODE_FROM_FILL, OTHER)

# The regions whose base is real: the buffers and the arena the harness hands
# out, and the stack. An address up to a tebibyte outside one is read as that
# base plus a garbage length or index - a 32-bit index scaled by 8 lands well
# inside it - while a fill word lands that close by chance under once in a
# million.
OURS = ((INPUT_REGION_BASE, INPUT_REGION_END), (ARENA_BASE, ARENA_END),
        (STACK_TOP - STACK_SIZE, STACK_TOP))
BEYOND_SPAN = 1 << 40


def fill_origin(address: int, constants, reach: int = REACH):
    """How `address` was reached from a word of our fill, or None.

    Returns (constant, offset, displacement): the fill constant that made the
    word, the word's byte offset in the fill block, and how far past the word's
    value the address lies. Nearer displacements are tried first.

    A random 64-bit value passes the test by accident with probability about
    (GAMMA / 8) / 2**64 for each displacement and constant tried - under one in
    a billion over the whole reach - so a match is not a coincidence.

    A pointer read from an unaligned position spans two words and is not of
    this form, so it is not recognised; nor is one read from the stack, whose
    garbage comes from random.Random rather than the arithmetic fill. Both
    leave the count a lower bound.
    """
    address &= MASK64
    for distance in range(reach + 1):
        for displacement in (distance, -distance) if distance else (0,):
            pointer = (address - displacement) & MASK64
            for constant in constants:
                offset = ((pointer - constant) * GOLDEN_INVERSE) & MASK64
                if offset < GAMMA and offset % WORD == 0:
                    return constant, offset, displacement
    return None


_STACK_WORDS: dict[int, frozenset] = {}


def _stack_words(seed: int) -> frozenset:
    """Every aligned word of the stack garbage a run under `seed` starts with.

    The stack is filled from random.Random, not the arithmetic fill, so it
    does not invert; but it is deterministic and a megabyte, so its 131,072
    words can simply be listed. A local the function never wrote is one.
    """
    words = _STACK_WORDS.get(seed)
    if words is None:
        block = _stack_garbage(seed)
        words = frozenset(int.from_bytes(block[i:i + 8], "little")
                          for i in range(0, len(block), 8))
        _STACK_WORDS[seed] = words
    return words


def _from_stack(address: int, seed: int, reach: int = REACH) -> bool:
    """Whether `address` is a word of the stack's garbage plus a field offset."""
    words = _stack_words(seed)
    return any(((address - d) & MASK64) in words
               for d in range(-reach, reach + 1))


def _beyond(address: int) -> bool:
    """Whether `address` lies just outside a region of ours: a real base plus
    or minus a garbage offset."""
    return any(low - BEYOND_SPAN <= address < high + BEYOND_SPAN
               for low, high in OURS)


def classify_wild(kind: str, address: int, seed: int, variant: int) -> str:
    """Where a run's first wild access came from, as one of ORIGINS.

    `kind` is how the harness touched it: "read", "write", "fetch", or "null"
    for a dereference of the guard page. `seed` and `variant` are the run's
    own, the only constants its memory could have been filled from.

    The sources are tried from the most specific: a pointer read out of the
    arithmetic fill, then out of the stack's garbage - an uninitialised local
    - then a real base with a garbage offset added. A fetch from invented
    data is code, a callback or an operations table; handing it a buffer
    cannot repair that, so it is kept apart from data pointers, which a
    buffer can.
    """
    if kind == "null" or address < NULL_GUARD:
        return NULL
    origin = fill_origin(address, (variant, seed))
    if origin is not None:
        if kind == "fetch":
            return CODE_FROM_FILL
        return INPUT_FILL if origin[0] == variant else UNINITIALISED_FILL
    if _from_stack(address, seed):
        return CODE_FROM_FILL if kind == "fetch" else STACK_GARBAGE
    if _beyond(address):
        return BEYOND
    return OTHER




# ---------------------------------------------------------------- the runs

COMPLETED = Status.COMPLETED.value

# The names the roadmap argued a compiler can neither invent nor remove - locks
# and stream output - reported as their own projection of the calls, to see
# whether the argument holds (docs/verifier.md, the census, observation 1).
ALLOW_LIST = frozenset(LOCKS) | frozenset(STREAM_OUTPUT)


@dataclass(frozen=True)
class RunTrace:
    """What the census keeps of one run: small, picklable, never compared.

    A comparison makes eight first-tier runs an input and the verdict needs
    their written pages; the census needs only this, so it can cross a
    process boundary with the pair's result.
    """
    status: str
    instructions: int
    calls: tuple
    calls_overflow: bool
    first_wild: tuple | None
    allocations: tuple
    buffers: frozenset          # indices of the argument buffers written
    arena_bytes: int            # bytes stored into the arena
    null_return: bool | None    # RAX was zero; None if the run stopped
    read_invented: bool
    seed: int
    variant: int


def trace(outcome, seed: int, variant: int) -> RunTrace:
    """The census's record of one run, from the harness's Outcome.

    A buffer's index is its distance from the region's base in strides,
    the layout inputs.input_vectors hands out - the same in both builds by
    construction, which is what makes it comparable.
    """
    buffers, arena_bytes = set(), 0
    for address, size in outcome.write_ranges:
        if INPUT_REGION_BASE <= address < INPUT_REGION_END:
            buffers.add((address - INPUT_REGION_BASE) // BUFFER_STRIDE)
        elif ARENA_BASE <= address < ARENA_END:
            arena_bytes += size
    completed = outcome.status is Status.COMPLETED
    return RunTrace(
        status=outcome.status.value,
        instructions=outcome.instructions,
        calls=tuple(outcome.calls),
        calls_overflow=outcome.calls_overflow,
        first_wild=outcome.first_wild,
        allocations=tuple(outcome.allocations),
        buffers=frozenset(buffers),
        arena_bytes=arena_bytes,
        null_return=(outcome.ret_int == 0) if completed else None,
        read_invented=outcome.read_invented,
        seed=seed,
        variant=variant,
    )


# ---------------------------------------------------------------- counting


def _multiset(items) -> tuple:
    return tuple(sorted(Counter(items).items()))


# Each observation as a function of one run's trace; None where the run
# cannot show it, as a call list that overflowed cannot.
OBSERVATIONS = {
    "calls": lambda t: None if t.calls_overflow else _multiset(t.calls),
    "calls on the allow-list": lambda t: (
        None if t.calls_overflow
        else _multiset(c for c in t.calls if c in ALLOW_LIST)),
    "buffers written": lambda t: t.buffers,
    "arena bytes written": lambda t: t.arena_bytes,
    "allocations": lambda t: _multiset(t.allocations),
}


def _settled(variant, extract):
    """(Q's value, K's value) under one input fill, or None if it cannot say.

    The gates are the verifier's own, so the census measures rules that could
    actually be written: all four runs completed, none read memory only we
    could have made (F32), and each version's two seeds agree (F22, F33).
    """
    if any(t.status != COMPLETED or t.read_invented for t in variant):
        return None
    q_a, q_b, k_a, k_b = (extract(t) for t in variant)
    if q_a is None or k_a is None or q_a != q_b or k_a != k_b:
        return None
    return q_a, k_a


def _judge_input(variants, extract):
    """Whether one input shows a difference: None if it cannot say, and True
    only when Q and K differ under both input fills (F24)."""
    settled = [_settled(variant, extract) for variant in variants]
    if any(s is None for s in settled):
        return None
    return all(q != k for q, k in settled)


def _status_split(variants):
    """Whether Q and K ended differently - one completed, one faulted - the
    same way under both seeds and both fills. A run that read invented memory
    says nothing: how it ended may be ours."""
    splits = []
    for q_a, q_b, k_a, k_b in variants:
        if any(t.read_invented for t in (q_a, q_b, k_a, k_b)):
            return None
        if q_a.status != q_b.status or k_a.status != k_b.status:
            return None
        splits.append(q_a.status != k_a.status)
    return all(splits)


def _first_wild(traces):
    """The run whose wild touch says most about how the pair ended, or None.

    A run that stopped - faulted, lost its walk - is preferred over one that
    completed, so a faulting pair is read by the run that faulted rather than
    by whichever run happened to touch wild memory first.
    """
    touched = [run_trace for variants in traces for variant in variants
               for run_trace in variant if run_trace.first_wild is not None]
    for run_trace in touched:
        if run_trace.status != COMPLETED:
            return run_trace
    return touched[0] if touched else None


@dataclass
class PairCensus:
    """What the census says about one pair, ready to be added up."""
    verdict: str
    forced_only: bool
    comparable: dict            # observation -> could it be compared at all
    differs: dict               # observation -> did it differ on some input
    call_names: tuple           # names whose counts differed, first such input
    instructions: tuple         # every first-tier run's count
    survival_mass: int | None   # compared bytes of the strongest survival
    survival_saw_return: bool
    wild_origin: str | None     # one of ORIGINS
    wild_status: str | None     # how the run that touched it ended
    differing: tuple = ()       # every observation that differed, for unions
    label: str = ""             # which pair, so one can be read back


def summarise(verdict: str, forced_only: bool, inputs, traces,
              ret_kind: str) -> PairCensus:
    """One pair's census, from its per-input results and run traces.

    `traces` holds, per input, one tuple per input fill of four traces in the
    order (Q seed A, Q seed B, K seed A, K seed B). `inputs` are the per-input
    results, read only for how much a survival compared. Nothing here can
    change the verdict it is handed.
    """
    observations = dict(OBSERVATIONS)
    if ret_kind == "pointer":
        # A returned pointer is never compared (F18); whether it was null is.
        observations["null return"] = lambda t: t.null_return

    comparable, differs, call_names = {}, {}, ()
    for name, extract in observations.items():
        judged = [_judge_input(variants, extract) for variants in traces]
        comparable[name] = any(j is not None for j in judged)
        differs[name] = any(j for j in judged)
        if name == "calls" and differs[name]:
            variants = traces[judged.index(True)]
            q, k = (dict(side) for side in _settled(variants[0], extract))
            call_names = tuple(sorted(n for n in set(q) | set(k)
                                      if q.get(n) != k.get(n)))

    splits = [_status_split(variants) for variants in traces]
    comparable["status split"] = any(s is not None for s in splits)
    differs["status split"] = any(s for s in splits)

    mass, saw_return = None, False
    if verdict == "survived" and not forced_only:
        for result in inputs:
            if result.verdict.value == "survived":
                mass = max(mass or 0, result.compared_bytes)
                saw_return = saw_return or result.saw_return

    wild = _first_wild(traces)
    origin = (classify_wild(*wild.first_wild, wild.seed, wild.variant)
              if wild is not None else None)

    return PairCensus(
        verdict=verdict,
        forced_only=forced_only,
        comparable=comparable,
        differs=differs,
        call_names=call_names,
        instructions=tuple(t.instructions for variants in traces
                           for variant in variants for t in variant),
        survival_mass=mass,
        survival_saw_return=saw_return,
        wild_origin=origin,
        wild_status=wild.status if wild is not None else None,
        differing=tuple(sorted(n for n, d in differs.items() if d)),
    )




# ---------------------------------------------------------------- adding up


def aggregate(entries, budget: int) -> dict:
    """Add up the census over many pairs, as JSON can hold it.

    `entries` are (bucket, PairCensus); V0 knows a pair's bucket, R0 passes
    None and its verdict stands in. The two halves are read together by
    experiments/census_read.py, which applies the protocol's thresholds.
    """
    observations, names, anatomy, outcomes = {}, Counter(), Counter(), Counter()
    runs, survivals, differing = [], [], []
    for bucket, pair in entries:
        ended = bucket or pair.verdict
        outcomes[ended] += 1
        for name, could in pair.comparable.items():
            row = observations.setdefault(
                name, {"comparable": 0, "differs": 0, "differs_unrefuted": 0})
            row["comparable"] += int(could)
            if pair.differs.get(name):
                row["differs"] += 1
                if pair.verdict != "refuted":
                    row["differs_unrefuted"] += 1
        names.update(pair.call_names)
        runs.extend(pair.instructions)
        if pair.survival_mass is not None:
            survivals.append([pair.survival_mass, pair.survival_saw_return])
        if pair.wild_origin is not None:
            anatomy[f"{ended} | {pair.wild_origin}"] += 1
        if pair.differing:
            differing.append([pair.verdict, list(pair.differing), pair.label])
    runs.sort()

    def percentile(p):
        return runs[min(len(runs) - 1, int(len(runs) * p))] if runs else None

    return {
        "pairs": len(entries),
        "outcomes": dict(outcomes),
        "observations": observations,
        "call_names": dict(names.most_common()),
        "runs": len(runs),
        "instructions_per_run": {
            "median": percentile(0.50), "p90": percentile(0.90),
            "p99": percentile(0.99), "max": runs[-1] if runs else None,
            "at_budget": sum(1 for r in runs if r >= budget),
        },
        "survivals": survivals,
        "anatomy": dict(anatomy),
        "differing": differing,
    }

def census_lines(summary: dict, side: str) -> list:
    """The census as V0 or R0 prints it beside its own report.

    V0 counts how often an observation differed on true pairs - its risk.
    R0 counts how often it differed on wrong pairs not already refuted - its
    ceiling. Neither half decides alone.
    """
    pairs = summary["pairs"] or 1
    lines = [f"\n=== observation census ({side}, stubbed layer, "
             f"{summary['pairs']} pairs)"]
    what = "not refuted" if side == "R0" else "differed"
    lines.append(f"  {'observation':26} {'comparable':>10} {what:>12}")
    for name, row in summary["observations"].items():
        count = row["differs_unrefuted"] if side == "R0" else row["differs"]
        lines.append(f"  {name:26} {row['comparable']:10} {count:12}"
                     f"  {100 * count / pairs:5.1f}%")
    per_run = summary["instructions_per_run"]
    lines.append(f"  instructions per run ({summary['runs']} runs): "
                 f"median {per_run['median']}, p90 {per_run['p90']}, "
                 f"p99 {per_run['p99']}, max {per_run['max']}, "
                 f"{per_run['at_budget']} at the budget")
    masses = sorted(mass for mass, _ in summary["survivals"])
    if masses:
        with_return = sum(1 for _, saw in summary["survivals"] if saw)
        lines.append(f"  survivals: {len(masses)}, compared bytes "
                     f"p10 {masses[len(masses) // 10]}, "
                     f"median {masses[len(masses) // 2]}; "
                     f"{with_return} with a compared return")
    if summary["anatomy"]:
        lines.append("  first wild touch (how the pair ended | where from):")
        for key, n in sorted(summary["anatomy"].items(),
                             key=lambda item: -item[1])[:12]:
            lines.append(f"      {key:62} {n:5}")
    if side == "V0" and summary["call_names"]:
        lines.append("  call names that differed on true pairs (top 15):")
        for name, n in list(summary["call_names"].items())[:15]:
            lines.append(f"      {name:30} {n:5}")
    return lines
