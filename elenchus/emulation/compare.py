"""Turn two functions' behaviour into a verdict on the claim that they match.

The harness runs one function and records an Outcome. This compares the two
sides of a claim - Q against reference K - on the same inputs, and returns
one of three verdicts (docs/verifier.md): REFUTED if they disagree
reproducibly, SURVIVED if every input that could be judged agreed, or
INCONCLUSIVE if no input could be judged either way.

The whole design turns on one rule: never refute a true claim. So a
difference counts against the claim only when it is *real* - present in both
seeds - and every way two compilations of one function can differ without
being different functions is handled here or excluded:

- The return value is compared masked to the reference type's width, and not
  at all for a void return (risk 1).
- Each version is run twice, with two fill seeds for uninitialised memory. A
  return value or a written byte that changes between the seeds *within one
  version* depends on garbage that version read, and is excluded from the
  comparison for that input (risk 2). Only observations stable across both
  seeds are compared across the two versions.
- Pointer values are never compared; memory is compared by content, and a
  page written by one version but not the other, or written differently, is
  a real effect only if it is stable across seeds.
- A float return is compared by its bits; if a calibration later shows -O0
  and -O3 differ in the last bits, a tolerance goes here, recorded with the
  verdict (risk 6). For now bit-equality is the assumption, to be tested.
- Anything that stops a clean pair of runs - an unsupported instruction, an
  unstubbed import, an exhausted budget, a fault in one version - makes that
  input inconclusive, never refuted.

Comparing is pure given the harness's Outcomes, so it is tested on
hand-built Outcomes as well as end to end through the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from elenchus.emulation.abi import Placement
from elenchus.emulation.harness import PAGE, Loader, Status, run

# The two fill seeds. Different, both non-zero (zero means input memory, which
# is filled from the address alone); a value that depends on garbage will
# differ between them, a real one will not.
SEED_A = 0x1111_1111
SEED_B = 0x2222_2222

# The two input-buffer fills. A pointer argument's buffer is data we invented,
# and some functions - a hash finaliser given a context, a parser given a
# header - branch on it, taking a different path for different bytes. Running
# the whole comparison under two fills tells that apart: if a version writes
# different places under the two, its behaviour turns on data we made up, the
# claim cannot be judged on this input, and it is inconclusive (F24). Never
# refuted: a difference we caused is not the function's.
VARIANT_A = 0
VARIANT_B = 0xA5A5_A5A5


class Verdict(Enum):
    REFUTED = "refuted"
    SURVIVED = "survived"
    INCONCLUSIVE = "inconclusive"


@dataclass
class InputResult:
    """What one input vector settled, and why."""
    verdict: Verdict
    reason: str = ""                 # what disagreed, or why undecidable
    detail: dict = field(default_factory=dict)
    status: object = None            # the harness Status when inconclusive,
    #                                  carried exactly so the caller need not
    #                                  parse `reason`; None otherwise


def _stable_return(a, b, placement: Placement):
    """The return value both seeds of one version agree on, or None.

    None means the return depended on the fill seed - garbage - so it cannot
    be compared. For a void return there is nothing to compare, which is not
    garbage: that is reported separately.
    """
    ret = placement.ret
    if ret.kind == "void":
        return "void"
    if ret.kind == "pointer":
        # A returned pointer is an address that differs between -O0 and -O3
        # (F18); its value is never compared. Like void, there is nothing to
        # compare here - the function's effect is in the memory it wrote.
        return "pointer"
    if ret.kind == "float":
        # XMM0 is 128 bits; a float defines its low 32, a double its low 64.
        # -O0 and -O3 leave the upper bits however they happen to, so the
        # return is compared masked to the type's width, the float analogue of
        # masking RAX for a narrow integer (F21). Without this a float return
        # that agrees in value refutes on the undefined upper bits.
        mask = (1 << (ret.size * 8)) - 1
        if (a.ret_float_bits & mask) != (b.ret_float_bits & mask):
            return None
        return ("float", a.ret_float_bits & mask)
    masked_a = a.ret_int & ret.mask
    masked_b = b.ret_int & ret.mask
    if masked_a != masked_b:
        return None
    return ("int", masked_a)


def _image_pointer_bytes(mine, theirs, page, ranges) -> set:
    """The byte offsets holding an address into either binary's image.

    A function that writes a pointer to one of its own globals writes an
    address, and that address differs between -O0 and -O3 exactly as a
    global's does (F15) and a returned pointer's does (F18). Only image
    addresses are excluded: a pointer into an input buffer or the arena sits
    at the same address in both versions and still compares.

    Both versions' bytes are needed, because the value is only recognisable
    as a pointer when the eight bytes are all there - and they are checked
    unaligned as well, since a compiler is free to store one anywhere (F31).
    """
    blanked = set()
    for start in sorted(set(mine) | set(theirs)):
        window = range(start, start + 8)
        if not all(offset in mine and offset in theirs for offset in window):
            continue
        for content in (mine, theirs):
            value = int.from_bytes(bytes(content[o] for o in window), "little")
            if any(low <= value < high for low, high in ranges):
                blanked.update(window)
                break
    return blanked


def _touched(outcome) -> dict:
    """Which bytes of which pages a run actually wrote: page -> set of offsets.

    A recorded page holds 4096 bytes, but a function may have written four of
    them. The rest is the fill it never touched, and comparing that is
    comparing our own pattern - which is how -O0 zeroing a struct's padding
    while -O3 leaves it alone came out as the two functions disagreeing
    (F31).
    """
    touched = {}
    for address, size in outcome.write_ranges:
        for offset in range(address, address + size):
            page = offset & ~(PAGE - 1)
            if page in outcome.writes:
                touched.setdefault(page, set()).add(offset - page)
    return touched


def _garbage_bytes(a, b) -> dict:
    """The bytes one version could not settle on, page -> set of offsets.

    A byte that differs between the two seeds came from memory the function
    read but never wrote. And a *store* with any such byte in it is garbage
    whole: one machine instruction writes one value, so half of it being
    garbage makes the value garbage - which is what page-granular detection
    missed when an eight-byte store straddled a page and only its low half
    landed on the page that got dropped (F31).
    """
    garbage = {}
    for page, content in a.writes.items():
        other = b.writes.get(page)
        if other is None:
            garbage.setdefault(page, set()).update(range(PAGE))
            continue
        differing = {i for i in range(PAGE) if content[i] != other[i]}
        if differing:
            garbage.setdefault(page, set()).update(differing)
    for page in b.writes:
        if page not in a.writes:
            garbage.setdefault(page, set()).update(range(PAGE))

    # A store only half of which we can see is not judgeable at all.
    for outcome in (a, b):
        for address, size in outcome.partial_writes:
            for offset in range(address, address + size):
                garbage.setdefault(offset & ~(PAGE - 1), set()).add(
                    offset & (PAGE - 1))

    # Spread each tainted byte over the whole store that wrote it.
    for outcome in (a, b):
        for address, size in outcome.write_ranges:
            span = [(offset & ~(PAGE - 1), offset & (PAGE - 1))
                    for offset in range(address, address + size)]
            if any(within in garbage.get(page, ()) for page, within in span):
                for page, within in span:
                    garbage.setdefault(page, set()).add(within)
    return garbage


def _stable_writes(a, b):
    """What one version wrote and can stand behind, page -> {offset: byte}.

    Only bytes the function actually wrote, and only those the two seeds
    agreed on, byte by byte and store by store.
    """
    touched = _touched(a)
    garbage = _garbage_bytes(a, b)
    stable = {}
    for page, offsets in touched.items():
        content = a.writes[page]
        bad = garbage.get(page, ())
        kept = {offset: content[offset] for offset in offsets
                if offset not in bad}
        if kept:
            stable[page] = kept
    return stable


def _judge(q_a, q_b, k_a, k_b, placement: Placement, image_ranges=()) -> InputResult:
    """Compare the two versions on one input, each already run twice."""
    # Any run that did not complete makes the input inconclusive. Carry the
    # first non-completed status exactly, so the caller buckets on the enum
    # rather than on the wording of `reason`.
    for outcome, who in ((q_a, "Q"), (q_b, "Q"), (k_a, "K"), (k_b, "K")):
        if outcome.status is not Status.COMPLETED:
            return InputResult(Verdict.INCONCLUSIVE,
                               f"{who} {outcome.status.value}",
                               {"detail": outcome.detail},
                               status=outcome.status)

    q_ret = _stable_return(q_a, q_b, placement)
    k_ret = _stable_return(k_a, k_b, placement)
    if q_ret is None or k_ret is None:
        # The return depended on garbage in at least one version; it cannot be
        # judged, but the written memory still can.
        ret_verdict = None
    elif q_ret != k_ret:
        return InputResult(Verdict.REFUTED, "return value differs",
                           {"Q": q_ret, "K": k_ret})
    else:
        ret_verdict = "agree"

    q_writes = _stable_writes(q_a, q_b)
    k_writes = _stable_writes(k_a, k_b)
    # Only bytes *both* versions wrote and both could stand behind are
    # evidence. A byte one wrote and the other never touched is that other's
    # padding, and padding is indeterminate in C - comparing it made -O0
    # zeroing a struct disagree with -O3 leaving it alone (F31).
    differing = []
    for page in set(q_writes) & set(k_writes):
        mine, theirs = q_writes[page], k_writes[page]
        shared = set(mine) & set(theirs)
        if image_ranges:
            shared -= _image_pointer_bytes(mine, theirs, page, image_ranges)
        if any(mine[offset] != theirs[offset] for offset in shared):
            differing.append(hex(page))
    if differing:
        return InputResult(Verdict.REFUTED, "written memory differs",
                           {"pages": differing})

    if ret_verdict is None and placement.ret.kind != "void":
        # Nothing could be compared: the return was garbage-dependent and no
        # memory was written. This input decides nothing.
        return InputResult(Verdict.INCONCLUSIVE, "return depended on the fill")

    return InputResult(Verdict.SURVIVED, "agreed on this input")


@dataclass
class Comparison:
    """The verdict on a claim, and the per-input results behind it."""
    verdict: Verdict
    inputs: list = field(default_factory=list)   # InputResult per vector
    refuting_input: int | None = None            # index of the first refutation
    max_instructions: int = 0                    # the longest run behind it,
    #                                              so V0 can see where the real
    #                                              budget sits
    # The second tier (docs/verifier.md): agreements seen on paths reached by
    # forcing a branch. Such a path is infeasible, so it can corroborate but
    # never refute, and a survival that rests only on it is weaker and says so.
    forced_attempts: int = 0
    forced_agreements: int = 0
    forced_only: bool = False


def _combine_variants(per_variant) -> InputResult:
    """One verdict for an input from its judgement under both fills.

    The rule is asymmetric, like everything else here:

    - Both fills refuted: the difference is there whatever we put in the
      buffers, so it is the functions' own. REFUTED.
    - One refuted, the other did not: the difference turns on bytes we
      invented, so the claim cannot be judged on this input. INCONCLUSIVE,
      never refuted - a difference we caused is not evidence.
    - Neither refuted and at least one survived: SURVIVED.
    - Neither could be judged: INCONCLUSIVE, carrying the first reason.
    """
    verdicts = [r.verdict for r in per_variant]
    if all(v is Verdict.REFUTED for v in verdicts):
        return per_variant[0]
    if any(v is Verdict.REFUTED for v in verdicts):
        refuting = next(r for r in per_variant if r.verdict is Verdict.REFUTED)
        return InputResult(Verdict.INCONCLUSIVE,
                           "differs under one input fill only",
                           refuting.detail)
    if any(v is Verdict.SURVIVED for v in verdicts):
        return next(r for r in per_variant if r.verdict is Verdict.SURVIVED)
    return per_variant[0]


# What the second tier is allowed to spend, measured rather than chosen
# (F29). Each attempt costs four runs, and each input two recording runs on
# top, so a pair the first tier could not judge costs
# FORCED_INPUTS * (2 + 4 * FORCED_ATTEMPTS) runs. Scanned over 150 pairs:
#
#   attempts  1 -> 4.0% forced agreement, 2 -> 5.3%, 4 -> 5.3%
#   inputs    1 -> 5.3%,                  2 -> 5.3%, 6 -> 5.3%
#
# Two attempts is where the gain stops. And the number of inputs changes it
# not at all: if forcing is going to work on a pair, it works on the first
# input, and the other five only cost time - 477 s against 319 s for exactly
# the same result. The first version spent 6 inputs and 4 attempts, and the
# 500-pair run took 27 minutes, nineteen of them in the kernel mapping images
# for runs that changed nothing.
FORCED_ATTEMPTS = 2
FORCED_INPUTS = 1


def _forced_agreement(q_loader, q_address, k_loader, k_address, placement,
                      args, budget, stub_resolver, q_prepare, k_prepare,
                      image_ranges, seed) -> tuple[int, int]:
    """Run both versions with corresponding branches forced, and count.

    Returns (attempts, agreements). A difference here is discarded: the path
    does not exist for any real input, so a difference on it is not the
    functions' and cannot refute. An agreement is evidence, collected from
    code the first tier could not reach.
    """
    from elenchus.emulation.forced import corresponding_flips

    def record(loader, address, prepare):
        return run(loader, address, placement, args, seed=SEED_A,
                   budget=budget, stub_resolver=stub_resolver,
                   input_variant=VARIANT_A, prepare=prepare,
                   record_predicates=True)

    q_seen = record(q_loader, q_address, q_prepare)
    k_seen = record(k_loader, k_address, k_prepare)

    attempts = agreements = 0
    for q_force, k_force in corresponding_flips(
            q_seen.predicates, k_seen.predicates, FORCED_ATTEMPTS, seed):
        attempts += 1
        runs = [
            run(loader, address, placement, args, seed=run_seed, budget=budget,
                stub_resolver=stub_resolver, input_variant=variant,
                prepare=prepare, force=force)
            for loader, address, prepare, force in (
                (q_loader, q_address, q_prepare, q_force),
                (k_loader, k_address, k_prepare, k_force))
            for run_seed in (SEED_A, SEED_B)
            for variant in (VARIANT_A,)
        ]
        judged = _judge(runs[0], runs[1], runs[2], runs[3], placement,
                        image_ranges)
        if judged.verdict is Verdict.SURVIVED:
            agreements += 1
    return attempts, agreements


def compare(q_loader: Loader, q_address: int,
            k_loader: Loader, k_address: int,
            placement: Placement, input_vectors,
            budget: int = 5_000_000, stub_resolver=None,
            q_prepare=None, k_prepare=None,
            force_when_unjudged: bool = True,
            forcing_seed: int = 1) -> Comparison:
    """Test whether Q behaves as K across the given inputs.

    Each side is run twice per input, at SEED_A and SEED_B, so a
    garbage-dependent observation can be told from a real one. The first
    input that refutes ends the comparison (there is no stronger evidence
    than one counterexample); with no refutation, the verdict is SURVIVED if
    at least one input was judged and INCONCLUSIVE if none could be.

    stub_resolver serves the two sides' import calls; both sides get the same
    resolver, so a claim is judged against one set of stubs. Passed None -
    the default - every import is inconclusive, which is layer A of V0.
    """
    # Both binaries' image ranges, so a written pointer into either one can be
    # recognised as an address rather than compared as data (F23).
    image_ranges = (
        (q_loader.base, q_loader.base + q_loader.size),
        (k_loader.base, k_loader.base + k_loader.size),
    )
    results = []
    judged_any = False
    peak = 0
    for index, args in enumerate(input_vectors):
        # Each input is judged twice, once per input-buffer fill. A refutation
        # counts only if both fills refute: a difference that appears under
        # one fill and not the other came from the bytes we invented, not from
        # the functions (F24).
        per_variant = []
        for variant in (VARIANT_A, VARIANT_B):
            runs = [
                run(loader, address, placement, args, seed=seed, budget=budget,
                    stub_resolver=stub_resolver, input_variant=variant,
                    prepare=prepare)
                for loader, address, prepare in (
                    (q_loader, q_address, q_prepare),
                    (k_loader, k_address, k_prepare))
                for seed in (SEED_A, SEED_B)
            ]
            q_a, q_b, k_a, k_b = runs[0], runs[1], runs[2], runs[3]
            peak = max(peak, *(r.instructions for r in runs))
            per_variant.append(_judge(q_a, q_b, k_a, k_b, placement, image_ranges))

        result = _combine_variants(per_variant)
        results.append(result)
        if result.verdict is Verdict.REFUTED:
            return Comparison(Verdict.REFUTED, results, refuting_input=index,
                              max_instructions=peak)
        if result.verdict is Verdict.SURVIVED:
            judged_any = True

    if judged_any:
        return Comparison(Verdict.SURVIVED, results, max_instructions=peak)

    # The first tier could judge nothing: the function faulted on the data we
    # invented, or ran out of budget in a loop that data sent it round, or
    # refused to work at all on a context it did not build. That is where the
    # unjudged pairs are, and it is what the second tier is for - run it again
    # with a branch forced, and count the agreements (docs/verifier.md).
    attempts = agreements = 0
    for args in (input_vectors[:FORCED_INPUTS] if force_when_unjudged else ()):
        made, agreed = _forced_agreement(
            q_loader, q_address, k_loader, k_address, placement, args, budget,
            stub_resolver, q_prepare, k_prepare, image_ranges, forcing_seed)
        attempts += made
        agreements += agreed
        if agreements:
            break              # one corroborated input is enough to report

    if agreements:
        # Survived, but only on paths no real input takes. Weaker than a
        # feasible survival, and the flag is how a caller says so rather than
        # quietly treating the two as the same thing.
        return Comparison(Verdict.SURVIVED, results, max_instructions=peak,
                          forced_attempts=attempts,
                          forced_agreements=agreements, forced_only=True)
    return Comparison(Verdict.INCONCLUSIVE, results, max_instructions=peak,
                      forced_attempts=attempts)
