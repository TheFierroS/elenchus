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
from elenchus.emulation.harness import Loader, Status, run

# The two fill seeds. Different, both non-zero (zero means input memory, which
# is filled from the address alone); a value that depends on garbage will
# differ between them, a real one will not.
SEED_A = 0x1111_1111
SEED_B = 0x2222_2222


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
        if a.ret_float_bits != b.ret_float_bits:
            return None
        return ("float", a.ret_float_bits)
    masked_a = a.ret_int & ret.mask
    masked_b = b.ret_int & ret.mask
    if masked_a != masked_b:
        return None
    return ("int", masked_a)


def _stable_writes(a, b):
    """The pages both seeds of one version wrote identically.

    A page whose contents differ between the two seeds was filled from
    garbage the version read, so it is dropped. What remains is the memory
    the function genuinely produced, keyed by page for comparison across
    versions.
    """
    stable = {}
    for page, content in a.writes.items():
        if page in b.writes and b.writes[page] == content:
            stable[page] = content
    return stable


def _judge(q_a, q_b, k_a, k_b, placement: Placement) -> InputResult:
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
    if q_writes != k_writes:
        pages = sorted(set(q_writes) | set(k_writes))
        differing = [hex(p) for p in pages if q_writes.get(p) != k_writes.get(p)]
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


def compare(q_loader: Loader, q_address: int,
            k_loader: Loader, k_address: int,
            placement: Placement, input_vectors,
            budget: int = 5_000_000, stub_resolver=None) -> Comparison:
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
    results = []
    judged_any = False
    peak = 0
    for index, args in enumerate(input_vectors):
        q_a = run(q_loader, q_address, placement, args, seed=SEED_A,
                  budget=budget, stub_resolver=stub_resolver)
        q_b = run(q_loader, q_address, placement, args, seed=SEED_B,
                  budget=budget, stub_resolver=stub_resolver)
        k_a = run(k_loader, k_address, placement, args, seed=SEED_A,
                  budget=budget, stub_resolver=stub_resolver)
        k_b = run(k_loader, k_address, placement, args, seed=SEED_B,
                  budget=budget, stub_resolver=stub_resolver)
        peak = max(peak, q_a.instructions, q_b.instructions,
                   k_a.instructions, k_b.instructions)

        result = _judge(q_a, q_b, k_a, k_b, placement)
        results.append(result)
        if result.verdict is Verdict.REFUTED:
            return Comparison(Verdict.REFUTED, results, refuting_input=index,
                              max_instructions=peak)
        if result.verdict is Verdict.SURVIVED:
            judged_any = True

    return Comparison(Verdict.SURVIVED if judged_any else Verdict.INCONCLUSIVE,
                      results, max_instructions=peak)
