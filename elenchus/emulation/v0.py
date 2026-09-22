"""V0 - how much of a real library the verifier can run.

The one number the project does not have (docs/verifier.md, V0 protocol):
across a sample of true pairs - the same function at -O0 and -O3, known the
same from debug information - how many run to completion, and of those, how
many the harness wrongly refutes. Everything after V0 - the real instruction
budget, which stubs to write next, whether the approach is worth building -
is set by what this measures.

Two layers on the same sample: bare (no stubs) and with the base stubs. The
gap between them is the value of those stubs. Every pair falls in one bucket,
and among the completed pairs the disagreements are counted and their causes
named, because those are the false refutations the design exists to drive to
zero.

This module builds the sample and buckets a pair; running it needs the corpus
binaries on disk, so the command lives here and the measurement is taken on
the machine that has them. The bucketing is tested on the fixtures.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field

from elenchus.emulation.abi import Undecidable, placement
from elenchus.emulation.compare import Verdict, compare
from elenchus.emulation.harness import Status
from elenchus.emulation.inputs import input_vectors

# The starting instruction budget. V0 measures whether it is right: a pair
# that does not finish under it is budget-exhausted, and the distribution of
# how many instructions the finishers took says where the real budget sits.
# Lowered from the 5M ceiling to 500k after the first pass: real functions
# that finish do so in far fewer, and a lower budget turns a 5M-instruction
# spin from most of the running time into a quick budget-exhausted verdict.
V0_BUDGET = 500_000

# How many input vectors per pair. Generated from the reference signature by
# inputs.input_vectors - real buffers for pointers, boundary values for
# integers - so a function is exercised where it can refute rather than
# crashed on a pointer argument given a small integer.
V0_INPUT_COUNT = 6


class Bucket:
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported instruction"
    IMPORT_MISSING = "import without a stub"
    BUDGET = "budget exhausted"
    TOO_MUCH_MEMORY = "mapped too much memory"
    FAULT = "fault (bad memory access)"
    SIGNATURE_DECLINED = "signature declined"
    LOAD_FAILED = "load failed"
    UNJUDGED = "ran but nothing to compare"
    # among completed pairs:
    AGREED = "agreed"
    DISAGREED = "disagreed"


# Each harness Status maps to exactly one bucket. No string matching: the
# comparison carries the Status enum, and this is the whole translation.
_STATUS_BUCKET = {
    Status.UNSUPPORTED_INSTRUCTION: Bucket.UNSUPPORTED,
    Status.IMPORT_WITHOUT_STUB: Bucket.IMPORT_MISSING,
    Status.BUDGET_EXHAUSTED: Bucket.BUDGET,
    Status.TOO_MUCH_MEMORY: Bucket.TOO_MUCH_MEMORY,
    Status.FAULT: Bucket.FAULT,
}


@dataclass
class PairResult:
    package: str
    name: str
    bucket: str
    agreement: str | None = None        # AGREED / DISAGREED, if completed
    detail: str = ""
    instructions: int = 0               # the longest run, when completed


@dataclass
class V0Report:
    layer: str                          # "bare" or "stubbed"
    buckets: Counter = field(default_factory=Counter)
    agreement: Counter = field(default_factory=Counter)
    imports_missing: Counter = field(default_factory=Counter)
    disagreements: list = field(default_factory=list)   # PairResult, for reading
    completed_instructions: list = field(default_factory=list)  # per completed pair

    def add(self, result: PairResult):
        self.buckets[result.bucket] += 1
        if result.bucket == Bucket.COMPLETED:
            self.agreement[result.agreement] += 1
            self.completed_instructions.append(result.instructions)
            if result.agreement == Bucket.DISAGREED:
                self.disagreements.append(result)
        elif result.bucket == Bucket.IMPORT_MISSING:
            self.imports_missing[result.detail] += 1

    @property
    def total(self):
        return sum(self.buckets.values())

    def instruction_percentiles(self):
        """Where completed pairs' longest runs fall: (median, p90, p99, max).

        This is what sets the real budget - if every finisher is under some
        number, the budget comes down to fit and a spin is caught far sooner.
        """
        counts = sorted(self.completed_instructions)
        if not counts:
            return None

        def pct(p):
            return counts[min(len(counts) - 1, int(len(counts) * p))]

        return pct(0.50), pct(0.90), pct(0.99), counts[-1]


def sample_pairs(conn, split="train", count=3000, seed=0):
    """Return `count` true pairs from `split`, drawn by `seed`.

    A pair is one identity - (package, decl_file, name) - present in both the
    -O0 and -O3 stripped binary, each with an address, a binary path and a
    resolved signature. Only identities the placement can be built for are
    eligible; the rest are counted as signature-declined without running.
    """
    rows = conn.execute("""
        SELECT cb.package        AS package,
               gt.decl_file      AS decl_file,
               gt.name           AS name,
               cb.opt_level      AS opt_level,
               gt.address        AS address,
               gt.abi            AS abi,
               b.path            AS path
        FROM ground_truth gt
        JOIN corpus_binaries cb ON cb.binary_id = gt.binary_id
        JOIN binaries b         ON b.id = gt.binary_id
        JOIN dataset_split ds   ON ds.package = cb.package
        WHERE cb.stripped = 1
          AND ds.split = ?
          AND cb.opt_level IN ('O0', 'O3')
          AND gt.decl_file IS NOT NULL
          AND gt.abi IS NOT NULL
    """, (split,)).fetchall()

    sides = {}
    for row in rows:
        key = (row["package"], row["decl_file"], row["name"])
        sides.setdefault(key, {})[row["opt_level"]] = row

    identities = [(key, s["O0"], s["O3"]) for key, s in sides.items()
                  if "O0" in s and "O3" in s]
    identities.sort()                    # deterministic before the seeded shuffle
    random.Random(seed).shuffle(identities)
    return identities[:count]


def bucket_pair(o0_loader, o0_addr, o3_loader, o3_addr, abi_json,
                resolver=None, budget=V0_BUDGET, inputs=None):
    """Run one pair and return its bucket and, if completed, whether it agreed.

    resolver None is the bare layer; a resolver is the stubbed layer. The
    comparison already runs each side twice per seed, so a disagreement here is
    a real one - a false refutation to investigate, since the pair is true.

    inputs, if given, overrides the generated vectors (the tests use this);
    otherwise they are built from the signature by inputs.input_vectors, so a
    pointer argument gets a real buffer rather than a small integer.
    """
    abi = json.loads(abi_json)
    try:
        place = placement(abi)
    except Undecidable as exc:
        return PairResult("", "", Bucket.SIGNATURE_DECLINED, detail=str(exc))

    vectors = inputs if inputs is not None else input_vectors(
        abi, count=V0_INPUT_COUNT)
    result = compare(o0_loader, o0_addr, o3_loader, o3_addr, place, vectors,
                     budget=budget, stub_resolver=resolver)

    if result.verdict is Verdict.REFUTED:
        return PairResult("", "", Bucket.COMPLETED, Bucket.DISAGREED,
                          detail=_first_refute_reason(result),
                          instructions=result.max_instructions)
    if result.verdict is Verdict.SURVIVED:
        return PairResult("", "", Bucket.COMPLETED, Bucket.AGREED,
                          instructions=result.max_instructions)

    # Inconclusive: nothing could be judged. Name the commonest reason, which
    # is the bucket V0 reports - a missing stub, the budget, an instruction.
    return _inconclusive_bucket(result)


def run_v0(conn, count=3000, seed=0, budget=V0_BUDGET, split="train"):
    """Run V0 both layers on a sample and return (bare, stubbed) reports.

    Loads each binary once and reuses it across the pairs that live in it, so
    a package's -O0 and -O3 files are opened once each rather than per pair.
    Needs the corpus binaries on disk; this is the command's body.
    """
    from elenchus.emulation.harness import Loader
    from elenchus.emulation.stubs import resolver

    pairs = sample_pairs(conn, split=split, count=count, seed=seed)
    loaders = {}

    def loader_for(path):
        if path not in loaders:
            loaders[path] = Loader(path)
        return loaders[path]

    bare = V0Report(layer="bare")
    stubbed = V0Report(layer="stubbed")
    for (package, _decl, name), o0, o3 in pairs:
        try:
            o0_loader = loader_for(o0["path"])
            o3_loader = loader_for(o3["path"])
        except Exception as exc:                       # noqa: BLE001
            for report in (bare, stubbed):
                report.add(PairResult(package, name, Bucket.LOAD_FAILED,
                                      detail=str(exc)))
            continue
        for report, res in ((bare, None), (stubbed, resolver)):
            result = bucket_pair(o0_loader, o0["address"], o3_loader,
                                 o3["address"], o0["abi"], resolver=res,
                                 budget=budget)
            result.package, result.name = package, name
            report.add(result)
    return bare, stubbed


def _print_report(report: V0Report):
    print(f"\n=== layer: {report.layer}   ({report.total} pairs)")
    for bucket, n in report.buckets.most_common():
        print(f"  {bucket:24} {n:6}  {100 * n / report.total:5.1f}%")
    completed = report.buckets.get(Bucket.COMPLETED, 0)
    if completed:
        agreed = report.agreement.get(Bucket.AGREED, 0)
        disagreed = report.agreement.get(Bucket.DISAGREED, 0)
        print(f"  of completed: {agreed} agreed, {disagreed} DISAGREED "
              f"(false refutations to investigate)")
        pcts = report.instruction_percentiles()
        if pcts:
            median, p90, p99, top = pcts
            print(f"  instructions of finishers: median {median}, p90 {p90}, "
                  f"p99 {p99}, max {top}")
    if report.imports_missing:
        print("  imports to stub next (top 15):")
        for name, n in report.imports_missing.most_common(15):
            print(f"      {name:24} {n:6}")
    for d in report.disagreements[:20]:
        print(f"      DISAGREED {d.package}/{d.name}: {d.detail}")


def cmd_verify_v0(args):
    from elenchus.db import connect

    conn = connect(args.db)
    bare, stubbed = run_v0(conn, count=args.count, seed=args.seed,
                           budget=args.budget)
    _print_report(bare)
    _print_report(stubbed)

    bare_done = bare.buckets.get(Bucket.COMPLETED, 0)
    stub_done = stubbed.buckets.get(Bucket.COMPLETED, 0)
    total = bare.total or 1
    print(f"\ncoverage: bare {100 * bare_done / total:.1f}%, "
          f"stubbed {100 * stub_done / total:.1f}%  "
          f"(the stubs are worth {100 * (stub_done - bare_done) / total:.1f} points)")
    return 0


def _first_refute_reason(comparison):
    for r in comparison.inputs:
        if r.verdict is Verdict.REFUTED:
            return r.reason
    return ""


def _inconclusive_bucket(comparison):
    """Map an all-inconclusive comparison to a bucket, by the exact Status.

    Every input that stopped a run carries the harness Status that stopped it
    (compare.InputResult.status). The bucket is the most common such status
    across the inputs, translated through _STATUS_BUCKET - no string matching,
    so the count of, say, faults is exactly the count of faults. An import is
    named from its detail so the histogram can point at the next stub.

    An input with no status is one that ran but had nothing to compare (a
    garbage-dependent return, no memory written); if that is all there is, the
    pair is UNJUDGED - it ran, it just decided nothing.
    """
    statuses = Counter()
    import_names = Counter()
    for r in comparison.inputs:
        if r.status is None:
            statuses[None] += 1
            continue
        statuses[r.status] += 1
        if r.status is Status.IMPORT_WITHOUT_STUB:
            name = r.detail.get("detail", "")
            import_names[name.split(":")[0].strip()] += 1

    ranked = [s for s in statuses if s is not None]
    if not ranked:
        return PairResult("", "", Bucket.UNJUDGED)

    status = max(ranked, key=lambda s: (statuses[s], s.value))
    bucket = _STATUS_BUCKET[status]
    detail = ""
    if status is Status.IMPORT_WITHOUT_STUB and import_names:
        detail = import_names.most_common(1)[0][0]
    return PairResult("", "", bucket, detail=detail)
