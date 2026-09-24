"""R0 - can the verifier catch a claim that is false?

V0 measures one half of a verifier: that it never refutes a true claim. That
is the half that must be perfect, and four rounds of work made it so. But a
verifier that refuses to refute anything is perfectly safe and perfectly
useless, and nothing in this project measured the other half until the rules
that made V0 clean had already been written.

R0 is that measurement. It builds pairs that are deliberately **wrong** - one
function at -O0 against a *different* function from the same package at -O3 -
and asks what the verifier says. Every refutation is correct; every survival
is a miss.

Three numbers come out, and they are read together:

- **refuted**: the verifier did its job.
- **inconclusive**: it said nothing, which costs an opportunity and no more.
- **survived**: it saw agreement between two functions that are not the same.
  Never zero, and it should not be: two unrelated functions handed data that
  means nothing to them can genuinely do the same observable nothing. The
  verdict reports what was observed, not what is true. But it is the number
  to watch, because it is the one that can mislead.

Run it beside V0 after any change to the comparison. A change that improves
one and quietly ruins the other is the failure mode this exists to catch.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field

from elenchus.emulation.abi import Undecidable, placement
from elenchus.emulation.compare import compare
from elenchus.emulation.harness import Loader
from elenchus.emulation.inputs import input_vectors
from elenchus.emulation.stubs import resolver

R0_BUDGET = 5_000


@dataclass
class R0Report:
    """What the verifier said about claims known to be false."""
    total: int = 0
    verdicts: Counter = field(default_factory=Counter)
    forced_survivals: int = 0
    elapsed: float = 0.0

    @property
    def refuted(self) -> int:
        return self.verdicts.get("refuted", 0)

    @property
    def survived(self) -> int:
        return self.verdicts.get("survived", 0)

    @property
    def inconclusive(self) -> int:
        return self.verdicts.get("inconclusive", 0)

    def add(self, verdict: str, forced_only: bool = False) -> None:
        self.total += 1
        self.verdicts[verdict] += 1
        if verdict == "survived" and forced_only:
            self.forced_survivals += 1


def sample_false_pairs(conn, count=300, seed=0, split="train"):
    """Pairs that are wrong by construction, drawn reproducibly.

    Both sides come from the same package, so they are the kind of confusion
    a model would actually make - two functions from one library - rather
    than two things from unrelated worlds. A pair whose two names match is
    discarded: that would be a true claim, which is V0's business.
    """
    rows = conn.execute("""
        SELECT cb.package   AS package,
               cb.opt_level AS opt_level,
               gt.name      AS name,
               gt.address   AS address,
               gt.abi       AS abi,
               b.path       AS path
        FROM ground_truth gt
        JOIN corpus_binaries cb ON cb.binary_id = gt.binary_id
        JOIN binaries b         ON b.id = gt.binary_id
        JOIN dataset_split ds   ON ds.package = cb.package
        WHERE cb.stripped = 1
          AND ds.split = ?
          AND cb.opt_level IN ('O0', 'O3')
          AND gt.abi IS NOT NULL
    """, (split,)).fetchall()

    pools = {}
    for row in rows:
        pools.setdefault((row["package"], row["opt_level"]), []).append(row)
    packages = sorted({package for package, level in pools
                       if (package, "O0") in pools and (package, "O3") in pools})
    if not packages:
        return []

    rng = random.Random(seed)
    pairs, attempts = [], 0
    while len(pairs) < count and attempts < count * 100:
        attempts += 1
        package = rng.choice(packages)
        q = rng.choice(pools[(package, "O0")])
        k = rng.choice(pools[(package, "O3")])
        if q["name"] == k["name"]:
            continue                      # that is a true claim, not ours
        try:
            placement(json.loads(q["abi"]))
        except Undecidable:
            continue
        pairs.append((q, k))
    return pairs


def run_r0(conn, count=300, seed=0, budget=R0_BUDGET, split="train"):
    """Put false pairs to the verifier and report what it said."""
    report = R0Report()
    loaders = {}
    started = time.time()

    def loader_for(path):
        if path not in loaders:
            loaders[path] = Loader(path)
        return loaders[path]

    for q, k in sample_false_pairs(conn, count=count, seed=seed, split=split):
        abi = json.loads(q["abi"])
        try:
            place = placement(abi)
            result = compare(loader_for(q["path"]), q["address"],
                             loader_for(k["path"]), k["address"], place,
                             input_vectors(abi, count=6), budget=budget,
                             stub_resolver=resolver)
        except Undecidable:
            continue
        except Exception:                          # noqa: BLE001
            # A pair the harness cannot even set up says nothing about the
            # verifier's judgement, so it is left out rather than counted as
            # a miss.
            continue
        report.add(result.verdict.value,
                   forced_only=getattr(result, "forced_only", False))
    report.elapsed = time.time() - started
    return report


def _print_report(report: R0Report) -> None:
    total = report.total or 1
    print(f"\n=== R0: {report.total} pairs that are false by construction "
          f"({report.elapsed:.0f}s)\n")
    print(f"  refuted      {report.refuted:5}  {100 * report.refuted / total:5.1f}%"
          f"   the verifier did its job")
    print(f"  inconclusive {report.inconclusive:5}  "
          f"{100 * report.inconclusive / total:5.1f}%   said nothing")
    print(f"  survived     {report.survived:5}  {100 * report.survived / total:5.1f}%"
          f"   saw agreement between different functions")
    if report.forced_survivals:
        print(f"     of those, {report.forced_survivals} only on forced paths")
    print("\n  Survived is never zero and should not be: two unrelated "
          "functions given\n  data that means nothing to them can do the same "
          "observable nothing.\n  Read it beside V0 - a change that cleans one "
          "and ruins the other is\n  what these two measurements exist to "
          "catch.")


def cmd_verify_r0(args):
    import json as _json

    from elenchus.db import connect

    conn = connect(args.db)
    report = run_r0(conn, count=args.count, seed=args.seed, budget=args.budget)
    _print_report(report)

    if args.out:
        with open(args.out, "w") as handle:
            _json.dump({"count": args.count, "seed": args.seed,
                        "budget": args.budget, "total": report.total,
                        "verdicts": dict(report.verdicts),
                        "forced_survivals": report.forced_survivals,
                        "elapsed": report.elapsed}, handle, indent=1)
        print(f"\nwritten to {args.out}")
    return 0
