"""L2 results: read the short learning-curve runs and apply the rule.

The protocol (docs/experiments.md, L2, written 18 September before any run):

  five runs: 100% seeds 0 and 1, package 50% seeds 0 and 1, identity 50%
  seed 0; each contrastive from random weights, 11,550 steps (30 epochs of
  full train at 385), the rate decaying over all of them, validated every
  385 steps, read at best.pt

  identity gain = MRR(100%, seed 0) - MRR(identity 50%, seed 0)
  package gain  = mean over seeds s of MRR(100%, s) - MRR(package 50%, s)
  T = max(0.01, M) = 0.01, M = 0.0014 from L's two full runs

  both gains <= T    levelled off: full training, its budget set from where
                     the 100% runs' best.pt lands
  larger in (T, 2T]  still climbing, by less than the doubling that produced
                     it: full training anyway, the gain recorded as what a
                     wave 3 would have been worth
  larger > 2T        wave 3 before full training: variety if the package
                     gain leads, large packages in existing domains if only
                     the identity gain is above

  noise check:  if the two 100% runs differ by more than T, the seed spread
                is as large as the threshold and the verdict is provisional
  budget check: if a 100% run's best beats its best by 75% of the steps by
                more than T / 2 it was still climbing; a "levelled off"
                verdict is then provisional and the runs repeat at twice the
                budget (a gain above T from under-trained runs stands)

MRR is over queries, O0 -> O3 on val, from best.pt measured again from disk.
The package mean is printed beside it and decides nothing: val has packages
with three queries and packages with four hundred (F10).

Run from the repo root when experiments/run_curve.sh has finished:
  python experiments/curve2_results.py
"""

import json
import sys
from pathlib import Path

STEPS = 11550
EVERY = 385
T = 0.01            # max(0.01, M); M = 0.0014 (docs/experiments.md, L, Result)
LATE = 0.75
PREFIX = "curve2"

# (run name, fraction, unit, seed); all five decide
RUNS = [
    ("full", 1.0, None, 0),
    ("identity-0.5", 0.5, "identity", 0),
    ("package-0.5", 0.5, "package", 0),
    ("package-0.5", 0.5, "package", 1),
    ("full", 1.0, None, 1),
]


def run_dir(root, name, seed):
    return Path(root) / f"{PREFIX}-{name}-seed-{seed}"


def read(summary, steps=STEPS):
    """The figures the rule and the table need, from one summary.json."""
    trained = [h for h in summary["history"] if h["epoch"] >= 0]
    best = summary["best_val_metrics"]
    kept = next(h for h in trained if h["epoch"] == summary["best_epoch"])
    by_late = [h["mrr"] for h in trained if h["step"] <= LATE * steps]
    subset = summary.get("train_subset", {})
    return {
        "mrr": best["mrr"],
        "package_mrr": best.get("package_mean", {}).get("mrr"),
        "recall@10": best["recall@10"],
        "best_step": kept["step"],
        "final_step": trained[-1]["step"] if trained else 0,
        "complete": bool(trained) and trained[-1]["step"] == steps,
        "late_gain": max(h["mrr"] for h in trained) - max(by_late) if by_late else None,
        "identities": subset.get("identities"),
        "packages": subset.get("packages"),
    }


def load(root="models", steps=STEPS):
    results = {}
    for name, _fraction, _unit, seed in RUNS:
        path = run_dir(root, name, seed) / "summary.json"
        results[(name, seed)] = (read(json.loads(path.read_text()), steps)
                                 if path.exists() else None)
    return results


def verdict(results):
    """Apply the rule. Returns a dict; "decision" is None while runs are missing."""
    missing = [f"{name} seed {seed}" for name, _f, _u, seed in RUNS
               if results.get((name, seed)) is None
               or not results[(name, seed)]["complete"]]
    if missing:
        return {"decision": None, "missing": missing}

    def mrr(name, seed):
        return results[(name, seed)]["mrr"]

    identity_gain = mrr("full", 0) - mrr("identity-0.5", 0)
    package_gains = [mrr("full", seed) - mrr("package-0.5", seed) for seed in (0, 1)]
    package_gain = sum(package_gains) / 2
    larger = max(identity_gain, package_gain)

    if larger <= T:
        decision = "levelled off: full training"
    elif larger <= 2 * T:
        decision = ("still climbing below a doubling's worth: full training, "
                    "the gain recorded")
    elif package_gain >= identity_gain:
        decision = "more packages still help: wave 3, the thinnest domains first"
    else:
        decision = ("only more functions help: wave 3, large packages in "
                    "existing domains")

    seed_spread = abs(mrr("full", 0) - mrr("full", 1))
    still_climbing = [seed for seed in (0, 1)
                      if results[("full", seed)]["late_gain"] > T / 2]
    provisional = (seed_spread > T
                   or (bool(still_climbing) and decision.startswith("levelled off")))

    return {"decision": decision, "missing": [], "identity_gain": identity_gain,
            "package_gains": package_gains, "package_gain": package_gain,
            "seed_spread": seed_spread, "still_climbing": still_climbing,
            "provisional": provisional,
            "best_steps": [results[("full", seed)]["best_step"] for seed in (0, 1)]}


def show(results):
    def cell(value, width, spec=".4f"):
        if value is None:
            return f"{'-':>{width}}"
        return format(value, spec).rjust(width)

    print(f"{'run':15} {'seed':>4} {'funcs':>6} {'pkgs':>4} {'MRR':>7} {'pkg MRR':>7} "
          f"{'R@10':>6} {'best at':>7} {'last 25%':>8}  done")
    for name, _fraction, _unit, seed in RUNS:
        run = results.get((name, seed))
        if run is None:
            print(f"{name:15} {seed:>4}  not finished")
            continue
        print(f"{name:15} {seed:>4} {run['identities'] or 0:>6} "
              f"{run['packages'] or 0:>4} {cell(run['mrr'], 7)} "
              f"{cell(run['package_mrr'], 7)} {cell(run['recall@10'], 6, '.3f')} "
              f"{run['best_step']:>7} {cell(run['late_gain'], 8)}"
              f"  {'yes' if run['complete'] else 'NO'}")


def main(root="models"):
    results = load(root)
    show(results)
    outcome = verdict(results)
    print()
    if outcome["decision"] is None:
        print("not finished  : " + ", ".join(outcome["missing"]))
        return 1

    print(f"identity gain : {outcome['identity_gain']:+.4f}  "
          f"({outcome['identity_gain'] / T:.1f} T)")
    print(f"package gain  : {outcome['package_gain']:+.4f}  "
          f"({outcome['package_gain'] / T:.1f} T), seeds "
          + ", ".join(f"{g:+.4f}" for g in outcome["package_gains"]))
    print(f"seed spread   : {outcome['seed_spread']:.4f} against T = {T}")
    print(f"full runs kept: steps {outcome['best_steps'][0]} and "
          f"{outcome['best_steps'][1]} of {STEPS}")
    if outcome["still_climbing"]:
        print("still climbing: seeds "
              + ", ".join(str(s) for s in outcome["still_climbing"]))
    print(f"verdict       : {outcome['decision']}"
          f"{'  (PROVISIONAL)' if outcome['provisional'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
