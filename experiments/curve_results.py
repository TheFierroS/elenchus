"""L results: read the learning-curve runs and apply the rule.

The protocol (docs/experiments.md, L, amended 17 September before any run):

  every run: contrastive from random weights, STEPS steps, the learning rate
  decaying over all of them, validated every 172 steps, read at best.pt

  identity gain = MRR(100%, seed 0) - MRR(identity 50%, seed 0)
  package gain  = mean over seeds s of MRR(100%, seed s) - MRR(package 50%, seed s)
  T = max(0.01, M) = 0.0148, M from E3

  both gains <= T      the curve has flattened: no wave 2, go to full training
  package gain > T     more packages help: wave 2, data-format and text first
  only identity > T    more functions help: wave 2, large packages in
                       existing domains

  budget check: if a 100% run's best MRR beats its best by 75% of the steps
  by more than T / 2, it was still climbing; a "flattened" verdict is then
  provisional and the runs repeat at twice the budget (a gain above T from
  under-trained runs stands: more training would not shrink it)

  the 25% -> 50% gains are the curve's shape, not the decision

MRR is over queries, O0 -> O3 on val, from best.pt measured again from disk.
The summaries are read as they are; nothing is trained or measured here.

Run from the repo root when experiments/run_curve.sh has finished:
  python experiments/curve_results.py
"""

import json
import sys
from pathlib import Path

STEPS = 5160
T = 0.0148          # max(0.01, M); M = 0.0148 (docs/experiments.md, E3, Result)
LATE = 0.75

# (run name, fraction, unit, seed); the first five decide
RUNS = [
    ("full", 1.0, None, 0),
    ("identity-0.5", 0.5, "identity", 0),
    ("package-0.5", 0.5, "package", 0),
    ("package-0.5", 0.5, "package", 1),
    ("full", 1.0, None, 1),
    ("identity-0.25", 0.25, "identity", 0),
    ("package-0.25", 0.25, "package", 0),
    ("package-0.25", 0.25, "package", 1),
]
DECIDING = RUNS[:5]


def run_dir(root, name, seed):
    return Path(root) / f"curve-{name}-seed-{seed}"


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
    missing = [f"{name} seed {seed}" for name, _f, _u, seed in DECIDING
               if results.get((name, seed)) is None
               or not results[(name, seed)]["complete"]]
    if missing:
        return {"decision": None, "missing": missing}

    def mrr(name, seed):
        return results[(name, seed)]["mrr"]

    identity_gain = mrr("full", 0) - mrr("identity-0.5", 0)
    package_gains = [mrr("full", seed) - mrr("package-0.5", seed) for seed in (0, 1)]
    package_gain = sum(package_gains) / 2

    if package_gain > T:
        decision = "more packages help: wave 2, data-format and text first"
    elif identity_gain > T:
        decision = ("only more functions help: wave 2, large packages in "
                    "existing domains")
    else:
        decision = "flattened: no wave 2, go to full training"

    still_climbing = [seed for seed in (0, 1)
                      if results[("full", seed)]["late_gain"] > T / 2]
    provisional = bool(still_climbing) and decision.startswith("flattened")

    def shape(unit):
        gains = []
        for seed in ((0,) if unit == "identity" else (0, 1)):
            half, quarter = (results.get((f"{unit}-0.5", seed)),
                             results.get((f"{unit}-0.25", seed)))
            if half is None or quarter is None or not quarter["complete"]:
                return None
            gains.append(half["mrr"] - quarter["mrr"])
        return sum(gains) / len(gains)

    return {"decision": decision, "missing": [], "identity_gain": identity_gain,
            "package_gains": package_gains, "package_gain": package_gain,
            "still_climbing": still_climbing, "provisional": provisional,
            "shape": {"identity": shape("identity"), "package": shape("package")}}


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
        print(f"{name:15} {seed:>4} {cell(run['identities'], 6, 'd')} "
              f"{cell(run['packages'], 4, 'd')} {cell(run['mrr'], 7)} "
              f"{cell(run['package_mrr'], 7)} {cell(run['recall@10'], 6, '.3f')} "
              f"{run['best_step']:>7} {cell(run['late_gain'], 8, '+.4f')}  "
              f"{'yes' if run['complete'] else 'NO, step ' + str(run['final_step'])}")

    result = verdict(results)
    print(f"\nT = {T:.4f}   (budget {STEPS} steps; 'last 25%' = best MRR minus the "
          f"best by step {int(LATE * STEPS)})")
    if result["decision"] is None:
        print("verdict: waiting for " + ", ".join(result["missing"]))
        return result
    seeds = ", ".join(f"seed {s} {g:+.4f}" for s, g in enumerate(result["package_gains"]))
    print(f"identity gain 50% -> 100%: {result['identity_gain']:+.4f}")
    print(f"package gain  50% -> 100%: {result['package_gain']:+.4f}  ({seeds})")
    shape = result["shape"]
    print("shape 25% -> 50% (not decided on): "
          f"identity {cell(shape['identity'], 0, '+.4f')}, "
          f"package {cell(shape['package'], 0, '+.4f')}")
    if result["still_climbing"]:
        print(f"budget check: 100% seed(s) {result['still_climbing']} gained more than "
              f"T/2 = {T / 2:.4f} in the last 25% of steps")
    else:
        print("budget check: both 100% runs had levelled off by 75% of the budget")
    label = "PROVISIONAL, repeat at twice the budget: " if result["provisional"] else ""
    print(f"verdict: {label}{result['decision']}")
    return result


def main():
    show(load())
    return 0


if __name__ == "__main__":
    sys.exit(main())
