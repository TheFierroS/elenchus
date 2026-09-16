"""B2-B4 results: the full runs, their seed noise, and the MLM verdict.

Reads the summary.json of every finished run under models/ (a run without
one was cut short and is not a result). Nothing is re-measured: a
contrastive run records its kept checkpoint's val metrics, measured again
from disk when training ended.

The rule was written before the runs (docs/experiments.md, full training):

  spread S   = max(max-min over seeds of B3's val MRR, same for B4)
  margin     = max(0.005, S)
  MLM helps  if mean(B3) - mean(B4) > margin and B3's mean package MRR is not
             below B4's by more than margin
  MLM hurts  if mean(B4) - mean(B3) > margin
  otherwise  no measured difference: later work drops MLM, and E1/E2 with it

S is also the run-to-run noise at full scale, which later experiments read
their differences against.

Run from the repo root:  python experiments/full_results.py
"""

import json
import sys
from pathlib import Path
from statistics import fmean

SEEDS = (0, 1, 2)
BAR = {"mrr": 0.124, "recall@10": 0.189}   # val baselines: import-jaccard, bm25


def load(path):
    summary = Path(path) / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else None


def contrastive_row(summary):
    best = summary["best_val_metrics"]
    trained = [h for h in summary["history"] if h["epoch"] >= 0]
    return {
        "mrr": best["mrr"], "recall@10": best["recall@10"],
        "package_mrr": best["package_mean"]["mrr"],
        "start_mrr": summary["start_val_metrics"]["mrr"],
        "best_epoch": summary["best_epoch"], "epochs": len(trained),
        "by_package_epoch": summary.get("best_epoch_by_package_mrr"),
        "improving_at_end": summary["best_epoch"] == trained[-1]["epoch"],
        "peak_gib": summary.get("peak_reserved_gib"),
    }


def collect(root="models", seeds=SEEDS):
    mlm = load(Path(root) / "b2-mlm")
    groups = {}
    for name, prefix in (("B3 mlm init", "b3-con-mlm-seed"),
                         ("B4 random init", "b4-con-random-seed")):
        groups[name] = {seed: (contrastive_row(s) if (s := load(
            Path(root) / f"{prefix}-{seed}")) else None) for seed in seeds}
    return mlm, groups


def verdict(groups):
    rows = {name: [r for r in runs.values() if r] for name, runs in groups.items()}
    b3, b4 = rows["B3 mlm init"], rows["B4 random init"]
    if len(b3) < 2 or len(b4) < 2:
        return None
    spread = max(max(r["mrr"] for r in g) - min(r["mrr"] for r in g) for g in (b3, b4))
    margin = max(0.005, spread)
    gain = fmean(r["mrr"] for r in b3) - fmean(r["mrr"] for r in b4)
    package_gain = (fmean(r["package_mrr"] for r in b3)
                    - fmean(r["package_mrr"] for r in b4))
    if gain > margin and package_gain >= -margin:
        result = "MLM helps: keep it"
    elif -gain > margin:
        result = "MLM hurts: drop it, and E1/E2 with it"
    else:
        result = "no measured difference: drop MLM, and E1/E2 with it"
    complete = len(b3) == len(b4) == len(SEEDS)
    return {"spread": spread, "margin": margin, "gain": gain,
            "package_gain": package_gain, "result": result, "complete": complete}


def show(mlm, groups):
    if mlm:
        epochs = [h for h in mlm["history"] if h["epoch"] >= 0]
        improving = mlm["best_epoch"] == epochs[-1]["epoch"]
        print(f"B2 MLM: start val loss {mlm['start_val_loss']:.4f}, "
              f"best {mlm['best_val_loss']:.4f} at epoch {mlm['best_epoch']} "
              f"of {len(epochs)}{'  (still improving at the end)' if improving else ''}")
    else:
        print("B2 MLM: not finished")

    print(f"\n{'group':16} {'seed':>4} {'start':>6} {'MRR':>6} {'pkg':>6} {'R@10':>6} "
          f"{'epoch':>7} {'by pkg':>6} {'GiB':>5}")
    for name, runs in groups.items():
        for seed, r in runs.items():
            if r is None:
                print(f"{name:16} {seed:>4}  not finished")
                continue
            flag = "*" if r["improving_at_end"] else " "
            print(f"{name:16} {seed:>4} {r['start_mrr']:6.3f} {r['mrr']:6.3f} "
                  f"{r['package_mrr']:6.3f} {r['recall@10']:6.3f} "
                  f"{r['best_epoch']:>3}/{r['epochs']:<2}{flag} "
                  f"{r['by_package_epoch']!s:>6} {r['peak_gib'] or 0:5.2f}")
    print("  * best epoch was the last: still improving when training stopped")

    finished = [r for runs in groups.values() for r in runs.values() if r]
    if finished:
        disagree = sum(r["by_package_epoch"] != r["best_epoch"] for r in finished)
        above = sum(r["mrr"] > BAR["mrr"] and r["recall@10"] > BAR["recall@10"]
                    for r in finished)
        print(f"\npackage mean would keep another epoch in {disagree} "
              f"of {len(finished)} runs")
        print(f"above the val bar (MRR > {BAR['mrr']}, recall@10 > {BAR['recall@10']}) "
              f"in {above} of {len(finished)} runs")

    v = verdict(groups)
    if v is None:
        print("\nverdict: needs at least two finished runs in both B3 and B4")
        return
    print(f"\nseed spread S = {v['spread']:.4f}   margin = {v['margin']:.4f}")
    print(f"B3 - B4: MRR {v['gain']:+.4f}   package MRR {v['package_gain']:+.4f}")
    print(f"verdict: {v['result']}"
          f"{'' if v['complete'] else '   (PROVISIONAL: not all seeds finished)'}")


def main():
    show(*collect())
    return 0


if __name__ == "__main__":
    sys.exit(main())
