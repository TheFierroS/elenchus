"""E3 results: measure every run's last checkpoint and apply the rule.

The protocol (docs/experiments.md, E3) was written before the runs:

  M = max(0.005, 2 * |seed 0 - seed 1| of the uniform runs' O0->O3 query MRR)
  a share P becomes the default only if, for both seeds,
    1. its O0->O3 MRR over queries beats uniform by more than M,
    2. its O0->O3 MRR over packages is not below uniform by more than M,
    3. its O0->O2 and O1->O3 MRR are not below uniform by more than M;
  if several qualify, the highest O0->O3 query MRR averaged over the two
  seeds; if none, uniform stays.

The last checkpoint (last.pt) is measured, not best.pt: under a fixed step
budget every run is read at the same point, and choosing the best of more
epochs would give some runs more chances.

Run from the repo root when experiments/run_e3.sh has finished:
  python experiments/e3_results.py
"""

import json
import os
import sys
from pathlib import Path

import torch

from elenchus.corpus.dataset import candidate_rows
from elenchus.db import connect
from elenchus.encoder.train import measure_checkpoint
from elenchus.encoder.vocab import Vocab
from elenchus.evaluation.task import retrieval_task

SHARES = ["none", "0.25", "0.5", "1.0"]
SEEDS = [0, 1]
TASKS = [("O0", "O3"), ("O0", "O2"), ("O1", "O3")]
STEPS = 1500
BATCH = 64


def run_dir(root, share, seed):
    return Path(root) / f"e3-share-{share}-seed-{seed}"


def measure_runs(conn, vocab, root="models", device=None, shares=SHARES, seeds=SEEDS,
                 steps=STEPS, batch=BATCH):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = candidate_rows(conn)
    tasks = {pair: retrieval_task(conn, split="val", query_opt=pair[0],
                                  pool_opt=pair[1], rows=rows) for pair in TASKS}
    results = {}
    for share in shares:
        for seed in seeds:
            path = run_dir(root, share, seed)
            # summary.json is written last: a run cut short has last.pt from
            # its latest epoch but no summary, and is not a result.
            if not (path / "summary.json").exists() or not (path / "last.pt").exists():
                results[(share, seed)] = None
                continue
            summary = json.loads((path / "summary.json").read_text())
            drawn = {}
            for entry in summary["history"][1:]:
                for pair, count in entry.get("pairs", {}).items():
                    drawn[pair] = drawn.get(pair, 0) + count
            total = sum(drawn.values())
            measured = {}
            for pair, (queries, pool, gold) in tasks.items():
                measured[pair] = (measure_checkpoint(path / "last.pt", vocab, queries,
                                                     pool, gold, device=device)
                                  if queries else None)
            results[(share, seed)] = {
                "draws": total, "complete": total == steps * batch,
                "eval_pair_share": drawn.get("O0-O3", 0) / total if total else None,
                "metrics": measured}
    return results


def mrr(result, pair, mean="query"):
    metrics = result["metrics"].get(pair)
    if metrics is None or not metrics.get("queries"):
        return None
    return metrics["mrr"] if mean == "query" else metrics["package_mean"]["mrr"]


def verdict(results, shares=SHARES, seeds=SEEDS):
    """Apply the protocol. Returns (margin, {share: reasons failed}, chosen)."""
    uniform = [results.get(("none", seed)) for seed in seeds]
    if any(run is None for run in uniform):
        return None, {}, None
    spread = abs(mrr(uniform[0], ("O0", "O3")) - mrr(uniform[1], ("O0", "O3")))
    margin = max(0.005, 2 * spread)
    failed = {}
    for share in shares[1:]:
        reasons = []
        for seed, base in zip(seeds, uniform):
            run = results.get((share, seed))
            if run is None:
                reasons.append(f"seed {seed} missing")
                continue
            if not mrr(run, ("O0", "O3")) - mrr(base, ("O0", "O3")) > margin:
                reasons.append(f"seed {seed}: O0->O3 query gain not above M")
            package, base_package = (mrr(run, ("O0", "O3"), "package"),
                                     mrr(base, ("O0", "O3"), "package"))
            if package < base_package - margin:
                reasons.append(f"seed {seed}: O0->O3 package MRR drops more than M")
            for pair in (("O0", "O2"), ("O1", "O3")):
                value = mrr(run, pair)
                if value is not None and value < mrr(base, pair) - margin:
                    reasons.append(f"seed {seed}: {pair[0]}->{pair[1]} drops more than M")
        failed[share] = reasons
    qualified = [s for s, reasons in failed.items() if not reasons]

    def mean_mrr(share):
        values = [mrr(results[(share, seed)], ("O0", "O3")) for seed in seeds]
        return sum(values) / len(values)

    chosen = max(qualified, key=mean_mrr) if qualified else "none"
    return margin, failed, chosen


def show(results):
    def cell(value):
        return f"{value:.4f}" if value is not None else "     -"

    print(f"{'share':6} {'seed':4} {'O0-O3 drawn':>11} {'complete':>8} "
          f"{'O0>O3 q':>8} {'O0>O3 pkg':>9} {'O0>O2 q':>8} {'O1>O3 q':>8}")
    for (share, seed), run in results.items():
        if run is None:
            print(f"{share:6} {seed:<4} {'not finished':>11}")
            continue
        print(f"{share:6} {seed:<4} {run['eval_pair_share']:>11.1%} "
              f"{str(run['complete']):>8} {cell(mrr(run, ('O0', 'O3'))):>8} "
              f"{cell(mrr(run, ('O0', 'O3'), 'package')):>9} "
              f"{cell(mrr(run, ('O0', 'O2'))):>8} {cell(mrr(run, ('O1', 'O3'))):>8}")
    margin, failed, chosen = verdict(results)
    if margin is None:
        print("\nverdict: the uniform runs are not both finished")
        return
    print(f"\nM = {margin:.4f}")
    for share, reasons in failed.items():
        status = "QUALIFIES" if not reasons else "fails: " + "; ".join(reasons)
        print(f"  {share:5} {status}")
    incomplete = [key for key, run in results.items() if run and not run["complete"]]
    if incomplete:
        print(f"WARNING: runs that did not take {STEPS} steps: {incomplete}")
    print(f"verdict: default becomes {chosen}")


def main():
    conn = connect(os.environ["ELENCHUS_DB"])
    show(measure_runs(conn, Vocab.load("models/vocab.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
