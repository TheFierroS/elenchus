"""One-off, read-only: the dataset as training sees it, split by split.

Answers "is the data good, and is there enough of it?" with counts instead
of impressions. Everything is read through load_examples, the same path the
encoder trains on, so every number here is a number the model meets.

  identities   distinct functions (package, source file, name)
  pairable     identities with >= 2 levels: usable by contrastive training
  O0+O3        identities the retrieval task can ask about
  tokens       normalised length; contrastive cuts at 1023 after [CLS]

Run from the repo root:  python experiments/corpus_report.py
"""

import os
import statistics
from collections import Counter, defaultdict

from elenchus.corpus.build import load_manifest
from elenchus.db import connect
from elenchus.encoder.data import load_examples

SPLITS = ("train", "val", "test")
LEVELS = ("O0", "O1", "O2", "O3")
CUT = 1023


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def split_report(examples, domain_of):
    levels = defaultdict(set)
    for e in examples:
        levels[e.identity].add(e.opt_level)
    lengths = sorted(len(e.tokens) for e in examples)
    per_package = Counter(identity[0] for identity in levels)
    per_domain = Counter(domain_of.get(identity[0], "unlabelled") for identity in levels)
    pairs = Counter()
    for found in levels.values():
        for i, a in enumerate(LEVELS):
            for b in LEVELS[i + 1:]:
                if a in found and b in found:
                    pairs[f"{a}-{b}"] += 1
    return {
        "packages": len(per_package),
        "rows": len(examples),
        "identities": len(levels),
        "pairable": sum(1 for found in levels.values() if len(found) >= 2),
        "O0+O3": sum(1 for found in levels.values() if {"O0", "O3"} <= found),
        "unique contents": len({e.content for e in examples}),
        "no import": sum(1 for e in examples
                         if not any(t.startswith("IMPORT:") for t in e.tokens)),
        "tokens p50/p90/p99/max": (percentile(lengths, 0.5),
                                   percentile(lengths, 0.9),
                                   percentile(lengths, 0.99),
                                   lengths[-1] if lengths else 0),
        "tokens mean": statistics.fmean(lengths) if lengths else 0.0,
        "cut at 1023": sum(1 for n in lengths if n > CUT),
        "per_package": per_package,
        "per_domain": per_domain,
        "pairs": pairs,
    }


def corpus_report(conn, domain_of):
    return {split: split_report(load_examples(conn, split), domain_of)
            for split in SPLITS}


def show(reports):
    def pct(part, whole):
        return f"{part:>7,} ({part / whole:6.1%})" if whole else f"{part:>7,}"

    print(f"{'':24}" + "".join(f"{s:>20}" for s in SPLITS))
    for key in ("packages", "rows", "identities", "unique contents"):
        print(f"{key:24}" + "".join(f"{reports[s][key]:>20,}" for s in SPLITS))
    for key, base in (("pairable", "identities"), ("O0+O3", "identities"),
                      ("no import", "rows"), ("cut at 1023", "rows")):
        print(f"{key:24}" + "".join(f"{pct(reports[s][key], reports[s][base]):>20}"
                                    for s in SPLITS))
    print(f"{'tokens mean':24}" + "".join(f"{reports[s]['tokens mean']:>20.0f}"
                                          for s in SPLITS))
    spread = {s: "/".join(map(str, reports[s]["tokens p50/p90/p99/max"]))
              for s in SPLITS}
    print(f"{'tokens p50/p90/p99/max':24}" + "".join(f"{spread[s]:>20}" for s in SPLITS))

    print("\nidentities per domain")
    domains = sorted(set().union(*(reports[s]["per_domain"] for s in SPLITS)))
    for d in domains:
        print(f"  {d:22}" + "".join(
            f"{pct(reports[s]['per_domain'][d], reports[s]['identities']):>20}"
            for s in SPLITS))

    print("\nlargest packages (share of the split's identities)")
    for s in SPLITS:
        top = reports[s]["per_package"].most_common(5)
        whole = reports[s]["identities"]
        listed = ", ".join(f"{p} {n / whole:.0%}" for p, n in top)
        print(f"  {s:6} {listed}")

    print("\nlevel pairs available in train (identities having both)")
    t = reports["train"]
    print("  " + "  ".join(f"{k} {v:,}" for k, v in sorted(t["pairs"].items())))


def main():
    conn = connect(os.environ["ELENCHUS_DB"])
    domain_of = {p.name: p.domain for p in load_manifest("elenchus/corpus/manifest.toml")}
    show(corpus_report(conn, domain_of))


if __name__ == "__main__":
    main()
