"""One-off, read-only: eligible functions whose source path is not in their package.

Every package is unpacked under data/corpus/<package>/, so a function's DWARF
decl_file should start there. A path outside it - an absolute path into the
repository root, say - means the compiler was told a file name without its
directory (typically by #line directives in an amalgamation) and resolved it
against wherever the build ran. The identity (package, decl_file, name) then
depends on the machine, and files that share a name in different original
directories collapse into one path.

Counts only rows the dataset keeps, so every number here is one training
meets.

Run from the repo root:  python experiments/decl_paths.py
"""

import os
from collections import Counter, defaultdict

from elenchus.corpus.dataset import eligible_rows
from elenchus.db import connect


def outside_paths(rows):
    """{package: (rows, outside rows, Counter of outside paths)} for every package."""
    total = Counter()
    outside = Counter()
    paths = defaultdict(Counter)
    for row in rows:
        package = row["package"]
        total[package] += 1
        home = f"data/corpus/{package}/"
        if not (row["decl_file"] or "").startswith(home):
            outside[package] += 1
            paths[package][row["decl_file"]] += 1
    return {p: (total[p], outside[p], paths[p]) for p in total}


def show(report):
    affected = sorted((p for p, (_t, o, _c) in report.items() if o),
                      key=lambda p: -report[p][1])
    rows = sum(t for t, _o, _c in report.values())
    moved = sum(o for _t, o, _c in report.values())
    print(f"packages {len(report)}, affected {len(affected)}; "
          f"eligible rows {rows:,}, outside their package {moved:,}")
    for p in affected:
        total, out, paths = report[p]
        print(f"\n{p}: {out:,} of {total:,} rows ({out / total:.1%}), "
              f"{len(paths)} distinct paths")
        for path, n in paths.most_common(5):
            print(f"  {n:6,}  {path}")


def main():
    conn = connect(os.environ["ELENCHUS_DB"])
    show(outside_paths(eligible_rows(conn)))


if __name__ == "__main__":
    main()
