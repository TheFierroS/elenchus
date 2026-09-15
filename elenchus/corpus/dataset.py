"""Decide what may be trained on, and split it so the metrics mean something.

Three things stand between a corpus and a usable dataset.

*Provenance.* Every binary here was linked against the MinGW runtime, so
every binary contains the same DllMain, the same pformat, the same dtoa. They
came with the toolchain rather than from the source we chose, they are
byte-identical across packages, and they make up a quarter of the ground
truth. Train on one package and test on another and the model meets them
again in the test set having already memorised them. Whatever score that
produces is not a measurement.

*Duplication.* The same reasoning applies to any source file two packages
happen to share, toolchain or not - a vendored dependency, a copied helper.
The rule is about the file being shared, not about whose file it is.

*Splitting by package, never by function.* The same source function compiled
at -O0 and at -O3 is two rows. Split rows at random and one lands in train
and the other in test, which is the exact pair the encoder is being trained
to recognise. The whole measurement collapses, and it collapses upward: the
numbers look excellent. So the unit of splitting is the package, and every
row belonging to a package moves together.

None of this can be verified by reading the code, so leakage() checks the
result directly: it looks for any function whose normalised form appears in
more than one split, and reports what it finds.
"""

import hashlib
from collections import defaultdict

from elenchus.encoder.normalise import normalise

SPLITS = ("train", "val", "test")
DEFAULT_SHARES = (0.70, 0.15, 0.15)

# Path fragments that mark a source file as belonging to the toolchain rather
# than to a package we chose. A second, path-independent rule (a file seen in
# more than one package) runs alongside this one, because these fragments
# describe today's MinGW on today's machine and will not describe tomorrow's.
TOOLCHAIN_MARKERS = (
    "/mingw-w64-crt/",
    "/x86_64-w64-mingw32/",
    "/lib/gcc/",
    "/gcc-",
    # Debian and Ubuntu install the MinGW headers here, and a static inline
    # from one of them (a stdio wrapper, a time helper) gets a DWARF entry in
    # whichever package happens to call it. Seen in only one package, the
    # shared-file rule cannot catch it, so the path has to.
    "/mingw-w64/include/",
)


def source_files(conn):
    """Return [(decl_file, package_count, row_count)] over all ground truth."""
    return conn.execute("""
        SELECT gt.decl_file                AS decl_file,
               COUNT(DISTINCT cb.package)  AS packages,
               COUNT(*)                    AS rows_
        FROM ground_truth gt
        JOIN corpus_binaries cb ON cb.binary_id = gt.binary_id
        WHERE gt.decl_file IS NOT NULL
        GROUP BY gt.decl_file
    """).fetchall()


def is_toolchain_path(path):
    """True if a source path looks like it came with the compiler."""
    return any(marker in path for marker in TOOLCHAIN_MARKERS)


def excluded_files(conn):
    """Return {decl_file: reason} for source files that may not be trained on.

    Both rules are applied and the reason is kept, so a file dropped by only
    one of them is visible. If the two ever disagree badly that is worth
    knowing: it means either the corpus grew a genuinely shared dependency,
    or the path markers have gone stale.
    """
    excluded = {}
    for row in source_files(conn):
        path = row["decl_file"]
        shared = row["packages"] > 1
        toolchain = is_toolchain_path(path)

        if shared and toolchain:
            excluded[path] = "shared+toolchain"
        elif shared:
            excluded[path] = "shared"
        elif toolchain:
            excluded[path] = "toolchain"

    return excluded


MIN_INSTRUCTIONS = 8


def candidate_rows(conn):
    """Return ground-truth functions that survived the source-file rules.

    This is the dataset before deduplication: named by the compiler, found in
    the stripped binary, extracted, and not from an excluded source file.
    """
    dropped = set(excluded_files(conn))

    rows = conn.execute("""
        SELECT gt.function_id        AS function_id,
               gt.name               AS name,
               gt.decl_file          AS decl_file,
               cb.package            AS package,
               cb.opt_level          AS opt_level,
               fc.listing            AS listing,
               fc.n_instructions     AS n_instructions
        FROM ground_truth gt
        JOIN corpus_binaries cb  ON cb.binary_id = gt.binary_id
        JOIN function_code fc    ON fc.function_id = gt.function_id
        WHERE gt.function_id IS NOT NULL AND gt.decl_file IS NOT NULL
    """).fetchall()

    return [
        row for row in rows
        if row["decl_file"] not in dropped
        and row["n_instructions"] >= MIN_INSTRUCTIONS
    ]


def duplicate_keys(rows):
    """Return normalised forms that occur in more than one package.

    Excluding a source file catches the runtime, but not a function that
    simply happens to compile to the same thing somewhere else. A one-line
    wrapper - set up a frame, forward the arguments, call, return - looks
    identical in every library that has one, and there is one in every
    library. Whichever split such a function lands in, its twin lands
    elsewhere, and the model meets in the test set something it memorised
    in training.

    Dropping them costs little. They carry almost no information: if two
    functions are indistinguishable after normalisation, no encoder reading
    normalised text could tell them apart, so keeping them would only teach
    it to guess.
    """
    packages = defaultdict(set)
    for row in rows:
        packages[content_key(row["listing"])].add(row["package"])

    return {key for key, seen in packages.items() if len(seen) > 1}


def eligible_rows(conn, rows=None):
    """Return the rows that may be trained on, deduplicated across packages.

    Normalising every function costs a second or two, so callers that need
    both the dataset and the leakage check should compute the candidates
    once and pass them in.
    """
    rows = candidate_rows(conn) if rows is None else rows
    duplicates = duplicate_keys(rows)

    return [row for row in rows if content_key(row["listing"]) not in duplicates]


def package_sizes(rows):
    """Return {package: eligible row count}."""
    sizes = defaultdict(int)
    for row in rows:
        sizes[row["package"]] += 1
    return dict(sizes)


DOMAINS = ("compression", "crypto", "data-format", "text", "interpreter",
           "systems", "media")


def assign_splits(sizes, shares=DEFAULT_SHARES, domains=None):
    """Assign whole packages to train/val/test.

    With domains ({package: domain}), each domain is split on its own, so
    every kind of code is represented in every split, and the split is
    stable: a change in one package's size can only move packages of its own
    domain.

    Both properties were missing without it. Splitting all packages in one
    pass by size, a 1.4% change in row counts after a rescan moved 32 of 48
    packages, and took the only crypto package left in test back into train
    - so what the exam covered was decided by coincidence.

    Without domains the whole corpus is one group, which is the old
    behaviour. A package with no domain goes into its own "unlabelled" group
    rather than being silently mixed into another.

    Measured on the 48-package corpus: 70.5 / 16.2 / 13.3% of rows, every
    domain in every split. Under +-3% noise on every package's row count, at
    most 12 packages moved, against 29 when splitting the corpus in one pass.
    """
    if domains is None:
        return _assign_by_size(sizes, shares)

    groups = defaultdict(dict)
    for package, size in sizes.items():
        groups[domains.get(package) or "unlabelled"][package] = size

    # Shared across domains: which split is furthest behind overall. Inside a
    # small domain the last packages are placed only to make sure every split
    # gets one, and without this the same split would take the smallest
    # package every time - test ended at 12% of rows instead of 15%.
    total = sum(sizes.values())
    ledger = {
        "targets": {name: total * share for name, share in zip(SPLITS, shares)},
        "assigned": {name: 0 for name in SPLITS},
    }

    result = {}
    for domain in sorted(groups):
        result.update(_assign_by_size(groups[domain], shares, ledger))
    return result


def _assign_by_size(sizes, shares=DEFAULT_SHARES, ledger=None):
    """Assign whole packages to train/val/test, balancing by row count.

    Packages differ in size by more than an order of magnitude here, so a
    random assignment would routinely put most of the corpus on one side.
    Largest first, each package goes to whichever split is furthest below its
    target - the same greedy rule used for bin packing, and deterministic, so
    the split is a property of the corpus rather than of the day it was run.

    Every split is guaranteed at least one package: an empty test set would
    make the exit criterion unmeasurable rather than merely unbalanced.
    """
    total = sum(sizes.values())
    targets = {name: total * share for name, share in zip(SPLITS, shares)}
    assigned = {name: 0 for name in SPLITS}
    result = {}

    ordered = sorted(sizes.items(), key=lambda item: (-item[1], item[0]))

    for index, (package, size) in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [name for name in SPLITS if not any(
            value == name for value in result.values()
        )]

        # Once only as many packages remain as there are empty splits, the
        # rest are spoken for: fill the empty ones rather than keep balancing.
        if empty and remaining <= len(empty):
            if ledger is None:
                choice = empty[0]
            else:
                choice = max(empty, key=lambda name: (
                    ledger["targets"][name] - ledger["assigned"][name]))
        else:
            choice = max(SPLITS, key=lambda name: targets[name] - assigned[name])

        result[package] = choice
        assigned[choice] += size
        if ledger is not None:
            ledger["assigned"][choice] += size

    return result


def training_runs(conn):
    """Return runs that trained a model on the recorded split.

    Once a model has learnt from the train packages, the split is part of
    that model. Moving a package from train to test afterwards would put
    functions the model has studied into its exam, and every number measured
    from then on would be inflated without any error being raised.
    """
    return conn.execute(
        "SELECT id, started_at, status FROM runs WHERE kind = 'train' ORDER BY id"
    ).fetchall()


def store_splits(conn, assignment):
    """Record the split each package belongs to, replacing any earlier one."""
    with conn:
        conn.execute("DELETE FROM dataset_split")
        conn.executemany(
            "INSERT INTO dataset_split (package, split) VALUES (?, ?)",
            sorted(assignment.items()),
        )


def load_splits(conn):
    """Return {package: split} as recorded, or an empty dict if never set."""
    return {
        row["package"]: row["split"]
        for row in conn.execute("SELECT package, split FROM dataset_split")
    }


def content_key(listing):
    """Hash a function's normalised form.

    Not the raw bytes: those still carry absolute addresses, so the same
    function in two binaries hashes differently and the duplicate slips
    through. The normalised token sequence is what the encoder actually sees,
    which makes it the right thing to ask about when the question is whether
    the model has met this function before.
    """
    return tokens_key(normalise(listing))


def tokens_key(tokens):
    """Hash an already normalised token sequence.

    Split out of content_key so a caller holding tokens does not normalise
    twice - and, more to the point, so that the dataset's dedup and the
    encoder's false-negative mask cannot drift into two notions of "the same
    code".
    """
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def cross_split_collisions(rows, assignment):
    """Return normalised forms that occur in more than one split.

    Each finding names every package involved and the split it sits in: a
    collision is between two packages, and reporting only one of them makes
    the result unreadable - the first version of this printed one name and
    made it look as though a package had been split in half.
    """
    members = defaultdict(dict)

    for row in rows:
        split = assignment.get(row["package"])
        if split is None:
            continue
        members[content_key(row["listing"])].setdefault(
            (row["package"], split), row["name"]
        )

    found = []
    for key, seen in members.items():
        if len({split for _, split in seen}) > 1:
            found.append((
                key[:12],
                sorted(
                    f"{name} ({package}/{split})"
                    for (package, split), name in seen.items()
                ),
            ))

    return sorted(found)


def leakage(conn, assignment=None, rows=None):
    """Check the finished dataset for functions shared between splits.

    Measured after deduplication, so it should always be empty - and that is
    the point. A check that can only pass is not a check, which is why the
    command reports the collisions found *before* deduplication alongside
    this one: that number says how much the filter actually prevented, and
    this one says the filter ran.

    A finding here means the filter was bypassed or the split came from
    somewhere other than assign_splits. It is a defect, not a threshold.
    """
    assignment = assignment or load_splits(conn)
    if not assignment:
        return []

    return cross_split_collisions(eligible_rows(conn, rows), assignment)


def report(conn, assignment, rows=None):
    """Return the lines summarising a dataset: what was dropped and why."""
    all_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM ground_truth WHERE function_id IS NOT NULL"
    ).fetchone()["n"]

    dropped = excluded_files(conn)
    candidates = candidate_rows(conn) if rows is None else rows
    eligible = eligible_rows(conn, candidates)
    sizes = package_sizes(eligible)
    total = sum(sizes.values())

    reasons = defaultdict(int)
    for reason in dropped.values():
        reasons[reason] += 1

    lines = [
        f"ground truth      : {all_rows} matched functions",
        f"excluded files    : {len(dropped)} "
        f"({dict(sorted(reasons.items()))})",
        f"after exclusion   : {len(candidates)} rows "
        f"(>= {MIN_INSTRUCTIONS} instructions)",
        f"cross-package dup : {len(candidates) - len(eligible)} rows dropped",
        f"eligible          : {total} rows "
        f"({100 * total // all_rows if all_rows else 0}% of ground truth)",
        "",
        f"  {'package':<14} {'rows':>6}  split",
    ]

    for package, size in sorted(sizes.items(), key=lambda i: -i[1]):
        lines.append(
            f"  {package:<14} {size:>6}  {assignment.get(package, '-')}"
        )

    prevented = cross_split_collisions(candidates, assignment)
    lines.append("")
    lines.append(
        f"  collisions prevented by dedup: {len(prevented)} "
        f"normalised forms spanning splits"
    )

    lines.append("")
    for name in SPLITS:
        size = sum(s for p, s in sizes.items() if assignment.get(p) == name)
        packages = sum(1 for p in sizes if assignment.get(p) == name)
        share = 100 * size / total if total else 0
        lines.append(
            f"  {name:<6} {packages} packages  {size:>6} rows  ({share:.0f}%)"
        )

    return lines
