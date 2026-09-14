"""Record what a measurement measured, so the number survives the terminal.

A metric printed to a screen is a rumour. Six weeks from now the ablation
table needs to say what the encoder scored and what it beat, and the honest
version of that sentence has to name the corpus, the code, and the split it
was measured on - otherwise two numbers from different months get compared
as though they were the same experiment.

So every run of the evaluation opens a run row, which already carries the git
commit and a timestamp, and writes its numbers against it. Added to that is a
fingerprint of the dataset: the packages, their split, and how many rows each
contributed. Any of those changing changes the fingerprint, which is what
makes "these two numbers are comparable" a checkable claim rather than a
recollection.

The numbers themselves are a projection, not evidence: they can be recomputed
from the corpus at any time. What cannot be recomputed is the circumstance -
which commit, which corpus, which day - and that is what is being kept.
"""

import hashlib
import json
from collections import defaultdict

from elenchus.corpus.dataset import candidate_rows, eligible_rows, load_splits


def dataset_fingerprint(conn, rows=None):
    """Return a short hash identifying the dataset a measurement used.

    Built from the packages, their split, and their row counts. Two runs
    sharing a fingerprint were asked the same questions; two that do not were
    not, however similar their numbers look.
    """
    assignment = load_splits(conn)

    counts = defaultdict(int)
    for row in eligible_rows(conn, rows):
        counts[row["package"]] += 1

    material = json.dumps(
        [
            [package, assignment.get(package, "-"), counts[package]]
            for package in sorted(counts)
        ],
        separators=(",", ":"),
    )

    return hashlib.sha256(material.encode()).hexdigest()[:16]


def record(conn, run_id, task, results):
    """Store one measurement per method for this run.

    task describes the question asked - split, query level, pool level - and
    results maps method name to its metrics.
    """
    rows = []
    for method, metrics in results.items():
        rows.append((
            run_id,
            method,
            task["split"],
            task["query_opt"],
            task["pool_opt"],
            metrics.get("queries", 0),
            metrics.get("pool", 0),
            json.dumps(metrics),
        ))

    with conn:
        conn.executemany(
            "INSERT INTO measurements "
            "(run_id, method, split, query_opt, pool_opt, n_queries, n_pool, "
            " metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            rows,
        )

    return len(rows)


def history(conn, split=None, method=None, limit=40):
    """Return past measurements, newest first."""
    clauses = []
    params = []
    if split:
        clauses.append("m.split = ?")
        params.append(split)
    if method:
        clauses.append("m.method = ?")
        params.append(method)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    return conn.execute(
        """
        SELECT m.id, m.method, m.split, m.query_opt, m.pool_opt,
               m.n_queries, m.n_pool, m.metrics, m.created_at,
               r.code_version, r.params
        FROM measurements m
        JOIN runs r ON r.id = m.run_id
        """
        + where
        + " ORDER BY m.id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()


def short_commit(code_version):
    """Abbreviate a commit, keeping the mark that says it was uncommitted.

    code_version() appends -dirty when the working tree had changes, and that
    suffix is the most important part of the string: it means the exact code
    behind this number is not in git and the run cannot be reproduced. Shown
    as a trailing asterisk so it survives abbreviation instead of being
    truncated into a stray dash.
    """
    if not code_version:
        return "?"

    dirty = code_version.endswith("-dirty")
    base = code_version[: -len("-dirty")] if dirty else code_version
    return base[:7] + ("*" if dirty else "")


def format_history(rows):
    """Render measurement history, newest first, one line each."""
    if not rows:
        return ["no measurements recorded yet"]

    lines = [
        f"  {'when':<20} {'commit':<9} {'corpus':<17} {'task':<14} "
        f"{'method':<16} {'mrr':>7} {'r@1':>7}",
        "  " + "-" * 96,
    ]

    for row in rows:
        metrics = json.loads(row["metrics"])
        params = json.loads(row["params"] or "{}")
        task = f"{row['split']} {row['query_opt']}->{row['pool_opt']}"

        lines.append(
            f"  {row['created_at']:<20} "
            f"{short_commit(row['code_version']):<9} "
            f"{params.get('dataset', '?')[:16]:<17} "
            f"{task:<14} {row['method']:<16} "
            f"{metrics.get('mrr', 0):>7.3f} {metrics.get('recall@1', 0):>7.3f}"
        )

    lines.append("")
    lines.append("  * uncommitted code: the exact run cannot be reproduced")

    return lines


def task_params(conn, split, query_opt, pool_opt, seed, rows=None):
    """Return the params recorded on the run, enough to repeat it."""
    rows = candidate_rows(conn) if rows is None else rows
    return {
        "split": split,
        "query_opt": query_opt,
        "pool_opt": pool_opt,
        "seed": seed,
        "dataset": dataset_fingerprint(conn, rows),
    }
