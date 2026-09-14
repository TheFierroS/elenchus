"""Re-read ground truth from the debug twins already on disk.

Ground truth is read from DWARF, and DWARF does not need Ghidra. When the
reader learns to extract something it previously skipped - a source file, a
type, an inline record - the corpus can be brought up to date by reading the
debug binaries again, without recompiling anything and without a Ghidra
session. That is minutes against hours, and it is only possible because
compilation, scanning, and truth extraction were kept as separate steps.

The stripped twin keeps its function rows and its instruction listings
untouched; only the ground_truth rows attached to it are rewritten, keyed by
address as before.
"""

import time

from elenchus.corpus.store import store_ground_truth
from elenchus.db import connect, finish_run, start_run


def corpus_pairs(conn):
    """Return [(stripped_id, package, opt_level, debug_path)] for the corpus.

    A corpus binary points at its twin; the debug half is where the DWARF
    lives, and its path was recorded when it was registered.
    """
    return conn.execute("""
        SELECT stripped.binary_id AS stripped_id,
               stripped.package   AS package,
               stripped.opt_level AS opt_level,
               debug_binary.path  AS debug_path
        FROM corpus_binaries stripped
        JOIN corpus_binaries debug ON debug.binary_id = stripped.twin_id
        JOIN binaries debug_binary ON debug_binary.id = debug.binary_id
        WHERE stripped.stripped = 1 AND debug.stripped = 0
        ORDER BY stripped.package, stripped.opt_level
    """).fetchall()


def cmd_refresh_truth(args):
    """Rewrite ground truth for every corpus binary from its debug twin."""
    started = time.time()
    conn = connect(args.db)

    pairs = corpus_pairs(conn)
    if not pairs:
        print("no corpus binaries with a debug twin")
        return 1

    run_id = start_run(
        conn, "extract", tool="dwarf", params={"stage": "ground-truth-refresh"}
    )

    total = 0
    matched = 0
    with_file = 0
    failures = []

    try:
        for row in pairs:
            try:
                count, hits = store_ground_truth(
                    conn, row["stripped_id"], row["debug_path"]
                )
            except (OSError, ValueError) as exc:
                failures.append((row["package"], row["opt_level"], str(exc)[:100]))
                print(f"  {row['package']:13} -{row['opt_level']}  FAILED")
                continue

            named = conn.execute(
                "SELECT COUNT(*) AS n FROM ground_truth "
                "WHERE binary_id = ? AND decl_file IS NOT NULL",
                (row["stripped_id"],),
            ).fetchone()["n"]

            total += count
            matched += hits
            with_file += named
            print(
                f"  {row['package']:13} -{row['opt_level']}  "
                f"gt={count:4}  matched={hits:4}  with source file={named:4}",
                flush=True,
            )
    except Exception:
        finish_run(conn, run_id, "failed")
        raise

    finish_run(conn, run_id, "ok")

    print()
    print(f"ground truth   : {matched}/{total} matched")
    print(f"source file    : {with_file}/{total} resolved")
    if failures:
        for package, opt, error in failures:
            print(f"  failed {package} -{opt}: {error}")
    print(f"elapsed        : {time.time() - started:.1f}s")
    return 0
