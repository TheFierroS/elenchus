"""Remove duplicate corpus registrations left by an interrupted rebuild.

A compiled PE carries a timestamp, so building the same source twice gives
two files with different hashes. The database, reasonably, treats them as two
binaries. The corpus then holds the same package at the same optimisation
level twice, and every function in it counts twice: in training, in the
retrieval pool, and in every measurement over either.

Nothing is wrong with the second copy - it is a perfectly good binary. The
problem is having both. So this keeps the newest registration of each
(package, level, stripped) triple and unregisters the rest, which is the
version whose paths still exist on disk after the most recent build.

Unregistering, not deleting. The binaries, their functions and their listings
stay where they are; only their membership in the corpus is withdrawn.
Deleting rows an event log points at would break the log, and the log is the
one thing in this project that is never rewritten.
"""


def duplicate_registrations(conn):
    """Return corpus_binaries rows that repeat a (package, level, side).

    Newest kept, older ones returned. Newest by binary id, which rises with
    insertion, so the survivor is the one the last build produced.
    """
    return conn.execute("""
        SELECT cb.binary_id, cb.package, cb.opt_level, cb.stripped
        FROM corpus_binaries cb
        WHERE cb.binary_id < (
            SELECT MAX(other.binary_id)
            FROM corpus_binaries other
            WHERE other.package = cb.package
              AND other.opt_level = cb.opt_level
              AND other.stripped = cb.stripped
        )
        ORDER BY cb.package, cb.opt_level, cb.binary_id
    """).fetchall()


def unregister(conn, binary_ids):
    """Withdraw binaries from the corpus, leaving every other row intact.

    Twin links pointing at a withdrawn binary are cleared too, so no
    surviving row is left referring to something no longer in the corpus.
    """
    if not binary_ids:
        return 0

    marks = ",".join("?" * len(binary_ids))
    with conn:
        conn.execute(
            f"UPDATE corpus_binaries SET twin_id = NULL "
            f"WHERE twin_id IN ({marks})",
            binary_ids,
        )
        cur = conn.execute(
            f"DELETE FROM corpus_binaries WHERE binary_id IN ({marks})",
            binary_ids,
        )

    return cur.rowcount


def cmd_prune_corpus(args):
    """Report duplicate corpus registrations, and drop them with --apply."""
    from elenchus.db import connect

    conn = connect(args.db)
    duplicates = duplicate_registrations(conn)

    if not duplicates:
        print("no duplicate registrations")
        return 0

    print(f"{len(duplicates)} duplicate registration(s):")
    for row in duplicates:
        side = "stripped" if row["stripped"] else "debug"
        print(f"  binary {row['binary_id']:5}  {row['package']:14} "
              f"-{row['opt_level']}  {side}")

    if not args.apply:
        print()
        print("Nothing changed. Pass --apply to unregister these.")
        return 0

    removed = unregister(conn, [row["binary_id"] for row in duplicates])
    print()
    print(f"unregistered {removed} binaries "
          f"(their functions and listings are untouched)")
    return 0
