"""Known gaps in the corpus: levels that are missing on purpose, and why.

Some levels cannot be stored. quickjs -O1 exhausts Ghidra's heap at 4 GB and
faster at 6 GB, while its other three levels scan normally. Left alone, that
one level makes every `elenchus check` warn and every `build-corpus` retry a
scan that is known to fail.

A warning that is always on is worse than none: it trains the eye to skip the
warnings line, and the next real problem - a package cut short by a crash -
reads the same as the familiar one. So a gap that has been looked at and
accepted is recorded, with its reason, and from then on it is reported as
known rather than warned about, and builds do not retry it.

A gap is a claim about the corpus, and the database checks it. A level cannot
be recorded as missing while it is stored, and if it is stored later the
record is flagged as stale rather than silently left to lie.
"""

from elenchus.corpus.build import OPT_LEVELS


class GapError(ValueError):
    """A gap record that would be untrue or cannot be changed as asked."""


def known_gaps(conn):
    """Return {package: {opt_level: reason}} for every recorded gap."""
    gaps = {}
    for row in conn.execute(
        "SELECT package, opt_level, reason FROM corpus_gaps "
        "ORDER BY package, opt_level"
    ):
        gaps.setdefault(row["package"], {})[row["opt_level"]] = row["reason"]
    return gaps


def stored_levels(conn, package):
    """Return the levels of package stored with both twins."""
    return {
        row["opt_level"]
        for row in conn.execute(
            "SELECT opt_level FROM corpus_binaries WHERE package = ? "
            "GROUP BY opt_level HAVING COUNT(DISTINCT stripped) = 2",
            (package,),
        )
    }


def record_gap(conn, package, opt_level, reason):
    """Record that one level of a package is missing on purpose."""
    if opt_level not in OPT_LEVELS:
        raise GapError(f"unknown level {opt_level!r}; expected one of {OPT_LEVELS}")

    reason = (reason or "").strip()
    if not reason:
        raise GapError("a gap needs a reason; it is the only record of the decision")

    in_corpus = conn.execute(
        "SELECT 1 FROM corpus_binaries WHERE package = ? LIMIT 1", (package,)
    ).fetchone()
    if in_corpus is None:
        raise GapError(f"{package} has nothing in the corpus; there is no gap to record")

    if opt_level in stored_levels(conn, package):
        raise GapError(f"{package} -{opt_level} is stored; it is not missing")

    existing = conn.execute(
        "SELECT reason FROM corpus_gaps WHERE package = ? AND opt_level = ?",
        (package, opt_level),
    ).fetchone()
    if existing is not None:
        raise GapError(
            f"{package} -{opt_level} is already recorded as a gap "
            f"({existing['reason']}); remove it first to change the reason"
        )

    with conn:
        conn.execute(
            "INSERT INTO corpus_gaps (package, opt_level, reason, recorded_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (package, opt_level, reason),
        )


def remove_gap(conn, package, opt_level):
    """Delete a gap record, so the level is warned about and retried again."""
    with conn:
        cur = conn.execute(
            "DELETE FROM corpus_gaps WHERE package = ? AND opt_level = ?",
            (package, opt_level),
        )
    if cur.rowcount == 0:
        raise GapError(f"{package} -{opt_level} is not recorded as a gap")


def cmd_corpus_gap(args):
    """Record, remove or list known corpus gaps."""
    from elenchus.db import connect

    conn = connect(args.db)

    try:
        if args.action == "add":
            record_gap(conn, args.package, args.level, args.reason)
            print(f"recorded: {args.package} -{args.level} ({args.reason.strip()})")
        elif args.action == "remove":
            remove_gap(conn, args.package, args.level)
            print(f"removed: {args.package} -{args.level}")
    except GapError as exc:
        print(f"refused: {exc}")
        return 1

    rows = conn.execute(
        "SELECT package, opt_level, reason, recorded_at FROM corpus_gaps "
        "ORDER BY package, opt_level"
    ).fetchall()
    if args.action == "list" or rows:
        if args.action != "list":
            print()
        if not rows:
            print("no known gaps")
        for row in rows:
            print(f"  {row['package']:14} -{row['opt_level']}  "
                  f"{row['recorded_at']}  {row['reason']}")
    return 0
