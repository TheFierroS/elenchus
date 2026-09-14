"""Consistency checks over the database.

Each check returns the violations it finds - an empty list means the
invariant holds. cmd_check in the CLI runs them all and reports. These are
the database's health guarantees, and they run in CI on every commit, so a
change that quietly breaks the event log is caught before it is merged.

The invariants that depend on the claim/verdict layer (an LLM-only claim may
not sit above 'supported', a confirmed claim needs a deterministic verdict)
arrive with that layer. What can be checked today is checked today.

A note on what belongs here. These are not tests of the code - the test suite
does that. They are questions asked of real data, of the kind that only go
wrong after a crash, a half-finished run, or two processes writing at once.
Every one of them below was added because it could have caught something
that actually happened.
"""

from elenchus.entities import ENTITY_TABLES


def check_links(conn):
    """Return [(event_id, entity_kind, entity_id)] for links with no entity.

    Every event link points at an entity - a function, block, or string.
    A link to a row that does not exist is a dangling reference: the event
    claims something about an entity the database cannot produce.

    One query per entity kind, not one per link. The first version asked
    about each link separately, which is unnoticeable over a few thousand
    and takes minutes over the several million a real corpus holds - and a
    check nobody is willing to wait for is a check nobody runs.
    """
    broken = []

    kinds = [
        row["entity_kind"]
        for row in conn.execute(
            "SELECT DISTINCT entity_kind FROM event_links"
        )
    ]

    for kind in kinds:
        table = ENTITY_TABLES.get(kind)

        if table is None:
            # An entity kind with no table at all: every link using it is
            # dangling, whatever it points to.
            rows = conn.execute(
                "SELECT event_id, entity_kind, entity_id FROM event_links "
                "WHERE entity_kind = ?",
                (kind,),
            )
        else:
            rows = conn.execute(
                f"""
                SELECT el.event_id, el.entity_kind, el.entity_id
                FROM event_links el
                LEFT JOIN {table} t ON t.id = el.entity_id
                WHERE el.entity_kind = ? AND t.id IS NULL
                """,
                (kind,),
            )

        broken.extend(
            (row["event_id"], row["entity_kind"], row["entity_id"])
            for row in rows
        )

    return broken


def check_orphan_events(conn):
    """Return event ids whose run never finished (still 'running').

    An event belongs to a run. A run left 'running' means the process died
    mid-scan, so those events are suspect - a torn write, not a clean record.
    A 'failed' run is deliberate and known, so its events are not flagged here.
    """
    rows = conn.execute("""
        SELECT e.id
        FROM events e
        JOIN runs r ON r.id = e.run_id
        WHERE r.status = 'running'
    """).fetchall()
    return [row["id"] for row in rows]


def check_seq_gaps(conn):
    """Return [(binary_id, expected, found)] where seq is not 1..N unbroken.

    Events carry a per-binary sequence that must run 1, 2, 3, ... with no
    gaps and no repeats. A gap means an event was lost; a repeat means two
    were written with the same position. Either breaks the ability to replay
    a binary's history in order.
    """
    # Two passes, because the cheap one answers for almost every binary.
    # Events carry UNIQUE (binary_id, seq), so duplicates cannot exist; a
    # sequence is therefore unbroken exactly when it starts at 1 and its
    # highest value equals how many there are. That is one aggregate query
    # over the whole table instead of one scan per binary, which matters at
    # a few million events.
    suspect = conn.execute("""
        SELECT binary_id
        FROM events
        GROUP BY binary_id
        HAVING MIN(seq) != 1 OR MAX(seq) != COUNT(*)
    """).fetchall()

    violations = []

    # Only the binaries that failed are walked row by row, to say where.
    # A number without a position is hard to act on.
    for row in suspect:
        binary_id = row["binary_id"]
        seqs = conn.execute(
            "SELECT seq FROM events WHERE binary_id = ? ORDER BY seq",
            (binary_id,),
        ).fetchall()

        for expected, found in enumerate(seqs, start=1):
            if found["seq"] != expected:
                violations.append((binary_id, expected, found["seq"]))
                break

    return violations


def check_unfinished_runs(conn):
    """Return runs still marked 'running' - processes that never closed.

    A run is opened before work starts and closed after, so one left open
    means the process was killed: out of memory, a crashed VM, a terminal
    closing. Six of these accumulated across one afternoon of interrupted
    corpus builds.

    Not corruption, but not nothing either: whatever those runs were writing
    stopped mid-way, so their output is partial by definition.
    """
    return [
        (row["id"], row["kind"], row["started_at"])
        for row in conn.execute(
            "SELECT id, kind, started_at FROM runs WHERE status = 'running' "
            "ORDER BY id"
        )
    ]


def check_code_orphans(conn):
    """Return function_code rows whose function is gone or in another binary.

    The listing belongs to a function, and that function belongs to a binary.
    If the two disagree the listing has been attached to the wrong code, which
    is the silent failure that would poison training without ever raising.
    """
    return [
        (row["function_id"], row["binary_id"])
        for row in conn.execute("""
            SELECT fc.function_id, fc.binary_id
            FROM function_code fc
            LEFT JOIN functions f ON f.id = fc.function_id
            WHERE f.id IS NULL OR f.binary_id != fc.binary_id
        """)
    ]


def check_ground_truth_binding(conn):
    """Return ground truth matched to a function in a different binary.

    Ground truth is matched by address, and an address only means something
    within one binary. A row pointing at another binary's function is a
    matching error of exactly the kind the metric cannot see: the corpus
    would look fully matched and be wrong.
    """
    return [
        (row["id"], row["name"], row["binary_id"], row["other"])
        for row in conn.execute("""
            SELECT gt.id, gt.name, gt.binary_id, f.binary_id AS other
            FROM ground_truth gt
            JOIN functions f ON f.id = gt.function_id
            WHERE gt.function_id IS NOT NULL AND f.binary_id != gt.binary_id
        """)
    ]


def check_twin_links(conn):
    """Return corpus binaries whose twin is missing or on the same side.

    Every corpus binary is half of a pair: one stripped, one with debug info,
    compiled from the same source at the same level. A twin pointing at
    nothing, or at another stripped binary, means ground truth would be read
    from the wrong file.
    """
    return [
        (row["binary_id"], row["package"], row["opt_level"], row["reason"])
        for row in conn.execute("""
            SELECT cb.binary_id, cb.package, cb.opt_level,
                   CASE WHEN twin.binary_id IS NULL THEN 'twin missing'
                        ELSE 'twin on the same side' END AS reason
            FROM corpus_binaries cb
            LEFT JOIN corpus_binaries twin ON twin.binary_id = cb.twin_id
            WHERE cb.twin_id IS NOT NULL
              AND (twin.binary_id IS NULL OR twin.stripped = cb.stripped)
        """)
    ]


def check_split_packages(conn):
    """Return split assignments naming a package the corpus does not have.

    A split left over from a smaller corpus would quietly send queries to a
    package that is no longer there, and the evaluation would run on fewer
    functions than it reported.
    """
    return [
        row["package"]
        for row in conn.execute("""
            SELECT package FROM dataset_split
            WHERE package NOT IN (SELECT package FROM corpus_binaries)
        """)
    ]


def check_measurement_runs(conn):
    """Return measurements whose run row is missing.

    A number without its run has lost the commit and the corpus it came from,
    which is the whole reason it was written down.
    """
    return [
        (row["id"], row["method"])
        for row in conn.execute("""
            SELECT m.id, m.method
            FROM measurements m
            LEFT JOIN runs r ON r.id = m.run_id
            WHERE r.id IS NULL
        """)
    ]


def check_corpus_duplicates(conn):
    """Return (package, level, stripped) recorded more than once.

    A compiled binary carries a timestamp, so recompiling the same source
    produces a different file hash and registers as a new binary rather than
    matching the old one. Build a package twice and the corpus holds both,
    and every function in it is counted twice - in the training set, in the
    pool, and in any measurement over either.

    It takes an interrupted run and a restart to get here, which is exactly
    what a long build invites.
    """
    return [
        (row["package"], row["opt_level"], row["stripped"], row["n"])
        for row in conn.execute("""
            SELECT package, opt_level, stripped, COUNT(*) AS n
            FROM corpus_binaries
            GROUP BY package, opt_level, stripped
            HAVING n > 1
            ORDER BY package, opt_level
        """)
    ]


def check_corpus_completeness(conn, levels=4):
    """Return packages missing some of their twins.

    A package should hold one stripped and one debug binary at every
    optimisation level. Fewer means a run stopped part way through it, and
    the package will contribute lopsided data: present at -O0, absent at -O3,
    so its functions can never form a cross-optimisation pair.

    A warning rather than a defect - the corpus is usable, just uneven.
    """
    return [
        (row["package"], row["n"])
        for row in conn.execute(
            "SELECT package, COUNT(*) AS n FROM corpus_binaries "
            "GROUP BY package HAVING n < ? ORDER BY package",
            (2 * levels,),
        )
    ]


# Checks that report a situation rather than a defect. An unfinished run
# means a process died; that is worth seeing, but it is not a broken
# database, and failing the command over it would teach everyone to ignore
# a red result - which would cost more than the warning is worth.
WARNINGS = {"unfinished runs", "corpus completeness"}


def close_stale_runs(conn, older_than_minutes=60):
    """Mark long-open runs as failed, and return how many were closed.

    A run open for an hour belongs to a process that is not coming back. The
    age threshold matters: a run opened a minute ago may well be a corpus
    build still working, and closing it would put a lie in the record.

    They become 'failed' rather than 'ok' because that is what happened -
    the work stopped part way, and the events it wrote are partial.
    """
    with conn:
        cur = conn.execute(
            "UPDATE runs SET status = 'failed', ended_at = datetime('now') "
            "WHERE status = 'running' "
            "AND started_at <= datetime('now', ?)",
            (f"-{int(older_than_minutes)} minutes",),
        )
    return cur.rowcount


# Every check, in the order cmd_check runs them. Each entry pairs a short
# label with the function, so adding a check is one line here.
CHECKS = [
    ("dangling links", check_links),
    ("orphan events", check_orphan_events),
    ("seq gaps", check_seq_gaps),
    ("unfinished runs", check_unfinished_runs),
    ("code orphans", check_code_orphans),
    ("ground truth binding", check_ground_truth_binding),
    ("twin links", check_twin_links),
    ("split packages", check_split_packages),
    ("measurement runs", check_measurement_runs),
    ("corpus duplicates", check_corpus_duplicates),
    ("corpus completeness", check_corpus_completeness),
]
