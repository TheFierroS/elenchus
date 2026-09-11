"""Consistency checks over the database.

Each check returns the violations it finds - an empty list means the
invariant holds. cmd_check in the CLI runs them all and reports. These are
the database's health guarantees, and they run in CI on every commit, so a
change that quietly breaks the event log is caught before it is merged.

The invariants that depend on the claim/verdict layer (an LLM-only claim may
not sit above 'supported', a confirmed claim needs a deterministic verdict)
arrive with that layer. What can be checked today is checked today.
"""

from elenchus.entities import ENTITY_TABLES


def check_links(conn):
    """Return [(event_id, entity_kind, entity_id)] for links with no entity.

    Every event link points at an entity - a function, block, or string.
    A link to a row that does not exist is a dangling reference: the event
    claims something about an entity the database cannot produce.
    """
    broken = []
    rows = conn.execute(
        "SELECT event_id, entity_kind, entity_id FROM event_links"
    ).fetchall()

    for row in rows:
        table = ENTITY_TABLES.get(row["entity_kind"])
        if table is None:
            broken.append((row["event_id"], row["entity_kind"], row["entity_id"]))
            continue

        found = conn.execute(
            f"SELECT 1 FROM {table} WHERE id = ?", (row["entity_id"],)
        ).fetchone()
        if found is None:
            broken.append((row["event_id"], row["entity_kind"], row["entity_id"]))

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
    violations = []
    binaries = conn.execute(
        "SELECT DISTINCT binary_id FROM events"
    ).fetchall()

    for b in binaries:
        binary_id = b["binary_id"]
        rows = conn.execute(
            "SELECT seq FROM events WHERE binary_id = ? ORDER BY seq",
            (binary_id,),
        ).fetchall()

        for expected, row in enumerate(rows, start=1):
            if row["seq"] != expected:
                violations.append((binary_id, expected, row["seq"]))
                break

    return violations


# Every check, in the order cmd_check runs them. Each entry pairs a short
# label with the function, so adding a check is one line here.
CHECKS = [
    ("dangling links", check_links),
    ("orphan events", check_orphan_events),
    ("seq gaps", check_seq_gaps),
]
