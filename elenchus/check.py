"""Consistency checks over the database."""

from elenchus.entities import ENTITY_TABLES


def check_links(conn):
    """Return [(event_id, entity_kind, entity_id)] for links with no matching entity."""
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
