"""Entity registry: which entity kinds exist and which table holds each one."""

import re

_VALID_NAME = re.compile(r"[a-z_]+")

ENTITY_TABLES = {
    "function": "functions",
    "string": "strings",
}


def entity_kind_check() -> str:
    """Build the SQL CHECK constraint for event_links.entity_kind."""
    names = sorted(ENTITY_TABLES)

    for name in names:
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid entity name: {name!r}")

    quoted = ", ".join(f"'{name}'" for name in names)
    return f"CHECK (entity_kind IN ({quoted}))"
