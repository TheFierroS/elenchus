"""Entity registry: which entity kinds exist and which table holds each one."""

import re

_VALID_NAME = re.compile(r"[a-z_]+")

# Imports are functions too - they live in the functions table with
# is_external = 1 - so they share it rather than getting a table of their own.
ENTITY_TABLES = {
    "function": "functions",
    "string": "strings",
    "basic_block": "basic_blocks",
}


def entity_kind_check() -> str:
    """Build the SQL CHECK constraint for event_links.entity_kind."""
    names = sorted(ENTITY_TABLES)

    for name in names:
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid entity name: {name!r}")

    quoted = ", ".join(f"'{name}'" for name in names)
    return f"CHECK (entity_kind IN ({quoted}))"
