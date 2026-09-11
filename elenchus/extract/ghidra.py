"""Extract facts from a binary using Ghidra, and record them in the database."""

import hashlib
from pathlib import Path

from elenchus.db import add_events


def file_sha256(path):
    """Return the hex SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def register_binary(conn, path, arch):
    """Insert the binary if new, or find the existing row. Return its id."""
    digest = file_sha256(path)
    row = conn.execute("SELECT id FROM binaries WHERE sha256 = ?", (digest,)).fetchone()
    if row is not None:
        return row["id"]

    with conn:
        cur = conn.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (str(Path(path).resolve()), digest, arch),
        )
    return cur.lastrowid


def extract_functions(conn, binary_id, program):
    """Read every function from the Ghidra program and store it.

    Returns {address: function_id}.
    """
    id_by_address = {}

    existing = conn.execute(
        "SELECT id, address FROM functions WHERE binary_id = ?", (binary_id,)
    ).fetchall()
    if existing:
        for row in existing:
            id_by_address[row["address"]] = row["id"]
        return id_by_address

    with conn:
        for f in program.getFunctionManager().getFunctions(True):
            address = f.getEntryPoint().getOffset()
            size = f.getBody().getNumAddresses()
            raw_name = f.getName()
            is_external = int(f.isExternal())
            is_thunk = int(f.isThunk())

            cur = conn.execute(
                "INSERT INTO functions "
                "(binary_id, address, size, raw_name, is_external, is_thunk) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (binary_id, address, size, raw_name, is_external, is_thunk),
            )
            id_by_address[address] = cur.lastrowid

    return id_by_address


def extract_calls(conn, binary_id, run_id, program, id_by_address):
    """Record every call relationship Ghidra found, as observation events."""
    ghidra_version = str(program.getMetadata()["Created With Ghidra Version"])
    records = []

    for f in program.getFunctionManager().getFunctions(True):
        caller_addr = f.getEntryPoint().getOffset()

        for callee in f.getCalledFunctions(None):
            callee_addr = callee.getEntryPoint().getOffset()

            if callee_addr not in id_by_address or caller_addr not in id_by_address:
                continue

            caller_id = id_by_address[caller_addr]
            callee_id = id_by_address[callee_addr]

            records.append((
                "observation.call",
                {"tool": "ghidra", "tool_version": ghidra_version},
                [
                    ("function", caller_id, "subject"),
                    ("function", callee_id, "target"),
                ],
            ))

    return add_events(conn, binary_id, run_id, records)


def _collect_strings(program, id_by_address, min_length):
    """Find candidate strings and the functions that reference them.

    Returns a list of (address, value, encoding, [(function_id, call_site)]).
    Ghidra marks a lot of debug metadata as string data, so the reference list
    is what tells a real program string apart from leftover DWARF text.
    """
    listing = program.getListing()
    fm = program.getFunctionManager()
    refs = program.getReferenceManager()

    candidates = []
    for data in listing.getDefinedData(True):
        if not data.hasStringValue():
            continue

        value = str(data.getValue())
        if len(value) < min_length:
            continue

        address = data.getAddress()
        encoding = str(data.getDataType().getName())

        referenced_by = []
        for ref in refs.getReferencesTo(address):
            caller = fm.getFunctionContaining(ref.getFromAddress())
            if caller is None:
                continue

            caller_addr = caller.getEntryPoint().getOffset()
            if caller_addr not in id_by_address:
                continue

            referenced_by.append((
                id_by_address[caller_addr],
                hex(ref.getFromAddress().getOffset()),
            ))

        candidates.append((address, value, encoding, referenced_by))

    return candidates


def extract_strings(
    conn,
    binary_id,
    run_id,
    program,
    id_by_address,
    min_length=4,
    keep_unreferenced=False,
):
    """Record strings and which functions reference them.

    By default only strings reachable from code are kept. Unreferenced strings
    are mostly debug metadata, but occasionally interesting on their own, so
    keep_unreferenced makes that a choice rather than a hardcoded rule.
    """
    existing = conn.execute(
        "SELECT COUNT(*) AS n FROM strings WHERE binary_id = ?", (binary_id,)
    ).fetchone()["n"]
    if existing:
        return 0

    ghidra_version = str(program.getMetadata()["Created With Ghidra Version"])
    candidates = _collect_strings(program, id_by_address, min_length)

    kept = [c for c in candidates if c[3] or keep_unreferenced]
    if not kept:
        return 0

    string_ids = {}
    with conn:
        for address, value, encoding, _ in kept:
            cur = conn.execute(
                "INSERT INTO strings (binary_id, address, value, encoding, length) "
                "VALUES (?, ?, ?, ?, ?)",
                (binary_id, address.getOffset(), value, encoding, len(value)),
            )
            string_ids[address.getOffset()] = cur.lastrowid

    provenance = {
        "tool": "ghidra",
        "tool_version": ghidra_version,
        "min_length": min_length,
        "keep_unreferenced": keep_unreferenced,
    }

    records = []
    for address, value, encoding, referenced_by in kept:
        string_id = string_ids[address.getOffset()]

        records.append((
            "observation.string",
            dict(provenance, referenced=bool(referenced_by)),
            [("string", string_id, "subject")],
        ))

        for function_id, site in referenced_by:
            records.append((
                "observation.string_ref",
                dict(provenance, site=site),
                [
                    ("function", function_id, "subject"),
                    ("string", string_id, "target"),
                ],
            ))

    add_events(conn, binary_id, run_id, records)
    return len(kept)