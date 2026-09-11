"""Store corpus membership and ground truth into the database.

The corpus is built from twinned binaries: a debug build carrying DWARF and
a stripped build that does not. The agent analyses the stripped one; its
ground truth is read from the debug one and attached to the stripped one's
functions by address, which strip leaves in place.

This module records that relationship (register_corpus_binary) and writes the
ground truth (store_ground_truth), matching each ground-truth function to the
stripped function Ghidra found at the same address. A ground-truth function
with no match is still stored, function_id left null - most often it was
inlined away, which is itself worth knowing.
"""

import json

from elenchus.corpus.dwarf import ground_truth


def register_corpus_binary(
    conn, binary_id, package, compiler, opt_level, stripped,
    version=None, twin_id=None,
):
    """Record that a binary is a corpus member, and how it was built."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO corpus_binaries "
            "(binary_id, package, version, compiler, opt_level, stripped, twin_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (binary_id, package, version, compiler, opt_level,
             int(stripped), twin_id),
        )


def link_twins(conn, debug_id, stripped_id):
    """Point each of a debug/stripped pair at the other."""
    with conn:
        conn.execute(
            "UPDATE corpus_binaries SET twin_id = ? WHERE binary_id = ?",
            (stripped_id, debug_id),
        )
        conn.execute(
            "UPDATE corpus_binaries SET twin_id = ? WHERE binary_id = ?",
            (debug_id, stripped_id),
        )


def store_ground_truth(conn, stripped_binary_id, debug_path):
    """Read ground truth from debug_path and attach it to the stripped binary.

    Returns (total, matched): how many ground-truth functions were stored and
    how many lined up with a function Ghidra found in the stripped binary.
    """
    function_id_by_address = {
        row["address"]: row["id"]
        for row in conn.execute(
            "SELECT id, address FROM functions "
            "WHERE binary_id = ? AND is_external = 0",
            (stripped_binary_id,),
        )
    }

    total = 0
    matched = 0
    with conn:
        for gt in ground_truth(debug_path):
            function_id = function_id_by_address.get(gt.address)
            if function_id is not None:
                matched += 1

            conn.execute(
                "INSERT OR REPLACE INTO ground_truth "
                "(binary_id, function_id, address, name, return_type, "
                " param_types, decl_line) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    stripped_binary_id,
                    function_id,
                    gt.address,
                    gt.name,
                    gt.return_type,
                    json.dumps(list(gt.param_types)),
                    gt.decl_line,
                ),
            )
            total += 1

    return total, matched
