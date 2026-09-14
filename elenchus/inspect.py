"""Show one function the way a person needs to see it to judge it.

A match rate says that a ground-truth address landed on a function Ghidra
found. It does not say that the function at that address is the one the
compiler named. A systematic error - an image base applied on one side and
not the other, every entry point resolving to its thunk - would keep the
match rate at a hundred percent and attach every name to its neighbour. The
metric cannot see that. A person reading the code can: a function called
deflate_stored should be copying bytes, and if it is dividing floats,
something upstream is wrong.

So this module exists to make that reading cheap enough to actually do fifty
times. It puts the compiler's truth, Ghidra's guess, the raw instructions,
and their normalised form on one screen. The second column is not decoration:
what the encoder learns from is the normalised text, so that is what has to
look sane, not the disassembly it came from.

Sampling is seeded and reproducible. The set of functions checked by hand is
then itself a fact that can be written down and revisited, rather than
whatever happened to be on screen that afternoon.
"""

import json
import random

from elenchus.encoder.normalise import normalise_line

# One SELECT serves every lookup below; the callers differ only in their
# WHERE clause. Ground truth is joined loosely because its absence is
# informative - those are the runtime functions that came with the toolchain
# rather than from the source we compiled.
_SELECT = """
SELECT f.id           AS function_id,
       f.address      AS address,
       f.raw_name     AS ghidra_name,
       cb.package     AS package,
       cb.opt_level   AS opt_level,
       fc.n_instructions,
       fc.code_size,
       fc.byte_hash,
       gt.name        AS truth_name,
       gt.return_type AS return_type,
       gt.param_types AS param_types,
       gt.decl_line   AS decl_line
FROM function_code fc
JOIN functions f        ON f.id = fc.function_id
JOIN corpus_binaries cb ON cb.binary_id = fc.binary_id
LEFT JOIN ground_truth gt ON gt.function_id = f.id
"""


def find_functions(conn, function_id=None, package=None, opt=None,
                   name=None, truth=None):
    """Return function rows matching the filters, in corpus order.

    truth=True keeps only functions the compiler named, truth=False only
    those it did not, and None keeps both.
    """
    clauses = []
    params = []

    if function_id:
        clauses.append("f.id = ?")
        params.append(function_id)

    if package:
        clauses.append("cb.package = ?")
        params.append(package)
    if opt:
        clauses.append("cb.opt_level = ?")
        params.append(opt)
    if name:
        clauses.append("gt.name = ?")
        params.append(name)
    if truth is True:
        clauses.append("gt.name IS NOT NULL")
    if truth is False:
        clauses.append("gt.name IS NULL")

    query = _SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY cb.package, cb.opt_level, f.address"

    return conn.execute(query, params).fetchall()


def sample_functions(rows, count, seed):
    """Pick count rows reproducibly.

    A fixed seed over a fixed corpus gives the same fifty functions every
    time, so 'the fifty I checked' is a statement that can be repeated.
    """
    if count >= len(rows):
        return list(rows)
    return random.Random(seed).sample(list(rows), count)


def signature(row):
    """Render the compiler's declaration, or say that there is none."""
    if row["truth_name"] is None:
        return "(no ground truth - runtime or library glue)"

    try:
        params = json.loads(row["param_types"] or "[]")
    except json.JSONDecodeError:
        params = []

    arguments = ", ".join(params) if params else "void"
    return f"{row['return_type']} {row['truth_name']}({arguments})"


def listing_of(conn, function_id):
    """Return the stored instruction listing for one function."""
    row = conn.execute(
        "SELECT listing FROM function_code WHERE function_id = ?",
        (function_id,),
    ).fetchone()
    return row["listing"] if row else ""


def header(row):
    """Return the identifying lines shown above any listing."""
    where = f"{row['package']} -{row['opt_level']}"
    lines = [
        f"function  #{row['function_id']}  at {row['address']:#x}  in {where}",
        f"ghidra    {row['ghidra_name']}",
        f"truth     {signature(row)}",
    ]
    if row["decl_line"]:
        lines.append(f"source    line {row['decl_line']}")
    lines.append(
        f"size      {row['code_size']} bytes, "
        f"{row['n_instructions']} instructions, hash {row['byte_hash'][:12]}"
    )
    return lines


def side_by_side(listing, width=44, limit=None):
    """Render raw instructions and their normalised form in two columns.

    Reading them apart is what hides mistakes: a scheme that throws away too
    much looks reasonable on its own and only looks wrong next to what it
    came from.
    """
    lines = [
        "  " + "raw".ljust(width) + "normalised",
        "  " + "-" * (width - 2) + "  " + "-" * 24,
    ]

    rows = listing.splitlines()
    shown = rows if limit is None else rows[:limit]

    for line in shown:
        parts = line.split("\t")
        if len(parts) < 4:
            continue

        offset, mnemonic, operands, target = parts[0], parts[1], parts[2], parts[3]
        raw = f"{offset:>5}  {mnemonic} {operands}".rstrip()
        if target:
            raw += f"  -> {target}"

        tokens = " ".join(normalise_line(line))

        if len(raw) > width - 1:
            raw = raw[: width - 2] + "…"
        lines.append("  " + raw.ljust(width) + tokens)

    if limit is not None and len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more instructions")

    return lines


def render(conn, row, limit=None):
    """Return the full report for one function as a list of lines."""
    lines = list(header(row))
    lines.append("")
    lines.extend(side_by_side(listing_of(conn, row["function_id"]), limit=limit))
    return lines


def summarise(row):
    """Return one line describing a function, for list output."""
    name = row["truth_name"] or "-"
    return (
        f"  #{row['function_id']:<7} {row['package']:<13} -{row['opt_level']}  "
        f"{row['n_instructions']:>5} instr  {name}"
    )
