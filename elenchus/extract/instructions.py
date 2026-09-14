"""Extract the instruction stream of every function, and hash its bytes.

Until now the database held the *shape* of a function - its blocks, its edges,
who it calls - but not the code itself. The encoder's primary input is the
instruction sequence, so it has to be recorded.

What is stored here is deliberately raw. Each instruction becomes one line:

    offset <TAB> mnemonic <TAB> operands <TAB> target

with operands as Ghidra prints them (spaces removed) and target naming where
a call or branch goes. Nothing is classified, nothing is thrown away yet.
Interpretation - turning `[RBP+-0x18]` into MEM_STACK - belongs to
encoder/normalise.py, which is pure Python and can be rerun in a second.
A Ghidra scan takes minutes; a normalisation scheme will be changed a dozen
times before the encoder trains. Keeping them apart means only the cheap half
is repeated.

Two things are address-free by construction. The offset is relative to the
function's entry point, and a branch inside the same function is recorded as a
signed distance rather than a destination. Both mean a function's listing is
identical wherever the linker happened to place it.

The byte hash covers the instruction bytes in order, and is what cross-binary
deduplication will use: the same helper compiled into two packages at the same
optimisation level produces the same hash.
"""

import hashlib

from elenchus.db import add_events


def _operands(instruction):
    """Return the operand texts, space-free, joined by a single space.

    Ghidra prints `qword ptr [RBP + -0x18]`; the spaces are removed so that a
    space can separate one operand from the next without ambiguity.
    """
    parts = []
    for index in range(instruction.getNumOperands()):
        text = str(instruction.getDefaultOperandRepresentation(index))
        parts.append("".join(text.split()))
    return " ".join(parts)


def _external_target(instruction):
    """Return IMPORT:library!name if this instruction refers to an import.

    A call through the import table has no flow destination inside the
    binary - the address is filled in by the loader - so the reference is
    the only place the name appears.
    """
    try:
        for ref in instruction.getReferencesFrom():
            if not ref.isExternalReference():
                continue
            return f"IMPORT:{ref.getLibraryName()}!{ref.getLabel()}"
    except AttributeError:
        pass
    return ""


def _target(instruction, function_manager, body):
    """Describe where a call or branch goes, without naming an address.

    Returns one of:
      IMPORT:lib!name  an imported function, by name - the strongest signal
                       a stripped Windows binary still carries
      LOCAL:+0x2f      a branch within this function, as a signed distance
      CALL:internal    a call to another function in this binary
      TAIL:internal    a jump into another function (a tail call)
      CALL:indirect    a computed call - through a register or a table
      CALL:unknown     a destination Ghidra could not resolve
      ""               not a call or branch at all
    """
    flow = instruction.getFlowType()
    is_call = flow.isCall()
    if not (is_call or flow.isJump()):
        return ""

    external = _external_target(instruction)
    if external:
        return external

    flows = list(instruction.getFlows())
    if not flows:
        return "CALL:indirect" if is_call else "TAIL:indirect"

    destination = flows[0]

    target_function = function_manager.getFunctionAt(destination)
    if target_function is not None:
        if target_function.isThunk():
            thunked = target_function.getThunkedFunction(True)
            if thunked is not None:
                target_function = thunked

        if target_function.isExternal():
            location = target_function.getExternalLocation()
            library = location.getLibraryName() if location is not None else "?"
            return f"IMPORT:{library}!{target_function.getName()}"

    if body.contains(destination):
        delta = destination.getOffset() - instruction.getAddress().getOffset()
        sign = "+" if delta >= 0 else "-"
        return f"LOCAL:{sign}{abs(delta):#x}"

    if target_function is not None:
        return "CALL:internal" if is_call else "TAIL:internal"

    return "CALL:unknown" if is_call else "TAIL:unknown"


def function_listing(function, listing, function_manager):
    """Build one function's listing text, byte hash, and counts.

    Returns (text, instruction_count, byte_size, byte_hash). A function whose
    body Ghidra never disassembled comes back empty, and the caller skips it:
    an empty listing is not evidence of anything.
    """
    entry = function.getEntryPoint().getOffset()
    body = function.getBody()

    digest = hashlib.sha256()
    lines = []
    size = 0

    for instruction in listing.getInstructions(body, True):
        raw = bytes(byte & 0xFF for byte in instruction.getBytes())
        digest.update(raw)
        size += len(raw)

        offset = instruction.getAddress().getOffset() - entry
        lines.append("\t".join((
            f"{offset:x}",
            str(instruction.getMnemonicString()).lower(),
            _operands(instruction),
            _target(instruction, function_manager, body),
        )))

    if not lines:
        return "", 0, 0, ""

    return "\n".join(lines), len(lines), size, digest.hexdigest()


def extract_code(conn, binary_id, run_id, program, id_by_address):
    """Record the instruction listing of every internal function.

    The listing is a projection keyed by function, replaced whenever a newer
    scan produces a better one; the fact that *this run* saw that code is an
    observation, appended like every other. Returns how many functions were
    recorded.
    """
    listing = program.getListing()
    function_manager = program.getFunctionManager()

    rows = []
    records = []

    for function in function_manager.getFunctions(True):
        if function.isExternal():
            continue

        function_id = id_by_address.get(function.getEntryPoint().getOffset())
        if function_id is None:
            continue

        text, count, size, digest = function_listing(
            function, listing, function_manager
        )
        if not count:
            continue

        rows.append((function_id, binary_id, count, size, digest, text))
        records.append((
            "observation.instructions",
            {"instructions": count, "bytes": size, "byte_hash": digest},
            [("function", function_id, "subject")],
        ))

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO function_code "
            "(function_id, binary_id, n_instructions, code_size, byte_hash, "
            " listing) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    add_events(conn, binary_id, run_id, records)
    return len(rows)


def function_index(conn, binary_id):
    """Return {address: function_id} for a binary already in the database.

    A rescan that only adds instructions does not repeat function extraction,
    so it reads the mapping from the database instead of rebuilding it.
    """
    return {
        row["address"]: row["id"]
        for row in conn.execute(
            "SELECT id, address FROM functions "
            "WHERE binary_id = ? AND is_external = 0",
            (binary_id,),
        )
    }
