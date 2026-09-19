"""Extract facts from a binary using Ghidra, and record them in the database.

Two kinds of thing are written here, and the distinction matters.

Entities - functions, basic blocks, strings - exist or they do not. The
function at a given address is the same function on every scan, so it is
written once and reused.

Observations - that a run saw a function, a call, an edge, a string
reference - happen on every scan. Two runs may disagree: a different Ghidra
version, or a deeper analysis pass, can surface what an earlier run missed.
That disagreement is evidence, so each run records what it saw.

Provenance (which tool, which version, which parameters) lives on the run,
not in every payload. A payload carries only what is specific to that one
observation.
"""

import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pefile

from elenchus.db import add_events


@contextmanager
def open_fresh(path):
    """Open a binary in a Ghidra project of its own, made for this call.

    pyghidra's default is a project beside the binary, named after it. Open
    the same path twice and the second call finds a program of that name
    already in the project and hands back the analysis of the file as it was
    the first time - silently, since nothing failed.

    That is only ever wrong. A rebuilt package keeps its path and changes
    its contents, so a rescan read the old layout while ground truth was
    read from the new one: tinyspline came back 20 of 343 functions matched,
    c-blosc 152 of 419, and the only reason it was caught is that the build
    prints the match rate. A fresh project each time costs one import and
    leaves nothing behind to go stale.
    """
    import pyghidra

    with tempfile.TemporaryDirectory(prefix="elenchus-ghidra-") as location:
        with pyghidra.open_program(
            str(path), project_location=location, project_name="scan",
            nested_project_location=False,
        ) as api:
            yield api

# UNWIND_INFO flag marking an entry that continues another function's unwind
# data rather than starting a function of its own.
_UNW_FLAG_CHAININFO = 0x4


def pdata_entry_points(path):
    """Return the absolute start address of every function .pdata records.

    Win64 requires an unwind entry for every function that touches the stack,
    and the compilers used here emit one for every function. That makes .pdata
    an inventory of function starts that needs no symbols and no call graph.

    Entries flagged as chained are skipped: they describe a further piece of a
    function already entered elsewhere (MSVC splits large functions this way),
    and creating a function at one would cut a real function in two.

    A file pefile cannot parse yields nothing rather than an error; the scan
    itself will report what is wrong with it.
    """
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXCEPTION"]]
        )
    except (pefile.PEFormatError, OSError):
        return []

    base = pe.OPTIONAL_HEADER.ImageBase
    starts = set()
    for entry in getattr(pe, "DIRECTORY_ENTRY_EXCEPTION", []):
        unwind = getattr(entry, "unwindinfo", None)
        if unwind is not None and unwind.Flags & _UNW_FLAG_CHAININFO:
            continue
        starts.add(base + entry.struct.BeginAddress)
    return sorted(starts)


def _ghidra_create(program, address, monitor):
    from ghidra.app.cmd.disassemble import DisassembleCommand
    from ghidra.app.cmd.function import CreateFunctionCmd

    DisassembleCommand(address, None, True).applyTo(program, monitor)
    return bool(CreateFunctionCmd(address).applyTo(program, monitor))


def _ghidra_reanalyse(program, monitor):
    from ghidra.app.plugin.core.analysis import AutoAnalysisManager

    AutoAnalysisManager.getAnalysisManager(program).startAnalysis(monitor)


def seed_functions_from_pdata(program, entry_points, create=None, reanalyse=None):
    """Create the functions Ghidra's analysis did not find at .pdata starts.

    Ghidra finds functions in a stripped PE by following calls. A function
    reached only through a pointer - a cipher registered in a descriptor
    table, an interpreter built-in, a callback, a virtual method - has no call
    to follow, and it was never disassembled at all: libtomcrypt -O3 lost a
    third of its functions this way, and no Ghidra analyzer reads .pdata to
    recover them. Measured before adopting on that binary: all 152 missing
    ground-truth functions recovered, and not one of the 436 functions found
    before changed its listing.

    Addresses that already start or fall inside a function are left alone.
    Analysis is re-run once afterwards, so seeded functions are analysed like
    every other. Returns the number of functions created.

    create and reanalyse exist so the decision logic can be tested without a
    JVM; by default they are Ghidra's own commands.
    """
    create = create or _ghidra_create
    reanalyse = reanalyse or _ghidra_reanalyse

    from_ghidra = create is _ghidra_create
    monitor = None
    if from_ghidra:
        from ghidra.util.task import ConsoleTaskMonitor
        monitor = ConsoleTaskMonitor()

    manager = program.getFunctionManager()
    space = program.getAddressFactory().getDefaultAddressSpace()

    created = 0
    transaction = program.startTransaction("seed functions from .pdata")
    try:
        for value in entry_points:
            address = space.getAddress(value)
            if manager.getFunctionContaining(address) is not None:
                continue
            try:
                if create(program, address, monitor):
                    created += 1
            except Exception:  # noqa: BLE001 - one bad entry costs one function
                continue
    finally:
        program.endTransaction(transaction, True)

    if created:
        reanalyse(program, monitor)
    return created


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


def ghidra_version(program):
    """Return the Ghidra version that analysed this program."""
    return str(program.getMetadata()["Created With Ghidra Version"])


def extract_functions(conn, binary_id, run_id, program):
    """Record every internal function Ghidra sees, and that this run saw it.

    Imports are handled separately by extract_imports; here we take the
    functions Ghidra places in the binary itself.

    Returns {address: function_id} for internal functions.
    """
    id_by_address = {
        row["address"]: row["id"]
        for row in conn.execute(
            "SELECT id, address FROM functions "
            "WHERE binary_id = ? AND is_external = 0",
            (binary_id,),
        )
    }

    records = []
    with conn:
        for f in program.getFunctionManager().getFunctions(True):
            address = f.getEntryPoint().getOffset()
            known = address in id_by_address

            if not known:
                cur = conn.execute(
                    "INSERT INTO functions "
                    "(binary_id, address, size, raw_name, is_external, is_thunk) "
                    "VALUES (?, ?, ?, ?, 0, ?)",
                    (
                        binary_id,
                        address,
                        f.getBody().getNumAddresses(),
                        f.getName(),
                        int(f.isThunk()),
                    ),
                )
                id_by_address[address] = cur.lastrowid

            records.append((
                "observation.function",
                {"first_seen": not known},
                [("function", id_by_address[address], "subject")],
            ))

    add_events(conn, binary_id, run_id, records)
    return id_by_address


def _import_index(conn, binary_id):
    """Return {(library, name): function_id} for this binary's imports."""
    return {
        (row["library"], row["raw_name"]): row["id"]
        for row in conn.execute(
            "SELECT id, library, raw_name FROM functions "
            "WHERE binary_id = ? AND is_external = 1",
            (binary_id,),
        )
    }


def extract_imports(conn, binary_id, run_id, program):
    """Record every imported function: name, source library, that it was seen.

    Imports are the strongest readable signal in a stripped Windows binary -
    a function that calls CreateFileW and ReadFile is doing file I/O whatever
    its own name was stripped to. They live in the functions table with
    is_external = 1, keyed by (library, name) rather than by address, since an
    external function has no address of its own inside the binary.

    Returns {(library, name): function_id}.
    """
    id_by_key = _import_index(conn, binary_id)

    records = []
    with conn:
        for f in program.getFunctionManager().getExternalFunctions():
            name = f.getName()
            library = f.getExternalLocation().getLibraryName()
            key = (library, name)
            known = key in id_by_key

            if not known:
                # No real address for an external function. Use a stable
                # negative synthetic so UNIQUE (binary_id, address) still holds
                # and internal addresses (always >= 0) never collide.
                address = -(len(id_by_key) + 1)

                cur = conn.execute(
                    "INSERT INTO functions "
                    "(binary_id, address, size, raw_name, library, "
                    " is_external, is_thunk) "
                    "VALUES (?, ?, 0, ?, ?, 1, 0)",
                    (binary_id, address, name, library),
                )
                id_by_key[key] = cur.lastrowid

            records.append((
                "observation.import",
                {"first_seen": not known, "library": library, "name": name},
                [("function", id_by_key[key], "subject")],
            ))

    add_events(conn, binary_id, run_id, records)
    return id_by_key


def extract_calls(conn, binary_id, run_id, program, id_by_address):
    """Record every call relationship Ghidra found, as observation events.

    Both internal calls and calls into imports are recorded. Import targets
    are looked up by (library, name) since they have no meaningful address.
    Run extract_imports first so those targets already exist.
    """
    import_id = _import_index(conn, binary_id)

    records = []
    for f in program.getFunctionManager().getFunctions(True):
        caller_addr = f.getEntryPoint().getOffset()
        if caller_addr not in id_by_address:
            continue
        caller_id = id_by_address[caller_addr]

        for callee in f.getCalledFunctions(None):
            if callee.isExternal():
                key = (callee.getExternalLocation().getLibraryName(), callee.getName())
                callee_id = import_id.get(key)
            else:
                callee_id = id_by_address.get(callee.getEntryPoint().getOffset())

            if callee_id is None:
                continue

            records.append((
                "observation.call",
                {},
                [
                    ("function", caller_id, "subject"),
                    ("function", callee_id, "target"),
                ],
            ))

    return add_events(conn, binary_id, run_id, records)


def extract_blocks(conn, binary_id, run_id, program, id_by_address):
    """Record basic blocks and the control-flow edges between them.

    A basic block is a straight run of instructions with one way in and one
    way out. The edges between them - a branch, a loop's back edge, a
    fall-through - are the shape of the function, and that shape survives
    optimisation better than the individual instructions do. It is the
    encoder's primary structural signal.

    Blocks are entities keyed by address; edges are observations linking two
    blocks with a flow type. Returns the number of blocks this run observed.
    """
    from ghidra.program.model.block import BasicBlockModel
    from ghidra.util.task import ConsoleTaskMonitor

    monitor = ConsoleTaskMonitor()
    bbm = BasicBlockModel(program)
    fm = program.getFunctionManager()

    id_by_block_address = {
        row["address"]: row["id"]
        for row in conn.execute(
            "SELECT id, address FROM basic_blocks WHERE binary_id = ?", (binary_id,)
        )
    }

    block_records = []
    edge_records = []
    seen = 0

    with conn:
        for f in fm.getFunctions(True):
            func_addr = f.getEntryPoint().getOffset()
            function_id = id_by_address.get(func_addr)
            if function_id is None:
                continue

            blocks = bbm.getCodeBlocksContaining(f.getBody(), monitor)
            while blocks.hasNext():
                block = blocks.next()
                addr = block.getMinAddress().getOffset()
                size = block.getMaxAddress().getOffset() - addr + 1
                known = addr in id_by_block_address

                if not known:
                    cur = conn.execute(
                        "INSERT INTO basic_blocks "
                        "(binary_id, function_id, address, size) "
                        "VALUES (?, ?, ?, ?)",
                        (binary_id, function_id, addr, size),
                    )
                    id_by_block_address[addr] = cur.lastrowid

                block_records.append((
                    "observation.basic_block",
                    {"first_seen": not known},
                    [("basic_block", id_by_block_address[addr], "subject")],
                ))
                seen += 1

        # A second pass for edges: every block now has an id, so both ends of
        # an edge can be linked. Edges that leave the binary (returns, tail
        # calls into unknown code) have no destination block and are skipped.
        for f in fm.getFunctions(True):
            if id_by_address.get(f.getEntryPoint().getOffset()) is None:
                continue

            blocks = bbm.getCodeBlocksContaining(f.getBody(), monitor)
            while blocks.hasNext():
                block = blocks.next()
                src_addr = block.getMinAddress().getOffset()
                src_id = id_by_block_address.get(src_addr)
                if src_id is None:
                    continue

                dests = block.getDestinations(monitor)
                while dests.hasNext():
                    d = dests.next()
                    dst_addr = d.getDestinationAddress().getOffset()
                    dst_id = id_by_block_address.get(dst_addr)
                    if dst_id is None:
                        continue

                    edge_records.append((
                        "observation.cfg_edge",
                        {"flow": str(d.getFlowType())},
                        [
                            ("basic_block", src_id, "subject"),
                            ("basic_block", dst_id, "target"),
                        ],
                    ))

    add_events(conn, binary_id, run_id, block_records)
    add_events(conn, binary_id, run_id, edge_records)
    return seen


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

    Returns the number of strings this run observed.
    """
    id_by_string_address = {
        row["address"]: row["id"]
        for row in conn.execute(
            "SELECT id, address FROM strings WHERE binary_id = ?", (binary_id,)
        )
    }

    candidates = _collect_strings(program, id_by_address, min_length)
    kept = [c for c in candidates if c[3] or keep_unreferenced]
    if not kept:
        return 0

    records = []
    with conn:
        for address, value, encoding, referenced_by in kept:
            offset = address.getOffset()
            known = offset in id_by_string_address

            if not known:
                cur = conn.execute(
                    "INSERT INTO strings "
                    "(binary_id, address, value, encoding, length) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (binary_id, offset, value, encoding, len(value)),
                )
                id_by_string_address[offset] = cur.lastrowid

            string_id = id_by_string_address[offset]

            records.append((
                "observation.string",
                {"first_seen": not known, "referenced": bool(referenced_by)},
                [("string", string_id, "subject")],
            ))

            for function_id, site in referenced_by:
                records.append((
                    "observation.string_ref",
                    {"site": site},
                    [
                        ("function", function_id, "subject"),
                        ("string", string_id, "target"),
                    ],
                ))

    add_events(conn, binary_id, run_id, records)
    return len(kept)
