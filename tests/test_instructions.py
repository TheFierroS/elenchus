"""Tests for instruction extraction, with Ghidra stood in for by fakes.

Ghidra is not installed in CI and starting it costs a JVM, so the extractor
is written against a handful of methods - getBytes, getMnemonicString,
getFlows - and those are what the fakes provide. The test then checks the
part that is ours: the line format, the relative offsets, the byte hash, and
the fact that a listing does not change when the same code is linked at a
different address.
"""

import hashlib

import pytest

from elenchus.extract.instructions import extract_code, function_index


class FakeAddress:
    def __init__(self, offset):
        self._offset = offset

    def getOffset(self):
        return self._offset


class FakeFlowType:
    def __init__(self, call=False, jump=False):
        self._call = call
        self._jump = jump

    def isCall(self):
        return self._call

    def isJump(self):
        return self._jump


class FakeInstruction:
    def __init__(self, address, mnemonic, operands=(), data=b"\x90",
                 flow=None, flows=(), refs=()):
        self._address = FakeAddress(address)
        self._mnemonic = mnemonic
        self._operands = list(operands)
        self._data = data
        self._flow = flow or FakeFlowType()
        self._flows = [FakeAddress(f) for f in flows]
        self._refs = list(refs)

    def getAddress(self):
        return self._address

    def getBytes(self):
        return list(self._data)

    def getMnemonicString(self):
        return self._mnemonic

    def getNumOperands(self):
        return len(self._operands)

    def getDefaultOperandRepresentation(self, index):
        return self._operands[index]

    def getFlowType(self):
        return self._flow

    def getFlows(self):
        return self._flows

    def getReferencesFrom(self):
        return self._refs


class FakeBody:
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def contains(self, address):
        return self._low <= address.getOffset() <= self._high


class FakeFunction:
    def __init__(self, entry, instructions, body=None, external=False,
                 thunk=False, name="FUN"):
        self._entry = FakeAddress(entry)
        self.instructions = instructions
        self._body = body or FakeBody(entry, entry + 0x1000)
        self._external = external
        self._thunk = thunk
        self._name = name

    def getEntryPoint(self):
        return self._entry

    def getBody(self):
        return self._body

    def isExternal(self):
        return self._external

    def isThunk(self):
        return self._thunk

    def getThunkedFunction(self, _recursive):
        return None

    def getName(self):
        return self._name


class FakeListing:
    def getInstructions(self, body, _forward):
        # The fakes keep each function's instructions on its body's owner, so
        # the body carries them here rather than the listing holding a map.
        return body.instructions


class FakeBodyWithCode(FakeBody):
    def __init__(self, low, high, instructions):
        super().__init__(low, high)
        self.instructions = instructions


class FakeFunctionManager:
    def __init__(self, functions, by_address=None):
        self._functions = functions
        self._by_address = by_address or {}

    def getFunctions(self, _forward):
        return self._functions

    def getFunctionAt(self, address):
        return self._by_address.get(address.getOffset())


class FakeProgram:
    def __init__(self, function_manager):
        self._fm = function_manager

    def getListing(self):
        return FakeListing()

    def getFunctionManager(self):
        return self._fm


def build_function(entry, instructions, **kwargs):
    body = FakeBodyWithCode(entry, entry + 0x1000, instructions)
    return FakeFunction(entry, instructions, body=body, **kwargs)


@pytest.fixture
def binary(db):
    """A binary with one function row, ready to receive a listing."""
    with db:
        binary_id = db.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES ('/tmp/x.dll', 'abc', 'x86:LE:64:default', datetime('now'))"
        ).lastrowid
        db.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, 0x401000, 16, 'FUN_00401000')",
            (binary_id,),
        )
        run_id = db.execute(
            "INSERT INTO runs (kind, code_version, params, started_at, status) "
            "VALUES ('extract', 'test', '{}', datetime('now'), 'ok')"
        ).lastrowid
    return binary_id, run_id


def simple_function(entry=0x401000):
    return build_function(entry, [
        FakeInstruction(entry + 0, "PUSH", ["RBP"], b"\x55"),
        FakeInstruction(entry + 1, "MOV", ["RBP", "RSP"], b"\x48\x89\xe5"),
        FakeInstruction(
            entry + 4, "JZ", [hex(entry + 0x20)], b"\x74\x1a",
            flow=FakeFlowType(jump=True), flows=[entry + 0x20],
        ),
    ])


def test_listing_shape_and_offsets(db, binary):
    binary_id, run_id = binary
    function = simple_function()
    program = FakeProgram(FakeFunctionManager([function]))

    written = extract_code(
        db, binary_id, run_id, program, function_index(db, binary_id)
    )
    assert written == 1

    row = db.execute("SELECT * FROM function_code").fetchone()
    assert row["n_instructions"] == 3
    assert row["code_size"] == 6

    lines = row["listing"].splitlines()
    assert lines[0] == "0\tpush\tRBP\t"
    assert lines[1] == "1\tmov\tRBP RSP\t"
    # A forward branch inside the function keeps its direction, not its target.
    assert lines[2].endswith("LOCAL:+0x1c")


def test_byte_hash_is_over_the_instruction_bytes(db, binary):
    binary_id, run_id = binary
    program = FakeProgram(FakeFunctionManager([simple_function()]))
    extract_code(db, binary_id, run_id, program, function_index(db, binary_id))

    expected = hashlib.sha256(b"\x55\x48\x89\xe5\x74\x1a").hexdigest()
    row = db.execute("SELECT byte_hash FROM function_code").fetchone()
    assert row["byte_hash"] == expected


def test_normalised_form_is_independent_of_load_address(db, binary):
    """The same code linked elsewhere must normalise to the same tokens.

    The stored listing still carries Ghidra's printed destination address, so
    the two listings differ by that one number. What has to match - and what
    the encoder actually reads - is the normalised form. Checking it here
    rather than in test_normalise.py is deliberate: this is the seam between
    extraction and normalisation, and a seam is where the guarantee can be
    lost without either side looking wrong on its own.
    """
    from elenchus.encoder.normalise import normalise
    binary_id, run_id = binary

    program = FakeProgram(FakeFunctionManager([simple_function(0x401000)]))
    extract_code(db, binary_id, run_id, program, function_index(db, binary_id))
    first = db.execute("SELECT listing FROM function_code").fetchone()["listing"]

    moved = simple_function(0x501000)
    # Point the same database row at the moved copy by rewriting its address.
    with db:
        db.execute("UPDATE functions SET address = 0x501000")
    program = FakeProgram(FakeFunctionManager([moved]))
    extract_code(db, binary_id, run_id, program, function_index(db, binary_id))
    second = db.execute("SELECT listing FROM function_code").fetchone()["listing"]

    assert first != second          # the printed address moved with the code
    assert normalise(first) == normalise(second)


def test_import_call_is_named(db, binary):
    binary_id, run_id = binary
    entry = 0x401000

    thunk = FakeFunction(0x402000, [], external=True, name="CreateFileW")
    thunk.getExternalLocation = lambda: type(
        "Loc", (), {"getLibraryName": lambda self: "KERNEL32.dll"}
    )()

    function = build_function(entry, [
        FakeInstruction(
            entry, "CALL", ["0x402000"], b"\xe8\x00\x00\x00\x00",
            flow=FakeFlowType(call=True), flows=[0x402000],
        ),
    ])
    manager = FakeFunctionManager([function], {0x402000: thunk})

    extract_code(
        db, binary_id, run_id, FakeProgram(manager), function_index(db, binary_id)
    )
    listing = db.execute("SELECT listing FROM function_code").fetchone()["listing"]
    assert listing.endswith("IMPORT:KERNEL32.dll!CreateFileW")


def test_call_leaving_the_function_is_internal(db, binary):
    binary_id, run_id = binary
    entry = 0x401000

    callee = FakeFunction(0x401800, [], name="FUN_00401800")
    function = build_function(entry, [
        FakeInstruction(
            entry, "CALL", ["0x401800"], b"\xe8\x00\x00\x00\x00",
            flow=FakeFlowType(call=True), flows=[0x401800],
        ),
    ])
    # The callee sits inside the caller's address range in this fake, so the
    # function-at lookup - not the range check - has to decide.
    manager = FakeFunctionManager([function], {0x401800: callee})

    extract_code(
        db, binary_id, run_id, FakeProgram(manager), function_index(db, binary_id)
    )
    listing = db.execute("SELECT listing FROM function_code").fetchone()["listing"]
    assert listing.endswith("LOCAL:+0x800")


def test_rescan_replaces_rather_than_duplicates(db, binary):
    binary_id, run_id = binary
    program = FakeProgram(FakeFunctionManager([simple_function()]))

    extract_code(db, binary_id, run_id, program, function_index(db, binary_id))
    extract_code(db, binary_id, run_id, program, function_index(db, binary_id))

    rows = db.execute("SELECT COUNT(*) AS n FROM function_code").fetchone()
    assert rows["n"] == 1

    # But both runs left their observation behind: the listing is a
    # projection, the fact that a run saw it is history.
    events = db.execute(
        "SELECT COUNT(*) AS n FROM events WHERE type = 'observation.instructions'"
    ).fetchone()
    assert events["n"] == 2


def test_function_without_instructions_is_skipped(db, binary):
    binary_id, run_id = binary
    empty = build_function(0x401000, [])
    program = FakeProgram(FakeFunctionManager([empty]))

    assert extract_code(
        db, binary_id, run_id, program, function_index(db, binary_id)
    ) == 0
    assert db.execute("SELECT COUNT(*) AS n FROM function_code").fetchone()["n"] == 0
