"""Tests for recovering functions from .pdata.

Ghidra finds functions in a stripped PE by following calls, so a function
reached only through a pointer was never found - a third of libtomcrypt -O3.
Seeding from .pdata fixed that without changing any function found before.
These tests pin the decisions around it: what counts as a function start,
what is left alone, and that one bad entry costs one function, not the scan.
No JVM is started; Ghidra's side is replaced by small fakes.
"""

import contextlib
import sys
import types

import pefile

sys.modules.setdefault("pyghidra", types.ModuleType("pyghidra"))

from elenchus.extract import ghidra as module  # noqa: E402
from elenchus.extract.ghidra import (  # noqa: E402
    pdata_entry_points,
    seed_functions_from_pdata,
)

SAMPLE = "tests/fixtures/sample.dll"


# ---------------------------------------------------------------- reading .pdata


def test_entry_points_are_function_starts_inside_executable_code():
    starts = pdata_entry_points(SAMPLE)
    pe = pefile.PE(SAMPLE)
    base = pe.OPTIONAL_HEADER.ImageBase
    code = [
        (base + s.VirtualAddress, base + s.VirtualAddress + s.Misc_VirtualSize)
        for s in pe.sections if s.Characteristics & 0x20000000
    ]

    assert starts, "the fixture has an exception directory"
    assert starts == sorted(set(starts))
    assert all(any(lo <= a < hi for lo, hi in code) for a in starts)


def test_a_file_that_is_not_a_pe_has_no_entry_points(tmp_path):
    path = tmp_path / "not_a_pe.bin"
    path.write_bytes(b"not a portable executable")
    assert pdata_entry_points(path) == []


def test_a_missing_file_has_no_entry_points(tmp_path):
    assert pdata_entry_points(tmp_path / "gone.dll") == []


def test_chained_entries_are_not_function_starts(monkeypatch):
    """A chained entry continues another function; a function there would cut
    a real one in two."""
    def entry(begin, flags):
        return types.SimpleNamespace(
            struct=types.SimpleNamespace(BeginAddress=begin),
            unwindinfo=types.SimpleNamespace(Flags=flags),
        )

    class FakePE:
        OPTIONAL_HEADER = types.SimpleNamespace(ImageBase=0x10000)
        DIRECTORY_ENTRY_EXCEPTION = [
            entry(0x1000, 0x0), entry(0x1200, 0x4), entry(0x1400, 0x1),
            types.SimpleNamespace(struct=types.SimpleNamespace(BeginAddress=0x1600),
                                  unwindinfo=None),
        ]

        def __init__(self, *_a, **_k):
            pass

        def parse_data_directories(self, **_k):
            pass

    monkeypatch.setattr(module.pefile, "PE", FakePE)
    assert pdata_entry_points("ignored") == [0x11000, 0x11400, 0x11600]


# ---------------------------------------------------------------- seeding


class FakeProgram:
    """Functions as (start, end) ranges; records what Ghidra would be told."""

    def __init__(self, functions=()):
        self.functions = list(functions)
        self.transactions = []
        manager = types.SimpleNamespace(getFunctionContaining=self._containing)
        space = types.SimpleNamespace(getAddress=lambda value: value)
        self.getFunctionManager = lambda: manager
        self.getAddressFactory = lambda: types.SimpleNamespace(
            getDefaultAddressSpace=lambda: space)

    def _containing(self, address):
        return next((f for f in self.functions if f[0] <= address < f[1]), None)

    def startTransaction(self, name):
        self.transactions.append(["open", name])
        return len(self.transactions) - 1

    def endTransaction(self, index, commit):
        self.transactions[index][0] = "committed" if commit else "rolled back"


class Recorder:
    def __init__(self, fail_at=()):
        self.created, self.analysed, self.fail_at = [], 0, set(fail_at)

    def create(self, program, address, _monitor):
        if address in self.fail_at:
            raise RuntimeError("Ghidra could not disassemble here")
        self.created.append(address)
        program.functions.append((address, address + 0x10))
        return True

    def reanalyse(self, _program, _monitor):
        self.analysed += 1


def seed(program, starts, recorder):
    return seed_functions_from_pdata(program, starts, recorder.create,
                                     recorder.reanalyse)


def test_missing_starts_become_functions():
    program, recorder = FakeProgram([(0x1000, 0x1100)]), Recorder()
    assert seed(program, [0x1000, 0x2000, 0x3000], recorder) == 2
    assert recorder.created == [0x2000, 0x3000]


def test_a_start_inside_an_existing_function_is_left_alone():
    """Found functions are never cut: the measured property was 0 changes."""
    program, recorder = FakeProgram([(0x1000, 0x1100)]), Recorder()
    assert seed(program, [0x1040], recorder) == 0
    assert recorder.created == []


def test_analysis_runs_once_and_only_when_something_was_added():
    program, recorder = FakeProgram(), Recorder()
    seed(program, [0x1000, 0x2000], recorder)
    assert recorder.analysed == 1

    idle = Recorder()
    seed(FakeProgram([(0x1000, 0x3000)]), [0x1000, 0x2000], idle)
    assert idle.analysed == 0


def test_one_entry_that_fails_costs_only_that_function():
    program, recorder = FakeProgram(), Recorder(fail_at={0x2000})
    assert seed(program, [0x1000, 0x2000, 0x3000], recorder) == 2
    assert recorder.created == [0x1000, 0x3000]


def test_the_transaction_is_committed_even_when_creation_fails():
    program = FakeProgram()
    seed(program, [0x1000], Recorder(fail_at={0x1000}))
    assert program.transactions == [["committed", "seed functions from .pdata"]]


def test_a_create_that_reports_failure_is_not_counted():
    program = FakeProgram()
    result = seed_functions_from_pdata(program, [0x1000],
                                       create=lambda *_a: False,
                                       reanalyse=lambda *_a: None)
    assert result == 0


# ---------------------------------------------------------------- wiring


def test_the_corpus_scan_seeds_before_it_extracts(monkeypatch, db, tmp_path):
    """Seeding after extract_functions would record nothing it found."""
    from elenchus.corpus import cli as corpus_cli

    order = []
    program = types.SimpleNamespace(getLanguageID=lambda: "x86:LE:64:default")
    api = types.SimpleNamespace(getCurrentProgram=lambda: program)

    @contextlib.contextmanager
    def fake_open(_path):
        yield api

    monkeypatch.setattr(corpus_cli.pyghidra, "open_program", fake_open, raising=False)
    monkeypatch.setattr(corpus_cli, "ghidra_version", lambda _p: "test")
    monkeypatch.setattr(corpus_cli, "pdata_entry_points", lambda path: [0x1000])
    monkeypatch.setattr(corpus_cli, "seed_functions_from_pdata",
                        lambda _p, starts: order.append(("seed", starts)) or 1)
    monkeypatch.setattr(corpus_cli, "register_binary", lambda *_a: 1)
    monkeypatch.setattr(corpus_cli, "extract_functions",
                        lambda *_a: order.append(("functions",)) or {})
    for name in ("extract_imports", "extract_calls", "extract_blocks", "extract_code"):
        monkeypatch.setattr(corpus_cli, name, lambda *_a: None)

    corpus_cli._scan(db, tmp_path / "x.dll", {"package": "p"})

    assert order == [("seed", [0x1000]), ("functions",)]
