"""Tests for surviving bad input.

A corpus run takes hours and touches two hundred binaries built from source
nobody here wrote. Some of them will be malformed - a compilation unit the
compiler emitted oddly, a debug section that ends early. The question is not
whether that happens but what it costs when it does, and the answer has to be
"one binary", never "the run".

This file exists because the answer was once "the run": an ELFParseError
escaped a catch list of standard exception types, fourteen packages into the
scanning phase, after several hours of compiling.
"""

import pytest
from elftools.common.exceptions import ELFError, ELFParseError

from elenchus.corpus.dwarf import DWARF_ERRORS, ground_truth


def test_pyelftools_errors_are_in_the_catch_list():
    """The specific gap that ended a run.

    pyelftools raises from its own exception tree, so a catch list of
    builtins looks thorough and lets every parse error straight through.
    """
    assert issubclass(ELFParseError, DWARF_ERRORS)
    assert issubclass(ELFError, DWARF_ERRORS)


def test_a_file_that_is_not_a_pe_yields_nothing(tmp_path):
    """Not an error: a file with no debug info simply has no ground truth."""
    path = tmp_path / "not_a_pe.bin"
    path.write_bytes(b"this is not a portable executable")

    assert list(ground_truth(str(path))) == []


def test_a_truncated_file_yields_nothing(tmp_path):
    """A PE header with nothing behind it must not raise."""
    path = tmp_path / "truncated.dll"
    path.write_bytes(b"MZ" + b"\x00" * 200)

    assert list(ground_truth(str(path))) == []


def test_a_unit_that_fails_to_parse_does_not_stop_the_file(monkeypatch):
    """One malformed compilation unit costs that unit, not the binary."""
    from elenchus.corpus import dwarf as module

    class Unit:
        def __init__(self, ok):
            self.ok = ok
            self.cu_offset = 0

        def iter_DIEs(self):
            if not self.ok:
                raise ELFParseError("expected 4, found 0")
            return []

    class Dwarf:
        def iter_CUs(self):
            return iter([Unit(False), Unit(True)])

        def line_program_for_CU(self, _cu):
            return None

    monkeypatch.setattr(module, "_dwarf_from_pe", lambda _path: Dwarf())

    # The bad unit is skipped and the good one is reached; neither raises.
    assert list(ground_truth("ignored")) == []


def test_a_broken_iterator_ends_the_file_quietly(monkeypatch):
    """When the unit iterator itself breaks, stop - do not propagate."""
    from elenchus.corpus import dwarf as module

    class Dwarf:
        def iter_CUs(self):
            def generate():
                raise ELFParseError("section ended early")
                yield  # pragma: no cover

            return generate()

    monkeypatch.setattr(module, "_dwarf_from_pe", lambda _path: Dwarf())

    assert list(ground_truth("ignored")) == []


def test_one_bad_binary_does_not_end_a_corpus_run(monkeypatch, db):
    """The loop keeps going and reports what it lost.

    Checked by making the middle optimisation level raise, and asserting the
    ones on either side were still stored.
    """
    import sys
    import types

    sys.modules.setdefault("pyghidra", types.ModuleType("pyghidra"))
    from elenchus.corpus import cli as module

    attempted = []

    def fake_store_level(conn, pkg, opt, debug_path, strip_path):
        attempted.append(opt)
        if opt == "O1":
            raise ELFParseError("expected 4, found 0")
        return 10, 10

    monkeypatch.setattr(module, "store_level", fake_store_level)

    class Result:
        binaries = [f"/tmp/{n}" for n in range(8)]
        ok = True

    class Package:
        name = "probe"
        version = "1"

    args = types.SimpleNamespace(
        db=":memory:", manifest=None, work_dir="/tmp", rebuild=False
    )

    # Drive only the scanning half, which is where the failure landed.
    monkeypatch.setattr(module, "load_manifest", lambda _p: [Package()])
    monkeypatch.setattr(module, "build_package", lambda _p, _w: Result())
    monkeypatch.setattr(module, "connect", lambda _p: db)
    monkeypatch.setattr(module.pyghidra, "start", lambda: None, raising=False)

    assert module.cmd_build_corpus(args) == 0
    assert attempted == ["O0", "O1", "O2", "O3"]


def test_the_run_reports_what_it_could_not_scan(monkeypatch, db, capsys):
    import sys
    import types

    sys.modules.setdefault("pyghidra", types.ModuleType("pyghidra"))
    from elenchus.corpus import cli as module

    def always_fails(*_args, **_kwargs):
        raise ELFParseError("expected 4, found 0")

    class Result:
        binaries = [f"/tmp/{n}" for n in range(8)]
        ok = True

    class Package:
        name = "probe"
        version = "1"

    monkeypatch.setattr(module, "store_level", always_fails)
    monkeypatch.setattr(module, "load_manifest", lambda _p: [Package()])
    monkeypatch.setattr(module, "build_package", lambda _p, _w: Result())
    monkeypatch.setattr(module, "connect", lambda _p: db)
    monkeypatch.setattr(module.pyghidra, "start", lambda: None, raising=False)

    args = types.SimpleNamespace(
        db=":memory:", manifest=None, work_dir="/tmp", rebuild=False
    )
    module.cmd_build_corpus(args)

    out = capsys.readouterr().out
    assert "failed scans    : 4" in out
    assert "ELFParseError" in out


@pytest.mark.parametrize("error", [
    ELFParseError("bad"),
    KeyError("missing"),
    ValueError("bad value"),
    AttributeError("no attribute"),
    IndexError("out of range"),
])
def test_every_expected_parse_failure_is_caught(error):
    assert isinstance(error, DWARF_ERRORS)
