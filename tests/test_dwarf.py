"""Tests for DWARF ground-truth extraction.

These run in CI without Ghidra: reading DWARF needs only pefile and
pyelftools, both installed from pip. The fixture is a tiny MinGW-built DLL
committed under tests/fixtures, so the test is self-contained.
"""

from pathlib import Path

from elenchus.corpus.dwarf import ground_truth

FIXTURE = Path(__file__).parent / "fixtures" / "sample.dll"


def _by_name():
    return {f.name: f for f in ground_truth(FIXTURE)}


def test_finds_the_source_functions():
    """The three functions we wrote are all recovered."""
    funcs = _by_name()
    assert {"add", "strlen_simple", "scale"} <= set(funcs)


def test_signatures_are_correct():
    """Return and parameter types are read, including const and double."""
    funcs = _by_name()
    assert funcs["add"].signature == "int add(int, int)"
    assert funcs["strlen_simple"].signature == "int strlen_simple(const char*)"
    assert funcs["scale"].signature == "double scale(double, int)"


def test_addresses_are_present():
    """Every recovered function has a real entry-point address."""
    for f in ground_truth(FIXTURE):
        assert f.address > 0


def test_decl_lines_are_read():
    """Source line numbers survive into the ground truth."""
    funcs = _by_name()
    assert funcs["add"].decl_line == 1


def test_no_debug_info_yields_nothing():
    """A file without DWARF produces no ground truth, not an error."""
    # The stripped fixture, if present, or any non-DWARF file. Here we point
    # at this test file itself: not a PE, so extraction yields nothing.
    result = list(ground_truth(Path(__file__)))
    assert result == []
