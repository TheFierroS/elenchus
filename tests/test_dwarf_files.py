"""Tests for resolving DW_AT_decl_file into a source path.

The file table is the basis of provenance: it is how a function compiled
from zlib's own crc32.c is told apart from one the MinGW runtime brought
along. Getting the indexing base wrong shifts every name by one file, which
is the kind of error that produces plausible paths and wrong answers, so the
two DWARF conventions are pinned here.
"""

from elenchus.corpus.dwarf import _file_table


class FakeEntry:
    def __init__(self, name, dir_index=0):
        self.name = name.encode()
        self.dir_index = dir_index


class FakeProgram:
    def __init__(self, version, directories, entries):
        self.header = {
            "version": version,
            "include_directory": [d.encode() for d in directories],
            "file_entry": entries,
        }


class FakeDwarf:
    def __init__(self, program):
        self._program = program

    def line_program_for_CU(self, _cu):
        return self._program


def test_dwarf5_counts_files_from_zero():
    program = FakeProgram(
        5,
        ["/build/zlib-1.3.1", "/usr/share/mingw-w64/include"],
        [FakeEntry("crc32.c", 0), FakeEntry("zlib.h", 1)],
    )
    table = _file_table(FakeDwarf(program), None)

    assert table[0] == "/build/zlib-1.3.1/crc32.c"
    assert table[1] == "/usr/share/mingw-w64/include/zlib.h"


def test_dwarf4_counts_files_from_one():
    """In DWARF 4 the directory index is also one-based, with 0 the comp dir."""
    program = FakeProgram(
        4,
        ["/build/zlib-1.3.1"],
        [FakeEntry("crc32.c", 1), FakeEntry("deflate.c", 0)],
    )
    table = _file_table(FakeDwarf(program), None)

    assert table[1] == "/build/zlib-1.3.1/crc32.c"
    assert table[2] == "deflate.c"      # index 0 means the unit's own dir
    assert 0 not in table


def test_absolute_names_keep_their_own_path():
    program = FakeProgram(5, ["/build"], [FakeEntry("/opt/crt/dtoa.c", 0)])
    assert _file_table(FakeDwarf(program), None)[0] == "/opt/crt/dtoa.c"


def test_missing_line_program_yields_nothing():
    """No line program means no claim about provenance, rather than a guess."""
    class NoProgram:
        def line_program_for_CU(self, _cu):
            return None

    assert _file_table(NoProgram(), None) == {}
