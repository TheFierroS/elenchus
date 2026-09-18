"""Tests for resolving DW_AT_decl_file into a source path.

The file table is the basis of provenance: it is how a function compiled
from zlib's own crc32.c is told apart from one the MinGW runtime brought
along. Getting the indexing base wrong shifts every name by one file, which
is the kind of error that produces plausible paths and wrong answers, so the
two DWARF conventions are pinned here.
"""

import types

import pytest

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


# ---------------------------------------------------------------- #line names


class FakeUnit:
    """A compilation unit whose top DIE names its own source file."""

    def __init__(self, source):
        self.source = source

    def get_top_DIE(self):
        attributes = {}
        if self.source is not None:
            attributes["DW_AT_name"] = types.SimpleNamespace(value=self.source.encode())
        return types.SimpleNamespace(attributes=attributes)


DUKTAPE = "data/corpus/duktape/src/duktape-2.7.0/src"


def duktape_like(version=5):
    """The shape measured in duktape's real DWARF: the build directory is
    directory 0, #line names sit under it without a directory, the unit's own
    source and headers under theirs."""
    if version >= 5:
        directories = ["/home/someone/elenchus", DUKTAPE, "/usr/share/mingw-w64/include"]
        return FakeProgram(5, directories, [
            FakeEntry("duktape.c", 1), FakeEntry("duk_api_stack.c", 0),
            FakeEntry("duk_config.h", 1), FakeEntry("math.h", 2)])
    return FakeProgram(4, [DUKTAPE, "/usr/share/mingw-w64/include"], [
        FakeEntry("duktape.c", 1), FakeEntry("duk_api_stack.c", 0),
        FakeEntry("math.h", 2)])


def test_a_line_directive_name_lives_in_its_package_not_in_the_build_directory():
    table = _file_table(FakeDwarf(duktape_like()), FakeUnit(f"{DUKTAPE}/duktape.c"))

    assert table[1] == f"{DUKTAPE}/duk_api_stack.c"
    assert not any(path.startswith("/home/") for path in table.values())
    # Everything that was not a bare name under directory 0 is untouched.
    assert table[0] == f"{DUKTAPE}/duktape.c"
    assert table[2] == f"{DUKTAPE}/duk_config.h"
    assert table[3] == "/usr/share/mingw-w64/include/math.h"


def test_dwarf4_line_directive_names_are_anchored_too():
    table = _file_table(FakeDwarf(duktape_like(4)), FakeUnit(f"{DUKTAPE}/duktape.c"))
    assert table[2] == f"{DUKTAPE}/duk_api_stack.c"
    assert table[1] == f"{DUKTAPE}/duktape.c"


def test_the_same_line_name_in_two_packages_stays_two_files():
    """Bare, both would read "utf8.c" - one file shared by two packages, which
    the dataset drops from both."""
    def path(package):
        program = FakeProgram(5, ["/build"], [FakeEntry("utf8.c", 0)])
        return _file_table(FakeDwarf(program),
                           FakeUnit(f"data/corpus/{package}/src/all.c"))[0]

    assert path("one") == "data/corpus/one/src/utf8.c"
    assert path("two") == "data/corpus/two/src/utf8.c"


@pytest.mark.parametrize("name", ["data/corpus/zlib/src/inflate.c", "sub/inner.c",
                                  "sub\\inner.c"])
def test_a_name_that_carries_a_directory_keeps_the_build_directory(name):
    """Only bare names are #line's; a path given with a directory was given
    relative to where the compiler ran."""
    program = FakeProgram(5, ["/build"], [FakeEntry(name, 0)])
    table = _file_table(FakeDwarf(program), FakeUnit("data/corpus/zlib/src/inflate.c"))
    assert table[0] == f"/build/{name}"


@pytest.mark.parametrize("name", ["/opt/crt/dtoa.c", "C:/crt/dtoa.c", "\\crt\\dtoa.c",
                                  "C:dtoa.c"])
def test_an_absolute_name_is_never_anchored(name):
    program = FakeProgram(5, ["/build"], [FakeEntry(name, 0)])
    assert _file_table(FakeDwarf(program), FakeUnit(f"{DUKTAPE}/duktape.c"))[0] == name


@pytest.mark.parametrize("unit", [None, FakeUnit(None), FakeUnit("duktape.c")])
def test_without_a_source_directory_to_anchor_at_nothing_changes(unit):
    program = FakeProgram(5, ["/build"], [FakeEntry("duk_api_stack.c", 0)])
    assert _file_table(FakeDwarf(program), unit)[0] == "/build/duk_api_stack.c"


def test_a_unit_that_cannot_be_read_falls_back_instead_of_failing():
    class Broken:
        def get_top_DIE(self):
            raise KeyError("malformed")

    program = FakeProgram(5, ["/build"], [FakeEntry("duk_api_stack.c", 0)])
    assert _file_table(FakeDwarf(program), Broken())[0] == "/build/duk_api_stack.c"


def test_a_windows_style_unit_path_anchors_with_forward_slashes():
    program = FakeProgram(5, ["C:/build"], [FakeEntry("duk_api_stack.c", 0)])
    unit = FakeUnit("data\\corpus\\duktape\\src\\duktape.c")
    assert _file_table(FakeDwarf(program), unit)[0] == (
        "data/corpus/duktape/src/duk_api_stack.c")


# ------------------------------------------------- generated files, a directory
#
# yara's parsers and lexers come from bison and flex, which ran in the package
# root and wrote "#line 1 \"libyara/lexer.c\"". GCC files those under a
# relative directory entry of their own, so the name arrives with a directory
# in front of it. Measured on yara 4.5.8 at all four levels: 437 rows in the
# four generated units, every one of them landing in the build directory
# before this, none after.

YARA = "data/corpus/yara/src/yara-4.5.8/libyara"


def yara_like(version=5):
    if version >= 5:
        directories = ["/home/someone/elenchus", YARA,
                       "/usr/share/mingw-w64/include", "libyara"]
        return FakeProgram(5, directories, [
            FakeEntry("hex_lexer.c", 1), FakeEntry("stdio.h", 2),
            FakeEntry("hex_lexer.c", 3), FakeEntry("hex_lexer.l", 3)])
    return FakeProgram(4, [YARA, "/usr/share/mingw-w64/include", "libyara"], [
        FakeEntry("hex_lexer.c", 1), FakeEntry("stdio.h", 2),
        FakeEntry("hex_lexer.c", 3), FakeEntry("hex_lexer.l", 3)])


def test_a_generated_units_own_directory_is_anchored_in_the_package():
    table = _file_table(FakeDwarf(yara_like()), FakeUnit(f"{YARA}/hex_lexer.c"))

    assert table[2] == f"{YARA}/hex_lexer.c"
    assert table[3] == f"{YARA}/hex_lexer.l"
    # The units and headers that carry an absolute directory are untouched.
    assert table[0] == f"{YARA}/hex_lexer.c"
    assert table[1] == "/usr/share/mingw-w64/include/stdio.h"


def test_dwarf4_relative_directories_are_anchored_too():
    table = _file_table(FakeDwarf(yara_like(4)), FakeUnit(f"{YARA}/hex_lexer.c"))
    assert table[3] == f"{YARA}/hex_lexer.c"
    assert table[4] == f"{YARA}/hex_lexer.l"


def test_the_deepest_matching_ancestor_wins():
    """A package whose own directory repeats the name: the generator ran in
    the nearer one, not in the checkout above it."""
    program = FakeProgram(5, ["/build", "src"], [FakeEntry("parser.c", 1)])
    unit = FakeUnit("data/corpus/pkg/src/vendor/src/unit.c")
    assert _file_table(FakeDwarf(program), unit)[0] == (
        "data/corpus/pkg/src/vendor/src/parser.c")


def test_a_relative_directory_no_ancestor_matches_is_left_alone():
    program = FakeProgram(5, ["/build", "generated"], [FakeEntry("parser.c", 1)])
    unit = FakeUnit("data/corpus/pkg/src/unit.c")
    assert _file_table(FakeDwarf(program), unit)[0] == "generated/parser.c"


def test_a_relative_directory_with_nothing_above_it_is_left_alone():
    """Anchoring at the first component would leave no directory at all, and
    a bare relative path is what the old resolution already gives."""
    program = FakeProgram(5, ["/build", "data"], [FakeEntry("parser.c", 1)])
    assert _file_table(FakeDwarf(program), FakeUnit("data/unit.c"))[0] == "data/parser.c"


def test_a_relative_directory_without_a_unit_to_anchor_at_is_left_alone():
    program = FakeProgram(5, ["/build", "libyara"], [FakeEntry("lexer.c", 1)])
    assert _file_table(FakeDwarf(program), None)[0] == "libyara/lexer.c"


def test_an_absolute_name_under_a_relative_directory_keeps_its_path():
    program = FakeProgram(5, ["/build", "libyara"], [FakeEntry("/opt/crt/dtoa.c", 1)])
    table = _file_table(FakeDwarf(program), FakeUnit(f"{YARA}/hex_lexer.c"))
    assert table[0] == "/opt/crt/dtoa.c"
