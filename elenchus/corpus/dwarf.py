"""Read ground truth from the DWARF debug info inside a MinGW-built PE.

The debug half of a compiled binary carries what the stripped half loses:
the real function names, their signatures, their types, the source line each
came from. This is the ground truth the whole project is measured against,
and the training pairs the encoder learns from - all of it for free, no
human labelling, because the compiler wrote it down.

Why extract the sections by hand. pyelftools reads DWARF well but expects an
ELF wrapper; a PE is not one. MinGW also stores long section names (.debug_info
is over eight characters) as a '/N' reference into the COFF string table
rather than inline, so the names have to be resolved before the sections can
be handed to pyelftools. Once resolved, pyelftools does the hard part -
walking the DIE tree, following type references, decoding forms.

Reading DWARF straight from the PE keeps ground truth independent of Ghidra,
whose own DWARF interpretation is unreliable here, and independent of the
container format: the same DIE-walking code would read an ELF's DWARF too,
which is where a future Linux corpus would begin.
"""

import io
from dataclasses import dataclass
from struct import error as struct_error

import pefile
from elftools.common.exceptions import ELFError
from elftools.dwarf.dwarfinfo import (
    DebugSectionDescriptor,
    DwarfConfig,
    DWARFInfo,
)


@dataclass(frozen=True)
class GroundTruthFunction:
    """One function as the compiler recorded it, before stripping.

    address is the key that ties this back to the stripped binary: strip
    removes symbols, not code, so the entry point stays put.

    decl_file is the source file the compiler recorded. It is what separates
    the code we chose to compile from the runtime the toolchain linked in
    behind us, and that separation decides what may be trained on.
    """

    address: int
    name: str
    return_type: str
    param_types: tuple[str, ...]
    decl_file: str | None
    decl_line: int | None

    @property
    def signature(self) -> str:
        return f"{self.return_type} {self.name}({', '.join(self.param_types)})"


def _resolve_section_names(pe, raw):
    """Return {resolved_name: bytes} for every section.

    A section named '/N' points at offset N in the COFF string table, which
    sits right after the symbol table. Short names are taken as-is.
    """
    strtab_off = (
        pe.FILE_HEADER.PointerToSymbolTable
        + pe.FILE_HEADER.NumberOfSymbols * 18
    )

    def resolve(name_field):
        name = name_field.rstrip(b"\x00").decode("utf-8", "replace")
        if name.startswith("/"):
            pos = strtab_off + int(name[1:])
            end = raw.index(b"\x00", pos)
            return raw[pos:end].decode("utf-8", "replace")
        return name

    return {resolve(s.Name): s.get_data() for s in pe.sections}


def _dwarf_from_pe(path):
    """Build a DWARFInfo from the debug sections of a PE, or return None.

    None means the binary carries no .debug_info - it was built without -g,
    or already stripped. That is a fact about the binary, not an error.
    """
    with open(path, "rb") as f:
        raw = f.read()

    try:
        pe = pefile.PE(path, fast_load=True)
    except pefile.PEFormatError:
        # Not a PE at all - a source file, an ELF, a text blob. No ground
        # truth to read, and that is a fact about the input, not an error.
        return None

    sections = _resolve_section_names(pe, raw)

    if not sections.get(".debug_info"):
        return None

    def desc(name):
        blob = sections.get(name, b"")
        if not blob:
            return None
        return DebugSectionDescriptor(io.BytesIO(blob), name, 0, len(blob), 0)

    return DWARFInfo(
        config=DwarfConfig(
            little_endian=True, default_address_size=8, machine_arch="x86-64"
        ),
        debug_info_sec=desc(".debug_info"),
        debug_aranges_sec=desc(".debug_aranges"),
        debug_abbrev_sec=desc(".debug_abbrev"),
        debug_frame_sec=desc(".debug_frame"),
        eh_frame_sec=None,
        debug_str_sec=desc(".debug_str"),
        debug_loc_sec=None,
        debug_ranges_sec=None,
        debug_line_sec=desc(".debug_line"),
        debug_addr_sec=desc(".debug_addr"),
        debug_str_offsets_sec=desc(".debug_str_offsets"),
        debug_line_str_sec=desc(".debug_line_str"),
        debug_pubtypes_sec=None,
        debug_pubnames_sec=None,
        debug_sup_sec=None,
        gnu_debugaltlink_sec=None,
        debug_loclists_sec=desc(".debug_loclists"),
        debug_rnglists_sec=desc(".debug_rnglists"),
        debug_types_sec=None,
    )


# Everything that can go wrong reading malformed debug information. ELFError
# belongs here and was missing: pyelftools raises from its own exception tree,
# not from Python's builtins, so a catch list of standard types looks
# thorough and silently lets ELFParseError through. One such error, in one
# compilation unit of one binary, ended a corpus run that had been going for
# hours.
DWARF_ERRORS = (ELFError, AssertionError, KeyError, ValueError, AttributeError,
                IndexError, struct_error)


def _decode(value):
    """Return a str for a DWARF string attribute, whatever form it took."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _file_table(dwarf, cu):
    """Return {index: path} for the source files this unit was built from.

    DW_AT_decl_file is an index, not a name: the compiler writes the file
    list once per unit and refers to it by number. Resolving it is what makes
    provenance possible - telling a function we compiled from one the
    toolchain brought along, without keeping a list of runtime function names
    that would rot with the next MinGW release.

    The indexing base moved in DWARF 5: file 0 became the unit's own primary
    source, where DWARF 4 started counting at 1. Both are handled because the
    corpus may be rebuilt with a different GCC.

    A unit with no line program yields an empty table; the caller then leaves
    decl_file unset rather than guessing.

    A relative name filed under directory 0 is anchored at the unit's own
    source directory, not at the compilation directory. The corpus compiles
    every source by its path (data/corpus/<package>/...), so its files carry
    their directory and headers come through -I; the only relative names left
    under directory 0 are ones a #line directive spelled without a directory.
    duktape's amalgamation has 145 of them ("duk_api_stack.c"), and resolving
    them against the compilation directory put every duktape function in the
    root of whichever checkout built it - an identity that changed from
    machine to machine. Left bare they would be worse: two packages each with
    a #line "utf8.c" would look like one file shared between packages, and
    the dataset drops those. Anchored, such a name lives in its package. The
    path names where the directive points, not a file that exists on disk.

    A name that carries a directory is anchored too, at the ancestor the
    generator ran in; see _anchored. That covers bison and flex output, which
    names its source relative to the package root rather than to the unit.
    """
    try:
        program = dwarf.line_program_for_CU(cu)
    except DWARF_ERRORS:
        return {}

    if program is None:
        return {}

    anchor = _unit_directory(cu)
    header = program.header
    version = header.get("version", 4)
    directories = [_decode(d) for d in header.get("include_directory", [])]
    entries = header.get("file_entry", [])

    table = {}
    for position, entry in enumerate(entries):
        name = _decode(entry.name)
        index = position if version >= 5 else position + 1

        directory = None
        dir_index = getattr(entry, "dir_index", None)
        if dir_index is not None:
            if version >= 5:
                if 0 <= dir_index < len(directories):
                    directory = directories[dir_index]
            elif 1 <= dir_index <= len(directories):
                directory = directories[dir_index - 1]

        anchored = None
        if not _is_absolute(name):
            if directory is not None and not _is_absolute(directory):
                anchored = _anchored(anchor, f"{directory}/{name}")
            elif dir_index == 0:
                anchored = _anchored(anchor, name)

        if anchored:
            table[index] = anchored
        elif directory and not _is_absolute(name):
            table[index] = f"{directory}/{name}"
        else:
            table[index] = name

    return table


def _anchored(anchor, name):
    """Where a relative name filed under directory 0 points, or None.

    A bare name belongs to the unit's own directory, as duktape's #line names
    do. A name that carries a directory was written by a generator that ran
    somewhere above the unit - bison and flex spell yara's lexers
    "libyara/lexer.c", relative to the package root they ran in, not to
    libyara/ where the unit sits. GCC files those under a relative directory
    entry of their own ("libyara"), so the caller hands the two back joined.
    Resolved against the compilation directory they land in the root of
    whichever checkout built them, the same machine-dependent identity bare
    duktape names once had.

    The generator's directory is found by walking up from the unit: the
    deepest ancestor whose name is the first component of the relative name.
    Nothing in this corpus is compiled by a name relative to the build
    directory - every source is passed as data/corpus/<package>/... - so a
    relative name under directory 0 can only be a directive's. When no
    ancestor matches, or the match is the path's first component and leaves
    no directory to anchor at, the caller keeps its old resolution rather
    than invent one.
    """
    if not anchor:
        return None

    name = name.replace("\\", "/")
    head, slash, _rest = name.partition("/")
    if not slash:
        return f"{anchor}/{name}"

    parts = anchor.split("/")
    for depth in range(len(parts), 0, -1):
        if parts[depth - 1] != head:
            continue
        base = "/".join(parts[:depth - 1])
        return f"{base}/{name}" if base else None

    return None


def _is_absolute(name):
    return name.startswith(("/", "\\")) or ":" in name[:3]


def _unit_directory(cu):
    """The directory of the unit's own source file, or None if it has none."""
    if cu is None:
        return None
    try:
        attribute = cu.get_top_DIE().attributes.get("DW_AT_name")
    except DWARF_ERRORS:
        return None
    if attribute is None:
        return None
    source = _decode(attribute.value).replace("\\", "/")
    directory, _slash, _file = source.rpartition("/")
    return directory or None


def _die_name(die):
    attr = die.attributes.get("DW_AT_name")
    if attr is None:
        return None
    return _decode(attr.value)


def _type_name(die, offsets, depth=0):
    """Turn a type DIE into a readable C type, following references.

    depth guards against a cyclic or pathological type graph rather than
    trusting the debug info to be well-formed.
    """
    if die is None:
        return "void"
    if depth > 16:
        return "?"

    tag = die.tag

    def inner():
        ref = die.attributes.get("DW_AT_type")
        if ref is None:
            return None
        return offsets.get(ref.value + die.cu.cu_offset)

    if tag in ("DW_TAG_base_type", "DW_TAG_typedef"):
        return _die_name(die) or "?"
    if tag == "DW_TAG_pointer_type":
        return _type_name(inner(), offsets, depth + 1) + "*"
    if tag == "DW_TAG_const_type":
        return "const " + _type_name(inner(), offsets, depth + 1)
    if tag == "DW_TAG_volatile_type":
        return "volatile " + _type_name(inner(), offsets, depth + 1)
    if tag == "DW_TAG_structure_type":
        return "struct " + (_die_name(die) or "anon")
    if tag == "DW_TAG_union_type":
        return "union " + (_die_name(die) or "anon")
    if tag == "DW_TAG_enumeration_type":
        return "enum " + (_die_name(die) or "anon")
    if tag == "DW_TAG_array_type":
        return _type_name(inner(), offsets, depth + 1) + "[]"

    name = _die_name(die)
    return name if name else tag.replace("DW_TAG_", "")


def ground_truth(path):
    """Yield a GroundTruthFunction for every defined function with an address.

    Compilation units that pyelftools cannot parse - a malformed header, an
    empty unit from the CRT - are skipped rather than aborting the whole file.
    A binary is a mix of the code we compiled and runtime glue; the glue is
    allowed to be messy.
    """
    dwarf = _dwarf_from_pe(path)
    if dwarf is None:
        return

    cu_iter = dwarf.iter_CUs()
    while True:
        try:
            cu = next(cu_iter)
        except StopIteration:
            break
        except DWARF_ERRORS:
            # One unit was malformed. The iterator cannot recover its
            # position, so we stop - the units we care about come first.
            break

        try:
            dies = list(cu.iter_DIEs())
        except DWARF_ERRORS:
            continue

        offsets = {die.offset: die for die in dies}
        file_table = _file_table(dwarf, cu)

        for die in dies:
            if die.tag != "DW_TAG_subprogram":
                continue

            # pyelftools decodes attributes lazily, so a malformed form only
            # raises here, on the one function that carries it. Losing that
            # function is acceptable; losing the unit around it is not.
            try:
                function = _subprogram(die, offsets, file_table, cu.cu_offset)
            except DWARF_ERRORS:
                continue

            if function is not None:
                yield function


def _subprogram(die, offsets, file_table, cu_offset):
    """Build a GroundTruthFunction from one subprogram DIE, or None."""
    name = _die_name(die)
    low = die.attributes.get("DW_AT_low_pc")
    if name is None or low is None:
        return None

    ret_ref = die.attributes.get("DW_AT_type")
    ret_die = offsets.get(ret_ref.value + cu_offset) if ret_ref else None

    params = []
    for child in die.iter_children():
        if child.tag != "DW_TAG_formal_parameter":
            continue
        p_ref = child.attributes.get("DW_AT_type")
        p_die = offsets.get(p_ref.value + cu_offset) if p_ref else None
        params.append(_type_name(p_die, offsets))

    line_attr = die.attributes.get("DW_AT_decl_line")
    file_attr = die.attributes.get("DW_AT_decl_file")

    return GroundTruthFunction(
        address=low.value,
        name=name,
        return_type=_type_name(ret_die, offsets),
        param_types=tuple(params),
        decl_file=file_table.get(file_attr.value) if file_attr else None,
        decl_line=line_attr.value if line_attr else None,
    )
