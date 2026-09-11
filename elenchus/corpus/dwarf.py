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

import pefile
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


def _die_name(die):
    attr = die.attributes.get("DW_AT_name")
    if attr is None:
        return None
    value = attr.value
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


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

    file_table = None

    cu_iter = dwarf.iter_CUs()
    while True:
        try:
            cu = next(cu_iter)
        except StopIteration:
            break
        except (AssertionError, KeyError, ValueError):
            # One unit was malformed. The iterator cannot recover its
            # position, so we stop - the units we care about come first.
            break

        try:
            dies = list(cu.iter_DIEs())
        except (AssertionError, KeyError, ValueError):
            continue

        offsets = {die.offset: die for die in dies}

        for die in dies:
            if die.tag != "DW_TAG_subprogram":
                continue

            name = _die_name(die)
            low = die.attributes.get("DW_AT_low_pc")
            if name is None or low is None:
                continue

            ret_ref = die.attributes.get("DW_AT_type")
            ret_die = offsets.get(ret_ref.value + cu.cu_offset) if ret_ref else None
            return_type = _type_name(ret_die, offsets)

            params = []
            for child in die.iter_children():
                if child.tag != "DW_TAG_formal_parameter":
                    continue
                p_ref = child.attributes.get("DW_AT_type")
                p_die = offsets.get(p_ref.value + cu.cu_offset) if p_ref else None
                params.append(_type_name(p_die, offsets))

            line_attr = die.attributes.get("DW_AT_decl_line")

            yield GroundTruthFunction(
                address=low.value,
                name=name,
                return_type=return_type,
                param_types=tuple(params),
                decl_file=file_table,
                decl_line=line_attr.value if line_attr else None,
            )
