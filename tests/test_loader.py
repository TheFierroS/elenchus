"""The Loader parses only what a run uses, and gets what a full parse gives.

Loader reads a PE's headers and its import table and nothing else (F47). The
emulator reads six things from it - base, size, the import slots and traps,
the header bytes and each section's bytes - and if a cheaper parse ever gave
a different one, every run on that binary would map different memory. So
each is compared with what a full parse yields, on both fixture builds.
"""

from pathlib import Path

import pytest

pytest.importorskip("unicorn")

import pefile  # noqa: E402

from elenchus.emulation.harness import PAGE, Loader  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"
BUILDS = ("cases_O0.dll", "cases_O3.dll")
FIELDS = ("base", "size", "slots", "traps", "_header", "_sections")


def _full_parse(path):
    """What the Loader held when it parsed everything, before F47."""
    pe = pefile.PE(str(path), fast_load=False)
    reference = Loader.__new__(Loader)
    reference.base = pe.OPTIONAL_HEADER.ImageBase
    reference.size = (pe.OPTIONAL_HEADER.SizeOfImage + PAGE - 1) & ~(PAGE - 1)
    reference.slots, reference.traps = reference._import_traps(pe)
    reference._header = bytes(pe.header)
    reference._sections = [(s.VirtualAddress, s.get_data())
                           for s in pe.sections if s.get_data()]
    pe.close()
    return reference


@pytest.mark.parametrize("build", BUILDS)
def test_the_loader_matches_a_full_parse(build):
    path = FIXTURES / build
    loader, reference = Loader(path), _full_parse(path)
    for field in FIELDS:
        assert getattr(loader, field) == getattr(reference, field), field


@pytest.mark.parametrize("build", BUILDS)
def test_the_loader_still_finds_the_imports(build):
    """Two parses that both missed the import table would agree with each
    other; the fixtures call imports, so there must be traps to find."""
    assert Loader(FIXTURES / build).traps
