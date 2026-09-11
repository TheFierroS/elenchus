"""Tests for the parts of the build driver that need neither network nor gcc.

Downloading and compiling are exercised by hand, not in CI - they need the
network and a cross-compiler. What is tested here is the manifest loading and
the pure helpers, so a broken manifest or a mangled Package is caught.
"""

from pathlib import Path

from elenchus.corpus.build import OPT_LEVELS, Package, load_manifest

MANIFEST = Path(__file__).parent.parent / "elenchus" / "corpus" / "manifest.toml"


def test_manifest_loads():
    """Every manifest entry becomes a Package with the required fields."""
    packages = load_manifest(MANIFEST)
    assert len(packages) >= 1
    for p in packages:
        assert isinstance(p, Package)
        assert p.name
        assert p.url.startswith("https://")
        assert p.style


def test_manifest_names_are_unique():
    """No two packages share a name - names key the corpus."""
    names = [p.name for p in load_manifest(MANIFEST)]
    assert len(names) == len(set(names))


def test_four_optimisation_levels():
    """The corpus is built at O0 through O3."""
    assert OPT_LEVELS == ("O0", "O1", "O2", "O3")
