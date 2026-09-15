"""Tests for turning a manifest entry into a list of sources to compile.

The old schema could name two places: the source root, or src/. Three of the
four candidate packages that failed on the first corpus run kept their
sources in lib/, and there was no way to say so. These tests pin the glob
behaviour that replaced it, and the failure modes that cost the most time to
diagnose - a pattern that matches nothing, and a file that another file
#includes being compiled twice.
"""

import pytest

from elenchus.corpus.build import Package, _c_sources, load_manifest

MANIFEST = "elenchus/corpus/manifest.toml"


@pytest.fixture
def tree(tmp_path):
    """A source tree shaped like the packages that gave us trouble."""
    for relative in (
        "main.c",
        "helper.c",
        "test.c",
        "lib/core.c",
        "lib/frame.c",
        "lib/x86/cpu_features.c",
        "src/parser.c",
        "tests/test_parser.c",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int f(void) { return 0; }\n")
    return tmp_path


def package(**kwargs):
    defaults = {
        "name": "probe", "version": "1", "url": "https://example.invalid/x.tar.gz",
        "license": "MIT",
    }
    return Package(**{**defaults, **kwargs})


def names(paths):
    return sorted(p.name for p in paths)


def test_root_glob_stays_in_the_root(tree):
    found = _c_sources(tree, package(sources=["*.c"]))
    assert names(found) == ["helper.c", "main.c", "test.c"]


def test_sources_can_live_anywhere(tree):
    """The case that broke lz4, libbase64 and libdeflate."""
    found = _c_sources(tree, package(sources=["lib/*.c", "lib/x86/*.c"]))
    assert names(found) == ["core.c", "cpu_features.c", "frame.c"]


def test_several_patterns_are_merged_without_duplicates(tree):
    found = _c_sources(tree, package(sources=["lib/*.c", "lib/*.c", "src/*.c"]))
    assert names(found) == ["core.c", "frame.c", "parser.c"]


def test_exclude_matches_by_file_name(tree):
    found = _c_sources(tree, package(sources=["*.c"], exclude=["test.c"]))
    assert names(found) == ["helper.c", "main.c"]


def test_a_pattern_matching_nothing_says_so(tree):
    """Otherwise this surfaces as a compiler error about missing input."""
    with pytest.raises(ValueError, match="no sources matched"):
        _c_sources(tree, package(sources=["nowhere/*.c"]))


def test_the_old_style_field_still_works(tree):
    """Entries written before sources existed must keep building."""
    assert names(_c_sources(tree, package(style="c_glob"))) == [
        "helper.c", "main.c", "test.c",
    ]
    assert names(_c_sources(tree, package(style="c_glob_src"))) == ["parser.c"]


def test_the_shipped_manifest_parses():
    packages = load_manifest(MANIFEST)

    assert len(packages) > 20
    assert len({p.name for p in packages}) == len(packages)

    for entry in packages:
        assert entry.url.startswith("https://")
        assert entry.license
        assert entry.sources or entry.style


def test_link_libraries_come_after_the_sources(tmp_path):
    """The GNU linker resolves in command-line order and never looks back.

    A -lws2_32 placed before the file that needs it silently resolves
    nothing, and the error that follows names an undefined symbol rather
    than the ordering that caused it.
    """
    from elenchus.corpus.build import _compile_level

    (tmp_path / "a.c").write_text("int f(void) { return 0; }\n")
    entry = package(sources=["*.c"], libs=["ws2_32"])

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded.setdefault("cmd", cmd)
        raise RuntimeError("stop before actually compiling")

    import subprocess
    original = subprocess.run
    subprocess.run = fake_run
    try:
        _compile_level(
            [tmp_path / "a.c"], tmp_path / "o.dll", tmp_path / "s.dll",
            "O0", entry, tmp_path,
        )
    except RuntimeError:
        pass
    finally:
        subprocess.run = original

    cmd = recorded["cmd"]
    assert cmd[-1] == "-lws2_32"
    assert cmd.index("-lws2_32") > cmd.index(str(tmp_path / "a.c"))


def test_the_shipped_manifest_stays_varied():
    """An encoder trained only on compressors learns compressors.

    Not a property of the code, but of the corpus, and the corpus is what
    every later measurement rests on - so it is checked here rather than
    remembered.
    """
    names_ = {p.name for p in load_manifest(MANIFEST)}

    for expected in ("mbedtls", "libexpat", "lua", "zstd",
                     "sqlite", "libuv", "miniaudio", "stb"):
        assert expected in names_, f"{expected} keeps the corpus varied"


def test_every_package_has_a_known_domain_and_every_domain_can_fill_three_splits():
    """The stratified split only guarantees coverage for domains of three or more."""
    from collections import Counter

    from elenchus.corpus.dataset import DOMAINS

    packages = load_manifest(MANIFEST)
    unknown = [p.name for p in packages if p.domain not in DOMAINS]
    assert not unknown, f"missing or unknown domain: {unknown}"

    counts = Counter(p.domain for p in packages)
    assert all(counts[d] >= 3 for d in DOMAINS), counts
