"""Tests for turning a manifest entry into a list of sources to compile.

The old schema could name two places: the source root, or src/. Three of the
four candidate packages that failed on the first corpus run kept their
sources in lib/, and there was no way to say so. These tests pin the glob
behaviour that replaced it, and the failure modes that cost the most time to
diagnose - a pattern that matches nothing, and a file that another file
#includes being compiled twice.
"""

from pathlib import Path

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

    # Wave 2 (docs/experiments.md, L): at the same number of functions, more
    # packages scored higher. The additions that carry the thin domains and
    # the new one are pinned so that none is dropped unnoticed.
    for expected in ("lexbor", "tidy-html5", "oniguruma", "mpack", "libcbor",
                     "libwebp", "flac", "opus", "c-ares", "cglm", "chipmunk2d",
                     "libtommath", "wasm3", "speexdsp", "enet"):
        assert expected in names_, f"{expected} was added in wave 2"

    # The second wave-2 batch. The large ones carry most of the identities and
    # the three binary-analysis packages are the whole of a new domain, so
    # losing any of them silently would change what the corpus teaches.
    for expected in ("libxml2", "libarchive", "freetype", "curl", "xz",
                     "libjpeg-turbo", "openjpeg", "libpng", "vorbis",
                     "minizip-ng", "secp256k1", "chibi-scheme", "pdcurses",
                     "qhull", "box2d", "capstone", "yara", "zydis"):
        assert expected in names_, f"{expected} was added in wave 2"


def test_the_wave_3_packages_that_carry_a_domain_are_there():
    """Wave 3 was chosen for variety, not size (docs/experiments.md, L2): the
    package unit paid 5.9 T where the identity unit paid 4.1 T. These carry
    the domains that were thinnest, so losing one silently would undo it."""
    names_ = {p.name for p in load_manifest(MANIFEST)}

    for expected in ("leptonica", "jerryscript", "nats-c", "libevent", "gravity",
                     "mathc", "sgscript", "nanomsg", "libmdbx", "superlu",
                     "portaudio", "speex", "plutovg", "rhash", "imath", "toy",
                     "umka-lang", "hiredis", "glfw", "libharu", "libxlsxwriter"):
        assert expected in names_, f"{expected} was added in wave 3"


def test_wave_2_did_not_leave_any_domain_thin():
    """After wave 2 the two thinnest domains in train had at least 15 packages."""
    from collections import Counter

    counts = Counter(p.domain for p in load_manifest(MANIFEST))
    assert counts["text"] >= 15 and counts["data-format"] >= 15, counts
    assert counts["math"] >= 3, counts
    # binary-analysis is the corpus's own subject matter: disassemblers and
    # scanners. It opened with exactly three, which is the least a stratified
    # split can place in all three parts, and three is where it stayed: every
    # other such library in C is either GPL or C++ (docs/experiments.md, F11).
    assert counts["binary-analysis"] >= 3, counts
    # Wave 3 split media and systems, which had grown to hold unlike things.
    # Each half has to stand on its own.
    for half in ("image", "audio", "networking", "systems"):
        assert counts[half] >= 6, counts


def test_every_package_has_a_known_domain_and_every_domain_can_fill_three_splits():
    """The stratified split only guarantees coverage for domains of three or more."""
    from collections import Counter

    from elenchus.corpus.dataset import DOMAINS

    packages = load_manifest(MANIFEST)
    unknown = [p.name for p in packages if p.domain not in DOMAINS]
    assert not unknown, f"missing or unknown domain: {unknown}"

    counts = Counter(p.domain for p in packages)
    assert all(counts[d] >= 3 for d in DOMAINS), counts


# ----------------------------------------------------------- exclude patterns


def test_exclude_takes_a_glob(tree):
    """nng keeps 52 tests beside the code they test, all of them *_test.c."""
    (tree / "lib" / "core_test.c").write_text("int t(void) { return 0; }\n")
    found = _c_sources(tree, package(sources=["lib/*.c"], exclude=["*_test.c"]))
    assert names(found) == ["core.c", "frame.c"]


def test_a_plain_exclude_name_is_still_one_file(tree):
    """A name with no wildcard in it must not start matching neighbours."""
    found = _c_sources(tree, package(sources=["*.c", "tests/*.c"],
                                     exclude=["test.c"]))
    assert names(found) == ["helper.c", "main.c", "test_parser.c"]


def test_exclude_matching_is_case_sensitive(tree):
    (tree / "Test.c").write_text("int f(void) { return 0; }\n")
    found = _c_sources(tree, package(sources=["*.c"], exclude=["test.c"]))
    assert "Test.c" in names(found)
    assert "test.c" not in names(found)


def test_the_matcher_does_not_follow_the_platforms_case_rules():
    """fnmatch folds case on Windows and macOS and not on Linux, so the test
    above cannot see the difference here - but which sources a package
    compiles must not depend on where it is compiled, and CI is to grow a
    Windows and a macOS runner."""
    import fnmatch

    from elenchus.corpus import build

    assert build.fnmatchcase is fnmatch.fnmatchcase


# ---------------------------------------------------------------- depends


DEPENDENT = """
[[package]]
name = "zydis"
version = "4.1.0"
url = "https://example.invalid/zydis.tar.gz"
license = "MIT"
domain = "binary-analysis"
sources = ["src/*.c"]
include_dirs = ["include"]

[[package.depends]]
name = "zycore"
version = "1.5.0"
url = "https://example.invalid/zycore.tar.gz"
license = "MIT"
sources = ["src/*.c"]
include_dirs = ["include"]
defines = ["ZYAN_NO_LIBC"]
"""


def test_a_dependency_is_loaded_as_a_package(tmp_path):
    path = tmp_path / "m.toml"
    path.write_text(DEPENDENT)

    zydis = load_manifest(path)[0]

    assert [d.name for d in zydis.depends] == ["zycore"]
    assert isinstance(zydis.depends[0], Package)
    assert zydis.depends[0].defines == ["ZYAN_NO_LIBC"]


def test_the_packages_that_need_another_library_name_it():
    """These four cannot be compiled from a tarball of their own."""
    depends = {p.name: [d.name for d in p.depends] for p in load_manifest(MANIFEST)
               if p.depends}

    assert depends == {
        "minizip-ng": ["zlib"], "libpng": ["zlib"], "vorbis": ["ogg"],
        "zydis": ["zycore"], "libxlsxwriter": ["zlib"], "nats-c": ["protobuf-c"],
        "usockets": ["libuv"], "libcyaml": ["libyaml"], "speex": ["ogg"],
        "libspng": ["zlib"], "opusfile": ["ogg", "opus"]}


def test_a_dependency_is_compiled_with_the_package(tmp_path, monkeypatch):
    """Its sources join the compile, its include dirs and macros come with
    them, and it is extracted outside the package's own source tree so the
    dataset can tell the two apart."""
    from elenchus.corpus import build as module

    def fake_download(url, dest_dir):
        root = Path(dest_dir) / ("zycore-1.5.0" if "zycore" in url else "zydis-4.1.0")
        (root / "src").mkdir(parents=True)
        (root / "include").mkdir()
        name = "zycore.c" if "zycore" in url else "zydis.c"
        (root / "src" / name).write_text("int f(void) { return 0; }\n")
        return root

    recorded = {}

    def fake_compile(sources, debug, stripped, opt, pkg=None, root=None, extra=()):
        recorded.setdefault("sources", [str(s) for s in sources])
        recorded.setdefault("extra", extra)
        Path(debug).write_bytes(b"")
        Path(stripped).write_bytes(b"")

    monkeypatch.setattr(module, "_download", fake_download)
    monkeypatch.setattr(module, "_compile_level", fake_compile)

    path = tmp_path / "m.toml"
    path.write_text(DEPENDENT)
    result = module.build_package(load_manifest(path)[0], tmp_path / "work")

    assert result.ok
    assert [Path(s).name for s in recorded["sources"]] == ["zydis.c", "zycore.c"]
    assert f"/{module.DEPENDENCY_DIR}/zycore/" in recorded["sources"][1]
    assert f"/{module.DEPENDENCY_DIR}/" not in recorded["sources"][0]
    assert [d.name for d, _root in recorded["extra"]] == ["zycore"]


def test_a_dependencys_flags_follow_its_sources(tmp_path):
    """-I and -D of both go on the command line, the package's first."""
    (tmp_path / "a.c").write_text("int f(void) { return 0; }\n")
    main = package(sources=["*.c"], include_dirs=["include"], defines=["MAIN=1"])
    dep = package(name="dep", include_dirs=["inc"], defines=["DEP=1"], libs=["ws2_32"])

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded.setdefault("cmd", cmd)
        raise RuntimeError("stop before actually compiling")

    import subprocess

    from elenchus.corpus.build import _compile_level
    original = subprocess.run
    subprocess.run = fake_run
    try:
        _compile_level([tmp_path / "a.c"], tmp_path / "o.dll", tmp_path / "s.dll",
                       "O0", main, tmp_path, [(dep, tmp_path / "dep")])
    except RuntimeError:
        pass
    finally:
        subprocess.run = original

    cmd = recorded["cmd"]
    assert f"-I{tmp_path / 'include'}" in cmd
    assert f"-I{tmp_path / 'dep' / 'inc'}" in cmd
    assert cmd.index("-DMAIN=1") < cmd.index("-DDEP=1")
    assert cmd[-1] == "-lws2_32"
    assert cmd.index("-lws2_32") > cmd.index(str(tmp_path / "a.c"))


# ------------------------------------------------- copy and create make paths


def test_create_builds_the_directories_its_path_needs(tmp_path):
    """lwIP ships no configuration of its own and expects arch/cc.h from
    whoever builds it - a directory its tarball does not have."""
    from elenchus.corpus.build import _prepare

    root = tmp_path / "src"
    root.mkdir()
    _prepare(root, package(create=[["arch/cc.h", "#define LWIP_NO_UNISTD_H 1\n"]]))

    assert (root / "arch" / "cc.h").read_text() == "#define LWIP_NO_UNISTD_H 1\n"


def test_copy_builds_the_directories_its_destination_needs(tmp_path):
    from elenchus.corpus.build import _prepare

    root = tmp_path / "src"
    (root / "win").mkdir(parents=True)
    (root / "win" / "config.h").write_text("#define HAVE_X 1\n")
    _prepare(root, package(copy=[["win/config.h", "build/gen/config.h"]]))

    assert (root / "build" / "gen" / "config.h").read_text() == "#define HAVE_X 1\n"
