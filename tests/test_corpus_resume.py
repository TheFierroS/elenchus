"""Tests for finishing a partly stored package without breaking the corpus.

A scan can die part way through a package - Ghidra ran out of heap on one
optimisation level of quickjs while the others went through. What happens
next decides whether the corpus stays clean. Recompiling gives every binary a
new link timestamp and so a new sha256, and the levels already stored get
registered a second time. So a partly stored package is finished from the
binaries already on disk, and only after proving they are the ones stored.

Ghidra is never started here: the scan is replaced by a recorder, so what is
under test is only the decision of what to compile and what to scan.
"""

import sys
import types

import pytest

sys.modules.setdefault("pyghidra", types.ModuleType("pyghidra"))

from elenchus.corpus import build as build_module  # noqa: E402
from elenchus.corpus import cli as module  # noqa: E402
from elenchus.corpus.build import (  # noqa: E402
    OPT_LEVELS,
    BuildResult,
    Package,
    binary_paths,
)
from elenchus.corpus.store import register_corpus_binary  # noqa: E402
from elenchus.extract.ghidra import file_sha256  # noqa: E402


def package(name):
    return Package(name=name, version="1", url="https://example.invalid/x.tar.gz",
                   license="MIT")


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def compile_to_disk(pkg, work_dir, stamp="first"):
    """Write every binary a build would produce; stamp stands in for the PE
    timestamp that makes two compilations of the same source differ."""
    for opt, debug, stripped in binary_paths(pkg, work_dir):
        write(debug, f"{pkg.name}-{opt}-debug-{stamp}".encode())
        write(stripped, f"{pkg.name}-{opt}-stripped-{stamp}".encode())


def store(db, pkg, work_dir, levels):
    """Register levels as a previous run would have, from the files on disk."""
    for opt, debug, stripped in binary_paths(pkg, work_dir):
        if opt not in levels:
            continue
        for path, is_stripped in ((debug, False), (stripped, True)):
            register_file(db, pkg.name, opt, is_stripped, path)


def register_file(db, name, opt, is_stripped, path):
    with db:
        binary_id = db.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (str(path), file_sha256(path)),
        ).lastrowid
    register_corpus_binary(db, binary_id, name, "gcc", opt, is_stripped,
                           version="1")


class Harness:
    """Runs build-corpus with compiling and scanning replaced by recorders."""

    def __init__(self, monkeypatch, db, work_dir):
        self.db = db
        self.work_dir = work_dir
        self.packages = []
        self.compiled = []
        self.scanned = []
        self.ghidra_started = False

        def fake_build(pkg, work_dir):
            self.compiled.append(pkg.name)
            compile_to_disk(pkg, work_dir, stamp="recompiled")
            binaries = []
            for _opt, debug, stripped in binary_paths(pkg, work_dir):
                binaries.extend([debug, stripped])
            return BuildResult(pkg.name, ok=True, binaries=binaries)

        def fake_store_level(conn, pkg, opt, debug_path, strip_path):
            self.scanned.append((pkg.name, opt, debug_path, strip_path))
            return 10, 10

        def fake_start():
            self.ghidra_started = True

        monkeypatch.setattr(module, "load_manifest", lambda _p: self.packages)
        monkeypatch.setattr(module, "build_package", fake_build)
        monkeypatch.setattr(module, "store_level", fake_store_level)
        monkeypatch.setattr(module, "connect", lambda _p: db)
        monkeypatch.setattr(module.pyghidra, "start", fake_start, raising=False)

    def run(self, rebuild=False, only=None):
        args = types.SimpleNamespace(
            db=":memory:", manifest=None, work_dir=str(self.work_dir),
            rebuild=rebuild, only=only,
        )
        return module.cmd_build_corpus(args)

    def scanned_levels(self, name):
        return [opt for pkg, opt, _d, _s in self.scanned if pkg == name]


@pytest.fixture
def harness(monkeypatch, db, tmp_path):
    return Harness(monkeypatch, db, tmp_path / "corpus")


# ---------------------------------------------------------------- what counts as stored


def test_a_level_needs_both_twins(harness):
    pkg = package("probe")
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0"})
    _opt, _debug, stripped = binary_paths(pkg, harness.work_dir)[1]
    register_file(harness.db, "probe", "O1", True, stripped)

    assert module._levels_in_corpus(harness.db) == {"probe": {"O0"}}


def test_a_duplicated_level_cannot_hide_a_missing_one(harness):
    """The row-counting rule this replaced called this package complete.

    O0 registered twice plus O1 and O2 is eight rows - the number a complete
    package has - with O3 never stored.
    """
    pkg = package("probe")
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0", "O1", "O2"})

    stale = harness.work_dir / "stale"
    for is_stripped in (False, True):
        path = stale / f"O0-{is_stripped}.dll"
        write(path, f"older compilation {is_stripped}".encode())
        register_file(harness.db, "probe", "O0", is_stripped, path)

    rows = harness.db.execute(
        "SELECT COUNT(*) FROM corpus_binaries WHERE package = 'probe'"
    ).fetchone()[0]
    assert rows == 2 * len(OPT_LEVELS)
    assert "probe" not in module._packages_in_corpus(harness.db)


# ---------------------------------------------------------------- the normal cases


def test_a_complete_package_is_neither_compiled_nor_scanned(harness):
    pkg = package("done")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, set(OPT_LEVELS))

    assert harness.run() == 0
    assert harness.compiled == []
    assert harness.scanned == []


def test_a_new_package_is_compiled_and_every_level_scanned(harness):
    harness.packages = [package("fresh")]

    assert harness.run() == 0
    assert harness.compiled == ["fresh"]
    assert harness.scanned_levels("fresh") == list(OPT_LEVELS)


# ---------------------------------------------------------------- resuming


def test_a_partly_stored_package_scans_only_what_is_missing(harness):
    """The quickjs case: one level lost to a heap error, the rest stored."""
    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0", "O2", "O3"})

    assert harness.run() == 0
    assert harness.compiled == [], "recompiling would duplicate the stored levels"
    assert harness.scanned_levels("quickjs") == ["O1"]


def test_resuming_scans_the_binaries_already_on_disk(harness):
    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0"})

    harness.run()

    expected = {opt: (d, s) for opt, d, s in binary_paths(pkg, harness.work_dir)}
    for _name, opt, debug, stripped in harness.scanned:
        assert (debug, stripped) == expected[opt]


def test_a_binary_changed_on_disk_refuses_the_resume(harness, capsys):
    """Files from another compilation would mix two builds in one package."""
    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0"})
    compile_to_disk(pkg, harness.work_dir, stamp="someone recompiled")

    assert harness.run() == 0
    assert harness.compiled == []
    assert harness.scanned == []

    out = capsys.readouterr().out
    assert "REFUSED" in out and "not the one stored" in out
    assert "failed packages : quickjs" in out


def test_missing_binaries_refuse_the_resume(harness, capsys):
    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0"})
    _opt, debug, _stripped = binary_paths(pkg, harness.work_dir)[3]
    debug.unlink()

    harness.run()

    assert harness.compiled == []
    assert harness.scanned == []
    assert "are gone" in capsys.readouterr().out


def test_the_resume_refusal_names_the_way_out(harness, capsys):
    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0"})
    compile_to_disk(pkg, harness.work_dir, stamp="changed")

    harness.run()

    out = capsys.readouterr().out
    assert "--rebuild" in out and "prune-corpus --apply" in out


def test_a_refused_package_does_not_stop_the_others(harness):
    broken, fresh = package("broken"), package("fresh")
    harness.packages = [broken, fresh]
    compile_to_disk(broken, harness.work_dir)
    store(harness.db, broken, harness.work_dir, {"O0"})
    compile_to_disk(broken, harness.work_dir, stamp="changed")

    assert harness.run() == 0
    assert harness.compiled == ["fresh"]
    assert harness.scanned_levels("fresh") == list(OPT_LEVELS)
    assert harness.scanned_levels("broken") == []


def test_rebuild_compiles_and_scans_every_level_again(harness):
    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0", "O1", "O2"})

    harness.run(rebuild=True)

    assert harness.compiled == ["quickjs"]
    assert harness.scanned_levels("quickjs") == list(OPT_LEVELS)


# ---------------------------------------------------------------- --only


def test_only_limits_the_run_to_the_named_packages(harness):
    harness.packages = [package("a"), package("b"), package("c")]

    assert harness.run(only=["b"]) == 0
    assert harness.compiled == ["b"]
    assert {name for name, *_ in harness.scanned} == {"b"}


def test_only_refuses_a_name_that_is_not_in_the_manifest(harness, capsys):
    """A typo must not quietly turn into a run that does nothing."""
    harness.packages = [package("quickjs")]

    assert harness.run(only=["quikjs"]) == 2
    assert harness.compiled == []
    assert not harness.ghidra_started
    assert "quikjs" in capsys.readouterr().out


# ---------------------------------------------------------------- one naming rule


def test_build_package_writes_where_binary_paths_says(monkeypatch, tmp_path):
    """Resume finds binaries by name; the builder must agree on the names."""
    pkg = package("probe")
    written = []

    monkeypatch.setattr(build_module, "_download", lambda _u, dest: dest)
    monkeypatch.setattr(build_module, "_prepare", lambda _r, _p: None)
    monkeypatch.setattr(build_module, "_c_sources", lambda _r, _p: ["x.c"])
    monkeypatch.setattr(
        build_module, "_compile_level",
        lambda _s, debug, stripped, opt, *_a: written.append((opt, debug, stripped)),
    )

    result = build_module.build_package(pkg, tmp_path)

    assert result.ok
    assert written == binary_paths(pkg, tmp_path)


# ---------------------------------------------------------------- known gaps


def test_a_recorded_gap_is_not_retried(harness, capsys):
    """quickjs -O1 fails however much heap it gets; a build must not keep trying."""
    from elenchus.corpus.gaps import record_gap

    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0", "O2", "O3"})
    record_gap(harness.db, "quickjs", "O1", "Ghidra heap exhausted")

    assert harness.run() == 0
    assert harness.compiled == []
    assert harness.scanned == []
    out = capsys.readouterr().out
    assert "already present : 1 (quickjs)" in out
    assert "quickjs -O1 (not retried)" in out


def test_rebuild_ignores_recorded_gaps(harness):
    from elenchus.corpus.gaps import record_gap

    pkg = package("quickjs")
    harness.packages = [pkg]
    compile_to_disk(pkg, harness.work_dir)
    store(harness.db, pkg, harness.work_dir, {"O0", "O2", "O3"})
    record_gap(harness.db, "quickjs", "O1", "Ghidra heap exhausted")

    harness.run(rebuild=True)

    assert harness.scanned_levels("quickjs") == list(OPT_LEVELS)
