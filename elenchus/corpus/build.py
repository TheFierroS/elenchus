"""Compile corpus packages into twinned binaries for later scanning.

For each package the driver downloads the source, applies any preparation the
package needs (a missing config header, a file to exclude), then for every
optimisation level compiles all its C sources into one debug shared library
and strips a copy. The result is a debug/stripped twin per level: the debug
half carries the DWARF ground truth, the stripped half is what the agent sees.

Compiling is kept separate from scanning. Compiling needs a cross-compiler and
the network; scanning needs Ghidra. Splitting them means the build driver can
be exercised without Ghidra, and a failed download does not waste a Ghidra
session.

A package that will not build is reported, not fatal - build systems disagree,
and chasing every one to green would stall the whole corpus. The caller decides
what to do with the failures.

Preparation covers the common reasons a plain "compile every .c" fails:

  exclude  drop sources that carry their own main, or a one-file amalgamation
           that re-includes the rest (both cause duplicate symbols)
  copy     put a package's prebuilt config header in place of the one its
           build system would normally generate
  create   write a small header the build system generates but we can supply
           verbatim, e.g. an empty export macro
"""

import shutil
import subprocess
import tarfile
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

OPT_LEVELS = ("O0", "O1", "O2", "O3")

# Where a package's dependencies are extracted, beside its own src/. The name
# is deliberately not one a source tree would use: the dataset drops every
# source file under it, and that rule matches on the directory name alone.
DEPENDENCY_DIR = "_deps"

# Flags that make a build reproducible and self-contained: emit DWARF, keep
# frame pointers so structure is legible, and do not pull in stack protectors
# or PIE machinery that would clutter the code we want to study.
_COMMON_FLAGS = (
    "-g",
    "-fno-omit-frame-pointer",
    "-fno-stack-protector",
    "-fno-PIE",
)

_CC = "x86_64-w64-mingw32-gcc"
_STRIP = "x86_64-w64-mingw32-strip"


@dataclass
class Package:
    """One library to compile, and what it takes to compile it.

    sources is a list of glob patterns relative to the extracted source root.
    It replaced a two-valued style field that could only express "the .c files
    in the root" or "the .c files in src/", which is not where most libraries
    keep them - lz4, libdeflate and zstd all use lib/, and that single
    limitation was the reason three of four candidate packages failed. A glob
    list costs nothing and expresses all of them.

    include_dirs, defines and libs exist for the same reason: a library whose
    headers sit in include/, that expects one macro to be set, or that calls
    a Windows API and must be linked against it, is not a hard package to
    build - but without these fields it is impossible to describe. Three of
    the first run's seven failures were a missing -l.

    exclude drops sources by file name, and a name may be a glob: nng keeps
    52 tests beside the code it tests, all of them *_test.c, and listing
    them one by one says nothing a reader could check.

    depends names libraries the package cannot compile without and that its
    own tarball does not carry: zydis needs zycore, libpng needs zlib. Each
    is a Package in its own right, written as a nested table in the manifest,
    and is compiled into the same binary. Its sources are extracted under
    DEPENDENCY_DIR rather than beside the package's own, and the dataset
    drops every function declared there: a dependency is compiled to make the
    package link, not to be learned from. Left in, a vendored zlib would
    either collide with the zlib package - one source file in two packages,
    which drops it from both - or, built with different macros, survive as a
    near-copy of it in another split.
    """

    name: str
    version: str
    url: str
    license: str
    domain: str = ""
    style: str = "c_glob"
    sources: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    libs: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    copy: list[list[str]] = field(default_factory=list)
    create: list[list[str]] = field(default_factory=list)
    depends: list["Package"] = field(default_factory=list)


@dataclass
class BuildResult:
    """What came of trying to build one package."""

    package: str
    ok: bool
    binaries: list[Path] = field(default_factory=list)
    error: str | None = None


def load_manifest(path):
    """Read the package manifest into a list of Package.

    A package's depends entries are nested tables with the same shape, so a
    dependency is described exactly as a package is and prepared by the same
    code.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [_package(entry) for entry in data["package"]]


def _package(entry):
    entry = dict(entry)
    entry["depends"] = [_package(d) for d in entry.get("depends", [])]
    return Package(**entry)


def _download(url, dest_dir):
    """Fetch a tarball and extract it, returning the extracted source root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "src.tar.gz"

    urllib.request.urlretrieve(url, archive)

    with tarfile.open(archive) as tar:
        members = tar.getnames()
        tar.extractall(dest_dir, filter="data")

    top = dest_dir / Path(members[0]).parts[0]
    return top if top.is_dir() else dest_dir


def _prepare(root, package):
    """Apply copy and create steps so a plain compile can succeed."""
    for src_rel, dst_rel in package.copy:
        shutil.copy2(root / src_rel, root / dst_rel)

    for name, content in package.create:
        (root / name).write_text(content)


def _c_sources(root, package):
    """Return the .c files to compile, honouring the package's exclude list.

    Raises when a package matches nothing. That failure used to surface as a
    compiler error about missing input, one step removed from the cause; said
    plainly it points straight at the pattern that is wrong.

    An exclude entry is matched against the file name as a glob, case
    sensitively, so a name with no wildcard in it still means that one file.
    """
    patterns = package.sources or (
        ["src/*.c"] if package.style == "c_glob_src" else ["*.c"]
    )

    excluded = package.exclude
    found = {
        path
        for pattern in patterns
        for path in root.glob(pattern)
        if not any(fnmatchcase(path.name, e) for e in excluded)
    }

    if not found:
        raise ValueError(
            f"no sources matched {patterns} under {root} "
            f"(excluding {sorted(excluded)})"
        )

    return sorted(found)


def _compile_level(sources, out_debug, out_stripped, opt, package=None, root=None,
                   extra=()):
    """Compile sources into a debug shared lib at opt, then strip a copy.

    --exclude-all-symbols is what makes the stripped twin actually stripped.
    MinGW exports every non-static symbol from a shared library by default,
    and an export table is not a symbol table: strip cannot remove it,
    because the loader needs it. Without the flag roughly half the functions
    in a "stripped" binary still carry their real names, which would hand the
    Ghidra-alone baseline half the answer and make the ablation table lie.

    Measured before adopting: zlib -O0 went from 136 named functions to 35,
    all of them import thunks and PE header entry points that a real target
    would have too, and Ghidra still found all 269 functions. That was once
    credited to the .pdata unwind table; it was wrong. zlib's functions are
    reached by direct calls, which is how Ghidra finds functions - it has no
    analyzer that reads .pdata. Functions reached only through pointers were
    being missed (a third of libtomcrypt -O3) until extraction began seeding
    functions from .pdata itself; see seed_functions_from_pdata.

    Returns (debug_path, stripped_path). Raises on compiler failure so the
    caller can mark the whole package failed.
    """
    includes = []
    defines = []
    libs = []
    entries = [(package, root)] if package is not None and root is not None else []
    for entry, entry_root in [*entries, *extra]:
        includes += [f"-I{Path(entry_root) / d}" for d in entry.include_dirs]
        defines += [f"-D{d}" for d in entry.defines]
        # Libraries go after the sources: the GNU linker resolves symbols in
        # command-line order and will not look back at an archive it has
        # already passed.
        libs += [f"-l{name}" for name in entry.libs]

    cmd = [
        _CC,
        f"-{opt}",
        *_COMMON_FLAGS,
        *includes,
        *defines,
        "-shared",
        "-Wl,--exclude-all-symbols",
        "-o",
        str(out_debug),
        *[str(s) for s in sources],
        *libs,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    shutil.copy2(out_debug, out_stripped)
    subprocess.run(
        [_STRIP, "-s", str(out_stripped)],
        check=True, capture_output=True, text=True,
    )

    return out_debug, out_stripped


def binary_paths(package, work_dir):
    """Return the paths build_package writes, as [(opt, debug, stripped), ...].

    The single place these names are decided. Resuming a partly scanned
    package reads the binaries a previous run compiled instead of compiling
    again, and it can only find them if both sides agree on where they are.
    """
    out_dir = Path(work_dir) / package.name / "out"
    return [
        (
            opt,
            out_dir / f"{package.name}_{opt}.dll",
            out_dir / f"{package.name}_{opt}_stripped.dll",
        )
        for opt in OPT_LEVELS
    ]


def build_package(package, work_dir):
    """Download, prepare, and compile one package at every optimisation level.

    Returns a BuildResult; on any failure ok is False and the error is captured
    rather than raised, so one bad package does not stop the corpus.
    """
    pkg_dir = Path(work_dir) / package.name
    out_dir = pkg_dir / "out"

    try:
        src_root = _download(package.url, pkg_dir / "src")
        _prepare(src_root, package)

        sources = _c_sources(src_root, package)
        if not sources:
            return BuildResult(package.name, ok=False,
                               error="no .c sources found")

        extra = []
        for dependency in package.depends:
            dep_root = _download(dependency.url,
                                 pkg_dir / DEPENDENCY_DIR / dependency.name)
            _prepare(dep_root, dependency)
            sources = sources + _c_sources(dep_root, dependency)
            extra.append((dependency, dep_root))

        out_dir.mkdir(parents=True, exist_ok=True)
        produced = []
        for opt, debug, stripped in binary_paths(package, work_dir):
            _compile_level(sources, debug, stripped, opt, package, src_root, extra)
            produced.extend([debug, stripped])

        return BuildResult(package.name, ok=True, binaries=produced)

    except (subprocess.CalledProcessError, OSError, tarfile.TarError,
            urllib.error.URLError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr
        else:
            detail = str(exc)
        return BuildResult(package.name, ok=False, error=str(detail)[:500])
