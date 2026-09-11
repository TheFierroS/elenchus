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
from pathlib import Path

OPT_LEVELS = ("O0", "O1", "O2", "O3")

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
    name: str
    version: str
    url: str
    license: str
    style: str
    exclude: list[str] = field(default_factory=list)
    copy: list[list[str]] = field(default_factory=list)
    create: list[list[str]] = field(default_factory=list)


@dataclass
class BuildResult:
    """What came of trying to build one package."""

    package: str
    ok: bool
    binaries: list[Path] = field(default_factory=list)
    error: str | None = None


def load_manifest(path):
    """Read the package manifest into a list of Package."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [Package(**entry) for entry in data["package"]]


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
    """Return the .c files to compile, honouring the package's exclude list."""
    search = root / "src" if package.style == "c_glob_src" else root
    excluded = set(package.exclude)
    return sorted(s for s in search.glob("*.c") if s.name not in excluded)


def _compile_level(sources, out_debug, out_stripped, opt):
    """Compile sources into a debug shared lib at opt, then strip a copy.

    Returns (debug_path, stripped_path). Raises on compiler failure so the
    caller can mark the whole package failed.
    """
    cmd = [
        _CC,
        f"-{opt}",
        *_COMMON_FLAGS,
        "-shared",
        "-o",
        str(out_debug),
        *[str(s) for s in sources],
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    shutil.copy2(out_debug, out_stripped)
    subprocess.run(
        [_STRIP, "-s", str(out_stripped)],
        check=True, capture_output=True, text=True,
    )

    return out_debug, out_stripped


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

        out_dir.mkdir(parents=True, exist_ok=True)
        produced = []
        for opt in OPT_LEVELS:
            debug = out_dir / f"{package.name}_{opt}.dll"
            stripped = out_dir / f"{package.name}_{opt}_stripped.dll"
            _compile_level(sources, debug, stripped, opt)
            produced.extend([debug, stripped])

        return BuildResult(package.name, ok=True, binaries=produced)

    except (subprocess.CalledProcessError, OSError, tarfile.TarError,
            urllib.error.URLError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr
        else:
            detail = str(exc)
        return BuildResult(package.name, ok=False, error=str(detail)[:500])
