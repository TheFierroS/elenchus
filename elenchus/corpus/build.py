"""Compile corpus packages into twinned binaries for later scanning.

For each package the driver downloads the source, then for every optimisation
level compiles all its C sources into one debug shared library and strips a
copy of it. The result is a debug/stripped twin per level: the debug half
carries the DWARF ground truth, the stripped half is what the agent will see.

Compiling is kept separate from scanning. Compiling needs a cross-compiler and
the network; scanning needs Ghidra. Splitting them means the build driver can
be exercised without Ghidra, and a failed download does not waste a Ghidra
session.

A package that will not build is reported, not fatal - build systems disagree,
and chasing every one to green would stall the whole corpus. The caller decides
what to do with the failures.
"""

import shutil
import subprocess
import tarfile
import tomllib
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


@dataclass
class BuildResult:
    """What came of trying to build one package."""

    package: str
    ok: bool
    binaries: list[Path] = field(default_factory=list)  # (debug, stripped) pairs flat
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
        # Tarballs from GitHub wrap everything in a single top-level directory.
        members = tar.getnames()
        tar.extractall(dest_dir, filter="data")

    top = dest_dir / Path(members[0]).parts[0]
    return top if top.is_dir() else dest_dir


def _c_sources(root, style):
    """Return the .c files to compile for a given build style."""
    search = root / "src" if style == "c_glob_src" else root
    return sorted(search.glob("*.c"))


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
    subprocess.run([_STRIP, "-s", str(out_stripped)], check=True,
                   capture_output=True, text=True)

    return out_debug, out_stripped


def build_package(package, work_dir):
    """Download and compile one package at every optimisation level.

    work_dir holds the extracted source and the output binaries. Returns a
    BuildResult; on any failure ok is False and the error is captured rather
    than raised, so one bad package does not stop the corpus.
    """
    pkg_dir = Path(work_dir) / package.name
    out_dir = pkg_dir / "out"

    try:
        src_root = _download(package.url, pkg_dir / "src")
        sources = _c_sources(src_root, package.style)
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
