"""The build-corpus command: compile packages, scan them, store ground truth.

This ties together the pieces proven separately. For each package it builds
the twinned binaries, then for every optimisation level scans the stripped
twin with Ghidra and attaches the ground truth read from the debug twin. The
debug and stripped binaries are both registered as corpus members and linked.

Ghidra starts once and every binary is scanned in the same session: starting
the JVM is a fixed cost paid per process, not per binary.

A package that will not build is reported and skipped, not fatal - the corpus
grows by what compiles, and the failures are printed so they can be fixed or
dropped later.
"""

import time

import pyghidra

from elenchus.corpus.build import OPT_LEVELS, build_package, load_manifest
from elenchus.corpus.store import (
    link_twins,
    register_corpus_binary,
    store_ground_truth,
)
from elenchus.db import (
    connect,
    finish_run,
    set_run_tool_version,
    start_run,
)
from elenchus.extract.ghidra import (
    extract_blocks,
    extract_calls,
    extract_functions,
    extract_imports,
    ghidra_version,
    register_binary,
)
from elenchus.extract.instructions import extract_code


def _scan(conn, path, run_kind_params, with_code=True):
    """Scan one binary into the database, returning its binary id.

    Only the extraction needed downstream is run here: functions (for ground
    truth matching), plus imports, calls, blocks and the instruction listing
    the encoder trains on. Strings are skipped for corpus binaries - library
    strings are ASCII and not the signal we train on.
    """
    run_id = start_run(conn, "extract", tool="ghidra", params=run_kind_params)
    try:
        with pyghidra.open_program(str(path)) as api:
            program = api.getCurrentProgram()
            set_run_tool_version(conn, run_id, ghidra_version(program))
            binary_id = register_binary(conn, str(path), str(program.getLanguageID()))
            functions = extract_functions(conn, binary_id, run_id, program)
            extract_imports(conn, binary_id, run_id, program)
            extract_calls(conn, binary_id, run_id, program, functions)
            extract_blocks(conn, binary_id, run_id, program, functions)
            if with_code:
                extract_code(conn, binary_id, run_id, program, functions)
    except Exception:
        finish_run(conn, run_id, "failed")
        raise
    finish_run(conn, run_id, "ok")
    return binary_id


def _packages_in_corpus(conn):
    """Return packages already fully recorded: both twins at every level.

    A corpus run can take hours and can be interrupted by something that has
    nothing to do with it - the first one died to a WSL crash partway
    through scanning, with fourteen of twenty-two packages stored. Recompiling
    those fourteen to get to the fifteenth is wasted work, so a rerun picks up
    where the last one stopped. --rebuild forces the whole thing.

    Fully recorded means the expected number of rows, not merely some: a
    package interrupted mid-scan must be redone, not skipped.
    """
    expected = 2 * len(OPT_LEVELS)
    return {
        row["package"]
        for row in conn.execute(
            "SELECT package, COUNT(*) AS n FROM corpus_binaries "
            "GROUP BY package HAVING n >= ?",
            (expected,),
        )
    }


def active_runs(conn, kind="extract", within_minutes=60):
    """Return runs of this kind opened recently and never closed.

    Two corpus builds writing to one database is not a hypothetical: it
    happened, three at once, because the command is long, silent and easy to
    start twice. SQLite takes one writer, Ghidra locks its project
    directories, and the result is two processes quietly spoiling each
    other's work.

    The time window keeps this from tripping over the wreckage of an old
    crash. A run opened yesterday and never closed is a dead process, not a
    competitor - `check --close-stale` is for those.
    """
    return conn.execute(
        "SELECT id, kind, started_at FROM runs "
        "WHERE status = 'running' AND kind = ? "
        "AND started_at > datetime('now', ?) ORDER BY id",
        (kind, f"-{int(within_minutes)} minutes"),
    ).fetchall()


def store_level(conn, pkg, opt, debug_path, strip_path):
    """Scan one optimisation level's twins and store their ground truth.

    Kept as a function so the loop around it can survive its failure. Returns
    (ground truth rows, rows matched to a scanned function).
    """
    params = {"package": pkg.name, "opt": opt}

    strip_id = _scan(conn, strip_path, {**params, "stripped": True})
    # The debug twin exists to be read for DWARF, not to be learnt from:
    # strip removes symbols, not code, so its instructions would be the
    # stripped twin's instructions stored a second time.
    debug_id = _scan(
        conn, debug_path, {**params, "stripped": False}, with_code=False
    )

    register_corpus_binary(
        conn, strip_id, pkg.name, "gcc", opt,
        stripped=True, version=pkg.version,
    )
    register_corpus_binary(
        conn, debug_id, pkg.name, "gcc", opt,
        stripped=False, version=pkg.version,
    )
    link_twins(conn, debug_id, strip_id)

    return store_ground_truth(conn, strip_id, debug_path)


def cmd_build_corpus(args):
    """Build, scan, and store ground truth for every package in the manifest."""
    started = time.time()
    packages = load_manifest(args.manifest)
    conn = connect(args.db)

    active = active_runs(conn)
    if active and not getattr(args, "force", False):
        print(f"another run appears to be working on {args.db}:")
        for row in active:
            print(f"  run {row['id']} ({row['kind']}) started {row['started_at']}")
        print()
        print("Wait for it, or pass --force if you are certain it is dead.")
        print("If it died, `elenchus check --close-stale` will tidy up.")
        return 1

    pyghidra.start()

    already = _packages_in_corpus(conn)

    built = []
    failed = []
    skipped = []
    for pkg in packages:
        if pkg.name in already and not args.rebuild:
            skipped.append(pkg.name)
            continue

        print(f"building {pkg.name} {pkg.version} ...", flush=True)
        result = build_package(pkg, args.work_dir)
        if not result.ok:
            failed.append((pkg.name, result.error))
            print(f"  FAILED: {result.error.splitlines()[0] if result.error else '?'}")
            continue
        built.append((pkg, result))

    print()
    total_matched = 0
    total_gt = 0

    broken = []

    for pkg, result in built:
        # binaries are [debug_O0, strip_O0, debug_O1, strip_O1, ...]
        pairs = list(zip(result.binaries[0::2], result.binaries[1::2]))
        for opt, (debug_path, strip_path) in zip(OPT_LEVELS, pairs):
            try:
                gt, matched = store_level(
                    conn, pkg, opt, debug_path, strip_path
                )
            except Exception as exc:
                # One binary out of two hundred must not end a run that has
                # been going for hours. A malformed compilation unit did
                # exactly that once, after fourteen packages had been stored
                # and the fifteenth was being read.
                broken.append((pkg.name, opt, f"{type(exc).__name__}: {exc}"))
                print(f"  {pkg.name:10} -{opt}  SCAN FAILED", flush=True)
                continue

            total_gt += gt
            total_matched += matched
            pct = 100 * matched // gt if gt else 0
            print(
                f"  {pkg.name:10} -{opt}  gt={gt:4}  "
                f"matched={matched:4} ({pct}%)",
                flush=True,
            )

    print()
    if skipped:
        print(f"already present : {len(skipped)} ({', '.join(sorted(skipped))})")
    print(f"packages built  : {len(built)}/{len(packages) - len(skipped)}")
    print(f"ground truth    : {total_matched}/{total_gt} matched "
          f"({100 * total_matched // total_gt if total_gt else 0}%)")
    if failed:
        print(f"failed packages : {', '.join(name for name, _ in failed)}")
    if broken:
        print(f"failed scans    : {len(broken)}")
        for name, opt, error in broken:
            print(f"  {name} -{opt}: {error[:110]}")
    print(f"elapsed         : {time.time() - started:.1f}s")
    return 0
