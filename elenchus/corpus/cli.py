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


def _scan(conn, path, run_kind_params):
    """Scan one binary into the database, returning its binary id.

    Only the extraction needed downstream is run here: functions (for ground
    truth matching), plus imports, calls and blocks for the encoder. Strings
    are skipped for corpus binaries - library strings are ASCII and not the
    signal we train on.
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
    except Exception:
        finish_run(conn, run_id, "failed")
        raise
    finish_run(conn, run_id, "ok")
    return binary_id


def cmd_build_corpus(args):
    """Build, scan, and store ground truth for every package in the manifest."""
    started = time.time()
    packages = load_manifest(args.manifest)
    conn = connect(args.db)

    pyghidra.start()

    built = []
    failed = []
    for pkg in packages:
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

    for pkg, result in built:
        # binaries are [debug_O0, strip_O0, debug_O1, strip_O1, ...]
        pairs = list(zip(result.binaries[0::2], result.binaries[1::2]))
        for opt, (debug_path, strip_path) in zip(OPT_LEVELS, pairs):
            params = {"package": pkg.name, "opt": opt}

            strip_id = _scan(conn, strip_path, {**params, "stripped": True})
            debug_id = _scan(conn, debug_path, {**params, "stripped": False})

            register_corpus_binary(
                conn, strip_id, pkg.name, "gcc", opt,
                stripped=True, version=pkg.version,
            )
            register_corpus_binary(
                conn, debug_id, pkg.name, "gcc", opt,
                stripped=False, version=pkg.version,
            )
            link_twins(conn, debug_id, strip_id)

            gt, matched = store_ground_truth(conn, strip_id, debug_path)
            total_gt += gt
            total_matched += matched
            pct = 100 * matched // gt if gt else 0
            print(f"  {pkg.name:10} -{opt}  gt={gt:4}  matched={matched:4} ({pct}%)")

    print()
    print(f"packages built  : {len(built)}/{len(packages)}")
    print(f"ground truth    : {total_matched}/{total_gt} matched "
          f"({100 * total_matched // total_gt if total_gt else 0}%)")
    if failed:
        print(f"failed packages : {', '.join(name for name, _ in failed)}")
    print(f"elapsed         : {time.time() - started:.1f}s")
    return 0
