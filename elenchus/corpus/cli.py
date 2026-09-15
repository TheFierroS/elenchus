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

from elenchus.corpus.build import (
    OPT_LEVELS,
    BuildResult,
    binary_paths,
    build_package,
    load_manifest,
)
from elenchus.corpus.gaps import known_gaps
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
    file_sha256,
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


def _levels_in_corpus(conn):
    """Return {package: {opt levels stored with both twins}}.

    A level counts only when its debug and its stripped binary are both
    registered. Counting rows instead - the rule this replaced - let a package
    with one level registered twice and another missing add up to "complete",
    and the missing level was then skipped for good.
    """
    levels = {}
    for row in conn.execute(
        "SELECT package, opt_level FROM corpus_binaries "
        "GROUP BY package, opt_level HAVING COUNT(DISTINCT stripped) = 2"
    ):
        levels.setdefault(row["package"], set()).add(row["opt_level"])
    return levels


def _packages_in_corpus(conn):
    """Return packages stored at every optimisation level.

    A corpus run can take hours and can be interrupted by something that has
    nothing to do with it - the first one died to a WSL crash partway
    through scanning, with fourteen of twenty-two packages stored. Recompiling
    those fourteen to get to the fifteenth is wasted work, so a rerun picks up
    where the last one stopped. --rebuild forces the whole thing.
    """
    return {
        package
        for package, levels in _levels_in_corpus(conn).items()
        if levels >= set(OPT_LEVELS)
    }


class ResumeRefused(Exception):
    """A partly stored package cannot be finished from what is on disk."""


def _resume_build(conn, pkg, work_dir):
    """Reuse a previous run's binaries to finish a partly stored package.

    Recompiling is not an option once some levels are stored. A PE carries a
    link timestamp, so the same source compiled twice gives a different
    sha256, and the levels already stored would be registered a second time -
    the duplicate registrations prune-corpus exists to clean up.

    Reuse is only safe if the files on disk are the ones that were stored, so
    that is proven rather than assumed: every registered binary of the package
    must still exist at its expected path with the recorded hash. Anything
    else is refused with the reason, and the package is left alone.
    """
    expected = {}
    for opt, debug, stripped in binary_paths(pkg, work_dir):
        expected[(opt, 0)] = debug
        expected[(opt, 1)] = stripped

    missing = sorted(str(p) for p in expected.values() if not p.exists())
    if missing:
        raise ResumeRefused(
            f"{pkg.name} is partly stored but {len(missing)} compiled "
            f"binaries are gone (first: {missing[0]}); recompiling would "
            "register the stored levels twice. Use --rebuild, then "
            "`elenchus prune-corpus --apply`."
        )

    registered = conn.execute(
        "SELECT cb.opt_level, cb.stripped, b.sha256 FROM corpus_binaries cb "
        "JOIN binaries b ON b.id = cb.binary_id WHERE cb.package = ?",
        (pkg.name,),
    ).fetchall()

    for row in registered:
        path = expected.get((row["opt_level"], row["stripped"]))
        if path is None or file_sha256(path) != row["sha256"]:
            side = "stripped" if row["stripped"] else "debug"
            raise ResumeRefused(
                f"{pkg.name} -{row['opt_level']} {side}: the binary on disk is "
                "not the one stored, so the remaining levels would come from "
                "a different compilation. Use --rebuild, then "
                "`elenchus prune-corpus --apply`."
            )

    binaries = []
    for _opt, debug, stripped in binary_paths(pkg, work_dir):
        binaries.extend([debug, stripped])
    return BuildResult(pkg.name, ok=True, binaries=binaries)


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

    only = getattr(args, "only", None)
    if only:
        unknown = sorted(set(only) - {pkg.name for pkg in packages})
        if unknown:
            print(f"not in the manifest: {', '.join(unknown)}")
            return 2
        packages = [pkg for pkg in packages if pkg.name in only]

    pyghidra.start()

    stored = _levels_in_corpus(conn)
    # A recorded gap is a decision not to retry: treated as done, unless the
    # whole package is being rebuilt. Remove the record to retry one level.
    gaps = known_gaps(conn)

    built = []
    failed = []
    skipped = []
    for pkg in packages:
        if args.rebuild:
            done = set()
        else:
            done = stored.get(pkg.name, set()) | set(gaps.get(pkg.name, {}))

        if done >= set(OPT_LEVELS):
            skipped.append(pkg.name)
            continue

        if done:
            remaining = [opt for opt in OPT_LEVELS if opt not in done]
            print(f"resuming {pkg.name} {pkg.version}: "
                  f"-{', -'.join(remaining)} (reusing compiled binaries)",
                  flush=True)
            try:
                result = _resume_build(conn, pkg, args.work_dir)
            except ResumeRefused as exc:
                failed.append((pkg.name, str(exc)))
                print(f"  REFUSED: {exc}")
                continue
        else:
            print(f"building {pkg.name} {pkg.version} ...", flush=True)
            result = build_package(pkg, args.work_dir)
            if not result.ok:
                failed.append((pkg.name, result.error))
                print(f"  FAILED: "
                      f"{result.error.splitlines()[0] if result.error else '?'}")
                continue

        built.append((pkg, result, done))

    print()
    total_matched = 0
    total_gt = 0

    broken = []

    for pkg, result, done in built:
        # binaries are [debug_O0, strip_O0, debug_O1, strip_O1, ...]
        pairs = list(zip(result.binaries[0::2], result.binaries[1::2]))
        for opt, (debug_path, strip_path) in zip(OPT_LEVELS, pairs):
            if opt in done:
                continue
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
    skipped_gaps = [
        f"{name} -{level}"
        for name, levels in sorted(gaps.items())
        for level in levels
        if not args.rebuild and name in {pkg.name for pkg in packages}
    ]
    if skipped_gaps:
        print(f"known gaps      : {', '.join(skipped_gaps)} (not retried)")
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
