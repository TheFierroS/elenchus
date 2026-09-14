"""Command line interface for Elenchus."""

import argparse
import sys
import time
from pathlib import Path

import pyghidra

from elenchus.check import CHECKS, WARNINGS, close_stale_runs
from elenchus.corpus.cli import cmd_build_corpus
from elenchus.corpus.dataset import (
    assign_splits,
    candidate_rows,
    eligible_rows,
    leakage,
    load_splits,
    package_sizes,
    report,
    store_splits,
)
from elenchus.corpus.prune import cmd_prune_corpus
from elenchus.corpus.refresh import cmd_refresh_truth
from elenchus.db import connect, finish_run, set_run_tool_version, start_run
from elenchus.evaluation.baselines import all_baselines
from elenchus.evaluation.metrics import evaluate, format_table
from elenchus.evaluation.store import (
    format_history,
    history,
    record,
    task_params,
)
from elenchus.evaluation.task import retrieval_task
from elenchus.extract.ghidra import (
    extract_blocks,
    extract_calls,
    extract_functions,
    extract_imports,
    extract_strings,
    ghidra_version,
    register_binary,
)
from elenchus.extract.instructions import extract_code, function_index
from elenchus.inspect import find_functions, render, sample_functions, summarise


def cmd_scan(args):
    """Analyse a binary with Ghidra and record what it finds."""
    started = time.time()
    pyghidra.start()
    conn = connect(args.db)

    run_id = start_run(
        conn,
        "extract",
        tool="ghidra",
        params={
            "binary": args.binary,
            "min_string_length": args.min_string_length,
            "keep_unreferenced_strings": args.keep_unreferenced_strings,
        },
    )

    try:
        with pyghidra.open_program(args.binary) as api:
            program = api.getCurrentProgram()
            arch = str(program.getLanguageID())
            set_run_tool_version(conn, run_id, ghidra_version(program))

            binary_id = register_binary(conn, args.binary, arch)
            functions = extract_functions(conn, binary_id, run_id, program)
            imports = extract_imports(conn, binary_id, run_id, program)
            calls = extract_calls(conn, binary_id, run_id, program, functions)
            blocks = extract_blocks(conn, binary_id, run_id, program, functions)
            coded = extract_code(conn, binary_id, run_id, program, functions)
            strings = extract_strings(
                conn,
                binary_id,
                run_id,
                program,
                functions,
                args.min_string_length,
                args.keep_unreferenced_strings,
            )
    except Exception:
        finish_run(conn, run_id, "failed")
        raise

    finish_run(conn, run_id, "ok")

    print(f"binary   : {args.binary}")
    print(f"arch     : {arch}")
    print(f"run      : {run_id}")
    print(f"id       : {binary_id}")
    print(f"functions: {len(functions)}")
    print(f"imports  : {len(imports)}")
    print(f"calls    : {calls}")
    print(f"blocks   : {blocks}")
    print(f"code     : {coded} functions")
    print(f"strings  : {strings}")
    print(f"elapsed  : {time.time() - started:.1f}s")
    return 0


def cmd_extract_code(args):
    """Add instruction listings to binaries already in the database.

    Week 4 introduced function_code, so every binary scanned before it has
    functions, blocks and calls but no code. Rescanning through build-corpus
    would recompile everything for nothing; this walks the binaries already
    recorded and extracts only what is missing. Ghidra reuses its cached
    analysis, so the second pass over a binary is cheap.
    """
    pyghidra.start()
    conn = connect(args.db)

    # The debug twin of a corpus binary is never an analysis target - it
    # exists to be read for DWARF - and its code is the stripped twin's code
    # stored twice, so it is left out whatever else is asked for.
    conditions = [
        "id NOT IN (SELECT binary_id FROM corpus_binaries WHERE stripped = 0)"
    ]
    if args.only_corpus:
        conditions.append("id IN (SELECT binary_id FROM corpus_binaries)")

    rows = conn.execute(
        "SELECT id, path FROM binaries WHERE "
        + " AND ".join(conditions)
        + " ORDER BY id"
    ).fetchall()

    done = 0
    skipped = []
    for row in rows:
        path = Path(row["path"])

        if args.skip_existing and conn.execute(
            "SELECT 1 FROM function_code WHERE binary_id = ? LIMIT 1",
            (row["id"],),
        ).fetchone():
            continue

        if not path.exists():
            skipped.append((row["id"], "file is gone"))
            continue

        run_id = start_run(
            conn, "extract", tool="ghidra", params={"binary": str(path),
                                                    "stage": "instructions"}
        )
        try:
            with pyghidra.open_program(str(path)) as api:
                program = api.getCurrentProgram()
                set_run_tool_version(conn, run_id, ghidra_version(program))
                count = extract_code(
                    conn, row["id"], run_id,
                    program, function_index(conn, row["id"]),
                )
        except Exception as exc:
            finish_run(conn, run_id, "failed")
            skipped.append((row["id"], str(exc)[:120]))
            print(f"  {path.name:34} FAILED")
            continue

        finish_run(conn, run_id, "ok")
        done += 1
        print(f"  {path.name:34} {count} functions", flush=True)

    print()
    print(f"binaries done   : {done}/{len(rows)}")
    for binary_id, reason in skipped:
        print(f"  skipped {binary_id}: {reason}")
    return 0


def cmd_inspect(args):
    """Show functions with their ground truth, instructions, and tokens.

    Without arguments this is a filter over the corpus; with --sample it is
    the hand-check the week's exit criterion asks for, drawn reproducibly so
    the same fifty functions can be pulled up again.
    """
    conn = connect(args.db)

    truth = None
    if args.with_truth:
        truth = True
    if args.without_truth:
        truth = False

    rows = find_functions(
        conn,
        function_id=args.function_id,
        package=args.package,
        opt=args.opt,
        name=args.name,
        truth=truth,
    )

    if not rows:
        print("no function matched")
        return 1

    if args.sample:
        rows = sample_functions(rows, args.sample, args.seed)

    if args.list:
        print(f"{len(rows)} functions")
        for row in rows:
            print(summarise(row))
        return 0

    for index, row in enumerate(rows):
        if index:
            print()
            print("=" * 72)
            print()
        limit = args.max_instructions or None
        for line in render(conn, row, limit=limit):
            print(line)

    return 0


def cmd_dataset(args):
    """Report the trainable dataset, and assign or verify its split.

    Running this without --assign is a read: it says what would be excluded
    and what the current split looks like. With --assign it writes the
    assignment. Either way it ends with the leakage check, because a split
    that has never been checked is a split that cannot be trusted.
    """
    conn = connect(args.db)

    # Normalising every function is the expensive part, so it happens once
    # here and the candidates are handed to each step that needs them.
    rows = candidate_rows(conn)
    assignment = load_splits(conn)

    if args.assign or not assignment:
        sizes = package_sizes(eligible_rows(conn, rows))
        if not sizes:
            print("no eligible functions - run refresh-truth and extract-code first")
            return 1

        assignment = assign_splits(sizes)
        if args.assign:
            store_splits(conn, assignment)
        else:
            print("(no split recorded yet - showing what would be assigned)")

    for line in report(conn, assignment, rows):
        print(line)

    print()
    violations = leakage(conn, assignment, rows)
    if not violations:
        print("leakage check    : ok, no function appears in two splits")
        return 0

    print(f"leakage check    : {len(violations)} FAILED")
    for key, where in violations[:15]:
        print(f"    {key}  {' = '.join(where)}")
    if len(violations) > 15:
        print(f"    ... and {len(violations) - 15} more")
    return 1


def cmd_baselines(args):
    """Measure the model-free baselines on cross-optimisation retrieval.

    This is the bar. Run before any encoder exists, so that whatever the
    encoder scores later can be read as better or worse than something,
    rather than as a number on its own.
    """
    conn = connect(args.db)

    queries, pool, gold = retrieval_task(
        conn,
        split=args.split,
        query_opt=args.query_opt,
        pool_opt=args.pool_opt,
    )

    if not queries:
        print(f"no queries for split={args.split} "
              f"{args.query_opt} -> {args.pool_opt}; run dataset --assign first")
        return 1

    params = task_params(
        conn, args.split, args.query_opt, args.pool_opt, args.seed
    )

    print(f"task    : {args.split} split, -{args.query_opt} query "
          f"-> -{args.pool_opt} pool")
    print(f"queries : {len(queries)}   pool: {len(pool)}")
    print(f"corpus  : {params['dataset']}")
    print()

    run_id = start_run(conn, "eval", tool="baselines", params=params,
                       seed=args.seed)

    try:
        results = {}
        for scorer in all_baselines(args.seed):
            started = time.time()
            results[scorer.name] = evaluate(scorer, queries, pool, gold)
            print(f"  {scorer.name:<24} {time.time() - started:6.1f}s",
                  flush=True)
    except Exception:
        finish_run(conn, run_id, "failed")
        raise

    if not args.no_record:
        record(conn, run_id, params, results)
    finish_run(conn, run_id, "ok")

    print()
    for line in format_table(results):
        print(line)

    return 0


def cmd_measurements(args):
    """Show what has been measured before, newest first."""
    conn = connect(args.db)
    for line in format_history(
        history(conn, split=args.split, method=args.method, limit=args.limit)
    ):
        print(line)
    return 0


def cmd_check(args):
    """Run every consistency check and report.

    Exits non-zero on a defect, not on a warning: an interrupted run is
    worth seeing but does not mean the database is broken, and a command
    that goes red for something harmless gets ignored when it goes red for
    something real.
    """
    conn = connect(args.db)

    if args.close_stale:
        closed = close_stale_runs(conn, args.stale_minutes)
        print(f"closed {closed} run(s) open for over "
              f"{args.stale_minutes} minutes")
        print()

    defects = 0
    warnings = 0

    for label, check in CHECKS:
        violations = check(conn)
        count = len(violations)

        if count == 0:
            status = "ok"
        elif label in WARNINGS:
            status = f"{count} warning"
            warnings += count
        else:
            status = f"{count} FAILED"
            defects += count

        print(f"{label:22} {status}")

        for violation in violations[:10]:
            print(f"    {violation}")
        if count > 10:
            print(f"    ... and {count - 10} more")

    if warnings and not defects:
        print()
        print("warnings only: nothing is corrupt, but some run was cut short")

    return 1 if defects else 0


def build_parser():
    parser = argparse.ArgumentParser(prog="elenchus")
    parser.add_argument(
        "--db",
        default="data/elenchus.db",
        help="path to the database file (default: data/elenchus.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="analyse a binary and record observations")
    scan.add_argument("binary", help="path to the binary to analyse")
    scan.add_argument(
        "--min-string-length",
        type=int,
        default=4,
        help="ignore strings shorter than this (default: 4)",
    )
    scan.add_argument(
        "--keep-unreferenced-strings",
        action="store_true",
        help="also record strings that no function references",
    )
    scan.set_defaults(func=cmd_scan)

    check = sub.add_parser("check", help="run consistency checks on the database")
    check.add_argument(
        "--close-stale",
        action="store_true",
        help="mark long-open runs as failed before checking",
    )
    check.add_argument(
        "--stale-minutes",
        type=int,
        default=60,
        help="how old an open run must be to count as abandoned (default: 60)",
    )
    check.set_defaults(func=cmd_check)

    code = sub.add_parser(
        "extract-code",
        help="add instruction listings to binaries already in the database",
    )
    code.add_argument(
        "--only-corpus",
        action="store_true",
        help="restrict to corpus binaries, leaving one-off scans alone",
    )
    code.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip binaries that already have listings",
    )
    code.set_defaults(func=cmd_extract_code)

    inspect = sub.add_parser(
        "inspect",
        help="show a function's ground truth, instructions, and tokens",
    )
    inspect.add_argument(
        "--function-id", type=int, help="inspect one function by its id"
    )
    inspect.add_argument("--package", help="restrict to one corpus package")
    inspect.add_argument("--opt", help="restrict to one optimisation level, e.g. O0")
    inspect.add_argument("--name", help="restrict to one ground-truth name")
    inspect.add_argument(
        "--with-truth",
        action="store_true",
        help="only functions the compiler named",
    )
    inspect.add_argument(
        "--without-truth",
        action="store_true",
        help="only functions with no ground truth, i.e. runtime glue",
    )
    inspect.add_argument(
        "--sample", type=int, help="draw this many at random instead of all"
    )
    inspect.add_argument(
        "--seed", type=int, default=1, help="seed for --sample (default: 1)"
    )
    inspect.add_argument(
        "--list",
        action="store_true",
        help="one line per function instead of the full listing",
    )
    inspect.add_argument(
        "--max-instructions",
        type=int,
        default=40,
        help="truncate long listings (default: 40, 0 for all)",
    )
    inspect.set_defaults(func=cmd_inspect)

    corpus = sub.add_parser(
        "build-corpus", help="compile, scan, and store ground truth for packages"
    )
    corpus.add_argument(
        "--force",
        action="store_true",
        help="start even if another run looks active on this database",
    )
    corpus.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild packages already in the corpus instead of skipping them",
    )
    corpus.add_argument(
        "--manifest",
        default="elenchus/corpus/manifest.toml",
        help="path to the package manifest",
    )
    corpus.add_argument(
        "--work-dir",
        default="data/corpus",
        help="where to download sources and write built binaries",
    )
    corpus.set_defaults(func=cmd_build_corpus)

    refresh = sub.add_parser(
        "refresh-truth",
        help="re-read ground truth from the debug twins, no Ghidra needed",
    )
    refresh.set_defaults(func=cmd_refresh_truth)

    prune = sub.add_parser(
        "prune-corpus",
        help="unregister duplicate corpus binaries left by a repeated build",
    )
    prune.add_argument(
        "--apply",
        action="store_true",
        help="actually unregister them (otherwise only report)",
    )
    prune.set_defaults(func=cmd_prune_corpus)

    dataset = sub.add_parser(
        "dataset",
        help="report the trainable dataset and check its split for leakage",
    )
    dataset.add_argument(
        "--assign",
        action="store_true",
        help="recompute and store the package split",
    )
    dataset.set_defaults(func=cmd_dataset)

    bases = sub.add_parser(
        "baselines",
        help="measure model-free retrieval baselines (the week 6 bar)",
    )
    bases.add_argument("--split", default="test", help="which split to score")
    bases.add_argument("--query-opt", default="O0", help="query side, e.g. O0")
    bases.add_argument("--pool-opt", default="O3", help="pool side, e.g. O3")
    bases.add_argument("--seed", type=int, default=1)
    bases.add_argument(
        "--no-record",
        action="store_true",
        help="print the numbers without storing them",
    )
    bases.set_defaults(func=cmd_baselines)

    hist = sub.add_parser(
        "measurements", help="show past evaluation results"
    )
    hist.add_argument("--split", help="restrict to one split")
    hist.add_argument("--method", help="restrict to one method")
    hist.add_argument("--limit", type=int, default=40)
    hist.set_defaults(func=cmd_measurements)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
