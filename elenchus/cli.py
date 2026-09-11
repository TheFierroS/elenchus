"""Command line interface for Elenchus."""

import argparse
import sys
import time

import pyghidra

from elenchus.check import check_links
from elenchus.db import connect, finish_run, set_run_tool_version, start_run
from elenchus.extract.ghidra import (
    extract_blocks,
    extract_calls,
    extract_functions,
    extract_imports,
    extract_strings,
    ghidra_version,
    register_binary,
)


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
    print(f"strings  : {strings}")
    print(f"elapsed  : {time.time() - started:.1f}s")
    return 0


def cmd_check(args):
    """Verify that every event link points at an entity that exists."""
    conn = connect(args.db)
    broken = check_links(conn)

    total = conn.execute("SELECT COUNT(*) AS n FROM event_links").fetchone()["n"]
    print(f"checked {total} links, {len(broken)} broken")

    for event_id, kind, entity_id in broken:
        print(f"  event {event_id} -> missing {kind} {entity_id}")

    return 1 if broken else 0


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
    check.set_defaults(func=cmd_check)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
