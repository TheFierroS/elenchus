"""Move training off this machine and bring its record back.

The database is mostly its own audit trail: of 14 GB, about 11.7 GB is
events and their links, and training reads none of it. What training reads
is ground truth, the listings, which binary belongs to which package, and
the split - about 1.3 GB. Exporting that makes a rented GPU a few minutes
of upload rather than hours, and keeps the full trail here, where it is
read.

The exported database carries the same row ids as this one, so anything
the remote writes that refers to a binary or a function still refers to
the same thing when it comes back. Only `runs` and `measurements` are
written remotely, and only their own ids have to be renumbered on the way
in.
"""

import sqlite3
from pathlib import Path

from elenchus.db import connect, schema

# What training and validation read. Anything not here is either the audit
# trail or something a training run never opens.
EXPORTED = (
    "binaries",
    "functions",
    "corpus_binaries",
    "ground_truth",
    "function_code",
    "dataset_split",
    "corpus_gaps",
)


def export_training(conn, out):
    """Write a database holding only what a training run reads.

    Returns {table: rows}. Refuses to overwrite: an export is cheap to
    repeat and a database is not cheap to lose.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; move it aside first")

    target = connect(out)
    target.executescript(schema())
    target.commit()
    target.close()

    counts = {}
    conn.execute("ATTACH DATABASE ? AS target", (str(out),))
    try:
        for table in EXPORTED:
            conn.execute(f"INSERT INTO target.{table} SELECT * FROM main.{table}")
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM target.{table}").fetchone()[0]
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE target")

    return counts


def _rows(conn, table, source="remote"):
    return conn.execute(f"SELECT * FROM {source}.{table}").fetchall()


def import_runs(conn, path, apply=False):
    """Copy the runs (and their measurements) from an exported database.

    Renumbered onto the end of this database's runs, because both sides
    started counting from whatever they inherited. Nothing else is copied:
    a remote training run writes a runs row and nothing more, and if it
    measured baselines those rows point at the run by id, which is the one
    thing that has to be rewritten.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    conn.execute("ATTACH DATABASE ? AS remote", (str(path),))
    try:
        runs = _rows(conn, "runs")
        measurements = _rows(conn, "measurements")
        if not runs:
            return {"runs": [], "measurements": 0}

        known = {(r["code_version"], r["started_at"], r["kind"])
                 for r in conn.execute("SELECT * FROM main.runs")}
        fresh = [r for r in runs
                 if (r["code_version"], r["started_at"], r["kind"]) not in known]

        next_id = (conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM main.runs").fetchone()[0]) + 1
        mapping = {r["id"]: next_id + i for i, r in enumerate(fresh)}
        moved = [m for m in measurements if m["run_id"] in mapping]

        if apply:
            columns = [c[1] for c in conn.execute("PRAGMA table_info(runs)")]
            placeholders = ", ".join("?" * len(columns))
            for run in fresh:
                values = [mapping[run["id"]] if c == "id" else run[c]
                          for c in columns]
                conn.execute(
                    f"INSERT INTO main.runs ({', '.join(columns)}) "
                    f"VALUES ({placeholders})", values)

            columns = [c[1] for c in conn.execute("PRAGMA table_info(measurements)")]
            placeholders = ", ".join("?" * len(columns))
            for row in moved:
                values = [None if c == "id" else
                          (mapping[row["run_id"]] if c == "run_id" else row[c])
                          for c in columns]
                conn.execute(
                    f"INSERT INTO main.measurements ({', '.join(columns)}) "
                    f"VALUES ({placeholders})", values)
            conn.commit()

        return {"runs": [(r["id"], mapping[r["id"]], r["kind"], r["started_at"])
                         for r in fresh],
                "measurements": len(moved),
                "already_here": len(runs) - len(fresh)}
    finally:
        conn.execute("DETACH DATABASE remote")


def cmd_export_training(conn, args):
    counts = export_training(conn, args.out)
    size = Path(args.out).stat().st_size / (1024 ** 3)
    for table, n in counts.items():
        print(f"  {table:18} {n:>8}")
    print(f"\nwrote {args.out} ({size:.2f} GiB)")
    print("training reads this; the events and their links stay here")
    return 0


def cmd_import_runs(conn, args):
    try:
        found = import_runs(conn, args.path, apply=args.apply)
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"cannot read {args.path}: {exc}")
        return 1

    if not found["runs"]:
        print("no new runs in that database")
        return 0

    for old, new, kind, started in found["runs"]:
        print(f"  run {old:>5} -> {new:<5} {kind:<12} {started}")
    print(f"\n{len(found['runs'])} run(s), "
          f"{found['measurements']} measurement(s)")
    if found.get("already_here"):
        print(f"{found['already_here']} run(s) were already here")
    if not args.apply:
        print("\nNothing changed. Pass --apply to record them.")
    return 0
