"""How much of a run is the megabyte of stack it maps and fills?

The worker scan said the measurement stops scaling at two processes on a
twenty-core machine, which means the work is not CPU-bound: it is waiting on
something shared, and the obvious candidate is the memory every run maps and
gives up again. The largest single piece of that is the stack - 1 MiB mapped
and 1 MiB written, per run, tens of thousands of times - and a function
starts at STACK_TOP - 0x2000 and uses a few kilobytes of it.

So: does a smaller stack cost anything? This runs R0 at each size and reads
two things together, as every parameter in this verifier has been chosen:

  *time*, which is the point of asking; and
  *the verdicts*, which must not move. A stack too small to hold a deep frame
  faults instead of reading garbage, which turns judged pairs into faults -
  a coverage loss that would not show in the clock. If the counts change at a
  size, that size is refused whatever it does for the time.

Serial on purpose: the sizes are compared against each other, and a worker
process would not see the patched constant anyway.

    python experiments/scan_stack.py 2>&1 | tee data/scan_stack.log

About eight minutes at the default count.
"""

from __future__ import annotations

import argparse
import os
import time

from elenchus.db import connect
from elenchus.emulation import harness
from elenchus.emulation.r0 import run_r0

SIZES = {
    "1M": ("1 MiB  (today)", 0x100000),
    "256K": ("256 KiB", 0x40000),
    "64K": ("64 KiB", 0x10000),
    "16K": ("16 KiB", 0x4000),
}

# Patching the constant reaches a *forked* worker, which begins with this
# process's memory, and not a *spawned* one, which imports the module again
# and gets the default. Python chooses between them and has changed its mind
# across versions, so this is checked rather than assumed: a scan that
# measured the default while printing the patched size would be worse than no
# scan at all. When the check fails the scan drops to one worker and says so.


def _size_a_worker_sees(_task):
    from elenchus.emulation import harness as inner
    return inner.STACK_SIZE


def _reaches_workers(size, workers):
    """Whether a worker runs the size this process just set."""
    from elenchus.emulation.parallel import mapped
    seen = mapped(_size_a_worker_sees, list(range(workers)), workers=workers)
    return all(value == size for value in seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="run each size at this worker count. The whole "
                             "question the second pass asks is whether a "
                             "smaller stack raises the ceiling or disappears "
                             "into it, and that needs more than one worker.")
    parser.add_argument("--sizes", default="1M,256K,64K,16K",
                        help="which sizes to run, comma separated: "
                             + ",".join(SIZES))
    parser.add_argument("--db", default=os.environ.get("ELENCHUS_DB"))
    options = parser.parse_args()
    if not options.db:
        raise SystemExit("set ELENCHUS_DB or pass --db")

    chosen = []
    for key in options.sizes.split(","):
        key = key.strip()
        if key not in SIZES:
            raise SystemExit(f"unknown size {key!r}; pick from "
                             + ", ".join(SIZES))
        chosen.append(SIZES[key])

    original = harness.STACK_SIZE
    rows = []
    for label, size in chosen:
        # The fill is cached per seed and its length is the stack's, so the
        # cache has to go with the constant or the next run writes the old
        # size into the new mapping.
        harness.STACK_SIZE = size
        harness._STACK_BLOCKS.clear()

        workers = options.workers
        if workers > 1 and _reaches_workers(size, workers):
            pass
        elif workers > 1:
            print("  the workers do not see the patched size on this "
                  "platform; falling back to one", flush=True)
            workers = 1

        conn = connect(options.db)
        started = time.time()
        report = run_r0(conn, count=options.count, seed=options.seed,
                        workers=workers)
        elapsed = time.time() - started
        rows.append((label, size, elapsed, dict(report.verdicts),
                     report.forced_survivals))
        print(f"\n{label:16} {elapsed:7.1f}s  "
              f"{options.count / elapsed:5.2f} pairs/s  "
              f"{dict(report.verdicts)}  forced {report.forced_survivals}",
              flush=True)

    harness.STACK_SIZE = original
    harness._STACK_BLOCKS.clear()

    print("\n" + "=" * 68)
    print(f"workers: {options.workers}")
    first = chosen[0][0]
    print(f"{'stack':16} {'seconds':>9} {'pairs/s':>9} {'vs ' + first:>14}"
          f"   verdicts")
    base = rows[0][2]
    for label, _size, elapsed, verdicts, forced in rows:
        print(f"{label:16} {elapsed:9.1f} {options.count / elapsed:9.2f} "
              f"{base / elapsed:13.2f}x   {verdicts} forced {forced}")

    moved = [label for label, _s, _e, verdicts, forced in rows[1:]
             if (verdicts, forced) != (rows[0][3], rows[0][4])]
    print()
    if moved:
        print("VERDICTS MOVED at: " + ", ".join(moved))
        print("Those sizes are refused whatever they did for the clock: a")
        print("stack too small faults where the old one read garbage, and")
        print("that is coverage lost, not time saved.")
    else:
        print("The verdicts held at every size. The smallest that holds them")
        print("is the candidate, and it is confirmed on V0 before it is kept -")
        print("R0 at this count cannot see a small loss of coverage.")


if __name__ == "__main__":
    main()
