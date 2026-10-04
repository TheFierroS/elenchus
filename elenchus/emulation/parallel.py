"""Run the measurements' pairs across processes instead of one at a time.

V0's 3000 pairs took 112 minutes, of which 81 were kernel time: every run maps
the binary's image into a fresh emulator, and there are tens of thousands of
runs. The runs are entirely independent - each one builds its own emulator,
its own arena and its own stack, and nothing is carried between them - so the
cost divides by however many cores are given to it.

Two properties this must hold, because the measurements are worthless without
them:

*Identical results.* A pair's verdict cannot depend on how many workers ran.
Results come back in the order the tasks were given, not the order they
finished, and each task carries everything its pair needs; the seeds and fills
are constants in `compare`, so two processes judging one pair reach the same
verdict as one process would. `workers=1` takes the serial path, which is what
the tests compare against.

*No leak carried across.* F19 and F20 were two memory leaks that killed a
3000-pair run at 9 GB, and the lesson there - every run releases the
emulator's handle in a `finally` - is untouched here: a worker runs exactly
the same `run()`. What a worker does accumulate is its own cache of parsed
binaries, so processes are recycled after a fixed number of tasks and the
cache goes with them. Memory is per worker, so four workers want roughly four
times one run's footprint; the verifier sits at about 800 MB, and the WSL
default of 10 GB is what the ceiling below is chosen against.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

# Workers take the platform's own start method, and are not recycled.
#
# Neither of those is the first thing that was tried. The first version asked
# for max_tasks_per_child, to bound a worker's cache of parsed binaries, and
# that silently changed the start method - Python refuses fork with
# max_tasks_per_child and falls back to spawn, where a worker starts a fresh
# interpreter and imports everything again, pyghidra included. That looked
# like the reason eight workers measured slower than two, so the next version
# forced fork. Measured, it was not: 1.45 / 2.37 / 2.40 / 1.97 pairs a second
# at 1, 2, 4 and 8 workers, against 1.50 / 2.47 / 2.43 / 2.16 with spawn. No
# difference, and forcing fork costs something real - from Python 3.14 forking
# a process that has threads is deprecated and can deadlock the child.
#
# So the start method goes back to the default. What the recycling was for is
# done where it belongs instead: v0 and r0 keep the last few loaders, so a
# worker's footprint does not depend on how long it lives.

# Measured, not chosen (experiments/scan_workers.sh, 100 R0 pairs on a
# twenty-core machine):
#
#   workers   1      2      4      8
#   pairs/s   1.50   2.47   2.43   2.16
#
# Two is the whole of it. Four buys nothing and eight is worse than two, on a
# machine with twenty cores - so the work is not CPU-bound past that point: it
# is waiting on something the processes share, and what every run asks the
# kernel for is memory, mapped and given up again. More workers then only add
# contention. The ceiling stands until that cost comes down.
MAX_WORKERS = 2


def default_workers() -> int:
    """Half the cores, capped, and never more than one per core.

    Half rather than all: the measurements are run on the machine someone is
    using, and a run that takes the whole machine for twenty minutes is a run
    they will not start.
    """
    cores = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, cores // 2))


# The verdicts were identical at 1, 2, 4 and 8 workers in that scan, which is
# the property this module exists to keep and the first thing it checked.


def mapped(work, tasks, workers: int = 1, order_by=None, on_result=None):
    """Apply `work` to every task, returning results in the tasks' order.

    `work` must be a module-level function and every task picklable, since
    they cross a process boundary. With `workers` at 1, or a single task, the
    serial path runs instead - same results, no processes, and it is what the
    parallel path is tested against.

    `order_by`, if given, is a key the tasks are run in rather than the order
    they arrive. The sample is shuffled, so consecutive pairs come from
    different binaries and a cache of parsed ones has to hold the whole
    sample; grouping by binary leaves two live at a time. The results are put
    back in the caller's order before returning, so this changes how long it
    takes and nothing about what it says.

    `on_result`, if given, is called with (done, total) as each result
    arrives. This exists because the function returns a list: a caller that
    counted while iterating the returned value would count a finished run all
    at once, which is how the first version managed to report no progress at
    all for the whole of a half-hour measurement.
    """
    tasks = list(tasks)
    if not tasks:
        return []

    order = range(len(tasks))
    if order_by is not None:
        order = sorted(order, key=lambda i: order_by(tasks[i]))
        tasks = [tasks[i] for i in order]

    total = len(tasks)
    results = []
    if workers <= 1 or total <= 1:
        for task in tasks:
            results.append(work(task))
            if on_result is not None:
                on_result(len(results), total)
    else:
        # Hand each worker a contiguous block rather than every Nth task.
        # With one task at a time, four workers walking a list grouped by
        # binary all reach the same binary at the same moment and all four
        # parse it, which undoes the grouping entirely - 16 ms a parse on a
        # small library and more on a real one, over a hundred of them.
        chunk = max(1, min(32, total // (workers * 4)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # .map yields in the order of the input, whatever order they
            # finish in, which is what makes a parallel report identical to a
            # serial one. Consumed one at a time rather than with list(), so a
            # caller can be told about each as it lands.
            for result in pool.map(work, tasks, chunksize=chunk):
                results.append(result)
                if on_result is not None:
                    on_result(len(results), total)

    if order_by is None:
        return results
    restored = [None] * len(results)
    for position, original in enumerate(order):
        restored[original] = results[position]
    return restored
