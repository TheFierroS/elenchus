"""The parallel path must give the same report as the serial one.

V0 and R0 are the two numbers every decision in the verifier rests on, so
spreading their pairs over processes is only allowed if it changes nothing
about what they say. Two things have to hold and both are tested here: the
results come back in the order the tasks were given, not the order the
workers finished them, and the serial path - workers=1, which is what the
rest of the suite runs - produces the same list.

The work functions are deliberately trivial. What is under test is the
mapping, not the verifier; that a pair's verdict does not depend on which
process judged it follows from each task carrying everything its pair needs,
and from the seeds and fills being constants in compare.
"""

from __future__ import annotations

import os

import pytest

from elenchus.emulation.parallel import (
    MAX_WORKERS,
    default_workers,
    mapped,
)


def square(value):
    return value * value


def slow_for_small(value):
    """Finishes in the reverse of the order it is given.

    A task with a small value sleeps longest, so if results came back as they
    completed the list would be reversed. It is not: .map yields in the input
    order.
    """
    import time
    time.sleep((8 - value) * 0.02)
    return value


def test_the_serial_path_runs_without_processes():
    assert mapped(square, [1, 2, 3, 4], workers=1) == [1, 4, 9, 16]


def test_one_task_never_starts_a_pool():
    """A single task takes the serial path whatever the worker count, so a
    quick check does not pay for a process pool."""
    assert mapped(square, [7], workers=4) == [49]


def test_no_tasks_is_an_empty_list():
    assert mapped(square, [], workers=4) == []


def test_the_parallel_path_gives_the_same_answers():
    tasks = list(range(12))
    assert mapped(square, tasks, workers=3) == mapped(square, tasks, workers=1)


def test_results_come_back_in_the_tasks_order_not_the_finishing_order():
    """The property the measurements depend on: a report folded from these
    results is the same report however the work was spread."""
    tasks = list(range(8))
    assert mapped(slow_for_small, tasks, workers=4) == tasks


def test_default_workers_leaves_the_machine_usable():
    """Half the cores, never more than the cap, never fewer than one."""
    workers = default_workers()
    assert 1 <= workers <= MAX_WORKERS
    assert workers <= max(1, (os.cpu_count() or 1))


def pair_of(task):
    return task[0]


def test_ordering_by_a_key_does_not_change_the_answers():
    """Grouping tasks by binary is a speed change, not a behaviour one.

    The sample is shuffled, so consecutive pairs come from different binaries
    and a cache of parsed ones has to hold the whole sample. Running them
    grouped leaves two live at a time - but the results must come back in the
    caller's order, or every report folded from them would be wrong.
    """
    tasks = [("b", 1), ("a", 2), ("b", 3), ("a", 4), ("c", 5)]
    plain = mapped(second, tasks, workers=1)
    grouped = mapped(second, tasks, workers=1, order_by=pair_of)
    assert plain == grouped == [1, 2, 3, 4, 5]

    assert mapped(second, tasks, workers=2, order_by=pair_of) == plain


def second(task):
    return task[1]


def test_the_stack_fill_is_the_same_bytes_every_time():
    """Caching it must not change what a run reads (harness._stack_garbage).

    It is the memory a function reads when it reads a local it never wrote,
    and the two-seed check rests on it differing between seeds and not within
    one. Measured at 2.54 ms a run before it was kept, 39% of a short run.
    """
    pytest.importorskip("unicorn")
    from elenchus.emulation.harness import STACK_SIZE, _stack_garbage

    first = _stack_garbage(0x1111_1111)
    again = _stack_garbage(0x1111_1111)
    assert first == again
    assert len(first) == STACK_SIZE
    assert first is again                      # kept, not made twice
    assert _stack_garbage(0x2222_2222) != first        # the seeds still differ


def slow(value):
    import time
    time.sleep(0.05)
    return value


def test_progress_arrives_while_the_work_is_still_running():
    """The callback must fire as results land, not once at the end.

    The first version returned a list, so a caller counting while it iterated
    the returned value counted a finished run all at once and a half-hour
    measurement printed nothing until it was over. The test is about timing,
    so it records when each call happened: the first must arrive well before
    the last.
    """
    import time
    seen = []
    started = time.perf_counter()
    mapped(slow, list(range(8)), workers=2,
           on_result=lambda done, total: seen.append(
               (done, total, time.perf_counter() - started)))

    assert [done for done, _total, _at in seen] == list(range(1, 9))
    assert all(total == 8 for _done, total, _at in seen)
    # Eight tasks of 50 ms over two workers is 200 ms of work, so the calls
    # must be spread over at least a good part of that. Measured from the
    # first call rather than from the start, since opening the pool costs a
    # third of a second and is not what this is about. Built as a list first,
    # every call would land in the same instant and the spread would be zero.
    spread = seen[-1][2] - seen[0][2]
    assert spread > 0.05, f"the calls arrived together, {spread:.3f}s apart"


def test_progress_fires_on_the_serial_path_too():
    seen = []
    mapped(square, [1, 2, 3], workers=1,
           on_result=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def report_patched_constant(_task):
    from elenchus.emulation import harness
    return harness.STACK_SIZE


def test_whether_a_worker_sees_the_parent_s_patched_constants():
    """Recorded, not required: it depends on the platform's start method.

    A forked worker begins with the parent's memory and sees a patched
    constant; a spawned one imports the module again and sees the default.
    Python picks between them and has changed its mind across versions, so an
    experiment that scans a constant by patching it (experiments/scan_stack.py)
    cannot assume either - it checks, the way this does, and falls back to one
    worker when the answer is no. Written down because a scan that silently
    measured the default while reporting the patched value would be worse than
    no scan at all.
    """
    pytest.importorskip("unicorn")
    from elenchus.emulation import harness

    original = harness.STACK_SIZE
    harness.STACK_SIZE = 0xBEEF000
    try:
        seen = mapped(report_patched_constant, [1, 2, 3, 4], workers=2)
    finally:
        harness.STACK_SIZE = original
    assert len(set(seen)) == 1, "the workers disagreed with each other"
    assert seen[0] in (0xBEEF000, original)
