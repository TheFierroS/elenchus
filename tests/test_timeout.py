"""A run past its time stops itself, from inside the emulation (F44).

Unicorn's own timeout stops the emulator from a second thread, which raced
with the hooks' own stops and segfaulted. The clock is now read from the
code hook, so these check that it still stops a long run and leaves a short
one alone.
"""

from pathlib import Path

import pytest

from elenchus.corpus.dwarf import ground_truth
from elenchus.emulation.abi import placement
from elenchus.emulation.harness import Loader, Status, run

pytest.importorskip("unicorn")

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"
BUF = 0x0000_2000_0000_0000


def _sum():
    dll = FIXTURES / "cases_O0.dll"
    signature = {f.name: f for f in ground_truth(dll)}["sum"]
    return Loader(dll), signature.address, placement(signature.abi)


def test_a_run_past_its_time_is_stopped_by_the_hook():
    """sum over a hundred thousand ints would take far more than the budget
    of one microsecond allows; the hook's first look at the clock stops it."""
    loader, address, place = _sum()
    out = run(loader, address, place, [BUF, 100_000],
              budget=5_000_000, timeout_us=1)
    assert out.status is Status.TIMED_OUT
    assert out.instructions < 5_000


def test_a_run_within_its_time_completes():
    loader, address, place = _sum()
    out = run(loader, address, place, [BUF, 9], budget=5_000_000)
    assert out.status is Status.COMPLETED
