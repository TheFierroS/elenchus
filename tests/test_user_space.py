"""An access above user space faults, as it does on Windows (F48).

The harness maps any page a run first touches, outside the null guard. Above
the end of user space - the kernel's half and the non-canonical hole - a real
machine faults instead, and mapping a page there for a wild pointer crashed
libunicorn in one process in five. These run the fixtures' `sum` with its
pointer on either side of the line.
"""

from pathlib import Path

import pytest

pytest.importorskip("unicorn")

from elenchus.corpus.dwarf import ground_truth  # noqa: E402
from elenchus.emulation.abi import placement  # noqa: E402
from elenchus.emulation.harness import (  # noqa: E402
    PAGE,
    USER_SPACE_END,
    Loader,
    Status,
    run,
)

FIXTURES = Path(__file__).parent / "fixtures" / "emulation"


def _sum():
    path = FIXTURES / "cases_O0.dll"
    for function in ground_truth(path):
        if function.name == "sum":
            return Loader(path), function.address, placement(function.abi)
    raise LookupError("sum is not in the fixture")


def test_an_access_above_user_space_faults():
    loader, address, place = _sum()
    outcome = run(loader, address, place, [USER_SPACE_END, 4], budget=5_000)
    assert outcome.status is Status.FAULT
    assert "above user space" in outcome.detail
    assert outcome.first_wild[1] == USER_SPACE_END


def test_an_access_just_below_user_space_is_still_mapped():
    """The line is where Windows draws it, not lower: a wild pointer below
    it is filled and read like any other, and marked as invented (F32)."""
    loader, address, place = _sum()
    outcome = run(loader, address, place, [USER_SPACE_END - PAGE, 4],
                  budget=5_000)
    assert outcome.status is Status.COMPLETED
    assert outcome.read_invented
