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


# ------------------------------------------- the translation buffer (F45)


def test_each_run_asks_for_a_small_translation_buffer(monkeypatch):
    """Unicorn's default is 1 GiB per emulator; a run asks for TCG_BUFFER,
    and still computes what it computed."""
    from unicorn import Uc

    from elenchus.emulation.harness import TCG_BUFFER
    asked = []
    original = Uc.ctl_set_tcg_buffer_size

    def spy(self, size):
        asked.append(size)
        return original(self, size)

    monkeypatch.setattr(Uc, "ctl_set_tcg_buffer_size", spy)
    loader, address, place = _sum()
    out = run(loader, address, place, [BUF, 9], budget=5_000_000)
    assert out.status is Status.COMPLETED
    assert asked == [TCG_BUFFER]


# ------------------------------------------- closing the emulator (F46)


def _recording(monkeypatch):
    """Make run() build its emulator through a function that keeps it.

    Unicorn's Uc hands back an architecture's own class when built, so a
    subclass of it breaks; a function that calls the real one and records
    what came back does not.
    """
    import unicorn

    made = []
    real = unicorn.Uc

    def make(*args, **kwargs):
        emulator = real(*args, **kwargs)
        made.append(emulator)
        return emulator

    monkeypatch.setattr(unicorn, "Uc", make)
    return made


def test_a_run_closes_its_emulator(monkeypatch):
    """F19's release_handle() had been raising TypeError since Unicorn 2.1,
    and a bare except hid it. This checks the emulator is closed, not that
    something was asked to close it."""
    made = _recording(monkeypatch)
    loader, address, place = _sum()
    out = run(loader, address, place, [BUF, 9], budget=5_000_000)
    assert out.status is Status.COMPLETED
    assert len(made) == 1
    assert not made[0]._Uc__finalizer.alive


def test_a_faulting_run_closes_its_emulator(monkeypatch):
    """The finally covers every way out, a fault included."""
    made = _recording(monkeypatch)
    loader, address, place = _sum()
    out = run(loader, address, place, [0, 9], budget=5_000_000)
    assert out.status is not Status.COMPLETED
    assert len(made) == 1
    assert not made[0]._Uc__finalizer.alive
