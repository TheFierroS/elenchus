"""A size no instruction has is not read (F42)."""

import pytest

from elenchus.emulation.branches import Decoder


def test_a_size_unicorn_could_not_decode_is_not_read():
    """Unicorn reports an undecodable instruction as 0xf1f1f1f1 bytes long.
    Reading that many cost most of a run; the answer was None either way."""
    pytest.importorskip("capstone")
    asked = []

    def read(address, size):
        asked.append(size)
        return b"\x74\x05" + b"\x90" * 14     # je +5, then padding

    decoder = Decoder(read)
    assert decoder.branch_at(0x1000, 0xF1F1F1F1) is None
    assert asked == []


def test_a_real_instruction_is_still_decoded():
    pytest.importorskip("capstone")
    asked = []

    def read(address, size):
        asked.append(size)
        return b"\x74\x05"                     # je +5

    found = Decoder(read).branch_at(0x2000, 2)
    assert found is not None
    assert found.target == 0x2007
    assert found.fall_through == 0x2002
    assert asked == [2]
