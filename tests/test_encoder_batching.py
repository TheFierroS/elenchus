"""Tests for the tensor layer: padding and masks.

The one mistake this layer can make on its own is letting padding count - as
tokens the model attends to, as positions the masked-LM loss is taken over,
or as a lost real token at a sequence's end. Each test checks one of those.
"""

import pytest
import torch

from elenchus.encoder.batching import collate_masked, collate_pairs, pad
from elenchus.encoder.data import IGNORE_INDEX, MaskedBatch, PairBatch

PAD = 0


def test_padding_goes_to_the_longest_sequence_in_the_batch():
    ids, mask = pad([[5, 6, 7], [8]], PAD)
    assert ids.tolist() == [[5, 6, 7], [8, PAD, PAD]]
    assert ids.dtype == torch.long


def test_the_mask_is_true_exactly_on_real_tokens():
    sequences = [[5, 6, 7], [8], [9, 10]]
    _ids, mask = pad(sequences, PAD)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [len(s) for s in sequences]
    assert mask.tolist() == [
        [True, True, True], [True, False, False], [True, True, False]]


def test_a_real_token_equal_to_the_pad_value_is_still_real():
    """The mask comes from lengths, never from comparing ids to PAD."""
    ids, mask = pad([[PAD, 3], [4]], PAD)
    assert mask.tolist() == [[True, True], [True, False]]


def test_an_empty_batch_or_sequence_is_refused():
    with pytest.raises(ValueError):
        pad([], PAD)
    with pytest.raises(ValueError):
        pad([[1], []], PAD)


def test_masked_labels_are_ignored_on_padding():
    batch = MaskedBatch(inputs=[[1, 9, 9], [1, 9]],
                        labels=[[IGNORE_INDEX, 42, IGNORE_INDEX], [IGNORE_INDEX, 7]])
    tensors = collate_masked(batch, PAD)

    assert tensors["labels"].tolist() == [[IGNORE_INDEX, 42, IGNORE_INDEX],
                                          [IGNORE_INDEX, 7, IGNORE_INDEX]]
    assert tensors["attention_mask"].tolist() == [[True, True, True], [True, True, False]]


def test_pair_tensors_keep_rows_aligned_and_carry_the_false_negative_mask():
    batch = PairBatch(
        anchor=[[1, 10, 11], [1, 12]],
        positive=[[1, 20], [1, 21, 22, 23]],
        identities=[("p", "a.c", "f"), ("p", "a.c", "g")],
        anchor_levels=["O0", "O0"], positive_levels=["O3", "O3"],
        anchor_content=["qa", "qb"],
        positive_content=["same", "same"],
    )
    tensors = collate_pairs(batch, PAD)

    assert tensors["anchor_ids"].shape == (2, 3)
    assert tensors["positive_ids"].shape == (2, 4)
    assert tensors["positive_mask"].sum(dim=1).tolist() == [2, 4]
    # Both positives are the same code: each is a false negative for the other.
    assert tensors["false_negatives"].tolist() == [[False, True], [True, False]]
