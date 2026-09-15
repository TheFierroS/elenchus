"""Turn the lists data.py builds into padded tensors.

The only torch in the data path, kept deliberately thin: every decision about
what a batch contains is made and tested in data.py. What is left here is
geometry - rectangular tensors, and a mask saying which positions are real -
and the one mistake this layer can make on its own is letting padding count.
"""

import torch

from elenchus.encoder.data import IGNORE_INDEX, false_negatives


def pad(sequences, pad_value):
    """Stack variable-length id lists into (ids, mask).

    ids is [batch, longest] filled with pad_value past each sequence's end;
    mask is True exactly where a real token sits. Padding goes to the longest
    sequence in the batch, not to a fixed maximum: with a median function of
    about 130 tokens and a limit of 1024, fixed padding would spend most of
    every batch on nothing.
    """
    if not sequences:
        raise ValueError("cannot pad an empty batch")
    if any(len(s) == 0 for s in sequences):
        raise ValueError("every sequence needs at least one token")

    longest = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), longest), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), longest), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = True
    return ids, mask


def collate_pairs(batch, pad_id):
    """Tensors for one contrastive batch.

    false_negatives[i][j] is True where positive j compiles to the same
    normalised code as anchor i or its positive, and so must not be counted
    as a negative for i. The loss applies it; it is built here so the loss
    never sees content hashes.
    """
    anchor_ids, anchor_mask = pad(batch.anchor, pad_id)
    positive_ids, positive_mask = pad(batch.positive, pad_id)
    mask = false_negatives(batch.anchor_content, batch.positive_content,
                           batch.positive_content)
    return {
        "anchor_ids": anchor_ids,
        "anchor_mask": anchor_mask,
        "positive_ids": positive_ids,
        "positive_mask": positive_mask,
        "false_negatives": torch.tensor(mask, dtype=torch.bool),
    }


def collate_masked(batch, pad_id):
    """Tensors for one masked-LM batch; padded label positions are ignored."""
    input_ids, attention_mask = pad(batch.inputs, pad_id)
    labels, _ = pad(batch.labels, IGNORE_INDEX)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
