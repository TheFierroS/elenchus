"""Training objectives: masked-LM loss, InfoNCE, and the MoCo memory queue.

InfoNCE, in one sentence: for each query, the loss is the cross-entropy of
picking its own key out of every candidate key, with similarities divided by
a temperature. It is the evaluation task turned into a loss - find the -O3
twin of an -O0 function among many.

Candidates that must not count as wrong are removed, not merely down-weighted:
  - keys whose normalised code is identical to the query's or its key's
    (no encoder reading normalised text could tell them apart)
  - queue entries from the same function (it was enqueued in an earlier step)
  - queue slots not yet filled
The true key is never removed.

MoCo: a batch that fits in 8 GB holds a few dozen functions - few negatives.
Keys from earlier batches are kept in a fixed-size queue and reused as
negatives. They must be comparable with current keys, so keys come from a
momentum copy of the encoder that drifts slowly towards the trained one.
"""

import copy

import torch
import torch.nn.functional as F

from elenchus.encoder.data import IGNORE_INDEX

_REMOVED = float("-inf")


def mlm_loss(logits, labels):
    """Cross-entropy over hidden positions only; labels are IGNORE_INDEX elsewhere."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           labels.reshape(-1), ignore_index=IGNORE_INDEX)


def info_nce(queries, keys, temperature, false_negatives=None,
             queue=None, queue_excluded=None):
    """InfoNCE loss; the key for query i is keys[i].

    queries, keys   [B, D], unit length
    false_negatives [B, B] bool, True where keys[j] must not count against i
    queue           [K, D] extra negative keys, or None
    queue_excluded  [B, K] bool, True where queue[k] must not count against i
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries, keys = queries.float(), keys.float()
    batch = queries.shape[0]

    logits = queries @ keys.T
    if false_negatives is not None:
        if false_negatives.shape != logits.shape:
            raise ValueError(f"false_negatives is {tuple(false_negatives.shape)}, "
                             f"logits are {tuple(logits.shape)}")
        # Built at the logits' own shape. torch.eye(batch) broadcast silently
        # over a wider key set and switched the whole mask off.
        answer = torch.zeros_like(false_negatives)
        answer[torch.arange(batch), torch.arange(batch)] = True
        logits = logits.masked_fill(false_negatives & ~answer, _REMOVED)

    if queue is not None and queue.shape[0]:
        extra = queries @ queue.float().T
        if queue_excluded is not None:
            extra = extra.masked_fill(queue_excluded, _REMOVED)
        logits = torch.cat([logits, extra], dim=1)

    targets = torch.arange(batch, device=queries.device)
    return F.cross_entropy(logits / temperature, targets)


class MemoryQueue:
    """A fixed-size FIFO of past keys, with what is needed to exclude them.

    Identities and content hashes are stored as integer codes so exclusion is
    one tensor comparison, not a Python loop: with a batch of 64 and a queue
    of 4,096 that is 262,144 comparisons every step.
    """

    def __init__(self, size, dim, device="cpu"):
        if size < 0:
            raise ValueError("queue size cannot be negative")
        self.size = size
        self.device = device
        self.keys = torch.zeros(size, dim, device=device)
        self.identity_codes = torch.full((size,), -1, dtype=torch.long, device=device)
        self.content_codes = torch.full((size,), -1, dtype=torch.long, device=device)
        self.filled = 0
        self.cursor = 0
        self._codes = {}

    def _code(self, value):
        return self._codes.setdefault(value, len(self._codes))

    def _codes_for(self, values):
        return torch.tensor([self._code(v) for v in values], dtype=torch.long,
                            device=self.device)

    def enqueue(self, keys, identities, contents):
        if self.size == 0:
            return
        keys = keys.detach().float()
        identity_codes = self._codes_for(identities)
        content_codes = self._codes_for(contents)
        for row in range(keys.shape[0]):
            self.keys[self.cursor] = keys[row]
            self.identity_codes[self.cursor] = identity_codes[row]
            self.content_codes[self.cursor] = content_codes[row]
            self.cursor = (self.cursor + 1) % self.size
            self.filled = min(self.filled + 1, self.size)

    def excluded(self, identities, query_contents, key_contents):
        """[B, size] bool: True where a queue slot must not be a negative for i."""
        identity = self._codes_for(identities)[:, None]
        query = self._codes_for(query_contents)[:, None]
        key = self._codes_for(key_contents)[:, None]
        slots = torch.arange(self.size, device=self.device)
        empty = (slots >= self.filled)[None, :]
        return (empty
                | (self.identity_codes[None, :] == identity)
                | (self.content_codes[None, :] == query)
                | (self.content_codes[None, :] == key))


def momentum_copy(model):
    """A frozen copy that will follow model through momentum_update."""
    target = copy.deepcopy(model)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


@torch.no_grad()
def momentum_update(model, target, momentum):
    """target <- momentum * target + (1 - momentum) * model."""
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    for online, slow in zip(model.parameters(), target.parameters()):
        slow.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
