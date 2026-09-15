"""What the encoder is shown during training, and in what order.

The model can only learn what the batches teach, and a batch can teach the
wrong thing without any error: the same function on both sides of a
comparison, two indistinguishable functions labelled as different, a
sequence cut so its two views no longer line up. None of these crash. They
produce a model that trains smoothly and learns something other than what was
intended. So this module holds every decision about what goes into a batch,
in plain Python, where each one can be tested on its own.

Two training stages draw from it.

Masked language modelling (MLM) shows one function with some tokens hidden
and asks for them back. It needs no labels, only code, and teaches the model
the grammar of normalised assembly before it is asked anything harder.

Contrastive training shows the same function at two optimisation levels and
asks the model to pick one view out of a batch given the other. That is the
evaluation task itself: find the -O3 twin of an -O0 query among many
functions.

Only the train split is ever loaded here, and loading refuses to guess when
no split is recorded.

Turning these lists into padded tensors is a separate, thin layer that needs
torch. Keeping the decisions here means they are tested without it.

Defaults that have not been measured are recorded in docs/experiments.md with
the measurement that will settle them.
"""

import random
from collections import defaultdict
from dataclasses import dataclass

from elenchus.corpus.dataset import eligible_rows, load_splits, tokens_key
from elenchus.encoder.normalise import normalise
from elenchus.encoder.vocab import SPECIALS

# The label value cross-entropy skips. -100 is torch's default ignore_index,
# so masked-LM labels built here need no translation later.
IGNORE_INDEX = -100

DEFAULT_MAX_LEN = 1024


@dataclass(frozen=True)
class Example:
    """One function at one optimisation level, normalised."""

    function_id: int
    identity: tuple[str, str, str]
    opt_level: str
    tokens: tuple[str, ...]
    content: str

    @property
    def package(self):
        return self.identity[0]


def load_examples(conn, split="train", rows=None):
    """Return an Example for every eligible function in one split.

    Refuses when the split has no packages recorded. Falling back to every
    package would train on val and test without a single error.
    """
    assignment = load_splits(conn)
    if split not in assignment.values():
        raise ValueError(
            f"no {split!r} split is recorded; run `elenchus dataset --assign` first"
        )

    examples = []
    for row in eligible_rows(conn, rows):
        if assignment.get(row["package"]) != split:
            continue
        tokens = tuple(normalise(row["listing"]))
        examples.append(Example(
            function_id=row["function_id"],
            identity=(row["package"], row["decl_file"], row["name"]),
            opt_level=row["opt_level"],
            tokens=tokens,
            content=tokens_key(tokens),
        ))

    examples.sort(key=lambda e: e.function_id)
    return examples


def epoch_rng(seed, epoch):
    """A random generator determined by (seed, epoch) and nothing else.

    random.Random seeded with a tuple would go through hash(), which varies
    between Python processes. An integer does not.
    """
    return random.Random(seed * 1_000_003 + epoch)


def with_cls(ids, cls_id, max_len):
    """Prefix [CLS] and keep the first max_len - 1 tokens.

    Always from the start, for both views of a pair. A random window would
    compare the middle of one view with the start of the other - a question
    with no right answer.
    """
    if max_len < 2:
        raise ValueError(f"max_len must leave room for a token, got {max_len}")
    return [cls_id] + list(ids[: max_len - 1])


# ---------------------------------------------------------------- contrastive


@dataclass
class PairBatch:
    """One contrastive batch. Index i of every list is the same function."""

    anchor: list[list[int]]
    positive: list[list[int]]
    identities: list[tuple[str, str, str]]
    anchor_levels: list[str]
    positive_levels: list[str]
    anchor_content: list[str]
    positive_content: list[str]

    def __len__(self):
        return len(self.anchor)


def false_negatives(query_content, own_content, key_content):
    """Return a matrix: [i][j] is True where key j must not count against i.

    In a batch, every key other than a query's own positive is treated as
    wrong. That is a lie when key j compiles to exactly the same normalised
    code as query i's positive, or as query i itself: no encoder reading
    normalised text can tell them apart, and asking it to push them apart
    teaches it to separate things by noise. The evaluation already scores
    identical functions as one answer; training has to agree with it.

    The diagonal is never masked - it is the answer.
    """
    n, m = len(query_content), len(key_content)
    return [
        [
            i != j and key_content[j] in (query_content[i], own_content[i])
            for j in range(m)
        ]
        for i in range(n)
    ]


class PairSampler:
    """Draws batches of positive pairs with hard negatives from one package.

    Only functions present at two or more levels can form a pair. Each epoch,
    every such function appears in exactly one batch, with two distinct levels
    chosen at random - so over many epochs all six level pairs are seen, not
    only the -O0/-O3 pair the evaluation measures (docs/experiments.md, E3).

    A batch is filled from a few packages at a time, taking up to per_package
    functions from each in turn (E4). Late in an epoch, when only a few
    packages still have functions left, the turns repeat and one package may
    fill most of a batch - its functions are then negatives for each other,
    which is harder, not wrong. Random batches are mostly easy: a compression routine
    is not hard to tell from a JSON parser. Two functions from the same
    library share an author, a style and helpers, which is the distinction the
    agent will actually face inside one binary.

    A function never appears twice in a batch; it would be its own negative.
    Incomplete final batches are dropped so every step has the same size.
    """

    def __init__(self, examples, vocab, batch_size=64, per_package=8,
                 max_len=DEFAULT_MAX_LEN, seed=0):
        if batch_size < 2:
            raise ValueError("a contrastive batch needs at least two functions")
        if per_package < 1:
            raise ValueError("per_package must be at least 1")

        self.vocab = vocab
        self.batch_size = batch_size
        self.per_package = per_package
        self.max_len = max_len
        self.seed = seed

        # The same source function can have more than one row at one level:
        # a static inline from a header is emitted out of line in each file
        # that uses it. Every row is a legitimate view of the identity.
        views = defaultdict(lambda: defaultdict(list))
        for example in examples:
            views[example.identity][example.opt_level].append(example)

        self.views = {
            identity: {level: rows for level, rows in sorted(levels.items())}
            for identity, levels in sorted(views.items())
            if len(levels) >= 2
        }

        by_package = defaultdict(list)
        for identity in self.views:
            by_package[identity[0]].append(identity)
        self.by_package = dict(sorted(by_package.items()))

    def __len__(self):
        return len(self.views) // self.batch_size

    def _schedule(self, rng):
        """Decide which identities share a batch, for one epoch."""
        queues = {}
        for package, identities in self.by_package.items():
            order = list(identities)
            rng.shuffle(order)
            queues[package] = order

        remaining = sum(len(q) for q in queues.values())
        batches = []

        while remaining >= self.batch_size:
            live = [package for package, q in queues.items() if q]
            rng.shuffle(live)

            batch = []
            while len(batch) < self.batch_size:
                before = len(batch)
                for package in live:
                    take = min(self.per_package,
                               self.batch_size - len(batch),
                               len(queues[package]))
                    batch.extend(queues[package][:take])
                    del queues[package][:take]
                    if len(batch) == self.batch_size:
                        break
                if len(batch) == before:
                    # Unreachable while remaining is counted correctly; if it
                    # ever is not, fail loudly instead of spinning forever.
                    raise RuntimeError(
                        f"batch stuck at {len(batch)}/{self.batch_size} "
                        f"with {remaining} functions reported remaining"
                    )

            remaining -= len(batch)
            batches.append(batch)

        return batches

    def batches(self, epoch):
        """Yield the PairBatch sequence for one epoch."""
        rng = epoch_rng(self.seed, epoch)
        cls_id = self.vocab.cls_id

        for identities in self._schedule(rng):
            batch = PairBatch([], [], [], [], [], [], [])
            for identity in identities:
                levels = self.views[identity]
                first, second = rng.sample(sorted(levels), 2)
                anchor = rng.choice(levels[first])
                positive = rng.choice(levels[second])

                batch.anchor.append(
                    with_cls(self.vocab.encode(anchor.tokens), cls_id, self.max_len))
                batch.positive.append(
                    with_cls(self.vocab.encode(positive.tokens), cls_id, self.max_len))
                batch.identities.append(identity)
                batch.anchor_levels.append(first)
                batch.positive_levels.append(second)
                batch.anchor_content.append(anchor.content)
                batch.positive_content.append(positive.content)
            yield batch


# ---------------------------------------------------------------- masked LM


def mask_tokens(ids, vocab, rng, rate=0.15):
    """Hide a share of tokens for masked language modelling.

    Returns (inputs, labels). labels holds the original id at every chosen
    position and IGNORE_INDEX everywhere else, so the loss is computed only
    where something was hidden.

    Of the chosen positions, 80% become [MASK], 10% a random ordinary token,
    10% stay as they were. If the model only ever had to think where it saw
    [MASK], that skill would be idle in real use, where [MASK] never appears;
    the other 20% make every token worth questioning.

    The number chosen is exact - round(rate * n), at least one - rather than a
    coin flip per token, so no sequence contributes nothing to the loss.

    [PAD], [CLS] and [MASK] are never chosen: they are structure, and
    predicting them teaches nothing. [UNK] and IMPORT:<rare> are chosen like
    any token - they stand for real code, and "a rarely used import is called
    here" is worth predicting. No special is ever used as a random
    replacement.

    Masking single tokens is the default, not a settled choice: whole
    instructions may teach more (docs/experiments.md, E1).
    """
    structural = {vocab.pad_id, vocab.cls_id, vocab.mask_id}
    first_ordinary = len(SPECIALS)
    if len(vocab) <= first_ordinary:
        raise ValueError("vocabulary has no ordinary tokens to substitute")

    candidates = [i for i, token in enumerate(ids) if token not in structural]
    inputs = list(ids)
    labels = [IGNORE_INDEX] * len(ids)
    if not candidates:
        return inputs, labels

    count = min(len(candidates), max(1, round(rate * len(candidates))))
    for position in rng.sample(candidates, count):
        labels[position] = ids[position]
        roll = rng.random()
        if roll < 0.8:
            inputs[position] = vocab.mask_id
        elif roll < 0.9:
            inputs[position] = rng.randrange(first_ordinary, len(vocab))
        # else: left unchanged, but still predicted

    return inputs, labels


@dataclass
class MaskedBatch:
    inputs: list[list[int]]
    labels: list[list[int]]

    def __len__(self):
        return len(self.inputs)


class MaskedSampler:
    """Draws masked-LM batches from single functions.

    Every example is usable, including functions present at one level only -
    no pair is needed to learn grammar.

    Examples whose normalised code is identical are kept once. -O2 and -O3
    often compile a function to the same thing, and a sequence seen twice as
    often is weighted twice as heavily for no reason.

    Unlike contrastive batches, a long function is not cut from the start: a
    random window is taken, so across epochs the model reads all of it. There
    is no second view here to keep aligned with.
    """

    def __init__(self, examples, vocab, batch_size=64, max_len=DEFAULT_MAX_LEN,
                 rate=0.15, seed=0):
        if max_len < 2:
            raise ValueError(f"max_len must leave room for a token, got {max_len}")

        self.vocab = vocab
        self.batch_size = batch_size
        self.max_len = max_len
        self.rate = rate
        self.seed = seed

        unique = {}
        for example in sorted(examples, key=lambda e: e.function_id):
            unique.setdefault(example.content, example)
        self.examples = list(unique.values())

    def __len__(self):
        return len(self.examples) // self.batch_size

    def batches(self, epoch):
        rng = epoch_rng(self.seed, epoch)
        order = list(self.examples)
        rng.shuffle(order)

        body = self.max_len - 1
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            batch = MaskedBatch([], [])
            for example in order[start:start + self.batch_size]:
                ids = self.vocab.encode(example.tokens)
                if len(ids) > body:
                    offset = rng.randrange(len(ids) - body + 1)
                    ids = ids[offset:offset + body]
                inputs, labels = mask_tokens(
                    [self.vocab.cls_id] + ids, self.vocab, rng, self.rate)
                batch.inputs.append(inputs)
                batch.labels.append(labels)
            yield batch
