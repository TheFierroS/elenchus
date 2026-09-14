"""Four ways to match functions without a model, to set the bar.

An encoder that scores 60% is neither good nor bad until something else has
tried. These four try, in increasing order of how much they actually look at
the code, and the strongest of them is the number week 6 has to beat.

Random exists to catch broken plumbing. If queries and pool were ever
shuffled apart, every other baseline would quietly collapse to random too,
and only having measured random makes that visible.

Structural looks at size alone - instructions, blocks, calls. Close to
nothing, but not nothing: large functions stay large.

Import overlap looks at which Windows functions get called. On this platform
that is a strong signal and an unfair-looking one, because import names pass
through the compiler untouched: -O0 and -O3 call the same memcpy. Any method
that ignores them is throwing away the one part of a stripped binary that
was never stripped.

BM25 treats the opcode sequence as text and retrieves it the way a search
engine retrieves documents. It is the serious competitor. If a transformer
trained from scratch cannot beat term-frequency statistics over three-token
opcode windows, it has not earned its place.
"""

import math
import random
from collections import Counter, defaultdict


class RandomBaseline:
    """Score every candidate at random, reproducibly."""

    name = "random"

    def __init__(self, seed=1):
        self.seed = seed
        self.pool = []

    def prepare(self, pool):
        self.pool = pool

    def scores(self, query):
        # Seeded per query, so the whole run repeats exactly while different
        # queries still get different orderings. A string seed rather than a
        # tuple: Python only accepts a few seed types, and a tuple is not
        # one of them.
        rng = random.Random(f"{self.seed}:{query.function_id}")
        return [rng.random() for _ in self.pool]


class StructuralBaseline:
    """Match on size: instruction count, block count, call count.

    The three are put on a log scale first, because the difference between
    10 and 20 instructions means far more than between 1000 and 1010, then
    standardised against the pool so that one feature with a wide range
    cannot drown the others.
    """

    name = "structural"

    def __init__(self):
        self.features = []
        self.centre = ()
        self.spread = ()

    @staticmethod
    def _raw(sample):
        return (
            math.log1p(sample.n_instructions),
            math.log1p(sample.n_blocks),
            math.log1p(sample.n_calls),
        )

    def prepare(self, pool):
        raw = [self._raw(sample) for sample in pool]
        if not raw:
            self.features, self.centre, self.spread = [], (), ()
            return

        columns = list(zip(*raw))
        self.centre = tuple(sum(c) / len(c) for c in columns)
        self.spread = tuple(
            max(
                math.sqrt(sum((v - mean) ** 2 for v in column) / len(column)),
                1e-6,
            )
            for column, mean in zip(columns, self.centre)
        )
        self.features = [self._standardise(values) for values in raw]

    def _standardise(self, values):
        return tuple(
            (value - mean) / spread
            for value, mean, spread in zip(values, self.centre, self.spread)
        )

    def scores(self, query):
        if not self.features:
            return []

        target = self._standardise(self._raw(query))
        return [
            -math.dist(target, candidate)
            for candidate in self.features
        ]


class ImportJaccard:
    """Match on the set of imported functions called.

    Jaccard similarity: how much of the union of the two sets they share.
    A function calling exactly {malloc, memcpy, free} and one calling
    {malloc, memcpy, free, strlen} overlap by three quarters.
    """

    name = "import-jaccard"

    def __init__(self):
        self.pool_imports = []

    def prepare(self, pool):
        self.pool_imports = [sample.imports for sample in pool]

    def scores(self, query):
        wanted = query.imports
        if not wanted:
            # No imports is not evidence of similarity to everything that
            # also has none, so it scores zero rather than one.
            return [0.0] * len(self.pool_imports)

        out = []
        for candidate in self.pool_imports:
            if not candidate:
                out.append(0.0)
                continue
            shared = len(wanted & candidate)
            out.append(shared / len(wanted | candidate))
        return out


class BM25Mnemonics:
    """Retrieve on opcode n-grams with BM25, the classic text ranking.

    Each function becomes a bag of overlapping opcode windows - push/mov/sub,
    mov/sub/cmp - and is scored the way a search engine scores a document
    against a query. Two ideas carry it. A window that appears in few
    functions says more than one appearing in all of them (that is the idf
    term), and a window appearing twenty times in one function does not say
    twenty times as much as appearing once (that is the saturation k1 gives).
    Length normalisation stops long functions from winning by sheer volume.

    Operands are deliberately ignored. The opcode sequence alone survives
    optimisation better than anything else that can be read off the listing
    without a model.
    """

    name = "bm25-mnemonic"

    def __init__(self, n=3, k1=1.5, b=0.75):
        self.n = n
        self.k1 = k1
        self.b = b
        self.postings = defaultdict(list)
        self.lengths = []
        self.average_length = 1.0
        self.idf = {}
        self.size = 0

    def _grams(self, sample):
        opcodes = sample.mnemonics
        if len(opcodes) < self.n:
            return [tuple(opcodes)] if opcodes else []
        return [
            tuple(opcodes[i:i + self.n])
            for i in range(len(opcodes) - self.n + 1)
        ]

    def prepare(self, pool):
        self.postings = defaultdict(list)
        self.lengths = []
        self.size = len(pool)

        for index, sample in enumerate(pool):
            counts = Counter(self._grams(sample))
            self.lengths.append(sum(counts.values()) or 1)
            for gram, count in counts.items():
                self.postings[gram].append((index, count))

        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        )

        # Robertson-Sparck Jones idf, the form BM25 is normally used with.
        self.idf = {
            gram: math.log(
                1 + (self.size - len(entries) + 0.5) / (len(entries) + 0.5)
            )
            for gram, entries in self.postings.items()
        }

    def scores(self, query):
        out = [0.0] * self.size

        for gram in set(self._grams(query)):
            entries = self.postings.get(gram)
            if not entries:
                continue

            weight = self.idf[gram]
            for index, frequency in entries:
                norm = 1 - self.b + self.b * self.lengths[index] / self.average_length
                out[index] += weight * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
                )

        return out


def all_baselines(seed=1):
    """Return the four baselines, weakest expected first."""
    return [
        RandomBaseline(seed),
        StructuralBaseline(),
        ImportJaccard(),
        BM25Mnemonics(),
    ]
