"""One-off: the val MLM loss of a model that knows only token frequencies.

A masked-LM model that has not learned to read context can still learn how
often each token appears. Its val loss then sits at the cross-entropy of the
train token frequencies on val - the "unigram floor". A loss that stays near
this number has learned no context yet; one clearly below it has.

Uses MaskedSampler's own de-duplication, so it counts what training counts.
Approximate: training reads random windows of long functions and scores the
10% of masked positions left unchanged, which a frequency model ignores.

Run from the repo root:  python experiments/unigram_floor.py
"""

import math
import os
from collections import Counter

from elenchus.db import connect
from elenchus.encoder.data import MaskedSampler, load_examples
from elenchus.encoder.vocab import Vocab


def token_counts(conn, vocab, split):
    sampler = MaskedSampler(load_examples(conn, split), vocab)
    counts = Counter()
    for example in sampler.examples:
        counts.update(vocab.encode(example.tokens))
    return counts


def unigram_floor(conn, vocab):
    train = token_counts(conn, vocab, "train")
    val = token_counts(conn, vocab, "val")
    total = sum(train.values())
    size = len(vocab)
    # Add-one smoothing: a val token never seen in train must not cost infinity.
    def p(i):
        return (train[i] + 1) / (total + size)
    n = sum(val.values())
    cross = -sum(c * math.log(p(i)) for i, c in val.items()) / n
    return {"floor": cross, "uniform": math.log(size), "val_tokens": n,
            "train_tokens": total}


def main():
    conn = connect(os.environ["ELENCHUS_DB"])
    vocab = Vocab.load("models/vocab.json")
    r = unigram_floor(conn, vocab)
    print(f"uniform guess ln(V) : {r['uniform']:.3f}")
    print(f"unigram floor (val) : {r['floor']:.3f}")
    print(f"tokens              : train {r['train_tokens']:,}  val {r['val_tokens']:,}")


if __name__ == "__main__":
    main()
