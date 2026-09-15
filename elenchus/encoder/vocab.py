"""The vocabulary: the contract between normalised tokens and model weights.

A network does arithmetic on numbers, not words, so every token becomes an
integer, and that integer selects a row of the model's embedding table. Once a
model is trained, row 6 *means* whatever token held id 6 while it learned. A
vocabulary rebuilt in a different order does not make the model fail loudly -
it makes it read one token as another and produce confident nonsense. Nothing
turns red. So the vocabulary is treated as an artifact, not a computation:

  deterministic  the same corpus yields the same ids, byte for byte
  persisted      written to disk and kept with every checkpoint that used it
  identified     it records what it was built from and with which threshold

Built from the train split only. A token that appears only in val or test must
look unknown to the model, because that is what an unseen binary will look
like to it in real use; letting it in would quietly leak the exam.

Frequency is counted in functions, not occurrences. One function calling
memcpy fifty times, compiled at four optimisation levels, is still one
function that uses memcpy. Counting raw occurrences would let a single long
function push any token it likes past the threshold - and a token learned from
one function is a token memorised, not understood.

Tokens below the threshold do not vanish. A rare import becomes
IMPORT:<rare>, which keeps the fact that an imported function is called while
dropping a name too rare to learn. Anything else rare becomes [UNK].

Pure Python on purpose: no torch import, so the suite and CI run without the
ml extra installed.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from elenchus.corpus.dataset import eligible_rows, load_splits
from elenchus.encoder.normalise import normalise

PAD = "[PAD]"
CLS = "[CLS]"
MASK = "[MASK]"
UNK = "[UNK]"
RARE_IMPORT = "IMPORT:<rare>"

# Fixed at the front, in this order, so their ids never depend on the corpus:
# padding is always 0, and code that builds masks may rely on it.
SPECIALS = (PAD, CLS, MASK, UNK, RARE_IMPORT)

IMPORT_PREFIX = "IMPORT:"

# Chosen from the week 6 measurement on the 28-package corpus: 135 distinct
# import names in train, 79 of them seen at least five times. Revisited once
# the vocabulary reports how much of val falls through to the fallbacks.
DEFAULT_MIN_FUNCTIONS = 5

FORMAT_VERSION = 1


def identity(row):
    """Return the key that makes two rows the same source function.

    The same triple the evaluation task uses: package, source file, name.
    The optimisation level is deliberately not part of it.
    """
    return (row["package"], row["decl_file"], row["name"])


def document_frequency(functions):
    """Count, for every token, the number of distinct functions containing it.

    functions is an iterable of (identity, tokens) pairs. The same identity
    may appear several times - once per optimisation level - and still counts
    once per token.
    """
    seen = {}
    for key, tokens in functions:
        seen.setdefault(key, set()).update(tokens)

    counts = Counter()
    for tokens in seen.values():
        counts.update(tokens)

    return counts


def build_vocab(functions, min_functions=DEFAULT_MIN_FUNCTIONS, source=None):
    """Build a vocabulary from (identity, tokens) pairs.

    Specials come first with fixed ids. Every other token seen in at least
    min_functions distinct functions follows, most frequent first, ties broken
    by the token itself so the order never depends on input order.

    source is free-form provenance (corpus fingerprint, code version) stored
    with the vocabulary; it does not affect the ids.
    """
    if min_functions < 1:
        raise ValueError(f"min_functions must be at least 1, got {min_functions}")

    counts = document_frequency(functions)
    for special in SPECIALS:
        counts.pop(special, None)

    kept = sorted(
        (token for token, count in counts.items() if count >= min_functions),
        key=lambda token: (-counts[token], token),
    )

    return Vocab(
        tokens=SPECIALS + tuple(kept),
        min_functions=min_functions,
        source=dict(source or {}),
    )


@dataclass(frozen=True)
class Vocab:
    """An immutable token <-> id mapping with its provenance."""

    tokens: tuple[str, ...]
    min_functions: int
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if tuple(self.tokens[: len(SPECIALS)]) != SPECIALS:
            raise ValueError(
                f"vocabulary must start with {SPECIALS}, "
                f"got {tuple(self.tokens[: len(SPECIALS)])}"
            )
        if len(set(self.tokens)) != len(self.tokens):
            duplicated = sorted(t for t, n in Counter(self.tokens).items() if n > 1)
            raise ValueError(f"vocabulary repeats tokens: {duplicated[:5]}")

        # A frozen dataclass cannot assign normally; the index is derived
        # state, rebuilt from tokens, so bypassing the freeze here is safe.
        object.__setattr__(
            self, "_index", {token: i for i, token in enumerate(self.tokens)}
        )

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_id(self):
        return self._index[PAD]

    @property
    def cls_id(self):
        return self._index[CLS]

    @property
    def mask_id(self):
        return self._index[MASK]

    @property
    def unk_id(self):
        return self._index[UNK]

    @property
    def rare_import_id(self):
        return self._index[RARE_IMPORT]

    def id_of(self, token):
        """Return the id for token, falling back to IMPORT:<rare> or [UNK]."""
        found = self._index.get(token)
        if found is not None:
            return found
        if token.startswith(IMPORT_PREFIX):
            return self.rare_import_id
        return self.unk_id

    def encode(self, tokens):
        """Map a token sequence to ids. Never fails: unknowns fall back."""
        return [self.id_of(token) for token in tokens]

    def decode(self, ids):
        """Map ids back to tokens. An id outside the vocabulary is an error.

        Unlike encoding, there is no sensible fallback: an out-of-range id
        means the ids and the vocabulary disagree, which is exactly the silent
        mismatch this module exists to prevent.
        """
        tokens = []
        for i in ids:
            if not 0 <= i < len(self.tokens):
                raise IndexError(f"id {i} outside vocabulary of {len(self.tokens)}")
            tokens.append(self.tokens[i])
        return tokens

    def to_json(self):
        return {
            "format": FORMAT_VERSION,
            "min_functions": self.min_functions,
            "source": self.source,
            "tokens": list(self.tokens),
        }

    @classmethod
    def from_json(cls, data):
        version = data.get("format")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported vocabulary format {version!r}, "
                f"expected {FORMAT_VERSION}"
            )
        return cls(
            tokens=tuple(data["tokens"]),
            min_functions=data["min_functions"],
            source=dict(data.get("source") or {}),
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text()))


def train_functions(conn, rows=None):
    """Return (identity, tokens) for every eligible function in the train split.

    Refuses to run before a split is recorded. Falling back to the whole
    corpus would build the vocabulary from val and test as well - a leak
    that produces no error and slightly better numbers.
    """
    assignment = load_splits(conn)
    if "train" not in assignment.values():
        raise ValueError(
            "no train split is recorded; run `elenchus dataset --assign` first"
        )

    return [
        (identity(row), normalise(row["listing"]))
        for row in eligible_rows(conn, rows)
        if assignment.get(row["package"]) == "train"
    ]


def coverage(vocab, token_lists):
    """Report how much of a token stream falls through to the fallbacks.

    Meant for val: a high rate there means the model is being examined in a
    language it was never taught, and a poor score may be the vocabulary's
    fault rather than the model's.
    """
    total = unknown = rare = 0
    functions = touched = 0

    for tokens in token_lists:
        functions += 1
        hit = False
        for i in vocab.encode(tokens):
            total += 1
            if i == vocab.unk_id:
                unknown += 1
                hit = True
            elif i == vocab.rare_import_id:
                rare += 1
                hit = True
        touched += hit

    return {
        "tokens": total,
        "unk_rate": unknown / total if total else 0.0,
        "rare_import_rate": rare / total if total else 0.0,
        "functions": functions,
        "functions_with_fallback": touched / functions if functions else 0.0,
    }
