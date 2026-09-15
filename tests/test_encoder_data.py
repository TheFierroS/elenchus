"""Tests for what the encoder is shown in training.

Every failure guarded against here is silent. A function paired with itself,
two indistinguishable functions labelled different, two views cut so they no
longer correspond, a val package slipping into train: training runs, the loss
falls, and the model learns something other than what was meant. So each
test builds the situation that would teach the wrong thing and checks that
it cannot reach a batch.
"""

from collections import Counter

import pytest

from elenchus.corpus.dataset import content_key, store_splits, tokens_key
from elenchus.encoder.data import (
    IGNORE_INDEX,
    Example,
    MaskedSampler,
    PairSampler,
    epoch_rng,
    false_negatives,
    load_examples,
    mask_tokens,
    with_cls,
)
from elenchus.encoder.vocab import SPECIALS, UNK, build_vocab

LEVELS = ("O0", "O1", "O2", "O3")


def example(fid, package, name, level, tokens, content=None):
    tokens = tuple(tokens)
    return Example(
        function_id=fid,
        identity=(package, f"{package}/src.c", name),
        opt_level=level,
        tokens=tokens,
        content=content or tokens_key(tokens),
    )


def corpus(packages=4, per_package=20, levels=LEVELS):
    """Distinct functions, each at every level with level-specific code."""
    examples, fid = [], 0
    for p in range(packages):
        for f in range(per_package):
            for level in levels:
                fid += 1
                examples.append(example(
                    fid, f"pkg{p}", f"fn{f}", level,
                    ["push", f"body_{p}_{f}", f"at_{level}", "ret"],
                ))
    return examples


def vocab_for(examples):
    return build_vocab(
        [(e.identity, list(e.tokens)) for e in examples], min_functions=1)


def pair_sampler(examples=None, **kwargs):
    examples = corpus() if examples is None else examples
    defaults = {"batch_size": 16, "per_package": 8, "seed": 7}
    return PairSampler(examples, vocab_for(examples), **(defaults | kwargs))


# ---------------------------------------------------------------- loading


def listing(mnemonics):
    return "\n".join(f"{i:x}\t{m}\tRAX RBX\t" for i, m in enumerate(mnemonics))


def add_function(conn, package, name, mnemonics, opt="O0"):
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (f"/tmp/{package}_{opt}.dll", f"{package}{opt}"),
        )
        binary_id = conn.execute(
            "SELECT id FROM binaries WHERE sha256 = ?", (f"{package}{opt}",)
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped) "
            "VALUES (?, ?, 'gcc', ?, 1)",
            (binary_id, package, opt),
        )
        address = 0x1000 + 0x100 * conn.execute(
            "SELECT COUNT(*) FROM functions WHERE binary_id = ?", (binary_id,)
        ).fetchone()[0]
        fid = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, ?)", (binary_id, address, f"FUN_{address:x}"),
        ).lastrowid
        conn.execute(
            "INSERT INTO ground_truth (binary_id, function_id, address, name, "
            "return_type, param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, fid, address, name, f"src/{package}/{name}.c"),
        )
        conn.execute(
            "INSERT INTO function_code (function_id, binary_id, n_instructions, "
            "code_size, byte_hash, listing) VALUES (?, ?, ?, 32, ?, ?)",
            (fid, binary_id, len(mnemonics), f"h{fid}", listing(mnemonics)),
        )


@pytest.fixture
def split_db(db):
    for package, op in (("trainpkg", "xor"), ("valpkg", "imul"), ("testpkg", "shl")):
        for i in range(2):
            add_function(db, package, f"{package}_{i}", [op, "push"] * 5 + [f"x{i}"])
    store_splits(db, {"trainpkg": "train", "valpkg": "val", "testpkg": "test"})
    return db


def test_loading_takes_only_the_requested_split(split_db):
    examples = load_examples(split_db, "train")
    assert {e.package for e in examples} == {"trainpkg"}


def test_loading_refuses_when_no_split_is_recorded(db):
    add_function(db, "trainpkg", "f", ["push"] * 10)
    with pytest.raises(ValueError, match="dataset --assign"):
        load_examples(db, "train")


def test_example_content_is_the_datasets_own_content_key(split_db):
    """The false-negative mask and the dataset dedup must mean the same thing."""
    row = split_db.execute(
        "SELECT fc.function_id, fc.listing FROM function_code fc "
        "JOIN corpus_binaries cb ON cb.binary_id = fc.binary_id "
        "WHERE cb.package = 'trainpkg' ORDER BY fc.function_id LIMIT 1"
    ).fetchone()
    loaded = next(e for e in load_examples(split_db) if e.function_id == row[0])
    assert loaded.content == content_key(row[1])


# ---------------------------------------------------------------- truncation


def test_cls_comes_first_and_the_start_is_kept():
    assert with_cls([10, 11, 12, 13, 14], cls_id=1, max_len=4) == [1, 10, 11, 12]


def test_short_sequences_are_not_padded_here():
    assert with_cls([10, 11], cls_id=1, max_len=8) == [1, 10, 11]


def test_max_len_must_leave_room_for_a_token():
    with pytest.raises(ValueError):
        with_cls([10], cls_id=1, max_len=1)


def test_both_views_of_a_pair_are_cut_from_the_start():
    """A random window on one side would compare unrelated stretches of code."""
    long = [
        example(i * 10 + j, "pkg", f"fn{i}", level,
                ["push"] + [f"tok{k}" for k in range(50)] + [f"at_{level}"])
        for i in range(4) for j, level in enumerate(("O0", "O3"))
    ]
    sampler = pair_sampler(long, batch_size=4, per_package=4, max_len=6)
    vocab = sampler.vocab
    batch = next(sampler.batches(0))

    expected = [vocab.cls_id] + vocab.encode(["push", "tok0", "tok1", "tok2", "tok3"])
    assert all(view == expected for view in batch.anchor + batch.positive)


# ---------------------------------------------------------------- pairs


def test_a_function_at_one_level_never_forms_a_pair():
    examples = corpus(packages=1, per_package=4)
    examples.append(example(999, "pkg0", "inlined_away", "O0", ["push", "solo"]))
    sampler = pair_sampler(examples, batch_size=2, per_package=2)

    assert ("pkg0", "pkg0/src.c", "inlined_away") not in sampler.views


def test_both_views_are_one_function_at_two_different_levels():
    sampler = pair_sampler()
    by_id = {}
    for e in corpus():
        by_id[(e.identity, e.opt_level)] = e.content

    for batch in sampler.batches(0):
        for i, identity in enumerate(batch.identities):
            assert batch.anchor_levels[i] != batch.positive_levels[i]
            anchor_key = (identity, batch.anchor_levels[i])
            positive_key = (identity, batch.positive_levels[i])
            assert batch.anchor_content[i] == by_id[anchor_key]
            assert batch.positive_content[i] == by_id[positive_key]


def test_no_function_appears_twice_in_an_epoch():
    """Twice in one batch, a function is its own negative."""
    sampler = pair_sampler()
    seen = [identity for batch in sampler.batches(0) for identity in batch.identities]
    assert len(seen) == len(set(seen))


def test_every_pairable_function_is_used_when_batches_divide_evenly():
    sampler = pair_sampler(batch_size=16)  # 80 functions, 5 batches
    seen = {identity for batch in sampler.batches(0) for identity in batch.identities}
    assert seen == set(sampler.views)


def test_every_batch_has_the_same_size_and_the_remainder_is_dropped():
    sampler = pair_sampler(batch_size=24)  # 80 functions
    sizes = [len(batch) for batch in sampler.batches(0)]
    assert sizes == [24, 24, 24]
    assert len(sampler) == 3


def test_all_six_level_pairs_are_seen_over_epochs():
    sampler = pair_sampler()
    pairs = {
        frozenset((a, b))
        for epoch in range(5)
        for batch in sampler.batches(epoch)
        for a, b in zip(batch.anchor_levels, batch.positive_levels)
    }
    assert len(pairs) == 6


def test_every_row_at_a_level_is_eventually_used():
    """A header inline emitted in two files is two rows of one identity."""
    examples = corpus(packages=1, per_package=2, levels=("O3",))
    examples += [
        example(500, "pkg0", "fn0", "O0", ["push", "copy_in_a"]),
        example(501, "pkg0", "fn0", "O0", ["push", "copy_in_b"]),
        example(502, "pkg0", "fn1", "O0", ["push", "other"]),
    ]
    sampler = pair_sampler(examples, batch_size=2, per_package=2)
    wanted = {e.content for e in examples if e.function_id in (500, 501)}

    as_anchor, as_positive = set(), set()
    for epoch in range(40):
        for batch in sampler.batches(epoch):
            as_anchor.update(batch.anchor_content)
            as_positive.update(batch.positive_content)

    assert wanted <= as_anchor, "a duplicate row is never a query"
    assert wanted <= as_positive, "a duplicate row is never a target"


# ---------------------------------------------------------------- hard negatives


def test_an_early_batch_takes_at_most_per_package_from_each():
    sampler = pair_sampler(corpus(packages=6, per_package=20),
                           batch_size=16, per_package=4)
    first = next(sampler.batches(0))
    counts = Counter(identity[0] for identity in first.identities)

    assert max(counts.values()) <= 4
    assert len(counts) == 4


def test_a_batch_draws_on_few_packages_not_a_random_mix():
    """Sixteen functions drawn at random from six packages would span nearly
    all of them. Built for hard negatives, a batch spans exactly
    batch_size / per_package - while every package still has that many left.
    """
    sampler = pair_sampler(corpus(packages=6, per_package=20),
                           batch_size=16, per_package=8)
    first = next(sampler.batches(0))
    assert len({identity[0] for identity in first.identities}) == 2


def test_which_packages_meet_in_a_batch_changes_between_epochs():
    """Fixed pairings would teach the same few library-vs-library contrasts."""
    sampler = pair_sampler(corpus(packages=8, per_package=20),
                           batch_size=16, per_package=8)
    pairings = {
        frozenset(identity[0] for identity in next(sampler.batches(epoch)).identities)
        for epoch in range(10)
    }
    assert len(pairings) > 3


def test_late_batches_still_fill_when_few_packages_remain():
    examples = corpus(packages=1, per_package=40)
    sampler = pair_sampler(examples, batch_size=16, per_package=4)
    batches = list(sampler.batches(0))

    assert [len(b) for b in batches] == [16, 16]
    for batch in batches:
        assert len(set(batch.identities)) == 16


# ---------------------------------------------------------------- reproducibility


def signature(batches):
    return [
        (tuple(b.identities), tuple(b.anchor_levels), tuple(b.positive_levels))
        for b in batches
    ]


def test_the_same_seed_and_epoch_give_the_same_batches():
    assert signature(pair_sampler(seed=3).batches(2)) == \
        signature(pair_sampler(seed=3).batches(2))


def test_a_new_epoch_gives_new_batches():
    sampler = pair_sampler()
    assert signature(sampler.batches(0)) != signature(sampler.batches(1))


def test_a_new_seed_gives_new_batches():
    assert signature(pair_sampler(seed=1).batches(0)) != \
        signature(pair_sampler(seed=2).batches(0))


def test_epoch_rng_does_not_depend_on_the_process():
    """Seeding with a tuple would go through hash() and vary between runs."""
    assert epoch_rng(5, 3).random() == epoch_rng(5, 3).random()
    assert epoch_rng(5, 3).random() != epoch_rng(3, 5).random()


def test_refuses_a_batch_that_cannot_hold_a_negative():
    with pytest.raises(ValueError):
        pair_sampler(batch_size=1)


# ---------------------------------------------------------------- false negatives


def test_the_answer_itself_is_never_masked():
    mask = false_negatives(["same", "x"], ["same", "y"], ["same", "y"])
    assert mask[0][0] is False and mask[1][1] is False


def test_a_key_identical_to_the_true_positive_is_not_a_negative():
    """Two functions compiled to one normalised form cannot be told apart."""
    mask = false_negatives(
        query_content=["q0", "q1", "q2"],
        own_content=["p0", "p1", "p2"],
        key_content=["p0", "p0", "p2"],
    )
    assert mask[0] == [False, True, False]


def test_a_key_identical_to_the_query_is_not_a_negative():
    mask = false_negatives(["q0", "q1"], ["p0", "p1"], ["p0", "q0"])
    assert mask[0] == [False, True]


def test_distinct_code_stays_a_negative():
    mask = false_negatives(["q0", "q1"], ["p0", "p1"], ["p0", "p1"])
    assert mask == [[False, False], [False, False]]


def test_the_mask_covers_keys_beyond_the_batch():
    """The memory queue is longer than the batch; the mask must be too."""
    mask = false_negatives(["q0"], ["p0"], ["p0", "a", "p0", "q0"])
    assert mask == [[False, False, True, True]]


# ---------------------------------------------------------------- masking


@pytest.fixture
def vocab():
    return build_vocab([(("p", "f", f"n{i}"), [f"t{k}" for k in range(20)])
                        for i in range(2)], min_functions=1)


def sequence(vocab, n):
    return [vocab.cls_id] + [len(SPECIALS) + (k % (len(vocab) - len(SPECIALS)))
                             for k in range(n)]


def test_structure_is_never_hidden(vocab):
    structure = [vocab.cls_id] + [vocab.mask_id] * 3 + [vocab.pad_id] * 3
    ids = structure + sequence(vocab, 40)[1:]
    for seed in range(50):
        _inputs, labels = mask_tokens(ids, vocab, epoch_rng(seed, 0))
        assert all(labels[i] == IGNORE_INDEX for i in range(7))


def test_unknown_and_rare_imports_can_be_hidden(vocab):
    """They stand for real code; only structure is exempt."""
    ids = [vocab.cls_id] + [vocab.unk_id] * 10 + [vocab.rare_import_id] * 10
    chosen = set()
    for seed in range(50):
        _inputs, labels = mask_tokens(ids, vocab, epoch_rng(seed, 0))
        chosen.update(label for label in labels if label != IGNORE_INDEX)
    assert chosen == {vocab.unk_id, vocab.rare_import_id}


def test_the_number_hidden_is_exact(vocab):
    ids = sequence(vocab, 100)
    _inputs, labels = mask_tokens(ids, vocab, epoch_rng(0, 0), rate=0.15)
    assert sum(label != IGNORE_INDEX for label in labels) == 15


def test_a_short_sequence_still_hides_one(vocab):
    ids = sequence(vocab, 2)
    _inputs, labels = mask_tokens(ids, vocab, epoch_rng(0, 0), rate=0.15)
    assert sum(label != IGNORE_INDEX for label in labels) == 1


def test_labels_keep_the_original_and_untouched_positions_are_unchanged(vocab):
    ids = sequence(vocab, 60)
    inputs, labels = mask_tokens(ids, vocab, epoch_rng(1, 0))
    for position, (original, label) in enumerate(zip(ids, labels)):
        if label == IGNORE_INDEX:
            assert inputs[position] == original
        else:
            assert label == original


def test_the_80_10_10_split(vocab):
    ids = sequence(vocab, 200)
    outcome = Counter()
    for seed in range(300):
        inputs, labels = mask_tokens(ids, vocab, epoch_rng(seed, 0))
        for position, label in enumerate(labels):
            if label == IGNORE_INDEX:
                continue
            if inputs[position] == vocab.mask_id:
                outcome["mask"] += 1
            elif inputs[position] == label:
                outcome["kept"] += 1
            else:
                outcome["random"] += 1

    total = sum(outcome.values())
    assert outcome["mask"] / total == pytest.approx(0.80, abs=0.02)
    # A random replacement can land on the original token, which counts as
    # kept; with this vocabulary that shifts about 0.4% between the two.
    assert outcome["random"] / total == pytest.approx(0.10, abs=0.02)
    assert outcome["kept"] / total == pytest.approx(0.10, abs=0.02)


def test_a_random_replacement_is_never_a_special(vocab):
    ids = sequence(vocab, 200)
    for seed in range(100):
        inputs, labels = mask_tokens(ids, vocab, epoch_rng(seed, 0))
        for position, label in enumerate(labels):
            if label != IGNORE_INDEX and inputs[position] != vocab.mask_id:
                assert inputs[position] >= len(SPECIALS)


def test_nothing_to_hide_is_not_an_error(vocab):
    inputs, labels = mask_tokens([vocab.cls_id], vocab, epoch_rng(0, 0))
    assert inputs == [vocab.cls_id] and labels == [IGNORE_INDEX]


def test_a_vocabulary_of_specials_only_is_refused():
    empty = build_vocab([], min_functions=1)
    with pytest.raises(ValueError, match="no ordinary tokens"):
        mask_tokens([empty.cls_id, empty.unk_id], empty, epoch_rng(0, 0))


# ---------------------------------------------------------------- masked sampler


def test_identical_code_is_kept_once():
    examples = [
        example(2, "pkg", "f", "O2", ["push", "same"]),
        example(1, "pkg", "f", "O3", ["push", "same"]),
        example(3, "pkg", "g", "O0", ["push", "other"]),
    ]
    sampler = MaskedSampler(examples, vocab_for(examples), batch_size=1)
    assert [e.function_id for e in sampler.examples] == [1, 3]


def test_functions_at_one_level_are_used_for_masked_lm():
    examples = [example(1, "pkg", "solo", "O0", ["push", "only_here"])]
    sampler = MaskedSampler(examples, vocab_for(examples), batch_size=1)
    assert len(list(sampler.batches(0))) == 1


def test_a_long_function_is_read_through_a_moving_window():
    tokens = ["push"] + [f"t{k}" for k in range(40)]
    examples = [example(1, "pkg", "long", "O0", tokens)]
    vocab = vocab_for(examples)
    sampler = MaskedSampler(examples, vocab, batch_size=1, max_len=9, rate=0.0001)
    ids = vocab.encode(tokens)

    starts = set()
    for epoch in range(40):
        batch = next(sampler.batches(epoch))
        inputs, labels = batch.inputs[0], batch.labels[0]
        original = [label if label != IGNORE_INDEX else token
                    for token, label in zip(inputs, labels)]

        assert original[0] == vocab.cls_id
        assert len(original) == 9
        window = original[1:]
        offset = ids.index(window[0])
        assert ids[offset:offset + 8] == window, "the window must be contiguous"
        starts.add(offset)

    assert len(starts) > 5


def test_masked_batches_are_reproducible():
    examples = corpus(packages=2, per_package=10)
    vocab = vocab_for(examples)
    one = MaskedSampler(examples, vocab, batch_size=8, seed=4)
    two = MaskedSampler(examples, vocab, batch_size=8, seed=4)

    assert [b.inputs for b in one.batches(1)] == [b.inputs for b in two.batches(1)]
    assert [b.inputs for b in one.batches(1)] != [b.inputs for b in one.batches(2)]


def test_unknown_tokens_in_a_sequence_do_not_break_encoding():
    examples = [example(1, "pkg", "f", "O0", ["push", "seen"])]
    vocab = vocab_for(examples)
    stranger = [example(2, "pkg", "g", "O0", ["push", "never_seen"])]
    sampler = MaskedSampler(stranger, vocab, batch_size=1, rate=0.0001)

    batch = next(sampler.batches(0))
    restored = [label if label != IGNORE_INDEX else token
                for token, label in zip(batch.inputs[0], batch.labels[0])]
    assert vocab.decode(restored) == ["[CLS]", "push", UNK]
