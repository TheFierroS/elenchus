"""Tests for the vocabulary: the ids a trained model's weights depend on.

The failures guarded against here are quiet ones. A vocabulary that shifts
one id does not crash anything - it makes a trained model read one token as
another. A vocabulary built from the whole corpus does not crash either - it
leaks val and test into training. So each test builds exactly the situation
that would go wrong and checks that it cannot.
"""

import json
import random

import pytest

from elenchus.corpus.dataset import store_splits
from elenchus.encoder.vocab import (
    CLS,
    MASK,
    PAD,
    RARE_IMPORT,
    SPECIALS,
    UNK,
    Vocab,
    build_vocab,
    coverage,
    document_frequency,
    train_functions,
)


def fn(name, *tokens, package="pkg", decl_file="a.c"):
    """One (identity, tokens) pair."""
    return ((package, decl_file, name), list(tokens))


# ---------------------------------------------------------------- counting


def test_a_token_repeated_inside_one_function_counts_once():
    counts = document_frequency([fn("f", "call", "IMPORT:memcpy", "IMPORT:memcpy")])
    assert counts["IMPORT:memcpy"] == 1


def test_the_same_function_at_four_levels_counts_once():
    """Optimisation levels are views of one function, not four functions."""
    views = [fn("f", "IMPORT:memcpy") for _ in ("O0", "O1", "O2", "O3")]
    assert document_frequency(views)["IMPORT:memcpy"] == 1


def test_same_name_in_different_packages_is_two_functions():
    counts = document_frequency([
        fn("init", "push", package="zlib"),
        fn("init", "push", package="lua"),
    ])
    assert counts["push"] == 2


def test_one_long_function_cannot_push_a_token_past_the_threshold():
    """The reason frequency is counted in functions."""
    greedy = fn("big", *(["IMPORT:obscure"] * 500))
    vocab = build_vocab([greedy], min_functions=5)
    assert "IMPORT:obscure" not in vocab.tokens


# ---------------------------------------------------------------- threshold


def test_threshold_is_inclusive():
    at = [fn(f"f{i}", "at_threshold") for i in range(5)]
    below = [fn(f"g{i}", "below_threshold") for i in range(4)]
    vocab = build_vocab(at + below, min_functions=5)

    assert "at_threshold" in vocab.tokens
    assert "below_threshold" not in vocab.tokens


def test_threshold_below_one_is_refused():
    with pytest.raises(ValueError):
        build_vocab([], min_functions=0)


# ---------------------------------------------------------------- fallbacks


@pytest.fixture
def small():
    functions = [fn(f"f{i}", "mov", "REG64", "IMPORT:memcpy") for i in range(3)]
    return build_vocab(functions, min_functions=2)


def test_an_unseen_import_keeps_its_import_ness(small):
    assert small.id_of("IMPORT:NeverSeenW") == small.rare_import_id


def test_an_unseen_ordinary_token_is_unknown(small):
    assert small.id_of("vfmadd231ps") == small.unk_id


def test_a_known_import_keeps_its_own_id(small):
    assert small.id_of("IMPORT:memcpy") not in (small.rare_import_id, small.unk_id)


def test_a_rare_import_seen_in_training_still_falls_back():
    functions = [fn(f"f{i}", "IMPORT:common") for i in range(3)]
    functions.append(fn("lonely", "IMPORT:rare_one"))
    vocab = build_vocab(functions, min_functions=2)

    assert vocab.id_of("IMPORT:rare_one") == vocab.rare_import_id


def test_encoding_never_fails(small):
    assert len(small.encode(["mov", "???", "IMPORT:x", ""])) == 4


# ---------------------------------------------------------------- stable ids


def test_specials_have_fixed_ids_at_the_front(small):
    assert small.tokens[: len(SPECIALS)] == SPECIALS
    assert small.pad_id == 0
    assert (small.cls_id, small.mask_id, small.unk_id, small.rare_import_id) == (
        SPECIALS.index(CLS), SPECIALS.index(MASK),
        SPECIALS.index(UNK), SPECIALS.index(RARE_IMPORT),
    )


def test_input_order_does_not_change_the_ids():
    """The property a trained checkpoint depends on."""
    functions = [
        fn(f"f{i}", *random.Random(i).sample(["a", "b", "c", "d", "e"], 3))
        for i in range(40)
    ]
    shuffled = functions[:]
    random.Random(99).shuffle(shuffled)

    assert build_vocab(functions, 2).tokens == build_vocab(shuffled, 2).tokens


def test_more_frequent_tokens_come_first_and_ties_break_by_name():
    functions = (
        [fn(f"x{i}", "often") for i in range(4)]
        + [fn(f"y{i}", "zeta", "alpha") for i in range(2)]
    )
    vocab = build_vocab(functions, min_functions=1)
    assert vocab.tokens[len(SPECIALS):] == ("often", "alpha", "zeta")


def test_a_corpus_token_spelled_like_a_special_cannot_move_the_specials():
    functions = [fn(f"f{i}", PAD, UNK, "mov") for i in range(3)]
    vocab = build_vocab(functions, min_functions=1)

    assert vocab.tokens[: len(SPECIALS)] == SPECIALS
    assert vocab.tokens.count(PAD) == 1


# ---------------------------------------------------------------- decoding


def test_decode_inverts_encode_for_known_tokens(small):
    tokens = ["mov", "REG64", "IMPORT:memcpy", PAD, CLS]
    assert small.decode(small.encode(tokens)) == tokens


@pytest.mark.parametrize("bad", [-1, 10_000])
def test_decode_rejects_an_id_outside_the_vocabulary(small, bad):
    with pytest.raises(IndexError):
        small.decode([bad])


# ---------------------------------------------------------------- persistence


def test_save_and_load_round_trip(small, tmp_path):
    small_with_source = build_vocab(
        [fn(f"f{i}", "mov") for i in range(3)], 2,
        source={"fingerprint": "f81cc0254c00e728", "code_version": "abc1234"},
    )
    path = tmp_path / "nested" / "vocab.json"
    small_with_source.save(path)

    loaded = Vocab.load(path)
    assert loaded == small_with_source
    assert loaded.source["fingerprint"] == "f81cc0254c00e728"


def test_equal_vocabularies_save_to_identical_bytes(tmp_path):
    """Provenance assembled in a different order must not change the file.

    Saving one object twice would prove nothing - its keys are already in
    one order. Two equal vocabularies whose source was built differently is
    the case that happens in practice.
    """
    functions = [fn(f"f{i}", "mov") for i in range(3)]
    one = build_vocab(functions, 2, source={"fingerprint": "x", "code_version": "y"})
    two = build_vocab(functions, 2, source={"code_version": "y", "fingerprint": "x"})

    one.save(tmp_path / "a.json")
    two.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_loading_refuses_an_unknown_format(small, tmp_path):
    data = small.to_json() | {"format": 999}
    (tmp_path / "v.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="format"):
        Vocab.load(tmp_path / "v.json")


def test_loading_refuses_reordered_specials(small):
    data = small.to_json()
    data["tokens"][0], data["tokens"][1] = data["tokens"][1], data["tokens"][0]
    with pytest.raises(ValueError, match="must start with"):
        Vocab.from_json(data)


def test_loading_refuses_a_repeated_token(small):
    data = small.to_json()
    data["tokens"].append("mov")
    with pytest.raises(ValueError, match="repeats"):
        Vocab.from_json(data)


# ---------------------------------------------------------------- leakage


def listing(mnemonics):
    return "\n".join(f"{i:x}\t{m}\tRAX RBX\t" for i, m in enumerate(mnemonics))


def add_function(conn, package, name, mnemonics, opt="O0"):
    with conn:
        binary_id = conn.execute(
            "INSERT OR IGNORE INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (f"/tmp/{package}_{opt}.dll", f"{package}{opt}"),
        ).lastrowid
        binary_id = conn.execute(
            "SELECT id FROM binaries WHERE sha256 = ?", (f"{package}{opt}",)
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped) "
            "VALUES (?, ?, 'gcc', ?, 1)",
            (binary_id, package, opt),
        )
        address = conn.execute(
            "SELECT COUNT(*) FROM functions WHERE binary_id = ?", (binary_id,)
        ).fetchone()[0] * 0x100 + 0x1000
        code = listing(mnemonics)
        fid = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, ?)",
            (binary_id, address, f"FUN_{address:x}"),
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
            (fid, binary_id, len(mnemonics), f"h{fid}", code),
        )


@pytest.fixture
def split_corpus(db):
    """A train package and a test package, each with a mnemonic of its own."""
    for i in range(3):
        add_function(db, "trainpkg", f"t{i}", ["push", "xor"] * 5 + [f"pad{i}"])
    for i in range(3):
        add_function(db, "testpkg", f"s{i}", ["push", "imul"] * 5 + [f"q{i}"])
    store_splits(db, {"trainpkg": "train", "testpkg": "test"})
    return db


def test_the_vocabulary_is_built_from_train_only(split_corpus):
    functions = train_functions(split_corpus)
    vocab = build_vocab(functions, min_functions=1)

    assert {key[0] for key, _ in functions} == {"trainpkg"}
    assert "xor" in vocab.tokens
    assert "imul" not in vocab.tokens, "a test-only token leaked into the vocabulary"


def test_building_without_a_recorded_split_is_refused(db):
    add_function(db, "trainpkg", "t0", ["push"] * 10)
    with pytest.raises(ValueError, match="dataset --assign"):
        train_functions(db)


# ---------------------------------------------------------------- coverage


def test_coverage_counts_each_fallback_separately(small):
    report = coverage(small, [
        ["mov", "REG64"],
        ["mov", "vfmadd231ps", "IMPORT:NeverSeenW", "IMPORT:memcpy"],
    ])

    assert report["tokens"] == 6
    assert report["unk_rate"] == pytest.approx(1 / 6)
    assert report["rare_import_rate"] == pytest.approx(1 / 6)
    assert report["functions_with_fallback"] == pytest.approx(1 / 2)


def test_coverage_of_nothing_is_zero_not_an_error(small):
    report = coverage(small, [])
    assert report["tokens"] == 0 and report["unk_rate"] == 0.0
