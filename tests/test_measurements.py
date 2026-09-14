"""Tests for recording measurements.

The value of a stored number is that it can be compared to another one, and
that only works if the circumstance is stored with it. So what is tested here
is mostly the fingerprint: that it changes when the dataset changes, and that
it does not change when nothing meaningful did.
"""

import json

import pytest

from elenchus.corpus.dataset import store_splits
from elenchus.evaluation.store import (
    dataset_fingerprint,
    format_history,
    history,
    record,
    short_commit,
    task_params,
)


def add_binary(conn, package, opt="O0"):
    with conn:
        binary_id = conn.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
            (f"/tmp/{package}_{opt}.dll", f"{package}{opt}"),
        ).lastrowid
        conn.execute(
            "INSERT INTO corpus_binaries "
            "(binary_id, package, compiler, opt_level, stripped) "
            "VALUES (?, ?, 'gcc', ?, 1)",
            (binary_id, package, opt),
        )
    return binary_id


def add_function(conn, binary_id, address, name, package, op="mov"):
    listing = "\n".join(
        f"{i:x}\t{op}\tRAX qwordptr[RBP+-0x{i + 8:x}]\t" for i in range(12)
    )
    with conn:
        function_id = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, ?)",
            (binary_id, address, f"FUN_{address:x}"),
        ).lastrowid
        conn.execute(
            "INSERT INTO ground_truth "
            "(binary_id, function_id, address, name, return_type, "
            " param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, function_id, address, name,
             f"data/corpus-build/{package}/src/x.c"),
        )
        conn.execute(
            "INSERT INTO function_code "
            "(function_id, binary_id, n_instructions, code_size, byte_hash, "
            " listing) VALUES (?, ?, 12, 32, ?, ?)",
            (function_id, binary_id, f"h{function_id}", listing),
        )
    return function_id


@pytest.fixture
def corpus(db):
    for index, (package, op) in enumerate((("zlib", "xor"), ("lua", "imul"))):
        binary_id = add_binary(db, package)
        add_function(db, binary_id, 0x1000 + index * 0x100,
                     f"{package}_one", package, op)
    store_splits(db, {"zlib": "train", "lua": "test"})
    return db


def test_fingerprint_is_stable_for_an_unchanged_corpus(corpus):
    assert dataset_fingerprint(corpus) == dataset_fingerprint(corpus)


def test_fingerprint_changes_when_the_split_changes(corpus):
    before = dataset_fingerprint(corpus)
    store_splits(corpus, {"zlib": "test", "lua": "train"})
    assert dataset_fingerprint(corpus) != before


def test_fingerprint_changes_when_a_package_arrives(corpus):
    before = dataset_fingerprint(corpus)

    binary_id = add_binary(corpus, "mpc")
    add_function(corpus, binary_id, 0x5000, "mpc_one", "mpc", "shl")
    store_splits(corpus, {"zlib": "train", "lua": "test", "mpc": "val"})

    assert dataset_fingerprint(corpus) != before


def test_recorded_numbers_come_back_with_their_circumstance(corpus):
    with corpus:
        run_id = corpus.execute(
            "INSERT INTO runs (kind, code_version, params, started_at, status) "
            "VALUES ('eval', 'abc1234', ?, datetime('now'), 'ok')",
            (json.dumps({"dataset": "deadbeef"}),),
        ).lastrowid

    task = {"split": "test", "query_opt": "O0", "pool_opt": "O3"}
    written = record(corpus, run_id, task, {
        "bm25-mnemonic": {"queries": 10, "pool": 12, "mrr": 0.4,
                          "recall@1": 0.2},
        "random": {"queries": 10, "pool": 12, "mrr": 0.05, "recall@1": 0.0},
    })
    assert written == 2

    rows = history(corpus)
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"bm25-mnemonic", "random"}

    # The commit that produced the number travels with it.
    assert all(row["code_version"] == "abc1234" for row in rows)

    text = "\n".join(format_history(rows))
    assert "deadbeef" in text
    assert "test O0->O3" in text


def test_history_can_be_narrowed(corpus):
    with corpus:
        run_id = corpus.execute(
            "INSERT INTO runs (kind, code_version, params, started_at, status) "
            "VALUES ('eval', 'abc', '{}', datetime('now'), 'ok')"
        ).lastrowid

    for split in ("train", "test"):
        record(corpus, run_id,
               {"split": split, "query_opt": "O0", "pool_opt": "O3"},
               {"random": {"queries": 1, "pool": 1, "mrr": 0.1}})

    assert len(history(corpus, split="test")) == 1
    assert len(history(corpus, method="random")) == 2
    assert len(history(corpus, method="bm25-mnemonic")) == 0


def test_an_uncommitted_run_is_marked_not_truncated():
    """The suffix is the point: it says this number cannot be reproduced."""
    assert short_commit("7722946abc") == "7722946"
    assert short_commit("7722946abc-dirty") == "7722946*"
    assert short_commit(None) == "?"


def test_empty_history_says_so(db):
    assert format_history([]) == ["no measurements recorded yet"]


def test_task_params_carry_everything_needed_to_repeat(corpus):
    params = task_params(corpus, "test", "O0", "O3", seed=7)

    assert params["split"] == "test"
    assert params["query_opt"] == "O0"
    assert params["pool_opt"] == "O3"
    assert params["seed"] == 7
    assert params["dataset"] == dataset_fingerprint(corpus)
