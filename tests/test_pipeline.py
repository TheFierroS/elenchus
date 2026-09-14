"""One test that walks the whole pipeline, from empty database to a metric.

Every other test file checks a part. This one checks that the parts fit: a
function goes in as ground truth and instructions, comes out as a dataset
row, lands in a split, becomes a query, and is found. Nothing here is clever;
what it catches is the seam.

That distinction earned its place today. task.py was right, metrics.py was
right, and their combination capped every measurement at half of what the
task allowed, because one treated an answer as a single function and the
other had no opinion. No unit test could have seen it.
"""

import random

import pytest

from elenchus.corpus.dataset import (
    assign_splits,
    eligible_rows,
    leakage,
    load_splits,
    package_sizes,
    store_splits,
)
from elenchus.evaluation.baselines import BM25Mnemonics, RandomBaseline
from elenchus.evaluation.metrics import evaluate
from elenchus.evaluation.store import dataset_fingerprint, history, record
from elenchus.evaluation.task import retrieval_task

CRT = "/usr/src/mingw-w64-13.0.0/mingw-w64-crt/crt/pesect.c"


def listing(seed):
    """A distinctive instruction listing, stable for a given seed.

    Drawn at random from a seeded generator rather than from a formula. The
    first version used arithmetic on the seed, which made different functions
    collide after normalisation - and deduplication, correctly, threw nearly
    all of them away. The fixture was the thing at fault, but it is a fair
    warning about how easily synthetic data becomes uniform.
    """
    rng = random.Random(seed)
    opcodes = ["mov", "add", "xor", "shl", "cmp", "jz", "call", "ret",
               "push", "pop", "sub", "imul", "shr", "and", "or", "test",
               "lea", "movzx", "sete", "neg"]

    lines = []
    for index in range(rng.randint(24, 40)):
        lines.append(
            f"{index:x}\t{rng.choice(opcodes)}\t"
            f"RAX qwordptr[RBP+-0x{index + 8:x}]\t"
        )
    return "\n".join(lines)


def optimised(listing_text):
    """The same function as the compiler would leave it at a higher level.

    Shorter, and with the stack traffic gone - which is what optimisation
    actually does, and what the encoder will have to see through. Enough of
    the opcode sequence survives for a text method to have a chance, which
    is the point: the task must be hard but not impossible.
    """
    lines = listing_text.splitlines()
    kept = [line for index, line in enumerate(lines) if index % 3 != 0]
    return "\n".join(
        line.replace("qwordptr[RBP+-0x8]", "RCX") for line in kept
    )


@pytest.fixture
def corpus(db):
    """Four packages, two optimisation levels, one shared runtime function.

    Small enough to reason about, complete enough that every stage of the
    pipeline has something to do: a package to exclude by provenance, a
    function duplicated across packages to deduplicate, and pairs to match.
    """
    packages = ["zlib", "lua", "mpc", "miniz"]

    for package_index, package in enumerate(packages):
        for opt in ("O0", "O3"):
            with db:
                binary_id = db.execute(
                    "INSERT INTO binaries (path, sha256, arch, imported_at) "
                    "VALUES (?, ?, 'x86:LE:64:default', datetime('now'))",
                    (f"/tmp/{package}_{opt}.dll", f"{package}{opt}"),
                ).lastrowid
                db.execute(
                    "INSERT INTO corpus_binaries "
                    "(binary_id, package, compiler, opt_level, stripped) "
                    "VALUES (?, ?, 'gcc', ?, 1)",
                    (binary_id, package, opt),
                )

            for function_index in range(6):
                seed = package_index * 100 + function_index
                text = listing(seed)
                if opt == "O3":
                    text = optimised(text)

                add_function(
                    db, binary_id,
                    address=0x1000 + function_index * 0x100,
                    name=f"{package}_fn{function_index}",
                    decl_file=f"data/corpus-build/{package}/src/{package}.c",
                    text=text,
                )

            # The runtime function every binary gets, identical everywhere.
            add_function(
                db, binary_id, address=0x9000, name="_CRT_INIT",
                decl_file=CRT, text=listing(999),
            )

    return db


def add_function(conn, binary_id, address, name, decl_file, text):
    with conn:
        function_id = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 64, ?)",
            (binary_id, address, f"FUN_{address:x}"),
        ).lastrowid
        conn.execute(
            "INSERT INTO ground_truth "
            "(binary_id, function_id, address, name, return_type, "
            " param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, function_id, address, name, decl_file),
        )
        conn.execute(
            "INSERT INTO function_code "
            "(function_id, binary_id, n_instructions, code_size, byte_hash, "
            " listing) VALUES (?, ?, ?, 64, ?, ?)",
            (function_id, binary_id, len(text.splitlines()),
             f"h{function_id}", text),
        )
    return function_id


def test_the_whole_pipeline_runs(corpus):
    """From an empty database to a number, with every stage in between."""
    # 1. What may be trained on: the runtime is gone, the libraries remain.
    rows = eligible_rows(corpus)
    names = {row["name"] for row in rows}
    assert "_CRT_INIT" not in names
    assert "zlib_fn0" in names

    # 2. Packages are split whole, and every split gets one.
    sizes = package_sizes(rows)
    assignment = assign_splits(sizes)
    store_splits(corpus, assignment)
    assert set(assignment.values()) == {"train", "val", "test"}
    assert load_splits(corpus) == assignment

    # 3. Nothing crosses from one split into another.
    assert leakage(corpus, assignment) == []

    # 4. A retrieval task exists on the test side.
    split = next(
        name for name in ("test", "val", "train")
        if any(assignment[p] == name for p in sizes)
    )
    queries, pool, gold = retrieval_task(corpus, split=split)
    assert queries and pool
    assert len(queries) == len(gold)

    # 5. Every query's answer is inside the pool it was given.
    for answers in gold:
        assert answers
        assert all(0 <= index < len(pool) for index in answers)

    # 6. A method that reads the code beats one that does not.
    bm25 = evaluate(BM25Mnemonics(), queries, pool, gold)
    chance = evaluate(RandomBaseline(1), queries, pool, gold)
    assert bm25["mrr"] > chance["mrr"]

    # 7. The number is written down with the corpus it came from.
    with corpus:
        run_id = corpus.execute(
            "INSERT INTO runs (kind, code_version, params, started_at, status) "
            "VALUES ('eval', 'abc', '{}', datetime('now'), 'ok')"
        ).lastrowid

    record(
        corpus, run_id,
        {"split": split, "query_opt": "O0", "pool_opt": "O3"},
        {"bm25-mnemonic": bm25, "random": chance},
    )
    assert len(history(corpus)) == 2
    assert dataset_fingerprint(corpus)


def test_queries_and_pool_come_from_different_optimisation_levels(corpus):
    """The task is cross-optimisation, or it is not measuring anything."""
    store_splits(corpus, assign_splits(package_sizes(eligible_rows(corpus))))

    for split in ("train", "val", "test"):
        queries, pool, _gold = retrieval_task(corpus, split=split)
        assert all(sample.opt_level == "O0" for sample in queries)
        assert all(sample.opt_level == "O3" for sample in pool)


def test_a_query_never_meets_a_function_from_another_split(corpus):
    """The property the whole week's work exists to guarantee."""
    assignment = assign_splits(package_sizes(eligible_rows(corpus)))
    store_splits(corpus, assignment)

    for split in ("train", "val", "test"):
        queries, pool, _gold = retrieval_task(corpus, split=split)
        for sample in queries + pool:
            assert assignment[sample.package] == split


def test_the_runtime_function_never_reaches_the_task(corpus):
    """It is in every binary, identical; it would inflate every number."""
    store_splits(corpus, assign_splits(package_sizes(eligible_rows(corpus))))

    for split in ("train", "val", "test"):
        queries, pool, _gold = retrieval_task(corpus, split=split)
        for sample in queries + pool:
            assert sample.name != "_CRT_INIT"


def test_the_fingerprint_follows_the_corpus(corpus):
    """Two measurements are comparable only if this matches."""
    store_splits(corpus, assign_splits(package_sizes(eligible_rows(corpus))))
    before = dataset_fingerprint(corpus)

    binary_id = corpus.execute(
        "SELECT binary_id FROM corpus_binaries LIMIT 1"
    ).fetchone()["binary_id"]
    add_function(
        corpus, binary_id, 0xA000, "newcomer",
        "data/corpus-build/zlib/src/zlib.c", listing(4242),
    )

    assert dataset_fingerprint(corpus) != before
