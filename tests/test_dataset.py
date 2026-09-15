"""Tests for dataset construction: exclusion, splitting, leakage.

These are the tests the week's exit criterion rests on. The failure they
guard against is not a crash - it is a dataset that looks fine and produces
excellent numbers because the model was shown the answers. So each test
builds a corpus containing exactly the mistake it is looking for and checks
that it is caught.
"""

import pytest

from elenchus.corpus.dataset import (
    assign_splits,
    candidate_rows,
    content_key,
    cross_split_collisions,
    eligible_rows,
    excluded_files,
    is_toolchain_path,
    leakage,
    load_splits,
    package_sizes,
    store_splits,
)

CRT = "/usr/src/mingw-w64-13.0.0/mingw-w64-crt/crt/pesect.c"


def listing(n, mnemonic="mov"):
    """A listing of n distinguishable instructions."""
    return "\n".join(
        f"{i:x}\t{mnemonic}\tRAX qwordptr[RBP+-0x{i + 8:x}]\t" for i in range(n)
    )


def add_function(conn, binary_id, address, name, decl_file, code, function_id=None):
    with conn:
        fid = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, ?)",
            (binary_id, address, f"FUN_{address:x}"),
        ).lastrowid
        conn.execute(
            "INSERT INTO ground_truth "
            "(binary_id, function_id, address, name, return_type, "
            " param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, fid, address, name, decl_file),
        )
        conn.execute(
            "INSERT INTO function_code "
            "(function_id, binary_id, n_instructions, code_size, byte_hash, "
            " listing) VALUES (?, ?, ?, 32, ?, ?)",
            (fid, binary_id, len(code.splitlines()), f"h{fid}", code),
        )
    return fid


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


@pytest.fixture
def corpus(db):
    """Two packages, each with its own code plus the same runtime function."""
    # Each package gets code of its own - different mnemonics, so the
    # normalised forms differ. Giving both packages identical bodies would
    # make this fixture itself a leak, which the leakage test would rightly
    # flag; that case is covered separately below.
    for package, source, op in (
        ("zlib", "crc32.c", "xor"), ("lua", "lapi.c", "imul")
    ):
        binary_id = add_binary(db, package)
        path = f"data/corpus-build/{package}/src/{source}"

        add_function(db, binary_id, 0x1000, f"{package}_work", path,
                     listing(30, op))
        add_function(db, binary_id, 0x2000, f"{package}_more", path,
                     listing(25, op + "x" if op == "xor" else "shl"))
        # The runtime function every binary gets, byte for byte the same.
        add_function(db, binary_id, 0x3000, "_CRT_INIT", CRT, listing(20, "sub"))

    return db


def test_toolchain_paths_are_recognised():
    assert is_toolchain_path(CRT)
    assert is_toolchain_path("/usr/x86_64-w64-mingw32/include/stdlib.h")
    assert is_toolchain_path("/usr/share/mingw-w64/include/stdio.h")
    assert not is_toolchain_path("data/corpus-build/zlib/src/zlib-1.3.1/crc32.c")


def test_runtime_source_is_excluded_by_both_rules(corpus):
    """It is shared across packages and it lives in the toolchain."""
    dropped = excluded_files(corpus)
    assert dropped[CRT] == "shared+toolchain"
    assert len(dropped) == 1


def test_eligible_rows_drop_the_runtime(corpus):
    names = {row["name"] for row in eligible_rows(corpus)}
    assert names == {"zlib_work", "zlib_more", "lua_work", "lua_more"}
    assert "_CRT_INIT" not in names


def test_split_keeps_a_package_whole():
    sizes = {"lua": 3000, "zlib": 500, "cjson": 400, "sds": 100}
    assignment = assign_splits(sizes)

    assert set(assignment) == set(sizes)
    assert set(assignment.values()) <= {"train", "val", "test"}
    # The largest package anchors the largest split.
    assert assignment["lua"] == "train"


def test_every_split_gets_at_least_one_package():
    """An empty test set makes the exit criterion unmeasurable."""
    for count in range(3, 9):
        sizes = {f"p{i}": 100 * (i + 1) for i in range(count)}
        assignment = assign_splits(sizes)
        assert set(assignment.values()) == {"train", "val", "test"}


def test_split_is_deterministic():
    sizes = {"lua": 3000, "zlib": 500, "cjson": 400, "mpc": 800, "sds": 90}
    assert assign_splits(sizes) == assign_splits(sizes)


def test_splits_round_trip_through_the_database(corpus):
    assignment = {"zlib": "train", "lua": "test"}
    store_splits(corpus, assignment)
    assert load_splits(corpus) == assignment

    store_splits(corpus, {"zlib": "val", "lua": "test"})
    assert load_splits(corpus)["zlib"] == "val"


def test_no_leakage_once_the_runtime_is_excluded(corpus):
    """The runtime function is in both packages but never in the dataset."""
    assert leakage(corpus, {"zlib": "train", "lua": "test"}) == []


def test_a_vendored_copy_is_found_and_then_removed(db):
    """The failure this whole module exists to prevent, and its fix.

    Two packages, two different source files, identical code - as a copied
    helper or a one-line wrapper would be. No source-file rule fires, so
    only deduplication can catch it.
    """
    shared = listing(40, "xor")
    assignment = {"zlib": "train", "lua": "test"}

    for package in ("zlib", "lua"):
        binary_id = add_binary(db, package)
        add_function(
            db, binary_id, 0x1000, f"{package}_copy",
            f"data/corpus-build/{package}/src/vendored.c", shared,
        )

    # Before deduplication the collision is real and spans both splits.
    before = cross_split_collisions(candidate_rows(db), assignment)
    assert len(before) == 1
    assert before[0][1] == ["lua_copy (lua/test)", "zlib_copy (zlib/train)"]

    # After it, the rows are gone from the dataset entirely.
    assert leakage(db, assignment) == []
    assert [row["name"] for row in eligible_rows(db)] == []


def test_tiny_functions_never_enter_the_dataset(db):
    """A three-instruction stub is identical everywhere and means nothing."""
    stub = listing(3)

    for package in ("zlib", "lua"):
        binary_id = add_binary(db, package)
        add_function(
            db, binary_id, 0x1000, f"{package}_stub",
            f"data/corpus-build/{package}/src/x.c", stub,
        )

    assert candidate_rows(db) == []
    assert leakage(db, {"zlib": "train", "lua": "test"}) == []


def test_content_key_ignores_what_normalisation_drops():
    """Two functions differing only in stack offsets are the same function."""
    a = "0\tmov\tRAX qwordptr[RBP+-0x18]\t"
    b = "0\tmov\tRCX qwordptr[RBP+-0x40]\t"
    assert content_key(a) == content_key(b)


def test_package_sizes_counts_rows(corpus):
    assert package_sizes(eligible_rows(corpus)) == {"zlib": 2, "lua": 2}


# ---------------------------------------------------------------- stratified split


def test_every_domain_of_three_or_more_reaches_every_split():
    """The crypto case: four packages, one of them must be in test."""
    sizes = {"mbedtls": 5500, "libsodium": 1810, "libtomcrypt": 1658, "monocypher": 377,
             "flecs": 6952, "sqlite": 5408, "libuv": 2019, "logc": 20}
    domains = {"mbedtls": "crypto", "libsodium": "crypto", "libtomcrypt": "crypto",
               "monocypher": "crypto", "flecs": "systems", "sqlite": "systems",
               "libuv": "systems", "logc": "systems"}
    assignment = assign_splits(sizes, domains=domains)

    for domain in ("crypto", "systems"):
        splits = {assignment[p] for p in sizes if domains[p] == domain}
        assert splits == {"train", "val", "test"}, domain


def test_without_domains_the_split_is_the_old_one():
    sizes = {f"p{i}": 100 * (i + 1) for i in range(12)}
    assert assign_splits(sizes, domains=None) == assign_splits(sizes)


def test_a_package_with_no_domain_is_not_mixed_into_another():
    sizes = {"a": 50, "b": 40, "c": 30, "stray": 1000}
    domains = {"a": "text", "b": "text", "c": "text"}
    assignment = assign_splits(sizes, domains=domains)

    # Alone in its group, the stray goes where a lone package goes: train.
    assert assignment["stray"] == "train"
    assert {assignment[p] for p in "abc"} == {"train", "val", "test"}


def test_the_smallest_package_of_every_domain_does_not_always_land_in_test():
    """Filling val before test in every small domain left test at 12% of rows;
    the global shortfall decides instead."""
    sizes, domains = {}, {}
    for d in range(6):
        for name, size in (("big", 1000), ("mid", 300), ("small", 10)):
            sizes[f"{name}{d}"] = size
            domains[f"{name}{d}"] = f"d{d}"
    assignment = assign_splits(sizes, domains=domains)

    total = sum(sizes.values())
    for split in ("val", "test"):
        rows = sum(size for p, size in sizes.items() if assignment[p] == split)
        assert rows >= 0.10 * total, f"{split} starved: {rows} of {total}"


def test_a_stratified_split_is_deterministic():
    sizes = {f"p{i}": 37 * i + 11 for i in range(20)}
    domains = {f"p{i}": f"d{i % 4}" for i in range(20)}
    assert assign_splits(sizes, domains=domains) == assign_splits(dict(
        reversed(list(sizes.items()))), domains=domains)
