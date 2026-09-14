"""Tests for the inspect view.

The view exists so a person can catch a matching error the match rate cannot
see, which means the thing worth testing is that it shows what it claims to:
the compiler's name next to Ghidra's, the raw instruction next to its
normalised form, and a missing ground truth stated rather than hidden.
"""

import json

import pytest

from elenchus.inspect import (
    find_functions,
    render,
    sample_functions,
    side_by_side,
    signature,
)

LISTING = "\n".join([
    "0\tpush\tRBP\t",
    "1\tmov\tRBP RSP\t",
    "4\tcmp\tdwordptr[RBP+-0x4] 0x1000\t",
    "b\tjz\t0x401040\tLOCAL:+0x12",
    "d\tcall\tqwordptr[0x404120]\tIMPORT:KERNEL32.dll!ReadFile",
])


@pytest.fixture
def corpus(db):
    """Two functions in one corpus binary: one named, one runtime glue."""
    with db:
        binary_id = db.execute(
            "INSERT INTO binaries (path, sha256, arch, imported_at) "
            "VALUES ('/tmp/zlib_O0_stripped.dll', 'aa', 'x86:LE:64:default', "
            "        datetime('now'))"
        ).lastrowid

        db.execute(
            "INSERT INTO corpus_binaries "
            "(binary_id, package, version, compiler, opt_level, stripped) "
            "VALUES (?, 'zlib', '1.3.1', 'gcc', 'O0', 1)",
            (binary_id,),
        )

        named = db.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, 0x401000, 20, 'FUN_00401000')",
            (binary_id,),
        ).lastrowid

        glue = db.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, 0x402000, 12, 'FUN_00402000')",
            (binary_id,),
        ).lastrowid

        db.execute(
            "INSERT INTO ground_truth "
            "(binary_id, function_id, address, name, return_type, "
            " param_types, decl_line) "
            "VALUES (?, ?, 0x401000, 'deflate_stored', 'int', ?, 1234)",
            (binary_id, named, json.dumps(["struct z_stream*", "int"])),
        )

        for function_id in (named, glue):
            db.execute(
                "INSERT INTO function_code "
                "(function_id, binary_id, n_instructions, code_size, "
                " byte_hash, listing) VALUES (?, ?, 5, 20, 'abcdef123456', ?)",
                (function_id, binary_id, LISTING),
            )

    return db, named, glue


def test_signature_is_rebuilt_from_the_compiler(corpus):
    db, named, _ = corpus
    row = find_functions(db, function_id=named)[0]
    assert signature(row) == "int deflate_stored(struct z_stream*, int)"


def test_missing_ground_truth_is_stated_not_hidden(corpus):
    db, _, glue = corpus
    row = find_functions(db, function_id=glue)[0]
    assert "no ground truth" in signature(row)


def test_truth_filter_separates_source_from_runtime(corpus):
    db, named, glue = corpus
    assert [r["function_id"] for r in find_functions(db, truth=True)] == [named]
    assert [r["function_id"] for r in find_functions(db, truth=False)] == [glue]
    assert len(find_functions(db)) == 2


def test_filters_narrow_by_package_and_level(corpus):
    db, _, _ = corpus
    assert len(find_functions(db, package="zlib", opt="O0")) == 2
    assert find_functions(db, package="lua") == []
    assert len(find_functions(db, name="deflate_stored")) == 1


def test_columns_pair_each_instruction_with_its_tokens(corpus):
    db, _, _ = corpus
    lines = side_by_side(LISTING)

    call = next(line for line in lines if "ReadFile" in line and "call" in line)
    # The raw side keeps the address it was called through; the normalised
    # side keeps only the name, which is the whole point of the scheme.
    assert "0x404120" in call
    assert "IMPORT:ReadFile" in call

    branch = next(line for line in lines if "jz" in line)
    assert "0x401040" in branch
    assert "BB_FWD" in branch


def test_long_listings_are_truncated(corpus):
    db, _, _ = corpus
    lines = side_by_side(LISTING, limit=2)
    assert any("3 more instructions" in line for line in lines)


def test_render_puts_both_names_in_the_header(corpus):
    db, named, _ = corpus
    row = find_functions(db, function_id=named)[0]
    text = "\n".join(render(db, row))

    assert "FUN_00401000" in text          # what Ghidra called it
    assert "deflate_stored" in text        # what the compiler called it
    assert "line 1234" in text
    assert "zlib -O0" in text


def test_sampling_is_reproducible():
    rows = list(range(100))
    assert sample_functions(rows, 10, seed=1) == sample_functions(rows, 10, seed=1)
    assert sample_functions(rows, 10, seed=1) != sample_functions(rows, 10, seed=2)
    assert len(sample_functions(rows, 500, seed=1)) == 100
