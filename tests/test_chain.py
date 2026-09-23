"""Finding a function's initialiser - and refusing to guess.

The finder decides whether a chain is run at all, so what it *declines* is as
important as what it finds: a wrong initialiser writes a context that looks
valid, which is worse than the noise it replaces. These check both.
"""

import json

from elenchus.emulation.chain import (
    _chainable_signature,
    candidate_names,
    find_initialiser,
)

POINTER = {"kind": "pointer", "size": 8, "signed": False}
OTHER_POINTER = {"kind": "pointer", "size": 4, "signed": False}
INT = {"kind": "int", "size": 4, "signed": True}
VOID = {"kind": "void", "size": 0, "signed": None}


def abi(params, ret=None, variadic=False):
    return {"return": ret or VOID, "params": list(params), "variadic": variadic}


def sibling(name, params, variadic=False):
    return (name, json.dumps(abi(params, variadic=variadic)))


# ------------------------------------------------------- candidate names


def test_the_last_segment_is_replaced_not_appended():
    names = candidate_names("rhash_ripemd160_final")
    assert "rhash_ripemd160_init" in names
    assert "rhash_ripemd160_new" in names
    assert "rhash_ripemd160_final_init" not in names       # not appended


def test_an_initialiser_itself_gets_no_candidates():
    """foo_init does not need an initialiser of its own."""
    assert candidate_names("foo_init") == []
    assert candidate_names("parser_new") == []


def test_a_bare_name_gets_no_candidates():
    """Guessing from a name with no separator would match far too much."""
    assert candidate_names("strlen") == []
    assert candidate_names("main") == []


# ------------------------------------------------------- what it accepts


def test_it_finds_the_matching_initialiser():
    found = find_initialiser(
        "rhash_ripemd160_final", abi([POINTER, POINTER]),
        [sibling("rhash_ripemd160_init", [POINTER]),
         sibling("rhash_ripemd160_update", [POINTER, POINTER, INT]),
         sibling("unrelated_helper", [INT])])
    assert found is not None
    assert found[0] == "rhash_ripemd160_init"


def test_an_initialiser_may_take_integers_after_the_context():
    """A size or a flag is fine: the chain can supply those."""
    found = find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("ctx_init", [POINTER, INT, INT])])
    assert found is not None


# ------------------------------------------------------- what it refuses


def test_a_different_context_type_is_not_ours():
    """The first parameter must be the same resolved type, or the
    'initialiser' builds something else entirely."""
    assert find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("ctx_init", [OTHER_POINTER])]) is None


def test_an_initialiser_needing_a_second_pointer_is_refused():
    """Filling that pointer is the very problem the chain exists to avoid."""
    assert find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("ctx_init", [POINTER, POINTER])]) is None


def test_a_variadic_initialiser_is_refused():
    assert find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("ctx_init", [POINTER], variadic=True)]) is None


def test_two_candidates_mean_no_choice():
    """Ambiguity is not a choice: both ctx_init and ctx_new fit, so neither
    is used."""
    assert find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("ctx_init", [POINTER]),
         sibling("ctx_new", [POINTER])]) is None


def test_a_function_without_a_context_gets_no_chain():
    """add3(int, int, int) has nothing to initialise."""
    assert find_initialiser(
        "math_add3", abi([INT, INT, INT]),
        [sibling("math_init", [POINTER])]) is None


def test_no_sibling_means_no_chain():
    assert find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("other_init", [POINTER])]) is None


def test_a_prefix_match_alone_is_not_enough():
    """ctx_helper_init shares a prefix but is not ctx_finish's initialiser."""
    assert find_initialiser(
        "ctx_finish", abi([POINTER]),
        [sibling("ctx_helper_init", [POINTER])]) is None


# ------------------------------------------------------- signature check


def test_chainable_signature_rules():
    assert _chainable_signature(abi([POINTER]))
    assert _chainable_signature(abi([POINTER, INT]))
    assert not _chainable_signature(abi([INT]))              # no context
    assert not _chainable_signature(abi([]))                 # nothing at all
    assert not _chainable_signature(abi([POINTER, POINTER]))
    assert not _chainable_signature(None)


# ------------------------------------------------------- the returning shape


def test_a_constructor_that_returns_the_context_is_accepted():
    """Most C libraries hand the context back rather than filling a buffer:
    cJSON_CreateObject, xmlNewDoc. Measured on the corpus, 275 of the
    initialisers a name match finds are this shape."""
    found = find_initialiser(
        "pool_total", abi([POINTER]),
        [("pool_create", json.dumps(abi([], ret=POINTER)))])
    assert found is not None
    assert found[0] == "pool_create"
    assert found[2] == "returns"


def test_a_returning_constructor_that_wants_a_pointer_is_refused():
    """cJSON_Parse(const char *text) returns a context but needs invented data
    for its own argument, which is the problem the chain exists to avoid. A
    single pointer of the *context's own type* is still the in-place shape and
    is allowed; a second one is not."""
    assert find_initialiser(
        "doc_free", abi([POINTER]),
        [("doc_create", json.dumps(abi([OTHER_POINTER], ret=POINTER)))]) is None
    assert find_initialiser(
        "doc_free", abi([POINTER]),
        [("doc_create", json.dumps(abi([POINTER, POINTER], ret=POINTER)))]) is None


def test_a_returning_constructor_may_take_integers():
    found = find_initialiser(
        "pool_total", abi([POINTER]),
        [("pool_create", json.dumps(abi([INT], ret=POINTER)))])
    assert found is not None and found[2] == "returns"


def test_both_shapes_at_once_are_ambiguous():
    """pool_init fills a buffer and pool_create returns one: two ways to make
    a context, so neither is chosen."""
    assert find_initialiser(
        "pool_total", abi([POINTER]),
        [("pool_init", json.dumps(abi([POINTER]))),
         ("pool_create", json.dumps(abi([], ret=POINTER)))]) is None
