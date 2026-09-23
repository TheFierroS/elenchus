"""Finding the initialiser that makes a function's context meaningful.

A function that takes a context - a hash state, a parser, an allocator -
expects that context to have been built by its own initialiser. Handed
invented bytes instead it walks garbage, faults, frees a pointer that was
never allocated or trips an assertion, and the verifier can only decline: the
500-pair run put 17.6% in faults and 35 of 46 stub declines down to exactly
this (docs/verifier.md, constructor chains).

This module finds the initialiser, from the ground truth the reference build
carries: the same package and source file, a name that is the function's with
its last segment replaced by init/new/create/..., and a first parameter of
the same type. It is deliberately strict - a wrong initialiser writes a
context that *looks* valid, which is worse than noise - and when nothing
fits, or more than one does, it finds none and the caller tests the function
as before.

It reads records and compares strings; it runs nothing, so it is tested on
its own.
"""

from __future__ import annotations

import json

# The last segment a constructor's name ends in, where the function under
# test has something else (final, free, destroy, process...). Ordered by how
# specific they are; an exact-name match against any of them qualifies.
CONSTRUCTOR_SUFFIXES = (
    "init", "new", "create", "setup", "start", "open", "alloc",
    "initialize", "initialise", "init_static", "reset",
)


def _split_last_segment(name: str):
    """('rhash_ripemd160', 'final') for rhash_ripemd160_final.

    Returns (None, None) for a name with no separator, which cannot be
    matched by suffix replacement.
    """
    if "_" not in name:
        return None, None
    head, _, tail = name.rpartition("_")
    if not head or not tail:
        return None, None
    return head, tail


def candidate_names(name: str) -> list[str]:
    """The initialiser names that would go with `name`.

    rhash_ripemd160_final -> rhash_ripemd160_init, _new, _create, ...
    A name with no underscore gets no candidates: guessing from a bare name
    would match far too much.
    """
    head, tail = _split_last_segment(name)
    if head is None:
        return []
    if tail in CONSTRUCTOR_SUFFIXES:
        return []                       # it is an initialiser itself
    return [f"{head}_{suffix}" for suffix in CONSTRUCTOR_SUFFIXES]


def _first_param(abi) -> dict | None:
    params = (abi or {}).get("params") or []
    return params[0] if params else None


def _chainable_signature(abi) -> bool:
    """Whether an initialiser's signature is one the chain can call.

    It must take the context pointer first, and nothing after it that would
    need invented data of its own - integers are fine (a size, a flag), a
    second pointer is not, because filling it is the very problem the chain
    exists to avoid.
    """
    if abi is None or abi.get("variadic"):
        return False
    params = abi.get("params") or []
    if not params or params[0]["kind"] != "pointer":
        return False
    return all(p["kind"] == "int" for p in params[1:])


def find_initialiser(name: str, abi, siblings):
    """Return the one initialiser for `name`, or None.

    siblings is an iterable of (name, abi_json) for the functions in the same
    package and source file. The match must be exact on the name and on the
    first parameter's resolved type; anything less would risk writing a
    context that looks valid but is not.

    None when nothing fits *and* when more than one does: an ambiguous choice
    is not a choice.
    """
    wanted = candidate_names(name)
    if not wanted:
        return None
    context = _first_param(abi)
    if context is None or context["kind"] != "pointer":
        return None

    found = []
    for sibling_name, sibling_abi_json in siblings:
        if sibling_name not in wanted:
            continue
        sibling_abi = (json.loads(sibling_abi_json)
                       if isinstance(sibling_abi_json, str) else sibling_abi_json)
        if not _chainable_signature(sibling_abi):
            continue
        if _first_param(sibling_abi) != context:
            continue                    # a different context type: not ours
        found.append((sibling_name, sibling_abi))

    if len(found) != 1:
        return None
    return found[0]
