"""Encoder commands.

Kept apart from the main CLI so the corpus and evaluation commands do not
grow model-side imports, and so the model side can import torch lazily.
"""

from elenchus.corpus.dataset import eligible_rows, load_splits
from elenchus.db import code_version, connect
from elenchus.encoder.normalise import normalise
from elenchus.encoder.vocab import (
    IMPORT_PREFIX,
    SPECIALS,
    build_vocab,
    coverage,
    train_functions,
)
from elenchus.evaluation.store import dataset_fingerprint


def cmd_vocab(args):
    """Build the vocabulary from train, save it, and report its coverage of val.

    Val coverage is reported rather than tuned against here: it answers
    whether a poor val score later could be the vocabulary's fault - the
    model being examined in tokens it was never taught.
    """
    conn = connect(args.db)
    rows = eligible_rows(conn)

    functions = train_functions(conn, rows)
    source = {
        "dataset_fingerprint": dataset_fingerprint(conn),
        "code_version": code_version(),
        "train_functions": len({key for key, _tokens in functions}),
        "train_rows": len(functions),
    }
    vocab = build_vocab(functions, min_functions=args.min_functions, source=source)
    vocab.save(args.out)

    ordinary = vocab.tokens[len(SPECIALS):]
    imports = sum(1 for token in ordinary if token.startswith(IMPORT_PREFIX))
    print(f"vocabulary      : {len(vocab)} tokens -> {args.out}")
    print(f"  specials      : {len(SPECIALS)}")
    print(f"  imports       : {imports}")
    print(f"  other         : {len(ordinary) - imports}")
    print(f"  threshold     : seen in >= {args.min_functions} train functions")
    print(f"  built from    : {source['train_rows']} rows, "
          f"{source['train_functions']} distinct functions")
    print(f"  fingerprint   : {source['dataset_fingerprint']}")

    assignment = load_splits(conn)
    val = [normalise(r["listing"]) for r in rows if assignment.get(r["package"]) == "val"]
    if val:
        report = coverage(vocab, val)
        print()
        print(f"val coverage    : {report['functions']} functions, "
              f"{report['tokens']} tokens")
        print(f"  [UNK]         : {report['unk_rate']:.3%} of tokens")
        print(f"  IMPORT:<rare> : {report['rare_import_rate']:.3%} of tokens")
        print(f"  functions touching either : {report['functions_with_fallback']:.1%}")
    return 0
