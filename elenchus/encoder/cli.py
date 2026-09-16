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


# Command-line flag -> EncoderConfig field.
SIZE_FLAGS = {"d_model": "d_model", "layers": "n_layers", "heads": "n_heads",
              "d_ff": "d_ff", "embed_dim": "embed_dim", "dropout": "dropout",
              "pooling": "pooling"}


def size_overrides(args):
    """The configuration fields given on the command line, and only those."""
    return {field: getattr(args, flag) for flag, field in SIZE_FLAGS.items()
            if getattr(args, flag, None) is not None}


def cmd_train(args):
    """Train one stage: masked-LM pre-training or contrastive training."""
    from elenchus.encoder.train import ConfigConflict, Settings, StaleVocabulary, train
    from elenchus.encoder.vocab import Vocab

    conn = connect(args.db)
    vocab = Vocab.load(args.vocab)
    settings = Settings(
        stage=args.stage, out=args.out, epochs=args.epochs,
        batch_size=args.batch_size, per_package=args.per_package, lr=args.lr,
        temperature=args.temperature, queue_size=args.queue_size,
        momentum=args.momentum, max_len=args.max_len, patience=args.patience,
        seed=args.seed, init=args.init, device=args.device,
        max_steps=args.max_steps, train_fraction=args.train_fraction,
        train_unit=args.train_unit, eval_pair_share=args.eval_pair_share,
    )
    try:
        summary = train(conn, vocab, settings, overrides=size_overrides(args))
    except (StaleVocabulary, ConfigConflict) as exc:
        print(f"refused: {exc}")
        return 1

    def retrieval(label, m):
        print(f"{label:12}: recall@1 {m['recall@1']:.3f}  recall@10 "
              f"{m['recall@10']:.3f}  mrr {m['mrr']:.3f}  (mean over queries)")
        if "package_mean" in m:
            p = m["package_mean"]
            print(f"{'':12}  recall@1 {p['recall@1']:.3f}  recall@10 "
                  f"{p['recall@10']:.3f}  mrr {p['mrr']:.3f}  "
                  f"(mean over {p['packages']} packages)")

    print()
    for key, value in summary.items():
        if key.startswith("start_val_") and not isinstance(value, dict):
            print(f"{key:12}: {value:.4f}  (before training)")
    if "start_val_metrics" in summary:
        retrieval("start val", summary["start_val_metrics"])
    if summary["checkpoint"] is None:
        print("trained     : nothing (--max-steps 0)")
    else:
        print(f"best epoch  : {summary['best_epoch']}")
        for key, value in summary.items():
            if key.startswith("best_val_") and not isinstance(value, dict):
                print(f"{key:12}: {value:.4f}")
        if "best_val_metrics" in summary:
            retrieval("val", summary["best_val_metrics"])
        by_package = summary.get("best_epoch_by_package_mrr")
        if by_package is not None:
            verdict = ("the same" if by_package == summary["best_epoch"]
                       else f"DIFFERENT from {summary['best_epoch']}")
            print(f"by packages : epoch {by_package} would be kept ({verdict})")
    if summary.get("peak_allocated_gib") is not None:
        print(f"peak memory : {summary['peak_allocated_gib']:.2f} GiB allocated, "
              f"{summary['peak_reserved_gib']:.2f} GiB reserved")
    if summary["checkpoint"] is not None:
        print(f"checkpoint  : {summary['checkpoint']}")
    return 0
