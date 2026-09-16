"""One-off: does the order of validation batches decide how much memory is held?

Validation embeds the pool, then the queries, in batches sorted by length.
Walked shortest first, each batch is larger than every one before it, so
the CUDA caching allocator can reuse none of the blocks it already holds and
keeps adding. Walked longest first, the first batch opens the largest blocks
and every later, smaller batch fits inside them. The batches themselves -
which functions share one - are identical either way; only the walk differs,
so the vectors should be bit-identical.

Run each order in a fresh process (the cache lives as long as the process):
  python experiments/alloc_probe.py asc
  python experiments/alloc_probe.py desc
  python experiments/alloc_probe.py compare
"""

import os
import sys
import time

import torch

from elenchus.db import connect
from elenchus.encoder.batching import pad
from elenchus.encoder.model import EncoderConfig, FunctionEncoder
from elenchus.encoder.scorer import EncoderScorer
from elenchus.encoder.vocab import Vocab
from elenchus.evaluation.task import retrieval_task

GIB = 1024 ** 3


@torch.no_grad()
def embed_walk(scorer, samples, reverse):
    """EncoderScorer.embed with the same batches, walked in either direction."""
    scorer.model.eval()
    sequences = [scorer._ids(s) for s in samples]
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    chunks = [order[start:start + scorer.batch_size]
              for start in range(0, len(order), scorer.batch_size)]
    if reverse:
        chunks = chunks[::-1]
    vectors = [None] * len(sequences)
    for chunk in chunks:
        ids, mask = pad([sequences[i] for i in chunk], scorer.vocab.pad_id)
        out = scorer.model.embed(ids.to(scorer.device), mask.to(scorer.device))
        for row, index in enumerate(chunk):
            vectors[index] = out[row].float().cpu()
    return torch.stack(vectors)


def run(order, device, db, vocab_path, out_dir):
    conn = connect(db)
    vocab = Vocab.load(vocab_path)
    queries, pool, _gold = retrieval_task(conn, split="val")
    torch.manual_seed(0)
    model = FunctionEncoder(EncoderConfig(vocab_size=len(vocab),
                                          pad_id=vocab.pad_id)).to(device)
    scorer = EncoderScorer(model, vocab, device=device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    vectors = torch.cat([embed_walk(scorer, pool, order == "desc"),
                         embed_walk(scorer, queries, order == "desc")])
    if cuda:
        torch.cuda.synchronize()
    seconds = time.time() - started
    torch.save(vectors, os.path.join(out_dir, f"alloc-{order}.pt"))
    allocated = torch.cuda.max_memory_allocated() / GIB if cuda else None
    reserved = torch.cuda.max_memory_reserved() / GIB if cuda else None
    config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)")
    print(f"order {order:4}  pool {len(pool)}  queries {len(queries)}  "
          f"{seconds:.1f}s  allocator {config}")
    if cuda:
        print(f"  peak allocated {allocated:.2f} GiB   peak reserved {reserved:.2f} GiB")
    return vectors


def compare(out_dir):
    asc = torch.load(os.path.join(out_dir, "alloc-asc.pt"))
    desc = torch.load(os.path.join(out_dir, "alloc-desc.pt"))
    print(f"vectors {tuple(asc.shape)}  bit-identical: {torch.equal(asc, desc)}  "
          f"max difference {(asc - desc).abs().max().item():.3g}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "compare":
        compare("data")
    elif what in ("asc", "desc"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run(what, device, os.environ["ELENCHUS_DB"], "models/vocab.json", "data")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
