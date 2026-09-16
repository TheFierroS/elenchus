"""The encoder as a scorer, beside the baselines in the same evaluation.

evaluate() asks a scorer to prepare the pool once and then score each query
against it. Here that means embedding the pool once, and a dot product per
query - the reason retrieval scales where reading every function would not.

Queries are embedded in batches too, ahead of scoring (prepare_queries). One
at a time, every query is a forward pass of its own length: 1,643 val
queries asked the CUDA allocator for 1,643 differently sized blocks, and its
cache grew to 9.00 GiB reserved for 2.46 GiB of tensors - past the card,
into shared system memory. With expandable segments the same epoch reserved
3.61 GiB and ran 30 s instead of 40 s, the same MRR: fragmentation, not need.

Batched and single vectors agree to ~1e-7, which can still swap two
candidates whose scores are that close. Neither order is more correct; what
matters is that every measurement is taken one way, and it is this one.
"""

import torch

from elenchus.encoder.batching import pad
from elenchus.encoder.data import DEFAULT_MAX_LEN, with_cls
from elenchus.encoder.normalise import normalise


class EncoderScorer:
    name = "encoder"

    def __init__(self, model, vocab, max_len=DEFAULT_MAX_LEN, device="cpu",
                 batch_size=64, name=None):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size
        if name:
            self.name = name
        self.pool = None
        self._queries = {}

    def _ids(self, sample):
        tokens = normalise(sample.listing)
        return with_cls(self.vocab.encode(tokens), self.vocab.cls_id, self.max_len)

    @torch.no_grad()
    def embed(self, samples):
        was_training = self.model.training
        self.model.eval()
        try:
            sequences = [self._ids(s) for s in samples]
            # Similar lengths share a batch: less padding, same result.
            order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
            vectors = [None] * len(sequences)
            for start in range(0, len(order), self.batch_size):
                chunk = order[start:start + self.batch_size]
                ids, mask = pad([sequences[i] for i in chunk], self.vocab.pad_id)
                out = self.model.embed(ids.to(self.device), mask.to(self.device))
                for row, index in enumerate(chunk):
                    vectors[index] = out[row].float().cpu()
            return torch.stack(vectors) if vectors else torch.empty(0)
        finally:
            self.model.train(was_training)

    def prepare(self, pool):
        self.pool = self.embed(pool)
        self._queries = {}

    def prepare_queries(self, queries):
        """Embed every query in length-sorted batches, before scoring them.

        Kept by object identity, with the object itself held, so a vector can
        only ever be handed back for the very query it was computed from.
        """
        vectors = self.embed(queries)
        self._queries = {id(query): (query, vector)
                         for query, vector in zip(queries, vectors)}

    def scores(self, query):
        if self.pool is None:
            raise RuntimeError("prepare(pool) must be called first")
        cached = self._queries.get(id(query))
        if cached is not None and cached[0] is query:
            vector = cached[1]
        else:
            vector = self.embed([query])[0]
        return (self.pool @ vector).tolist()
