"""Tests for the encoder model, its losses, the memory queue and the scorer.

The model's numbers cannot be checked by eye, so the tests check properties:
padding does not change a vector, a removed candidate really contributes
nothing to the loss, the true key is never removed, the momentum copy moves
by exactly the amount asked, the queue forgets in order.
"""

import math
import types

import pytest
import torch

from elenchus.encoder.batching import pad
from elenchus.encoder.data import IGNORE_INDEX
from elenchus.encoder.losses import (
    MemoryQueue,
    info_nce,
    mlm_loss,
    momentum_copy,
    momentum_update,
)
from elenchus.encoder.model import EncoderConfig, FunctionEncoder
from elenchus.encoder.scorer import EncoderScorer
from elenchus.encoder.vocab import build_vocab

torch.manual_seed(0)


def tiny(pooling="mean", vocab_size=40):
    config = EncoderConfig(vocab_size=vocab_size, max_len=32, d_model=32, n_layers=2,
                           n_heads=2, d_ff=64, dropout=0.0, embed_dim=16,
                           pooling=pooling)
    return FunctionEncoder(config).eval()


# ---------------------------------------------------------------- model


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_padding_does_not_change_a_functions_vector(pooling):
    """The property every batch depends on."""
    model = tiny(pooling)
    short = [1, 7, 8, 9]
    alone_ids, alone_mask = pad([short], 0)
    batch_ids, batch_mask = pad([short, [1] + list(range(10, 30))], 0)

    with torch.no_grad():
        alone = model.embed(alone_ids, alone_mask)[0]
        in_batch = model.embed(batch_ids, batch_mask)[0]
    assert torch.allclose(alone, in_batch, atol=1e-5)


def test_embeddings_are_unit_length():
    model = tiny()
    ids, mask = pad([[1, 5, 6], [1, 7]], 0)
    with torch.no_grad():
        vectors = model.embed(ids, mask)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_different_code_gives_different_vectors():
    model = tiny()
    ids, mask = pad([[1, 5, 6, 7], [1, 20, 21, 22]], 0)
    with torch.no_grad():
        a, b = model.embed(ids, mask)
    assert not torch.allclose(a, b)


def test_cls_pooling_reads_the_first_position_only():
    model = tiny("cls")
    with torch.no_grad():
        hidden = model.hidden(*pad([[1, 5, 6]], 0))
        expected = torch.nn.functional.normalize(model.projection(hidden[:, 0]), dim=-1)
        assert torch.allclose(model.embed(*pad([[1, 5, 6]], 0)), expected, atol=1e-6)


def test_a_sequence_longer_than_max_len_is_refused():
    model = tiny()
    ids, mask = pad([[1] * 40], 0)
    with pytest.raises(ValueError, match="max_len"):
        model.embed(ids, mask)


def test_mlm_logits_cover_the_vocabulary_at_every_position():
    model = tiny(vocab_size=40)
    ids, mask = pad([[1, 5, 6], [1, 7]], 0)
    assert model.mlm_logits(ids, mask).shape == (2, 3, 40)


@pytest.mark.parametrize("seed", range(3))
def test_an_untrained_model_starts_masked_lm_at_an_even_guess(seed):
    """The loss of guessing uniformly over V tokens is ln(V).

    Too large a starting scale makes an untrained model confidently wrong:
    run 492 began at 78 instead of ~5.8. A wide model shows it most, because
    the logits grow with d_model; width 64 is wide enough to fail loudly at
    std 1.0 (loss ~57) and still quick.
    """
    vocab_size = 321
    torch.manual_seed(seed)
    model = FunctionEncoder(EncoderConfig(
        vocab_size=vocab_size, max_len=128, d_model=64, n_layers=2, n_heads=2,
        d_ff=128, dropout=0.0, embed_dim=16)).eval()
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, vocab_size, (16, 96), generator=generator)
    hidden = torch.rand(ids.shape, generator=generator) < 0.15
    labels = torch.where(hidden, ids, torch.full_like(ids, IGNORE_INDEX))
    inputs = torch.where(hidden, torch.full_like(ids, 2), ids)  # [MASK]
    mask = torch.ones_like(ids, dtype=torch.bool)

    with torch.no_grad():
        loss = mlm_loss(model.mlm_logits(inputs, mask), labels).item()
    assert abs(loss - math.log(vocab_size)) < 1.0, loss


def test_embeddings_start_small_and_the_padding_row_at_zero():
    """The value itself, not only its effect: the positions never reach the
    MLM logits, so the loss test alone would not notice them starting large."""
    config = EncoderConfig(vocab_size=500, pad_id=3, max_len=512, d_model=128)
    model = FunctionEncoder(config)
    tokens = model.tokens.weight.detach()
    others = torch.cat([tokens[:3], tokens[4:]])

    assert torch.count_nonzero(tokens[3]) == 0
    assert abs(others.std().item() - 0.02) < 0.002
    assert abs(model.positions.weight.detach().std().item() - 0.02) < 0.002


def test_padding_stays_harmless_after_training_moves_the_padding_row():
    """The padding row starts at zero but does not stay there: padding_idx
    stops the gradient of the input lookup only, and the tied MLM head reaches
    the same row from the output side (it learns never to predict [PAD]).
    What protects a function's vector is the attention mask, so check it with
    a padding row that has grown as large as any other."""
    model = tiny()
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.1)
    ids, mask = pad([[1, 5, 6], [1, 7]], 0)
    for _ in range(20):
        optimiser.zero_grad()
        model.mlm_logits(ids, mask).sum().backward()
        optimiser.step()
    assert model.tokens.weight[0].norm() > 1.0, "the padding row did not move"

    model.eval()
    short = [1, 7, 8, 9]
    with torch.no_grad():
        alone = model.embed(*pad([short], 0))[0]
        in_batch = model.embed(*pad([short, [1] + list(range(10, 30))], 0))[0]
    assert torch.allclose(alone, in_batch, atol=1e-5)


def test_unknown_pooling_is_refused():
    with pytest.raises(ValueError):
        FunctionEncoder(EncoderConfig(vocab_size=10, pooling="max"))


@pytest.mark.parametrize("fields, message", [
    (dict(d_model=256, n_heads=3), "divisible"),
    (dict(n_layers=0), "n_layers"),
    (dict(d_ff=-1), "d_ff"),
    (dict(embed_dim=12.5), "embed_dim"),
    (dict(max_len=0), "max_len"),
    (dict(dropout=1.0), "dropout"),
    (dict(dropout=-0.1), "dropout"),
])
def test_an_impossible_size_is_refused_by_name_before_any_layer_is_built(fields,
                                                                          message):
    with pytest.raises(ValueError, match=message):
        EncoderConfig(vocab_size=10, **fields)


def test_every_configuration_field_is_either_shape_or_free():
    """A new field must be classified, or --init would silently ignore it."""
    import dataclasses

    from elenchus.encoder.model import FREE_FIELDS, SHAPE_FIELDS

    names = {f.name for f in dataclasses.fields(EncoderConfig)}
    assert set(SHAPE_FIELDS) | set(FREE_FIELDS) == names
    assert not set(SHAPE_FIELDS) & set(FREE_FIELDS)


def test_a_different_head_count_loads_silently_and_computes_something_else():
    """Why n_heads counts as shape: PyTorch does not refuse it."""
    a = tiny()
    b = FunctionEncoder(dataclasses_replace(a.config, n_heads=4)).eval()
    b.load_state_dict(a.state_dict())  # no error
    ids, mask = pad([[1, 5, 6, 7, 8]], 0)
    with torch.no_grad():
        assert not torch.allclose(a.embed(ids, mask), b.embed(ids, mask), atol=1e-4)


def dataclasses_replace(config, **changes):
    import dataclasses

    return dataclasses.replace(config, **changes)


# ---------------------------------------------------------------- mlm loss


def test_mlm_loss_ignores_positions_that_were_not_hidden():
    logits = torch.randn(1, 3, 10)
    labels = torch.tensor([[IGNORE_INDEX, 4, IGNORE_INDEX]])
    expected = torch.nn.functional.cross_entropy(logits[0, 1:2], torch.tensor([4]))
    assert torch.allclose(mlm_loss(logits, labels), expected)


# ---------------------------------------------------------------- info_nce


def unit(*rows):
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float), dim=-1)


def test_perfectly_matched_pairs_give_a_low_loss_and_swapped_pairs_a_high_one():
    q = unit([1, 0], [0, 1])
    assert info_nce(q, q, 0.1) < 0.01
    assert info_nce(q, unit([0, 1], [1, 0]), 0.1) > 5


def test_a_removed_false_negative_contributes_nothing():
    """Same as if that key were not in the batch at all."""
    q = unit([1, 0], [0.9, 0.1], [0, 1])
    k = unit([1, 0.1], [1, 0], [0, 1])
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 1] = True  # key 1 is the same code as query 0's positive

    masked = info_nce(q[:1], k, 0.2, false_negatives=mask[:1])
    without = torch.nn.functional.cross_entropy(
        (q[:1] @ k[[0, 2]].T) / 0.2, torch.tensor([0]))
    assert torch.allclose(masked, without, atol=1e-6)


def test_a_mask_of_the_wrong_shape_is_refused_not_broadcast():
    """The bug this test file found: a batch-square guard broadcast over a
    wider key set and silently disabled the whole mask."""
    q = unit([1, 0])
    with pytest.raises(ValueError, match="false_negatives"):
        info_nce(q, unit([1, 0], [0, 1]), 0.1,
                 false_negatives=torch.zeros(1, 1, dtype=torch.bool))


def test_the_true_key_is_never_removed_even_if_marked():
    q = unit([1, 0], [0, 1])
    everything = torch.ones(2, 2, dtype=torch.bool)
    loss = info_nce(q, q, 0.1, false_negatives=everything)
    assert math.isfinite(loss.item())
    assert loss < 1e-3


def test_queue_negatives_raise_the_loss_and_excluded_ones_do_not():
    q, k = unit([1, 0]), unit([1, 0.2])
    queue = unit([0.95, 0.1], [0, 1])
    base = info_nce(q, k, 0.1)
    with_queue = info_nce(q, k, 0.1, queue=queue)
    excluded = info_nce(q, k, 0.1, queue=queue,
                        queue_excluded=torch.tensor([[True, True]]))
    assert with_queue > base
    assert torch.allclose(excluded, base, atol=1e-6)


def test_temperature_must_be_positive():
    q = unit([1, 0])
    with pytest.raises(ValueError):
        info_nce(q, q, 0)


# ---------------------------------------------------------------- queue


def test_the_queue_forgets_the_oldest_keys_first():
    queue = MemoryQueue(size=3, dim=2)
    for step in range(5):
        queue.enqueue(torch.tensor([[float(step), 0.0]]), [f"id{step}"], [f"c{step}"])
    assert queue.filled == 3
    assert sorted(queue.keys[:, 0].tolist()) == [2.0, 3.0, 4.0]
    # The three survivors are the last three enqueued, by identity as well.
    kept = queue.excluded(["id0"], ["x"], ["y"])[0]
    assert not kept.any(), "id0 was forgotten, so nothing should match it"
    assert queue.excluded(["id4"], ["x"], ["y"])[0].sum() == 1


def test_empty_slots_and_the_same_function_are_excluded():
    queue = MemoryQueue(size=4, dim=2)
    queue.enqueue(unit([1, 0], [0, 1]), ["f", "g"], ["cf", "cg"])
    mask = queue.excluded(identities=["f", "h"], query_contents=["qf", "cg"],
                          key_contents=["kf", "kh"])
    # Row 0 (function f): its own old key is out; g stays; empty slots out.
    assert mask[0].tolist() == [True, False, True, True]
    # Row 1 (function h): g's code equals h's query code, so g is out too.
    assert mask[1].tolist() == [False, True, True, True]


def test_enqueued_keys_carry_no_gradient():
    queue = MemoryQueue(size=2, dim=2)
    key = torch.ones(1, 2, requires_grad=True)
    queue.enqueue(key * 2, ["f"], ["c"])
    assert not queue.keys.requires_grad


# ---------------------------------------------------------------- momentum


def test_the_momentum_copy_is_frozen_and_moves_by_the_asked_amount():
    model = tiny()
    target = momentum_copy(model)
    assert all(not p.requires_grad for p in target.parameters())

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = [p.clone() for p in target.parameters()]
    momentum_update(model, target, 0.9)
    for old, new, online in zip(before, target.parameters(), model.parameters()):
        assert torch.allclose(new, 0.9 * old + 0.1 * online, atol=1e-6)


def test_momentum_outside_zero_to_one_is_refused():
    model = tiny()
    with pytest.raises(ValueError):
        momentum_update(model, momentum_copy(model), 1.5)


# ---------------------------------------------------------------- scorer


def sample(listing):
    return types.SimpleNamespace(listing=listing)


def listing(*mnemonics):
    return "\n".join(f"{i:x}\t{m}\tRAX RBX\t" for i, m in enumerate(mnemonics))


def test_the_scorer_scores_every_pool_entry_and_ranks_identical_code_first():
    functions = [(("p", "f", str(i)), [m, "REG64"]) for i, m in
                 enumerate(["push", "xor", "imul", "shl", "add", "sub"] * 2)]
    vocab = build_vocab(functions, min_functions=1)
    model = FunctionEncoder(EncoderConfig(vocab_size=len(vocab), max_len=32,
                                          d_model=32, n_layers=2, n_heads=2,
                                          d_ff=64, dropout=0.0, embed_dim=16))
    scorer = EncoderScorer(model, vocab, max_len=32, batch_size=2)
    pool = [sample(listing("push", "xor")), sample(listing("imul", "shl", "add")),
            sample(listing("sub"))]
    scorer.prepare(pool)

    scores = scorer.scores(sample(listing("imul", "shl", "add")))
    assert len(scores) == 3
    assert max(range(3), key=scores.__getitem__) == 1


def varied(count, seed):
    import random

    rng = random.Random(seed)
    names = ["push", "xor", "imul", "shl", "add", "sub", "lea", "cmp"]
    return [sample(listing(*[rng.choice(names) for _ in range(rng.randint(1, 30))]))
            for _ in range(count)]


def scorer_with_counter(batch_size=4):
    names = ["push", "xor", "imul", "shl", "add", "sub", "lea", "cmp"] * 2
    functions = [(("p", "f", str(i)), [m, "REG64"]) for i, m in enumerate(names)]
    vocab = build_vocab(functions, min_functions=1)
    torch.manual_seed(0)
    model = FunctionEncoder(EncoderConfig(vocab_size=len(vocab), max_len=32,
                                          d_model=32, n_layers=2, n_heads=2,
                                          d_ff=64, dropout=0.0, embed_dim=16))
    calls = []
    real = model.embed
    model.embed = lambda ids, mask: calls.append(ids.shape) or real(ids, mask)
    return EncoderScorer(model, vocab, max_len=32, batch_size=batch_size), calls


def test_queries_are_embedded_in_batches_not_one_forward_pass_each():
    """The fix for the allocator filling the card during validation: 1,643
    single passes of 1,643 lengths became a handful of batches."""
    from elenchus.evaluation.metrics import evaluate

    scorer, calls = scorer_with_counter(batch_size=4)
    pool, queries = varied(10, 1), varied(9, 2)
    evaluate(scorer, queries, pool, [{0}] * len(queries))
    assert len(calls) == 3 + 3  # ceil(10 / 4) pool batches, ceil(9 / 4) query batches


def test_batched_queries_score_like_single_ones():
    from elenchus.evaluation.metrics import evaluate

    pool, queries = varied(12, 3), varied(11, 4)
    gold = [{i % 12} for i in range(len(queries))]
    batched, _ = scorer_with_counter()
    single, _ = scorer_with_counter()
    single.prepare(pool)
    expected = [single.scores(query) for query in queries]  # nothing prepared

    batched.prepare(pool)
    batched.prepare_queries(queries)
    for query, want in zip(queries, expected):
        assert batched.scores(query) == pytest.approx(want, abs=1e-5)
    assert evaluate(batched, queries, pool, gold)["recall@10"] == pytest.approx(
        evaluate(single, queries, pool, gold)["recall@10"])


def test_a_query_that_was_not_prepared_is_embedded_on_its_own():
    scorer, calls = scorer_with_counter()
    pool, prepared, stranger = varied(4, 5), varied(3, 6), varied(1, 7)[0]
    scorer.prepare(pool)
    scorer.prepare_queries(prepared)
    before = len(calls)
    scorer.scores(prepared[0])
    assert len(calls) == before, "a prepared query must not be embedded again"
    scorer.scores(stranger)
    assert len(calls) == before + 1


def test_a_new_pool_forgets_the_old_queries():
    """A vector cached for one evaluation must not answer in the next."""
    scorer, calls = scorer_with_counter()
    queries = varied(2, 8)
    scorer.prepare(varied(4, 9))
    scorer.prepare_queries(queries)
    scorer.prepare(varied(4, 10))
    before = len(calls)
    scorer.scores(queries[0])
    assert len(calls) == before + 1


def test_a_cached_vector_is_only_returned_for_the_object_it_was_made_from():
    scorer, _calls = scorer_with_counter()
    query, impostor = varied(2, 11)
    scorer.prepare(varied(4, 12))
    scorer.prepare_queries([query])
    # Simulate a reused id: the cache entry under impostor's id holds query.
    scorer._queries[id(impostor)] = (query, scorer._queries[id(query)][1])
    fresh, _ = scorer_with_counter()
    fresh.prepare(varied(4, 12))
    assert scorer.scores(impostor) == pytest.approx(fresh.scores(impostor), abs=1e-5)


def test_batches_are_formed_shortest_first_and_run_longest_first():
    """Longest first lets later batches reuse the allocator's blocks; the
    batches themselves must not change, or the vectors would."""
    scorer, calls = scorer_with_counter(batch_size=4)
    samples = varied(10, 13)
    lengths = [len(scorer._ids(s)) for s in samples]
    ascending = sorted(range(10), key=lambda i: lengths[i])
    formed = [ascending[0:4], ascending[4:8], ascending[8:10]]

    vectors = scorer.embed(samples)

    assert [shape[0] for shape in calls] == [2, 4, 4]
    assert [shape[1] for shape in calls] == [
        max(lengths[i] for i in chunk) for chunk in reversed(formed)]

    # Bit-identical to walking the same batches shortest first.
    with torch.no_grad():
        scorer.model.eval()
        expected = [None] * 10
        for chunk in formed:
            ids, mask = pad([scorer._ids(samples[i]) for i in chunk], scorer.vocab.pad_id)
            for row, index in zip(range(len(chunk)), chunk):
                expected[index] = scorer.model.embed(ids, mask)[row]
    assert torch.equal(vectors, torch.stack(expected))


def test_scoring_before_prepare_is_refused():
    vocab = build_vocab([(("p", "f", "g"), ["push"])], min_functions=1)
    scorer = EncoderScorer(tiny(vocab_size=len(vocab)), vocab)
    with pytest.raises(RuntimeError):
        scorer.scores(sample(listing("push")))


def test_the_scorer_leaves_the_model_in_training_mode_if_it_found_it_so():
    vocab = build_vocab([(("p", "f", "g"), ["push"])], min_functions=1)
    model = tiny(vocab_size=len(vocab)).train()
    EncoderScorer(model, vocab, max_len=32).embed([sample(listing("push"))])
    assert model.training
