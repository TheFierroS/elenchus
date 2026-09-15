"""End-to-end tests for training, on a tiny database and a tiny model, on CPU.

These train for real, briefly. They check what a long run on a GPU cannot be
trusted to show by itself: that runs are recorded, that the test split is
never read, that a stale vocabulary is refused, that checkpoints come back
identical, and that the model actually learns on data where learning is easy.
"""

import json
import types

import pytest
import torch

from elenchus.corpus.dataset import store_splits
from elenchus.encoder import train as module
from elenchus.encoder.batching import pad
from elenchus.encoder.model import EncoderConfig
from elenchus.encoder.train import (
    Settings,
    StaleVocabulary,
    learning_rate,
    load_checkpoint,
    measure_checkpoint,
    train,
)
from elenchus.encoder.vocab import build_vocab, train_functions
from elenchus.evaluation.store import dataset_fingerprint

MNEMONICS = ["push", "pop", "xor", "imul", "shl", "shr", "add", "sub", "lea", "cmp",
             "test", "and", "or", "not", "neg", "inc", "dec", "sar", "rol", "ror"]


def add_function(conn, package, name, level, body):
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO binaries (path, sha256, arch, imported_at) "
            "VALUES (?, ?, 'x', datetime('now'))",
            (f"/{package}{level}", package + level))
        binary_id = conn.execute("SELECT id FROM binaries WHERE sha256 = ?",
                                 (package + level,)).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO corpus_binaries (binary_id, package, compiler, "
            "opt_level, stripped) VALUES (?, ?, 'gcc', ?, 1)",
            (binary_id, package, level))
        count = conn.execute("SELECT COUNT(*) FROM functions WHERE binary_id = ?",
                             (binary_id,)).fetchone()[0]
        fid = conn.execute(
            "INSERT INTO functions (binary_id, address, size, raw_name) "
            "VALUES (?, ?, 32, 'F')", (binary_id, 0x1000 + count)).lastrowid
        conn.execute(
            "INSERT INTO ground_truth (binary_id, function_id, address, name, "
            "return_type, param_types, decl_file, decl_line) "
            "VALUES (?, ?, ?, ?, 'int', '[]', ?, 1)",
            (binary_id, fid, 0x1000 + count, name, f"src/{package}.c"))
        listing = "\n".join(f"{i:x}\t{m}\tRAX RBX\t" for i, m in enumerate(body))
        conn.execute("INSERT INTO function_code VALUES (?, ?, ?, 32, ?, ?)",
                     (fid, binary_id, len(body), f"h{fid}", listing))


@pytest.fixture
def corpus(db):
    """Functions whose -O3 form is a recognisable rewrite of their -O0 form."""
    splits = {}
    for p, split in enumerate(["train", "train", "train", "val", "test"]):
        package = f"pkg{p}"
        splits[package] = split
        for f in range(8):
            # A (package, function) signature keeps every body distinct: the
            # dataset drops code that appears in more than one package.
            signature = [MNEMONICS[p], MNEMONICS[5 + f], MNEMONICS[(13 + p + f) % 20]]
            base = signature + [MNEMONICS[(p * 7 + f * 3 + k) % 20] for k in range(7)]
            add_function(db, package, f"fn{f}", "O0", base + ["push"] * 2)
            add_function(db, package, f"fn{f}", "O3", base[:9] + ["ret"])
    store_splits(db, splits)
    return db


@pytest.fixture
def vocab(corpus):
    functions = train_functions(corpus)
    return build_vocab(functions, min_functions=1,
                       source={"dataset_fingerprint": dataset_fingerprint(corpus)})


CONFIG = dict(max_len=32, d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0,
              embed_dim=16)


def settings(tmp_path, stage, **overrides):
    values = dict(stage=stage, out=str(tmp_path / stage), epochs=2, batch_size=4,
                  per_package=2, queue_size=8, warmup_steps=2, max_len=32,
                  patience=5, device="cpu", amp=False, lr=1e-3)
    return Settings(**(values | overrides))


def config(vocab):
    return EncoderConfig(vocab_size=len(vocab), pad_id=vocab.pad_id, **CONFIG)


def quiet(*_args, **_kwargs):
    pass


# ---------------------------------------------------------------- contrastive


def test_contrastive_training_runs_records_and_checkpoints(corpus, vocab, tmp_path):
    summary = train(corpus, vocab, settings(tmp_path, "contrastive"),
                    config(vocab), log=quiet)

    run = corpus.execute("SELECT kind, status, params FROM runs WHERE id = ?",
                         (summary["run_id"],)).fetchone()
    assert (run["kind"], run["status"]) == ("train", "ok")
    params = json.loads(run["params"])
    assert params["dataset"] == dataset_fingerprint(corpus)
    assert params["stage"] == "contrastive"

    assert (tmp_path / "contrastive" / "best.pt").exists()
    assert (tmp_path / "contrastive" / "summary.json").exists()
    measured = corpus.execute(
        "SELECT split, method FROM measurements WHERE run_id = ?",
        (summary["run_id"],)).fetchall()
    assert [(m["split"], m["method"]) for m in measured] == [
        ("val", f"encoder-{summary['run_id']}")]


def test_training_never_reads_the_test_split(corpus, vocab, tmp_path, monkeypatch):
    asked = []
    real_task, real_examples = module.retrieval_task, module.load_examples

    def spy_task(conn, split="test", *a, **k):
        asked.append(split)
        return real_task(conn, split, *a, **k)

    def spy_examples(conn, split="train", *a, **k):
        asked.append(split)
        return real_examples(conn, split, *a, **k)

    monkeypatch.setattr(module, "retrieval_task", spy_task)
    monkeypatch.setattr(module, "load_examples", spy_examples)
    train(corpus, vocab, settings(tmp_path, "contrastive"), config(vocab), log=quiet)
    train(corpus, vocab, settings(tmp_path, "mlm"), config(vocab), log=quiet)

    assert "test" not in asked
    assert set(asked) == {"train", "val"}


def test_a_vocabulary_from_another_dataset_is_refused(corpus, vocab, tmp_path):
    stale = build_vocab([(("p", "f", "g"), list(vocab.tokens[5:]))], min_functions=1,
                        source={"dataset_fingerprint": "someothercorpus"})
    with pytest.raises(StaleVocabulary):
        train(corpus, stale, settings(tmp_path, "contrastive"), log=quiet)
    trained = corpus.execute("SELECT COUNT(*) FROM runs WHERE kind = 'train'")
    assert trained.fetchone()[0] == 0


def test_a_run_that_fails_is_recorded_as_failed(corpus, vocab, tmp_path, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("loss exploded")

    monkeypatch.setattr(module, "info_nce", broken)
    with pytest.raises(RuntimeError):
        train(corpus, vocab, settings(tmp_path, "contrastive"), config(vocab), log=quiet)
    status = corpus.execute("SELECT status FROM runs WHERE kind = 'train'").fetchone()[0]
    assert status == "failed"


def test_the_model_learns_when_learning_is_easy(corpus, vocab, tmp_path):
    """Not a quality claim - a check that gradients flow and the loss moves."""
    torch.manual_seed(0)
    summary = train(corpus, vocab,
                    settings(tmp_path, "contrastive", epochs=15, queue_size=0,
                             patience=15),
                    config(vocab), log=quiet)
    losses = [h["train_loss"] for h in summary["history"]]
    assert losses[-1] < losses[0] * 0.7, losses


# ---------------------------------------------------------------- mlm and init


def test_mlm_then_contrastive_starts_from_the_pretrained_weights(corpus, vocab, tmp_path):
    mlm = train(corpus, vocab, settings(tmp_path, "mlm"), config(vocab), log=quiet)
    pretrained, _, _ = load_checkpoint(mlm["checkpoint"], vocab=vocab)

    captured = {}
    real_optimiser = module._optimiser

    def spy(model, s):
        captured["state"] = {k: v.clone() for k, v in model.state_dict().items()}
        return real_optimiser(model, s)

    module._optimiser = spy
    try:
        train(corpus, vocab, settings(tmp_path, "contrastive", init=mlm["checkpoint"]),
              log=quiet)
    finally:
        module._optimiser = real_optimiser

    for name, value in pretrained.state_dict().items():
        assert torch.equal(captured["state"][name], value), name


def test_a_checkpoint_comes_back_identical(corpus, vocab, tmp_path):
    summary = train(corpus, vocab, settings(tmp_path, "contrastive"), config(vocab),
                    log=quiet)
    model, loaded_vocab, meta = load_checkpoint(summary["checkpoint"], vocab=vocab)
    again, _, _ = load_checkpoint(summary["checkpoint"])
    assert loaded_vocab.tokens == vocab.tokens
    assert meta["stage"] == "contrastive" and meta["run_id"] == summary["run_id"]

    ids, mask = pad([[vocab.cls_id, 6, 7, 8]], vocab.pad_id)
    model.eval()
    again.eval()
    with torch.no_grad():
        assert torch.equal(model.embed(ids, mask), again.embed(ids, mask))


def test_loading_with_a_different_vocabulary_is_refused(corpus, vocab, tmp_path):
    summary = train(corpus, vocab, settings(tmp_path, "contrastive"), config(vocab),
                    log=quiet)
    other = build_vocab([(("p", "f", "g"), ["only", "these"])], min_functions=1)
    with pytest.raises(ValueError, match="different vocabulary"):
        load_checkpoint(summary["checkpoint"], vocab=other)


def test_a_trained_run_locks_the_split(corpus, vocab, tmp_path, monkeypatch):
    from elenchus import cli

    train(corpus, vocab, settings(tmp_path, "mlm"), config(vocab), log=quiet)
    monkeypatch.setattr(cli, "connect", lambda _p: corpus)
    monkeypatch.setattr(cli, "load_manifest", lambda _p: [])
    store_splits(corpus, {"pkg0": "test", "pkg1": "train", "pkg2": "train",
                          "pkg3": "val", "pkg4": "train"})
    code = cli.cmd_dataset(types.SimpleNamespace(db=":memory:", assign=True,
                                                 force=False, manifest=None))
    assert code == 1


# ---------------------------------------------------------------- schedule


def test_learning_rate_warms_up_then_decays_and_stays_positive():
    s = Settings(stage="mlm", out="x", lr=1.0, warmup_steps=10)
    rates = [learning_rate(step, s, total_steps=100) for step in range(100)]
    assert rates[0] < rates[9]
    assert rates[9] == pytest.approx(1.0)
    assert rates[99] < rates[50] < rates[10]
    assert min(rates) >= 0.1 - 1e-9


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError):
        Settings(stage="finetune", out="x")


def test_training_passes_the_false_negative_mask_to_the_loss(corpus, vocab, tmp_path,
                                                            monkeypatch):
    """The mask being correct is tested elsewhere; this checks it is used."""
    seen = []
    real = module.info_nce

    def spy(q, k, t, false_negatives=None, **kwargs):
        seen.append(false_negatives)
        return real(q, k, t, false_negatives=false_negatives, **kwargs)

    monkeypatch.setattr(module, "info_nce", spy)
    train(corpus, vocab, settings(tmp_path, "contrastive"), config(vocab), log=quiet)

    assert seen and all(isinstance(m, torch.Tensor) and m.dtype == torch.bool
                        for m in seen)


def test_the_recorded_val_score_is_the_kept_checkpoints(corpus, vocab, tmp_path):
    from elenchus.evaluation.task import retrieval_task

    summary = train(corpus, vocab, settings(tmp_path, "contrastive", epochs=3),
                    config(vocab), log=quiet)
    queries, pool, gold = retrieval_task(corpus, split="val")
    again = measure_checkpoint(summary["checkpoint"], vocab, queries, pool, gold)

    stored = corpus.execute("SELECT metrics FROM measurements WHERE run_id = ?",
                            (summary["run_id"],)).fetchone()[0]
    assert json.loads(stored) == pytest.approx(again)
    assert again["mrr"] == pytest.approx(summary["best_val_mrr"])


def test_a_trained_checkpoint_joins_the_baselines_table(corpus, vocab, tmp_path,
                                                        monkeypatch, capsys):
    from elenchus import cli

    summary = train(corpus, vocab, settings(tmp_path, "contrastive"), config(vocab),
                    log=quiet)
    monkeypatch.setattr(cli, "connect", lambda _p: corpus)
    args = cli.build_parser().parse_args(
        ["baselines", "--split", "val", "--encoder", summary["checkpoint"]])
    args.db = ":memory:"

    assert cli.cmd_baselines(args) == 0
    out = capsys.readouterr().out
    assert f"encoder (run {summary['run_id']})" in out
    assert "bm25-mnemonic" in out


def test_the_train_command_parses_its_stages():
    from elenchus import cli

    args = cli.build_parser().parse_args(
        ["train", "contrastive", "--out", "models/c", "--init", "models/m/best.pt",
         "--max-steps", "5"])
    assert (args.stage, args.init, args.max_steps) == (
        "contrastive", "models/m/best.pt", 5)
    assert args.func.__name__ == "cmd_train"
