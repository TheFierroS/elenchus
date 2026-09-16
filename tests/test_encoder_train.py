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
    ConfigConflict,
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
        "SELECT split, method FROM measurements WHERE run_id = ? ORDER BY method",
        (summary["run_id"],)).fetchall()
    assert [(m["split"], m["method"]) for m in measured] == [
        ("val", f"encoder-{summary['run_id']}"),
        ("val", f"encoder-{summary['run_id']}-start")]


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
    losses = [h["train_loss"] for h in summary["history"] if h["epoch"] >= 0]
    assert len(losses) == 15
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

    stored = corpus.execute(
        "SELECT metrics FROM measurements WHERE run_id = ? AND method = ?",
        (summary["run_id"], f"encoder-{summary['run_id']}")).fetchone()[0]
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


# ---------------------------------------------------------------- model size


def train_runs(conn):
    return conn.execute("SELECT COUNT(*) FROM runs WHERE kind = 'train'").fetchone()[0]


def recorded_params(conn, run_id):
    row = conn.execute("SELECT params FROM runs WHERE id = ?", (run_id,)).fetchone()
    return json.loads(row[0])


def test_a_new_model_takes_the_sizes_asked_for(corpus, vocab, tmp_path):
    asked = dict(d_model=48, n_layers=2, n_heads=4, d_ff=96, embed_dim=12,
                 dropout=0.0, pooling="cls")
    summary = train(corpus, vocab, settings(tmp_path, "mlm", max_steps=1, epochs=1),
                    log=quiet, overrides=asked)
    model, _, _ = load_checkpoint(summary["checkpoint"], vocab=vocab)
    for name, value in asked.items():
        assert getattr(model.config, name) == value, name
    assert model.config.max_len == 32
    assert recorded_params(corpus, summary["run_id"])["config"]["d_model"] == 48


def test_max_len_left_unset_is_the_models_own(corpus, vocab, tmp_path):
    summary = train(corpus, vocab,
                    settings(tmp_path, "mlm", max_len=None, max_steps=1, epochs=1),
                    config(vocab), log=quiet)
    params = recorded_params(corpus, summary["run_id"])
    assert params["settings"]["max_len"] == params["config"]["max_len"] == 32


def test_a_max_len_the_given_model_cannot_read_is_refused_before_a_run(corpus, vocab,
                                                                       tmp_path):
    with pytest.raises(ConfigConflict):
        train(corpus, vocab, settings(tmp_path, "mlm", max_len=64), config(vocab),
              log=quiet)
    assert train_runs(corpus) == 0


def test_max_len_given_twice_with_different_values_is_refused(corpus, vocab, tmp_path):
    with pytest.raises(ConfigConflict, match="twice"):
        train(corpus, vocab, settings(tmp_path, "mlm", max_len=32), log=quiet,
              overrides=dict(max_len=64))
    assert train_runs(corpus) == 0


@pytest.fixture
def pretrained(corpus, vocab, tmp_path):
    summary = train(corpus, vocab, settings(tmp_path, "mlm", max_steps=2, epochs=1),
                    config(vocab), log=quiet)
    return summary["checkpoint"]


@pytest.mark.parametrize("asked", [
    dict(d_model=64), dict(n_layers=2), dict(n_heads=1), dict(d_ff=128),
    dict(embed_dim=8), dict(max_len=16),
])
def test_init_refuses_a_shape_that_differs_from_the_checkpoint(corpus, vocab, tmp_path,
                                                               pretrained, asked):
    before = train_runs(corpus)
    s = settings(tmp_path, "contrastive", init=pretrained, max_len=None)
    with pytest.raises(ConfigConflict, match=next(iter(asked))):
        train(corpus, vocab, s, log=quiet, overrides=asked)
    assert train_runs(corpus) == before, "a refusal must not record a run"


def test_init_refuses_a_settings_max_len_the_checkpoint_cannot_read(corpus, vocab,
                                                                    tmp_path, pretrained):
    s = settings(tmp_path, "contrastive", init=pretrained, max_len=64)
    with pytest.raises(ConfigConflict, match="max_len"):
        train(corpus, vocab, s, log=quiet)


def test_init_accepts_its_own_shape_given_again(corpus, vocab, tmp_path, pretrained):
    s = settings(tmp_path, "contrastive", init=pretrained, max_steps=1, epochs=1)
    summary = train(corpus, vocab, s, log=quiet,
                    overrides=dict(d_model=32, n_heads=2, dropout=0.0))
    assert recorded_params(corpus, summary["run_id"])["init_changes"] == {}


def test_init_may_change_dropout_and_pooling_and_says_so(corpus, vocab, tmp_path,
                                                         pretrained, monkeypatch):
    weights, _, _ = load_checkpoint(pretrained, vocab=vocab)
    captured = {}
    real_optimiser = module._optimiser

    def spy(model, s):
        captured["model"] = model
        captured["state"] = {k: v.clone() for k, v in model.state_dict().items()}
        return real_optimiser(model, s)

    monkeypatch.setattr(module, "_optimiser", spy)
    lines = []
    s = settings(tmp_path, "contrastive", init=pretrained, max_steps=1, epochs=1)
    summary = train(corpus, vocab, s, log=lines.append,
                    overrides=dict(dropout=0.2, pooling="cls"))

    assert captured["model"].config.pooling == "cls"
    assert captured["model"].config.dropout == 0.2
    assert captured["model"].body.layers[0].dropout.p == 0.2
    for name, value in weights.state_dict().items():
        assert torch.equal(captured["state"][name], value), name
    assert recorded_params(corpus, summary["run_id"])["init_changes"] == {
        "dropout": [0.0, 0.2], "pooling": ["mean", "cls"]}
    assert any("pooling: mean" in line for line in lines)


def test_size_flags_reach_the_overrides_and_unset_ones_do_not():
    from elenchus import cli
    from elenchus.encoder.cli import size_overrides

    args = cli.build_parser().parse_args(
        ["train", "mlm", "--out", "m", "--d-model", "384", "--layers", "6",
         "--heads", "6", "--d-ff", "1536", "--embed-dim", "96", "--dropout", "0.2",
         "--pooling", "cls"])
    assert size_overrides(args) == dict(d_model=384, n_layers=6, n_heads=6, d_ff=1536,
                                        embed_dim=96, dropout=0.2, pooling="cls")
    bare = cli.build_parser().parse_args(["train", "mlm", "--out", "m"])
    assert size_overrides(bare) == {}
    assert bare.max_len is None


def test_the_defaults_in_the_size_help_are_the_real_defaults():
    """The help text states defaults it cannot import; keep them honest."""
    import re

    from elenchus import cli
    from elenchus.encoder.cli import SIZE_FLAGS

    parser = cli.build_parser()
    train_parser = next(a for a in parser._subparsers._group_actions[0].choices.values()
                        if a.prog.endswith(" train"))
    defaults = EncoderConfig(vocab_size=1)
    for action in train_parser._actions:
        flag = action.dest
        if flag in SIZE_FLAGS:
            stated = re.search(r"\(([^()]*)\)$", action.help).group(1)
            assert stated == str(getattr(defaults, SIZE_FLAGS[flag])), flag


def test_the_train_command_refuses_a_conflict_and_says_why(corpus, vocab, tmp_path,
                                                          pretrained, monkeypatch,
                                                          capsys):
    from elenchus import cli

    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)
    monkeypatch.setattr("elenchus.encoder.cli.connect", lambda _p: corpus)
    args = cli.build_parser().parse_args(
        ["train", "contrastive", "--out", str(tmp_path / "c"), "--init", pretrained,
         "--vocab", str(vocab_path), "--d-model", "64", "--device", "cpu"])
    args.db = ":memory:"
    assert args.func(args) == 1
    assert "d_model 32 in the checkpoint, 64 asked" in capsys.readouterr().out


# ---------------------------------------------------------------- gpu memory


def test_off_a_gpu_memory_is_recorded_as_none_never_zero(corpus, vocab, tmp_path):
    lines = []
    summary = train(corpus, vocab, settings(tmp_path, "mlm"), config(vocab),
                    log=lines.append)
    start, *trained = summary["history"]
    assert start["train_memory"] is None, "nothing was trained before the start"
    assert start["val_memory"] == {"allocated_gib": None, "reserved_gib": None}
    assert trained
    for entry in trained:
        assert entry["train_memory"] == {"allocated_gib": None, "reserved_gib": None}
        assert entry["val_memory"] == {"allocated_gib": None, "reserved_gib": None}
    assert summary["peak_allocated_gib"] is None
    assert summary["peak_reserved_gib"] is None
    assert recorded_params(corpus, summary["run_id"])["gpu"] is None
    assert not any("mem" in line for line in lines)


def test_the_probe_reads_cuda_peaks_in_gib_for_its_own_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda d: calls.append(("reset", d)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                        lambda d: calls.append(("allocated", d)) or 3 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                        lambda d: calls.append(("reserved", d)) or 4 * 1024 ** 3)
    card = types.SimpleNamespace(name="Card", total_memory=8 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: card)

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    probe = module.MemoryProbe("cuda:1")
    probe.reset()
    assert probe.read() == {"allocated_gib": 3.0, "reserved_gib": 4.0}
    assert probe.describe() == {"name": "Card", "total_gib": 8.0,
                                "allocator": "expandable_segments:True"}
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    assert probe.describe()["allocator"] is None
    assert {device for _name, device in calls} == {torch.device("cuda:1")}
    assert [name for name, _device in calls] == ["reset", "allocated", "reserved"]


class FakeProbe:
    """Stands in for a GPU: every read is a new, larger peak."""

    events = []

    def __init__(self, device):
        self.readings = 0

    def reset(self):
        FakeProbe.events.append("reset")

    def read(self):
        self.readings += 1
        FakeProbe.events.append("read")
        return {"allocated_gib": float(self.readings),
                "reserved_gib": self.readings + 0.5}

    def describe(self):
        return {"name": "Fake", "total_gib": 8.0}


def test_each_epoch_measures_training_and_validation_apart(corpus, vocab, tmp_path,
                                                          monkeypatch):
    FakeProbe.events = []
    monkeypatch.setattr(module, "MemoryProbe", FakeProbe)
    real_step, real_evaluate = module._step, module.evaluate
    monkeypatch.setattr(module, "_step", lambda *a, **k: (
        FakeProbe.events.append("step"), real_step(*a, **k))[1])
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: (
        FakeProbe.events.append("eval"), real_evaluate(*a, **k))[1])

    lines = []
    summary = train(corpus, vocab, settings(tmp_path, "contrastive", epochs=2),
                    config(vocab), log=lines.append)

    # The start: reset, validation, read. Per epoch: reset, the steps, read |
    # reset, validation, read. (The kept checkpoint's final measurement
    # comes after, outside the probe.)
    compact = []
    for event in FakeProbe.events:
        if not (event == "step" and compact and compact[-1] == "step"):
            compact.append(event)
    epoch = ["reset", "step", "read", "reset", "eval", "read"]
    assert compact[:15] == ["reset", "eval", "read"] + epoch * 2

    start, first, second = summary["history"]
    assert start["train_memory"] is None
    assert start["val_memory"] == {"allocated_gib": 1.0, "reserved_gib": 1.5}
    assert first["train_memory"] == {"allocated_gib": 2.0, "reserved_gib": 2.5}
    assert first["val_memory"] == {"allocated_gib": 3.0, "reserved_gib": 3.5}
    assert second["val_memory"] == {"allocated_gib": 5.0, "reserved_gib": 5.5}
    assert summary["peak_allocated_gib"] == 5.0
    assert summary["peak_reserved_gib"] == 5.5
    assert recorded_params(corpus, summary["run_id"])["gpu"] == {"name": "Fake",
                                                                 "total_gib": 8.0}
    assert "mem val 1.00/1.50 GiB" in lines[1] and "mem train" not in lines[1]
    assert "mem train 2.00/2.50 val 3.00/3.50 GiB" in lines[2]
    written = json.loads((tmp_path / "contrastive" / "summary.json").read_text())
    assert written["peak_allocated_gib"] == 5.0
    assert written["history"][1]["val_memory"]["reserved_gib"] == 3.5


def test_the_train_command_prints_the_peak_only_when_there_is_one(monkeypatch, capsys,
                                                                 tmp_path):
    from elenchus import cli
    from elenchus.encoder import vocab as vocab_module

    monkeypatch.setattr("elenchus.encoder.cli.connect", lambda _p: None)
    monkeypatch.setattr(vocab_module.Vocab, "load", staticmethod(lambda _p: None))
    args = cli.build_parser().parse_args(["train", "mlm", "--out", str(tmp_path)])
    args.db = ":memory:"
    base = {"best_epoch": 0, "best_val_loss": 2.0, "checkpoint": "x/best.pt"}

    for peak, shown in ((None, False), (2.25, True)):
        summary = base | {"peak_allocated_gib": peak,
                          "peak_reserved_gib": None if peak is None else 3.5}
        monkeypatch.setattr(module, "train", lambda *a, _s=summary, **k: _s)
        assert args.func(args) == 0
        out = capsys.readouterr().out
        assert ("peak memory : 2.25 GiB allocated, 3.50 GiB reserved" in out) == shown
        assert ("peak memory" in out) == shown


# ---------------------------------------------------------------- the start


def test_the_start_is_measured_before_the_first_step(corpus, vocab, tmp_path,
                                                     monkeypatch):
    order = []
    real_step, real_evaluate = module._step, module.evaluate
    monkeypatch.setattr(module, "_step", lambda *a, **k: (
        order.append("step"), real_step(*a, **k))[1])
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: (
        order.append("eval"), real_evaluate(*a, **k))[1])
    lines = []
    summary = train(corpus, vocab, settings(tmp_path, "contrastive", epochs=1),
                    config(vocab), log=lines.append)

    assert order[0] == "eval" and order[1] == "step"
    assert summary["history"][0]["epoch"] == -1
    assert summary["history"][0]["mrr"] == summary["start_val_mrr"]
    assert lines[1].startswith("start ")


def test_the_start_score_belongs_to_the_starting_weights(corpus, vocab, tmp_path,
                                                        pretrained):
    """Started from a checkpoint, the start is that checkpoint's own score."""
    from elenchus.evaluation.task import retrieval_task

    summary = train(corpus, vocab,
                    settings(tmp_path, "contrastive", init=pretrained, epochs=1),
                    log=quiet)
    queries, pool, gold = retrieval_task(corpus, split="val")
    expected = measure_checkpoint(pretrained, vocab, queries, pool, gold)

    assert summary["start_val_metrics"] == pytest.approx(expected)
    stored = corpus.execute(
        "SELECT metrics FROM measurements WHERE run_id = ? AND method = ?",
        (summary["run_id"], f"encoder-{summary['run_id']}-start")).fetchone()[0]
    assert json.loads(stored) == pytest.approx(expected)


def test_the_start_is_never_kept_as_the_best_checkpoint(corpus, vocab, tmp_path,
                                                        monkeypatch):
    """Even when training only makes things worse, best.pt is a trained epoch:
    the start is a reference, and its weights are not the run's to keep."""
    scores = iter([0.9, 0.1, 0.2])
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: {
        "queries": 1, "pool": 1, "recall@1": 0.0, "recall@10": 0.0,
        "mrr": next(scores, 0.2), "median_rank": 1})
    summary = train(corpus, vocab, settings(tmp_path, "contrastive", epochs=2),
                    config(vocab), log=quiet)
    assert summary["start_val_mrr"] == 0.9
    assert summary["best_epoch"] == 1 and summary["best_val_mrr"] == 0.2
    kept = torch.load(tmp_path / "contrastive" / "best.pt", weights_only=False)
    assert kept["meta"]["epoch"] == 1


@pytest.mark.parametrize("stage, metric", [("mlm", "loss"), ("contrastive", "mrr")])
def test_max_steps_zero_measures_the_start_and_trains_nothing(corpus, vocab, tmp_path,
                                                             monkeypatch, stage, metric):
    """0 used to read as "no limit" and trained a full run."""
    steps = []
    real_step = module._step
    monkeypatch.setattr(module, "_step", lambda *a, **k: (
        steps.append(1), real_step(*a, **k))[1])
    summary = train(corpus, vocab, settings(tmp_path, stage, max_steps=0),
                    config(vocab), log=quiet)

    assert steps == []
    assert summary["checkpoint"] is None and summary["best_epoch"] is None
    assert summary[f"start_val_{metric}"] is not None
    assert [h["epoch"] for h in summary["history"]] == [-1]
    assert not (tmp_path / stage / "best.pt").exists()
    assert not (tmp_path / stage / "last.pt").exists()
    run = corpus.execute("SELECT status FROM runs WHERE id = ?",
                         (summary["run_id"],)).fetchone()
    assert run["status"] == "ok"
    methods = [r["method"] for r in corpus.execute(
        "SELECT method FROM measurements WHERE run_id = ?", (summary["run_id"],))]
    assert methods == ([f"encoder-{summary['run_id']}-start"]
                       if stage == "contrastive" else [])


def test_a_step_cap_stops_exactly_there(corpus, vocab, tmp_path, monkeypatch):
    steps = []
    real_step = module._step
    monkeypatch.setattr(module, "_step", lambda *a, **k: (
        steps.append(1), real_step(*a, **k))[1])
    summary = train(corpus, vocab, settings(tmp_path, "mlm", max_steps=3, epochs=5),
                    config(vocab), log=quiet)
    assert len(steps) == 3
    assert [h["epoch"] for h in summary["history"]] == [-1, 0]


def test_a_negative_step_cap_is_refused():
    with pytest.raises(ValueError, match="max_steps"):
        Settings(stage="mlm", out="x", max_steps=-1)


def test_the_train_command_reports_the_start_and_an_untrained_run(monkeypatch, capsys,
                                                                  tmp_path):
    from elenchus import cli
    from elenchus.encoder import vocab as vocab_module

    monkeypatch.setattr("elenchus.encoder.cli.connect", lambda _p: None)
    monkeypatch.setattr(vocab_module.Vocab, "load", staticmethod(lambda _p: None))
    args = cli.build_parser().parse_args(["train", "contrastive", "--out", str(tmp_path)])
    args.db = ":memory:"
    start = {"queries": 5, "pool": 5, "recall@1": 0.2, "recall@10": 0.4, "mrr": 0.3,
             "median_rank": 2}
    untrained = {"start_val_mrr": 0.3, "start_val_metrics": start, "best_epoch": None,
                 "best_val_mrr": None, "checkpoint": None, "peak_allocated_gib": None}
    monkeypatch.setattr(module, "train", lambda *a, **k: untrained)

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "start_val_mrr: 0.3000  (before training)" in out
    assert "start val   : recall@1 0.200  recall@10 0.400  mrr 0.300" in out
    assert "trained     : nothing (--max-steps 0)" in out
    assert "checkpoint" not in out and "best epoch" not in out


def test_the_train_command_parses_its_stages():
    from elenchus import cli

    args = cli.build_parser().parse_args(
        ["train", "contrastive", "--out", "models/c", "--init", "models/m/best.pt",
         "--max-steps", "5"])
    assert (args.stage, args.init, args.max_steps) == (
        "contrastive", "models/m/best.pt", 5)
    assert args.func.__name__ == "cmd_train"
