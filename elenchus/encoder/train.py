"""Training: masked-LM pre-training, then contrastive training.

Every run is a `runs` row of kind "train", which also locks the split (see
cmd_dataset). Its params hold the model configuration, the settings, the
dataset fingerprint and a hash of the vocabulary, so a checkpoint can always
be traced to the data and the ids it was trained on.

Only train and val are read. Model selection uses val - validation loss for
masked LM, val MRR on the evaluation task for contrastive training - and the
best checkpoint by that measure is kept. The test split is never loaded
here; its single look belongs at the end, through `elenchus baselines`.

A vocabulary built from a different dataset than the one being trained on is
refused: its ids would be right for some other corpus.

The model's configuration is the one authority on its size. Settings.max_len
left as None takes the model's; given, it must agree, because the samplers
cut functions to it and the model cannot read past its own. Starting from a
checkpoint (init), a requested shape that differs from the checkpoint's is
refused before any run is recorded; dropout and pooling may differ, own no
weights, and the change is written into the run's params.
"""

import contextlib
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from elenchus.db import code_version, finish_run, start_run
from elenchus.encoder.batching import collate_masked, collate_pairs
from elenchus.encoder.data import MaskedSampler, PairSampler, load_examples
from elenchus.encoder.losses import (
    MemoryQueue,
    info_nce,
    mlm_loss,
    momentum_copy,
    momentum_update,
)
from elenchus.encoder.model import (
    FREE_FIELDS,
    SHAPE_FIELDS,
    EncoderConfig,
    FunctionEncoder,
    count_parameters,
)
from elenchus.encoder.scorer import EncoderScorer
from elenchus.encoder.vocab import Vocab
from elenchus.evaluation.metrics import evaluate
from elenchus.evaluation.store import dataset_fingerprint, record, task_params
from elenchus.evaluation.task import retrieval_task


class StaleVocabulary(ValueError):
    """The vocabulary was built from a different dataset."""


class ConfigConflict(ValueError):
    """A requested model size disagrees with the model it has to fit."""


@dataclass
class Settings:
    stage: str                      # "mlm" or "contrastive"
    out: str
    epochs: int = 20
    batch_size: int = 64
    per_package: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    temperature: float = 0.07
    queue_size: int = 4096
    momentum: float = 0.999
    mask_rate: float = 0.15
    max_len: int | None = None      # None: the model's own max_len
    grad_clip: float = 1.0
    patience: int = 3
    seed: int = 0
    init: str | None = None
    device: str = "auto"
    amp: bool = True
    max_steps: int | None = None    # for smoke tests; None trains fully

    def __post_init__(self):
        if self.stage not in ("mlm", "contrastive"):
            raise ValueError(f"unknown stage {self.stage!r}")


def vocab_hash(vocab):
    return hashlib.sha256("\n".join(vocab.tokens).encode()).hexdigest()[:16]


def check_vocabulary(conn, vocab):
    built = vocab.source.get("dataset_fingerprint")
    current = dataset_fingerprint(conn)
    if built != current:
        raise StaleVocabulary(
            f"vocabulary was built for dataset {built}, the database is {current}; "
            "run `elenchus vocab` again")


def learning_rate(step, settings, total_steps):
    """Linear warmup, then cosine decay to a tenth of the peak."""
    if step < settings.warmup_steps:
        return settings.lr * (step + 1) / settings.warmup_steps
    span = max(1, total_steps - settings.warmup_steps)
    progress = min(1.0, (step - settings.warmup_steps) / span)
    return settings.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path, model, vocab, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "vocab": vocab.to_json(),
        "vocab_hash": vocab_hash(vocab),
        "meta": meta,
    }, path)


def load_checkpoint(path, device="cpu", vocab=None):
    """Return (model, vocab, meta). With vocab given, it must be the same one."""
    data = torch.load(path, map_location=device, weights_only=False)
    stored = Vocab.from_json(data["vocab"])
    if vocab_hash(stored) != data["vocab_hash"]:
        raise ValueError(f"{path}: vocabulary does not match its recorded hash")
    if vocab is not None and vocab_hash(vocab) != data["vocab_hash"]:
        raise ValueError(f"{path} was trained with a different vocabulary")
    model = FunctionEncoder(EncoderConfig(**data["config"])).to(device)
    model.load_state_dict(data["state_dict"])
    return model, stored, data["meta"]


def measure_checkpoint(path, vocab, queries, pool, gold, device="cpu"):
    """Evaluate a checkpoint read back from disk.

    Takes a path, not a model, on purpose: the number recorded for a run must
    belong to the checkpoint that is kept, and a model still in memory after
    the last epoch is usually not that one.
    """
    model, _, _ = load_checkpoint(path, device, vocab)
    scorer = EncoderScorer(model, vocab, max_len=model.config.max_len, device=device)
    return evaluate(scorer, queries, pool, gold)


def build_model(vocab, settings, config=None, overrides=None, device="cpu"):
    """Return (model, settings with max_len resolved, init changes, init meta).

    overrides holds only the configuration fields a user actually asked for.
    Nothing is recorded here, so a refusal leaves no run behind and does not
    lock the split.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(SHAPE_FIELDS) - set(FREE_FIELDS)
    if unknown:
        raise ValueError(f"not configuration fields: {sorted(unknown)}")
    if settings.max_len is not None:
        if overrides.get("max_len", settings.max_len) != settings.max_len:
            raise ConfigConflict("max_len given twice with different values")
        overrides["max_len"] = settings.max_len

    changes = {}
    if settings.init:
        if config is not None:
            raise ConfigConflict("give a configuration or a checkpoint to start "
                                 "from, not both")
        model, _, init_meta = load_checkpoint(settings.init, device, vocab)
        stored = model.config
        conflicts = {name: (getattr(stored, name), value)
                     for name, value in overrides.items()
                     if name in SHAPE_FIELDS and getattr(stored, name) != value}
        if conflicts:
            listed = ", ".join(f"{name} {old} in the checkpoint, {new} asked"
                               for name, (old, new) in sorted(conflicts.items()))
            raise ConfigConflict(f"{settings.init} cannot take this size: {listed}")
        changes = {name: [getattr(stored, name), value]
                   for name, value in overrides.items()
                   if name in FREE_FIELDS and getattr(stored, name) != value}
        if changes:
            # dropout and pooling own no weights: rebuild, then load them all.
            rebuilt = FunctionEncoder(dataclasses.replace(
                stored, **{name: new for name, (_old, new) in changes.items()}))
            rebuilt.load_state_dict(model.state_dict())
            model = rebuilt.to(device)
    else:
        base = config or EncoderConfig(vocab_size=len(vocab), pad_id=vocab.pad_id)
        if config is not None:
            conflicts = {name: (getattr(config, name), value)
                         for name, value in overrides.items()
                         if getattr(config, name) != value}
            if conflicts:
                raise ConfigConflict(f"the given configuration disagrees: {conflicts}")
        model = FunctionEncoder(dataclasses.replace(base, **overrides)).to(device)
        init_meta = None

    if model.config.vocab_size != len(vocab):
        raise ValueError("model and vocabulary disagree on the vocabulary size")
    settings = dataclasses.replace(settings, max_len=model.config.max_len)
    return model, settings, changes, init_meta


def _device(settings):
    if settings.device != "auto":
        return settings.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast(device, settings):
    if device.startswith("cuda") and settings.amp:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(conn, vocab, settings, config=None, log=print, overrides=None):
    """Train one stage and return a summary; the best checkpoint is on disk.

    overrides: configuration fields asked for by name (see build_model).
    """
    check_vocabulary(conn, vocab)
    torch.manual_seed(settings.seed)
    device = _device(settings)
    model, settings, changes, init_meta = build_model(vocab, settings, config,
                                                      overrides, device)

    fingerprint = dataset_fingerprint(conn)
    params = {
        "stage": settings.stage,
        "settings": asdict(settings),
        "config": model.config.to_dict(),
        "dataset": fingerprint,
        "vocab": vocab_hash(vocab),
        "init": settings.init,
        "init_changes": changes,
        "parameters": count_parameters(model),
    }
    run_id = start_run(conn, "train", tool="elenchus-encoder", params=params,
                       seed=settings.seed)
    log(f"run {run_id}: {settings.stage}, {count_parameters(model):,} parameters, "
        f"device {device}")
    for name, (old, new) in changes.items():
        log(f"  {name}: {old} in {settings.init}, {new} here")

    try:
        examples = load_examples(conn, "train")
        if settings.stage == "mlm":
            summary = _train_mlm(conn, model, vocab, settings, examples, device,
                                 run_id, log)
        else:
            summary = _train_contrastive(conn, model, vocab, settings, examples,
                                         device, run_id, log)
    except BaseException:
        finish_run(conn, run_id, "failed")
        raise

    summary.update(run_id=run_id, dataset=fingerprint, init=init_meta)
    Path(settings.out).mkdir(parents=True, exist_ok=True)
    (Path(settings.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    finish_run(conn, run_id, "ok")
    return summary


def _meta(settings, run_id, epoch, metric_name, metric):
    return {"stage": settings.stage, "run_id": run_id, "epoch": epoch,
            metric_name: metric, "code_version": code_version(),
            "seed": settings.seed}


def _optimiser(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings.lr,
                             weight_decay=settings.weight_decay)


def _step(model, optimiser, loss, settings, step, total_steps):
    for group in optimiser.param_groups:
        group["lr"] = learning_rate(step, settings, total_steps)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
    optimiser.step()


def _selection_loop(settings, epoch_fn, evaluate_fn, better, metric_name,
                    model, vocab, run_id, log):
    best, best_epoch, waited = None, -1, 0
    history = []
    out = Path(settings.out)
    for epoch in range(settings.epochs):
        started = time.time()
        train_loss, steps, stop = epoch_fn(epoch)
        metric = evaluate_fn()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        metric_name: metric, "seconds": time.time() - started})
        improved = best is None or better(metric, best)
        log(f"epoch {epoch:3}  loss {train_loss:.4f}  val {metric_name} {metric:.4f}"
            f"{'  *' if improved else ''}  ({steps} steps, {time.time() - started:.0f}s)")
        meta = _meta(settings, run_id, epoch, metric_name, metric)
        save_checkpoint(out / "last.pt", model, vocab, meta)
        if improved:
            best, best_epoch, waited = metric, epoch, 0
            save_checkpoint(out / "best.pt", model, vocab, meta)
        else:
            waited += 1
            if waited >= settings.patience:
                log(f"no improvement for {settings.patience} epochs; stopping")
                break
        if stop:
            break
    return {"best_epoch": best_epoch, f"best_val_{metric_name}": best,
            "history": history, "checkpoint": str(out / "best.pt")}


def _train_mlm(conn, model, vocab, settings, examples, device, run_id, log):
    sampler = MaskedSampler(examples, vocab, batch_size=settings.batch_size,
                            max_len=settings.max_len, rate=settings.mask_rate,
                            seed=settings.seed)
    val = MaskedSampler(load_examples(conn, "val"), vocab,
                        batch_size=settings.batch_size, max_len=settings.max_len,
                        rate=settings.mask_rate, seed=settings.seed + 1)
    optimiser = _optimiser(model, settings)
    total_steps = settings.max_steps or len(sampler) * settings.epochs
    state = {"step": 0}

    def epoch_fn(epoch):
        model.train()
        losses = []
        for batch in sampler.batches(epoch):
            tensors = _to(collate_masked(batch, vocab.pad_id), device)
            with _autocast(device, settings):
                logits = model.mlm_logits(tensors["input_ids"], tensors["attention_mask"])
            loss = mlm_loss(logits, tensors["labels"])
            _step(model, optimiser, loss, settings, state["step"], total_steps)
            losses.append(loss.item())
            state["step"] += 1
            if settings.max_steps and state["step"] >= settings.max_steps:
                break
        stop = bool(settings.max_steps and state["step"] >= settings.max_steps)
        return sum(losses) / max(len(losses), 1), len(losses), stop

    @torch.no_grad()
    def evaluate_fn():
        # The same masks every time (epoch 0 of a fixed seed), so epochs are
        # compared on one question paper.
        model.eval()
        total, count = 0.0, 0
        for batch in val.batches(0):
            tensors = _to(collate_masked(batch, vocab.pad_id), device)
            with _autocast(device, settings):
                logits = model.mlm_logits(tensors["input_ids"], tensors["attention_mask"])
            total += mlm_loss(logits, tensors["labels"]).item()
            count += 1
        return total / max(count, 1)

    return _selection_loop(settings, epoch_fn, evaluate_fn, lambda a, b: a < b,
                           "loss", model, vocab, run_id, log)


def _train_contrastive(conn, model, vocab, settings, examples, device, run_id, log):
    sampler = PairSampler(examples, vocab, batch_size=settings.batch_size,
                          per_package=settings.per_package, max_len=settings.max_len,
                          seed=settings.seed)
    if len(sampler) == 0:
        raise ValueError("not enough pairable functions for one batch")
    target = momentum_copy(model)
    queue = MemoryQueue(settings.queue_size, model.config.embed_dim, device)
    optimiser = _optimiser(model, settings)
    total_steps = settings.max_steps or len(sampler) * settings.epochs
    state = {"step": 0}

    queries, pool, gold = retrieval_task(conn, split="val")
    task = task_params(conn, "val", "O0", "O3", settings.seed)
    last_metrics = {}

    def epoch_fn(epoch):
        model.train()
        losses = []
        for batch in sampler.batches(epoch):
            tensors = _to(collate_pairs(batch, vocab.pad_id), device)
            with _autocast(device, settings):
                q = model.embed(tensors["anchor_ids"], tensors["anchor_mask"])
                with torch.no_grad():
                    k = target.embed(tensors["positive_ids"], tensors["positive_mask"])
            excluded = queue.excluded(batch.identities, batch.anchor_content,
                                      batch.positive_content) if queue.filled else None
            loss = info_nce(q, k, settings.temperature,
                            false_negatives=tensors["false_negatives"],
                            queue=queue.keys if queue.filled else None,
                            queue_excluded=excluded)
            _step(model, optimiser, loss, settings, state["step"], total_steps)
            momentum_update(model, target, settings.momentum)
            queue.enqueue(k, batch.identities, batch.positive_content)
            losses.append(loss.item())
            state["step"] += 1
            if settings.max_steps and state["step"] >= settings.max_steps:
                break
        stop = bool(settings.max_steps and state["step"] >= settings.max_steps)
        return sum(losses) / max(len(losses), 1), len(losses), stop

    def evaluate_fn():
        scorer = EncoderScorer(model, vocab, max_len=settings.max_len, device=device,
                               name="encoder")
        metrics = evaluate(scorer, queries, pool, gold)
        last_metrics.clear()
        last_metrics.update(metrics)
        return metrics.get("mrr", 0.0)

    summary = _selection_loop(settings, epoch_fn, evaluate_fn, lambda a, b: a > b,
                              "mrr", model, vocab, run_id, log)

    best_metrics = measure_checkpoint(summary["checkpoint"], vocab, queries, pool,
                                      gold, device=device)
    record(conn, run_id, task, {f"encoder-{run_id}": best_metrics})
    summary["best_val_metrics"] = best_metrics
    return summary
