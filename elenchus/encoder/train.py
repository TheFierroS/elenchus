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

Peak GPU memory is recorded for every epoch, training and validation apart,
because the larger of the two is not known in advance: contrastive
validation embeds the whole val pool at once. "allocated" is what tensors
needed, "reserved" what PyTorch's caching allocator held - the part of
nvidia-smi's number that belongs to this process's tensors. Both are GiB. On
a CPU they are None, never a made-up zero. Validation's reserved figure
still holds what training cached before it; its allocated figure is the one
that belongs to validation alone.

Every run measures its starting point on val before the first step, logged
as "start" and kept in the history as epoch -1; a contrastive run records it
beside its result as encoder-<run>-start. Without it a score cannot be
credited to learning: an untrained Transformer already gives functions with
similar tokens similar vectors, and a 50-step smoke run, still in warm-up,
reached val MRR 0.131 - above the model-free bar. The start is a reference,
never a checkpoint. max_steps=0 measures it and trains nothing.
"""

import contextlib
import dataclasses
import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from elenchus.corpus.dataset import candidate_rows
from elenchus.db import code_version, finish_run, start_run
from elenchus.encoder.batching import collate_masked, collate_pairs
from elenchus.encoder.data import (
    MaskedSampler,
    PairSampler,
    load_examples,
    subset_examples,
)
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
    max_steps: int | None = None    # None trains fully; 0 measures the start only
    train_fraction: float | None = None   # learning curve; None is all of train
    train_unit: str = "identity"          # "identity" or "package"
    eval_pair_share: float | None = None  # E3; None is uniform level pairs
    eval_every: int | None = None   # validate every N steps; None: after each epoch
    checkpoint_activations: bool = False  # recompute layers in backward (B5, memory)

    def __post_init__(self):
        if self.stage not in ("mlm", "contrastive"):
            raise ValueError(f"unknown stage {self.stage!r}")
        if self.train_fraction is not None and not 0 < self.train_fraction <= 1:
            raise ValueError(
                f"train_fraction must be in (0, 1], got {self.train_fraction}")
        if self.train_unit not in ("identity", "package"):
            raise ValueError(f"unknown train_unit {self.train_unit!r}")
        if self.eval_pair_share is not None and not 0 <= self.eval_pair_share <= 1:
            raise ValueError(
                f"eval_pair_share must be in [0, 1], got {self.eval_pair_share}")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError(f"max_steps must be None or >= 0, got {self.max_steps}")
        if self.eval_every is not None and self.eval_every < 1:
            raise ValueError(f"eval_every must be None or >= 1, got {self.eval_every}")


def vocab_hash(vocab):
    return hashlib.sha256("\n".join(vocab.tokens).encode()).hexdigest()[:16]


def check_vocabulary(conn, vocab, rows=None):
    built = vocab.source.get("dataset_fingerprint")
    current = dataset_fingerprint(conn, rows)
    if built != current:
        # The likelier cause is the wrong database (ELENCHUS_DB unset points at
        # data/elenchus.db), and rebuilding the vocabulary then would strand every
        # checkpoint trained on the right one - so that is asked about first.
        where = next((row[2] for row in conn.execute("PRAGMA database_list")
                      if row[1] == "main"), "") or "an in-memory database"
        raise StaleVocabulary(
            f"vocabulary was built for dataset {built}, but {where} is dataset "
            f"{current}. Is this the database you meant (ELENCHUS_DB, --db)? Only if "
            "the data itself changed, run `elenchus vocab` again - models trained "
            "on the old vocabulary will not load with the new one")


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


def local_run(conn, path, model, vocab, meta):
    """Return this database's id for the training run that made a checkpoint.

    A checkpoint records the run number of the database it was trained in,
    and a run number means nothing anywhere else: the checkpoints trained on
    rented hardware carry 2, 4 and 5, which here are extract runs from
    14 September. The run is found by what it was instead - code version,
    seed, stage, model shape, vocabulary and the directory it wrote to -
    and None is returned unless exactly one training run fits.
    """
    config = model.config.to_dict()
    vocabulary = vocab_hash(vocab)
    directory = Path(path).parent.name

    found = []
    for row in conn.execute(
            "SELECT id, code_version, seed, params FROM runs WHERE kind = 'train'"):
        params = json.loads(row["params"] or "{}")
        if (row["code_version"] == meta.get("code_version")
                and row["seed"] == meta.get("seed")
                and params.get("stage") == meta.get("stage")
                and params.get("config") == config
                and params.get("vocab") == vocabulary
                and Path(params.get("settings", {}).get("out", "")).name == directory):
            found.append(row["id"])
    return found[0] if len(found) == 1 else None


def encoder_label(conn, path, model, vocab, meta):
    """The name a checkpoint's scores are recorded under.

    encoder (run N) only when N is this database's run; otherwise the
    number it came with is kept but marked as belonging elsewhere, so that
    nobody follows it to the wrong run.
    """
    run = local_run(conn, path, model, vocab, meta)
    if run is not None:
        return f"encoder (run {run})"
    return f"encoder (run {meta.get('run_id')} elsewhere)"


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


GIB = 1024 ** 3


class MemoryProbe:
    """Peak GPU memory of one phase: reset before it, read after it.

    Everything goes through torch.cuda, so a test can stand in for a GPU by
    replacing those functions. On a device that is not CUDA every reading is
    None: no GPU memory was used, and a zero would read as a measurement.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"

    def reset(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def read(self):
        """{"allocated_gib": ..., "reserved_gib": ...} since the last reset."""
        if not self.cuda:
            return {"allocated_gib": None, "reserved_gib": None}
        return {
            "allocated_gib": torch.cuda.max_memory_allocated(self.device) / GIB,
            "reserved_gib": torch.cuda.max_memory_reserved(self.device) / GIB,
        }

    def describe(self):
        """The card itself, for the run's params; None off a GPU."""
        if not self.cuda:
            return None
        properties = torch.cuda.get_device_properties(self.device)
        return {"name": properties.name,
                "total_gib": properties.total_memory / GIB,
                # The readings depend on it: the same epoch reserved 9.00 GiB
                # with the default allocator and 3.61 with expandable segments.
                "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}


def _memory_text(train_peak, eval_peak):
    if eval_peak["allocated_gib"] is None:
        return ""
    parts = [("train", train_peak), ("val", eval_peak)]
    text = " ".join(f"{name} {peak['allocated_gib']:.2f}/{peak['reserved_gib']:.2f}"
                    for name, peak in parts if peak is not None)
    return f"  mem {text} GiB"


def _peak(history, key):
    values = [entry[phase][key] for entry in history
              for phase in ("train_memory", "val_memory")
              if entry[phase] is not None and entry[phase][key] is not None]
    return max(values) if values else None


def period_batches(sampler, settings):
    """The batches of each validation period, as a function of the period.

    Unset eval_every, a period is an epoch: exactly sampler.batches(epoch).
    Set, the epochs run on as one stream, and each period takes the next
    eval_every batches of it, across epoch boundaries. Each epoch's order comes
    from its own seeded generator, so where validation falls cannot change
    which batches are drawn. When the stream (all of `settings.epochs`) runs out,
    the last period is short, and the one after it takes no steps and ends
    the run.

    Why: an epoch of a 25% subset is a quarter of a full one, so validating per
    epoch gives runs on different fractions different chances and different
    patience in steps. A fixed interval in steps gives every run the same.
    """
    if settings.eval_every is None:
        return sampler.batches
    stream = (batch for epoch in range(settings.epochs)
              for batch in sampler.batches(epoch))
    return lambda _period: itertools.islice(stream, settings.eval_every)


def reached(step, settings):
    """Whether a run capped by max_steps has taken them all. 0 is a cap too."""
    return settings.max_steps is not None and step >= settings.max_steps


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

    The candidate rows are read from the database once, and every stage - the
    vocabulary check, the fingerprint recorded, train, val, the val task -
    works from that one list, so all of them see the same data.
    """
    rows = candidate_rows(conn)
    check_vocabulary(conn, vocab, rows)
    torch.manual_seed(settings.seed)
    device = _device(settings)
    model, settings, changes, init_meta = build_model(vocab, settings, config,
                                                      overrides, device)
    model.checkpoint_layers = settings.checkpoint_activations
    memory = MemoryProbe(device)

    fingerprint = dataset_fingerprint(conn, rows)
    params = {
        "stage": settings.stage,
        "settings": asdict(settings),
        "config": model.config.to_dict(),
        "dataset": fingerprint,
        "vocab": vocab_hash(vocab),
        "init": settings.init,
        "init_changes": changes,
        "parameters": count_parameters(model),
        "gpu": memory.describe(),
    }
    run_id = start_run(conn, "train", tool="elenchus-encoder", params=params,
                       seed=settings.seed)
    log(f"run {run_id}: {settings.stage}, {count_parameters(model):,} parameters, "
        f"device {device}")
    for name, (old, new) in changes.items():
        log(f"  {name}: {old} in {settings.init}, {new} here")
    if settings.checkpoint_activations:
        log("activation checkpointing: each layer recomputed in the backward pass")

    try:
        everything = load_examples(conn, "train", rows)
        examples = subset_examples(everything, settings.train_fraction,
                                   settings.train_unit, settings.seed)
        subset = {"rows": len(examples),
                  "identities": len({e.identity for e in examples}),
                  "packages": len({e.identity[0] for e in examples}),
                  "of_rows": len(everything),
                  "of_identities": len({e.identity for e in everything}),
                  "of_packages": len({e.identity[0] for e in everything})}
        if settings.train_fraction is not None:
            log(f"train subset ({settings.train_fraction:g} by {settings.train_unit}): "
                f"{subset['identities']:,} of {subset['of_identities']:,} functions, "
                f"{subset['packages']} of {subset['of_packages']} packages")
        if settings.stage == "mlm":
            summary = _train_mlm(conn, model, vocab, settings, examples, device,
                                 run_id, log, memory, rows)
        else:
            summary = _train_contrastive(conn, model, vocab, settings, examples,
                                         device, run_id, log, memory, rows)
        summary["train_subset"] = subset
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
                    model, vocab, run_id, log, memory, step_of):
    best, best_epoch, waited = None, -1, 0
    out = Path(settings.out)

    started = time.time()
    memory.reset()
    start = evaluate_fn()
    start_peak = memory.read()
    history = [{"epoch": -1, "step": 0, "train_loss": None, metric_name: start,
                "seconds": time.time() - started, "train_memory": None,
                "val_memory": start_peak}]
    log(f"start      val {metric_name} {start:.4f}  (before training, "
        f"{time.time() - started:.0f}s){_memory_text(None, start_peak)}")

    summary = {f"start_val_{metric_name}": start}
    if settings.max_steps == 0:
        log("max_steps 0: the start is measured; nothing is trained")
        return summary | {"best_epoch": None, f"best_val_{metric_name}": None,
                          "history": history, "checkpoint": None,
                          "peak_allocated_gib": _peak(history, "allocated_gib"),
                          "peak_reserved_gib": _peak(history, "reserved_gib")}

    # "epoch" in the history and checkpoints is the validation period: an
    # epoch by default, eval_every steps when that is set ("step" says where).
    by_steps = settings.eval_every is not None
    periods = itertools.count() if by_steps else range(settings.epochs)
    unit = "validations" if by_steps else "epochs"
    for epoch in periods:
        started = time.time()
        memory.reset()
        train_loss, steps, stop = epoch_fn(epoch)
        if by_steps and steps == 0:
            break               # the stream ran out exactly at the last period
        train_peak = memory.read()
        memory.reset()
        metric = evaluate_fn()
        eval_peak = memory.read()
        history.append({"epoch": epoch, "step": step_of(), "train_loss": train_loss,
                        metric_name: metric, "seconds": time.time() - started,
                        "train_memory": train_peak, "val_memory": eval_peak})
        improved = best is None or better(metric, best)
        label = f"eval {epoch:3}  step {step_of():6}" if by_steps else f"epoch {epoch:3}"
        log(f"{label}  loss {train_loss:.4f}  val {metric_name} {metric:.4f}"
            f"{'  *' if improved else ''}  ({steps} steps, {time.time() - started:.0f}s)"
            f"{_memory_text(train_peak, eval_peak)}")
        meta = _meta(settings, run_id, epoch, metric_name, metric)
        save_checkpoint(out / "last.pt", model, vocab, meta)
        if improved:
            best, best_epoch, waited = metric, epoch, 0
            save_checkpoint(out / "best.pt", model, vocab, meta)
        else:
            waited += 1
            if waited >= settings.patience:
                log(f"no improvement for {settings.patience} {unit}; stopping")
                break
        if stop:
            break
    return summary | {
            "best_epoch": best_epoch, f"best_val_{metric_name}": best,
            "history": history, "checkpoint": str(out / "best.pt"),
            "peak_allocated_gib": _peak(history, "allocated_gib"),
            "peak_reserved_gib": _peak(history, "reserved_gib")}


def _train_mlm(conn, model, vocab, settings, examples, device, run_id, log, memory,
               rows):
    sampler = MaskedSampler(examples, vocab, batch_size=settings.batch_size,
                            max_len=settings.max_len, rate=settings.mask_rate,
                            seed=settings.seed)
    val = MaskedSampler(load_examples(conn, "val", rows), vocab,
                        batch_size=settings.batch_size, max_len=settings.max_len,
                        rate=settings.mask_rate, seed=settings.seed + 1)
    optimiser = _optimiser(model, settings)
    total_steps = settings.max_steps or len(sampler) * settings.epochs
    state = {"step": 0}
    batches = period_batches(sampler, settings)

    def epoch_fn(epoch):
        model.train()
        losses = []
        for batch in batches(epoch):
            tensors = _to(collate_masked(batch, vocab.pad_id), device)
            with _autocast(device, settings):
                logits = model.mlm_logits(tensors["input_ids"], tensors["attention_mask"])
            loss = mlm_loss(logits, tensors["labels"])
            _step(model, optimiser, loss, settings, state["step"], total_steps)
            losses.append(loss.item())
            state["step"] += 1
            if reached(state["step"], settings):
                break
        stop = reached(state["step"], settings)
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
                           "loss", model, vocab, run_id, log, memory,
                           lambda: state["step"])


def _train_contrastive(conn, model, vocab, settings, examples, device, run_id, log,
                       memory, rows):
    sampler = PairSampler(examples, vocab, batch_size=settings.batch_size,
                          per_package=settings.per_package, max_len=settings.max_len,
                          seed=settings.seed, eval_pair_share=settings.eval_pair_share)
    if len(sampler) == 0:
        raise ValueError("not enough pairable functions for one batch")
    target = momentum_copy(model)
    queue = MemoryQueue(settings.queue_size, model.config.embed_dim, device)
    optimiser = _optimiser(model, settings)
    total_steps = settings.max_steps or len(sampler) * settings.epochs
    state = {"step": 0}
    batches = period_batches(sampler, settings)

    queries, pool, gold = retrieval_task(conn, split="val", rows=rows)
    task = task_params(conn, "val", "O0", "O3", settings.seed, rows)
    measured = []
    pair_counts = []   # per trained epoch: how often each level pair was drawn

    def epoch_fn(epoch):
        model.train()
        losses = []
        drawn = defaultdict(int)
        pair_counts.append(drawn)
        for batch in batches(epoch):
            for first, second in zip(batch.anchor_levels, batch.positive_levels):
                drawn["-".join(sorted((first, second)))] += 1
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
            if reached(state["step"], settings):
                break
        stop = reached(state["step"], settings)
        return sum(losses) / max(len(losses), 1), len(losses), stop

    def evaluate_fn():
        scorer = EncoderScorer(model, vocab, max_len=settings.max_len, device=device,
                               name="encoder")
        metrics = evaluate(scorer, queries, pool, gold)
        measured.append(metrics)
        return metrics.get("mrr", 0.0)

    summary = _selection_loop(settings, epoch_fn, evaluate_fn, lambda a, b: a > b,
                              "mrr", model, vocab, run_id, log, memory,
                              lambda: state["step"])

    # The epoch the package mean would have kept, beside the one kept: whether
    # the two criteria disagree is measured on every run, not assumed.
    for entry, metrics in zip(summary["history"], measured):
        entry["package_mrr"] = metrics.get("package_mean", {}).get("mrr")
    trained = [entry for entry in summary["history"]
               if entry["epoch"] >= 0 and entry["package_mrr"] is not None]
    best_by_package = max(trained, key=lambda entry: entry["package_mrr"], default=None)
    summary["best_epoch_by_package_mrr"] = best_by_package and best_by_package["epoch"]

    for entry, drawn in zip(summary["history"][1:], pair_counts):
        entry["pairs"] = dict(sorted(drawn.items()))

    results = {f"encoder-{run_id}-start": measured[0]}
    summary["start_val_metrics"] = measured[0]
    if summary["checkpoint"] is not None:
        best_metrics = measure_checkpoint(summary["checkpoint"], vocab, queries, pool,
                                          gold, device=device)
        results[f"encoder-{run_id}"] = best_metrics
        summary["best_val_metrics"] = best_metrics
    record(conn, run_id, task, results)
    return summary
