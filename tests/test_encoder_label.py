"""Naming a checkpoint by the run that made it, in this database.

A checkpoint carries the run number of the database it was trained in. The
test-split look scored three checkpoints trained on rented hardware, which
carried 2, 4 and 5 - here, extract runs from 14 September. These tests pin
how a checkpoint is matched to its local run instead, and that elenchus
check notices a score named after a run that is not a training run.
"""

import json
import types

import pytest

from elenchus.check import check_measurement_names
from elenchus.encoder import train
from elenchus.encoder.train import encoder_label, local_run

CONFIG = {"d_model": 384, "layers": 6, "heads": 6, "d_ff": 1536}
VOCAB = "vocab-69b6d888"


@pytest.fixture(autouse=True)
def fixed_vocab_hash(monkeypatch):
    monkeypatch.setattr(train, "vocab_hash", lambda vocab: VOCAB)


def model(config=CONFIG):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(to_dict=lambda: dict(config)))


def training_run(db, *, out="models/b3-con-mlm-11m-seed-0", seed=0,
                 code="22239c5", stage="contrastive", config=CONFIG, vocab=VOCAB):
    params = {"stage": stage, "config": config, "vocab": vocab,
              "settings": {"out": out, "seed": seed}}
    return db.execute(
        "INSERT INTO runs (kind, code_version, params, seed, started_at, status) "
        "VALUES ('train', ?, ?, ?, '2026-09-21 08:46', 'ok') RETURNING id",
        (code, json.dumps(params), seed)).fetchone()["id"]


def other_run(db, kind="extract"):
    return db.execute(
        "INSERT INTO runs (kind, code_version, params, started_at, status) "
        "VALUES (?, 'x', '{}', '2026-09-14 11:06', 'ok') RETURNING id",
        (kind,)).fetchone()["id"]


META = {"run_id": 2, "code_version": "22239c5", "seed": 0, "stage": "contrastive"}
PATH = "data/cloud-2026-09-21/elenchus/models/b3-con-mlm-11m-seed-0/best.pt"


def test_a_checkpoint_is_named_by_the_local_run_it_matches(db):
    other_run(db)                                  # whatever "run 2" is here
    run = training_run(db)

    assert local_run(db, PATH, model(), None, META) == run
    assert encoder_label(db, PATH, model(), None, META) == f"encoder (run {run})"


def test_a_checkpoint_trained_here_keeps_its_own_number(db):
    run = training_run(db)
    meta = dict(META, run_id=run)

    assert encoder_label(db, PATH, model(), None, meta) == f"encoder (run {run})"


def test_a_checkpoint_from_nowhere_here_says_so(db):
    assert encoder_label(db, PATH, model(), None, META) == "encoder (run 2 elsewhere)"


@pytest.mark.parametrize("change", [
    {"config": dict(CONFIG, layers=4)},            # another size
    {"seed": 1},
    {"code": "a0a562b"},                           # the laptop's stage 1
    {"out": "models/b3-con-mlm-seed-0"},           # another directory
    {"vocab": "some-other-vocabulary"},
    {"stage": "mlm"},
])
def test_any_difference_is_another_run(db, change):
    training_run(db, **change)

    assert local_run(db, PATH, model(), None, META) is None


def test_two_runs_that_fit_name_neither(db):
    """Two identical training runs - a repeat - leave the checkpoint's own
    run undecidable, and a guess is worse than saying so."""
    training_run(db)
    training_run(db)

    assert encoder_label(db, PATH, model(), None, META) == "encoder (run 2 elsewhere)"


# ------------------------------------------------------------------- check


def score(db, run_id, method):
    db.execute(
        "INSERT INTO measurements (run_id, method, split, query_opt, pool_opt, "
        "n_queries, n_pool, metrics, created_at) "
        "VALUES (?, ?, 'test', 'O0', 'O3', 1, 1, '{}', 'now')", (run_id, method))
    db.commit()


def test_check_flags_a_score_named_after_a_run_that_did_not_train(db):
    extract = other_run(db)
    evaluation = other_run(db, kind="eval")
    score(db, evaluation, f"encoder (run {extract})")

    assert [w[2] for w in check_measurement_names(db)] == [f"encoder (run {extract})"]


def test_check_accepts_a_score_named_after_a_training_run(db):
    run = training_run(db)
    evaluation = other_run(db, kind="eval")
    score(db, evaluation, f"encoder (run {run})")

    assert check_measurement_names(db) == []


def test_check_leaves_a_score_marked_as_from_elsewhere_alone(db):
    evaluation = other_run(db, kind="eval")
    score(db, evaluation, "encoder (run 2 elsewhere)")

    assert check_measurement_names(db) == []
