"""The full-training rule, applied to hand-built run summaries.

experiments/full_results.py decides whether MLM pre-training is kept. Its
rule was written before the runs (docs/experiments.md, full training,
rewritten 20 September for the wave 3 corpus); these tests build summaries
with chosen scores and check each branch, the seed-spread margin and the
budget check.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "full_results.py"
spec = importlib.util.spec_from_file_location("full_results", SCRIPT)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

STEPS = full.STEPS
EVERY = 543


def summary(mrr, late_gain=0.0, package_mrr=None):
    """A run validated every epoch whose best, `mrr`, beats its best by 75%
    of the budget by `late_gain`."""
    steps = list(range(EVERY, STEPS + 1, EVERY))
    history = [{"epoch": -1, "step": 0, "mrr": 0.02, "recall@10": 0.03}]
    for epoch, step in enumerate(steps):
        score = mrr if step == steps[-1] else mrr - late_gain
        history.append({"epoch": epoch, "step": step, "mrr": score,
                        "recall@10": 0.3})
    return {
        "history": history,
        "best_epoch": len(steps) - 1,
        "best_val_metrics": {
            "mrr": mrr, "recall@10": 0.3,
            "package_mean": {"mrr": package_mrr if package_mrr is not None else mrr},
        },
        "start_val_metrics": {"mrr": 0.02},
        "best_epoch_by_package_mrr": len(steps) - 1,
        "peak_reserved_gib": 3.7,
    }


def write(root, name, data):
    path = root / name
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(data))


def build(root, mlm=(0.60, 0.60), random=(0.60, 0.60), late=(0.0, 0.0),
          mlm_package=None):
    write(root, "b2-mlm", {"history": [{"epoch": 0, "step": 1}],
                           "best_epoch": 0, "start_val_loss": 5.0,
                           "best_val_loss": 2.0})
    for seed in (0, 1):
        write(root, f"b3-con-mlm-seed-{seed}",
              summary(mlm[seed], late_gain=late[seed],
                      package_mrr=mlm_package[seed] if mlm_package else None))
        write(root, f"b4-con-random-seed-{seed}", summary(random[seed]))
    return full.collect(root)[1]


def test_a_gain_beyond_the_margin_keeps_mlm(tmp_path):
    result = full.verdict(build(tmp_path, mlm=(0.64, 0.64), random=(0.60, 0.60)))
    assert result["gain"] == pytest.approx(0.04)
    assert result["result"].startswith("MLM helps")
    assert not result["provisional"]


def test_a_loss_beyond_the_margin_drops_mlm(tmp_path):
    result = full.verdict(build(tmp_path, mlm=(0.56, 0.56), random=(0.60, 0.60)))
    assert result["result"].startswith("MLM hurts")


def test_no_measured_difference_also_drops_mlm(tmp_path):
    """An hour of pre-training an iteration is kept only for a measured gain."""
    result = full.verdict(build(tmp_path, mlm=(0.602, 0.602), random=(0.60, 0.60)))
    assert result["result"].startswith("no measured difference")


def test_the_margin_is_the_seed_spread_when_that_is_wider(tmp_path):
    """Two seeds 0.02 apart make a 0.01 mean gain unreadable."""
    result = full.verdict(build(tmp_path, mlm=(0.62, 0.60), random=(0.60, 0.60)))
    assert result["spread"] == pytest.approx(0.02)
    assert result["margin"] == pytest.approx(0.02)
    assert result["result"].startswith("no measured difference")


def test_the_margin_has_a_floor(tmp_path):
    result = full.verdict(build(tmp_path, mlm=(0.6060, 0.6060), random=(0.60, 0.60)))
    assert result["spread"] == 0
    assert result["margin"] == 0.005
    assert result["result"].startswith("MLM helps")


def test_a_gain_the_package_mean_contradicts_is_not_a_win(tmp_path):
    """The rule asks both means to agree before an hour a run is kept."""
    result = full.verdict(build(tmp_path, mlm=(0.64, 0.64), random=(0.60, 0.60),
                                mlm_package=(0.50, 0.50)))
    assert result["gain"] > result["margin"]
    assert result["result"].startswith("no measured difference")


def test_runs_still_climbing_make_a_null_result_provisional(tmp_path):
    result = full.verdict(build(tmp_path, mlm=(0.602, 0.602), random=(0.60, 0.60),
                                late=(0.02, 0.0)))
    assert result["climbing"] == 1
    assert result["result"].startswith("no measured difference")
    assert result["provisional"]


def test_a_real_difference_stands_even_if_the_budget_was_short(tmp_path):
    result = full.verdict(build(tmp_path, mlm=(0.64, 0.64), random=(0.60, 0.60),
                                late=(0.02, 0.02)))
    assert result["climbing"] == 2
    assert not result["provisional"]


def test_one_finished_seed_decides_nothing(tmp_path):
    groups = build(tmp_path)
    groups["B3 mlm init"][1] = None
    assert full.verdict(groups) is None
