"""The short learning-curve rule, applied to hand-built run summaries.

experiments/curve2_results.py decides whether the corpus grows again before
full training. Its rule was written before the runs (docs/experiments.md,
L2); these tests build summary.json files with chosen scores and check that
each branch, each boundary and both provisional checks come out as written.

L2's rule differs from L's in three ways this file pins: a middle band
between T and 2T that goes to full training rather than another wave, a
threshold taken from fully decayed runs, and a noise check on the two full
runs of this experiment itself.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "curve2_results.py"
spec = importlib.util.spec_from_file_location("curve2_results", SCRIPT)
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)

T = curve.T
STEPS = curve.STEPS
EVERY = curve.EVERY


def summary(mrr, late_gain=0.0, final_step=STEPS, identities=100, packages=5,
            best_step=None):
    """A run validated every 385 steps whose best, `mrr`, beats its best by
    75% of the budget by `late_gain`, reached at the last validation."""
    steps = list(range(EVERY, final_step + 1, EVERY))
    if steps[-1] != final_step:
        steps.append(final_step)
    history = [{"epoch": -1, "step": 0, "mrr": 0.03}]
    for epoch, step in enumerate(steps):
        score = mrr if step == steps[-1] else mrr - late_gain
        history.append({"epoch": epoch, "step": step, "mrr": score})
    best_epoch = len(steps) - 1
    if best_step is not None:
        best_epoch = steps.index(best_step)
    return {"history": history, "best_epoch": best_epoch,
            "best_val_metrics": {"mrr": mrr, "recall@10": 0.5,
                                 "package_mean": {"mrr": mrr + 0.05}},
            "train_subset": {"identities": identities, "packages": packages}}


def write(root, name, seed, data):
    path = root / f"{curve.PREFIX}-{name}-seed-{seed}"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(data))


def build(root, full=(0.50, 0.50), identity_half=0.50, package_half=(0.50, 0.50),
          late=(0.0, 0.0)):
    for seed in (0, 1):
        write(root, "full", seed, summary(full[seed], late_gain=late[seed]))
        write(root, "package-0.5", seed, summary(package_half[seed]))
    write(root, "identity-0.5", 0, summary(identity_half))
    return curve.load(root)


# ------------------------------------------------------------------ branches


def test_both_gains_within_t_send_the_project_to_full_training(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.495,
                                 package_half=(0.493, 0.495)))
    assert result["decision"].startswith("levelled off")
    assert result["identity_gain"] == pytest.approx(0.005)
    assert not result["provisional"]


def test_a_gain_between_t_and_twice_t_still_goes_to_full_training(tmp_path):
    """The band L2 adds: climbing, but by less than the doubling that
    produced it, and re-splitting after a trained model costs more."""
    result = curve.verdict(build(tmp_path, identity_half=0.485,
                                 package_half=(0.485, 0.485)))
    assert result["package_gain"] == pytest.approx(0.015)
    assert T < result["package_gain"] <= 2 * T
    assert result["decision"].startswith("still climbing")


def test_a_package_gain_above_twice_t_calls_for_another_wave(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.48,
                                 package_half=(0.45, 0.45)))
    assert result["decision"].startswith("more packages still help")


def test_only_an_identity_gain_above_twice_t_favours_large_packages(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.45,
                                 package_half=(0.495, 0.495)))
    assert result["decision"].startswith("only more functions help")


def test_the_package_branch_wins_when_the_two_gains_are_equal(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.45,
                                 package_half=(0.45, 0.45)))
    assert result["identity_gain"] == pytest.approx(result["package_gain"])
    assert result["decision"].startswith("more packages still help")


# ----------------------------------------------------------------- boundaries


def test_a_gain_of_exactly_t_is_not_above_it(tmp_path, monkeypatch):
    # A T with an exact binary form (1/64), and scores whose differences are
    # exact too, so the boundary is not decided by a rounding error.
    monkeypatch.setattr(curve, "T", 1 / 64)
    result = curve.verdict(build(tmp_path, identity_half=0.484375,
                                 package_half=(0.484375, 0.484375)))
    assert result["identity_gain"] == 1 / 64
    assert result["decision"].startswith("levelled off")


def test_a_gain_of_exactly_twice_t_is_not_another_wave(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "T", 1 / 64)
    result = curve.verdict(build(tmp_path, identity_half=0.46875,
                                 package_half=(0.46875, 0.46875)))
    assert result["package_gain"] == 2 * (1 / 64)
    assert result["decision"].startswith("still climbing")


def test_the_package_gain_pairs_each_seed_with_its_own_full_run(tmp_path):
    """Seed 1 lower everywhere: paired by seed both gains are 0.005, while
    crossing the seeds would carry the seed difference into the gain."""
    result = curve.verdict(build(tmp_path, full=(0.50, 0.47), identity_half=0.495,
                                 package_half=(0.495, 0.465)))
    assert result["package_gains"] == pytest.approx([0.005, 0.005])
    assert result["decision"].startswith("levelled off")


def test_the_package_gain_is_the_mean_over_seeds(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.50,
                                 package_half=(0.475, 0.500)))
    # seed 0 alone is above T; the mean (0.0125) is in the middle band
    assert result["package_gains"] == pytest.approx([0.025, 0.0])
    assert result["decision"].startswith("still climbing")


# ------------------------------------------------------------ the two checks


def test_a_full_run_still_climbing_makes_a_levelled_verdict_provisional(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.495,
                                 package_half=(0.495, 0.495),
                                 late=(0.02, 0.0)))
    assert result["still_climbing"] == [0]
    assert result["decision"].startswith("levelled off")
    assert result["provisional"]


def test_a_gain_above_t_stands_even_if_the_runs_were_still_climbing(tmp_path):
    """Under-trained runs would not be expected to shrink a real gain."""
    result = curve.verdict(build(tmp_path, identity_half=0.44,
                                 package_half=(0.44, 0.44), late=(0.02, 0.02)))
    assert result["still_climbing"] == [0, 1]
    assert not result["provisional"]


def test_seeds_further_apart_than_t_make_any_verdict_provisional(tmp_path):
    """Noise as large as the threshold: the branch is reported, not trusted."""
    result = curve.verdict(build(tmp_path, full=(0.50, 0.48), identity_half=0.44,
                                 package_half=(0.44, 0.42)))
    assert result["seed_spread"] == pytest.approx(0.02)
    assert result["decision"].startswith("more packages still help")
    assert result["provisional"]


def test_seeds_exactly_t_apart_are_not_provisional(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "T", 1 / 64)
    result = curve.verdict(build(tmp_path, full=(0.5, 0.484375),
                                 identity_half=0.5,
                                 package_half=(0.5, 0.484375)))
    assert result["seed_spread"] == 1 / 64
    assert not result["provisional"]


# --------------------------------------------------------------- bookkeeping


def test_an_unfinished_run_decides_nothing(tmp_path):
    results = build(tmp_path)
    results[("full", 1)]["complete"] = False
    result = curve.verdict(results)
    assert result["decision"] is None
    assert result["missing"] == ["full seed 1"]


def test_a_missing_run_decides_nothing(tmp_path):
    for seed in (0, 1):
        write(tmp_path, "full", seed, summary(0.5))
        write(tmp_path, "package-0.5", seed, summary(0.5))
    result = curve.verdict(curve.load(tmp_path))
    assert result["decision"] is None
    assert result["missing"] == ["identity-0.5 seed 0"]


def test_the_shape_runs_of_l_are_not_read(tmp_path):
    """L2 drops the 25% runs; a leftover directory must not change anything."""
    results = build(tmp_path)
    write(tmp_path, "package-0.25", 0, summary(0.1))
    assert curve.load(tmp_path).keys() == results.keys()


def test_a_run_that_stopped_at_twenty_epochs_is_not_complete():
    """7,700 steps is the budget L2 considered and did not take."""
    assert not curve.read(summary(0.5, final_step=7700))["complete"]
    assert curve.read(summary(0.5))["complete"]


def test_the_step_a_full_run_kept_is_reported():
    """Full training's budget is set from it, so it has to survive the read."""
    assert curve.read(summary(0.5, best_step=8_855))["best_step"] == 8_855
