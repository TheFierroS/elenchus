"""The learning-curve rule, applied to hand-built run summaries.

experiments/curve_results.py decides whether the corpus grows before full
training. Its rule was written before the runs (docs/experiments.md, L);
these tests build summary.json files with chosen scores and check that each
branch, each boundary and the budget check come out as written.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "curve_results.py"
spec = importlib.util.spec_from_file_location("curve_results", SCRIPT)
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)

T = curve.T
STEPS = curve.STEPS
EVERY = 172


def summary(mrr, late_gain=0.0, final_step=STEPS, identities=100, packages=5):
    """A run validated every 172 steps whose best, `mrr`, beats its best by
    75% of the budget by `late_gain`, reached at the last validation."""
    steps = list(range(EVERY, final_step + 1, EVERY))
    if steps[-1] != final_step:
        steps.append(final_step)
    history = [{"epoch": -1, "step": 0, "mrr": 0.03}]
    for epoch, step in enumerate(steps):
        score = mrr if step == steps[-1] else mrr - late_gain
        history.append({"epoch": epoch, "step": step, "mrr": score})
    return {"history": history, "best_epoch": len(steps) - 1,
            "best_val_metrics": {"mrr": mrr, "recall@10": 0.5,
                                 "package_mean": {"mrr": mrr + 0.05}},
            "train_subset": {"identities": identities, "packages": packages}}


def write(root, name, seed, data):
    path = root / f"curve-{name}-seed-{seed}"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(data))


def build(root, full=(0.40, 0.40), identity_half=0.40, package_half=(0.40, 0.40),
          quarters=True, late=(0.0, 0.0)):
    for seed in (0, 1):
        write(root, "full", seed, summary(full[seed], late_gain=late[seed]))
        write(root, "package-0.5", seed, summary(package_half[seed]))
        if quarters:
            write(root, "package-0.25", seed, summary(package_half[seed] - 0.02))
    write(root, "identity-0.5", 0, summary(identity_half))
    if quarters:
        write(root, "identity-0.25", 0, summary(identity_half - 0.03))
    return curve.load(root)


def test_both_gains_within_t_mean_the_curve_has_flattened(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.395,
                                 package_half=(0.392, 0.394)))
    assert result["decision"].startswith("flattened")
    assert result["identity_gain"] == pytest.approx(0.005)
    assert result["package_gain"] == pytest.approx(0.007)
    assert not result["provisional"]


def test_a_package_gain_above_t_grows_the_corpus_with_packages(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.37,
                                 package_half=(0.38, 0.38)))
    assert result["decision"].startswith("more packages help")


def test_the_package_branch_wins_when_both_gains_are_above_t(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.30,
                                 package_half=(0.35, 0.35)))
    assert result["identity_gain"] > T and result["package_gain"] > T
    assert result["decision"].startswith("more packages help")


def test_only_an_identity_gain_above_t_grows_existing_domains(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.38,
                                 package_half=(0.395, 0.395)))
    assert result["decision"].startswith("only more functions help")


def test_a_gain_of_exactly_t_is_not_above_it(tmp_path, monkeypatch):
    # 0.0148 has no exact binary form, so the boundary is tested at a T that
    # has one (1/64), with scores whose differences are exact too.
    monkeypatch.setattr(curve, "T", 1 / 64)
    result = curve.verdict(build(tmp_path, full=(0.5, 0.5), identity_half=0.484375,
                                 package_half=(0.484375, 0.484375)))
    assert result["identity_gain"] == 1 / 64 and result["package_gain"] == 1 / 64
    assert result["decision"].startswith("flattened")


def test_a_package_gain_of_exactly_t_alone_is_not_above_it(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "T", 1 / 64)
    result = curve.verdict(build(tmp_path, full=(0.5, 0.5), identity_half=0.5,
                                 package_half=(0.484375, 0.484375)))
    assert result["decision"].startswith("flattened")


def test_an_identity_gain_of_exactly_t_alone_is_not_above_it(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "T", 1 / 64)
    result = curve.verdict(build(tmp_path, full=(0.5, 0.5), identity_half=0.484375,
                                 package_half=(0.5, 0.5)))
    assert result["decision"].startswith("flattened")


def test_the_package_gain_pairs_each_seed_with_its_own_full_run(tmp_path):
    # Seed 1 starts lower everywhere (as in E3). Paired by seed, both gains are
    # 0.005; pairing seed 1's half run with seed 0's full run would add the
    # seed difference and cross T.
    result = curve.verdict(build(tmp_path, full=(0.40, 0.37), identity_half=0.395,
                                 package_half=(0.395, 0.365)))
    assert result["package_gains"] == pytest.approx([0.005, 0.005])
    assert result["decision"].startswith("flattened")


def test_the_package_gain_is_the_mean_over_seeds(tmp_path):
    result = curve.verdict(build(tmp_path, identity_half=0.40,
                                 package_half=(0.375, 0.400)))
    # seed 0 alone is above T; the mean (0.0125) is not
    assert result["package_gains"] == pytest.approx([0.025, 0.0])
    assert result["decision"].startswith("flattened")


def test_a_full_run_still_climbing_makes_a_flat_verdict_provisional(tmp_path):
    result = curve.verdict(build(tmp_path, late=(0.0, T / 2 + 0.001)))
    assert result["still_climbing"] == [1]
    assert result["decision"].startswith("flattened") and result["provisional"]


def test_climbing_by_exactly_half_t_is_not_still_climbing(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "T", 1 / 64)
    build(tmp_path)
    (tmp_path / "curve-full-seed-0" / "summary.json").write_text(
        json.dumps(summary(0.5, late_gain=1 / 128)))
    results = curve.load(tmp_path)
    assert results[("full", 0)]["late_gain"] == 1 / 128
    assert curve.verdict(results)["still_climbing"] == []


def test_the_budget_check_starts_at_three_quarters_of_the_steps(tmp_path):
    # Climbing between 50% and 75% of the budget, flat after: levelled off.
    build(tmp_path)
    data = summary(0.40)
    for entry in data["history"][1:]:
        if entry["step"] <= 0.5 * STEPS:
            entry["mrr"] = 0.38
    (tmp_path / "curve-full-seed-0" / "summary.json").write_text(json.dumps(data))
    results = curve.load(tmp_path)
    assert results[("full", 0)]["late_gain"] == 0.0
    assert curve.verdict(results)["still_climbing"] == []


def test_climbing_just_under_half_t_is_not_still_climbing(tmp_path):
    # "More than T / 2": just under it is levelled off. (Exactly T / 2 is not
    # testable: 0.40 - (0.40 - 0.0074) is 0.0074000000000000177 in floats.)
    result = curve.verdict(build(tmp_path, late=(T / 2 - 1e-6, 0.0)))
    assert result["still_climbing"] == [] and not result["provisional"]


def test_a_gain_above_t_stands_even_if_the_full_runs_were_still_climbing(tmp_path):
    result = curve.verdict(build(tmp_path, package_half=(0.36, 0.36), late=(0.02, 0.02)))
    assert result["still_climbing"] == [0, 1]
    assert result["decision"].startswith("more packages help")
    assert not result["provisional"]


def test_the_quarter_runs_do_not_decide_and_their_absence_does_not_block(tmp_path):
    result = curve.verdict(build(tmp_path, quarters=False))
    assert result["decision"].startswith("flattened")
    assert result["shape"] == {"identity": None, "package": None}


def test_the_shape_is_read_from_the_quarter_runs(tmp_path):
    build(tmp_path)
    # package seeds gain 0.02 and 0.04 from 25% to 50%: the shape is their mean
    (tmp_path / "curve-package-0.25-seed-1" / "summary.json").write_text(
        json.dumps(summary(0.36)))
    result = curve.verdict(curve.load(tmp_path))
    assert result["shape"]["identity"] == pytest.approx(0.03)
    assert result["shape"]["package"] == pytest.approx(0.03)


def test_a_missing_deciding_run_gives_no_verdict(tmp_path):
    results = build(tmp_path)
    results[("package-0.5", 1)] = None
    result = curve.verdict(results)
    assert result["decision"] is None
    assert result["missing"] == ["package-0.5 seed 1"]


def test_a_run_that_did_not_take_every_step_gives_no_verdict(tmp_path):
    build(tmp_path)
    (tmp_path / "curve-full-seed-1" / "summary.json").write_text(
        json.dumps(summary(0.40, final_step=3000)))
    results = curve.load(tmp_path)
    assert not results[("full", 1)]["complete"]
    assert curve.verdict(results)["missing"] == ["full seed 1"]


def test_scores_come_from_the_kept_checkpoint_measured_again(tmp_path):
    data = summary(0.40)
    data["history"][-1]["mrr"] = 0.401          # the in-run validation
    write(tmp_path, "full", 0, data)
    run = curve.load(tmp_path)[("full", 0)]
    assert run["mrr"] == 0.40                   # best.pt, measured from disk
    assert run["package_mrr"] == pytest.approx(0.45)
    assert run["best_step"] == STEPS and run["complete"]


def test_the_report_prints_the_table_and_the_verdict(tmp_path, capsys):
    curve.show(build(tmp_path, package_half=(0.37, 0.37)))
    out = capsys.readouterr().out
    assert "identity-0.25" in out and "package-0.5" in out
    assert f"T = {T:.4f}" in out
    assert "package gain  50% -> 100%: +0.0300  (seed 0 +0.0300, seed 1 +0.0300)" in out
    assert "verdict: more packages help" in out


def test_the_report_says_what_it_is_waiting_for(tmp_path, capsys):
    write(tmp_path, "full", 0, summary(0.40))
    curve.show(curve.load(tmp_path))
    out = capsys.readouterr().out
    assert "identity-0.5       0  not finished" in out
    assert "verdict: waiting for identity-0.5 seed 0" in out
