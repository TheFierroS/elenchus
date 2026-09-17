"""Scenario tests for the long-run experiment scripts.

run_e3.sh and run_full.sh start hours of GPU work, and every run records
code_version(): the commit, plus "-dirty" when `git status --porcelain`
prints anything. The scripts must refuse to start a run that would be
recorded as dirty, by that same definition, and must stop before a run if the
commit moved while they were going (E3's eight runs were recorded under two
commits because one landed mid-way).

Each test copies the scripts into a throwaway git repository and runs them
with a fake `elenchus` that records its arguments and writes the run's
summary.json, and a fake `pgrep`, so a real training run on the machine
running the tests cannot change the outcome.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("guard.sh", "run_e3.sh", "run_full.sh")
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
       "-c", "commit.gpgsign=false"]

FAKE_ELENCHUS = """#!/bin/sh
# Records each call; "trains" by writing summary.json into --out.
echo "$*" >> "$FAKE_CALLS"
[ "$1" = check ] && exit "${FAKE_CHECK_EXIT:-0}"
out=""; prev=""
for arg in "$@"; do [ "$prev" = "--out" ] && out="$arg"; prev="$arg"; done
mkdir -p "$out" && echo '{}' > "$out/summary.json"
if [ "$out" = "${FAKE_COMMIT_AFTER:-}" ]; then
  echo more >> tracked.txt
  git -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qam mid-run
fi
if [ "$out" = "${FAKE_NEW_FILE_AFTER:-}" ]; then
  echo "x = 1" > new_module.py
fi
exit "${FAKE_TRAIN_EXIT:-0}"
"""


def _executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _git(root, *args):
    return subprocess.run([*GIT, *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(REPO / "experiments" / name, root / "experiments" / name)
    (root / ".gitignore").write_text("data/\nmodels/\n")
    (root / "tracked.txt").write_text("code\n")
    (root / "data").mkdir()
    (root / "models").mkdir()
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "start")
    return root


@pytest.fixture
def fakes(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    _executable(fake / "elenchus", FAKE_ELENCHUS)
    _executable(fake / "pgrep", "#!/bin/sh\nexit \"${FAKE_PGREP_EXIT:-1}\"\n")
    return fake


def run_script(repo, fakes, script, shell="sh", **env):
    calls = repo.parent / "calls.txt"
    calls.unlink(missing_ok=True)
    environ = {**os.environ, "PATH": f"{fakes}{os.pathsep}{os.environ['PATH']}",
               "ELENCHUS": str(fakes / "elenchus"), "FAKE_CALLS": str(calls),
               **env}
    for name in ("SHARE", "SEEDS"):
        if name not in env:
            environ.pop(name, None)
    done = subprocess.run([shell, f"experiments/{script}"], cwd=repo, env=environ,
                          capture_output=True, text=True, timeout=60)
    lines = calls.read_text().splitlines() if calls.exists() else []
    trained = [line for line in lines if line.startswith("train ")]
    return done, lines, trained


def out_of(call):
    words = call.split()
    return words[words.index("--out") + 1]


# -- run_full.sh ------------------------------------------------------------

def test_full_runs_every_step_in_order_on_a_clean_tree(repo, fakes):
    done, calls, trained = run_script(repo, fakes, "run_full.sh", SHARE="none")
    assert done.returncode == 0, done.stdout + done.stderr
    assert calls[0] == "check"
    assert [out_of(c) for c in trained] == (
        ["models/b2-mlm"]
        + [f"models/b3-con-mlm-seed-{s}" for s in (0, 1, 2)]
        + [f"models/b4-con-random-seed-{s}" for s in (0, 1, 2)])
    assert trained[0].startswith("train mlm ")
    assert all("--eval-pair-share" not in c for c in trained)
    assert all("--init models/b2-mlm/best.pt" in c for c in trained[1:4])
    assert all("--init" not in c for c in trained[4:])


@pytest.mark.parametrize("share", [None, "", "0.3", "uniform"])
def test_full_refuses_without_a_valid_share(repo, fakes, share):
    env = {} if share is None else {"SHARE": share}
    done, calls, _ = run_script(repo, fakes, "run_full.sh", **env)
    assert done.returncode == 2
    assert "REFUSED" in done.stdout and "SHARE" in done.stdout
    assert calls == []


def test_full_passes_the_share_to_contrastive_runs_only(repo, fakes):
    done, _, trained = run_script(repo, fakes, "run_full.sh", SHARE="0.5", SEEDS="0")
    assert done.returncode == 0, done.stdout
    assert "--eval-pair-share" not in trained[0]
    assert all("--eval-pair-share 0.5" in c for c in trained[1:])


def test_full_refuses_when_another_run_is_training(repo, fakes):
    done, calls, _ = run_script(repo, fakes, "run_full.sh", SHARE="none",
                                FAKE_PGREP_EXIT="0")
    assert done.returncode == 2 and "another training run" in done.stdout
    assert calls == []


def test_full_refuses_when_check_fails(repo, fakes):
    done, calls, _ = run_script(repo, fakes, "run_full.sh", SHARE="none",
                                FAKE_CHECK_EXIT="1")
    assert done.returncode == 2 and "check failed" in done.stdout
    assert calls == ["check"]


def test_full_skips_finished_runs(repo, fakes):
    (repo / "models" / "b2-mlm").mkdir()
    (repo / "models" / "b2-mlm" / "summary.json").write_text("{}")
    done, _, trained = run_script(repo, fakes, "run_full.sh", SHARE="none", SEEDS="0")
    assert done.returncode == 0, done.stdout
    assert "skip  models/b2-mlm" in done.stdout
    assert [out_of(c) for c in trained] == ["models/b3-con-mlm-seed-0",
                                            "models/b4-con-random-seed-0"]


def test_full_stops_when_mlm_fails(repo, fakes):
    done, _, trained = run_script(repo, fakes, "run_full.sh", SHARE="none",
                                  FAKE_TRAIN_EXIT="1")
    assert done.returncode == 2 and "B2 (MLM) failed" in done.stdout
    assert len(trained) == 1


# -- the working tree, both scripts ------------------------------------------

LONG_RUNS = [("run_full.sh", {"SHARE": "none"}), ("run_e3.sh", {})]


@pytest.mark.parametrize("script,env", LONG_RUNS)
def test_refuses_a_modified_tracked_file(repo, fakes, script, env):
    (repo / "tracked.txt").write_text("edited\n")
    done, calls, _ = run_script(repo, fakes, script, **env)
    assert done.returncode == 2
    assert "-dirty" in done.stdout and "tracked.txt" in done.stdout
    assert calls == []


@pytest.mark.parametrize("script,env", LONG_RUNS)
def test_refuses_a_new_uncommitted_file(repo, fakes, script, env):
    (repo / "experiments" / "run_curve.sh").write_text("#!/bin/sh\n")
    # The old check (git diff --quiet HEAD) saw nothing here, while
    # code_version() would record the runs as dirty.
    old_check = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=repo)
    assert old_check.returncode == 0
    done, calls, _ = run_script(repo, fakes, script, **env)
    assert done.returncode == 2
    assert "-dirty" in done.stdout and "run_curve.sh" in done.stdout
    assert calls == []


@pytest.mark.parametrize("script,env", LONG_RUNS)
def test_ignored_data_and_models_do_not_count(repo, fakes, script, env):
    (repo / "data" / "notes.log").write_text("a log\n")
    (repo / "models" / "scratch").mkdir()
    (repo / "models" / "scratch" / "last.pt").write_text("weights\n")
    done, _, trained = run_script(repo, fakes, script, **{**env, "SEEDS": "0"})
    assert done.returncode == 0, done.stdout
    assert trained


@pytest.mark.parametrize("script,env", LONG_RUNS)
def test_refuses_outside_a_git_checkout(repo, fakes, script, env):
    shutil.rmtree(repo / ".git")
    done, calls, _ = run_script(repo, fakes, script, **env)
    assert done.returncode == 2 and "git status failed" in done.stdout
    assert calls == []


@pytest.mark.parametrize("script,env", LONG_RUNS)
def test_refuses_when_status_fails_but_the_commit_reads(repo, fakes, script, env):
    # A broken index: `git status` fails while `git rev-parse HEAD` still
    # answers, so only the status check can tell that nothing is known.
    (repo / ".git" / "index").write_bytes(b"not an index")
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                          capture_output=True).returncode != 0
    assert _git(repo, "rev-parse", "--short", "HEAD").strip()
    done, calls, _ = run_script(repo, fakes, script, **env)
    assert done.returncode == 2 and "git status failed" in done.stdout
    assert calls == []


@pytest.mark.parametrize("script,env", LONG_RUNS)
def test_refuses_a_repository_with_no_commit(repo, fakes, script, env):
    # Status is clean (everything excluded) but there is no commit to record.
    shutil.rmtree(repo / ".git")
    _git(repo, "init", "-q")
    (repo / ".git" / "info" / "exclude").write_text("*\n")
    assert _git(repo, "status", "--porcelain") == ""
    done, calls, _ = run_script(repo, fakes, script, **env)
    assert done.returncode == 2 and "no commit yet" in done.stdout
    assert calls == []


@pytest.mark.parametrize("script,env,first", [
    ("run_full.sh", {"SHARE": "none"}, "models/b2-mlm"),
    ("run_e3.sh", {}, "models/e3-share-none-seed-0"),
])
def test_stops_before_the_next_run_when_head_moves(repo, fakes, script, env, first):
    before = _git(repo, "rev-parse", "--short", "HEAD").strip()
    done, _, trained = run_script(repo, fakes, script, FAKE_COMMIT_AFTER=first, **env)
    after = _git(repo, "rev-parse", "--short", "HEAD").strip()
    assert done.returncode == 2
    assert f"HEAD moved from {before} to {after}" in done.stdout
    assert [out_of(c) for c in trained] == [first]
    assert (repo / first / "summary.json").exists()

    # Started again on the new commit, by choice: the finished run is kept
    # and skipped, the rest go on.
    again, _, trained = run_script(repo, fakes, script, **env)
    assert again.returncode == 0, again.stdout
    assert first not in [out_of(c) for c in trained]
    assert len(trained) == (6 if script == "run_full.sh" else 7)


@pytest.mark.parametrize("script,env,first", [
    ("run_full.sh", {"SHARE": "none"}, "models/b2-mlm"),
    ("run_e3.sh", {}, "models/e3-share-none-seed-0"),
])
def test_stops_before_the_next_run_when_a_file_appears(repo, fakes, script, env, first):
    done, _, trained = run_script(repo, fakes, script, FAKE_NEW_FILE_AFTER=first, **env)
    assert done.returncode == 2
    assert "-dirty" in done.stdout and "new_module.py" in done.stdout
    assert [out_of(c) for c in trained] == [first]


# -- run_e3.sh ----------------------------------------------------------------

def test_e3_runs_eight_with_the_right_flags(repo, fakes):
    done, calls, trained = run_script(repo, fakes, "run_e3.sh")
    assert done.returncode == 0, done.stdout
    assert calls[0] == "check"
    shares = ("none", "0.25", "0.5", "1.0")
    expected = [(seed, share) for seed in (0, 1) for share in shares]
    assert [out_of(c) for c in trained] == [
        f"models/e3-share-{share}-seed-{seed}" for seed, share in expected]
    for call, (seed, share) in zip(trained, expected):
        assert call.startswith("train contrastive ")
        assert f"--seed {seed}" in call
        assert "--max-steps 1500 --epochs 100 --patience 100" in call
        assert ("--eval-pair-share" not in call) if share == "none" \
            else f"--eval-pair-share {share}" in call
    assert "=== all done" in done.stdout


def test_e3_refuses_when_another_run_is_training(repo, fakes):
    done, calls, _ = run_script(repo, fakes, "run_e3.sh", FAKE_PGREP_EXIT="0")
    assert done.returncode == 2 and "another training run" in done.stdout
    assert calls == []


def test_e3_refuses_when_check_fails(repo, fakes):
    done, calls, _ = run_script(repo, fakes, "run_e3.sh", FAKE_CHECK_EXIT="1")
    assert done.returncode == 2 and "check failed" in done.stdout
    assert calls == ["check"]


@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_e3_logs_the_training_exit_code(repo, fakes, shell):
    # In bash, a $(date) on the same line as $? used to reset it to 0.
    done, _, _ = run_script(repo, fakes, "run_e3.sh", shell=shell, FAKE_TRAIN_EXIT="3")
    assert "(exit 3)" in done.stdout
    assert "(exit 0)" not in done.stdout
