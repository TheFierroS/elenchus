# Checks shared by the long-run scripts (run_e3.sh, run_full.sh), sourced
# from the repo root after `cd`:  . experiments/guard.sh
#
# A run records code_version(), which is the commit plus "-dirty" when
# `git status --porcelain` prints anything - tracked changes and new,
# uncommitted files alike (data/ and models/ are ignored, so they do not
# count). The checks here use that same definition, so a script never starts
# a run that would be recorded as dirty.
#
# They are repeated before every run, not only at the start: during E3 a
# commit landed while the script was going, and its eight runs were recorded
# under two commits. A script now stops before its next run if the tree is no
# longer clean or HEAD has moved; finished runs are kept, and starting the
# script again continues on the new commit, by choice.

refuse() { echo "REFUSED: $*"; exit 2; }

# guard_clean: git must answer, and the tree must be clean.
guard_clean() {
  guard_status=$(git status --porcelain 2>/dev/null) \
    || refuse "git status failed (not a git checkout, or a broken one); the tree cannot be shown clean"
  if [ -n "$guard_status" ]; then
    refuse "uncommitted changes or new files; runs would be recorded as -dirty:
$guard_status"
  fi
}

# guard_start: the checks before the first run; remembers the commit.
guard_start() {
  guard_clean
  GUARD_HEAD=$(git rev-parse --short HEAD 2>/dev/null) \
    || refuse "git rev-parse HEAD failed (no commit yet?); runs would record no commit"
}

# guard_before_run: the same checks again, and the commit must not have moved.
guard_before_run() {
  guard_clean
  guard_now=$(git rev-parse --short HEAD 2>/dev/null) \
    || refuse "git rev-parse HEAD failed (no commit yet?); runs would record no commit"
  if [ "$guard_now" != "$GUARD_HEAD" ]; then
    refuse "HEAD moved from $GUARD_HEAD to $guard_now while runs were going; finished runs are kept, start the script again to continue on $guard_now"
  fi
}
