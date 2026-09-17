#!/bin/sh
# E3: which level pairs to train on. Protocol in docs/experiments.md (E3),
# written before these runs. Eight contrastive runs from random weights,
# 1,500 steps each, no early stopping: eval-pair-share unset / 0.25 / 0.5 /
# 1.0, seeds 0 and 1. A run whose summary.json exists is skipped, so the
# script can be started again after an interruption. The working tree must
# be clean and stay on one commit (experiments/guard.sh).
#
#   nohup sh experiments/run_e3.sh > data/e3.log 2>&1 &
#   python experiments/e3_results.py      (when it has finished)

set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
ELENCHUS="${ELENCHUS:-elenchus}"

. experiments/guard.sh     # refuse, guard_start, guard_before_run

guard_start
# [b]in: the pattern must not match itself, or any command line quoting it.
if pgrep -f "[b]in/elenchus train" > /dev/null 2>&1; then
  refuse "another training run is on the GPU (see: pgrep -af 'bin/elenchus train')"
fi
"$ELENCHUS" check > data/e3-check.log 2>&1 || refuse "elenchus check failed; see data/e3-check.log"

for seed in 0 1; do
  for share in none 0.25 0.5 1.0; do
    out="models/e3-share-$share-seed-$seed"
    if [ -f "$out/summary.json" ]; then
      echo "skip $out (already done)"
      continue
    fi
    guard_before_run
    flag=""
    [ "$share" != "none" ] && flag="--eval-pair-share $share"
    echo "=== $(date '+%H:%M:%S') start $out"
    # shellcheck disable=SC2086
    "$ELENCHUS" train contrastive --out "$out" --seed "$seed" \
      --max-steps 1500 --epochs 100 --patience 100 $flag \
      > "data/e3-share-$share-seed-$seed.log" 2>&1
    # Kept before the next line: in bash, $(date) there would reset $?.
    code=$?
    echo "=== $(date '+%H:%M:%S') end   $out  (exit $code)"
  done
done
echo "=== all done $(date '+%H:%M:%S')"
