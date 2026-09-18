#!/bin/sh
# The learning curve: is there enough data? Protocol in docs/experiments.md,
# L (written before those runs, amended 17 September), and L2, the short
# repeat on the corpus wave 2 built.
#
# Every run trains contrastively from random weights for the same budget of
# steps, with the learning rate decaying over all of it, validates every 172
# steps (one full-train epoch) and keeps its best validation (best.pt); none
# stops early. Eight runs, the ones that decide first:
#
#   full      100% of train             seeds 0 1
#   identity  50%, 25% of functions     seed 0
#   package   50%, 25% of packages      seeds 0 1
#
# SHAPE=0 leaves out the three 25% runs, which describe the curve's shape and
# never decided anything: that is L2's five runs. PREFIX names the output
# directories, so a second curve cannot silently reuse the first one's runs -
# its models are trained on another split and would be skipped as "done".
#
#   L : SHARE=none sh experiments/run_curve.sh
#   L2: SHARE=none PREFIX=curve2 STEPS=11550 EVERY=385 SHAPE=0 \
#         nohup sh experiments/run_curve.sh > data/curve2.log 2>&1 &
#
# SHARE must be given: the eval-pair-share E3 chose ("none" for uniform).
# Runs whose summary.json exists are skipped, so the script can be started
# again after an interruption; a run cut short starts over. The working tree
# must be clean and stay on one commit, and ELENCHUS_DB must be set
# (experiments/guard.sh).
#
#   SHARE=none nohup sh experiments/run_curve.sh > data/curve.log 2>&1 &
#   python experiments/curve_results.py

set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
ELENCHUS="${ELENCHUS:-elenchus}"
STEPS="${STEPS:-5160}"         # 30 full-train epochs of 172 steps
EVERY="${EVERY:-172}"
PREFIX="${PREFIX:-curve}"
SHAPE="${SHAPE:-1}"

. experiments/guard.sh     # refuse, guard_start, guard_before_run

[ -n "${SHARE:-}" ] || refuse "set SHARE to E3's verdict (none, 0.25, 0.5 or 1.0)"
case "$SHARE" in
  none) share_flag="" ;;
  0.25|0.5|1.0) share_flag="--eval-pair-share $SHARE" ;;
  *) refuse "SHARE must be none, 0.25, 0.5 or 1.0, got '$SHARE'" ;;
esac
guard_start
# [b]in: the pattern must not match itself, or any command line quoting it.
if pgrep -f "[b]in/elenchus train" > /dev/null 2>&1; then
  refuse "another training run is on the GPU (see: pgrep -af 'bin/elenchus train')"
fi
"$ELENCHUS" check > data/curve-check.log 2>&1 || refuse "elenchus check failed; see data/curve-check.log"

run() {  # run <name> <seed> [<fraction> <unit>]
  name="$1"; seed="$2"
  out="models/$PREFIX-$name-seed-$seed"
  if [ -f "$out/summary.json" ]; then
    echo "skip  $out (already done)"
    return 0
  fi
  guard_before_run
  subset=""
  [ $# -eq 4 ] && subset="--train-fraction $3 --train-unit $4"
  echo "=== $(date '+%H:%M:%S') start $out"
  # --epochs and --patience only need to be out of the way: the step budget
  # ends every run, and no run stops early.
  # shellcheck disable=SC2086
  "$ELENCHUS" train contrastive --out "$out" --seed "$seed" \
    --max-steps "$STEPS" --eval-every "$EVERY" --epochs 100000 --patience 100000 \
    $subset $share_flag > "data/$PREFIX-$name-seed-$seed.log" 2>&1
  code=$?
  echo "=== $(date '+%H:%M:%S') end   $out (exit $code)"
}

echo "=== learning curve: SHARE=$SHARE STEPS=$STEPS EVERY=$EVERY PREFIX=$PREFIX SHAPE=$SHAPE"
echo "=== commit $(git rev-parse --short HEAD)"

# The runs the decision reads, first; the 25% runs give the curve's shape.
run full 0
run identity-0.5 0 0.5 identity
run package-0.5 0 0.5 package
run package-0.5 1 0.5 package
run full 1

if [ "$SHAPE" != "0" ]; then
  run identity-0.25 0 0.25 identity
  run package-0.25 0 0.25 package
  run package-0.25 1 0.25 package
fi

echo "=== all done $(date '+%H:%M:%S')"
