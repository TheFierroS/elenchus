#!/bin/sh
# Full training, B2-B4. Protocol in docs/experiments.md ("Full training").
#
#   B2  MLM pre-training, seed 0                          models/b2-mlm
#   B3  contrastive from B2's best.pt, seeds 0 1           models/b3-con-mlm-seed-N
#   B4  contrastive from random weights, seeds 0 1         models/b4-con-random-seed-N
#
# One fixed budget for every run, no early stopping, read at best.pt: L2
# showed a run stopped on patience is measuring the stopping rule, not the
# thing under test. STEPS and EVERY come from the corpus (30 epochs of the
# train split); they must be the same for every run being compared.
#
# SIZE names a model size for stage 2 and is part of the output directory,
# so 3.6M and 11M runs do not overwrite each other:
#
#   SIZE=11m  D_MODEL=384 LAYERS=6 HEADS=6 D_FF=1536 CHECKPOINT=1
#   SIZE=15m  D_MODEL=384 LAYERS=8 HEADS=6 D_FF=1536 CHECKPOINT=1
#
# SHARE must be given: the eval-pair-share E3 chose ("none" for uniform).
# Runs whose summary.json exists are skipped, so the script can be started
# again after an interruption; a run cut short starts over. The working tree
# must be clean and stay on one commit (experiments/guard.sh).
#
#   SHARE=none nohup sh experiments/run_full.sh > data/full.log 2>&1 &
#   python experiments/full_results.py

set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
ELENCHUS="${ELENCHUS:-elenchus}"
SEEDS="${SEEDS:-0 1}"
STEPS="${STEPS:-16290}"          # 30 epochs of 543 steps on the wave 3 corpus
EVERY="${EVERY:-543}"            # one validation an epoch
SIZE="${SIZE:-}"                 # "" is the 3.6M default; else part of the name
D_MODEL="${D_MODEL:-}"
LAYERS="${LAYERS:-}"
HEADS="${HEADS:-}"
D_FF="${D_FF:-}"
CHECKPOINT="${CHECKPOINT:-}"

budget="--max-steps $STEPS --eval-every $EVERY --epochs 100000 --patience 100000"
shape=""
[ -n "$D_MODEL" ] && shape="$shape --d-model $D_MODEL"
[ -n "$LAYERS" ] && shape="$shape --layers $LAYERS"
[ -n "$HEADS" ] && shape="$shape --heads $HEADS"
[ -n "$D_FF" ] && shape="$shape --d-ff $D_FF"
[ -n "$CHECKPOINT" ] && shape="$shape --checkpoint-activations"
suffix=""
[ -n "$SIZE" ] && suffix="-$SIZE"

. experiments/guard.sh     # refuse, guard_start, guard_before_run

# -- before anything: every check here protects hours of GPU time
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
"$ELENCHUS" check > data/full-check.log 2>&1 || refuse "elenchus check failed; see data/full-check.log"

run() {  # run <out> <log> <args...>
  out="$1"; log="$2"; shift 2
  if [ -f "$out/summary.json" ]; then
    echo "skip  $out (already done)"
    return 0
  fi
  guard_before_run
  echo "=== $(date '+%H:%M:%S') start $out"
  "$ELENCHUS" train "$@" --out "$out" > "$log" 2>&1
  code=$?
  echo "=== $(date '+%H:%M:%S') end   $out (exit $code)"
  return $code
}

echo "=== full training: SHARE=$SHARE SEEDS='$SEEDS' STEPS=$STEPS EVERY=$EVERY SIZE='${SIZE:-3.6m}'"
echo "=== commit $(git rev-parse --short HEAD)"

# shellcheck disable=SC2086
run "models/b2-mlm$suffix" "data/full-b2-mlm$suffix.log" mlm \
    --seed 0 $budget $shape \
  || refuse "B2 (MLM) failed; B3 needs its checkpoint"

for seed in $SEEDS; do
  # shellcheck disable=SC2086
  run "models/b3-con-mlm$suffix-seed-$seed" "data/full-b3$suffix-seed-$seed.log" \
      contrastive --init "models/b2-mlm$suffix/best.pt" --seed "$seed" \
      $budget $shape $share_flag
done

for seed in $SEEDS; do
  # shellcheck disable=SC2086
  run "models/b4-con-random$suffix-seed-$seed" "data/full-b4$suffix-seed-$seed.log" \
      contrastive --seed "$seed" $budget $shape $share_flag
done

echo "=== all done $(date '+%H:%M:%S')"
