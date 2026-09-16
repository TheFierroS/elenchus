#!/bin/sh
# Full training, B2-B4. Protocol in docs/experiments.md ("Full training").
#
#   B2  MLM pre-training, seed 0, early stopping          models/b2-mlm
#   B3  contrastive from B2's best.pt, seeds 0 1 2        models/b3-con-mlm-seed-N
#   B4  contrastive from random weights, seeds 0 1 2      models/b4-con-random-seed-N
#
# SHARE must be given: the eval-pair-share E3 chose ("none" for uniform).
# Runs whose summary.json exists are skipped, so the script can be started
# again after an interruption; a run cut short starts over.
#
#   SHARE=none nohup sh experiments/run_full.sh > data/full.log 2>&1 &
#   python experiments/full_results.py

set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
ELENCHUS="${ELENCHUS:-elenchus}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-3}"

refuse() { echo "REFUSED: $*"; exit 2; }

# -- before anything: every check here protects hours of GPU time
[ -n "${SHARE:-}" ] || refuse "set SHARE to E3's verdict (none, 0.25, 0.5 or 1.0)"
case "$SHARE" in
  none) share_flag="" ;;
  0.25|0.5|1.0) share_flag="--eval-pair-share $SHARE" ;;
  *) refuse "SHARE must be none, 0.25, 0.5 or 1.0, got '$SHARE'" ;;
esac
if ! git diff --quiet HEAD -- . 2>/dev/null; then
  refuse "tracked files have uncommitted changes; runs would be recorded as -dirty"
fi
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
  echo "=== $(date '+%H:%M:%S') start $out"
  "$ELENCHUS" train "$@" --out "$out" > "$log" 2>&1
  code=$?
  echo "=== $(date '+%H:%M:%S') end   $out (exit $code)"
  return $code
}

echo "=== full training: SHARE=$SHARE SEEDS='$SEEDS' EPOCHS=$EPOCHS PATIENCE=$PATIENCE"
echo "=== commit $(git rev-parse --short HEAD)"

run models/b2-mlm data/full-b2-mlm.log mlm \
    --seed 0 --epochs "$EPOCHS" --patience "$PATIENCE" \
  || refuse "B2 (MLM) failed; B3 needs its checkpoint"

for seed in $SEEDS; do
  # shellcheck disable=SC2086
  run "models/b3-con-mlm-seed-$seed" "data/full-b3-seed-$seed.log" contrastive \
      --init models/b2-mlm/best.pt --seed "$seed" \
      --epochs "$EPOCHS" --patience "$PATIENCE" $share_flag
done

for seed in $SEEDS; do
  # shellcheck disable=SC2086
  run "models/b4-con-random-seed-$seed" "data/full-b4-seed-$seed.log" contrastive \
      --seed "$seed" --epochs "$EPOCHS" --patience "$PATIENCE" $share_flag
done

echo "=== all done $(date '+%H:%M:%S')"
