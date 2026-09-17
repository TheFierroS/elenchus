#!/bin/sh
# B5 memory plan: does the ~11M model fit the card as it is?
# Protocol in docs/experiments.md ("B5 - model size, memory first").
#
# Three 50-step smoke runs, each with nvidia-smi sampling the whole card once
# a second (PyTorch's figures cover only its own tensors and cache; whether
# the card overflows into shared memory depends on everything on it):
#   ref    3.6M contrastive                       (the reference)
#   mlm11  11M MLM         d_model 384, 6 layers, 6 heads, d_ff 1536
#   con11  11M contrastive the same size
#   mlm11ac, con11ac  the same two with --checkpoint-activations, the plan if
#          11M does not fit (17 September: mlm11 left 0.29 GiB, con11 failed)
#   mlm15ac, con15ac  ~15M (14,910,401 with this vocabulary: 8 layers of the
#          same width), with --checkpoint-activations; measured, not decided on
# Every flag other than size stays at its default: batch 64, length 1024.
# The working tree must be clean and stay on one commit, and ELENCHUS_DB must
# be set (experiments/guard.sh).
#
#   nohup sh experiments/b5_memory.sh > data/b5.log 2>&1 &    (~6 min)

set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
ELENCHUS="${ELENCHUS:-elenchus}"
NVSMI="${NVSMI:-nvidia-smi}"
SIZE="--d-model 384 --layers 6 --heads 6 --d-ff 1536"
SIZE15="--d-model 384 --layers 8 --heads 6 --d-ff 1536"
IDLE_SECONDS="${IDLE_SECONDS:-5}"    # the card sampled before each run
TAIL_SECONDS="${TAIL_SECONDS:-3}"    # and after it

. experiments/guard.sh     # refuse, guard_start, guard_before_run

guard_start
# [b]in: the pattern must not match itself, or any command line quoting it.
if pgrep -f "[b]in/elenchus train" > /dev/null 2>&1; then
  refuse "another training run is on the GPU; its memory would be counted too"
fi

measure() {  # measure <name> <train args...>
  name="$1"; shift
  out="models/b5-$name"; csv="data/b5-$name-gpu.csv"
  guard_before_run
  rm -rf "$out"
  "$NVSMI" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l 1 \
    > "$csv" &
  sampler=$!
  sleep "$IDLE_SECONDS"                    # the card before the run
  echo "=== $(date '+%H:%M:%S') $name"
  "$ELENCHUS" train "$@" --out "$out" --epochs 1 --max-steps 50 \
    > "data/b5-$name.log" 2>&1
  echo "    exit $?"
  sleep "$TAIL_SECONDS"
  kill "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null
}

# shellcheck disable=SC2086
measure ref   contrastive
# shellcheck disable=SC2086
measure mlm11 mlm $SIZE
# shellcheck disable=SC2086
measure con11 contrastive $SIZE
# shellcheck disable=SC2086
measure mlm11ac mlm $SIZE --checkpoint-activations
# shellcheck disable=SC2086
measure con11ac contrastive $SIZE --checkpoint-activations
# shellcheck disable=SC2086
measure mlm15ac mlm $SIZE15 --checkpoint-activations
# shellcheck disable=SC2086
measure con15ac contrastive $SIZE15 --checkpoint-activations

python3 - <<'PY'
import json
from pathlib import Path

HEADROOM_GIB = 0.5
print(f"\n{'run':7} {'torch alloc':>11} {'torch resv':>10} {'card idle':>9} "
      f"{'card peak':>9} {'card total':>10} {'free at peak':>12}  fits  "
      f"{'epoch s':>7}")
for name in ("ref", "mlm11", "con11", "mlm11ac", "con11ac", "mlm15ac", "con15ac"):
    summary_path = Path(f"models/b5-{name}/summary.json")
    rows = [line.split(",") for line in Path(f"data/b5-{name}-gpu.csv").read_text().split("\n")
            if line.strip()]
    used = [int(r[0]) / 1024 for r in rows]
    total = int(rows[0][1]) / 1024 if rows else float("nan")
    if not summary_path.exists() or not used:
        print(f"{name:7} did not finish (see data/b5-{name}.log)")
        continue
    s = json.loads(summary_path.read_text())
    idle = min(used[:5]) if len(used) >= 5 else min(used)
    peak = max(used)
    free = total - peak
    trained = [h for h in s.get("history", []) if h.get("epoch", -1) >= 0]
    seconds = f"{trained[0]['seconds']:>7.0f}" if trained else f"{'-':>7}"
    print(f"{name:7} {s['peak_allocated_gib']:>11.2f} {s['peak_reserved_gib']:>10.2f} "
          f"{idle:>9.2f} {peak:>9.2f} {total:>10.2f} {free:>12.2f}  "
          f"{'yes ' if free >= HEADROOM_GIB else 'NO  '}  {seconds}")
print(f"\nfits: at least {HEADROOM_GIB} GiB of the card left free at the peak (GiB throughout);"
      "\nepoch s: the 50 steps and their validation, in seconds")
PY
