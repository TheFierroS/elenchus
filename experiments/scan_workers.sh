#!/bin/sh
# Where does spreading the measurement over processes stop paying?
#
# Two changes were made to the cost of a measurement at once - keeping the
# stack fill, and judging pairs across processes - and the run that followed
# changed both, so neither has a number of its own. That is the mistake F27
# recorded and this undoes it: one axis, several points, small sample, read
# together.
#
# R0 rather than V0 because it is a third of the work and asks the emulator
# for exactly the same thing. --count 100 rather than 500 because the shape
# of the curve is what is wanted, not the verdicts; the verdict counts at
# this size mean nothing and are not read.
#
#   sh experiments/scan_workers.sh 2>&1 | tee data/scan_workers.log
#
# About six minutes. Reads ELENCHUS_DB from the environment like everything
# else.

set -e

COUNT=${COUNT:-100}
WORKERS=${WORKERS:-"1 2 4 8"}

echo "machine:"
echo "  cores visible to WSL : $(nproc)"
echo "  memory               :"
free -m | sed 's/^/    /'
echo

echo "scanning --workers over $COUNT pairs"
echo
printf '%8s  %10s  %12s\n' "workers" "seconds" "pairs/s"

for w in $WORKERS; do
    start=$(date +%s.%N)
    elenchus verify-r0 --count "$COUNT" --workers "$w" \
        --out "data/scan-w$w.json" > "data/scan-w$w.log" 2>&1
    end=$(date +%s.%N)
    secs=$(echo "$end - $start" | bc)
    rate=$(echo "scale=2; $COUNT / $secs" | bc)
    printf '%8s  %10.1f  %12s\n' "$w" "$secs" "$rate"
done

echo
echo "The verdicts are in data/scan-w*.json and should be identical across"
echo "the four - they are judged from the same tasks whatever ran them. If"
echo "they are not, the parallel path is wrong and nothing else here matters:"
for w in $WORKERS; do
    printf '  workers %s: ' "$w"
    python - "data/scan-w$w.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdicts"], "forced:", d["forced_survivals"])
PY
done

echo
echo "Reading it: the serial row is what the stack-fill cache is worth on its"
echo "own, against the 332 s / 1.5 pairs-per-second of the run before it. The"
echo "rest is what processes are worth, and where the curve flattens is where"
echo "the work stops being CPU-bound and starts waiting on something shared."
