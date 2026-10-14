#!/usr/bin/env bash
# Local smoke test: 40 true pairs (V0) and 40 false pairs (R0), about a minute.
#
# CI has no corpus, so this is the check run by hand before a release, a long
# run, or any change that could move a verdict. verify-v0 exits 0 whatever it
# found, so the verdicts are read from the JSON here rather than trusted from
# the exit code.
#
# It fails hard on what must never happen - a true pair refuted, or an R0 pair
# silently dropped - and separately on a drift from the expected counts. A
# drift can be harmless: the 2 s wall clock is the one load-dependent result,
# and a pair that times out moves between buckets. It is also what a rule
# change does on purpose, which is why it has its own exit code.
#
# Exit codes: 0 as expected; 1 DISAGREED, a dropped pair, or a failed run;
# 2 the counts drifted from the expected ones below.
#
# The expected counts are v0.7.1's on seed 0's first 40 pairs. A change meant
# to move verdicts updates them in the same commit as its measurement.
set -euo pipefail

: "${ELENCHUS_DB:?set ELENCHUS_DB to the corpus database first}"

COUNT=40
OUT=data/smoke
mkdir -p "$OUT"

# The numbers belong to a commit only if the tree is that commit. git diff
# alone misses staged changes and new files, so ask git status instead
# (data/ is ignored, so the run's own output does not count).
STATE=""
if [ -n "$(git status --porcelain)" ]; then
    STATE=" (dirty)"
fi
echo "smoke: $(git rev-parse --short HEAD)$STATE, $COUNT pairs each"

export PYTHONFAULTHANDLER=1
if ! elenchus verify-v0 --count "$COUNT" --out "$OUT/v0.json" > "$OUT/v0.log" 2>&1; then
    echo "smoke: verify-v0 failed, see $OUT/v0.log"
    exit 1
fi
if ! elenchus verify-r0 --count "$COUNT" --out "$OUT/r0.json" > "$OUT/r0.log" 2>&1; then
    echo "smoke: verify-r0 failed, see $OUT/r0.log"
    exit 1
fi

python - "$OUT/v0.json" "$OUT/r0.json" "$COUNT" <<'EOF'
import json
import sys

EXPECTED_AGREED = {"bare": 14, "stubbed": 18, "chained": 19}
EXPECTED_R0 = {"refuted": 10, "survived": 1}

v0 = json.load(open(sys.argv[1]))
r0 = json.load(open(sys.argv[2]))
count = int(sys.argv[3])
fatal, drift = [], []

for layer, expected in EXPECTED_AGREED.items():
    report = v0[layer]
    # agreement is a Counter written as a dict: a zero count has no key.
    disagreed = report["agreement"].get("disagreed", 0)
    if disagreed or report["disagreements"]:
        fatal.append(f"V0 {layer}: {disagreed} DISAGREED")
        for d in report["disagreements"]:
            fatal.append(f"    {d['package']}/{d['name']}: {d['detail']}")
    agreed = report["agreement"].get("agreed", 0)
    coverage = 100 * agreed / (report["total"] or 1)
    timed_out = report["buckets"].get("timed out", 0)
    print(f"V0 {layer:8} agreed {agreed:3} ({coverage:4.1f}%), "
          f"disagreed {disagreed}, timed out {timed_out}")
    if agreed != expected:
        drift.append(f"V0 {layer}: agreed {agreed}, expected {expected}")

# R0 leaves out a pair it could not set up rather than counting it
# (r0.judge_false_pair), so a total short of the count is a silent loss.
if r0["total"] != count:
    fatal.append(f"R0: judged {r0['total']} of {count} pairs, "
                 f"{count - r0['total']} dropped without a trace")
verdicts = r0["verdicts"]
print(f"R0 refuted {verdicts.get('refuted', 0)}, "
      f"survived {verdicts.get('survived', 0)} "
      f"({r0['forced_survivals']} forced), "
      f"inconclusive {verdicts.get('inconclusive', 0)}")
for verdict, expected in EXPECTED_R0.items():
    if verdicts.get(verdict, 0) != expected:
        drift.append(f"R0 {verdict}: {verdicts.get(verdict, 0)}, "
                     f"expected {expected}")

if fatal:
    print("SMOKE FAILED")
    print("\n".join(fatal))
    sys.exit(1)
if drift:
    print("SMOKE DRIFTED - a timed-out pair can do this; read the "
          "'timed out' counts, then run it again")
    print("\n".join(drift))
    sys.exit(2)
print("smoke ok")
EOF
