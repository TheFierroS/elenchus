# Open experiments

Decisions that sounded right but have not been measured. Each one is taken
with a default so work can continue, and each names the measurement that will
settle it. Nothing moves from here into "decided" on argument alone.

The rule for every entry: tune on `val`, never on `test`. An entry closes with
the number that decided it, the commit it was measured at, and the corpus
fingerprint.

---

## Encoder pre-training (MLM)

### E1 — Mask whole instructions instead of single tokens

**Default:** mask 15% of tokens independently (BERT: 80% `[MASK]`, 10% random,
10% unchanged).

**Doubt:** single-token masking may be too easy to be useful. In
`mov REG64 [MASK]` the missing operand is half-predictable from `mov` alone,
so the model can score well by learning local grammar and nothing about what
the function does. Masking a whole instruction (`[MASK] [MASK] [MASK]`) forces
it to use the surrounding code instead.

**Risk if wrong:** a pre-training loss that falls nicely while teaching
shortcuts - the kind of learning that looks fine and is not.

**Settles it:** pre-train twice with the same seed and token budget, once per
scheme; fine-tune both contrastively the same way; compare val MRR. Loss
values are not comparable between the two schemes and must not be used.

### E2 — Pre-train on unlabelled functions too

**Default:** MLM uses only dataset-eligible train rows.

**Doubt:** MLM needs no ground truth. The stripped train binaries hold more
functions than the eligible rows (18,097 vs 13,045 on the 28-package corpus),
and more data is the thing this corpus is shortest of.

**Settles it:** same comparison as E1. Requires dedup and the same
runtime exclusion the dataset applies, so no toolchain code enters.

---

## Contrastive training

### E3 — Which optimisation-level pairs to sample

**Default:** for each function, two distinct levels chosen uniformly, so all
six pairs appear.

**Doubt:** the evaluation task is `-O0 → -O3`, which is one pair in six.
Uniform sampling may under-train the case that is measured; weighting towards
it may over-fit the exam.

**Settles it:** val MRR on `-O0 → -O3`, and also on the other pairs, so a gain
on the exam that costs everything else is visible.

### E4 — Hard negatives: functions per package in a batch

**Default:** batches are built from several packages, up to 8 functions each.

**Doubt:** too few per package and the negatives are easy; too many and a
batch is one library's style, which may teach package identification instead
of function identity.

**Settles it:** val MRR at 4, 8, 16 per package, and fully random batches.

### E5 — Temperature

**Default:** to be set when training code lands.

**Settles it:** val sweep.

---

## Model input and output

### E6 — Long functions: truncate or split

**Default:** keep the first 1024 tokens (`[CLS]` included). Measured coverage
at 1024: 94.6% of `-O0` functions, 89.8% of `-O3`.

**Doubt:** the tail of a long function is lost, and it is lost more often on
the `-O3` side, which is the pool.

**Settles it:** val MRR restricted to functions longer than the limit, truncate
vs split-and-average.

### E7 — One vector per function: `[CLS]` or mean of tokens

**Settles it:** val MRR.

### E8 — Instruction boundaries

**Doubt:** normalised tokens are a flat stream. A mnemonic starts each
instruction so boundaries are recoverable, but the model is not told. An
instruction-index embedding would tell it.

**Settles it:** val MRR, only if E1-E7 leave the model short of the bar.

### E9 — Vocabulary threshold

**Default:** a token must appear in at least 5 distinct train functions.

**Settles it:** the vocabulary's own coverage report on val - the share of
tokens falling to `[UNK]` and `IMPORT:<rare>` - then val MRR at 2, 5, 10.

---

## Open questions (not experiments, but unexplained)

### Q1 — Lower match rates on some packages — RESOLVED

The 45-package build matched 98.8% of ground truth; some packages far less
(mujs -O3 72%, libtomcrypt -O2/-O3 67%, flecs -O3 93%).

**First hypothesis, refuted.** "Pointer-registered leaf functions have no
.pdata entry, so discovery has nothing to start from." In the sandbox builds
of libsodium, libtomcrypt and monocypher, every DWARF function at every level
has a .pdata entry - 100%, no exceptions.

**Measured on the corpus** (`pe_evidence.py`): the unmatched addresses of
libtomcrypt -O2/-O3, mujs -O3, duktape -O1 are all in .pdata, none is covered
by any Ghidra function, and none has a basic block. Ghidra never disassembled
them. flecs and libuv added a handful of true merges (7) at nop padding.

**Cause.** Ghidra finds functions in a stripped PE by following calls, and
none of its analyzers reads .pdata (checked: no analyzer name mentions
exception, pdata or unwind). A function reached only through a pointer -
libtomcrypt's descriptor tables, mujs built-ins - is never found. The old
claim that .pdata kept discovery intact (zlib, 269 → 269) credited the wrong
mechanism: zlib's functions are called directly.

**Fix, measured before adopting** on libtomcrypt -O3 (`pdata_seed_probe.py`):
creating functions at the 251 .pdata starts with no function, then
re-running analysis, recovered 152/152 missing ground-truth functions and
changed the listing of 0 of the 436 functions found before. Adds 5 seconds.
Now `seed_functions_from_pdata` in extraction.

**Consequence for the corpus.** Because existing listings do not change,
only packages with unmatched ground truth need rescanning. Rescanned 11
packages (`build-corpus --rescan`, 394 s): 37,295 of 37,341 ground-truth
functions matched (99.88%); libtomcrypt -O2/-O3 67% → 100%, mujs -O3 72% →
99.7%, duktape -O1 95% → 100%. The 46 left are mostly true merges at nop
padding (flecs).

**Why it matters beyond the corpus.** Virtual methods, window procedures,
callbacks and crackme validation routines are all reached through pointers.
The agent would have been blind to them.

### Q1b — "IAT calls classified as internal" — NOT AN IMPORT PROBLEM

An old note said some `call [0x...]` were `FUNC_INTERNAL` instead of
`IMPORT:`. Measured (`pe_evidence.py`): of 15,400 absolute memory calls not
resolved as imports, 2 point into the import table; the rest point into
.data or .bss - global function pointers such as flecs' `ecs_os_api`, sqlite's
allocator configuration, tree-sitter's `ts_current_malloc`. No import name
was lost. Ghidra reported them as internal because it read the pointer's
initial value; the normaliser now names them `FUNC_INDIRECT`.

### Q2 — Ghidra heap exhaustion on quickjs -O1

quickjs `-O1` stripped fails to scan with `OutOfMemoryError: Java heap space`.
`-O0`, `-O2` and `-O3` of the same package scan normally under the same limit.

**Measured.**

| heap | outcome | time to failure |
|---|---|---|
| `-Xmx4G` | out of memory | about 2 minutes |
| `-Xmx6G` | out of memory | 45 seconds |

Under 6G, process memory sampled every 5 seconds went 0.3 GB → 1.0 → 4.2 →
6.8 GB within 15 seconds, then stayed flat at the ceiling for 25 seconds until
the error. Memory does not creep up across a large binary; it explodes at
once and holds.

**Ruled out: "the heap is a bit small."** More heap failed sooner, not later.
(The second attempt may have skipped import - Ghidra had kept a project from
the first - which would explain the shorter time but not the failure.)

**Ruled out: "-O1 has an unusually large function."** Instruction counts of
the largest functions, from the debug twins:

| level | largest | second |
|---|---|---|
| -O0 | JS_CallInternal 16,307 | resolve_labels 2,980 |
| -O1 | JS_CallInternal 9,030 | resolve_labels 3,472 |
| -O2 | JS_CallInternal 10,116 | resolve_labels 3,611 |
| -O3 | JS_CallInternal 11,844 | js_create_function 3,925 |

-O1's largest function is the smallest of the four.

**Still open.** The stack trace points at `StackVariableAnalyzer`, running in
Ghidra's concurrent analysis queue. Something about the shape of some -O1
function - not its size - sends that analysis into runaway allocation.

**Decision for now:** quickjs is kept at three levels. -O1 is one training
view; the evaluation task (-O0 → -O3) is unaffected.

**Not done, and why:** turning the stack analyser off for this one binary.
Ghidra's stack analysis can change how operands are written, and the
normaliser reads operand text. One binary extracted differently from the
other 359 would give its functions different tokens, which the encoder would
learn as a compiler difference - a silent inconsistency. Only acceptable if
a small binary extracted with and without the analyser is shown to produce
identical listings.

**Why it matters later.** The agent will meet binaries like this. A scan that
dies on one pathological function must cost that function, not the binary -
a requirement for graceful degradation (week 11), not something to fix in
the corpus.
