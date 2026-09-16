# Open experiments

Decisions that sounded right but have not been measured. Each one is taken
with a default so work can continue, and each names the measurement that will
settle it. Nothing moves from here into "decided" on argument alone.

The rule for every entry: tune on `val`, never on `test`. An entry closes with
the number that decided it, the commit it was measured at, and the corpus
fingerprint.

The scripts that run the protocols and produced the findings below live in
`experiments/`, run from the repo root; their logs and outputs go to
`data/`, which is not committed.

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

**Measured (at 1da5e5e, fingerprint `47c82734a3cb7f08`).** Uniform holds
within a function, not across the dataset. Of 11,009 pairable train
functions only 5,561 have both `-O0` and `-O3`; many small functions are
inlined away from `-O2` up and can only form O0-O1. Counted over 33,024 draws
of the real `PairSampler` (3 epochs), agreeing with the exact expectation:

| level pair | share of draws | identical code |
|---|---|---|
| O0-O1 | 46.0% | 0.0% |
| O0-O2 | 15.2% | 0.0% |
| O1-O2 | 11.3% | 1.3% |
| O0-O3 | **10.1%** | 0.0% |
| O2-O3 | 10.0% | 43.1% |
| O1-O3 | 7.5% | 1.2% |

So training spends 46% of its pairs on the pair nearest to identity and 10%
on the pair the exam asks. That O0-O1 is "easy" is a hypothesis; O0-O1 pairs
still teach that register choice and frame layout do not change identity,
and a sampler that only drew O0-O3 would halve the pairable data. Moved to
the front of the experiments.

**Option:** `--eval-pair-share P` draws O0-O3 with probability P wherever a
function has both levels, and otherwise draws uniformly; unset, the sampler
is exactly the uniform one. Each trained epoch records the level pairs it
drew (`history[].pairs`), so the share actually trained on is visible.

**Check on real data (run 509, P = 0.5, 25% of train).** 29.9% of draws were
O0-O3, against 30.3% computed (50.5% of pairable functions have both levels;
among them uniform draws O0-O3 20% of the time; 0.505 × (0.5 + 0.5 × 0.20)).
Even P = 1 cannot exceed ~50%.

**Protocol (written 16 September, before any E3 run).**

- Eight contrastive runs from random weights (not the MLM probe, which saw
  all of train and would also colour the learning curve): P unset, 0.25,
  0.5, 1.0, each with seeds 0 and 1. `experiments/run_e3.sh`.
- Fixed budget: 1,500 steps, `--epochs 100 --patience 100`, no early stop.
  Each run is read at its last checkpoint (`last.pt`), not `best.pt`, so
  every run is measured at the same point and none gets more chances.
- Measured on val by `experiments/e3_results.py`: O0→O3 MRR over queries and over
  packages; O0→O2 and O1→O3 MRR over queries. A run that did not draw
  1,500 × 64 pairs is flagged.
- M = max(0.005, 2 × |seed 0 − seed 1| of the uniform runs' O0→O3 query MRR).
- A value of P becomes the default only if, for both seeds: (1) its O0→O3
  query MRR beats uniform by more than M; (2) its O0→O3 package MRR is not
  below uniform by more than M; (3) its O0→O2 and O1→O3 query MRR are not
  below uniform by more than M. If several qualify, the highest O0→O3 query
  MRR averaged over both seeds. If none qualify, uniform stays, and that is
  the result.

**Identical pairs (same measurement).** 4.6% of draws (4.5% expected) pair
two identical listings, almost all O2-O3; 49 functions (0.4%) are one code at
every level. Under the 5% threshold set before measuring, so the sampler is
not changed for it. Such a function is still a real negative for the rest of
its batch.

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

**Measured (fingerprint `47c82734a3cb7f08`).** Rows cut at 1023 tokens after
`[CLS]`: train 6.3%, val 6.2%, test 5.3%. Token length p50/p90/p99: train
145/715/4251, max 112,709. The longest are real, not scan errors: yyjson's
`yyjson_read_opts` is 40,096 instructions at -O0 and shrinks steadily
(15,415 / 12,914 / 12,923 at O1-O3) because yyjson forces inlining; the rest
are interpreter loops and macro-expanded cipher rounds (`sqlite3VdbeExec`,
`duk__js_execute_bytecode_inner`, pcre2 `match`, `saferp_ecb_*`). Contrastive
views are cut from the start, and -O3 reorders code, so the first 1023 tokens
of the two views of a long function may show different parts of it.

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

### L — Is there enough data? (learning curve)

**Default:** all of train.

**Doubt:** 16,651 train identities from 23 packages. If val MRR is still
climbing at 100%, the corpus should grow (wave 2) - and that has to be
decided before full training, since new packages change the split, which is
locked by the first trained model.

**Two questions, two units.** `--train-fraction F --train-unit identity`
keeps that share of functions, every level of each: do more functions from
the same packages help? `--train-unit package` keeps that share of packages
whole: do more *different* packages help, which is what wave 2 adds - but
with 23 packages, which few are kept moves the result. Subsets are a prefix
of one shuffle per (unit, seed), so 25% ⊂ 50% ⊂ 100%. The vocabulary stays
the one built from all of train.

**Settles it:** val MRR (query and package means) at 25/50/100% for both
units, read against the run-to-run noise (F5). Val is the same at every
step, so the steps differ only by the training data.

**Protocol (written 16 September, before any run; after E3, with the level
pair sampling E3 chooses).**

- Contrastive from random weights, as in E3. Unlike E3, each fraction is
  trained until val stops improving (`--epochs 40 --patience 3`) and read
  at `best.pt`: a fixed budget would under-train the larger fractions and
  flatten the curve for a reason that has nothing to do with data. Epochs
  are not comparable across fractions (a 25% epoch is a quarter as long),
  which early stopping makes irrelevant.
- Runs: identity 25%, 50%; package 25%, 50%; 100% (shared by both units).
  The package unit also with seed 1 at 25% and 50%, since which packages
  are kept moves the result.
- Gain = MRR(100%) − MRR(50%), query mean; threshold T = max(0.01, M) with
  M from E3.
  - Both units' gain ≤ T: the curve has flattened; no wave 2; go to full
    training.
  - Package gain > T: more packages help; wave 2, data-format and text
    first.
  - Only identity gain > T: more functions help but new packages are not
    shown to; wave 2 still, favouring large packages in existing domains.
  - The 25→50% gains are recorded as the curve's shape, not used to decide.
  - For the package unit, the gain is the mean over its two seeds.

**Wave 2, if it is needed:** train's thinnest domains are data-format (5.1%
of identities) and text (6.1%); val and test lean on a few large packages, so
a few mid-sized packages help more than many small ones.

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

### Q2 — Ghidra heap exhaustion on quickjs -O1 — CLOSED, permanent gap

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

**Stack analyser off: listings unchanged, failure not fixed.** On quickjs -O2
(which scans normally), re-extracting with `Stack` disabled gave 1,684
functions whose listings were identical to the stored ones - 0 differences.
But -O1 with `Stack` disabled still ran out of heap, now in a second
analyzer: `X86Analyzer.flowConstants` → `SymbolicPropogator.pushMemState`,
the x86 Constant Reference Analyzer saving one memory state per branch of
`JS_CallInternal`, an interpreter loop with thousands of branches.
Analysis then "finished", with that analyzer's work abandoned part way.

**Constant analyser off as well: listings change.** On -O2 with both
disabled, 188 of 1,684 listings differed, and the differences are import
names: `call R12` stored as `IMPORT:KERNEL32.DLL!Sleep` became
`CALL:indirect`; `call [IAT slot]` stored as `IMPORT:MSVCRT.DLL!_assert`
became `CALL:indirect`. Constant propagation is part of how Ghidra resolves
imports reached through a register or an import-table slot. The analyzer
has no size limit to skip a single function (its options: threads,
speculative reference bounds, pointer analysis switches).

**Decision: permanent gap.** Scanning -O1 with both analyzers off would make
one binary's import calls look indirect where the other 383 name them, and
the encoder would learn that as something -O1 does. Import names are the
strongest signal measured (import-bearing test queries: import Jaccard MRR
0.249; the rest 0.002). The 550 training identities -O1 would add are not
worth teaching that. Recorded as a corpus gap.

**Side finding.** The two import-table calls left unresolved in tree-sitter
(Q1b) are probably places this analyzer did not reach.

**Why it matters later.** The agent will meet binaries like this. A scan that
dies on one pathological function must cost that function, not the binary -
a requirement for graceful degradation (week 11), not something to fix in
the corpus.

---

## Findings before the first full training (week 6)

Measured on 16 September, corpus fingerprint `47c82734a3cb7f08` throughout;
none of them changed the dataset.

### F1 — Embedding scale: the untrained model was confidently wrong (84b0762)

Token and position embeddings started at PyTorch's std 1.0. The MLM head
reads logits through the token embedding, so logits spread by ~18: smoke run
492 averaged a loss of 78 over its first 50 steps, against ln(321) = 5.77 for
an even guess, and gradients measured 16-28 against a clipping norm of 1.0.
At std 0.02 the start is 5.8. Setting every Linear to 0.02 as well changed
nothing in the loss and made untrained vectors of different functions more
alike (cosine 0.84 → 0.90), so the other layers keep their defaults.

Probe run 495 (1,500 MLM steps): val loss 2.48 / 2.17 / 2.02 over three
epochs, against a unigram floor of 3.206 - the model reads context, not only
token frequency. Sandbox runs on synthetic data had plateaued at their
unigram floor; that was a short, warm-up-dominated schedule, not the model.

Side finding: the padding row does not stay zero. `padding_idx` stops only
the input lookup's gradient; the tied head trains the row from the output
side. Harmless - padding is masked out of attention and pooling - and tested
with a padding row that has moved.

### F2 — Model size flags, and two silent failures (9720c45)

A 4-head checkpoint loads into an 8-head model without an error and computes
different vectors (the attention matrices have one shape for any head
count). `--init` used to ignore a requested size silently; shape fields that
differ from the checkpoint are now refused, dropout and pooling may differ
and are recorded. `max_len` lived in both Settings and the model config;
the config is now the authority.

### F3 — GPU memory (b8e188f, e9b70e1, 6a69b30)

Peak memory is recorded per epoch, training and validation apart (GiB,
allocated / reserved). 3.6M MLM, batch 64, length 1024: 3.23 / 3.52. **11M
MLM: 6.85 / 7.30** on an 8 GiB card that also drives the desktop - at the
limit; B5 needs a memory plan (batch 32, accumulation, or length 512, each
with a cost to compare).

Validation was the problem. Contrastive run 498 reserved 9.00 GiB for 2.46
GiB of tensors - past the card, into shared memory. Queries were embedded one
forward pass each (1,643 lengths); batching them cut 713 passes to 24.
Batches ran shortest first, so each outgrew every block held. Measured on
the val pool and queries, each setting in a fresh process:

| allocator | shortest first | longest first |
|---|---|---|
| default | 2.40 / 6.52 GiB, 5.0 s | 2.40 / 4.61, 4.7 s |
| expandable segments | 2.40 / 2.67, 5.4 s | 2.40 / 2.69, 4.7 s |

Vectors bit-identical between orders. The command line now sets expandable
segments unless the user chose otherwise, and batches run longest first; a
contrastive smoke epoch went from 40 s and 9.00 GiB to 26 s and 3.59 GiB.

### F4 — Where an untrained encoder starts (f302c2a)

Every run now scores its starting point on val first. Val, -O0 → -O3:

| model | MRR | recall@10 |
|---|---|---|
| random baseline | 0.007 | 0.008 |
| untrained encoder (run 502) | 0.034 | 0.053 |
| MLM probe 495, before any contrastive step (run 503) | 0.047 | 0.088 |
| best baseline, import-jaccard | 0.124 | 0.164 |
| 50 contrastive steps from random (run 501) | 0.131 | 0.250 |

50 steps, most of them in warm-up, already quadrupled the untrained score -
an expectation that they would teach almost nothing was wrong. MLM
pre-training gives a better start (+37% MRR); its effect after full
contrastive training is B4.

### F5 — Run-to-run noise

Evaluation is deterministic: the start MRR was 0.034163 in every run. Training
on the GPU is not: four identical 50-step contrastive runs (same seed) gave
val MRR 0.1309, 0.1312, 0.1313, 0.1314 - a spread of ~0.0005. Three runs had
suggested 0.0002; few samples understate noise. This is a floor: noise after
full training, and between seeds, is expected to be larger and is measured
in B3 with 2-3 seeds. An experiment's difference is read against it.

### F6 — `decl_file` of `#line` names was machine-dependent (1f2f2e4)

duktape's amalgamation names 142 files in 145 `#line` directives without a
directory; DWARF resolved them against the build directory, so all 2,443
eligible duktape rows read `/home/<user>/elenchus/duk_*.c`. 47 packages were
clean, and no identities had merged (the names are distinct, the source tree
has no two files of one name). Anchored at the unit's source directory
instead. Refreshed on the real database: duktape 2,793 paths changed and
nothing else, fingerprint unchanged. `src-noline` was not an alternative:
duktape's error macros embed `__FILE__`/`__LINE__`, so it compiles
differently.

### F7 — Loading: the dedup was repeated, not the read (1da5e5e)

A 25 s smoke epoch took minutes. Every `eligible_rows` call normalised all
56,747 listings again for the cross-package dedup (31.6 s), four times a run.
`content_key` now remembers each listing; a run reads its rows once. Loading
150 s → 44 s; a digest of every eligible row's id, identity and content key
was identical before and after.

### F8 — Baselines per package (f42a34d)

Val, -O0 → -O3, MRR:

| method | mean over queries | mean over 11 packages |
|---|---|---|
| import-jaccard | 0.124 | 0.091 |
| bm25-mnemonic | 0.099 | 0.090 |
| structural | 0.063 | 0.074 |

The bar's lead is largely one package: import-jaccard scores 0.389 on libuv
(372 queries, heavy Windows API use) and 0.002-0.152 elsewhere; on stb, with
no imports, 0.038. Each method has its own ground: bm25 on libdeflate
(0.267), structural on inih. The package mean has its own weakness - inih
has 4 queries, libcsv 9 - so neither mean is read alone. The week's success
criterion stays as written (test, mean over queries); both means are
reported.

### F9 — The dataset as training sees it

| | train | val | test |
|---|---|---|---|
| packages | 23 | 11 | 14 |
| identities | 16,651 | 3,209 | 2,787 |
| pairable (≥ 2 levels) | 66.1% | 75.9% | 76.9% |
| with -O0 and -O3 | 33.4% | 49.8% | 42.4% |
| rows without an import | 69.5% | 67.4% | 67.0% |

The three splits look alike in import share, cut share and length, so val is
a fair stand-in for test. Val identities: stb 24%, libuv 21%, libsodium 20%.
Test: mujs 25%. Train's largest: flecs 16%, sqlite 13%, mbedtls 13%.

---

## Open questions carried into full training

- **Seed noise.** Measured only for one seed (F5). B3 runs 2-3 seeds.
- **Is O0-O1 the easy pair?** A hypothesis behind E3 (see E3), not measured.
- **Which mean selects the model?** Selection stays on the query mean; every
  contrastive run prints the epoch the package mean would keep. Decide only
  if they disagree. A third mean over packages with ≥ 20 queries is an option.
- **11M memory plan** (F3), before B5.
- **Validation's reserved figure** includes what training cached before it;
  its allocated figure is validation's own.
- **Learning-curve runs use the full-train vocabulary.** Tokens only seen in
  the dropped part stay in it; noted, not expected to matter.
- **Short runs start from random weights**, not from MLM, so their absolute
  scores sit below the full pipeline's; E3 and the learning curve compare
  runs with each other, never with the week's success criterion.
- **Identity and package means can disagree between models**, not only
  between epochs: run 509 (42 steps, P = 0.5, 25% of train) scored 0.097
  over queries and 0.127 over packages; run 501 had 0.131 over queries.

## Full training (B2-B6) - protocol

Written 16 September, before any full run. Run by `experiments/run_full.sh`
(which refuses to start with uncommitted changes, with another training run
on the GPU, if `elenchus check` fails, or without the level-pair setting
given explicitly), read by `experiments/full_results.py`.

- **B2:** MLM, seed 0, `--epochs 20 --patience 3`, kept at `best.pt`.
- **B3:** contrastive from B2's `best.pt`, seeds 0, 1, 2, same limits, with
  the level-pair sampling E3 chose.
- **B4:** the same from random weights, seeds 0, 1, 2.
- **Seed noise at full scale:** S = the larger of B3's and B4's max-min val
  MRR over seeds. This replaces F5's 50-step floor as the noise later
  experiments are read against.
- **Rule (B4):** margin = max(0.005, S).
  - Mean B3 − mean B4 > margin, and B3's mean package MRR not below B4's by
    more than margin: **MLM helps, keep it.**
  - Mean B4 − mean B3 > margin: **MLM hurts:** drop it, and E1/E2 with it.
  - Otherwise, **no measured difference:** drop it too - an hour of
    pre-training per iteration is kept only for a measured gain.
- **Also reported, not decided on:** runs whose best epoch was the last
  (still improving when stopped: if most are, 20 epochs is too few, and the
  runs are extended before B5); runs where the package mean would have kept
  another epoch; runs above the val bar (MRR 0.124, recall@10 0.189).
- **B6:** `elenchus baselines --split val --encoder <best of B3/B4>`, both
  means, recorded.

### B5 - model size, memory first

The ~11M configuration (d_model 384, 6 layers, 6 heads, d_ff 1536) reserved
7.30 GiB for MLM with the default allocator (F3), on an 8 GiB card that also
drives the desktop. `experiments/b5_memory.sh` runs 50 steps each of 3.6M
contrastive (reference), 11M MLM and 11M contrastive, sampling the whole card
with nvidia-smi, since PyTorch reports only its own memory.

- **Fits** if at least 0.5 GiB of the card is free at nvidia-smi's peak, for
  both 11M runs. Then B5 repeats B2-B4 at 11M unchanged.
- **Does not fit:** batch 32 and length 512 are not options - each changes
  the experiment (half the in-batch negatives; many more functions cut), so
  3.6M against 11M would stop comparing size alone. Instead, activation
  checkpointing (recompute activations in the backward pass; ~30% slower,
  same gradients), added with a test that it changes no loss beyond
  run-to-run noise, then measured again.
- **Assumption, stated:** a 50-step peak stands for a full run. With 6% of
  rows past the 1024-token cut, most batches hold a full-length function; a
  full run's per-epoch peaks are recorded anyway, and a run that overflows
  is stopped and re-planned.

## Before full training - checklist

- [x] F1-F9 above, each committed and measured
- [x] E3 and learning-curve options (dcf29a1), checked on real data (run 509)
- [x] E3 protocol and decision rule written before the runs
- [x] Learning-curve protocol and decision rule written before the runs
- [ ] E3 runs (`experiments/run_e3.sh`, ~100 min) and verdict
      (`experiments/e3_results.py`)
- [ ] Learning-curve runs and verdict
- [x] Full training protocol, B4 rule and B5 memory plan written before runs
- [ ] B5 memory measurement (`experiments/b5_memory.sh`), when the GPU is free
- [ ] Corpus decision (wave 2 or not); if wave 2: build, `check`,
      `dataset --assign --force`, `vocab`, baselines again
- [ ] B1 vocabulary fingerprint check, then B2 onwards
