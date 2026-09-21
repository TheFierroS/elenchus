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

**Result (runs 510-517, 16 September; fingerprint `47c82734a3cb7f08`).
Uniform stays.** `experiments/e3_results.py` applied the rule above
unchanged: `verdict: default becomes none`.

Runs 510-512 recorded code `156bf13`, runs 513-517 `00ca9f4`; none is
dirty, and nothing under `elenchus/` or `tests/` differs between the two
commits (only `docs/` and `experiments/`), so all eight ran the same code.
All eight exited 0, drew 1,500 × 64 = 96,000 pairs, and passed
`elenchus check` afterwards.

Val, last checkpoint (`last.pt`), MRR:

| P | seed | run | O0-O3 drawn | O0→O3 queries | O0→O3 packages | O0→O2 | O1→O3 |
|---|---|---|---|---|---|---|---|
| unset | 0 | 510 | 10.1% | 0.3968 | 0.4688 | 0.4729 | 0.6547 |
| unset | 1 | 514 | 10.2% | 0.3894 | 0.4645 | 0.4697 | 0.6488 |
| 0.25 | 0 | 511 | 20.2% | 0.3983 | 0.4551 | 0.4686 | 0.6513 |
| 0.25 | 1 | 515 | 20.3% | 0.3946 | 0.4889 | 0.4730 | 0.6445 |
| 0.5 | 0 | 512 | 30.3% | 0.3871 | 0.4666 | 0.4687 | 0.6512 |
| 0.5 | 1 | 516 | 30.2% | 0.3860 | 0.4787 | 0.4671 | 0.6427 |
| 1.0 | 0 | 513 | 50.6% | 0.3671 | 0.4080 | 0.4367 | 0.6327 |
| 1.0 | 1 | 517 | 50.6% | 0.3445 | 0.4189 | 0.4145 | 0.6428 |

Means over the two seeds, and the difference from uniform:

| P | O0→O3 queries | O0→O3 packages | O0→O2 | O1→O3 |
|---|---|---|---|---|
| unset | 0.3931 | 0.4667 | 0.4713 | 0.6518 |
| 0.25 | 0.3965 (+0.003) | 0.4720 (+0.005) | 0.4708 (−0.001) | 0.6479 (−0.004) |
| 0.5 | 0.3866 (−0.007) | 0.4727 (+0.006) | 0.4679 (−0.003) | 0.6470 (−0.005) |
| 1.0 | 0.3558 (−0.037) | 0.4134 (−0.053) | 0.4256 (−0.046) | 0.6378 (−0.014) |

M = max(0.005, 2 × |0.3968 − 0.3894|) = **0.0148**.

- **0.25** fails condition 1 on both seeds: gains over uniform of +0.0015
  (seed 0) and +0.0052 (seed 1), neither above M.
- **0.5** fails condition 1 on both seeds: −0.0097 and −0.0034.
- **1.0** fails condition 1 on both seeds (−0.0297, −0.0449); on seed 0 it
  also fails conditions 2 and 3 on every task, on seed 1 condition 2 and
  O0→O2.

**Reading.** The more the sampler pushes O0-O3, the lower the scores, and
at P = 1 every task falls, the exam pair included. Displacing the other
pairs with the exam pair did not help; the idea behind E3 that those pairs
are wasted on the near-identical O0-O1 is not supported. The shares
actually drawn match the computed ones (10.1 / 20.2 / 30.3 / 50.5%), so the
option did what it says; the result is about the sampling, not a bug in it.

**How the verdict is used.** The code default does not change:
`Settings.eval_pair_share` stays `None`, which is the uniform sampler bit
for bit. Later runs pass no `--eval-pair-share`; `run_full.sh` and the
learning-curve script take `SHARE=none`. Changing the default instead would
make the recorded settings of earlier runs misleading.

**Consequence for the learning curve.** T = max(0.01, M) = **0.0148**.

**Observations, not decided on:**

- **M rests partly on one epoch.** Uniform seed 1 fell from 0.3942 (epoch 7)
  to 0.3894 in its last, partial epoch. Read at `best.pt`, the seed spread
  would be 0.0026 and M 0.005. The verdict does not depend on it (at
  M = 0.005 no share passes condition 1 on both seeds: 0.25's seed-0 gain is
  0.0015), but the learning-curve threshold does. The rule stays as written;
  M measures seed and last-epoch variation together.
- **Seed noise is far larger than F5's floor.** Between seeds at 1,500 steps
  the uniform runs differ by 0.0074 MRR, ~15 times the ~0.0005 of four
  same-seed 50-step runs (F5). Full training's S (B3/B4) remains the noise
  later experiments are read against.
- **The seed moves the starting point.** Start MRR was 0.0342 in every
  seed-0 run and 0.0298 in every seed-1 run: the seed sets the initial
  weights, and evaluation stays deterministic for a given seed (F5).
- **Still improving when stopped.** The best epoch was the last in 6 of 8
  runs, and training loss was still falling in all of them. Full training
  uses early stopping and is expected to go higher.
- **Absolute level.** Without MLM, from random weights, after 1,500 steps:
  val MRR ~0.39 and, from the `best.pt` measurements training recorded,
  recall@10 0.51-0.56, about three times the val bar (0.124 / 0.189). This is
  val, used for tuning; it is not evidence for the week's criterion, which
  is read once on test.
- **Task difficulty.** O1→O3 (~0.65) > O0→O2 (~0.47) > O0→O3 (~0.39), in the
  expected order: the further apart the levels, the harder.
- **Which mean selects.** The package mean would have kept a different epoch
  in 2 of 8 runs (512: 8 instead of 7; 517: 7 instead of 8), one epoch apart
  each time. Not enough to change the selection rule.
- **Run 517's epoch 7 took 2,623 s instead of ~75 s**, stretching the run to
  54 minutes. Memory was unchanged and there was no error; the likely cause
  is the laptop sleeping or throttling. It does not affect the verdict (P = 1
  fails every condition on seed 0 alone), but long runs need the power
  settings (lid: do nothing on AC; sleep on AC: never) first.
- **Memory.** Peak reserved 3.66-3.70 GiB, allocated 3.48 GiB in every run.

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

**Amendment (17 September, before any learning-curve run).** Two faults in
the budget above, found reading the schedule, not a result. This supersedes
the first two bullets and the runs; the gains, T and the branches stand.

- *Patience was not equal.* `--patience 3` counts epochs, and an epoch
  shrinks with the fraction: 3 epochs are 516 steps at 100%, 258 at 50%, 129
  at 25%. Smaller fractions would be stopped by fewer, noisier validations.
- *Runs could stop before the learning rate came down.* The cosine decay
  spans all `--epochs 40`: at 100% the rate is still 89% of its peak at epoch
  10 and 75% at epoch 15, so an early stop would keep an un-annealed model.
  In E3, whose schedule ended at 1,500 steps, every run gained 0.026-0.039
  MRR after step 860, while its rate fell from 54% to 10% of peak; how much
  of that was the decay rather than the extra steps was not measured, so it
  is not assumed away.

The protocol now:

- Every run: contrastive from random weights, **5,160 steps** (30 epochs of
  full train), the learning rate decaying over all of them, validation
  **every 172 steps** (`--eval-every 172`, one full-train epoch; identical
  training wherever validation falls, tested), no early stopping, read at
  `best.pt`. Every run gets the same steps and the same 30 validations.
  Smaller fractions see their data many more times; `best.pt` keeps the
  best before any over-fitting. `experiments/run_curve.sh`,
  `experiments/curve_results.py`.
- **Eight runs:** 100% with seeds 0 **and 1**; identity 50%, 25% (seed 0);
  package 50%, 25% (seeds 0, 1). The added 100% seed-1 run pairs each
  package seed with a full run of its own seed: E3 measured 0.0074 between
  two seeds, half of T, and a gain taken across seeds would carry it.
  Identity gain = MRR(100%, 0) − MRR(identity 50%, 0); package gain = mean
  over s of MRR(100%, s) − MRR(package 50%, s). T = 0.0148, branches as above.
- **Budget check.** If a 100% run's best MRR exceeds its best up to 75% of the
  steps (3,870) by more than T / 2, it was still climbing when the budget
  ended. A "flattened" verdict is then provisional, and the runs repeat at
  twice the budget. A gain above T stands either way: longer training of
  under-trained runs would not be expected to shrink it.
- **Time, estimated:** ~0.41 s a step (E3: 75 s for 172 steps with
  validation), so ~40 minutes a run and ~5.5 hours for eight.

**Result (runs 522-529, 17 September, code `6866821`, fingerprint
`47c82734a3cb7f08`). Wave 2: more packages help.** All eight ran the whole
budget and exited 0 (10:20-15:27, 36.5-38.8 minutes each);
`experiments/curve_results.py` applied the amended rule: `verdict: more
packages help: wave 2, data-format and text first`.

Val, `best.pt` measured again from disk:

| run | seed | functions | packages | MRR (queries) | MRR (packages) | recall@10 | best at step |
|---|---|---|---|---|---|---|---|
| 100% | 0 | 16,651 | 23 | 0.4329 | 0.5213 | 0.612 | 3,784 |
| 100% | 1 | 16,651 | 23 | 0.4343 | 0.5140 | 0.595 | 4,300 |
| identity 50% | 0 | 8,326 | 23 | 0.3915 | 0.4603 | 0.546 | 3,268 |
| package 50% | 0 | 7,360 | 12 | 0.3720 | 0.4561 | 0.547 | 2,408 |
| package 50% | 1 | 9,834 | 12 | 0.3840 | 0.4489 | 0.550 | 2,580 |
| identity 25% | 0 | 4,163 | 23 | 0.3491 | 0.4037 | 0.530 | 2,236 |
| package 25% | 0 | 1,786 | 6 | 0.2664 | 0.3738 | 0.433 | 688 |
| package 25% | 1 | 4,159 | 6 | 0.3094 | 0.3783 | 0.475 | 1,548 |

- Identity gain 50% → 100%: **+0.0413**, 2.8 T.
- Package gain 50% → 100%: **+0.0556**, 3.8 T (seed 0 +0.0608, seed 1
  +0.0503). Both above T; the rule reads the package gain first.
- Budget check: both 100% runs had levelled off by step 3,870 (their gains
  after it: 0 and +0.0010, against T / 2 = 0.0074), so the budget was
  enough and the verdict is not provisional.

**Reading.** The data is what limits the model, and variety most of all.

- *Doubling helps by a steady amount so far.* Identity unit: +0.0424 from 25%
  to 50%, +0.0413 from 50% to 100%. On this curve, a corpus 40% larger would
  gain about half a doubling (~+0.02); that is an extrapolation from three
  points of one seed and one set of packages, not a prediction to rely on.
- *Variety counts about as much as doubling.* Two runs saw almost the same
  number of functions: identity 25% (4,163 from 23 packages) scored 0.3491,
  package 25% seed 1 (4,159 from 6 packages) 0.3094, a gap of 0.040. Identity
  50% (8,326 functions, 23 packages) also beat package 50% seed 1 (9,834
  functions, 12 packages), 0.3915 against 0.3840. The seeds differ in both
  pairs; a strong sign, not a controlled result.
- The package 25% → 50% shape (+0.0901) mixes subsets of very different
  sizes (1,786 and 4,159 functions at 25%) and is not read on its own.

**Observations, not decided on:**

- **Seeds agreed far more closely than E3 suggested.** The two 100% runs
  differ by 0.0014, where E3's two uniform runs differed by 0.0074 and set
  M = 0.0148. E3 read `last.pt` after 1,500 steps, one seed dipping in its
  last partial epoch (E3, Result); fully decayed runs read at `best.pt`
  vary much less. T was the rule as written and the gains are 3-4 T either
  way; a threshold for later experiments should come from runs like these.
- **Small data is learned, then over-fitted, early.** Package 25% seed 0
  kept step 688; the full runs kept steps 3,784 and 4,300. On this data a
  full run levels off around 4,000 steps; the full-training budget is set
  from the same kind of curve on the corpus it will train on.
- **The package mean would have kept another validation in some runs**, one
  validation apart (identity 50%: 17 instead of 18), as in E3.
- **Wall time** was 38.3 minutes a run on average (5 h 6 min for eight),
  about 0.45 s a step with validation and loading, close to the estimate.

**Next (decided with the user, 17 September).** Wave 2 before any full
training: 19 candidate packages were compiled at all four levels in the
sandbox, ~9,900 identities at ≥ 2 levels (text, data-format, media, systems,
and a new math domain of three), each with no `decl_file` outside its
package; they go into the manifest with the build. After the build, a
shorter repeat of this experiment (the five deciding runs) on the new
corpus, written as its own protocol before it runs, says whether the data
has levelled off and what budget full training takes.

**Wave 2, if it is needed:** train's thinnest domains are data-format (5.1%
of identities) and text (6.1%); val and test lean on a few large packages, so
a few mid-sized packages help more than many small ones.

### L2 — Has the enlarged corpus levelled off? (short learning curve)

**Default:** all of train, which wave 2 took from 16,651 identities in 23
packages to 35,510 in 38 (F10).

**Doubt:** L found the curve still climbing at 3-4 T, the package unit
fastest, and wave 2 followed. The same question has to be asked of the
corpus that answered it, and now it costs more to get wrong in either
direction: the split is already locked (F10), so a wave 3 afterwards means
recomputing it and discarding whatever was trained, while a wave 3 that was
not needed spends days on packages the model would not have learnt from.

**Settles it:** val MRR over queries at 100% and 50% for both units, read
against T.

**Protocol (written 18 September, before any run; after the wave 2 build and
the split of F10).**

- **Five runs**, L's deciding subset: 100% with seeds 0 and 1; package 50%
  with seeds 0 and 1; identity 50% with seed 0. L's 25% runs described the
  shape of the curve and were explicitly not used to decide, so they are not
  repeated.
- **Every run:** contrastive from random weights, **11,550 steps** (30 epochs
  of full train at 385 steps an epoch), the learning rate decaying over all
  of them, validation **every 385 steps** (`--eval-every 385`, 30
  validations), no early stopping, read at `best.pt`. This is L's amended
  budget in epochs, not in steps: L's full runs peaked at 3,784 and 4,300 of
  5,160 steps, that is 22 and 25 epochs of 30, and where the peak sits on a
  corpus 2.2 times larger is exactly what is not known. Twenty epochs (7,700
  steps) would be enough if the peak stays where it was in steps, and would
  fail its own budget check if it moved with the epochs - and a failed check
  costs the whole set again at twice the budget.
- **Gains.** Identity gain = MRR(100%, seed 0) − MRR(identity 50%, seed 0).
  Package gain = mean over s of MRR(100%, s) − MRR(package 50%, s), each
  seed against the full run of its own seed.
- **Threshold T = 0.01.** L's rule was T = max(0.01, M); only M changes. L
  took M = 0.0148 from E3, which read `last.pt` at 1,500 steps with one seed
  dipping in its last partial epoch. L's own two 100% runs, fully decayed and
  read at `best.pt` - the runs below are of that kind - differ by 0.0014, and
  L recorded that "a threshold for later experiments should come from runs
  like these". M = 0.0014 falls under the floor, so T is the floor: 0.01.
- **Noise check on this experiment's own runs.** If the two 100% runs differ
  by more than T, the seed spread is as large as the threshold and every
  verdict below is provisional, whatever it says.
- **Budget check**, as in L: if a 100% run's best MRR exceeds its best up to
  75% of the steps (8,662) by more than T / 2, it was still climbing when the
  budget ended; a "levelled off" verdict is then provisional and the runs
  repeat at twice the budget. A gain above T stands either way.
- **Branches.**
  - Both gains <= T: the curve has flattened. Full training, with its budget
    set from where the 100% runs' `best.pt` lands.
  - Larger gain in (T, 2T]: still climbing, but by less than the doubling
    that produced it. Full training anyway, and the gain is recorded as what
    a wave 3 would have been worth - re-splitting after a trained model
    costs more than this is worth.
  - Larger gain > 2T: wave 3 before any full training. If the package gain is
    the larger, it favours variety - the thinnest domains after wave 2 are
    binary-analysis (3 packages), math (5) and crypto (6). If only the
    identity gain is above, it favours large packages in domains that exist.
  - The units are read as L read them: the package gain first.
- **Time, estimated:** ~0.45 s a step with validation and loading (L
  measured 38.3 minutes for 5,160 steps), so ~87 minutes a run and ~7.3
  hours for five.

**Result (runs 863-868, 18-19 September, code `7c959f1`, corpus
`ad45cb78e6f1135a`). Still climbing, and packages pay most.**
`experiments/curve2_results.py` applied the rule unchanged:
`verdict: more packages still help: wave 3, the thinnest domains first`.

| run | seed | functions | packages | val MRR | best at | last 25% |
|---|---|---|---|---|---|---|
| 100% | 0 | 35,510 | 38 | **0.5942** | 8,470 | 0.0000 |
| 100% | 1 | 35,510 | 38 | **0.5926** | 10,395 | 0.0017 |
| identity 50% | 0 | 17,755 | 38 | 0.5530 | 6,545 | 0.0000 |
| package 50% | 0 | 16,724 | 19 | 0.5343 | 6,160 | 0.0000 |
| package 50% | 1 | 16,674 | 19 | 0.5350 | 6,160 | 0.0000 |

- identity gain **+0.0412** (4.1 T), package gain **+0.0587** (5.9 T; seeds
  +0.0599 and +0.0575). The package unit pays **42% more** than the identity
  unit, the same ordering L found and by a wider margin.
- The verdict is **not provisional**. The two full runs differ by 0.0016,
  a sixth of T, which is also the first measurement of seed noise on this
  corpus and close to L's 0.0014 on the old one - so T = 0.01 is still a
  floor well above the noise. Both full runs' late gains (0.0000, 0.0017)
  are under T / 2, so the budget held; 30 epochs was the right call, and
  20 would not have been: run 863's best sits at 8,470 steps, above the
  7,700 a twenty-epoch budget would have allowed.
- Doubling the corpus did not flatten the curve. L measured an identity gain
  of +0.0413 on 16,651 train identities; L2 measures +0.0412 on 35,510. The
  gain per doubling is not shrinking, so "train until the curve flattens" is
  not a stopping rule that terminates - a wave has to be sized by what the
  ecosystem holds, not by the curve.
- One run was lost: the machine restarted overnight and cut run 866
  (package 50%, seed 1) part way. It was closed as `failed` and repeated
  from the start as run 867; nothing resumed from a half-finished state.

### F11 — Wave 3: what the ecosystem actually holds

L2 said variety, so wave 3 was chosen by package count rather than size, and
the acceptance threshold dropped from ~80 identities to 25 - decided before
the scan resumed, on the grounds that the unit the curve rewards is the
package and that below ~25 a package lands in one split and measures
nothing. 48 packages were added, ~17,800 identities: 88 packages to **136**
(1.55x). `media` and `systems` were split into image/audio and
systems/networking, which adds no packages but makes the stratified split
keep unlike code apart.

About 120 candidates were tried. What stopped the rest, in order of
frequency: **POSIX headers** MinGW does not have (`termios.h`,
`sys/select.h`, `poll.h`, `regex.h`, `dlfcn.h`, `sys/wait.h`) - 17
packages; **generated sources or absent submodules** (bison and flex output,
`qdldl`, AMD, snowball's stemmer compiler) - 9; **too small** for even the
lowered threshold - 9; **licence** - 4; **header case** on a case-sensitive
filesystem - 5, of which several were recovered.

Two of those recoveries were corrections of our own mistakes, recorded
because both were wrong for the same reason - treating a build-environment
gap as a property of the library:

- hiredis was rejected as needing `clock_gettime`, which msvcrt lacks. MinGW
  has it in winpthreads: the entry needed `pthread` in `libs`, nothing more.
- Libraries that write `#include <Windows.h>` were rejected as unbuildable.
  A one-line forwarding header with the capitalisation a case-insensitive
  filesystem would accept is the same header, not a substitute. This is not
  the same as the `regex.h` bridge that was refused for `file`: that one
  would have given the library a different regular-expression engine, which
  changes what the binary does.

The two together were 1,038 identities, including gravity at 837.

**binary-analysis stays at three.** capstone, yara and zydis are the only
disassemblers or scanners in C under a permissive licence with a release
tag; everything else is C++ (LIEF, pe-parse, Detours, keystone), GPL
(radare2, binutils, elfutils), or generates its tables at build time
(udis86, XED). Three is the least a stratified split can place in all three
parts, so the domain holds, but it will not grow without changing one of
those constraints.

**What is missing, and it is not size.** Every binary in this corpus is
built by MinGW GCC. Most Windows executables an analyst meets are built by
MSVC, which writes recognisably different code - `/GS` stack cookies,
different prologue idioms, SEH tables, its own runtime calls. A model
trained only on GCC output may transfer poorly to them, and no amount of
extra GCC-built packages tests that. MSVC also writes PDB rather than DWARF,
so ground truth would need a second reader. The cheap version of the
question comes first: build a handful of packages with MSVC and score the
existing model on them.

### F12 — Libraries carry copies of each other

The wave 3 split was read before it was locked, and one thing in it was
wrong: lz4, a test package, had lost 45% of its rows. c-blosc ships lz4's
sources in `internal-complibs/`, and when both the copy and the original
are packages, the content dedup drops those functions from **both**. Adding
a package emptied out one that was already in the corpus.

Measured on the 136-package corpus, pairs sharing the most normalised
forms:

| package | carries | splits | shared forms |
|---|---|---|---|
| plutovg | stb's image and truetype headers | test / test | 546 |
| c-blosc | lz4 | val / test | 172 |
| tinyspline | parson | val / test | 109 |
| speex | shared ancestry with speexdsp | val / train | 93 |
| lizard | zstd's FSE and Huffman coders | test / train | 82 |
| hiredis | sds | val / val | 67 |
| zstd | xxhash | train / val | 62 |

The long tail is ordinary: 1,882 forms are shared by exactly two packages
and one form by 54 of them - a one-line wrapper looks the same everywhere.
Nothing leaked: the dedup drops both sides, so no function is in two
splits. The damage is hollowing, not leakage.

Two of these are separable and were fixed: c-blosc now takes lz4 and
tinyspline takes parson as dependencies, so the copies land under `_deps/`
and are dropped by the dependency rule before the content dedup sees them,
and the originals keep their functions. c-blosc goes from 170 identities to
73 and tinyspline from 195 to 90, which is what each contributes on its
own.

The rest cannot be separated without changing what the library is: plutovg
includes stb as headers, with no translation unit to move; lizard is a fork
of lz4 with the shared parts unmodified; speex and speexdsp were one
project. Those stay, recorded here.

`elenchus check` grew a **vendored copies** warning - package pairs sharing
more than 50 normalised forms, counting only forms that appear in three
packages or fewer so a generic wrapper does not implicate everyone. Had it
existed, it would have fired while the wave 3 entries were being written.

---

### F13 — A rescan was reading the previous scan

Fixing F12 meant rebuilding two packages, and the rebuild came back with
tinyspline at 20 of 343 ground-truth functions matched and c-blosc at 152
of 419. Both had been 100%.

The addresses had not shifted and the binaries were fine. pyghidra's
default is a project beside the binary, named after it
(`tinyspline_O0_stripped.dll_ghidra/`), and those projects were still there
from the wave 3 build. Opening the same path again found a program of that
name already imported and returned **the old analysis** - so Ghidra's
functions described the previous build's layout while ground truth was
read from the new binary's DWARF. Only the three CRT stubs, which sit at
the same address in both builds, lined up. Deleting the project
directories and rescanning gave 2,335 of 2,335.

Nothing in the corpus was wrong before this: every package until now was
scanned once, on a path that had no project yet. It would have gone wrong
the first time a package was rebuilt - a version bump, a corrected entry -
and it would not have raised an error, only a lower match rate in a line
of build output.

Scans now open each binary in a temporary project of their own
(`open_fresh`), so nothing persists to be reopened. The match rate printed
per level is what caught this; it is worth keeping an eye on for exactly
this reason.

---

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

*Update after E3:* "every run" held because those runs shared seed 0. The
seed sets the initial weights, so seed-1 runs start at 0.0298; evaluation is
still deterministic for a given seed. And the floor was far too low for
comparisons across seeds: two uniform E3 runs at 1,500 steps differ by
0.0074 (E3, Result).

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

### F10 — The corpus and split after wave 2 (`a7140cc`)

Wave 2 built 40 packages in 2 h 40 min, matching 113,129 of 113,151 ground
truth functions (99%); the corpus is 88 packages. The split was recomputed
once, with `--force`, before any full training: 74 packages moved, 34 of the
48 the earlier models had learnt from, so every measurement taken before it
(E3, L runs 522-529) compares nothing that exists now.

| | before | after |
|---|---|---|
| eligible rows | 55,997 | 132,657 (69% of ground truth) |
| train | 23 packages, 16,651 identities | 38 packages, **35,510** identities |
| pairable train identities | 11,008 | **24,703** (385 steps an epoch at batch 64) |
| val / test share of rows | 7% / 6% of eligible | 15% / 15% |
| domains in all three splits | 7 | **9** |
| dedup collisions across splits | 247 forms | 748 forms |

- The dependency rule (`_deps/`) dropped 38 source files - zlib inside
  libpng and minizip-ng, ogg inside vorbis, zycore inside zydis - so those
  packages contributed their own code and nothing else.
- **Val and test no longer lean on one package.** Their largest members are
  12% and 10% of their rows, where before val was a quarter stb and test a
  quarter mujs. L read its val means against that concentration.
- Vocabulary rebuilt from the new train: 448 tokens at the same threshold
  (E9's default, a token in >= 5 train functions), fingerprint
  `ad45cb78e6f1135a`. Val coverage 0.003% `[UNK]` and 0.025%
  `IMPORT:<rare>` of tokens, 3.7% of functions touching either, so the
  threshold was left alone.
- **Baselines on the new val** (3,222 queries, 3,351 pool, -O0 -> -O3):
  bm25-mnemonic 0.090 MRR over queries (0.100 over packages), import-jaccard
  0.075, structural 0.050, random 0.003. 0.090 is the bar now; the week 6
  numbers were measured on a split that no longer exists.
- **The package mean is noisier than it was.** Val has 23 packages, and logc
  contributes 3 queries and heatshrink 9, each weighing as much as
  libjpeg-turbo's 400 - which is why logc reads 0.375 on bm25. Decisions are
  read on the query mean, as in L; the package mean is recorded beside it.

**A limitation, recorded rather than fixed.** The split now puts miniz in
train and zlib in val, and lodepng in train and libpng in val: independent
implementations of the same formats, in different splits. The dedup drops
functions whose normalised form appears in two packages (748 across splits
here), so no copy is shared, but near-copies are not caught and may lift val
a little. It is not hand-corrected, because the split's worth is that it
follows from the rule - a package's size changing can only move packages of
its own domain - and an exception written by hand would end that.

---

## Open questions carried into full training

- **Seed noise.** Measured only for one seed (F5). B3 runs 2-3 seeds.
  *Update after E3:* two seeds at 1,500 steps differ by 0.0074 MRR (E3,
  Result); S from B3/B4 is still the figure to read experiments against.
  *Update after L:* two fully decayed 100% runs read at `best.pt` differ by
  0.0014 (L, Result).
- **Is O0-O1 the easy pair?** A hypothesis behind E3 (see E3), not measured.
  *Update after E3:* not supported - drawing O0-O3 in place of the other
  pairs did not raise O0→O3 and at P = 1 lowered every task (E3, Result).
- **Which mean selects the model?** Selection stays on the query mean; every
  contrastive run prints the epoch the package mean would keep. Decide only
  if they disagree. A third mean over packages with ≥ 20 queries is an option.
- **11M memory plan** (F3), before B5. *Resolved 17 September:* 11M fits
  with `--checkpoint-activations` (B5, Result).
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

### L3 — not run, and why

L2's rule said a gain above 2 T means wave 3 before full training. Wave 3
was built. Nothing in the rule asks for an L3 after it, and one was not
run. The reason is written here rather than left as a silence:

Every branch of an L3 leads to the same action. The gain per doubling has
not shrunk across two measurements (+0.0413 in L, +0.0412 in L2), so "still
climbing" is the expected verdict and wave 3 is not even a doubling - 1.41x
in pairable train identities. And "still climbing" would call for a wave 4,
which F11 measured to be out of reach: about 120 candidates were tried and
what remains is POSIX-only, C++, GPL, or generated at build time. A wave 4
needs a different target (MSVC, ELF), which is v2's question, not v1's.

An experiment whose branches all end in the same action is a record, not a
decision, and this one costs ten to twelve hours of GPU time. The same time
answers a question that is open: whether 3.6M parameters is the limit.

**The last measurement of the data curve is therefore L2**, taken on an
88-package corpus. The corpus trained on below is 136 packages, and no
curve was measured on it.

## Full training (B2-B6) - protocol

Written 16 September, before any full run, and **rewritten 20 September**,
before any full run, for the wave 3 corpus: the budget the open note below
asked for, two seeds instead of three, and the size comparison staged so
each stage decides whether the next is worth its hours. Run by
`experiments/run_full.sh` (which refuses to start with uncommitted changes,
with another training run on the GPU, if `elenchus check` fails, or without
the level-pair setting given explicitly), read by
`experiments/full_results.py`.

**The budget, one for every run being compared:** 16,290 steps - 30 epochs
of 543 on this corpus - the learning rate decaying over all of it,
validation every 543 steps, **no early stopping**, read at `best.pt`. L
found that a run stopped on patience measures the stopping rule rather than
the thing under test, and L2's full runs peaked at 73% and 90% of a 30-epoch
budget with 0.0000 and 0.0017 gained in the last quarter - so 30 epochs
holds, and it is 41% more steps than L2 had because the corpus is larger.
Extending it is not a free improvement: `best.pt` already discards the
epochs after the peak, and a different budget is a different learning-rate
schedule, so a longer run is not a shorter run continued.

**Stage 1 - is MLM worth an hour a run? (3.6M, ~11 hours)**

- **B2:** MLM, seed 0, the budget above, kept at `best.pt`.
- **B3:** contrastive from B2's `best.pt`, seeds 0 and 1, with the
  level-pair sampling E3 chose (uniform).
- **B4:** the same from random weights, seeds 0 and 1.

Two seeds, not three: L2 measured the seed spread at 0.0016 on this
architecture, a sixth of the threshold it was read against, and the third
seed costs two hours to narrow an interval that is already narrow. The
spread the two seeds do show is what the margin is built from, so a wide
spread still widens the margin and still refuses a close call.

**Stage 2 - does size help? (~14 hours)**

The winning recipe from stage 1, at 11M (d_model 384, 6 layers, 6 heads,
d_ff 1536) and 15M (the same, 8 layers), both with
`--checkpoint-activations`, which B5 measured to fit and to change no loss.
Seed 0 only, compared against stage 1's seed 0: same corpus, same budget,
same recipe, one variable.

- Larger size ahead by more than the margin: it is the candidate for v1.
- Within the margin: **3.6M is v1.** A model three times the size that
  cannot be told apart is a worse thing to ship - slower, and needing
  activation checkpointing to fit at all.
- 15M is run only if 11M is ahead of 3.6M. If the first step up does
  nothing, the second is not worth eight hours.

**Stage 3 - only if a size wins (~6-8 hours)**

A second seed at the winning size, to show the win is larger than that
size's own seed spread. If it is not, v1 is 3.6M.

**Sizes are not mixed into the MLM decision.** Stage 1 is read at 3.6M
alone; stage 2 asks a different question and is read on its own.
- **Seed noise at full scale:** S = the larger of B3's and B4's max-min val
  MRR over seeds. This replaces F5's 50-step floor as the noise later
  experiments are read against.
- **Rule (B4):** margin = max(0.005, S).
  - Mean B3 − mean B4 > margin, and B3's mean package MRR not below B4's by
    more than margin: **MLM helps, keep it.**
  - Mean B4 − mean B3 > margin: **MLM hurts:** drop it, and E1/E2 with it.
  - Otherwise, **no measured difference:** drop it too - an hour of
    pre-training per iteration is kept only for a measured gain.
- **Budget check**, as in L: if any run's best beats its own best up to 75%
  of the budget by more than margin / 2, it was still climbing when the
  budget ended, and a "no measured difference" read off such runs is the
  budget talking - the verdict is provisional and the pair repeats at twice
  the budget. A difference wider than the margin stands either way.
- **Also reported, not decided on:** runs whose best epoch was the last;
  runs where the package mean would have kept another epoch; runs above the
  val bar, which on this corpus is import-jaccard at **MRR 0.076** and
  bm25-mnemonic at **recall@10 0.115** (4,748 queries, 4,959 pool).
- **B6:** `elenchus baselines --split val --encoder <best of B3/B4>`, both
  means, recorded.
- *Closed (20 September):* the open note asked for the budget to be
  rewritten from L's validation curves before B2. It was, above, from L2's:
  16,290 steps, no early stopping.

**Result, stage 1 (runs 1287-1291, 20 September, code `a0a562b`, corpus
`69b6d888c94c2175`). MLM helps: keep it.** `experiments/full_results.py`
applied the rule unchanged: `verdict: MLM helps: keep it`.

| group | seed | start MRR | val MRR | package MRR | recall@10 | best epoch |
|---|---|---|---|---|---|---|
| B3, from MLM | 0 | 0.043 | **0.659** | 0.651 | 0.799 | 20 / 30 |
| B3, from MLM | 1 | 0.043 | **0.658** | 0.644 | 0.796 | 20 / 30 |
| B4, from random | 0 | 0.011 | 0.533 | 0.521 | 0.681 | 27 / 30 |
| B4, from random | 1 | 0.015 | 0.520 | 0.497 | 0.676 | 21 / 30 |

- Gain **+0.1320** on the query mean and **+0.1385** on the package mean,
  against a margin of 0.0125 - **ten times the margin**, and the two means
  agree. All four runs are above the val bar; recall@10 goes from 0.681 to
  0.799, so the share of queries whose answer reaches the verifier's list of
  ten rises from about two thirds to four fifths.
- **The budget held.** No run's best beat its own best by three quarters of
  the budget by more than margin / 2, and the latest peak was epoch 27 of
  30. Thirty epochs was enough for all four; the 41% more steps than L2 had
  were not needed but cost nothing, since `best.pt` discards what comes
  after the peak.
- **MLM also makes the result repeatable, which was not the question.** The
  two runs started from MLM differ by 0.001; the two started from random
  weights differ by 0.0125, ten times as much. The margin is built from the
  wider of the two, so this is where 0.0125 came from. A second, smaller
  benefit than the gain itself, and the reason the earlier reading of "the
  seeds agree to 0.001" was too generous: it was true of B3 only.
- **Reported, not decided on:** B2's MLM loss was still falling at the last
  epoch (6.1333 to 0.3919), so a longer pre-training budget might hand
  contrastive a better starting point - open for v2. The package mean would
  have kept a later epoch in three of the four runs.

**Result, stages 2 and 3 (runs 1292-1296, 21 September, code `22239c5`,
corpus `69b6d888c94c2175`). Size helps: v1 is 11M.** Trained on a rented
RTX PRO 4500 (32 GB, Blackwell) - see F14 for how, and for the night that
was lost first.

| run | size | seed | val MRR | package MRR | recall@1 | recall@10 | best epoch |
|---|---|---|---|---|---|---|---|
| 1295 | 3.6M | 0 | 0.6525 | 0.6490 | 0.577 | 0.794 | 27 / 30 |
| 1293 | **11M** | 0 | **0.6833** | 0.6901 | 0.605 | 0.820 | 21 / 30 |
| 1296 | **11M** | 1 | **0.6833** | 0.6810 | 0.604 | 0.818 | 19 / 30 |

MLM pre-training: 3.6M reached 0.3866 (run 1294), 11M 0.3236 (run 1292),
both still falling at the last epoch. Seed 1 started from the same MLM
checkpoint as seed 0, as B3's two seeds did in stage 1.

- **Stage 2, one seed each:** 11M ahead by **+0.0308** on the query mean and
  **+0.0411** on the package mean, against a margin of 0.0125 - two and a
  half times the margin, the two means agreeing.
- **Stage 3, the rule written before the run:** v1 is 11M if the mean of two
  11M seeds beats 3.6M by more than max(0.0125, S), S being the two 11M
  seeds' spread. The seeds agree to four decimals, S = 0.0000; the gain is
  **+0.0308**. **v1 is 11M.**
- **Budget:** no run's best beat its best by 75% of the budget by more than
  margin / 2 (0.0004, 0.0000, 0.0000). Thirty epochs held for both sizes.
- **The reference was measured again on the same card, and it mattered.**
  The protocol compared against stage 1's 3.6M seed 0, trained on the
  laptop. Run on the rented card, the same model, seed and data scored
  **0.6525 against the laptop's 0.659** (MLM 0.3866 against 0.3919). A 0.0065
  difference from hardware alone is half the margin; read against the
  laptop's number the size gain would have been 0.024 rather than 0.031.
- **The package mean is the noisier of the two.** The seeds agree exactly on
  the query mean and differ by 0.009 on the package mean, which weights a
  six-query package the same as a five-hundred-query one. This is why the
  decision is read off queries.
- **11M overfits after its peak.** Train loss kept falling (0.37 at the
  best epoch, 0.28 at the last) while val MRR held between 0.677 and 0.683.
  With a third of the parameters, 3.6M peaked at 0.6525 with train loss
  still at 0.44. The larger model uses the data up.
- **Which checkpoint is v1:** the two seeds tie on the decision metric;
  seed 0 (run 1293) is taken, being higher on the package mean and on
  recall@10. The file is
  `data/cloud-2026-09-21/elenchus/models/b3-con-mlm-11m-seed-0/best.pt`.

**15M was not run.** The protocol ran 15M only if 11M was ahead, and it
was. It was skipped on a reading, not a measurement: tripling the model
paid 0.031 - a quarter of what MLM paid (0.132) - and 11M was overfitting
past epoch 21, which points at the data as the limit rather than the size.
A further 1.4x was not expected to clear the margin, and the size question
is better asked again on a larger corpus (Q3). Recorded as a decision so
that it is not read as an omission.

**What the two levers were worth:** MLM pre-training +0.132, three times the
parameters +0.031. The larger lever was the recipe, and the next one is
probably the data.

### F14 — Training elsewhere, and the night that was lost first

The laptop ran at 82-87 C for eleven hours a night, and stage 2 would have
needed two nights more. Training moved to a rented GPU. Two things about
doing that are worth keeping.

**The database did not have to move.** Of its 14 GB, 11.7 GB is events and
their links, which training never opens. `elenchus export-training` writes
the seven tables it does read - 1.02 GiB - carrying row ids across
unchanged, and the export reproduced the corpus exactly: the same split,
the same 187,403 eligible rows, and the vocabulary fingerprint
`69b6d888c94c2175` on both machines. `elenchus import-runs` brings the runs
back renumbered. Its first real use found that it renumbered a
measurement's run_id but not the run number in its name, leaving
encoder-2 against run 1293; that was fixed (`197b87f`), the four affected
rows renamed in place, and `elenchus check` now fails on the mismatch.

**The first night's results were lost, and the cause was a platform detail
stated as fact without being checked.** The script trained both sizes and
then stopped the pod to stop the billing. Everything had been written to
the pod's container disk, which RunPod erases on stop; only a volume
survives. The deploy screen said so ("Nothing mounted at the template's
path") and the warning was dismissed. The GPU was then taken by another
user, so the pod could not even be restarted to look. Cost: one night and
about $4.

The second attempt did what the first should have: RunPod's storage
documentation read before anything was built on it; a network volume,
which outlives any pod and can be opened from a CPU-only one; `df` to see
that `/workspace` really was that volume; and a file written, the pod
terminated, a new pod started, and the file read back - five minutes and a
few cents, before the long run began. Results were then brought home and
compared by sha256 before anything was deleted.

Cost of stages 2 and 3 together, including the volume and the CPU pods
used to fetch results: about $4.50.

### Q3 — the third axis was never measured

L and L2 measured two ways of growing the corpus: more **functions**
(identity) and more **packages**. There is a third, and it was not in
either design: **more views of the same function.** Every identity has at
most four here (`-O0` to `-O3`). Adding `-Os`, `-Og` or LTO would make six
or seven, and training pairs grow with the square of the views - four give
six pairs, six give fifteen.

Its cost is not a candidate hunt but a rebuild: the manifest already holds
136 packages. Wave 3 spent three days finding 48 new ones because L2 said
packages pay more than identities; it never asked what views pay, because
the experiment had two arms and both were mine to choose. The design that
would have answered it is L2 with a third arm: identity 50%, package 50%,
**levels 50%** - half of each identity's views dropped.

A cheap first reading is available on this corpus: restrict training to
`-O0` and `-O3` and run once (~2 hours). If the slope is steep, wave 4 is a
night of recompiling rather than a week of searching - and MSVC is the same
axis, at its expensive end.

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

**Result (17 September). 11M does not fit as it is; with activation
checkpointing both 11M runs fit.** Two measurements, the same code path:

- *Morning (runs 518-520, code `c0432d7`):* ref ok; mlm11
  left **0.29 GiB** free (NO); con11 failed a few steps into training with
  `CUDA driver error: device not ready / Failed to create GPU mapping`, the
  card at 7.67 of 8.00 GiB. On WSL an overflow ends the run rather than
  spilling slowly into system memory.
- *Afternoon (code `12450b9`, `--checkpoint-activations` added and tested):*

| run | torch allocated | torch reserved | card idle | card peak | free at peak | fits | epoch s |
|---|---|---|---|---|---|---|---|
| ref (3.6M) | 3.48 | 3.59 | 0.94 | 4.66 | 3.34 | yes | 26 |
| mlm11 | 6.84 | 7.13 | 0.94 | 7.63 | 0.37 | NO | 85 |
| con11 | - | - | - | - | - | failed again (same error) | - |
| mlm11ac | 2.13 | 2.33 | 0.39 | 2.85 | **5.14** | yes | 66 |
| con11ac | 3.79 | 4.23 | 0.39 | 4.83 | **3.16** | yes | 67 |

GiB throughout; "epoch s" is the 50 steps with their validation.

- **Decision, by the rule:** B5 runs 11M with `--checkpoint-activations`.
  The flag changes no loss or gradient (bit-identical on CPU, tests in
  `12450b9`), so 3.6M against 11M still compares size alone.
- **Memory:** MLM's allocated peak fell from 6.84 to 2.13 GiB (-69%).
- **Speed is not measured cleanly here.** mlm11ac (66 s) was faster than
  mlm11 (85 s), most likely because mlm11 ran against the card's limit;
  on this card, checkpointed 11M costs no more time than 11M without it.
- **Card idle** read 0.39 GiB for the checkpointed runs and 0.94 for the
  others, likely freed after con11's failure; with 0.94 they would still
  leave ~4.6 and ~2.6 GiB.

**~15M, measured again (code `f6eb793`, 16:27-16:41).** Eight layers of the
11M width (14,910,401 parameters), checkpointed; the whole script ran again:

| run | torch allocated | torch reserved | card idle | card peak | free at peak | fits | epoch s |
|---|---|---|---|---|---|---|---|
| ref (3.6M) | 3.48 | 3.59 | 0.52 | 4.24 | 3.76 | yes | 25 |
| mlm11 | 6.84 | 7.13 | 0.51 | 7.63 | 0.36 | NO | 83 |
| con11 | - | - | - | - | - | failed (third time, same script) | - |
| mlm11ac | 2.13 | 2.33 | 0.38 | 3.01 | 4.99 | yes | 67 |
| con11ac | 3.79 | 4.23 | 0.47 | 4.87 | 3.13 | yes | 67 |
| mlm15ac | 2.35 | 2.53 | 0.42 | 3.20 | **4.79** | yes | 88 |
| con15ac | 3.86 | 4.30 | 0.49 | 4.99 | **3.00** | yes | 88 |

- **15M fits with checkpointing**, 3.00 GiB free for contrastive. Measured,
  not decided on: whether B5 includes 15M is settled on the new corpus.
- **Two more layers cost almost no memory** when only layer inputs are kept:
  contrastive allocated 3.79 → 3.86 GiB (+0.07).
- **They cost time in proportion:** 67 → 88 s (+31%) for 6 → 8 layers
  (+33% layer compute).
- **The measurement repeats:** the 11M rows match the earlier run to within
  0.16 GiB of card peak, idle this time steady at 0.38-0.52 GiB.

## Before full training - checklist

- [x] F1-F9 above, each committed and measured
- [x] E3 and learning-curve options (dcf29a1), checked on real data (run 509)
- [x] E3 protocol and decision rule written before the runs
- [x] Learning-curve protocol and decision rule written before the runs
- [x] E3 runs (`experiments/run_e3.sh`, ~100 min) and verdict
      (`experiments/e3_results.py`): uniform stays, M = 0.0148
- [x] Learning-curve protocol amended before the runs (fixed step budget,
      validation every 172 steps, 100% with two seeds)
- [x] Learning-curve runs (`experiments/run_curve.sh`, 5 h 6 min) and verdict
      (`experiments/curve_results.py`): more packages help, wave 2
- [x] Full training protocol, B4 rule and B5 memory plan written before runs
- [x] B5 memory measurement (`experiments/b5_memory.sh`): 11M fits with
      `--checkpoint-activations`
- [x] Corpus decision: wave 2 (L, Result)
- [ ] Wave 2: manifest, build, `check`, split preview, `dataset --assign
      --force`, `vocab`, baselines again
- [ ] Short learning-curve repeat on the new corpus (protocol first); sets
      the full-training budget
- [ ] B1 vocabulary fingerprint check, then B2 onwards
