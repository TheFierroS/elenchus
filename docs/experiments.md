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

### Q1 — Lower match rates on some packages

The 45-package build matched 41,859 of 42,365 ground-truth functions (98.8%).
The 28-package corpus had averaged 99.96%. Most new packages matched at 100%
on every level (sqlite, stb, lodepng, brotli, bzip2, pcre2, wren, md4c); the
loss is concentrated:

| package | -O0 | -O1 | -O2 | -O3 |
|---|---|---|---|---|
| mujs | 99% | 96% | 81% | 72% |
| flecs | 99% | 99% | 96% | 93% |
| duktape | 100% | 95% | 98% | 98% |
| libuv | 99% | 97% | 98% | 98% |
| quickjs | 100% | (scan failed) | 99% | 99% |

mujs and flecs alone account for over 300 of the 506 unmatched functions. The
drop grows with optimisation in both.

**Why it matters beyond lost rows.** If a particular kind of function goes
unmatched, the dataset does not just shrink - it is missing that kind
systematically, and the encoder never sees it. And whatever stops Ghidra
finding these functions here will stop it on a real stripped binary too, so
it bounds how far the agent can trust its function list.

**Hypothesis, not verified.** The packages that lose most register functions
by address rather than calling them: mujs its built-ins in tables, flecs its
systems, hooks and observers as callbacks. Nothing in the code calls such a
function; its address sits in data. Win64 leaf functions that use no stack
need no `.pdata` unwind entry, and more functions become such leaves at -O2
and -O3. With no call site and no unwind entry, stripped-binary function
discovery has nothing to start from.

**Against it:** quickjs is also an interpreter with built-in tables and
matches at 99%. So "interpreter" is not the cause, and "registered by address"
may not be either - it has to be checked, not argued.

**Settles it:** for the unmatched addresses of mujs -O3 and flecs -O3, check
(a) whether `.pdata` holds an entry for them, (b) whether any instruction
calls them, (c) whether a data section holds their address. Do the same for a
sample of quickjs -O3 functions that did match, as the contrast. If the
unmatched are mostly (no, no, yes) and the matched are not, the hypothesis
stands, and the fix belongs in extraction (seeding functions from address
tables), not in the dataset.

### Q2 — Ghidra heap exhaustion

quickjs `-O1` failed to scan with `OutOfMemoryError: Java heap space` under
`-Xmx4G`; `-O0`, `-O2` and `-O3` of the same package did not. Whether the
cause is heap size alone or parallel analysis across 20 threads holding many
functions at once is not known.
