# The verifier - design

Written 21 September, before any verifier code exists. `elenchus/emulation/`
and `elenchus/agent/` are empty packages; nothing below describes code that
runs yet. What is decided here is decided before it is built, so that the
building can be measured against it.

## What the verifier is for

The encoder proposes. For a function nobody has named, it ranks the known
functions it most resembles; on 39 libraries it never saw, the right one is
first two times in three and in the first ten 86% of the time (run 1297).
That is a good proposer and a bad witness: a ranking is not evidence, and
the third of queries where the first answer is wrong look, from the outside,
exactly like the two thirds where it is right.

The verifier's job is to test the proposals - to try to show that a
proposed identity is false - and to say, for every proposal, what it found
and how anyone can check it. The name of the project is the method: an
*elenchus* is a refutation, a claim put to a test that it can fail.

## Claims

A **claim** in v1 is an identity:

> **Q is K.** The function at address `a` in binary `B` was compiled from
> the same source function as the reference function `K`.

`K` is a function of a reference build - a library compiled by us, with
debug information, so its name, source file and **signature** are known.
`Q` is not: in use it is a function of a stripped binary; in evaluation it
is a function of a corpus binary whose debug information is withheld from
the verifier and used only to score it afterwards.

Claims come from the encoder's top ten for `Q`, with their rank and score.
Nothing else creates claims in v1.

## What the verifier can and cannot conclude

The verifier tests **behaviour**. If `Q` is `K`, then compiled at any
optimisation level from the same source, they compute the same function:
the same outputs and the same effects for the same inputs. So:

- **A behavioural difference refutes identity.** If one input makes them
  disagree - and the disagreement is not an artefact of how they were run
  (below) - `Q` is not `K`.
- **Agreement does not confirm identity.** Two different source functions
  can behave identically - two libraries' `min`, two wrappers around the
  same call. Behavioural equivalence is necessary for identity, not
  sufficient. And even equivalence is only sampled: N inputs agreeing says
  nothing about the N+1th.

Every claim therefore ends in exactly one of three **verdicts**:

| verdict | meaning | what it is not |
|---|---|---|
| **refuted** | a counterexample was found: an input on which `Q` and `K` disagree, reproducibly | - |
| **survived** | the tests that could be run were run, and none refuted it | not a proof; not "verified" |
| **inconclusive** | the claim could not be put to a test that would be trusted either way | not a failure of `Q`; not a vote against |

"Survived" is deliberately not "verified". The project reports what was
attempted and what happened, and a claim that has survived thirty-two
executions is described as exactly that.

## The requirement that everything else serves

> **The verifier must never refute a true claim.**

A proposer that is wrong a third of the time is useful; a witness that
lies once is not, because after that no refutation can be trusted. So the
design is asymmetric on purpose: whenever the verifier is not certain a
disagreement is real, the verdict is **inconclusive**, never refuted.
Coverage can be bought back later. Trust cannot.

This is measurable, and it is the first thing measured: the corpus holds
tens of thousands of **true pairs** - the same source function at -O0 and
at -O3, known from debug information. Every one of them is a claim that
must not be refuted. The false-refutation count on true pairs must be
**zero**; with none in N, the 95% upper bound on the rate is 3/N (the rule
of three), so N is reported alongside it.

## How a true claim could be wrongly refuted

Each of these is a way that two compilations of the *same* function can
look different when run. Each needs an answer before the verifier is
trusted, and each answer is tested on true pairs.

1. **The return register holds garbage for a narrow or absent return.** A
   function returning `int` defines the low 32 bits of RAX; the upper 32
   are whatever was last computed there, and -O0 and -O3 compute different
   things. `char` and `bool` define 8 bits. `void` defines nothing. The
   return value is compared **masked to the width the reference signature
   gives**, and not at all for `void`.
2. **Reads of uninitialised memory.** C allows reading uninitialised
   locals only as undefined behaviour, but real code does it, and -O0 and
   -O3 read different garbage. Answer: every execution is run **twice per
   version, with different fill patterns for uninitialised memory** - the
   stack below the entry stack pointer, and memory the allocator stubs hand
   out. An observable that changes with that fill pattern *within one
   version* depends on garbage and is excluded from the comparison for that
   input.

   Memory reached through arguments is **not** uninitialised: it is the
   input. It is filled with a pattern of its address alone, the same in
   both runs and both versions. *(Corrected 21 September by the first smoke
   test, before any verifier code: the first draft filled all unset memory
   from the seed, and `sum(const int *p, int n)` then returned a different
   value for each seed - correctly, since it reads its input. The rule as
   drafted would have called every function that reads its argument
   garbage-dependent and excluded it, so the verifier could have tested
   almost nothing.)*
3. **Other undefined behaviour in the inputs.** Signed overflow, shifts by
   the width, out-of-bounds indexing: the compiler may assume they never
   happen, and -O3 exploits the assumption. Generated inputs favour values
   that do not trigger it (see Inputs), and the true-pair calibration is
   what shows whether that is enough.
4. **Inlining changes the calls.** -O3 inlines helpers that -O0 calls, and
   turns loops into `memcpy` or `memset` calls that -O0 never made. So the
   **sequence of calls is never compared** - only final effects. Calls into
   the same binary are emulated through; calls to imports are served by
   stubs that perform their effect (below), so an inlined copy and a call
   leave the same state.
5. **Allocation can be elided.** -O3 may remove a `malloc`/`free` pair, so
   heap addresses differ between versions. Pointers are never compared by
   value; memory reached through them is compared by content.
6. **Floating point.** Neither build uses `-ffast-math`, so results should
   be bit-identical; this is an assumption to be tested on true pairs, not
   taken on trust. If it fails, float returns are compared within a
   tolerance recorded with the verdict.
7. **Speed.** -O3 is faster, sometimes by orders of magnitude. The
   instruction budget is generous and generous for both; a version that
   exhausts it makes the input **inconclusive**, not a refutation.
8. **Stack contents and locals.** Everything below the entry stack pointer
   is the function's own and is never compared.
9. **Global state.** A function may read and write globals. Reads see each
   binary's own initialised data, which is the same source data; writes are
   **not compared in v1**, because in a stripped binary there is no way to
   say which global in `Q` is which global in `K`. A known blind spot,
   recorded as one: a function whose only effect is on globals can survive
   a claim it should not.

## How a claim is tested

### Tier 1 - static evidence

Cheap facts compared without running anything: which imports are
reachable, string and constant references, size. Most wrong candidates
differ from the query in ways a glance shows.

But every item in the list above that makes behaviour differ also makes
static features differ: -O3 adds `memcpy` calls and drops dead strings.
So static evidence **refutes only through rules calibrated to zero false
refutations on true pairs** (train split), and confirmed at zero on val.
A static signal that has not passed that bar may reorder candidates; it
may not refute.

### Tier 2 - emulation

`Q` and `K` are each run on the same inputs, in isolation, and their
effects compared.

- **The experiment is defined by `K`'s signature.** The claim says `Q` is
  `K`; if it is, `Q` has `K`'s parameters, return type and calling
  convention. So `K`'s signature - from its debug information - decides how
  many arguments are passed, which are pointers, which are floating point
  and how wide the return value is, for both. This is why the verifier
  does not need `Q`'s types.
- **Calling convention:** Win64, as MinGW uses - the first four arguments
  in RCX, RDX, R8, R9 (or XMM0-3 for floating point), the rest on the
  stack above 32 bytes of shadow space.
- **Memory:** the binary's own sections mapped at its image base.
  Everything the function reaches that the verifier did not map is mapped
  **on first touch**, filled with a pattern that is a function of the
  address alone, so the same address holds the same bytes for both
  versions and a pointer argument's buffer exists without its size being
  known. Only the stack and allocator memory take the seed-dependent fill
  (item 2 above).
- **Calls are handled in three tiers, widest first** (see "Coverage" below).
  A call **within the binary** is followed into the real code and emulated -
  no stub, and the coverage is free: a statically-linked zlib's `deflate`
  runs as itself. A call to an **import the harness has a stub for** is
  served by the stub. A call to **any other import** makes that input
  **inconclusive** - never refuted.
- **Observed:** the masked return value, and the final contents of every
  page the function wrote outside its own stack - argument-reachable
  memory and the heap, compared by content.
- **Engine:** Unicorn, a CPU emulator built on QEMU, is the candidate: it
  runs x86-64 code without an operating system and lets memory and calls be
  intercepted, which is what the above needs. That it handles the
  instructions these binaries contain is **to be measured**, not assumed
  (V0 below).

### Inputs

For each claim, a fixed number of input vectors, generated from a seed
recorded with the verdict, per parameter from `K`'s signature: integers
from boundary values (0, 1, -1, small positives, the type's limits) and
seeded random values, kept away from values that overflow common
arithmetic; pointers to distinct regions far apart; floating point from
ordinary values and exact small integers. Testing stops at the first input
that refutes.

## Coverage - three tiers, widest first

The verifier's reach is not the number of stubs. Most of what a library
function calls is other code in the same binary, and most of the rest falls
into a few families that share one tested core. The design widens coverage
by these, in order, and only the last tier narrows it.

**Tier 1 - calls within the binary, followed for free.** Libraries are
usually linked statically, so when a function calls another function that is
*in this binary*, the emulator does not stub it: it follows the call into
the real code and runs it, exactly as it ran the function under test. zlib's
`deflate` calling `deflate_stored` calling `longest_match` needs no stubs at
all, because all three are present and all three are emulated. This is where
most calls go, and it costs nothing but letting the emulator continue. The
budget covers the whole call tree, so a genuinely huge computation is
budget-exhausted (inconclusive), not refuted.

**Tier 2 - imports, served by stub families.** A call that leaves the binary
- to the C runtime - is an import, and these cluster into a few behaviour
families, each built on one tested core rather than one stub per name:

| family | members | shared core |
|---|---|---|
| block memory | memcpy, memmove, memset, memcmp, and the `__builtin_` and bcopy/bzero spellings | copy / fill / compare a byte range |
| C string | strlen, strcpy, strncpy, strcat, strcmp, strncmp, strchr, strrchr, strstr, strdup | a NUL-terminated byte range |
| character | toupper, tolower, isdigit, isalpha, isspace, ... | a one-byte table |
| number/text | atoi, atol, strtol, strtoul | parse an integer from text |
| allocation | malloc, calloc, realloc, free, aligned variants | a deterministic arena |

A family's core is written and tested once against the C standard; each
member is a few lines on top, and each member still gets its own test. This
is how coverage is wide without being a pile of hand-written stubs: forty
string functions are one tested string core and forty short, tested wrappers.
A member the core cannot honour exactly (a `realloc` that would have to move
a block the arena cannot) makes the input inconclusive, never refuted.

**Tier 3 - an import in no family: inconclusive.** A call to something
genuinely outside these - `deflate` as an *import* because zlib was linked
dynamically, `getaddrinfo`, a callback into code the harness does not have -
makes the input inconclusive. It is not stubbed, because a hand-written
`deflate` would be unverified code in the heart of the verifier, and a wrong
stub refutes true claims. Silence is safer than a guess.

**Widening is measured, not guessed.** V0's import histogram counts, across
the sample, which imports reach tier 3 and how often. A family or a member
is added when the histogram shows it would move real coverage, and it enters
by the stub rules - the C standard, its own test, deterministic - never to
make one pair pass. So coverage grows toward where the functions actually
are, and a bigger binary that leans on an unhandled family shows up as a
spike in that histogram, to be answered by adding the family, not by
loosening anything.

## Every verdict can be replayed

A refutation is a counterexample, and a counterexample is only worth
something if it can be run again. Every verdict records: the claim, the
verifier's code version, the seed, the inputs, what each version produced
for each, and the reason for any exclusion. `elenchus verify --replay`
reruns a recorded verdict and must reproduce it exactly.

Claims and verdicts are events in the existing event log - which already
has a binary, a run, a sequence and links from events to functions with
roles - so they are appended and never edited, and each is tied to the run
and code that made it.

- `claim.identity`: payload {rank, score, encoder run}; links: the query
  function as `subject`, the reference function as `candidate`.
- `verdict`: payload {claim event, verdict, tier, rule or harness version,
  seed, inputs, observations, exclusions}; the same links.

## What must exist before it

- **Resolved signatures.** `ground_truth` stores parameter and return types
  as names (`uint32_t`, `size_t`), because `dwarf.py` stops at typedefs.
  The verifier needs each resolved to a kind (integer, pointer, float,
  struct, void), a width and, for integers, signedness. Structs passed by
  value have their own Win64 rules (by reference above 8 bytes) and are
  **inconclusive in v1** until handled.
- **The reference functions' binaries**, at every level. They are the
  corpus binaries already built; nothing new is compiled.

## How it is measured

On pairs where the truth is known, from the corpus:

- **True pairs** - the same identity at -O0 and -O3. The requirement: zero
  refuted. Also reported: how many survived and how many were
  inconclusive, which is the verifier's coverage.
- **False pairs** - a query and each wrong candidate in its encoder top
  ten. Reported: the share refuted (the verifier's power), survived,
  inconclusive.
- **End to end** - for each query, the encoder's top ten, then the
  verdicts. The answer is the highest-ranked candidate that survived; if
  none survived, the highest-ranked one not refuted, marked unverified; if
  all ten were refuted, *none of these*. Reported against the encoder alone
  (0.670 top-1 on test): accuracy, accuracy among answers backed by a
  survival, and how often the system abstains.

The splits keep their roles: rules and harness are calibrated on
**train**, measured on **val**, and the end-to-end number is taken once on
**test**, with its protocol written first, as the encoder's was.

## The first smoke test (21 September)

Before V0's protocol, one question: does the candidate engine run a
MinGW DLL's code at all? A three-function C file was built with the corpus
toolchain (GCC 13, MinGW-w64) at -O0 and -O3 and each function run under
Unicorn 2.1.4 with the Win64 convention, memory mapped on first touch.

- `add3(7, 5, 2)`: 24 from both versions, the right answer.
- `fill(buf, 20, 'A')`, a `void` function: the buffer written identically
  by both, and RAX left at **20 by -O0 and 16 by -O3**. Item 1 above, seen
  in the first function tried: comparing RAX here would have refuted a
  pair known to be the same function.
- `sum(buf, 9)`: the same value from both versions, but a different value
  for each fill seed, which is what corrected item 2.

A smoke test on three hand-written functions says the engine runs this
code; it says nothing about how much of a real library will run. That is
V0.

## V0 - feasibility protocol

Written 22 September, before any harness code. V0 answers one question with
a number the project does not have: **how much of a real library can the
verifier run at all?** Everything after it - the instruction budget, which
stubs to write first, whether the whole approach is worth building - is set
by what V0 measures, so its protocol is fixed here before it is run.

**The sample.** 3,000 true pairs drawn from the **train** split, by a seed
recorded with the result. A true pair is one identity at -O0 and -O3, known
the same from debug information. 3,000 is deliberately more than a quick
look needs: it is enough that each of the eleven domains lands a few hundred,
so the result is a map of *where* the verifier runs - crypto, strings,
parsing - not a single average that hides it. Train, not val or test, because
V0 shapes the harness and the harness must not be tuned on the split it will
be judged on.

**Two layers, run on the same sample.** The gap between them is the
measurement that matters.

- **Layer A - bare.** No stubs at all. A call to any import stops that pair.
  The most honest floor: how many functions run start to finish touching
  nothing outside this binary.
- **Layer B - with the base stubs.** The handful below implemented. How far
  that floor rises when the commonest imports are served.

If A says 40% and B says 75%, that 35 points is the value of eight stubs,
and it tells us the approach is worth building and where the next effort
goes. If B is barely above A, the imports are long-tailed and the plan
changes. Either way it is a measured number, not a guess.

**The instruction budget starts at 5,000,000** and V0 measures whether that
is right. A function that has not returned in five million instructions is
either genuinely non-terminating on this input or stuck in garbage we set
up wrong; either way its pair is counted **budget-exhausted**, and V0
reports, for the pairs that did finish, the distribution of how many
instructions they took. If every real function finishes under 200,000, the
budget comes down to fit; the starting value is a ceiling to measure under,
not a decision.

**Every pair falls in exactly one bucket:**

| bucket | meaning |
|---|---|
| completed | both versions returned within budget |
| unsupported instruction | the engine refused an instruction one version contains |
| import without a stub | a version called an import not served in this layer |
| budget exhausted | a version ran past the instruction budget |
| signature declined | the reference signature has a form the harness declines (a by-value struct, a vararg, long double) |
| load failed | the binary or the function could not be set up at all |

**And, among the completed pairs, the first false-refutation count.** Every
completed true pair *should* agree - it is the same function. Any that
disagree are the harness's own errors surfacing, and each is read and its
cause written down (the way F1-F14 were), because these are exactly the
false refutations the whole design exists to drive to zero. V0 does not
have to reach zero; it has to find and name every cause, so the calibration
that follows knows what it is fixing.

**What V0 decides:** the real instruction budget; the first stub list, in
the order their absence costs the most coverage; whether by-value structs
and the rest are rare enough to leave declined for v1; and the number the
project has never had - the verifier's reachable coverage. Recorded as a
measurement run, with the sample seed and the code version, like every
other.

## Stubs - how each one is trusted

A stub stands in for an imported function the emulator has no code for: the
emulator stops at the call, the stub performs the function's effect on
emulated memory and registers, and the caller continues as if the real one
had run. Stubs are what make a string or buffer function testable at all,
and a **wrong stub is the worst thing in the system** - it changes what a
version computes, so it can make one true function disagree with itself and
refute a claim that is true. The requirement that the verifier never refute
a true claim rests on every stub being exactly right.

So each stub is held to three rules, and none is admitted otherwise:

1. **It implements the C standard behaviour of the function it replaces,
   nothing more.** `memcpy` copies n bytes; it does not decide the copy
   looks wrong. Its only job is to leave memory and registers as the real
   function would.
2. **It is tested on its own, against that behaviour, before it is used** -
   including the corners that break naive versions: `memcpy` of zero bytes,
   `memmove` of overlapping regions (where `memcpy` may not be used),
   `strlen` on an empty string, `malloc(0)`, alignment of returned pointers.
   The test is the C semantics, not what happens to pass.
3. **It is deterministic and self-contained.** The allocator hands out
   addresses from a fixed arena in a fixed order, so a run replays exactly;
   `malloc` never fails except where the design says it may, and the same
   inputs always give the same addresses.

Anything a stub cannot honour makes the input **inconclusive**, never
refuted. If `realloc` would have to move a block and the arena cannot, that
input is inconclusive. Silence is always safer than a wrong answer.

**The base stubs (layer B, and the first written) - the block-memory and
allocation cores, plus the first string members - each with what its test
must cover. The families above say how the rest follow from these:**

| stub | behaviour | the test's corners |
|---|---|---|
| `memcpy` | copy n bytes, no overlap promised | n = 0; exact byte range; caller gets dest back in RAX |
| `memmove` | copy n bytes, overlap safe | forward and backward overlap; n = 0 |
| `memset` | write a byte n times | n = 0; the byte is truncated to 8 bits |
| `memcmp` | compare n bytes | equal; first-byte and last-byte difference; sign of result |
| `strlen` | bytes before the first NUL | empty string; the NUL is not counted |
| `strcmp` | compare until NUL or difference | equal; prefix; sign of result |
| `malloc` | n bytes from the arena, aligned | n = 0; 16-byte alignment; two calls do not overlap |
| `calloc` | like malloc, zeroed | the memory is zero; count × size overflow is refused (returns null) |
| `realloc` | resize, preserving contents | grow copies the old bytes; realloc(null, n) is malloc; shrink in place |
| `free` | return a block to the arena | free(null) does nothing; a freed block is not handed out corrupt |

The list is the starting set, not the final one: V0's import histogram says
which imports beyond these are worth a stub, and each new one enters by the
same three rules and its own test. A stub is never widened to make a
particular pair pass; it is widened to match the standard, and the pairs
follow.

## Order of work

1. [done] Resolved signatures in `dwarf.py` and migration 010.
2. **V0 layer A** - the harness far enough to run a function with no stubs:
   loader, Win64 call, memory on first touch, fill patterns, budget, masked
   return. Measured on the 3,000-pair sample.
3. **The base stubs**, each by its three rules and its own test; then
   **V0 layer B** on the same sample.
4. Read V0: set the real budget, the stub order, what stays declined, and
   the coverage number. Written down like a finding.
5. Calibration on train true pairs until zero refutations - every cause
   found written down, the way F1-F14 were.
6. Tier 1 static rules, each admitted only at zero false refutations.
7. Claims and verdicts as events; `elenchus verify` and `--replay`.
8. Measurement on val; then the end-to-end protocol; then test, once.

## V0's first run, and the first false refutation (F15)

The first V0 pass on the real corpus did what V0 is for: it surfaced a
harness flaw the fixtures never could. Of a 50-pair sample, the completed
pairs agreed except one - `lua_upvaluejoin` - which was refuted on "written
memory differs". Since the pair is the same function at two optimisation
levels, that was a false refutation, the one thing the design forbids.

The cause was risk 9, exactly as written but not yet coded around. The
function writes into the binary's own `.rdata` - a static string, the Lua
version banner - and -O0 and -O3 place that data at different addresses
(0x25264b000 against 0x202be8000), with different surrounding bytes. The
harness recorded those writes and compared them, so two placements of the
same global read as a difference and refuted a true claim.

The fix codes the rule the design already stated: the harness no longer
records writes into the binary's own image, only writes to memory it handed
out - argument buffers and the arena, which sit at the same address in both
versions and are the function's matchable effect. A miniature of the bug is
now a fixture (`record`, which writes a global array) and a test, so it
cannot return.

This narrows the blind spot to its stated v1 edge and no further: a function
whose only effect is on globals still survives a claim it should not, until
item 6 below compares those writes by image offset. Zero false refutations
holds again on the fixtures; the next real V0 pass is where it is retested.

## V0's speed, and the lost walk (F16)

The first V0 passes were slow - 50 pairs in 20-37 minutes - and the
instruction count, added to find out why, showed the cause was not
instructions. The finishers took a median of 40 and a p90 of 153; the time
went to a few functions that never came near the budget. `flecs_query_pred
_eq_name` took 35 seconds a run at 37,000 instructions, ending "mapped too
much memory".

The cause: a function that walks a data structure - a query engine over a
list or hash table - given a garbage pointer, chases it as if it were a real
node, touching a fresh far page each step. The harness maps and fills each
page on first touch, in Python, and the page limit was 4096 (16 MiB), so a
lost walk filled thousands of pages before stopping. Three functions were
43% of a 50-pair pass.

Two fixes, both keeping the verdict unchanged and only reaching it sooner:
the page limit drops to 512 (2 MiB), which a real function never approaches
but a lost walk hits in a fraction of the time; and a wall-clock timeout
(2 s by default) is the last safety net, so a function that stalls in any
way the budget and the page limit do not foresee is TIMED_OUT - inconclusive,
like the budget, never a refutation. Budget and timeout are told apart by
whether the instruction count reached the budget.

## A stub crashed on a garbage size (F17)

The first 3000-pair run died in seconds with a MemoryError: memset(dest, c,
n) was called with n in the tens of billions - a pointer the function passed
where a count belonged, since V0 feeds garbage - and `bytes([c]) * n` tried
to build forty billion bytes. A real memset of that size would exhaust
memory too, but the verifier must not: it has to decline, not die.

The fix is one sanity bound, MAX_TRANSFER (16 MiB), checked in one place -
the Machine's read and write, and a `bound` a stub calls before it
constructs a buffer itself. A size past it raises StubDeclined, so the input
is inconclusive, never a crash and never a wrong result. Every stub, present
and future, is covered because the bound sits in the Machine they all go
through, not in each stub. 16 MiB is far past any real single transfer and
past the page-map limit, so nothing legitimate is affected.

## Pointer returns refuted true claims (F18)

The first 500-pair run found eight false refutations, all "return value
differs", all on functions whose names give them away: utf8proc_version,
opus_strerror, pagetype_caption, opj_mct_get_mct_norms and four more. Every
one returns a pointer - checked in the signatures, six of six `return
pointer` - into the binary's own image or the heap: a version string, an
error message, a static table. That address is at a different value in -O0
and -O3, the same way a global's address is (F15), and the harness compared
the returned RAX as an integer and refuted a true claim.

This is F15's sibling on the return side, and the design's own rule - a
pointer is never compared by value - simply had not been applied to the
return. A returned pointer now has its own kind in the placement, and
compare does not compare it, exactly as it does not compare void; its mask is
zero as a second guard, so even read as an integer nothing of it is
compared. The function's effect is judged by the memory it wrote, not the
address it handed back. A pointer-returning fixture (`banner`, returning a
.rdata string) and a test hold it, mutation-checked against both guards.

All eight V0 disagreements were this one cause; with it fixed the next real
pass is where zero false refutations is retested at scale.

## The 3000-pair run leaked memory (F19)

The 500-pair run was clean, but 3000 pairs killed WSL: the process reached
9 GB by 150 pairs and was OOM-killed, with system time (12 min) dwarfing user
time (3 min) - the signature of runaway allocation, not computation. Measured
directly, memory grew about 60 MB per pair and never came back.

The cause: each run creates a Unicorn emulator that maps the whole binary
image plus the stack and arena, and holds it C-side; the Python object does
not free that when collected, and the hook closures reference the emulator, a
cycle the collector breaks slowly if at all. Over V0's tens of thousands of
runs it leaks gigabytes.

The fix: run() creates the emulator, delegates the body to a helper, and
releases the handle in a finally - so every return path, including a fault,
frees the C-side memory at once. Measured after: 2500 runs hold flat at
~115 MB where before 150 pairs reached 9 GB. Two tests check the release
happens on a normal run and on a faulting one, each with a fresh spy so no
count leaks between tests.

## A second leak: the retained pefile objects (F20)

Fixing F19 cut the leak but did not end it: 3000 pairs still grew past memory
and was killed, now more slowly. Measured, the growth tracked the number of
distinct binaries opened - about 135 MB per Loader, 96 loaders by 200 pairs -
not the number of runs. The Loader kept its pefile object (opened with
fast_load=False, which holds the whole file and its parsed structures), and
V0 caches one Loader per binary, so a hundred large binaries retained a
hundred heavy objects.

The Loader now keeps only what mapping needs - the header bytes and each
section's bytes at its virtual address, a few MB - and drops the pefile
object at construction. write_sections maps from those raw bytes. Measured
after: 200 loaders hold ~100 KB each instead of 135 MB, a thousandfold drop.
A test checks the pefile object is gone and that mapping from the raw bytes
still runs a function correctly.

Together F19 (the emulator's C memory) and F20 (the loaders' Python memory)
are the two leaks that killed the 3000-pair run; with both closed it holds
flat.

## The 3000-pair run: three false refutations, one fixed (F21)

The full run held coverage at 43% (stubbed) and kept zero false refutations
almost: three of 3000 disagreed, all "return value differs". They were three
different causes, not one.

cglm/glmc_vec4_norm_inf returns a float, and its XMM0 was 0x4f82a000 at -O0
and 0x4f82a0004f82a000 at -O3 - the same low 32 bits, the float's actual
value, differing only in the upper bits XMM0 leaves undefined. The float
return was compared over all 64 bits, not masked to the type's 4, so a float
that agreed in value refuted on undefined bits. Fixed (F21): a float return
is masked to its width, the analogue of masking RAX for a narrow integer, and
a fixture-built test with differing upper bits holds it, mutation-checked.

The other two - libarchive/lzx_read_bitlen (two pointers and an int) and
capstone/AArch64_map_vregister (an int, likely a table lookup) - return
integers that genuinely differed, and are a different class: a function whose
result depends on input data we filled with garbage, or on a global table
read at a different address. These are input-quality, the province of richer
input generation (roadmap item 2), and are left recorded until then rather
than papered over. Two in 3000 is 0.07%, and neither is a harness bug the way
F21 was.

## The fills were the wrong way round (F22)

Looking into the remaining disagreements turned up something worse than they
were: the two fills the design specifies had ended up inverted in the code.

The design says argument memory is the function's *input* - filled from its
address alone, identical in every run - while the stack is *uninitialised* and
takes the seed, which is what lets running twice with two seeds spot a result
that depends on garbage. The code did the opposite: every page mapped on
first touch, argument buffers included, took the run's seed, and the stack
was left as Unicorn's zero fill, identical in both seeds.

Both halves broke something. A function reading its own input buffer gave a
different answer per seed, so the two-seed check called a legitimate,
input-dependent result garbage and excluded it - the 3.1% "ran but nothing to
compare", and judging power lost across the board. And a function reading an
uninitialised local read zeros in both seeds, so the check that exists to
catch exactly that could never fire; risk 2 was unguarded.

Fixed: a page in the input-buffer region is filled from its address alone, a
page anywhere else - the stack, a wild pointer's page - from the seed, and
the stack is filled with seed garbage at setup instead of left zero. Tests
hold both halves, and the first version of the input test passed even with
the fix disabled (count_nonzero counts non-zero bytes, and random fill has
almost none zero); it was sharpened to `sum`, whose result depends on the
bytes themselves, before it caught the mutation.

## A written pointer is an address too (F23)

Fixing the fills (F22) raised coverage to 44.8% and cut "nothing to compare"
from 3.4% to 0.8%, and it changed which pairs disagreed: the three that
remained were now all "written memory differs" - rhash_ripemd160_final, FLAC
cuesheet, lexbor style_mutation_init.

Read byte by byte, the difference was a pointer. lexbor's buffer held
...02000000 identically in both versions with four bytes before it differing:
one 64-bit value, 0x298695_0e0 against 0x20605_6020 - the address of a global,
different in the two builds because their image bases are. The function
writes the address of something of its own into the caller's buffer, and we
compared that address as data.

This is the third sibling: F15 excluded writes *into* the image, F18 excluded
a *returned* pointer, and a pointer *written into a buffer* was still
compared. Now, before written pages are compared, each aligned 8-byte word
that falls inside either binary's image is blanked - only image addresses,
so a pointer into an input buffer or the arena, which sits at the same
address in both versions, still compares, and so does all real data. A
fixture that writes a global's address beside a plain count (`publish`) and
a unit test of the mask hold it, mutation-checked.

## A difference we caused is not evidence (F24)

F23 left one disagreement in 500: rhash_ripemd160_final, and only in the
stubbed layer. Read closely, -O0 ran 2576 instructions and wrote one buffer;
-O3 ran 1710 and wrote two, the second exactly 20 bytes - a RIPEMD-160
digest. The two versions took different paths, because the context they were
handed is a buffer we filled with an invented pattern, not a real hash
context. A function given a meaningless context does something meaningless,
and the two builds need not do the same meaningless thing.

So the input fill becomes an axis of the comparison, the way the stack fill
already was. Every input is now judged under two input-buffer fills, and a
refutation counts only if **both** fills refute. A difference that appears
under one fill and not the other came from bytes we invented: the claim
cannot be judged on that input, and it is inconclusive - never refuted,
because a difference we caused is not the function's.

The cost is double the runs per input; the tests check that it does not
blunt anything - two genuinely different functions are still refuted through
both fills, and a true pair still survives - and each branch of the rule has
its own test, mutation-checked.

This closes the class, but only by declining it. Judging such a function
properly needs a *valid* context, which means calling its initialiser first;
that is the constructor-chain work, the next item on the roadmap, and it is
how these move from inconclusive to actually verified.

## A stub's writes were invisible (F25)

F24 raised coverage to 47% but left rhash_ripemd160_final refuted, and under
*both* input fills - so it was not the invented data after all. Following it
down: the context buffer came out **byte for byte identical** in the two
versions, so both had done the same work; the difference was only that -O3
wrote the 20-byte digest into the caller's buffer and -O0 wrote nothing. And
the import trace said why: -O0 called memcpy, -O3 had inlined it.

The harness records writes from Unicorn's write hook, which fires for
emulated instructions. A stub writes through the emulator's API instead, and
that does not fire the hook - so every effect produced by a stub was
invisible. A function whose output is written by memcpy looked as if it
produced nothing, while the build that inlined the copy looked as if it did,
and the pair was refuted. This was not one function: memcpy, memset, strcpy,
calloc - every stub that writes - had its effects dropped.

The write rule is now one function that both the hook and Machine.write go
through, so a stub's write counts exactly like the function's own, across
whole pages the write spans. Two tests hold it, mutation-checked.

This is the deepest bug the verifier has had: not a false difference it
invented, but a real effect it could not see.

## Constructor chains - protocol

Written before the code. The 500-pair run after the stub families says where
the remaining coverage goes, and it is one thing in three disguises:

| bucket / decline | share | why |
|---|---|---|
| fault | 17.6% | a function walks a structure we filled with invented bytes |
| abort, _assert | 19 of 46 declines | a precondition we violated - an assertion the function never fails in practice |
| free of a bad pointer | 10 of 46 | a pointer that came out of our invented bytes |
| a size past the sanity bound | 6 of 46 | a count that came out of our invented bytes |

All of it is the same cause: **a function that takes a context expects that
context to have been built by its own initialiser, and we hand it noise.**
rhash_ripemd160_final wants a context from rhash_ripemd160_init; a parser
wants a state from parser_new. Given noise they take paths that mean nothing,
and the verifier can only decline.

**The chain.** Before calling the function under test, call its initialiser,
so the context it receives is one the library built:

```
rhash_ripemd160_init(ctx)        the initialiser, run first
rhash_ripemd160_final(ctx, out)  the function under test
```

**Each side runs its own initialiser.** Q's context is built by the
initialiser in Q's binary, K's by the one in K's. The alternative - build one
context and hand it to both - is unsafe: a context can hold a pointer into
the binary that made it (a function pointer, a table), and that address does
not exist on the other side, so the other version faults and a true claim is
refuted. Two initialisers, two contexts, same inputs: the comparison stays
honest.

**Finding the initialiser, cautiously.** A candidate must satisfy all of:

- same package and same source file as the function under test;
- a name that is the function's name with its last segment replaced by
  `init`, `new`, `create`, `setup`, `start`, `open` or `alloc` - so
  `rhash_ripemd160_final` looks for `rhash_ripemd160_init`, and a prefix
  match alone is not enough;
- a first parameter of the same resolved type as the function's first
  parameter (the context), and that parameter a pointer;
- a signature the harness can place, and no more than that one pointer
  argument plus integers, so the chain call needs no invented data of its own.

If none fits, or more than one fits, **no chain is run** and the function is
tested as it is today. A wrong initialiser would write a wrong context, and a
wrong context is worse than noise: it looks valid.

**Both sides must succeed.** If either initialiser does not run to completion
- it calls an unstubbed import, it faults - the chain is abandoned and the
claim is inconclusive for that input, never refuted. A context half-built is
not a context.

**What it does not do.** A function whose context needs more than an
initialiser - a parser that must be fed data, a socket that must be connected
- is still out of reach, and stays inconclusive. This closes the common case,
not every case.

**Measured how.** V0 gains a third layer, `chained`, on the same sample: the
same pairs, the stubs, and a chain where one is found. The three layers -
bare, stubbed, chained - say what each is worth, and the false-refutation
count must stay zero through all three. A chain that turns an inconclusive
pair into a refuted one is a bug in the chaining, not a discovery, and is
investigated as such.

## What the chain was actually worth (F26)

The constructor chain was built to answer the largest remaining bucket, and
measured on 500 pairs it moves coverage 0.6 points: 53.0% stubbed to 53.6%
chained, from 16 pairs where both sides had an initialiser, 7 of which could
not run it and fell back. Two rounds of work for half a point.

The first round *cost* 0.8 points, because a chain that failed lost a pair
that had been judged before. A chain is an attempt to improve, not a
precondition; it now falls back to the unchained run.

The second round asked why it applied so rarely, and the measurement broke
the assumption the design was built on. Across 26,285 context-taking
functions in train, a name match finds an initialiser for 3.1%. The
`X_init(ctx)` shape - the one the design pictured, from hash APIs - is
uncommon in C libraries; of the initialisers a name match does find, 275
take no arguments because they *return* the context (cJSON_CreateObject,
xmlNewDoc), and 266 want a second pointer of their own, which the chain
still refuses since filling it is the problem it exists to avoid. Accepting
the returning shape moved it from 13 pairs to 16.

**The lesson is about where the remaining coverage is.** What blocks it is
not architecture: faults (17.4%), stub declines (8.8%, of which 19 are an
abort or assert the function reaches because a precondition we cannot know
was violated), and exhausted budgets (7.2%) are all one thing - a function
given data that means nothing to it. Producing data that *does* mean
something requires knowing the library's own contract, and that contract is
not in the binary. The chain reaches the small part of it that naming
conventions expose, and no more.

Kept because it is free: it never refutes, it falls back when it fails, and
where it applies the function is verified on a context its own library
built, rather than declined.

## A bounded fill and chunked mapping (F27)

The first thing taken from the related work (docs/related-work.md). PEM folds
invalid addresses into one bounded random block rather than materialising
memory, and names the two properties such a model must hold:
equivalence-preserving, so two equivalent runs read the same bytes at the same
addresses, and difference-revealing, so two different addresses read different
ones. A constant fill holds the first and fails the second; ours held both but
built each page in a Python loop, which was most of what a lost walk cost
(F16), and was the reason the page budget had to be cut to 2 MiB.

The fill is now a window into a cached block of 1021 pages, chosen by the
address modulo the block's size: 22x faster, both properties intact. 1021 is
prime deliberately - a block size sharing a factor with the stride between
input buffers would give two different pointer arguments identical bytes, and
a test holds that sixteen consecutive buffers stay distinct.

Measuring the rest of a lost walk turned up something better. The cost was
never mostly the fill: Unicorn's mapping grows faster than linearly in the
number of separate regions, so 4096 pages mapped one at a time take 19
seconds where the same 16 MiB in 64-page chunks takes 53 ms. Memory is now
mapped a chunk at a time, clamped away from the null guard and falling back to
a single page where a chunk would overlap something already there. A walk
moves forward anyway, so the neighbours a chunk brings are usually the ones it
wants next.

The first version of this got the budget wrong and the measurement caught it.
It kept the limit at 4096 *pages* while making a touch map 64 of them, so a
walk that used to get 512 distinct touches now got 64: coverage fell 2.2
points and "mapped too much memory" more than doubled, 2.6% to 5.6%, from a
change meant to reduce it. What the budget has to bound is touches, not
pages - a function reaches few separate places, a lost walk reaches many.

Measured by region count, 512 touches of 16 pages is 32 MiB and 83 ms, where
2 MiB page by page was 390 ms. So: a 16-page chunk, a budget of 512 touches,
sixteen times the memory at a fifth of the cost. A test asserts the touch
count rather than the page count, so raising the chunk again cannot quietly
shrink the walk.

## Measuring the two knobs F27 left open (F28)

F27 changed two things at once - what the fill contains and how memory is
mapped - and the run that followed was worse on both counts it was meant to
improve: coverage 54.2% to 52.0%, and then, after the touch budget was fixed,
a run of 15m40 against 7m22. Changing two things and measuring the pair is
how that happens; this scanned them separately, 150 pairs, four
configurations.

| fill | touches | coverage | time |
|---|---|---|---|
| random | 128 | 54.7% | 142 s |
| random | 512 | 54.7% | 249 s |
| arithmetic | 128 | **57.3%** | **134 s** |
| arithmetic | 512 | 57.3% | 368 s |

Both knobs point one way. **The arithmetic fill beats random by 2.6 points**,
which is the reverse of what was assumed when PEM's random block was copied:
their idea - one bounded block, indexed by address - is right and is kept, but
their *content* is worse here. A borrowed idea is worth having; a borrowed
parameter has to be measured on the corpus it will run on.

**And 128 touches judge exactly the same pairs as 512, in a third of the
time.** A walk that has reached 128 separate places has already lost its way;
letting it reach 512 buys nothing and costs 234 seconds. F16's lesson stands
and was briefly forgotten: stopping a lost walk early is both faster and no
less accurate, because the verdict is inconclusive either way.

Zero false refutations in all four configurations.

## Two tiers: refutation and corroboration - protocol

Written before the code. The related work (docs/related-work.md) says plainly
where our remaining 38% goes and how everyone else got past it: BLEX starts
execution at any uncovered instruction, PEM flips branch outcomes to force a
function past its own input checks. Both reach near-full coverage by giving up
natural execution, and both are explicit that what they then measure is
similarity, not equivalence - BLEX says so in as many words.

We cannot give that up: the whole value here is a verdict, and a verdict rests
on a refutation that is never wrong. So the two are separated rather than
merged.

**Tier 1, refutation.** Exactly what exists today. The function runs from its
entry, on arguments the input generator produced, along whatever path it
actually takes. A difference seen here is the functions' own and refutes the
claim. Nothing in this tier changes, and its rules - the two seeds, the two
input fills, the pointer masks - continue to hold.

**Tier 2, corroboration.** The function is run again with one or more branch
outcomes forced. Such a path is *infeasible*: no real input would take it. So

  - a **difference** on a forced path proves nothing and **can never refute**;
  - an **agreement** on a forced path is evidence that the two functions do the
    same thing, collected from code tier 1 could not reach.

That asymmetry is the whole design. It lets us take the coverage trick from
the literature without taking the cost that comes with it.

**A verdict gains a strength.** `survived` stops being one thing:

| verdict | meaning |
|---|---|
| survived (feasible) | agreed on paths the function really takes |
| survived (forced) | agreed only on forced paths - weaker, and said so |
| refuted | differed on a feasible path |
| inconclusive | neither tier could judge |

A pair that today reports "inconclusive, the function faulted" can report
"agreed on 12 forced paths covering most of its instructions", which is worth
less than a feasible survival and much more than silence.

**Which branch to force.** If the two versions force *different* predicates,
the comparison is meaningless. We cannot know which predicate in -O0
corresponds to which in -O3, but PEM's answer is that we do not have to: rank
each version's predicates by *dynamic selectivity* - the distance |x - y| at
the comparison - and pick at the same rank in each. Their argument is that
optimisations remove, duplicate and move predicates but do not invent them, so
the extremes of that ranking are the part most likely to survive; they measure
it holding over 80% of the time, and sample the ranked list from a U-shaped
Beta so both ends are favoured. We take the mechanism as theirs and will
measure the parameters here (F28's lesson).

**Selectivity needs the compared values**, which means decoding the compare
that set the flags - so capstone joins unicorn in the `verify` extra. The
decode is cached per address and only runs when predicate recording is asked
for, so tier 1 pays nothing.

**Built in three measured steps**, one at a time:

1. *Record* predicate instances during a run - address, instruction count,
   both targets, which way it went, selectivity. No behaviour change; the
   measurement is what the ranking will be built on.
2. *Force* a chosen outcome at a chosen instruction count, and check the run
   goes where it was told.
3. *Judge* with the two tiers, and measure what corroboration is worth
   against the false-refutation count, which must stay at zero.

## What forcing is worth, and what it costs (F29)

The second tier works. On 500 pairs it moved 8.0% from unjudged to agreed -
and it drew from exactly the buckets it was aimed at: faults 18.6% to 15.2%,
exhausted budgets 5.8% to 4.2%. Zero false refutations, which the asymmetry
guarantees and the run confirms. The constructor chain bought 0.6 points for
two rounds of work; this bought 8.

The first version also took 27 minutes where the run before it took 5, and
nineteen of those minutes were kernel time - mapping an image per run, for
runs that turned out to change nothing. Scanned over 150 pairs:

| attempts | inputs | forced agreement | time |
|---|---|---|---|
| 1 | 1 | 4.0% | 274 s |
| 2 | 1 | **5.3%** | **319 s** |
| 4 | 1 | 5.3% | 322 s |
| 2 | 6 | 5.3% | 477 s |
| 4 | 6 | 5.3% | 482 s |

Two findings. A third attempt buys nothing, so two is the budget. And **the
number of inputs changes the gain not at all**: if forcing is going to work
on a pair it works on the first input, and the other five only cost time.
That is worth understanding rather than just using - the input decides which
path the function takes to the check, but the check is the same check, and
forcing it is the same act whichever way it was reached.

## Thirteen in three thousand (F30)

500 pairs had held zero false refutations for several rounds. 3000 turned up
thirteen - 0.43%, and they were two faults in this design rather than
thirteen surprises. Six of the thirteen were the same two functions in
different packages, `__g__fmt` and `__freedtoa`, which are David Gay's dtoa
bundled into half the corpus.

**Compared memory that was never ours.** The write rule excluded the stack
and the image and recorded everything else, while its own comment said only
memory the harness handed out is comparable. Those are not the same thing. A
function walking a pointer we invented writes wherever that pointer led, and
that address differs between the two builds for a reason that is ours - so
one version writes a page the other does not, the dictionaries differ, and a
true claim is refuted. The dtoa cases showed it plainly: pages at
0xfffff1f8aa000 written by one build only, and an eight-byte value straddling
the boundary just *below* an input buffer, where a function had left the
space we gave it. Only two regions are comparable, the input buffers and the
arena, both at the same addresses in both versions, and only those are
recorded now.

**Refuted on memory neither version could settle on.** A page whose content
differed between the two seeds was dropped as garbage - correctly - but then
the two versions' dictionaries were compared whole, so a page dropped in one
and kept in the other looked like a difference in what the functions wrote.
A page either version could not settle on is not evidence about either, and
is now left out of the comparison rather than counted as a difference.

Both are mutation-checked, and the tests say what they protect: a write
pointed outside the given buffers is not recorded while the same write into a
buffer is, and a page dropped in one version does not refute while a real
difference in written memory still does.

The lesson is about sample size. Five hundred pairs said zero for four
consecutive rounds of work; three thousand said thirteen. A rate of 0.4%
needs thousands of pairs to show itself at all, and the rule this project
rests on is exactly the kind that fails rarely and matters every time.

## Byte-precise writes: ten of the thirteen (F31)

F30 fixed one of the thirteen. The write log - added to stop guessing about
the other twelve - showed the traces of the two builds were *identical in
structure*: the same twelve stores to the same places. The differences were
in what those stores contained, and reading them byte by byte gave three
rules, each of which a page-granular comparison could not express.

**Only bytes the function wrote.** A recorded page holds 4096 bytes and a
function may have written four; the rest is the fill it never touched.
Comparing that compares our own pattern.

**Only bytes *both* versions wrote.** box2d showed why: -O0 zeroed a struct's
padding and -O3 left it holding the fill, and we called that the two
functions disagreeing. Padding is indeterminate in C, and a byte one version
never touched is that version's padding whatever it holds.

**A store with one garbage byte is garbage whole.** One instruction writes
one value. dtoa's `__g__fmt` wrote eight bytes four below a buffer, and the
low half - seed-dependent, so garbage - landed on one page while the high
half landed on the next; page-granular detection dropped the page with the
garbage and kept the page that refuted. The taint now spreads over the store.

**And a store only half of which we can see cannot be judged either.** The
same eight bytes, when they straddle out of comparable memory, leave part of
the value somewhere we never look - `__freedtoa` writes four bytes below the
first buffer. Half of one value is not a value.

Ten of the thirteen are gone: both dtoa families, both box2d cases,
leptonica's sort, jerryscript's descriptor. The image-pointer mask is now a
set of byte offsets and checks unaligned positions too, since a compiler may
store a pointer anywhere.

## The last three: a difference we read into being (F32)

F31 left three, and the disassembly settled all three at once.

capstone's `AArch64_map_vregister(0)` is the clearest. It expects a register
id of at least one, and at -O0 it computes `id - 1` in a 32-bit register,
which for zero underflows to 0xFFFFFFFF, doubles it, and reads a 16-bit entry
*eight gigabytes* past its table. That address is nowhere the binary has
anything; it is a page our harness mapped and filled. The value returned is
ours. -O3 does something else with the same undefined input, as a compiler
may, and we called the two builds different functions.

The other two are the same shape: `lzx_read_bitlen` loads a pointer out of a
buffer we filled and dereferences it, and opus's quantiser decides its single
byte of output after reading the same kind of memory.

**A page outside the input buffers and the arena can only be at an address
the function computed from the bytes we invented** - a garbage pointer, an
index that underflowed. What it reads there is our fill pattern, not the
binary's data, and a difference that follows from it is ours.

So a run records whether it touched such memory, and a difference between two
runs where either did cannot refute - it is inconclusive. The asymmetry is
the same one the second tier rests on, and for the same reason: agreement is
unaffected. Two builds agreeing, even while reading our pattern, is still two
builds agreeing, and V0's coverage is agreements, so this costs nothing
there.

What it does cost is refutation power against a wrong candidate that reads
wild memory - we can no longer tell its difference from ours. That is the
honest side of the trade, and the right side to be on: the rule is that a
true claim is never refuted.

Thirteen to zero, in three findings.

## Garbage-dependence belongs to the run (F33)

F32 left one: opus's `silk_NLSF_del_dec_quant`. It writes a single byte, and
the two builds wrote 0x00 and 0x01 under one input fill and 0xa5 and 0xa6
under the other - the fill's own byte at that offset, and that byte plus one.
The difference is there under both fills, so F24's rule does not fire, and
the function stays inside the buffers we gave it, so F32's does not either.

The proof is in the return: it changes with the fill seed. That is a run
reading memory it never wrote, and the byte it wrote came out of the same
computation.

The rule was applied too narrowly. Garbage-dependence was treated as a
property of one *output* - the return was excluded and the writes were still
compared - when it is a property of the *run*. A run whose return depends on
the fill read uninitialised memory, and everything else it produced came
through the same undefined arithmetic; a difference there is no more the
functions' than the return was. So it cannot refute, and, as everywhere else
in this design, agreement still counts.

Thirteen to zero, in four findings: memory that was never ours (F30), bytes
neither version wrote or one could not settle on (F31), memory reachable only
through an address we invented (F32), and now outputs of a run that read some
(F33). Every one of them is the same sentence said more precisely - a
difference we caused is not evidence.

## The other half of the measurement: R0 (F34)

Four rounds of work took V0's false refutations from thirteen to zero, and
each round made the verifier more careful about what counts as evidence. None
of them measured what that carefulness cost, because nothing in this project
measured it: **V0 only has true pairs.** A verifier that refuses to refute
anything passes V0 perfectly and is worthless.

R0 is the other half. It builds pairs that are wrong by construction - one
function at -O0 against a different function from the same package at -O3 -
and asks what the verdict is. Every refutation is correct; every survival is
a miss.

The first measurement, 300 wrong pairs:

| | refuted | inconclusive | survived |
|---|---|---|---|
| before F32 and F33 | 16.7% | 59.3% | 24.0% |
| after | **12.7%** | 62.3% | 25.0% |

So the two taint rules cost four points of refutation power: twelve wrong
pairs in three hundred that used to be caught now go unjudged. That is the
price of the thirteen false refutations, and it is the right way round - a
miss costs an opportunity, a false refutation costs the verifier's word.
Written down so the next change to the comparison is made knowing it.

**And the measurement found something the V0 side could not see.** A quarter
of the wrong pairs *survived*, and a sixth of those were pairs where neither
function produced anything observable at all: a void return, nothing written,
on both sides. The verifier was calling that agreement. It is not - two
functions that each declined to show us anything have not agreed on anything,
and that is exactly what inconclusive is for. A return that is never compared
(void, or a pointer by F18) is not an observation, and neither is a page with
no comparable byte in it; agreement now requires at least one of the two.

That change lowers V0's coverage a little, honestly: pairs that agreed on
doing nothing were being counted.

`elenchus verify-r0` is permanent, beside `verify-v0`. They are read
together, and the failure mode they exist to catch is a change that cleans
one while quietly ruining the other.

## A pointer into the stack is an address too (F35)

The 3000-pair run left one false refutation that the thirteen had not
included: `libspng/compress2`, refuted in the stubbed and chained layers on
*written memory differs*. It appeared only after F31 made the comparison
byte-precise, which is the first thing it says about itself - a page-granular
comparison had been hiding it behind coarser noise.

Everything the earlier rules test came back clean. Both builds completed, at
4,616 and 2,154 instructions. The return was stable under both seeds and both
fills, so F22 and F33 do not fire. Neither run touched memory outside the
buffers and the arena, so F32 does not. No store was partial. The refuting
page was the arena's first block, and of the 855 bytes the judge compared
there, **one** differed.

Read as the eight bytes it belongs to, that one byte is this:

| | value | points into |
|---|---|---|
| -O0 | `0x00007fefffffdf80` | the stack |
| -O3 | `0x00007fefffffdf50` | the stack |

The arena's first block is zlib's `deflate_state`, and its first field is
`strm` - a back-pointer to the `z_stream` that `compress2` holds in its own
frame. -O0 and -O3 size that frame differently, so the stored address differs
in its low byte, and the two builds looked like two functions.

**The mask was narrower than the sentence it serves.** F23 excluded a written
pointer into either binary's *image*, and its comment said a pointer into an
input buffer or the arena still compares because those sit at the same
address in both versions. That is right, and it is also the whole rule: only
what the harness hands out corresponds between the two builds. The image was
one region that does not. The stack is another, and nothing excluded it.

So the ranges the mask is given are now the two images **and the stack**, and
the helper is named for what it does - `_foreign_pointer_bytes`, the offsets
holding an address the harness did not hand out. A pointer into an input
buffer or the arena still compares, as before; the diagnosis confirmed that
it must, because every other pointer in that `deflate_state` is an arena
address and all of them matched, which also rules out the alternative reading
that the two builds allocate differently.

This is a class and not a case, which is why it is worth the paragraph: a
context structure holding a back-pointer to something on the caller's stack
is an ordinary C shape, and `X_init(ctx, caller_struct)` is exactly what the
constructor chain goes looking for. The same pattern would have refuted every
library that writes one.

The cost is the same one F23 accepted: a genuine data value that happens to
land in the stack's megabyte can no longer refute. R0 says what that is worth
in refutation power, and it is measured beside V0 rather than argued about.

The tests say what they protect - that a pointer into the stack is masked
while a pointer into the arena beside it is not, and that this exact pair
survives - and the mutation check removes the stack from the ranges and
watches both fail with *written memory differs* on page `0x300000000000`, the
line V0 reported.

**Confirmed at three thousand pairs.** The rule was written against one case
and checked on 500, which is the sample size that had never shown the fault in
the first place. The 3000-pair run is the one that could, and it is clean:
zero disagreements in all three layers, over 1105, 1364 and 1371 completed
pairs. Coverage did not move at all - 36.8 / 45.5 / 45.7 per cent against
36.8 / 45.5 / 45.7 before the fix, and forced-only agreement 5.4 against 5.4.

That the coverage is identical to the decimal is the part worth recording. A
wider mask excludes more bytes, and the way it could cost something here is by
leaving a pair with no comparable byte at all, which would move it into "ran
but nothing to compare". No pair in three thousand was left that way: where a
written pointer into the stack appears, real data appears beside it. The other
half of the cost was already measured and was also nothing - R0's refutation
power at 500 wrong pairs is unchanged by this rule.

**And the same run corrects what the chain is worth.** F26 measured the
constructor chain at 0.6 points on 500 pairs. At 3000 it is 0.2: 45.5 per cent
stubbed to 45.7 chained, from 86 pairs with an initialiser on both sides, 31
of which fell back when it failed. This is F30's lesson in the other
direction - a small sample hides a rare fault, and it also flatters a small
gain. The chain stays, on exactly the grounds it was kept on before: it never
refutes, it falls back, and where it applies the function is tested on a
context its own library built. It is worth a fifth of what was recorded.

**What the import histograms say at this size.** The stub families are done:
the commonest import still without a stub is getenv, ten times in three
thousand pairs, and the tail below it is either unstubbable by rule (longjmp,
SystemFunction036) or too rare to pay for. The declines point somewhere else.
After the preconditions we violate ourselves - abort at 80, a free of a
pointer our own invented bytes produced at 60, _assert at 35 - the largest is
InitOnceExecuteOnce at 54, declined because the one-time initialiser's
callback is not run. That callback is code in the same binary, which tier 1
follows for free everywhere else; running it needs re-entry from inside a
stub and nothing more. At 1.8 per cent it is nine times what the constructor
chain bought, and it is the next thing to measure after the import
observation.

## Where the measurement's time goes, and what did not move it (F36)

The 3000-pair V0 took 112 minutes, 81 of them kernel time, and the handover
named the cause: mapping the binary's image into a fresh emulator for every
one of tens of thousands of runs. Four things were tried against it. One
worked, three did not, and the three are worth more than the one.

**Judging the pairs across processes: 1.65x, and it stops at two.** The runs
are independent, so this should divide by the cores given to it. Measured on
R0, 100 pairs, on a machine with twenty cores:

| workers | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| pairs/s | 1.45 | 2.37 | 2.40 | 1.97 |

Two is the whole of it. Four buys nothing and eight is worse than two, which
is not Amdahl's shape - a serial fraction would still improve at four. It is a
flat ceiling, so what the processes share is saturated, and what every run
asks for is memory mapped and given back. The verdicts were identical at
every worker count, which is the property that makes any of this allowed.

**Keeping the stack fill: nothing.** Every run generated a megabyte of
random bytes for the stack, deterministic per seed and thrown away after.
Cached per seed - the same shape as the page fill already had - a fixture run
went from 6.50 ms to 2.47 ms, 2.6x. On the corpus it moved the rate from 1.50
to 1.45 pairs a second, which is noise. The fixture is a poor proxy: its
functions run tens of instructions where real pairs have a p90 at the 5000
budget, so the fixed cost the cache removes is a large share of one and a
small share of the other. The change is kept because it removes work that was
plainly repeated, and it is recorded here as worth nothing measurable.

**A smaller stack: 1.27x serial, 1.14x at two workers - declined.** A run maps
a megabyte of stack and writes a megabyte into it, while a function starts
0x2000 below the top and uses a few kilobytes. Scanned at 1 MiB, 256 KiB,
64 KiB and 16 KiB, with R0's verdicts read beside the clock at every size:
they did not move, and the time fell 68.1s to 53.5s at the smallest. Declined
anyway, for two reasons. The gain shrinks to 1.14x under the two workers we
are keeping, so it is worth about ten per cent of a real run. And 16 KiB
leaves 8 KiB below the entry frame, so a function with a large local array
faults where it used to run - a coverage loss that R0 at 100 pairs cannot
see and that only a V0 run would show. Ten per cent is not worth buying with
a risk to the thing V0 measures.

**Forcing fork rather than spawn: nothing, and it cost something.** Asking a
pool for max_tasks_per_child makes Python refuse fork and fall back to spawn,
where each worker starts a fresh interpreter and imports everything again,
`pyghidra` included - a plausible reason eight workers beat two. Measured, it
was not the reason: forking gave 1.45 / 2.37 / 2.40 / 1.97 against spawn's
1.50 / 2.47 / 2.43 / 2.16. And from Python 3.14 forking a process that has
threads is deprecated and can deadlock the child. Reverted to the platform's
own choice; what the recycling was for is done where it belongs instead, by
bounding the loader cache to the last few binaries.

**What this says about the obvious next lever.** Reusing the emulator between
runs - mapping the image once and resetting only what changed - is the move
the handover named, and it is a real refactor against F19's rule that every
run releases the handle. The evidence now says not to make it. Removing seven
eighths of the stack removes most of what a run maps and moved the ceiling by
fourteen per cent; removing the rest cannot be worth several times that. The
ceiling is not the bytes each run maps, and a refactor aimed at them would
reach the same wall with a leak risk attached.

So the measurement is 1.65x faster and that is where it is left: a 3000-pair
V0 falls from about 110 minutes to about 67. The daily pulse, 150 pairs, was
already a minute or two. Recorded with the rejected options because the next
person to look at this will have the same four ideas, and three of them have
now been paid for.

## What else could be observed: the census - protocol

Written 10 October, before any code. Refutation power is 10.6% on R0, and it
rests on two observations: the masked return, and the bytes both versions
wrote. The roadmap asked for a third, the sequence of imported calls; the
second survey (docs/related-work.md, section 5) says that is the weakest of
BLEX's seven features and points at three things that may be worth more. This
protocol measures all of them in one pair of runs, and decides nothing until
the numbers are read.

**Nothing in it can change a verdict.** No rule is added to `_judge`; `_judge`
reports two counts it already computes. The harness records three more things a
run already sees. The exit criterion is that nothing moves: V0 at 3,000 pairs
gives the same buckets in every layer, to the pair, with zero disagreements,
and R0 at 500 gives the same verdicts, both against the runs taken after F38
and F39 - which moved them on purpose, and before the census code exists. A
census that moves a bucket has changed behaviour, and is reverted rather than
explained.

### What is recorded

- **The imports a run called**, in order, when the stub served the call. A
  declined or missing stub ends the run, so that pair is not compared anyway.
  Capped at 256 with an overflow flag; a truncated list is reported apart,
  since a cut multiset misleads as much as a cut sequence.
- **The first wild access**: the kind (read, write, fetch) and address of the
  first touch outside the buffers and the arena - the event that sets
  `read_invented` - or the address of a null dereference.
- **The arena's allocations**: the size of each block an allocation stub
  handed out, in order, leaving out the runtime-state blocks (`_errno`, the
  stdio table).

All three are reset after a chained initialiser runs, where `written` is: what
the initialiser did is setup. (`write_ranges`, `partial_writes` and
`read_invented` are not reset there today, against the comment beside
`written.clear()`. That is a behaviour question with a measurement of its own,
recorded under Hardening, and the census does not touch it.)

The pure logic - inverting the fill, the counting gates, the aggregation -
lives in `elenchus/emulation/census.py`, which holds no emulator state and is
tested without Unicorn, like `abi`, `inputs`, `branches`, `forced` and `chain`.

### The eight observations

Everything is read on the **stubbed** layer of V0, which is the configuration
R0 runs, so the two halves compare like with like.

1. **Import multiset** - the count of each name served. Also projected onto
   the allow-list the roadmap derived by argument - locks and stream output, 25
   names a compiler can neither invent nor remove - to see whether the argument
   holds.
2. **Instructions per run** - every first-tier run, not the maximum over a
   pair's runs, which is what V0 prints today as "instructions of finishers".
   That figure's p90 of 5,000 has been read as a single run's; it is not, and
   S6 and F36's open suspicion cannot be weighed without the per-run figure.
3. **Which buffers were written** - the indices of the argument buffers each
   version wrote into. `_judge` compares only bytes both versions wrote, so two
   functions writing *different* buffers share no byte and that difference is
   never seen. A buffer's index is the same in both builds by construction.
   Arena writes are reported as a byte count only: -O3 may remove an
   allocation (risk 5), after which every later arena address shifts, and a
   block-level comparison would then measure us.
4. **Allocation pattern** - the multiset of block sizes. Risk 5 says this may
   not survive optimisation; it is measured because it is cheap and ARCTURUS
   uses it.
5. **How the runs ended** - each side's status on each input. Needs nothing
   new.
6. **Null or not** - for a returned pointer, which is never compared (F18),
   whether it was zero.
7. **Observation mass of a survival** - the compared bytes, and whether a
   return was compared, for every input that survived. This is BSim's
   significance in our terms: a survival on one byte and a survival on eight
   hundred are the same verdict today.
8. **Fault anatomy** - for every run that faulted or read invented memory,
   where its first wild address came from. The fill makes this exact. A word
   of the fill is `offset * GOLDEN + c` modulo 2^64, `c` being the input
   variant inside the buffers and the run's seed elsewhere, and GOLDEN is odd,
   so it inverts. An address `a` was reached through a pointer read out of our
   fill if, for some displacement `d` with `|d| <= 1024`,
   `(a - d - c) * GOLDEN^-1` is a multiple of eight below the block size; the
   chance of that by accident is under one in a billion a run. Classified as
   null; data pointer from the input fill; data pointer from uninitialised
   fill; code pointer from either (a fetch); other.

### How a difference is counted

A difference the existing rules would call ours is not counted, or the census
would measure the ceiling of a rule that cannot be written. For observations 1,
3, 4 and 6 an input counts as differing only if all four runs under a fill
completed, each version's two seeds agree on the observation, neither version
read invented memory, and the difference appears under both input fills -
F22, F32 and F24, applied to the new observation exactly as to the old ones.

Each of them gets two numbers:

- **Risk** - on V0, the share of judged true pairs where it differs. Every one
  is a false refutation the observation would have made.
- **Ceiling** - on R0, the share of all wrong pairs where it differs *and the
  verifier does not already refute them*. A difference on a pair already
  refuted adds nothing.

### What decides, written now

- An observation becomes a **rule candidate** only with zero risk in 3,000 true
  pairs - by the rule of three, a rate below 0.1% - and a ceiling of at least
  two points of R0. Anything less is recorded and not built. A candidate is
  then a separate change with its own protocol, tests and mutation check; the
  census itself never writes a rule.
- **Survival grading** (7) is pursued if some threshold on compared bytes
  keeps at least 80% of V0's survivals and at least doubles the likelihood
  ratio of a survival, about 8 today.
- **Input repair** (8) - Tinbergen's pointer derivation, read from DWARF types
  rather than tainted - is pursued if pointers read out of our fill explain at
  least a third of the faulting pairs, about six points of coverage at stake.
- **Instructions per run** (2) are recorded for S6, which they inform and do
  not decide.

### Runs

- R0 at 150 pairs: a pulse, about a minute.
- V0 and R0 at 500 pairs: nothing moved against the numbers they gave before.
- **V0 at 3,000** (about 67 minutes) and **R0 at 3,000** (about 21): the census.
  R0 at 500 would see a two-point ceiling as ten pairs, too few to read.

The V0 run is the one v0.7.0 was waiting for anyway; the census rides on it.

### Tests

Each recorded field has a test that it records and a test that it is reset
after a chained initialiser, each mutation-checked by deleting the line that
records it. The fill inversion is tested on words built by `_fill` itself: a
pointer read out of the fill is recognised at displacements 0 and 1,024, and
random addresses are not. The counting gates are tested on hand-built
`Outcome`s - a difference under one fill only, an observation the two seeds
disagree on, a run that read invented memory - each asserted not counted, and
a clean difference asserted counted. The existing suite passing unchanged is
the first evidence that no verdict moved; the 500-pair runs are the second.

## A stub could not touch what the run had mapped (F38)

Found while reading the harness to place the census's recording, and the
largest coverage change since forcing. Faults - "a function walked a pointer
we invented" - were a sixth of every pair and the biggest bucket left. For
most of them that explanation was wrong.

A function's own memory access goes through Unicorn, which calls the unmapped
hook only for memory that is unmapped. A stub's goes through
`_ensure_mapped`, which maps every page not in `mapped` - and `mapped` holds
only first touches, because it is what the walk budget counts. The image and
the stack are mapped by the run before the call and were never in it. So a
stub reading a string constant, or writing a local, tried to map memory that
was already there; Unicorn refused with `UC_ERR_MAP`, the error came back
through `emu_start`, and the run ended as a fault. `strlen` of a literal,
`strcmp` against one, `memset` on a local struct: some of the most ordinary
calls in C, made by a function doing nothing wrong.

**Confirmed before it was fixed, then counted.** A `Machine` over a mapped
stack and image refused both, with an input buffer as control. On the first
300 pairs of the V0 sample, stubbed layer, 40 of 56 faulting pairs met it -
71% - with 1,163 collisions in the image against 144 on the stack: it was
mostly constants being read, not locals being written. That was an upper
bound, since a pair can meet this and a genuine fault both.

It never risked a false refutation - a fault is inconclusive - which is why
a clean 3,000-pair V0 never showed it. It was silencing the verifier, not
making it lie.

**The fix** hands a stub the regions the run mapped itself (image, stack,
sentinel and trap pages) and leaves them alone. Two fixture tests reuse
functions that were already there - `strlen` on the `.rdata` string
`banner()` returns, and `memcpy` into a stack address - and both failed
before the fix, which is the strongest mutation check there is. Four more
mutations each break what they should: disabling the check or the wiring
breaks both, and removing the stack or the image from the regions breaks
only the test for that one.

**Measured on the same 500 pairs as before it** (chained layer):

| bucket | before | after |
|---|---|---|
| completed | 45.0% | **50.4%** |
| fault | 17.0% | **6.4%** |
| budget exhausted | 5.0% | 7.8% |
| import without a stub | 4.8% | 5.8% |
| mapped too much memory | 4.4% | 5.2% |
| ran but nothing to compare | 7.4% | 8.0% |
| stub declined | 8.2% | 8.2% |
| completed only on forced paths | 6.4% | 6.4% |

Bare stays at 36.2% to the decimal, which is the control: the bare layer has
no stubs, so a change to how stubs touch memory must not move it, and it did
not. Zero disagreements in all three layers. The stubs are now worth 13.6
points where they were worth 8.2, because they no longer kill the runs they
serve.

Of the 10.6 points that left the fault bucket, half became agreements and
half met the function's next obstacle - an exhausted budget, an import with
no stub, a walk past the touch limit. Functions that used to die at their
first `strlen` now get as far as their real problems.

**R0 reads it differently, and both readings stand.** 53 / 414 / 33 before,
54 / 409 / 37 after (refuted / inconclusive / survived; forced-only survivals
10 then 9). More true pairs are judged, and almost no more false pairs are
caught: the wrong pairs this lets run mostly agree on what little is
compared, and four more survive. The likelihood ratio of a survival stays
near eight. A coverage gain on true claims is not a gain in refutation
power, and that is exactly what the census's seventh observation - how much
a survival compared - exists to take apart.

**What it says about how this project explains its numbers.** The fault
bucket had an explanation that fitted, and it was repeated in the handover
without having been measured. It was right for about a third of the bucket.
F26 made the same mistake from the other side - the constructor shape was
assumed common and measured at 3% - and the remedy is the same: an
explanation that fits is a hypothesis until it is counted.

**And the stub is still a second door.** F25 made a stub's *writes* go
through the same rule as the function's own. Its *touches* did not, and this
was one of four places the two paths disagree. The other three are still
open: a stub dereferencing near null maps the guard page instead of
faulting, and leaves it mapped for the rest of the run; a stub reaching a
page outside the buffers and the arena does not set `read_invented`, so F32
does not see it; and a stub's first touches are not counted against the walk
budget. They push in different directions - the first and third towards
fewer completions, the second towards fewer refutations - so they are one
change, F39, measured on its own: every touch, the function's or a stub's,
through one rule.

## Hardening - the core is built, these make it stronger

The core runs (abi, harness, compare) and refutes true fixture pairs zero
times. Everything below is a way it can be made harder to fool or wider in
reach, gathered here so none is lost in the code. The verifier is the part
the project stands on, so each of these is planned work before v1, not a
someday-list, and each enters the same way everything has: measured, tested,
and never by loosening the rule that a true claim is never refuted.

**Not yet built, in the order they buy the most:**

1. **Stub families (tier 2).** Without stubs every function that calls an
   import is inconclusive, so this is the largest coverage gain and comes
   first. Block memory, then C string, character, number/text, allocation,
   each on one tested core (Coverage, above).
2. **Input generation.** The generator that builds input vectors from the
   reference signature - boundary values, seeded random, distinct pointer
   regions - so a claim is tested without hand-written inputs. Its seed is
   recorded with the verdict.
3. **Static tier (tier 1 evidence).** The cheap checks before emulation -
   reachable imports, string and constant references - each admitted only
   at zero false refutations on train, to reorder candidates and refute the
   obvious mismatches without running anything.

**Sharpening what is built, so a true claim is even harder to refute:**

4. **Byte-precise write tracking.** Writes are compared a page (4 KB) at a
   time; a function that writes four bytes of a page leaves the rest as
   whatever was there, and the two fill seeds catch most but not all of that
   noise. Recording exactly which byte offsets a version wrote, and
   comparing only those, removes a class of page-level false differences.
   This directly lowers the false-refutation risk, so it comes right after
   the first stubs.
5. **Stack-frame alignment.** The frame is 16-aligned; Win64 wants RSP to be
   8 mod 16 at entry (the return address just pushed). MinGW emits movups so
   nothing faults today (measured), but MSVC (v2) and hand-tuned SSE would,
   and getting entry alignment exactly right removes that as a variable.
6. **Global state - the full fix for risk 9 (F15).** Today a write into the
   binary's own image is excluded, so a global write never refutes but never
   confirms either. The way to compare it: read each binary's data sections
   before the run, and record an image write as (section, offset-from-section
   -start) rather than an absolute address, which is stable across the twins
   when the layout matches. Two versions are then compared on those offsets -
   *but only when the layout lines up*: -O0 and -O3 may drop unused data,
   reorder it or merge strings, and a mismatched offset must make the input
   inconclusive, never refuted. This widens coverage (global-writing
   functions become judgeable) while keeping the one rule, and it is how the
   blind spot F15 narrowed is properly closed.
7. **Float tolerance.** Float returns are compared bit-for-bit. Neither build
   uses -ffast-math, so this should hold; calibration on true pairs will show
   whether it does, and if not a per-type tolerance goes in compare, recorded
   with the verdict.
8. **Sub-page and heap-content comparison.** Once an allocator stub exists,
   memory it handed out is part of the effect and is compared by content like
   argument memory; the write tracker (4) makes that precise.

**Widening reach further (toward v1's ceiling, and v2):**

9. **By-value structs and long double.** Declined today. The Win64 rules for
   aggregates (in a register up to 8 bytes, by reference above, returned
   through a hidden pointer above 8) are mechanical; adding them turns a band
   of inconclusive claims into judged ones.
10. **A deeper input budget where it pays.** More input vectors refute more
    false pairs but cost time; V0 and the false-pair measurement say where a
    larger budget is worth it, per domain.
11. **MSVC and ELF (v2).** The calling convention here is one place, the
    debug format another; both change for MSVC, and the design keeps each in
    one module so the harness body is reused.

Each item is taken up in this order unless a measurement - V0's histogram, a
calibration false-refutation, the false-pair power - says a later one matters
more. The order serves one end: the widest reach that never once refutes a
function that is what it is claimed to be.

## Open questions

- **How much will be inconclusive?** Functions that do I/O, take
  callbacks, walk global lists or allocate through their own allocators
  may be most of a real library. V0 gives the first number. If coverage is
  low, that is reported, not hidden; widening it is the work after v1.
- **How many inputs is enough?** Each refutation needs one; each survival
  is only as strong as the inputs behind it. The count is fixed per claim
  and recorded; whether more would refute more is itself measured on false
  pairs.
- **Behaviourally equal but different functions.** The corpus dedup removed
  identical *code* across packages; it did not remove equal *behaviour*.
  How often a wrong candidate is genuinely indistinguishable is worth
  knowing, and the true/false pair measurement shows it as false pairs
  that survive.
- **Other compilers.** Nothing above assumes MinGW beyond the calling
  convention and the debug format; MSVC (v2) changes both, and the design
  keeps them in one place each.
