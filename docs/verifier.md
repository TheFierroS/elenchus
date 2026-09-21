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
   version, with different fill patterns** for all memory the verifier did
   not set on purpose. An observable that changes with the fill pattern
   *within one version* depends on garbage and is excluded from the
   comparison for that input.
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
  address, so the same address holds the same bytes for both versions and
  a pointer argument's buffer exists without its size being known.
- **Calls within the binary** are followed. **Calls to imports** are
  served by stubs: memory and string functions (`memcpy`, `memset`,
  `memcmp`, `strlen`, ...) and allocation (`malloc`, `calloc`, `realloc`,
  `free`, from a deterministic heap) are implemented; any import without a
  stub makes that input **inconclusive**. Which stubs exist is a list in
  the code, and each is tested against its C semantics on its own.
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

## Order of work

1. **V0 - feasibility, before any design is built on it.** Take a sample
   of true pairs from train. Load each binary, set up the call from the
   reference signature with no stubs at all, and run both versions. Count:
   how many run to completion, how many hit an instruction the engine does
   not support, how many call an import, how many touch memory outside
   their arguments. And the first false-refutation count: among pairs that
   ran, how many disagreed, and why. Protocol written first.
2. Resolved signatures in `dwarf.py`, and a migration for the corpus.
3. The harness: loader, calling convention, memory on first touch, fill
   patterns, instruction budget, masked return.
4. Stubs, one import at a time, each tested on its own.
5. Calibration on train true pairs until zero refutations - and every
   cause found written down, the way F1-F14 were.
6. Tier 1 static rules, each admitted only at zero false refutations.
7. Claims and verdicts as events; `elenchus verify` and `--replay`.
8. Measurement on val; then the end-to-end protocol; then test, once.

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
