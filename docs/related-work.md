# Related work: who else has stood where we are standing

Written after the verifier reached 54% coverage with zero false refutations and
three rounds of work (constructor chains, by-value structs, more stub families)
each bought under a point. The measurements said the remaining 38% is one
cause - a function handed data that means nothing to it - and that producing
meaningful data needs the library's own contract, which is not in the binary.

Before inventing a fourth thing, this is a survey of the people who have
already stood here, what they did, where they beat us, and what we should take.
Everything below is paraphrased from the papers; the citations are so the
claims can be checked.

---

## 1. BLEX - blanket execution (USENIX Security 2014)

Egele, Woo, Chapman, Brumley. *Blanket Execution: Dynamic Similarity Testing
for Program Binaries and Components.*

**This is our design, from 2014.** They run each function under a fixed
randomised environment, let it touch whatever memory it likes, map a dummy
page wherever it faults, cap instructions and wall-clock, and compare the side
effects. Their seven features are heap reads, heap writes, stack reads, stack
writes, library calls through the PLT, system calls, and the return in RAX.
Environments are reproducible by construction, exactly as our seeds are.

What we independently rediscovered and they had already: the dummy page for
unmapped memory, the instruction cap (theirs 10,000), the wall-clock timeout
(theirs 3 s), and the finding that more randomised environments stop helping
after about three - we use two seeds and two input fills, which lands in the
same place.

**Their answer to the coverage problem is the important part.** Rather than
trying to make the function's input valid, they give up on natural execution:
a function is run repeatedly, each run starting at the first instruction not
yet executed, until every instruction has been covered. They call this
blanket execution, and they are explicit that it works by sacrificing the
ordinary meaning of calling a function.

They are also explicit about what this costs them, in a section titled to that
effect: what they measure is *similarity, not equivalence*. Two functions get
a weighted Jaccard score over the seven features, with weights fit by an SVM.
They say plainly that declaring two functions inequivalent from one differing
environment would require capturing execution behaviour precisely, which is
unsolved for binaries.

Results: on coreutils across optimisation levels, the right match ranks first
64% of the time and in the top ten 77%. Against BinDiff they lose slightly on
-O2 vs -O3 (syntactically similar) and win by up to 3.5x on -O0 vs -O3.

## 2. UC-KLEE - under-constrained symbolic execution (USENIX Security 2015)

Ramos and Engler, best paper. *Under-Constrained Symbolic Execution:
Correctness Checking for Real Code.*

Instead of feeding a function concrete garbage, they start it on **symbolic**
memory and allocate lazily: a pointer begins unbound, and the first time it is
dereferenced a fresh object is allocated and bound to it, whose own contents
are again unbound, recursively. The solver then finds the values that make a
path feasible - so a check like `ctx->magic != MAGIC` is satisfied rather than
failed. A depth bound stops the structure growing forever.

They use it for exactly our shape of problem: checking that a patched version
of a function is equivalent to the old one, which they run as crash
equivalence, and they report verifying (with caveats) 115 patches while
finding 12 bugs in others.

**They reached our F18 and F23 independently and state the rule cleanly**: only
scalars are compared, because a difference in pointer *addresses* does not
matter as long as the objects those pointers refer to hold the same values.
They implement it by tracking pointer referents rather than by trying to
interpret symbolic expressions. Our image-range masking is a coarser version
of the same idea.

## 3. PEM - probabilistic execution model (ESEC/FSE 2023)

Xu, Xuan, Feng, Cheng, Ye, Shi, Tao, Yu, Zhang, Zhang, Purdue.

The current state of the art among execution-based methods, and the most
directly useful to us. Three ideas.

**Predicate flipping.** Run the function on a garbage seed; the run dies in an
input check, as ours do. Then take that path and *flip* a branch, forcing the
function past the check and into its real work. A path reached by flipping k
predicates is k-edge-off. This is their answer to the problem that sank our
constructor chain: they never produce a valid input, they simply refuse to let
the function's own guard stop them.

**Which predicate to flip.** Not any - the ones whose *dynamic selectivity*,
the runtime distance |x - y| at the comparison, is extreme. Their argument is
that optimisations remove, duplicate and move predicates but do not invent new
ones, so the ranking of the most and least selective predicates is the part
most likely to survive optimisation; they prove a bound and measure the
ranking holding over 80% of the time. They sample the ranked list from a
Beta(0.03, 0.03), which is U-shaped, so both extremes are favoured and the
middle is rare. This is what makes the two versions flip *corresponding*
predicates - the property we would need.

**Probabilistic memory model.** Invalid addresses are not mapped; they are
folded into a fixed random array of size 64k by `PM[addr mod gamma]`. They
state the two properties such a model must have, and these are worth adopting
as words: it must be *equivalence-preserving* (two equivalent paths in two
equivalent programs must see the same sequence of invalid addresses and
values) and *difference-revealing* (two different paths must see different
ones). A model that returns a constant satisfies the first and fails the
second; their ablation measures it - no model 76%, constant 83%, PMM 86%.

They also feed the *same value to every parameter*, which makes parameter
order and count irrelevant - a neat trick for stripped binaries where the
signature is unknown.

Results: 96% precision at rank 1 on coreutils across compilers and
optimisation levels, against 77% for in-memory fuzzing and 69% for BLEX. 90%
of functions reach near-full code coverage. Three functions per second, and
AArch64 support added in half a person-day.

## 4. The function-naming line, and the hole in it

Debin (CCS 2018), Nero (2020), NFRE (2021), SymLM (CCS 2022), XFL, SymGen
(NDSS 2025). SymLM reports precision 0.50 and recall 0.71 on its threshold;
SymGen, which domain-adapts an LLM, reports large gains over that line.

**None of them verify anything at deployment.** They report precision and
recall against held-out ground truth, which tells an analyst the average error
rate of the tool and nothing at all about the name in front of them. The
decompilation line (LLM4Decompile and successors) does better - it scores
*re-executability*, running the reconstructed code against the original's
tests - but that requires source and a test suite, which the analyst facing a
stripped binary does not have.

That hole is where this project sits.

---

## 5. A second survey, October: who else identifies functions by running them

Written after F37, when the next piece of work was a census of what else the
verifier could observe. The first survey (sections 1-4) asked how others
reached coverage; this one asks who else has used *behaviour* to identify a
function, what they observed, and where their tools broke. Numbers in the
sections below this one are from September; at the time of writing V0 holds
zero false refutations over 3,000 pairs and judges 45.7% of them on feasible
paths, 51.1% with forcing.

### 5.1 Software Ethology (Tinbergen, IOVFI) - our closest relative

McKee, Burow, Payer. *Software Ethology: An Accurate, Resilient, and
Cross-Architecture Binary Analysis Framework.* Open source as HexHive/IOVFI.

The same thesis as ours, from the identification side: a function is
characterised by the program-state changes it makes from a given input state,
and those changes survive the compiler, the optimisation level, obfuscation and
even the architecture. They generate input/output vectors (IOVecs) for each
known function by coverage-guided fuzzing, arrange them in a decision tree, and
identify an unknown function by asking it to reproduce them. F-score 0.779
across eight gcc and clang environments on coreutils, 25-53% above BLEX and
IMF-sim when the environments differ, and AArch64 identified from x64 IOVecs at
0.811.

**Two of their reported failure modes are two of our findings.** `c_isprint`
returns a byte; at -O0 an extra movzx writes the upper bits of RAX, their
strict return comparison saw a difference, and the function was mislabelled -
our F21 and risk 1, which mask the return to its type's width.
`set_program_name` is void, but RAX happened to hold a readable pointer on
exit, so it matched a class of string-returning functions - our F34, which
refuses to treat an uncompared return as an observation. Independent
confirmation that these rules are not fussiness: without them a published tool
mislabels.

**What they have that we do not: pointer derivation.** With no types, they
learn empirically which input bytes are pointers. On a segfault they taint the
faulting address backwards to the input location it was loaded from, allocate
a fresh object there, and **restart the function from its entry** with the
repaired input. They reject the simpler fix - substituting a valid address at
the faulting instruction - for the reason F32 exists: the output would no
longer follow from the input state. A repaired input is a legitimate input, a
structure whose pointer fields point at memory, handed identically to both
versions; a difference on it would be first-tier evidence under our rules, not
a forced path.

It is aimed at our largest bucket, faults at 18.2%, and we have two things they
lacked. The reference function's DWARF states which fields of an argument's
structure are pointers, so the repair can be read rather than inferred. And our
fill is invertible: an input-buffer word is `offset * GOLDEN + variant` modulo
2^64 with GOLDEN odd, so a wild address can be tested exactly for having been
loaded from our fill, and from which offset - the result of their taint
analysis without a taint analysis. How much of the fault bucket that explains
is the eighth item of the observation census; whether to build the repair
waits on that number.

**What they say they are not: sound.** A function that accepts every IOVec on
its path through the tree is labelled, and they state this can be wrong. That
is the direction we do not claim. Their result is an identification; ours is a
refutation, or an honest silence.

### 5.2 BLEX, feature by feature

Section 1 lists BLEX's seven features. Their own sensitivity analysis says what
each is worth alone (-O2 against -O3), and an SVM fit what each is worth
together:

| feature | alone | weight | here |
|---|---|---|---|
| heap writes | 57% | 2.50 | the arena and the input buffers |
| stack reads | 58% | 0.38 | excluded (F30, F35) |
| stack writes | 53% | 0.41 | excluded (F30, F35) |
| heap reads | 40% | 0.38 | not observed |
| system calls | 39% | 0.88 | none - library code |
| library calls | 17% | 0.11 | the import census |
| return in RAX | 13% | 0.32 | compared |

Three readings. We hold BLEX's strongest feature and its second weakest. The
feature the roadmap wanted next, library calls, is its weakest - though part of
that weakness is BLEX's alone, since static against dynamic linking changes it
and our two builds link alike. And much of BLEX's strength is stack contents,
which ARCTURUS calls unstable under code transformation and which F35 excluded
for that reason. BLEX buys similarity with a signal we refuse because it can
refute a true claim. That is a design choice, and these numbers are its price.

### 5.3 Others who ran functions to name them

**angr's Identifier** (Shellphish, for the Cyber Grand Challenge) names
statically linked libc functions in stripped binaries by running hand-written
test cases - memcpy, strcmp and printf each have their own. It is the oldest
working precedent for identification by execution, and its documentation names
its limit: the tests are written by hand, and symbolic execution could generate
them. We need neither, because the reference build is the oracle: every
candidate is tested against the function it is claimed to be, compiled from the
same source.

**BinMatch** (Hu et al., ICSME 2018, TSE 2019) runs the reference with real
test cases, records the values it reads, and replays them into the target in
the order they were read - an answer to meaningless input by borrowing the
reference's meaningful state. **Not taken:** -O3 reorders, merges and drops
loads, so a value migrated by position arrives at the wrong read, and the
difference that follows is ours. Fit for similarity, unsafe for refutation.

**IMF-sim** (Wang and Wu, ASE 2017) fuzzes each function in memory and
introduced the backward taint Tinbergen built on; its similarity is a trained
model over trace scores.

**ARCTURUS** (Zhou, Hu, Xu, Zhang, TOSEM 2024) reaches every block not by
flipping branches but by constructing executions that reach a chosen block, and
proves their results match a real execution - 87.8% precision at full block
coverage, 0.15 s a function. A feasible path is one a difference on can refute,
so this is the route by which the second tier, which today can only
corroborate, could one day refute. It needs lifting to LLVM IR. Recorded for
v2.

### 5.4 What the analyst already has: Ghidra's FID and BSim

Function ID matches hashed bodies exactly. BSim (NSA, Ghidra 11) matches
decompiler feature vectors and reports two numbers: similarity, and a
*significance* that grows with how many and how rare the shared features are. A
function that returns a constant matches hundreds of others at similarity 1.0
and almost no significance.

Two consequences. **BSim is the baseline a reader will ask for**, and ours
(import Jaccard, BM25, structural) are not it; it ships with the Ghidra this
project already drives. And **significance is what our survivals lack.** A
survival is one bit; F34 only removed the case where nothing was observed. A
pair that agreed on one compared byte and a pair that agreed on eight hundred
are both "survived". How much was observed is already counted inside `_judge`;
the census reports it for V0's survivals against R0's, and if R0's false
survivals sit at small observation, a survival's strength can be graded without
touching refutation.

A product consequence follows. Nothing here needs the encoder to be the
proposer - the thesis already says the model only proposes. A verifier that
takes candidate pairs from BSim, FID, the encoder or an LLM and returns a
verdict is useful the day it ships, whatever the analyst already uses.

### 5.5 Standing in for Windows: two designs

**Speakeasy** (Mandiant) and **Qiling** model the Win32 API in Python handlers
and keep a sample on its "happy path", as our stub families do. **Dumpulator**
and Sogen load the real system DLLs, emulate them, and stub only the system
calls. For the MSVC work that is the choice between writing UCRT stub families
and loading `ucrtbase.dll`: fewer and better-defined stubs, at the cost of a
real heap allocator that the deterministic arena would have to replace at the
NT layer. Recorded as an input to that decision.

None of the four calls back into emulated code from inside an API handler as a
reusable operation, so `InitOnceExecuteOnce` has no design to borrow. Ours
exists already: a forced branch stops the run in a hook and resumes elsewhere
through `execute()` (`state["resume_at"]`), which is the machinery a callback
needs.

### 5.6 An idea from none of them: the compiler as an undefined-behaviour oracle

F32 and F33 exist because a function given invented data can execute undefined
behaviour, after which -O0 and -O3 owe each other nothing; Alive2 frames
compiler correctness as refinement rather than equivalence for this reason.
Both taints are conservative heuristics, and R0 measured their cost at four
points.

The reference build is ours. GCC and Clang can compile with
`-fsanitize=undefined -fsanitize-undefined-trap-on-error`, which turns each
detected undefined operation into a trap and needs no runtime library, so it
should work under MinGW. A fifth build of each reference, run on an input
before the comparison, says whether *that input triggers undefined behaviour in
the source* - a property of source and input, so it holds for the stripped
candidate too. It does not see uninitialised reads or most out-of-bounds
pointers, so it would add to the taints, never replace them. Not measured; a
fixture build says whether the toolchain accepts the flags.

### 5.7 The naming line is a different task

Name generators are scored on tokens: SymGen (NDSS 2025) and Hieronym (2026)
report F1 near 0.33 and 0.47, and they name any function, including ones never
seen. The encoder retrieves a known function from a reference set (MRR 0.735),
which is library identification - the space of FLIRT, FID, BSim and Lumina. An
MRR is not comparable to a token F1, and any write-up should say so before a
reader compares them.


## Where we are ahead

**A verdict, not a score.** BLEX and PEM both produce a similarity number and
rank candidates. BLEX says outright that equivalence is beyond what they can
claim. We produce verified / refuted / inconclusive, with the rule that a true
claim must never be refuted, and we measure that rule - currently zero false
refutations in 500 pairs, across three layers. For an analyst deciding whether
to trust a name, "this is wrong" and "I cannot tell" are different and both
more useful than 0.62.

**The inconclusive verdict is a feature, not an admission.** The literature
has no room for it: a ranking must rank. Ours is what lets the refutation rule
hold.

**Adversarial self-testing.** F15 through F26 are twelve real defects found by
measuring our own tool against a corpus - two memory leaks that killed the
run, three separate classes of pointer comparison, an inverted memory fill, a
stub whose writes were invisible. Each fix is mutation-checked. Research
prototypes are rarely held to this.

**Declining rather than guessing.** The stub trust rules, the signature
decline, the sanity bound, the two-fill rule - each is a place we choose to
say nothing rather than risk saying something false. PEM and BLEX let library
calls run or record their names; neither can afford our caution because
neither is making a claim that could be wrong.

**Win64 and PE.** Nearly all of this literature is Linux/ELF.

## Where they are ahead, and it is not close

**Coverage.** PEM gets near-full code coverage on 90% of functions; BLEX gets
full instruction coverage by construction. We judge 54% of pairs at all. The
metrics differ - they extract *behaviour* from nearly every function, we
*compare* half of them - but the gap is real and it is the gap we have spent
three rounds failing to close.

**And the reason is now clear.** Both of them reach that coverage by
abandoning natural execution: BLEX starts anywhere, PEM flips branches. We
insisted on running the function properly from its entry with plausible
arguments, and the 38% we cannot judge is precisely the price of that
insistence.

**Scale.** PEM: 35k functions, three per second, a new architecture in half a
person-day. BLEX: 195k functions.

**Bounded invalid memory.** PEM's `mod gamma` never runs out; our page limit
(F16) is a patch over the same problem and still costs us 2.6% of pairs.

---

## What to take, in order

**1. Flipped and blanket paths as *corroboration only*.** This is the design
that reconciles their coverage with our rule. A path reached by flipping a
predicate is infeasible: the function would never take it on any real input.
So a difference observed on such a path is **not evidence of inequivalence** -
it cannot refute, ever. But agreement observed on such a path *is* evidence
for equivalence, and it is evidence we can collect from the 38% we currently
decline. The two-tier design writes itself:

- *Refutation* stays where it is - feasible paths from the entry, real
  arguments, the rules we have. It is bounded by input quality and that is
  honest.
- *Corroboration* may use flipped paths, blanket starts, and anything else
  that gets the function to do work, because the only thing at stake is how
  much agreement we have seen.

A pair that today is "inconclusive, the function faulted" could become
"survived: agreed on 12 flipped paths covering 80% of its instructions" -
which is a weaker claim than a feasible-path survival and should be reported
as such, but it is worth far more than silence.

**2. PEM's predicate selection.** If we flip, we must flip *corresponding*
predicates in the two versions or the comparison is meaningless. Dynamic
selectivity with the U-shaped sampling is their answer and it is measured at
over 80%. Nothing we would invent is likely to beat it.

**3. The `mod gamma` memory model.** Small, self-contained, removes the page
limit and its 2.6%, and gives us their two properties to state explicitly in
our own terms - which we have been implying without naming.

**4. UC-KLEE's pointer referents.** Our image-range mask (F18, F23) handles
pointers into the binary. Theirs handles pointers into *anything* by comparing
what is pointed at rather than the address. That would close the remaining
pointer blindness properly.

**5. The same-value-for-all-arguments trick**, when we leave the corpus. In V0
we have DWARF signatures; on a real stripped binary the signature is itself a
claim, and PEM's trick makes the comparison indifferent to getting it wrong.

## What not to take

**Similarity scores.** The moment the output is a weighted Jaccard, the
refutation rule is gone and we are the fourth-best entry in a crowded field.

**Symbolic execution, for now.** UC-KLEE's lazy initialisation is the
principled answer to invalid contexts and would genuinely raise refutation
coverage - but it means a solver, a large dependency, and per-function costs
in seconds. It is the right thing to reach for if flipping turns out not to be
enough, not before.

---

## Sources

- Egele, Woo, Chapman, Brumley. Blanket Execution. USENIX Security 2014.
  https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/egele
- Ramos, Engler. Under-Constrained Symbolic Execution. USENIX Security 2015.
  https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/ramos
- Xu et al. PEM: Representing Binary Program Semantics via a Probabilistic
  Execution Model. ESEC/FSE 2023. https://arxiv.org/abs/2308.15449
- Wang, Wu. In-Memory Fuzzing for Binary Code Similarity Analysis. ASE 2017.
- Jin et al. SymLM. ACM CCS 2022. https://dl.acm.org/doi/10.1145/3548606.3560612
- Beyond Classification: Inferring Function Names via Domain Adapted LLMs.
  NDSS 2025.
- Tan et al. LLM4Decompile. EMNLP 2024. https://arxiv.org/abs/2403.05286
- Marcelli et al. How Machine Learning Is Solving the Binary Function
  Similarity Problem. USENIX Security 2022.
- McKee, Burow, Payer. Software Ethology: An Accurate, Resilient, and
  Cross-Architecture Binary Analysis Framework. https://arxiv.org/abs/1906.02928
  Code: https://github.com/HexHive/IOVFI
- angr Identifier. https://docs.angr.io/en/latest/analyses/identifier.html
- Hu et al. BinMatch. ICSME 2018, https://arxiv.org/abs/1808.06216; extended
  in TSE 2019, https://arxiv.org/abs/1907.01374
- Zhou, Hu, Xu, Zhang. ARCTURUS: Full Coverage Binary Similarity Analysis with
  Reachability-guided Emulation. TOSEM 33(4), 2024.
  https://dl.acm.org/doi/10.1145/3640337
- Ghidra BSim. https://ghidra.re/ghidra_docs/GhidraClass/BSim/BSimTutorial_Basic_Queries.html
- Speakeasy https://github.com/mandiant/speakeasy; Qiling
  https://github.com/qilingframework/qiling; Dumpulator
  https://github.com/mrexodia/dumpulator
- Patrick et al. XFL. IEEE S&P 2023. https://arxiv.org/abs/2107.13404
- Hieronym. https://arxiv.org/html/2609.12457
- MaskRay. All about UndefinedBehaviorSanitizer.
  https://maskray.me/blog/2023-01-29-all-about-undefined-behavior-sanitizer
