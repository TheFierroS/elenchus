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
