# 🔬 I made my world small enough to compute everything exactly. It caught none of my eight mistakes 🇫🇷

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21726216.svg)](https://doi.org/10.5281/zenodo.21726216)

---

## 👤 What this is

My [first article](https://huggingface.co/blog/RDTvlokip/teaching-a-network-to-write-with-reward-only) trained networks to produce language from a scalar reward alone. My [second](https://huggingface.co/blog/RDTvlokip/i-published-my-rl-experiments) was about a reader running the code and four of my published numbers not surviving.

This one is about the third experiment, which had never been run. It is a referential game with 27 referents and 27 messages — small enough that the search space, the optimum, the null distribution and the gradient are all computable **exactly**, by enumeration or in closed form. Not estimated. Computed.

I ran the entire seven-question investigation programme in one day. Here is what it found, and here is the part I did not expect:

> Eight of my own hypotheses died that day. **Not one of them was an arithmetic error.** Making the world small enough to compute everything removed one class of mistake entirely and left the class that was actually killing me completely untouched.

The scientific result and the methodological result are both in here, and I am not sure which one is worth more.

---

## 📐 The world, and why it is this small

| element | value |
|---|---|
| attributes | 3 |
| values per attribute | 3 |
| referents | **27** |
| message length | 3 tokens |
| vocabulary | 3 |
| possible messages | **27** |
| sender | a 27 × 27 stochastic matrix |
| receiver | a 27 × 27 stochastic matrix |
| reward | 1 if the receiver reconstructs the referent, 0 otherwise, shared |
| chance | 1/27 = 3.7 % |

A **perfect code** is a bijection between referents and messages. There are 27! ≈ 1.09 × 10²⁸ of them. A **compositional** code is one where token *j* encodes attribute σ(*j*): there are 3! × (3!)³ = **1 296** of those.

Two facts follow before any training:

**All 27! perfect codes earn exactly 1.** They are tied. The reward cannot separate them.

**So compositionality is 1 296 / 27! ≈ 1.19 × 10⁻²⁵ of the optimum set.** It is not that RL fails to find compositional codes. Nothing in the objective asks for them.

That was in the design document from the start. The whole programme exists to ask a harder question: *what, if anything, does select them?*

---

## 🧨 The comment that started it, and the contradiction he did not point at

`dipankarsarkar` — who took apart four of my numbers in article 2 — read the design document for this unrun experiment and went straight at its one derived threshold.

The document said: a concentration above ~0.35 would be out of reach of a uniform draw, and that this threshold was not arbitrary but **derived** from the null distribution. He reproduced my null at 20 000 draws exactly, then reran it at 10 000 000:

| | 20 000 | 10 000 000 |
|---|---|---|
| mean | 0.1273 | 0.1269 |
| sd | 0.0332 | 0.0330 |
| q99.9 | 0.2537 | 0.2525 |
| **max** | **0.3305** | **0.3979** |

> "500x the draws moves q99.9 by 0.0012 and the max by 0.067. The body of the null is settled at 20,000. The max is the only row still moving, and the ~0.35 line is the one thing built on it."

My own independent 10M draw reproduced him: max 0.39788, and **14 draws above 0.35, exactly as he got**.

He was right, and there is a stronger version of his point that costs no simulation at all.

**The 1 296 compositional codes *are* bijections.** They are inside the null, with probability 1.19 × 10⁻²⁵, and they score exactly 1. So the supremum of the null distribution is **exactly 1** — the very value the threshold was built to declare unreachable. A sample maximum is not estimating a threshold here. It is estimating 1, infinitely slowly, and no sample size fixes that.

Twelve independent blocks of 10 000 000 draws each:

| row | mean over 12 blocks | spread across blocks |
|---|---|---|
| mean | 0.1269 | 0.0000 |
| q99.9 | 0.2527 | **0.0006** |
| **maximum** | 0.3950 | **0.0509** |

The maximum wanders by 1.54 standard deviations of the null itself. A threshold placed on that row inherits that: it measures how long I was willing to sample.

**And the worse problem was three paragraphs upstream, in my own document.** §5 says, in writing, *"we deliberately abandon the pass/fail criterion"* and records a commitment about a **distribution**. §6.1, three paragraphs later, reintroduces a pass/fail threshold and congratulates itself for it being derived rather than arbitrary.

I wrote both. It took an outside reader going to the unrun part of the project for me to read them next to each other.

---

## 🔒 The certificate I had been importing does not survive two agents

Everything above rested on a result I had carried over from experiment 2: *under maximum entropy, optima with equal reward are equiprobable.* That is what turns "1 296 out of 27!" into a statement about what training produces.

It was established for a **single** agent maximising `E[R] + β·H`. Here two agents share a reward. The design document flagged this as *"the first place in the project where a published result could collapse on a technical point I have not checked"*, and put it first in the execution order.

It collapsed.

**The certificate requires that the tied objects be the support of the distribution whose entropy is in the objective.** In experiment 2 they were: the tied objects were sequences, and the entropy was the entropy of the distribution over sequences, so spreading mass across the tied optima was free. Here the tied objects are **codes**, and no distribution over codes appears in the objective at all — the entropy acts on the rows of the sender and the receiver.

And the reward is a coordination reward, so spreading destroys it. Mixing K codes, both agents mixed the same way:

| K | 1 | 2 | 3 | 5 | 10 | 27 |
|---|---|---|---|---|---|---|
| E[R] | 1.0000 | 0.5000 | 0.3416 | 0.2237 | 0.1511 | 0.0713 |

There is no optimal law that "charges the 27! optima equally". The sentence has no referent in this setting.

---

## 🎯 What replaces it, and the corollary that should worry practitioners

The number survives, by an argument that is stronger and narrower.

Relabelling the 27 messages by a permutation π acts on codes by `c → π ∘ c`, and that action is **transitive** on the 27! bijections: to go from `c₁` to `c₂`, take `π = c₂ ∘ c₁⁻¹`. If the parametrization and the initialization are equivariant under that group, then the 27! codes are **exactly** equiprobable. No Gibbs assumption, no entropy argument.

Verified numerically — the ascent is exact, therefore deterministic — **8 runs out of 8** return exactly `π ∘ c` after the messages are relabelled.

> At the optimum, for a tabular parametrization, the probability that the code is compositional is exactly 1 296/27! ≈ 1.19 × 10⁻²⁵. That is now a **theorem about the parametrization**, valid for any equivariant algorithm, not a property of the optimiser.

This immediately reorganises the experimental programme. **Running the emergence measurement on a tabular sender cannot discover anything** — the outcome is a theorem. It keeps value as a bug detector: a departure from chance would prove the implementation broke a symmetry somewhere.

**And here is the corollary I did not see coming, which cost me a wasted experiment.**

I built what I thought was the contrasting parametrization: an autoregressive sender, `p(m₁)·p(m₂|m₁)·p(m₃|m₁,m₂)`, tabular at each stage so it has exactly the same expressivity as the 27 × 27 matrix. Same objective, same optimiser, same receiver. Only the parameter map differs.

The gap between the compositional code and a random one, under that parametrization, was **3.3 × 10⁻¹⁶**. Machine precision. It contrasted nothing.

The reason took me a while and it generalises well beyond this toy. **Relabelling the referents,** `c → c ∘ ρ⁻¹`, is *also* transitive on the 27! bijections. My autoregressive sender indexed its parameters **by referent**, with no sharing between referents — so it was equivariant on the referent side, and equivariance on **either side alone** is enough to tie every code.

> **A free per-referent embedding table cancels, in advance, anything the message structure could contribute.** No training trick, no auxiliary loss, no constraint on the message side can recover it, because the symmetry argument does not care what you do next.

Most implementations would do this without noticing. It is checkable by reading the architecture, without running anything.

The parametrization that actually breaks the tie is one where the referent enters **through its attributes** with **shared weights**: 81 + 9 parameters instead of 729 free ones.

---

## 🧪 Representable, reachable, stable — and the ceiling I nearly mismeasured

With a hand-built compositional code available, three questions that share a name and demand opposite remedies.

**Representable.** Supervised fitting toward an imposed code:

| parametrization | compositional | random | gap |
|---|---|---|---|
| tabular | 0.99947 in **2 198** steps | 0.99947 in **2 198** steps | −1.1 × 10⁻⁷ |
| autoregressive-per-referent | 0.99939 in 2 365 steps | 0.99939 in 2 364–2 365 steps | −1.1 × 10⁻⁷ |
| attribute-structured | 0.99939 in 2 367 steps | **0.11573, never reached** | **+0.884** |

The first two rows are equivariance made visible **down to the step count**, identical to the unit. The third says representability itself is not the same for all codes once the parametrization sees the attributes. Pushed to 20 000 steps at a tenfold learning rate, the structured sender reaches 1.00000 on the compositional code in 704 steps and plateaus at 0.09–0.24 on arbitrary bijections. That is a capacity limit, not an optimisation failure.

**Reachable, and this one surprised me.** Exact gradient ascent, no sampling anywhere, from a near-uniform start, lands on:

> 23/27, 24/27, 25/27, 26/27, 27/27.

**One start in forty reaches a perfect bijection.** The rest settle on codes where one to four referents collide. Started *on* a perfect code, it stays at E[R] = 1.0000 forever. Reachable and stable are different answers, and there is no sampling noise to blame — there is no sampling.

**And that opened a trap I closed the same day.** My fast vectorised path for the null assumed both margins of the joint distribution are uniform, which is true only for a bijection. On a non-bijective code it silently returned 0.110573 instead of 0.108071 — an error of 0.0025, about a fifth of the effect the whole experiment is designed to resolve. No exception, no warning. Guarded now.

**Then the ceiling.** When I first measured "the cost of coordination", I compared the free pair (E[R] = 0.911) against an agent frozen on a **bijection** (0.9999) and wrote down 0.049. That was wrong twice over.

A code with *k* collisions is capped **arithmetically** at (27 − *k*)/27: two referents sent to the same message are indistinguishable, whatever the receiver does. So I was comparing two ceilings, not two learning problems. And my speed threshold — steps to reach 0.99 — is **unreachable from the first collision onward**, so it measured a capacity while claiming to measure a speed.

Fixed, with the ceiling verified rather than assumed (frozen on a 2-collision code, the free agent reaches exactly 25/27), the result reverses:

| both free | E[R] | collisions | ceiling | **E[R] / ceiling** |
|---|---|---|---|---|
| tabular sender | 0.9111 | 2.40 | 0.9110 | **1.0000** |
| structured sender | 0.8777 | 3.30 | 0.8777 | **1.0000** |

> The free pair executes its chosen code **exactly as well** as an agent handed that same code ready-made. The deficit is not in the learning at all. It is entirely in *which code* the two settle on.

---

## ✍️ Who writes the code? Neither

Freeze one agent, let the other learn:

| condition | steps to 99 % of its own final value | final E[R] |
|---|---|---|
| sender frozen compositional, receiver free | 139 | 0.99992302 |
| sender frozen random, receiver free | 139 | 0.99992303 |
| receiver frozen compositional, sender free | 139 | 0.99992302 |
| receiver frozen random, sender free | 139 | 0.99992302 |

**139 steps in both directions, and the same final value to eight decimals.** The problem is exactly symmetric — the bilinearity of the objective made visible. Neither agent writes the code; the coordination does.

And freezing on the compositional code versus an arbitrary bijection is the same problem, to 6 × 10⁻⁹. Equivariance again, in a third disguise.

---

## ⏱️ The first-step gradient sees nothing. The preference appears at step 30

The design document had derived, before any experiment, that at initialization both policies are near-uniform, so both gradients are near-uniform, so **no direction is preferred**. Measured: coefficient of variation 1.0 × 10⁻² and 9.8 × 10⁻³. It holds. Sharp contrast with experiment 2, where a lexicon imbalance imposed a direction from step 1.

I then added a prediction of my own, before measuring: the structured parametrization, which ends at z = +9.9, must prefer the compositional code from the first step.

**Measured: z = −0.08 ± 0.24.** Nothing at all.

So I asked when it appears. Same measurement at several training depths, restarting from the same initialization each time — z of the compositional code against 100 control bijections:

| step | 0 | 10 | **30** | 100 | 300 | 1 000 | 3 000 |
|---|---|---|---|---|---|---|---|
| tabular | +0.07 | +0.07 | −0.30 | +0.30 | +0.19 | +0.16 | +0.19 |
| structured | −1.18 | −0.29 | **+4.36** | +4.25 | +3.91 | +5.81 | +5.85 |

> The tabular parametrization never prefers the compositional code, at any depth. The structured one does not either at first, then does so **abruptly between step 10 and step 30**, and never leaves.

The mechanism is nameable, which is what the question asked for: near the uniform point the parametrization's constraint **does not bite**, because every law is representable at low confidence. It appears as the law concentrates.

**Is the outcome written into the initialization?** With controls matched to the fiber profile of the reached code, the tabular parametrization ranks its eventual code **first out of 300**, z = +6.80. But only **8.7 %** of that code's referents are already the argmax of the initial weights, against 3.7 % by chance.

Two readings, and they have to be reconciled rather than picked between. The initialization biases strongly *in aggregate* without writing the code — about 2.3 referents out of 27. Saying "the outcome is decided at initialization" would overstate it. I added the no-null, no-cosine measurement precisely so I could not settle for the more flattering of the two numbers.

For the structured parametrization the initial imprint is **exactly nil**. Everything comes from the trajectory.

---

## 📊 The negative, stated with its bound

At 100 seeds, against a null drawn from the orbit matched to each run's collision profile:

| parametrization | n | concentration | z | KS *p* | past q99.9 |
|---|---|---|---|---|---|
| tabular | 100 | 0.1164 | **−0.01 ± 0.10** | 0.386 | 0 / 100 |
| autoregressive-per-referent | 100 | 0.1152 | **−0.05 ± 0.10** | 0.613 | 0 / 100 |
| attribute-structured | 20 | 0.3971 | **+9.01 ± 0.60** | 0.000 | 20 / 20 |

The theorem holds **in distribution**, not only in the mean. Under uniformity the percentile of each run within its own null must be uniform on [0, 1], and Kolmogorov–Smirnov gives *p* = 0.386 and 0.613 for the two equivariant parametrizations — compatible with uniform — against *p* ≈ 0 for the structured one. The standard deviation of the z values is **1.0**, which is the sharper check: the matched null has the right *shape*, not merely the right centre. A wrong reference class would show up as over- or under-dispersion, and it does not.

**"We saw nothing" means nothing without saying what we would have seen.** With a null standard deviation of 0.0312 at n = 100:

> Any residual selection by the dynamics, on an equivariant parametrization, is **smaller than 0.0087** in concentration units. That is a bound, not an absence.

And it justifies rerunning: at 20 seeds the bound was 0.027, so the scenario my reader had described as far likelier than an isolated outlier — a weak pressure lifting every run by 0.02 — would have gone unnoticed.

---

## 🌊 The constraint curve, and the one-line proof that killed my own rationale

The reward is provably indifferent to compositionality, so structure can only come from an **external** constraint. My design document listed four, with channel noise as the most promising, justified like this:

> "a compositional code loses only one attribute when a token is corrupted; a holistic code loses everything"

**That is false for this reward, and it takes one line to show without training anything.** For a deterministic sender on code `c` with the optimal decoder,

```
E[R]* = (1/27) · Σ_m' max_r C[c(r), m']
```

and since `c` is a bijection onto **all** 27 messages, `max_r C[c(r), m'] = max_m C[m, m']` — independent of `c`. Measured, compositional minus 200 random bijections:

| ε | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| gap | 0 | −1.1e−16 | −1.1e−16 | 0 | +1.1e−16 | −5.6e−17 |

Losing "only one attribute" earns nothing when the credit is all-or-nothing on the exact referent. The argument never mentions 27, so it holds at any size and for any channel.

**But the channel does break the symmetry**, and that is what made the experiment worth running. At ε = 0.2, the gap between the channel matrix and its permuted version is **0.00e+00** over the 1 296 structure-preserving permutations and at least **0.050** over 200 arbitrary ones. So:

> The equal-optima certificate still says nothing separates the bijections in reward, exactly, at every ε. And the equivariance theorem no longer applies. **Any selection observed would operate entirely outside the reward** — the cleanest setup this bench can produce.

**Nothing happens.**

| ε | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| E[R] | 0.9333 | 0.8344 | 0.7598 | 0.6104 | 0.4840 | 0.0370 |
| z | −0.44 | −0.05 | −0.28 | +0.38 | +0.15 | −0.02 |

No run out of 90 past the 99.9th percentile. The bound at 15 seeds is |z| < 0.72, against about +9 for the structured parametrization — a factor of more than twelve. At ε = 0.5 the channel destroys the code before structuring it: pure babble, 9.4 collisions.

> **Breaking the symmetry is necessary, and not sufficient.** That refutes the unifying hypothesis I had drawn from the equivariance theorem that same morning.

**Population turnover does nothing either, and that one was predicted.** Replacing a tabular receiver with a fresh one is an exchangeable operation, so equivariance survives and the theorem still applies. z from −0.34 to +0.33 across four renewal periods. If iterated learning produces compositionality, on this bench it cannot come from turnover alone — it needs the relearner's inductive bias. I have not surveyed the literature, so that is a statement about my setup, not about anyone's published work.

**What would have been needed, computed exactly.** A per-attribute partial-credit reward *does* break the tie: +0.034 at ε = 0.05, **+0.108** at ε = 0.2, +0.153 at ε = 0.5. The mechanism in my table is real — it just works by changing the **reward**, not by adding a constraint outside it. And a per-attribute reward tells the agent that attributes count separately, which is exactly what a compositional code encodes.

### The conclusion, and it is harsher than the curve I expected

Of everything tested that day, **one thing produced compositionality: the parametrization.** And it did so by making the alternatives unwritable, not by choosing between them. The one environmental constraint that would work does so by putting the preference into the reward.

> On this bench, compositionality was never **selected**. It was either impossible or specified.

---

## 🪦 Eight dead hypotheses in one day, and none of them arithmetic

This is the part I did not plan to write.

| what I believed | what killed it |
|---|---|
| "the 0.35 threshold is derived, so it's solid" | a sample maximum has no estimand here; the null's supremum is the value being excluded |
| "the double-count inflation grows with concentration" | true on the null, false on a structured population at the same level |
| "the equal-optima certificate applies as-is to two agents" | tied objects are not the support of the entropy in the objective |
| "the gap from 1/27 comes from the perturbation size" | it was Adam, whose normalised steps do not slow where the gradient vanishes |
| "the top of the concentration scale is safe" | true only for bijections, and the reached codes are not bijections |
| "the structured parametrization prefers compositional from step 1" | z = −0.08 ± 0.24; the preference appears at step 30 |
| "channel noise favours compositional codes" | one line of algebra; the gap is 1.1 × 10⁻¹⁶ at every ε |
| "breaking the symmetry is enough" | it is necessary and not sufficient |

Plus five protocol defects caught before publishing: statistics read off the last seed instead of averaged, twice; nulls not matched to the observation's orbit, twice; a success threshold arithmetically unreachable in half the conditions.

**Every one of these is a specification error** — what to compare, against what, under which condition. Not a single arithmetic error, in a world where every quantity is computed exactly.

So the uncomfortable finding: shrinking a problem until everything is exact removes the estimation-error class and leaves the class that was actually doing the damage completely untouched. Worse, exactness prints fifteen decimals, which are easier to over-read. Several of these errors were over-readings of an exact number — *"gap 3.3 × 10⁻¹⁶, so my contrast works"* actually meant the contrast was empty.

**What did catch them**, in order of yield: an outside reader who reran the code before speaking, every time; a **second independent route to the same quantity** (the Hessian against the bisection, Hamming distance against mutual information, argmax agreement against a z-score); and predictions written down before measuring, which killed two of the eight.

Exactness helped only by making the second route cheap. That is real value. It is not the value I thought I was buying.

**And there is a case where a reviewer and I were both wrong**, which matters enough that the next section is about it. His argument that a concentration of 1.0 forces an injective argmax is correct; I verified it and adopted it in my reply. Its premise was bijectivity, and the reached codes are not bijections. So this scores a perfect 1.0000 under the published statistic:

```
m₁ = a₁,  m₂ = a₁,  m₃ = a₂          9 of 27 messages used, one attribute discarded

concentration, published form : 1.000000     ← the top of the scale
concentration, matched form   : 0.666667     ← two attributes read out of three
```

A code that throws away one attribute in three and uses a third of the message space is handed the value reserved for compositional codes.

---

## 🎣 Two questions I wrote down in July and could not answer

The notebook has a section of thirty uncomfortable questions, written under a rule: no question whose answer I already know, none that flatters the project, and for each one what would settle it. Two of them were about me rather than about the network. Running this experiment settled both, and neither answer is the one I was hoping for.

**Q29 — "If this project's value comes entirely from a toy small enough to enumerate, is the honest way to do ML research to shrink until exactness and then argue about extrapolation? And what becomes of everything done at scale — is it necessarily better-funded anecdote?"**

Answered, and by the negative. This day is the data: a fully enumerable world, everything computed exactly, and eight dead hypotheses of which **not one was arithmetic**. Shrinking to exactness removes the estimation-error class and leaves the specification-error class exactly where it was. It can even make things worse, because exact numbers carry fifteen decimals and invite over-reading.

And the second half sets up a false alternative. **The variable is controls, not size.** My own toy produced anecdote every time it lacked a null — the 0.35 threshold, and in article 2 a bias ratio read off three seeds. A scale experiment with an ablation, a baseline and seed variance is not anecdote; a toy without a null is. And scale does one thing no toy will ever do: establish that a phenomenon **exists** in the regime anyone cares about. A toy refutes universal claims, scale establishes existence. Two jobs, not two levels of honesty.

The premise is also wrong, which is the useful part: what survived this day is not the toy but the statements with no number in them. Smallness did not produce them. It made them cheap to find and complete to check. Hence *shrink to search, not to prove*.

**Q30 — "At what point does 'I am measuring my own specification' apply to me and not only to the agent? I wrote the environment, the reward, the diagnostics, the metrics and the interpretation. The diagnostic I built detects the agent's degenerate sublanguages. What diagnostic detects mine?"**

That was the one question in the list I had no lead on at all. I do now, and it is humbling: **another person.** Five rounds from one outside reader produced five corrections, including the bound that carried the previous article.

But it is not their humanity that does the work. His decisive property was **rerunning the code before speaking, every time**, and re-deriving from the published artifact rather than from my intentions. That puts the burden back on me: publish enough that re-derivation is possible, or lose the diagnostic.

And the day supplied the limit of that answer, in the counterexample above. Two of us, agreeing, both wrong — and what broke it was not a third reader but running a step that had never been run, which produced the non-bijective codes his premise excluded. **Two people can share a frame**, especially when the second is reading the first's documents.

> So: anything that does not inherit your specification. A reader who re-derives from the artifact, a second route to the same quantity, a prediction with a date on it — and when all of those agree and are wrong together, **the step of the programme you have not yet executed.**

Three of those four depend on someone else's goodwill. The fourth is the only one I control alone, and it is a concrete argument for running the programme in order instead of commenting on the parts I have not run.

---

## 💬 His comment, in his words

He went to the unrun part of the project, which is where mistakes are cheapest to fix and where nobody has any reason to look.

> "Thank you for writing this up. Test 3 is the interesting part, so I went there instead. The one number in your null is the one you called derived."

> "I ran grammaire3.py at seed 0 before saying anything. Your table reproduces exactly: mean 0.1273, sd 0.0332, min 0.0305, max 0.3305."

> "That does not hurt you and I would rather say so than dress it up. At 100 seeds the chance any run clears 0.35 under the null is 1.4e-4. The threshold is safe for the experiment you planned. The reason to drop it anyway is that §6.2 is already better without it."

> "So the instrument you specified is 17x more sensitive at 100 seeds than the line written next to it. And a weak outside pressure is far likelier to lift every run by 0.02 than to throw one outlier past 0.35."

And on the statistic, where he flagged his own uncertainty:

> "Separately, on the statistic itself, and this one I am less sure about. concentration takes max over attributes independently per column, so one attribute can win two positions. I checked how often that happens: 74.6% of 500,000 uniform draws."

> "Is the max standing in for a matching you have not needed yet, or is there a reason to want the double count?"

Standing in for a matching. There was no reason. The unconstrained form does match the field-standard metric — positional disentanglement takes its argmax independently per position — so both are now published, and the matched one is what the position reading uses.

His power calculation reproduces exactly: 0.0130 at 100 seeds, 0.0184 at 50, against 0.223 on a single run for the threshold. Ratio 17.

---

## 🧰 What transfers

Everything above happens in a 27 × 27 world. The honest filter is to **rewrite each conclusion without any number** and discard what does not survive. What survives mentions no 27:

- **A no-go.** If your training procedure — parametrization, initialization, algorithm — is equivariant under a group acting transitively on the outcome set, the outcome is uniform on that set. No algorithm change alters it.
- **Its corollary.** A free per-referent embedding table makes the procedure exchangeable on the referent side, which alone is enough. Checkable by reading the architecture.
- **Channel noise does not break a tie between codes** under an exact-match reward, for any code size and any channel. Partial credit does.
- **The critical entropy coefficient is 1/N**, not "1/27" — the linearization gives a round-trip factor of (1/(Nβ))².
- **A sample maximum estimates nothing** when the value you want to exclude belongs to the support of the null.
- **A concentration of 1 does not imply compositionality** once the code stops being injective, and the counterexample generalises.

Hence the rule, already stated as the answer to Q29 and worth repeating as the operative one: **shrink to search, not to prove.** A small enumerable world is a discovery device — it makes arguments cheap to find and complete to check. What leaves it are the arguments that no longer mention its size. There is then no extrapolation to argue about; there are a theorem's hypotheses to verify, which is a finite operation.

And the asymmetry has to be stated: a toy is good at **refuting** universal claims and can never **establish** that something happens at scale. Everything negative in this article is worth more than the one positive result — which is itself half circular, since I built the parametrization that produced it and then measured that it can barely write anything else.

---

## ⚠️ Limits

**One world, one algorithm, one reward.** 27 referents, 27 messages, exact gradient ascent, all-or-nothing reconstruction reward. Nothing here is evidence about language models.

**Exact gradients do not scale.** Sampled REINFORCE gives different results, not approximate ones — experiment 2 established that. Everything measured here through the exact objective is a statement about the exact objective.

**No literature review.** I have not surveyed emergent-communication work. Where something here looks new to me, that is a statement about my reading and nothing else. The equivariance argument is elementary; someone has very likely written it down.

**The positive result is close to tautological.** The attribute-structured sender cannot write most bijections. Finding that it writes a structured code is not emergence, it is capacity. It also does not reach a compositional code — it stops at 0.42 rather than 1.0 — and it pays for the structure in task success, 0.92 down to 0.86.

**The recorded commitment was underspecified, and that is the most serious defect I found.** It was dated before any data, and presented as what made the test sharply falsifiable. It failed to name the parametrization, which is what decides the answer; its interpretation clause said an excess would prove a flaw in the equal-optima reasoning, which does not follow; and its first half predicted near-perfect bijections, of which there was one in twenty. Recording a prediction in advance protects against fitting after the fact. It does not protect against omitting a variable, nor against writing down the wrong interpretation in advance.

---

## 🗂️ Summary

- A referential game with 27 referents and 27 messages, small enough for the optimum, the null and the gradient to be **computed exactly**.
- All 27! perfect codes are tied in reward, so compositionality is 1.19 × 10⁻²⁵ of the optimum set **before any training**.
- The single-agent equal-optima certificate **does not survive** two agents; an equivariance theorem replaces it and gives the same number for any equivariant algorithm.
- **A free per-referent embedding table cancels in advance anything the message structure could contribute.**
- Exact gradient ascent reaches a perfect bijection **once in forty starts**; it executes whichever code it lands on to within 0.0000 of that code's arithmetic ceiling.
- Neither agent writes the code: freezing either gives **139 steps and the same value to eight decimals**.
- The first-step gradient prefers nothing; the structured parametrization's preference appears **between step 10 and step 30**.
- At 100 seeds, any residual selection is **smaller than 0.0087** in concentration.
- Channel noise breaks the symmetry, leaves the reward tie exact, and **produces nothing**. Breaking symmetry is necessary and not sufficient.
- **Eight of my hypotheses died in one day, none of them arithmetic.**
- Two questions I wrote down in July got answered: shrinking to exactness does **not** protect you, and the diagnostic for your own specification is **anything that does not inherit it** — including the experiment you have not run.

---

## ❓ Q&A

**Why 27 referents and not something realistic?**
Because at 27 the null distribution, the optimum set and the gradient are exact rather than estimated, and a second independent route to any quantity costs nothing. That is the entire justification, and it buys less than I assumed — see the eight dead hypotheses.

**Doesn't a result on a toy prove nothing?**
It cannot establish that a phenomenon occurs at scale. It can refute a universal claim, and it can produce arguments that never mention the toy's size. Those are the ones I would defend.

**Isn't "the parametrization decides, not the reward" already the consensus?**
As far as I can tell, yes. I have not surveyed the field, so I make no novelty claim. What may be worth something is that here it is a theorem with an exactly computed baseline rather than an empirical regularity — and that the corollary about per-referent embeddings is checkable without running anything.

**Would a bigger vocabulary or longer messages change it?**
The equivariance argument does not depend on either. The specific numbers do, entirely.

**What is the single most useful thing in here for someone else?**
Probably the embedding corollary, and the habit of computing a second independent route to every number. The second one caught more of my errors than exactness did.

---

## 💡 Did you know?

The 1 296 compositional codes are **exactly the orbit** of the canonical code under the group of message relabellings that respect the token-and-position structure — a group whose order is exactly 1 296, counted by backtracking rather than merely constructed. The two sets, built by two independent code paths, are identical. The only parametrization whose symmetry group is smaller than the full one is precisely the one whose symmetry group singles out the compositional codes.

---

## 🙏 Credit

The threshold critique, the power calculation, and the matching question are **`dipankarsarkar`** ([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)). This is his fifth round on this project, and the fifth to produce a correction. He is also the author of the product bound that carried article 2.

He went to the part that had never run. That is where mistakes are cheapest to fix and where nobody has any reason to look.

---

## 💻 Code and citation

Everything is reproducible: `src/test3_communication/` contains the eleven scripts behind every number here — ten of them written that day — `docs/TEST3.md` the design document with its dated corrections, and `docs/CARNET.md` the notebook with all sixteen dated dead hypotheses.

🔗 **GitHub:** https://github.com/RDTvlokip/RDTRL
📦 **DOI (all versions):** [10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216)

MIT licensed.