# 🔍 I published my RL experiments. A reader ran the code, and four of my numbers didn't survive 🇫🇷

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21726216.svg)](https://doi.org/10.5281/zenodo.21726216)

---

## 👤 What this is

My [previous article](https://huggingface.co/blog/RDTvlokip/teaching-a-network-to-write-with-reward-only) was about training a network to write from a reward signal alone — random weights, no pretraining, no data. Two experiments, both of which "worked" and both of which fell apart on inspection. The code went on GitHub with it.

Then someone read the code.

**Dipankar Sarkar** ([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)) commented four times over about a day. Each time he had run something. Not "have you considered" — he re-derived my closed forms, rebuilt my statistics in numpy because he had no torch on his machine, and pointed at line numbers.

By the end:

- he had found a **closed-form bound** on my results that I had missed entirely, and it is the strongest thing in the project now;
- **four numbers I had published turned out to be one seed each** — including one I had produced specifically to fix the previous three;
- an anomaly I had blamed on multithread non-determinism turned out to be **the dtype of one scalar**, on one line, in one function.

And then, separately and with nobody watching, the experiment I designed to validate his bound turned out to **test nothing at all** — an isomorphism of the original dressed up as a control. That one I caught myself, barely, and it is the most useful mistake in here.

This is that, written down. It is not a story about a clever result. Most of it is me being wrong in ways I could have caught and didn't.

Everything is reproducible: [RDTvlokip/RDTRL](https://github.com/RDTvlokip/RDTRL), MIT, archived at [10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216).

---

## 📐 The setup, in one table

You don't need the first article. Here is the whole environment.

A GRU generates three tokens. A hand-written parser checks them: `determiner noun verb`, with gender and number agreement. Reward is the mean of three sub-scores. No AI in the reward, no target sentence, dozens of valid answers.

| | |
|---|---|
| vocabulary | 20 tokens (6 determiners, 8 nouns, 6 verbs) |
| space | 20³ = **8,000 sequences — enumerable** |
| valid sentences | 48 |
| the two "corners" | 24 all-singular, 24 all-plural |
| training | REINFORCE + baseline, Adam 1e-3, entropy coefficient β |

That the space is enumerable is the only reason any of this is checkable. Every number below is **exact** — the true distribution of the policy over all 8,000 sequences — not estimated from samples.

The known failure from article 1: the agent reaches 99.9% grammaticality **without learning the agreement rule**, by hiding in one number-family where the constraint is vacuously satisfied. It uses 12 to 24 of the 48 valid sentences.

---

## 🧮 His bound: a policy with no coupling has a product support

Here is the argument, and it is his.

Suppose the policy has **no dependency between the determiner and the noun** — the noun distribution is the same whichever determiner was emitted. Then the set of sequences it can produce is a **product**: some set of determiners, times some set of nouns, times some set of verbs.

If that policy is also 100% valid, its support must fit inside the **largest fully valid product** contained in whichever corner it landed in. That is a hard ceiling on `2^H`, the effective number of modes, and it is computable by enumeration before any training happens.

I computed it exhaustively over every subset of nouns:

| | valid sentences | largest valid product |
|---|---|---|
| short grammar, plural corner | 24 | **24** = {des, les} × 4 nouns × 3 verbs |
| short grammar, singular corner | 24 | **12** — gender-locked |
| long grammar, each corner | 144 | **72** |

The two corners hold the same number of sentences and do not have the same ceiling. The reason is a `None` I wrote in the lexicon without thinking: `les` and `des` are gender-neutral, so they remove the determiner–noun gender constraint. The plural corner **is** a single product. The singular corner is a **union of two** products, masculine and feminine, and no product can span both.

Exactly one bit apart. And it explains a row I had published without understanding it. My table showed `12.0 modes, 100.00% valid, three seeds of three` and I had presented that as an interesting coincidence. His reading: *"is not a lottery outcome, it is the ceiling for an uncoupled policy inside that corner."*

### Measured against 70 seeds

One condition, β = 0.02, 20,000 episodes each.

| | |
|---|---|
| runs exceeding their corner's ceiling | **0 / 70** |
| max observed, singular corner (ceiling 12) | **12.0** |
| max observed, plural corner (ceiling 24) | **24.0** |
| the modal outcome | **the ceiling itself** |
| `I(determiner ; noun)` median | **0.0000 bits** |
| `I(determiner ; noun)` max over 70 | 0.0158 bits |

The coupled solution needs 1.0 bit of mutual information. The best any of seventy runs manages is 0.016. **Not one run acquires the conditional.**

One caveat he raised himself, and it halves the claim: the plural corner has 24 valid sentences and a largest product of 24, so **its gap is zero by construction** and a fully valid plural run cannot exceed its ceiling no matter what. So "zero violations in 70 runs" is really zero in the 37 singular ones. My grammar is asymmetric as an instrument. That became a design constraint on the test at the end of this article.

### The ceiling is a plateau, not a basin

I tracked `I(determiner ; noun)` step by step with an exact probe. Exact gradient, no sampling at all, seed 0:

```
  step     0 : I = 0.0045 |  47.54 modes | valid   0.60 %
  step   100 : I = 0.0000 |  12.00 modes | valid  99.99 %
  step  1000 : I = 0.0000 |  12.00 modes | valid  99.97 %
  step  1250 : I = 0.8518 |  17.87 modes | valid  99.92 %
  step  1500 : I = 0.9980 |  24.00 modes | valid  99.98 %
```

Exactly 12.00 modes with mutual information identically zero for a thousand steps, then it leaves. So the ceiling can be escaped with no noise whatsoever — it is a flat region, not a trap. Escape times vary enormously: step 1250, step 2875, and never within 4,000 steps on the third seed.

Nothing at initialization predicts which: `I` starts at 0.0045 bits in every seed and the six determiner masses all sit between 0.042 and 0.057, with no structure separating the seed that will couple from the one that won't.

### What actually separates exact from sampled

Every trajectory starts at **47.5 effective modes** — that's the untrained network, near-uniform over tokens. Training destroys diversity. But look at the minimum along the way:

| procedure | minimum modes reached | at step |
|---|---|---|
| exact gradient, 3 seeds | 10.74 / 11.06 / 11.24 | 25 |
| sampled REINFORCE, 3 seeds | **1.09 / 1.88 / 1.18** | 400 to 800 |

Sampled REINFORCE **crushes the policy onto a single sentence** before rebuilding it. The exact gradient never drops below 10.7.

The hypothesis this suggests, and I am labelling it as a hypothesis because I have not proven it: rebuilding from a near-deterministic point happens **position by position**, which is what a per-token entropy bonus can do, and a position-by-position rebuild yields a product by construction. Coupling would require opening a *joint* direction, which that term never opens. It predicts something checkable — the depth of the transient collapse should predict whether coupling is ever acquired — and I have not run that.

It also gives β annealing a mechanism it didn't have. In article 1 I explained the annealing fix with a story about keeping every conditional trained while the shared representation forms. The simpler reading: high β early **prevents the crush to a point**, so the policy never has to rebuild from a product.

![Four panels on the same runs. Panel A, a scatter of effective modes on one numeric path against the other, seventy seeds, coloured by which corner they landed in: every point keeps its colour across the two axes while only a third sit on the diagonal, and dashed lines at 12 and 24 are never crossed. Panel B, the same seventy runs as a strip plot per corner, with a solid bar at each corner's largest valid product and a wall of points stopping exactly there. Panel C, effective modes against training step on a log axis: all six trajectories start at 47.5, the exact-gradient runs dip only to about 11 while the sampled runs are crushed to roughly 1 before rebuilding. Panel D, twenty seeds sorted by early-stopping gap, most of them a single dot meaning no gap at all.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/zHTD2rlbzwKc5B6j7BMDI.png)

---

## 🪦 Four numbers I published that were one seed

This is the part I'd rather not write, which is why it's here and not in a footnote.

### 1. "The branch is biased about 2 to 1 toward singular"

I wrote that in a public reply, from 15 singular runs out of 24.

Those 24 were **3 seeds × 8 values of β**. And in the collapse regime the branch is decided by the *seed*: seed 0 went plural at every β, seeds 1 and 2 singular at every β. So I had **3 draws, not 24**. Wilson goes from [0.43, 0.79] to [0.21, 0.94] — no information at all.

Worse, my denominator was wrong before the independence question even arose: **11 of those 24 runs were not collapses**. At β ≥ 0.08 both families are alive, and a 50.1 / 49.9 split had been labelled "singular" by a bare argmax.

Redone properly, 70 seeds at one condition:

| | |
|---|---|
| singular / plural | **37 / 33** |
| Wilson 95% | [0.413, 0.641] |
| p against a fair coin | **0.72** |
| p against my claimed 2:1 | **0.016** |

Indistinguishable from a coin. My claim rejected. The order-1 marginal edge toward singular (+0.0167, computable in closed form) is real and does not survive the sampled dynamics.

### 2. "It hides in an all-plural sublanguage"

That sentence is in article 1, in its summary, and in the README. It was true of seed 0.

Over 70 seeds: **37 singular, 33 plural.** It hides in a single-number sublanguage; *which* one is a coin flip.

The uncomfortable part isn't the error, it's that **I already had the multi-seed sweep on disk when I wrote it.** The refuting data was measured, saved and pushed. The mistake was in the sentence, not the experiment.

### 3. "Early stopping would beat convergence by +12.5 modes"

From one seed: modes peak at 24.0 around episode 4,750, final state 11.5.

He noticed something I hadn't: that run and the sweep it was compared against came from **two different numeric paths** in my own code — the subject of the next section. On the sweep's path, he predicted from arithmetic alone, without running anything, that the same measurement would give **+5.4 rather than +12.5**. Measured: **+5.38**. Right to two decimals.

Then I traced 20 seeds on each path, and it got worse than his correction. On the canonical path:

| | |
|---|---|
| mean gap | +1.24 modes |
| **median gap** | **+0.03** |
| runs gaining more than one mode | **5 / 20** |
| runs gaining essentially nothing (< 0.05) | **10 / 20** |

Half the runs gain nothing at all, and fifteen of twenty gain less than one mode. His correction was too gentle: it isn't +5.4 instead of +12.5, it's a **median of zero**, and the +12.5 was a single run.

### 4. The fix for #3, which was also wrong

Having retracted it, I wrote something conditional that I liked: on one numeric path, all three runs that gained were in the plural corner — 3 of 8 against **0 of 12** singular — which fits the ceiling story, since that corner has twice the height to reach and lose.

Then I ran the same twenty seeds on the canonical path:

```
                        path A       path B (canonical)
  median gap             +0.00           +0.03
  runs above one mode     3/20            5/20
    of which plural       3/8             2/8
    of which singular     0/12            3/12
```

Three of five are singular. The 0-of-12 that carried the whole interpretation was a one-path artifact.

**This is the one that stings.** The first three were "one seed", a mistake I already knew about. The fourth was **twenty seeds** — I had applied the fix — and it was still wrong, because I had changed the dimension I was sampling and not the dimension that actually varied. It looked like the correction rather than the same mistake in better clothes.

The rule I take out of it is not "replicate", which I already knew. It is: **replicate what is convenient first**, and check the control you just built against the claim you just made with it.

---

## 🔬 An anomaly I blamed on non-determinism was one scalar's dtype

Article 1 carries this caveat, quoted exactly:

> *"Caveat before anyone quotes this: single seed, and I observed run-to-run variation at nominally identical settings (11.5 here vs 18.6 in the sweep), most likely torch CPU multithread non-determinism. I have not replicated it and I don't claim it as established."*

Hedged in the right places, and still wrong about the cause. The shape of the evidence should have told me: **both numbers were perfectly reproducible.** Non-determinism does not reproduce. I wrote "most likely non-determinism" about two values I could regenerate on demand.

He found it. The line that computes the advantage exists in two versions across my repository:

```python
rl_grammaire.py:141              (recompenses_t - baseline).detach()
stabilite_et_trajectoire.py:79   torch.tensor(r - baseline, dtype=torch.float32)
parametrisation_et_recuit.py:90  idem
localisation_effondrement.py:55  idem
trajectoire_couplage.py:84       torch.tensor(r - base).detach()
```

The first subtracts **in float32**: `recompenses_t` is already float32 and `baseline` is a Python float, so tensor–scalar promotion rounds the baseline *before* subtracting. Two roundings. The others subtract two float64 values and round once.

```
  float32 first : 0.08333331346511841
  float64 first : 0.0833333358168602
```

My reward function returns thirds and ninths, none of which are binary-exact, so the two lines stop being the same computation almost immediately:

| seed | first disagreement | % of first 2,000 steps | max relative gap |
|---|---|---|---|
| 0 | step **5** | 57.7 | 5.45e-06 |
| 1 | step **5** | 78.3 | 7.96e-06 |
| 2 | step **4** | 79.0 | 5.45e-06 |

One bit is enough because `distribution.sample()` is a threshold on a uniform draw: perturb the logits in the last place and eventually one token flips, after which the two runs share the seed and nothing else. And both stay reproducible because **both roundings are deterministic**. Two deterministic roundings explain "both are reproducible"; multithread noise never did.

Same seed, same loop, only the advantage line differs:

| path | final modes, seed 0 |
|---|---|
| float32 | **18.62** — the sweep's number |
| float64 then rounded | **11.50** — the trajectory script's number |

One correction to his reading, which he got right in numpy and wrong in torch: he counted this as **three** paths, treating `torch.tensor(r - base)` with no dtype as leaving the advantage in float64. `torch.get_default_dtype()` is float32, so that call returns a float32 tensor. Same dtype, same value to the bit. There are two paths, not three.

### The bug is a better experiment than any of my seeds

This is the part I did not expect.

Changing the seed changes the initialization **and** the whole sampling trajectory at once — a confound this project has carried since its first sweep. Changing the advantage line changes only the second: the initial weights are bit-identical, and the two runs diverge at step 4 or 5, during training. **Same starting point, different trajectory.** No seed can do that.

So I ran 70 seeds on each path:

| | float32 | float64 |
|---|---|---|
| singular / plural | 37 / 33 | **37 / 33** |
| Wilson 95% | [0.413, 0.641] | **[0.413, 0.641]** |
| p against 1/2 | 0.7202 | **0.7202** |
| ceiling violations | 0 | **0** |
| runs with I > 0.05 bits | 0/70 | **0/70** |

**All 70 of 70 keep the same corner.** Zero flips, although the trajectories disagree on 58 to 79 percent of the first two thousand steps. But only **21 of 70 keep the same effective-mode count**, correlation 0.68, mean absolute difference 2.87 modes and up to 12.7.

> **The initialization decides the corner. The trajectory decides how much of it gets filled.**

Two levels, two causes, separated by a manipulation that touches only one of them. And the conclusions the repository rests on — the fair coin, the ceiling, the null mutual information — are identical on both paths. They were not rounding artifacts.

### Which line is canonical, and a surprise

I expected a trade-off between accuracy and speed. There isn't one:

```
  float32 path : 19.46 µs per call
  float64 path :  4.57 µs per call   (77 % faster)
```

The name misleads. Nothing is stored in double; the produced tensor is float32 either way. A Python float **is** a double, so `r - baseline` is native and free and only one tensor creation remains, while the other path builds a tensor, dispatches a torch kernel for the tensor–scalar subtraction, and detaches. More operations *and* one more rounding.

In total that's 0.2% of a training step, so this is not a performance argument. It's that performance does not oppose accuracy here. float32 was never a choice — it was a writing accident in one function, already in the minority six files to five in its own repository. It is now the explicit non-default.

---

## 🔁 The reversal test, whose first version tested nothing

The bound above is computed from **my** lexicon. So the obvious objection: is it a law about the reward's product structure, or a coincidence of the French vocabulary I happened to write?

One experiment decides that. Build a lexicon where the ceilings come out different, record the prediction in advance, check.

### My first version was empty

I built one: move gender-neutrality from the plural determiners to the singular ones. Same twenty tokens, same 8,000 space, same 48 valid sentences, same two corners of 24. It swapped the ceilings to 24 and 12, and swapped both order-1 marginals as well. Everything flipped cleanly.

**Too cleanly.** The nouns and verbs in my lexicon are already symmetric in number — two per gender and number, three verbs of each. So swapping the determiners' number **is** the `sg ↔ pl` relabelling and nothing else. Verified on the feature multisets:

```
  det   standard with sg<->pl : {('f','pl'): 2, ('m','pl'): 2, (None,'sg'): 2}
  det   "reversed"            : {('f','pl'): 2, ('m','pl'): 2, (None,'sg'): 2}
  isomorphic : True
```

Seventy seeds would have returned the mirror image **by construction** and proved only that my code does not branch on the strings `"sg"` and `"pl"`.

I caught it before launching, but only barely, and only because the result table looked *suspiciously* tidy: every single quantity flipped, exactly, with nothing left over. That tidiness was the tell. The principle I was missing:

> **A relabelling can permute, but it cannot change a ratio.** A perfectly symmetric control is often a perfectly empty one.

I kept the empty variant in the code, documented. Deleting it would delete the lesson.

### The version that tests something

Three genders instead of two. The singular determiners are gender-neutral, so that corner is a single product; the plural ones are marked across three genders, so the gender must be fixed.

| | standard | three genders |
|---|---|---|
| tokens | 20 | 26 |
| space | 8,000 | 17,576 |
| valid sentences | 48 | 72 (brute-force checked) |
| size of each corner | 24 / 24 | 36 / 36 |
| **ceilings** | 12 and 24 | **36 and 12** |
| **ratio** | **2** | **3** |

Both corners still hold the same number of valid sentences. But the ceilings are now in a ratio of **3**, and no relabelling of a two-gender grammar can produce that, because the largest product is an isomorphism invariant.

Ceilings computed by enumeration and **committed to the repository before the run**. Seventy seeds:

| corner | n | predicted ceiling | max observed | violations | exactly on it |
|---|---|---|---|---|---|
| singular | 33 | **36** | **36.0** | **0** | 2 |
| plural | 37 | **12** | **12.0** | **0** | 7 |

And the quantitative form is stronger than the violation count:

| grammar | ceiling ratio | observed mean ratio |
|---|---|---|
| standard, 2 genders | 2.0 | 1.82 |
| **three genders** | **3.0** | **3.01** |

The mean tracks the ratio, not just the ordering.

The rest of the prediction holds too. The branch is still a coin at 33 / 37, p = 0.72, **despite both order-1 marginals having flipped**. `I(determiner ; noun)` has median 0.0000 and zero of seventy runs above 0.05 bits.

**One sub-prediction of mine was written too strongly.** I predicted "effective modes land on integer products". Measured: **66% in the standard grammar, 67% here**, within 0.05 of an integer. A stable fraction across two grammars, so a real fact, but not a rule — with six nouns instead of four, a non-uniform policy more easily gives a non-integer `2^H`. The ceiling is a law; the quantization is not.

![Two panels on the reversal test. Left, the three-gender grammar: seventy runs plotted by effective modes, split into a plural row and a singular row, with a solid bar at each corner's predicted ceiling, 12 and 36. No point lies past either bar, and the densest cluster in each row sits exactly on it. Right, the quantitative test: for each grammar a dash marks the computed ratio of the two ceilings and a dot marks the ratio of the observed means. The standard grammar reads 2.0 predicted against 1.82 observed, the three-gender grammar 3.0 against 3.01, and a note records that the first version of this test could only ever have landed on the left column.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/aAnMFn9RV-K0L6nzpqKVS.png)

**His bound survives a change of grammar, in value and not only in order.**

---

## 📈 The sweep, redone

The entropy sweep table in article 1 was three seeds, on the non-canonical path. Since the whole point of this article is that single-seed and single-path numbers move, leaving it there would have been indefensible. Redone: **8 β values × 10 seeds = 80 runs**, canonical path.

| β | validity % | effective modes / 48 | seeds covering both families | sg / pl |
|---|---|---|---|---|
| 0.0 | 100.0 ± 0.0 | 1.0 ± 0.0 | 0/10 | 6 / 4 |
| 0.01 | 100.0 ± 0.0 | 9.3 ± 5.6 | 0/10 | 4 / 6 |
| 0.02 | 99.7 ± 0.6 | 14.1 ± 5.5 | 0/10 | 3 / 7 |
| 0.05 | 97.0 ± 3.2 | 22.2 ± 2.0 | 0/10 | 4 / 6 |
| 0.08 | 86.4 ± 5.4 | **31.4 ± 10.9** | **5/10** | 4 / 6 |
| 0.12 | 58.6 ± 4.4 | 43.8 ± 1.3 | 10/10 | 7 / 3 |
| 0.2 | 21.4 ± 1.7 | 44.1 ± 1.9 | 10/10 | 3 / 7 |
| 0.35 | 5.4 ± 0.5 | 44.9 ± 1.1 | 10/10 | 4 / 6 |

The transition is still at β ≈ 0.08 and that is where the spread on modes explodes — **±10.9**, because five of ten seeds cover both families and five don't. The sg/pl column, which the old table didn't have, confirms across 80 more runs that the branch is a coin at every β.

Two numbers from article 1 also moved on the canonical path. One is cosmetic — the all-or-nothing control on the short grammar goes from 99.58% to 99.91%. The other is not: the long grammar's graded validity goes from **6.4% to 15.8%**, two and a half times larger, and it is quoted in the published article.

Neither is a surprise, and that is the point. Both are single-seed numbers, and I had written in my own notebook — two days before publishing — that a single-seed sweep cannot draw a frontier. Then I published one anyway.

---

## 💬 The exchange, in his words

He asked for this himself, at the end of the last round: *"On credit, raise it when there is a draft, and put the thread in it."* So here is the thread, quoted rather than summarized. Four rounds, roughly one day.

**Round one.** He opened by running my parser with no training at all:

> *"The collapse was decided before episode 1. I ran your Grammaire class, no training. […] So the vacuous corner was not found by 20,000 episodes of search. It was the steepest direction at step 0, and it is closed form."*
>
> *"Does the long run collapse to plurals too?"*

He was right about the mechanism and about the closed form. He was reading one position of a probe that has three; the noun position points the other way, which is why the order-1 greedy sequence is invalid. And the answer to his question was no: the long grammar collapses to a single number family, singular in four seeds of five.

**Round two**, the one that produced the bound:

> *"Both corners hold 24 sentences. Only one of them holds 24 for a policy that has not learned a dependency."*
>
> *"I re-enumerated your grammar rather than assume the arithmetic."*

And in the same message, the statistics:

> *"p = 0.31 against a fair coin, Wilson 0.43 to 0.79, and it is 3 seeds pooled across 8 betas rather than 24 draws of one condition. Separating 2:1 from even needs about 70 runs. Your balayage_graines docstring already makes that argument one level down."*

That last clause is the sharpest thing anyone has said about this project. I had written the argument myself, in a docstring, about a different table, and then not applied it to the one I published.

**Round three**, on the statistic I had used to claim no rule was learned:

> *"The statistic that says nobody acquired the conditional reads 0.333 on a policy that acquired it perfectly. […] So the quantity is determiner coverage, not the conditional."*
>
> *"And the part you will not enjoy. analyse_exacte returns masse_par_determinant. balayage_70_graines saves p_nom_sachant_det and drops it."*
>
> *"Do you still have the 70 policies, or only the rows?"*

Only the rows. He was right that the quantity was coverage; his proposed fix — mass-weighting — turns out to read 1.0000 for all four structures and separate nothing, because perfect agreement is reached by restriction as well as by conditioning. The quantity that works is the mutual information, and that is mine.

**Round four**, the dtype:

> *"Your two loops are algebraically identical. I checked that as well, at taille_lot=1 […] The difference is not in the algebra. It is the dtype of one scalar, which is exactly the kind of thing reading for algebra does not catch."*
>
> *"They stop being the same computation at step 5, and they disagree most across the first two thousand steps. That is the window your own table says decides everything."*
>
> *"Nondeterminism never explained 'both are reproducible'. Two deterministic roundings do."*
>
> *"What I could not do: there is no torch on this box, so I simulated the two roundings in numpy against your reward stream instead of retraining."*

He found a numerical bug in a library he could not run, on a machine without the dependency, by reading source and reimplementing the arithmetic. And he was wrong on exactly one point, for a reason worth keeping: he counted three paths where there are two, because `torch.tensor(x)` on a Python float returns float32, not float64 as numpy would.

**And the part I want on the record**, because it is the thing I have found hardest to do all week. Between his fourth message and my reply, he retracted his own diagnosis:

> *"And you are right about 11.50, I was wrong about why. Effective modes is 2^H of the joint restricted to the valid set, not a count of support, so an unbalanced les/des product sits under 12 without ever leaving the plural corner. I read a diversity number as a support size."*

He corrected himself, unprompted and in public, **faster than I delivered the numbers he had asked for**. I had those numbers on disk and hadn't sent them yet.

---

## 🧰 What transfers

The grammar is 20 tokens. These aren't.

### 1. A closed-form ceiling for uncoupled policies

If your reward's solution set is a union of products, and your policy hasn't learned the dependency, then its achievable diversity is capped by the **largest single product** in whichever region it entered — computable before training, from the reward alone. It converts "mode collapse" from a qualitative complaint into a number with a target. Credit where it belongs: this is Dipankar Sarkar's.

### 2. Perturb the numerics to separate initialization from trajectory

Changing a seed changes two things at once. Changing a **last-bit rounding** changes only the trajectory: same initial weights, different sampling stream. It is a free, already-available intervention that isolates a confound most experiments carry silently. Here it gave a clean answer — initialization picks the region, trajectory picks how much of it gets used.

It is also, incidentally, a robustness check: if your headline moves when you change a rounding, it was never a result.

### 3. A relabelling cannot change a ratio

When you build a control by *swapping* two conditions, check whether the swap is an isomorphism of your setup. If it is, the control is guaranteed to confirm and guarantees nothing. Make the predicted **quantity** take a different value, not a different label.

### 4. An artifact must carry its provenance

Six times in one session I lost or mislabelled data the same way: a filename or a caption that omitted a dimension the run actually varied.

| artifact | omitted | consequence |
|---|---|---|
| saved policy weights | the numeric path | **70 policies overwritten** an hour after I wrote they were on disk so nobody would need to retrain |
| a merge glob | the path, and its own output | 13 files for 6 shards, seeds counted twice |
| trace outputs | the seed range | two parallel shards into one file |
| result files | the path | a rerun would have clobbered the previous series |
| a figure caption | hardcoded next to a conditional load | the figure was about to name one path while plotting the other |
| an aggregation glob | the experiment | one row reported n=80 instead of 10 |

The fifth is the instructive one: I had *just* added path captions to all four panels of a figure, precisely because three of them came from different paths. An hour later I hardcoded one. **A caption written by hand next to a conditional load is a lie waiting to happen** — compute it from the data actually loaded, and make the fallback say it fired.

### 5. A defect you wrote down is not a defect you fixed

My saturation metric could exceed 100%, and a value above 100% meant mass leaking onto incompatible nouns — a failure that read as a success. I had that written in my notebook for two days before fixing it, and I only fixed it when someone asked for exactly that field. Writing it down made it *feel* handled, so nobody looked at it again, me included.

---

## ⚠️ Limits

- **One toy grammar family, 26 tokens at most.** The reversal test uses a second grammar, which is one more than I had, and still not many.
- **One algorithm.** "The sampled procedure" is a claim about a family, drawn from REINFORCE with a moving-average baseline. Not PPO, not a lower-variance estimator.
- **One β for the 70-seed studies.** The sweep covers eight, but the ceiling work is all at β = 0.02.
- **The ceiling binds at constant β in the collapse regime, not universally.** Annealing β from 0.2 to 0.01 reaches 45.3 modes, above the largest product over the whole valid set (24). So the bound describes constant-β sampling, not sampling as such.
- **The mechanism is localized, not proven.** I know sampling is necessary for the ceiling to bind. The "rebuild from a point is position-by-position" story is a hypothesis with a testable consequence I haven't run.
- **The plural corner has zero gap by construction**, so half my ceiling evidence in the standard grammar is uninformative. The three-gender grammar was designed to fix that, and does.
- **I have not surveyed the literature.** Where something here looks new to me, that is a statement about my reading, not about the field.

---

## 🗂️ Summary

I published two experiments. A reader ran the code and found a closed-form bound I had missed: a policy with no determiner–noun coupling has a product support, so at full validity it cannot exceed the largest valid product in the region it entered. Measured over 70 seeds, that bound is never crossed and the modal outcome is the bound itself; no run of seventy acquires the coupling. The same reader's questions killed four numbers I had published — three were a single seed, and the fourth was twenty seeds on one numeric path, produced while fixing the first three. An anomaly I had attributed to multithread non-determinism turned out to be a float32-versus-float64 rounding on one line, which also turned out to be a better controlled experiment than any seed: same initialization, different trajectory, and it shows the initialization picks the region while the trajectory picks how much of it gets filled. Finally, the reversal test — whose first version was an isomorphism of the original and tested nothing — was rebuilt with three genders so the ceiling ratio changes from 2 to 3, and the observed maxima follow exactly.

---

## ❓ Q&A

**— Isn't this just "my code had a bug"?**

Partly, and I'd rather say so than dress it up. But the interesting part isn't the bug, it's that the bug was a **better experiment** than the ones I designed. It held the initialization fixed and perturbed only the trajectory, which none of my seed sweeps could do. I would not have thought to build that intervention.

**— Why does a 5e-06 relative difference change anything?**

Because `sample()` is a threshold on a uniform draw. A last-place perturbation eventually flips one token, and after that the two runs share only the seed. What it does *not* do is change any aggregate conclusion here — the branch statistics, the ceiling and the mutual information are identical on both paths. Per-seed numbers moved; claims didn't.

**— The bound seems obvious in hindsight. Is it?**

It was obvious to him and not to me, after I had spent a day and a half staring at those exact numbers and writing a long article about them. I had published `12.0 modes, three seeds of three` as an interesting coincidence.

**— Should I trust the numbers in article 1?**

The aggregate conclusions, yes — they survive both numeric paths and 70-seed replication. The single-seed tables, no: use the ones here. Four specific sentences in article 1 are wrong and are listed above. That is the point of publishing the notebook alongside the code.

---

## 💡 Did you know?

`torch.tensor(0.1)` gives you a **float32** tensor, not float64, because `torch.get_default_dtype()` is float32. In numpy the same expression gives float64. That single difference is why a very careful reader, working from my source without a torch install, counted three numeric paths in my repository where there are two. Both of us reasoned correctly; one of us was reasoning about the wrong library's defaults.

---

## 🙏 Credit

The product bound, and the four rounds of criticism that produced most of this article, are due to **Dipankar Sarkar** — [ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367).

He derived the order-1 marginals independently from the published article, showed that the two degenerate corners hold the same number of valid sentences but not the same largest product, caught a statistic of mine that measured determiner coverage rather than agreement, caught a saturation metric that could exceed 100%, caught a sample-size claim that pooled three seeds into twenty-four rows, and located the advantage-line dtype by reading source he could not run.

He also corrected himself, in public, faster than I delivered the numbers he had asked for.

---

## 💻 Code and citation

Repository: **[RDTvlokip/RDTRL](https://github.com/RDTvlokip/RDTRL)** — MIT.

New in this round: `sonde_ordre1.py` (order-1 marginals at every position, closed form), `produit_et_saturation.py` (the ceilings by exhaustive enumeration), `balayage_70_graines.py` (the seed studies), `optimum_produit.py` (the optimum of the coupling-free class), `trajectoire_couplage.py` (step-by-step mutual information), `chemin_avantage.py` (the two numeric paths).

The lab notebook, `docs/CARNET.md`, is where all of this actually happened, with eight refuted hypotheses carrying their date of death.

```bibtex
@software{charlet_rdtrl_2026,
  author    = {Charlet, Théo},
  title     = {{RDTRL — Can a network learn to write from reward alone?}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21726216},
  url       = {https://doi.org/10.5281/zenodo.21726216}
}
```

---

**Théo CHARLET**

TSSR Graduate (IT Systems & Networks Technician) - AI/ML Specialization

Creator of AG-BPE (Attention-Guided Byte-Pair Encoding)

🔗 LinkedIn: https://www.linkedin.com/in/théo-charlet

🔎 RDTvlokip Search (my search engine): https://search.rdtvlokip.fr

🚀 Seeking internship opportunities

🔗 Website : https://rdtvlokip.fr
