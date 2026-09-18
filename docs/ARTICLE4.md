# 🧵 Fifty-two rounds with the same reader, then a saddle I chased alone. He found a rule I broke ten times before naming it; I found a col that took six tries to even locate 🇫🇷

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21726216.svg)](https://doi.org/10.5281/zenodo.21726216)

---

## 👤 What this is

[Article 1](https://huggingface.co/blog/RDTvlokip/teaching-a-network-to-write-with-reward-only) trained networks from a reward signal alone. [Article 2](https://huggingface.co/blog/RDTvlokip/i-published-my-rl-experiments) was a reader running the code and four of my numbers not surviving. [Article 3](https://huggingface.co/blog/RDTvlokip/i-made-my-world-small-enough-to-compute-everything) was the same reader's fifth round: a closed-form no-go theorem, a phase diagram, a literature review that found the whole thing published in 2021.

This is rounds six through fifty-two, plus a stretch after round fifty-two with no round at all. Same reader, `dipankarsarkar` ([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)), same project. Three eras: two with him in the room, one without.

**The first era** (rounds 6–24) is a fight over one small table: how many of a 27-referent game's runs land on a "collision," and how that count relates to a hand-searched worst case. It produced eleven numbered rules — most of them because something I had already published broke under a check that cost less than a minute — and two separate times where the number under a dozen rounds of dispute had already been computed and saved, correctly, on the first day.

**The second era** (rounds 25–47) is what happens once you stop trusting the aggregate table and look inside a single collision. "Collision" turned out to be three different things wearing one name: a genuine tie protected by a symmetry argument, a referent that never engaged with anything, and — the one this whole era kept circling back to — an Adam optimizer floor that froze a receiver mid-transfer and made a real transfer look like a permanent stall.

**The third era** (rounds 48–52, then two days with no round) closes a fixed-point question on a second, unrelated collision, derives a closed form for the fold that separates a graceful capture from a collapse, and then — once the reader went quiet — keeps going alone: a real saddle found hiding under an optimizer that had simply never been given time to warm up, proven by a causal test rather than argued from a hunch, and an exact identity question it still couldn't close after six honestly diagnosed attempts.

Read start to finish, this is a long argument for one habit: **before you contrast a statistic, check what it's a function of — and check again when the reconstruction it was measured on changes underneath you.** The third era adds a corollary: **that habit doesn't need someone else in the room to keep applying it, including to yourself.**

Everything is reproducible: [RDTvlokip/RDTRL](https://github.com/RDTvlokip/RDTRL), MIT, archived at [10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216).

---

## 📐 Where article 3 left off

27 referents, 27 messages, 3 attributes × 3 values. Sender and receiver are 27×27 stochastic matrices, `E[R] = tr(S·Rᵀ)/N`, objective `J = E[R] + β(H_S + H_R)`. All 27! bijections tie at reward 1; the 1,296 compositional codes are 1.19×10⁻²⁵ of that tie. Article 3 proved an equivariance no-go — any equivariant parametrization is uniform over the 27! codes, exactly, no Gibbs assumption needed — ran a one-day seven-question program on it, and found the whole argument published in 2021, twenty minutes into a literature review done after the fact.

What article 3 didn't do: run the game long enough, or look closely enough at any single collision, to notice that the entire investigation to that point ran through one specific optimizer, on one specific statistic, whose own null distribution had already quietly answered most of what was being asked of it.

---

## 🧮 Part one: the table that took nineteen rounds to read correctly

### Sixth round — annotating a stale bound is not fixing it

Published in article 3: three worst-case bounds — largest gap 0.1443, highest published score while double-counting 0.6314, highest non-compositional score 0.9294 — found by hill-climbing over **permutations**. Article 3 §6.5 and §6.7 had already shown that codes the dynamics actually reaches carry 1 to 4 collisions. I'd written "conditional on bijectivity" next to the bounds and left them there. Dipankar's point: the next step wasn't annotating them, it was rerunning them.

**The diagnosis was worse than "wrong regime."** My climber moved by transpositions, and a transposition of a permutation is a permutation — the move set couldn't leave the bijective regime **even in principle**. Not an unchecked assumption; an operator welded shut. Rebuilt on distinct-message floors:

| R floor | 27 | 26 | 25 | 24 | 23 |
|---|---|---|---|---|---|
| his largest gap | 0.0396 | 0.1628 | 0.1850 | 0.2002 | 0.2112 |
| my largest gap | 0.0526 | **0.1362** | 0.1783 | 0.1850 | 0.2152 |
| my max double-counting | 0.1943 | **0.6409** | 0.6530 | 0.6700 | 0.7015 |

**One collision does almost all of it.** At R=26 our two independent climbers agree to four decimals — 0.6409, apparied 0.5812 — and one collision beats a bijection-only search over the *entire* space (0.6314). On 100 tabular seeds at β=0.02: R=27 in 3 runs, 26 in 37, **25 in 43**, 24 in 16, 23 in 1. Ninety-seven of a hundred sit at or past the first collision; the median run is a collision beyond where the effect already arrived. It holds across the whole β range, not one setting — mean R climbs from 24.05 at β=0.005 to 25.25 at β=0.037, never reaching 27 on average — and the other two parametrizations sit deeper still: modal R=21 for the autoregressive-per-referent sender, R=23 for the attribute-structured one. The three bounds were replaced by a table indexed by R, with the operator's flaw written next to it rather than deleted.

### Seventh round — I'd fixed the denominator; the damage was in the numerator

Dipankar crossed his own R-indexed null against my measured `p(R)` over 100 seeds. His null: 0.11679 at R=27, 0.11697 at R=24; mixed over my `p(R)`, **0.11696** — flat, agreeing with my own fiber-profile calculation (eleven profiles, −0.0001 to +0.0005) by a completely different route.

**But his worst-case bound moves enormously across the same mixture**: 0.0396 at R=27 to 0.1751 mixed over `p(R)`. The null moved 0.00016; the worst case moved 0.1355. **A whole section had gone into rebuilding the reference when the damage was in the reachable set.** My three bounds weren't too generous — they were **too small**, off by 11%, not the factor-of-four his first framing suggested. A 150-run sweep (5 β × 30 seeds) then asked whether the effect shrinks with β or whether R stops predicting it:

| β | 0.005 | 0.010 | 0.020 | 0.030 | 0.037 |
|---|---|---|---|---|---|
| observed gap | 0.0077 | 0.0094 | 0.0100 | 0.0175 | 0.0106 |

| R | 27 | 26 | 25 | 24 | 23 |
|---|---|---|---|---|---|
| observed gap | 0.0096 | 0.0120 | 0.0139 | 0.0076 | 0.0106 |
| n | 8 | 30 | 53 | 47 | 12 |

**Neither.** Correlation with β: +0.158, no trend. Correlation with R: +0.091, 4% of variance explained. **The worst-case-attainable-under-a-floor and the gap-the-dynamics-actually-produces are two different functions of R** — one strongly monotone (0.0526 at R=27 to 0.2152 at R=23), the other flat. Ratio between them: 15.2. Neither number was wrong on its own axis; they had never been the same axis. His own β reservation settled the same way, by computation rather than argument: mixed over the `p(R)` specific to each β, the worst case runs from 0.1564 to 0.1773 — 0.0209 of amplitude across the whole β range, small relative to the quantity it bounds.

### A contrast that didn't survive its own second draw

A two-sigma contrast in the R table (R=25 vs R=24, t=2.53 by my own within-bin estimator, t=2.4 by his better-justified pooled one) got thirty more seeds each — sixty new runs, since R is an output of the run rather than a setting, and those two cells alone carried 100 of the original 150. The sign inverted on the sixty alone (−0.0053, SE 0.0033, t=−1.60), and on the pooled 210, everything drifted toward null: η² 4.11% → **1.2%**, F(4,145) p=0.19 → F(6,203) p=0.87, slope per unit R +0.00118 (t=1.12) → +0.00051 (t=0.61). The power-based ratio strengthened rather than weakened on replication — from 13.9 to **17.4× my own detectable resolution**, still pointing the wrong direction. The lesson generalized past R: *the cheapest way to know whether a two-sigma contrast is real is to withdraw more seeds, not to argue about it* — and this was the first time I did that before publishing rather than after.

### Ninth round — the table offered twenty contrasts, and the biggest wasn't the one I'd read

Dipankar corrected for ten pairwise contrasts on my five R-cells and got the two-sigma R-contrast down to a corrected p of 0.10. Rebuilt his null law by three more independent routes rather than take his simulation on trust — parametric with a re-estimated σ, a permutation of the 150 observed gaps over the design's own labels (no distributional assumption at all), and a parametric version with cell sizes redrawn each time (since R, being an output of the run, is itself a realization):

| | E[max\|t\|] | q90 | P(≥2.40) |
|---|---|---|---|
| his | 1.620 | 2.427 | 0.1066 |
| mine, σ known | 1.619 | 2.427 | **0.1066** |
| mine, σ re-estimated | 1.628 | 2.452 | 0.1130 |
| permutation on the real design | 1.624 | 2.428 | 0.1069 |
| cell sizes redrawn | 1.623 | 2.439 | 0.1100 |

His law matched the sigma-known row to three digits everywhere; the only real gap is that a re-estimated σ has slightly heavier tails, and letting R's own cell sizes vary moved things less than that — the fixed-cell approximation he'd been forced to make from outside was free. And the two-sigma R-contrast itself needed a correction first: my published t=2.53 was the within-bin estimator, already flagged as the worse one at n=8 and used anyway for the headline number — pooled properly across all five cells (the estimator a contrast test actually uses), it's **2.430**, closer to his 2.40 than my own number was, bringing the corrected p to 0.10. **The table actually offered twenty** — the same 150 runs also carry a five-level β line, ten more contrasts, printed in the same reply and read the same afternoon. His largest correction was |t|=2.968; mine, buried in a line I'd called "no trend beyond noise," was **already bigger** than the one we'd spent two rounds discussing.

**Five contrasts over two sigma, not one**, and the biggest was in the line I'd waved off without a test — that line's own omnibus was F(4,145)=2.595, p=0.039, computed and unread. Corrected on the true twenty: my R contrast moved from p=0.10 to **0.200**; the β contrast, 0.053. And the β line died on replication just as hard: sixty independent seeds took its lead contrast from t=−2.97 to **t=−0.34**.

A pooling mismatch surfaced in the same exchange, and it was neither of us being wrong: his own null-check compared his pooled contrast (+0.00256, inverse-variance across two halves) against my *published* +0.0028 — but that published number wasn't an inverse-variance pool at all, it was a fresh re-estimate from all 210 raw runs, a different estimator that doesn't have to agree. Pooled his way, my two actual halves (discovery +0.00631, replication −0.00535) give **+0.00206** — and against his own null, run end to end, that number sits at the 63rd to 82nd percentile of a table with nothing in it (P=0.822 unconditional, 0.634 conditioned on the winning pair). **My properly-pooled number was never worth publishing**; the only unselected estimate that survives is the replication alone, −0.0053.

**Rule adopted, four lines:** *K is declared before the data* (every pairwise contrast the design offers, summed across every factor read in the same sitting, reported or not); *every |t|≥2 gets printed*, not just the largest; *the corrected p comes from a permutation of the output over the design's own labels* (seconds, no assumptions, and the arbiter whenever the parametric version and a reader's version disagree); *a selected contrast never publishes pooled with its own replication* — the replication alone, with its own error bar. For this design the rule gives q95=2.99, against the 1.98 I'd been using.

### Tenth round — R is not a factor, it's the objective divided by 27

`|reward − k/27| < 10⁻³` for an integer k, in **150 cases out of 150** (k=R in 141, k=R−1 in 9), `corr(R, reward) = +0.9725`. **Stratifying by R stratifies runs by the value of the objective the optimizer was maximizing**, then asks whether a measurement bias differs between the ones that scored 25/27 and the ones that scored 24/27 — every cell in the table has a single reward value to five decimals. His own "bump" (a 13-run cell driving the contrast) reproduced exactly — and then dissolved: `corr(max, gap) = +0.4317` against `corr(R, gap) = +0.0172`. **The gap tracks where an unconstrained maximum happened to land, which has nothing to do with R at all.** For completeness, since it decides whether the β row is contaminated too: β doesn't move R (χ²(16)=15.67, p=0.476, full 5×5 crossing), just a weak linear component (corr=+0.197, p=0.016) — β stays a legitimate, assigned factor, its only real problems already on record (corrected p=0.053, Scheffé 0.072, t=−0.34 on replication).

> **Rule 5: a contrast is a fact about a column only if the column was set before the run** — not measured, not derived, not "a convenient output to index by." R fails at the strongest possible level: it's the objective on a grid. No diagnostic could have saved that row; it was disqualified by reading the generator.
> **Rule 6, for columns that pass rule 5: publish the breakdown number** — the smallest number of adversarially-removed runs that flips a contrast below its bar. Here: 2 of 150 and 3 of 150. A contrast that dies at 1.3% of the sample is reported with that integer attached, or not reported.

His own bump diagnostic — "removing any one of nineteen other cells moves the R-contrast by at most 0.27" — was itself a maximum over twenty-five one-at-a-time removals, exactly the fault he'd just taught me one floor up: a permutation null on the maximal drop gave E=0.430, q99=1.070, against an observed drop of 1.295, p=0.0008 — **it survives its own correction.** But his own count needed a correction too: not thirteen runs driving the bump, but a breakdown number of **two** — the smallest adversarial removal that flips the R-contrast below 1.98 — and the "cell" wasn't a cell at all, it was bimodal (four unusual runs sitting on nine ordinary ones, average concentration 0.182 against a plan-wide q90 of 0.171).

And rule 1, from two rounds earlier, carried its own hidden diary: "every factor read in the same sitting" makes the published p depend on how far I happened to scroll. Replaced with **Scheffé at 145 degrees of freedom** — a fixed bar of 2.817 at family-wise 0.10, giving 0.213 to the R contrast (against my scrolling-dependent 0.200) and 0.072 to β (against 0.053). Same numbers, fixed by the design before the first seed, no diary required.

Sixth time in this exchange the answer was already on disk — this time a column in the same file, twelve characters from the one under discussion.

### Eleventh round — the detection floor, and the threshold nobody had written down

Attacking both rules born the night before. Rule 6's bare integer — "dies at 2 of 150" — means nothing without knowing how many runs a **real** effect of that size would need to die at. Planting a real effect and remeasuring against four different sources of resampled residuals gave four different answers (medians 2, 3, 4, 4; powers 0.63 to 0.80) — **no single specification produced all four of his numbers**, and the spread between them was the actual finding: an ambiguous statistic whose model lived hidden inside a calibration step, exactly the same fault as a bare integer with the model hidden in the reader's head. Retracted, not amended.

His "the plan has two-thirds power at the size it found" turned out to be his own p in a different costume — power computed at the *observed* effect size is a one-to-one decreasing function of p and can contain nothing p doesn't already. His actual finding held up: `R = len(unique(code))` bounds `reward ≤ R/27` in all 210 runs, never off by more than 1 — a counting fact, not a regression.

The **detection floor**, which replaced the p column — `2.80 × SE`, a property of the design alone, needing no data:

| contrast | n | SE | floor | observed | observed/floor |
|---|---|---|---|---|---|
| R 27 vs 26 | 8+30 | 0.00516 | 0.01445 | −0.00249 | 0.17 |
| R 25 vs 24 | 53+47 | 0.00260 | 0.00728 | +0.00631 | **0.87** |
| β .005 vs .03 | 30+30 | 0.00330 | 0.00925 | −0.00981 | **1.06** |

**Every observed effect sits at or below its own floor** — including the two that had occupied four rounds. What neither the power calculation nor the floor asks is "what would it have taken to see something?" — the original round-seven table existed to compare the dynamics' own gap (0.0110) against the hand-searched worst case (0.1443), ratio 13.1:

| target ratio | required increase | against the largest observed effect |
|---|---|---|
| 2 | 0.0611 | 6.2× |
| 3 | 0.0371 | 3.8× |
| 5 | 0.0178 | 1.8× |
| 8 | 0.0070 | 0.7× |
| 10 | 0.0034 | 0.3× |

**I stopped there rather than pick the number that flatters the argument.** The verdict flips between "5" and "8": call "nowhere near" a factor of two, and the whole table sits six times below what could count; call it a factor of eight, and the contrasts were in range and the exercise was legitimate. A relevance threshold — how far above the floor an effect must sit to matter — turned out to be the one number in this whole exchange that had to be fixed before the data and never was: naming it after seeing the results makes it unusable no matter which way it points. The detection floor stays honestly computable today; the relevance threshold became uncomputable the moment the results were seen.

### Twelfth round — the floor was a p in disguise, and the variable isn't continuous

`floor = 2.80 × se` and `t = d / se`, in the same file — so "observed/floor" **is** `|t|/2.80`, identically, before any data exists. "Every effect is at its floor or below" is `p > 0.0058` at 145 df, an alpha 8.6× stricter than the 0.05 embedded in the floor's own definition — and I'd published the same fact twice in one message under two names, flagging only one. The column was retired.

The floor's *prospective* use survived and reached further back than where I'd applied it: at replication sample sizes (12+12, 30+12), neither of the two contested contrasts could ever have been confirmed, even if true. I'd reported the sixty-seed replication as a failed test; it was only ever an underpowered estimate.

Dipankar's proposed fix — a floor built from a pilot standard deviation, `2.80 × σ_pilot × √(1/na + 1/nb)` — fails structurally: Bartlett's test across β levels gives χ²=19.176, p=0.0007, a 2.07× ratio of standard deviations, and a common σ mis-estimates the floor by 12% to 38% depending on the cell. The mechanism is why: the gap is bounded by zero below (the unconstrained maximum is always at least the matched value), and across eighteen cells with n≥4, `corr(mean, sd) = +0.874` Pearson, +0.917 Spearman, slope 0.82, median CV 1.07. **σ isn't a nuisance scale here — it's approximately the quantity being measured.** A pilot only fixes the floor if its mean happens to match the run's, which means already knowing the effect. What I found instead reaches further back than his proposal, in a different unit: if `sd ≈ CV × mean` with CV stable near 1, then `floor/mean = 2.80 × CV × √(1/na+1/nb)` needs no σ at all. At 30 seeds per cell and CV≈1.2, this predicts **0.89** — checked against the real table: **0.894** (0.00925/0.01035). Writable before the first seed, one line, no pilot: *at thirty seeds per cell, this design sees a near-doubling of the effect and nothing smaller.*

**The bigger find, from asking "what does this measure feed?" for the first time**: the gap-max-minus-appariée statistic existed to bound the inflation from publishing concentration unconstrained rather than matched — and its only consumer, the 0.35 threshold, **had been retired on day one of this critique, at round six.** Rounds six through twelve had priced a measurement whose consumer no longer existed. The one version of this question that *did* still have a consumer — a plain one-sample estimate of the mean gap, not a contrast — had already been answered, at a precision none of the contrast tables came close to: mean 0.01035, SE 0.00085, 95% CI [0.00869, 0.01201], distance to the worst case **158 standard errors**, against a one-sample floor of 0.00237 (three to six times finer than any of the 0.00728–0.01445 contrast floors). **Rule 7: when a claim is retracted, list every measurement whose sole consumer it was, and stop measuring it.**

### The variable that was never continuous

Found the same evening, checking a proposed calibration pilot: **63 of 210 runs (30%) have an inflation gap of exactly zero** — precisely when the unconstrained argmax is already a bijection. Not a continuous quantity with a floor: **a point mass plus a skewed positive part**, and every t-test, permutation correction, and bootstrap across a dozen rounds had treated it as neither.

| quantity | value | ratio to 0.1443 |
|---|---|---|
| E[gap], published | 0.01035 | 13.9 |
| E[gap \| gap>0], matched comparison | 0.01479 | 9.8 |
| max observed over 210 runs | 0.05927 | **2.4** |

The four-round contrast decomposed into two different quantities with two different consumers: `P(collision) = 0.700`, `E[inflation | collision] = 0.01479`, whose product is exactly the published mean. `min`, `max`, and an exact-zero count would have caught this on round one.

### Thirteenth round — the answer was already in a field named `inflation_moyenne_globale`

Testing his proposed pilot statistic exposed that I'd read a Fisher p=0.66 with no floor either — the same fault, one round after retiring the same lesson for power. His three legs on the mixture's coefficient of variation all reproduced exactly: `CV_pos²/p = 0.9736`, `(1−p)/p = 0.4286`, summing to 1.4022 against a measured mixture CV² of 1.4022 — **30.6% of the CV² a design document would write down is just the collision rate itself**, with no scale in it at all. Relative floors: mixed at 30+30 gives 0.8561, conditional at 21+21 gives 0.7133 — a 17% gain from conditioning. A bootstrap window on how much of the effect the zeros carry: [1%, 53%] and [0%, 56%] — an interval reaching halfway doesn't support "they carry the size of the inflation." **That sentence was retired.**

But chasing where the CV-preinscription idea should have been checked led to a file generated on the **first day of the whole test**, before any of this critique: `results_test3/loi_nulle_longue_n10000000_g0.json`, field `inflation_moyenne_globale`:

```json
"taux_global": 0.7464519,
"inflation_moyenne_globale": 0.010049794802284647,
"inflation_maximale": 0.10807050074977963
```

| quantity | null, 10⁷ draws (day one) | 210 runs (thirteen rounds later) | |
|---|---|---|---|
| P(collision) | 0.7465 | 0.7000 | p=0.13 |
| E[inflation] | **0.01005** | **0.01035** | z=0.36 |
| E[inflation \| collision] | 0.01346 | 0.01479 | z=1.32 |

**The quantity I had measured, bounded, contrasted, corrected for multiplicity, replicated, and defended over eight rounds was the null distribution's own mean.** Not close to it — it. A Kolmogorov-Smirnov test nobody had run: 210 concentrations against 20,000 null draws, **D=0.0508, p=0.638**. And this was exactly what my own §6.2, published the same first day, had already established at 100 seeds: z=−0.0098±0.1025, KS p=0.386.

**Rule 8: when a result is established, list every quantity still being measured that it answers, and stop measuring it.** Rule 7 fires on retraction — cheap, and it caught the dead threshold. Rule 8 fires on establishment, and it's the one that would have stopped everything the day it started: §6.2 landed the same day the null file was generated, saying emergent codes are drawn from the null, and the null file already held the answer under its own honest name.

### Fourteenth and fifteenth rounds — the maximum was the first fifth of the draw, and the ratio's limit is 1

A reservoir sampler capped at 2,000,000 draws kept serving a stale maximum past that point — 13 of 10⁷ draws exceeded the number I'd published as *the* maximum. Fixed in five accumulator fields. Then Dipankar showed the whole "13.9× beats chance" ratio decays with sample size by construction, since the denominator is an order statistic: 1.335 at 2×10⁶ draws, 1.179 at 10⁷, 1.038 at 3×10⁹. Auditing every number he'd sent me since round one: about twenty reproduced exactly, one was conceded wrong, and **two never reproduced** — both of them the two numbers that had carried their round's argument. I had noted the discrepancies and continued because the conclusion didn't move — but it never moved because my own numbers always drifted *further into his favor*, never less. A disagreement that only ever strengthens one side is a disagreement never actually tested.

His own 24-restart worst-case bound (0.1443) turned out to be **one cell of a two-axis table with no coordinates printed**: 6 restarts gave 0.143824, 24 gave 0.146685, 384 gave **0.154322**. The ratio to the null's maximum was two variables disguised as one. **The limit of the ratio is exactly 1**: both procedures — sample-and-take-max, and hill-climb over the same bijective space — estimate the *same* supremum, one by random draw, the other by local search, and computing that supremum directly (1,500 restarts, two independent neighborhoods, 43 seconds) settled it: **0.154322**, reached by 1 restart in 600.

| | value | share of supremum |
|---|---|---|
| supremum, direct search | **0.154322** | 100.0% |
| my published bound, 24 restarts | 0.144297 | 93.5% |
| his 3×10⁹-draw max | 0.139048 | 90.1% |
| max of 210 emergent runs | 0.059270 | 38.4% |
| mean of 210 emergent runs | 0.010350 | 6.7% |

The null's tail-quantile logic also came apart the same round: the reservoir's zero-overshoot count at some thresholds was a bound, not an estimate, so I'd fit a log-linear extrapolation (R²=0.990) to convert it — then re-fit on different sub-windows and got estimates spanning a **factor of 3.5** (2.60×10⁻⁸ down to 7.45×10⁻⁹) depending on which window I'd have chosen after seeing it. Left unpublished; the honest column stayed his — zero overshoots in 10⁷ draws, p < 3.0×10⁻⁷ by the rule of three, plus a sentence saying the tail's shape places the true value one to two orders below without saying which.

The optimum's own structure nearly misled me a second way. Its information matrix has exactly one non-zero row — all three message positions carry information about a single attribute, so the greedy per-position maximum collects it three times while the matched assignment can only use it once (`conc_max`=0.2465, matched=0.0922). First reading: the worst inflation must live on degenerate, low-concentration codes that couldn't mislead a real reader. **Checked, and wrong** — the maximum inflation under a floor on `conc_max` stays essentially flat from 0.30 to 0.60 (0.140, 0.140, 0.140, 0.140, 0.133), and at floor 0.50 the matched concentration reaches 0.4735 while the same code reads **0.6137** on the published statistic. The bound is serious at every concentration level a reader would actually act on, not just the degenerate corner the first optimum happened to sit in.

His n-vs-n/10 test, applied to my own published fields, separated two kinds of quantity I'd been treating as one: estimates (mean, sd, q50, q99, `E[inflation]`, `P(inflation>0)`) move by 0.00–0.03% between 10⁶ and 10⁷ draws; order statistics (`q99.9999`, the maximum, `max inflation`) move by 2.45%, 12.96%, and 17.95% over the same range. My own `quantiles_queue_exacts` docstring argued it was "exact, not estimated from a subsample" — true, and irrelevant, since at 10⁷ draws the 1−10⁻⁶ quantile just *is* the tenth-highest draw: exact in the sense of not being subsampled, not in the sense of being stable. One docstring, two different meanings of "exact," conflated.

> **Rule 9, on his question of whether rule 8 fires on a method too: yes, and it bites here.** A choice of method is a claim about a class of quantities; write the class down in the same commit as the method. Its own limit, which I don't think is solved: writing the class moves the judgment call to where the boundary is drawn, by whoever sees the least of what will later come close to it — a version I couldn't have honestly written in August, before an adversarial optimum existed anywhere in the project.

### Sixteenth round — his habit is real; mine is its mirror

Dipankar conceded two unpublished conditioning choices in his own statistics. The first, `sub = counts[counts > 0]`, silently dropped replicates where a simulated contrast never crossed the bar — his `P(rupture ≤ 2) = 0.252` was conditional on "the contrast is even reportable," a caveat never in the sentence. Recalibrated per arm: reach rates 0.665–0.684, `P(≤2|reached)` 0.242–0.285, unconditional `P(≤2)` **0.496–0.514** — and it exposed a matching bug of my own: my earlier "0.488/0.354/0.518 depending on where the residuals came from" had fixed one calibration delta from the per-R-level sigma, then drawn residuals whose own standard deviation was 10% smaller in the "per-cell" arm — that arm carried a mechanically larger effective effect (power 0.796 vs 0.682) and resisted mechanically better. Recalibrated per arm, all three land between 0.496 and 0.514; the source of the residuals moved nothing. The second concession, on pooling: five of his six lines reproduced, including 0.8013 exact to four digits — but **the second didn't**, 0.7066 on my side against his 0.4499, and it contradicted the very mechanism he stated in the same message (conditioning on the lower-error-bar pair should shrink the discovery mean from 0.00657 to 0.00414, a 37% drop carrying 63.5% of the weight — his own fifth and sixth lines show exactly that shrinkage, his second line showed 0.0155). His question "which of the two were you tracking" answered itself: not a different object or conditioning — **a different threshold.** He'd been checking my published +0.0028 (raw repooled 210 runs); I'd been checking my +0.00206 (inverse-variance across the two halves, matched to his own procedure). Against his threshold, my numbers came back 0.4523 and 0.7066 — neither of us was wrong, neither had written down what we were measuring against.

His mechanism, turned back on me: conditioning on a selection event shrinks a statistic toward null every time, so his numbers had been systematically conservative — which in this exchange means favorable to me. One habit applied twice, not two accidents. Audited my own notebook: twenty-five dead hypotheses, twenty-four weakening something I'd claimed, **one** strengthening it (a held-out-token test I'd predicted at chance and measured at 0.9966). "Every correction weakens me" isn't proof of bias by itself — it's also what honest convergence from overconfidence looks like. The test that separates the two: *have I ever spent compute to make a negative stronger?* Never — my §6.2 bound (any residual selection below 0.0087) is a pure function of sample size, tightening it costs the same as everything I'd spent weakening positive claims, and in six days nobody had proposed it, me included.

> **Rule 10: the name of a statistic carries every argument its value depends on.** `inflation_maximale` hides its sample size. `P(rupture ≤ 2)` hides its conditioning on reachability. `quantiles_queue_exacts` at 0.999999 hides that it's the tenth-highest of ten million draws. If I keep one rule from sixteen rounds, it's this one — it costs a variable name.

### Seventeenth round — outside code arrives, enumeration alone doesn't decide, and a negative I finally tightened held

Dipankar rebuilt the whole measurement pipeline from the published definition rather than my code — `sklearn.metrics.mutual_info_score`, `scipy.optimize.linear_sum_assignment`, the world reread from the design document. Agreement to 3.05×10⁻¹⁶ on 3,000 bijections, and exactly 1.0 on all 1,296 compositional codes. Fifteen rounds of numbers were not two expressions of one shared misreading of the definition. And a validator I'd trusted for six days turned out to test only itself: its two halves import the feature tables and both take the argmax by column — a vectorization check against a lookup table, nothing else — and above all, **it only ever draws permutations**, so `matrices_information_generale`, the function underlying every emergent-code result across six different scripts, never appears in it at all. His control on 2,000 codes carrying real collisions: 2.50×10⁻¹⁶ agreement. And the price of the guard failing, if it ever had: **0.0205**, twice the published mean inflation — the number that should have been published next to the guard since day one.

A closed-form lattice bound for the worst-case inflation (using an information-theoretic marginal constraint on each row and column of the information matrix, capped at `log₂3` since a bijective code over 27 uniform referents makes the three message positions mutually independent) narrowed the search space to 3,123 candidates without ever training anything — and still **didn't settle it**: direct maximization on the lattice reached 0.340, a factor of 1.53 short of the true supremum. *The enumeration transformed "restart forever" into "3,123 candidates, almost none of which are codes" — a position worse than the search it was meant to replace.* His own claimed witness code, reported not reproducible on 1,500 restarts of my own search (his rarity arithmetic — 0 of 1,500 at a rate of 1/600 gives p=0.082 — meant my empty search wasn't evidence against him either), turned out real once he shared it directly the following round — my own script had a line printing "inflation of this code = 0.060758" right next to the witness row, under a constraint I hadn't imposed. The witness was in my own message; the matrix wasn't.

**And a negative result got tightened rather than defended.** Rerunning §6.2 at 600 seeds instead of 100, against a null of 200,000:

| arm | seed | n | z mean | SE | detectable |
|---|---|---|---|---|---|
| tabular | 0 | 100 | −0.0098 | 0.1025 | 0.00874 |
| tabular | 11 | 600 | +0.0099 | 0.0421 | 0.00359 |
| tabular | 907 | 600 | +0.0195 | 0.0412 | 0.00359 |
| factorized | 0 | 100 | −0.0514 | 0.1014 | 0.00869 |
| factorized | 11 | 600 | **+0.0935** | 0.0429 | 0.00356 |
| factorized | 907 | 600 | **−0.0280** | 0.0417 | 0.00356 |

the bound sharpened by a factor of **2.45** — 0.00874 to 0.00359/0.00356 — and a 2.18σ excursion on one arm (the factorized parametrization, seed 11, z=+0.0935, CI touching [+0.009,+0.177]) did not survive being pooled with a second 600-seed run on a different seed (seed 907, z=−0.0280, sign flipped, zero inside the interval — combined **+0.0327 ± 0.0299**, |z/SE|=1.09). **The sixteenth contrast in this exchange to die on its second draw**, and the first one I retracted before writing an interpretation around it. One thing pooling hides and is worth saying anyway: the two 600-seed runs on the factorized arm differ from each other by +0.1214 with SE 0.0598 — t=+2.03 — meaning a single 600-seed run's own standard error understates the true run-to-run variability even when that one run is entirely honest.

### Eighteenth round — a closed form for his unexplained constant

The masses his search kept landing on were multiples of an unexplained quantum, U ≈ 0.018156. Derived in closed form:

> **U = (2/27)·log₂(32/27) = (2/27)(5 − 3·log₂3) = 0.018156481321225**

against his measured 0.018156481321 — agreement to 2.25×10⁻¹³. And `32/27 = 2⁵/3³` isn't arbitrary: `log₂(32/27) = 5 − log₂27` is exactly the gap between five bits and the width of this 27-referent world. His own claim from the same round — "the seven highest optima are all integer multiples of U" — didn't survive a re-ordering of his own data stream I ran independently: a seventh optimum sat at 0.46U off the lattice. Both of us had been partly right; his closed-form quantum was real, his universal-quantization claim wasn't.

### The premise describes a population the dynamics almost never reaches

Same round, pushed further on my own initiative, on two prompts: *what was under our noses from the start*, and *we're never sure the code was right from the beginning*. Five candidates checked against the repository before claiming anything, and **four were already documented** — the reward cost of the structured parametrization (0.930 tabular against 0.861 structured, already in §7.19), whether concentration tracks compositionality at all (already measured, Spearman 0.814, concordance 0.863), whether the "outcome written at initialization" result (z=+6.80 at percentile 1.000) was buried (it isn't, it's in §7.21), and whether the exact-gradient design was hidden (it isn't, stated twice in the design document). Saying so is half the useful result.

**The fifth wasn't documented, and it mattered most.** `reinforce()` is defined once in the whole test-3 codebase and called from exactly one site, inside the *stable* branch, starting from a state exact ascent had *already* reached, to ask only whether it stays there. Every reachable result up to this point — §6.1 through §6.7, every concentration distribution, every null comparison, eighteen rounds of this exchange — is Adam on the closed-form `E[R]`. No sampling, no reward variance, no credit assignment, ever. The project's own conclusion is "on this bench, compositionality was never selected," and the project's actual question is whether *reinforcement learning* selects it — a distinction the design had already been burned by once (§1.12: a measured "critical β" of 0.0381 that turned out to be Adam's own stopping behavior, not the objective's, with the Hessian giving 1/27=0.037037037 to 2.4×10⁻¹¹).

And a second, bigger problem sat under the premise itself. The headline arithmetic — all 27! bijections tied at reward 1, so a compositional outcome must come from outside the reward — only applies to codes that **are** bijections, and every one of the 1,296 compositional codes is one:

| arm | n | bijective | share | compositional | 95% upper bound |
|---|---|---|---|---|---|
| tabular | 1,200 | 60 | 5.0% | 0 | 6.0% |
| factorized | 1,200 | 1 | 0.1% | 0 | 97.5% |
| structured | 40 | 1 | 2.5% | 0 | 97.5% |

**95% of tabular runs, and 99.9% of factorized ones, never enter the population the premise is even about.** They stop at reward ≈0.93 with ~1.8 collisions. The published bound (z=+0.0147±0.0294) was computed on all 1,200 runs; on the population where the question is actually askable, it's 5.4× looser (z=+0.0507±0.1585, 60 bijections). The direct form of the design's own question — of the runs that reached the tied set, how many are compositional — had never been computed: **0 of 60, upper bound 6.0%, against a null of 1.19×10⁻²⁵**, twenty-four orders of magnitude of slack, a test with no power at all. Minimum Hamming distance to any compositional code ever reached: 19 of 27 for tabular, 7 for structured — nothing in the equivariant arms ever came close. This doesn't make the conclusion wrong (a real pull toward structure would raise concentration in the non-bijective majority too, and it doesn't), but it separates two different claims the design document had fused: a within-fiber-class uniformity result, measured at high precision, and a tied-optima selection test, measured at n=60 with a useless bound.

Then the dynamics itself, checked rather than assumed. **First, the 0.93 plateau is real convergence, not truncation:**

| steps | mean E[R] | bijections | collisions |
|---|---|---|---|
| 3,000 | 0.92896 | 2/12 | 1.83 |
| 12,000 | 0.92901 | 3/12 | 1.83 |
| 30,000 | 0.92901 | 2/12 | 1.92 |

Ten times the budget moves the fifth decimal. **Second, it converges to a strictly worse point of its own objective** — `J` from random init reaches 0.96395; fitted onto a compositional code and held there, `J=1.00000` exactly. The gap is real, not an artifact of the entropy term (which would predict the opposite): **the landscape has local optima, and ascent from random init falls into one about 95% of the time, 0.036 below the global point.** The premise isn't that reward fails to pick among tied bijections — it's that the dynamics never arrives at the tied set to begin with.

**Third, a correction to my own first-draft instinct.** On the §1.12 precedent, I nearly blamed Adam again — and checked the raw gradient instead of the training loop:

| steps | E[R] | ‖grad J‖ | relative | collisions |
|---|---|---|---|---|
| 0 | 0.037037 | 2.107×10⁻⁵ | 5.58×10⁻⁵ | 10 |
| 1,000 | 0.888496 | 5.005×10⁻⁵ | 2.69×10⁻⁷ | 3 |
| 30,000 | **0.888889** | **3.653×10⁻⁷** | **1.13×10⁻⁹** | 3 |

The gradient goes to zero, and 20,000 steps of plain SGD at lr=1.0 from the plateau move `E[R]` by 7×10⁻⁵. **A genuine critical point of the objective, not Adam stalling** — and `0.888889` is exactly 24/27, the reward of a 24-distinct-message code. The sub-optimal attractor is a property of the landscape, not the optimizer — which turned out to matter for the opposite reason I expected once REINFORCE was actually run the next round.

### Nineteenth round — the 5% is a property of my optimizer, not the bench

Whether REINFORCE from random init is a fidelity check or an escape mechanism, tested rather than argued: two runs, and the second reverses the first. **First run**: exact ascent 0.939 with 1/25 bijections; REINFORCE (batch 64) 0.892 with 0/25; batch 8, 0.374 with 0/25 and 6.76 collisions — monotone, exactly what an unpaired-budget artifact looks like, and I nearly sent it as confirmation. What stopped me: `monter` runs Adam at lr=0.05 for 3,000 steps, `reinforce` runs lr=0.01 for 4,000 — publishing that gap as variance would have been `plafond_beta` all over again.

**Second run, both budgets, both learning rates, 12 seeds/cell:**

| batch | steps | lr | E[R] | bijections | collisions |
|---|---|---|---|---|---|
| 64 | 4,000 | 0.01 | 0.895 | 0/12 | 1.92 |
| 64 | **20,000** | 0.01 | **0.992** | **11/12** | **0.08** |
| 64 | 4,000 | 0.05 | 0.975 | 5/12 | 0.58 |
| 64 | 20,000 | 0.05 | 0.989 | 9/12 | 0.25 |
| 8 | 4,000 | 0.01 | 0.386 | 0/12 | 6.75 |
| 8 | 20,000 | 0.01 | 0.929 | 3/12 | 1.08 |
| 8 | 4,000 | 0.05 | 0.856 | 0/12 | 3.33 |
| 8 | 20,000 | 0.05 | 0.925 | 2/12 | 1.92 |

The 0.374 was **undertraining**, not variance — extending exact ascent itself to 30,000 steps gave only 2/12 bijections, gradient still at 7×10⁻¹¹. Matched at 20,000 steps, same seeds, same learning rate, one difference: exact ascent 0/12 bijections at both lr=0.01 and lr=0.05, REINFORCE 11/12 at lr=0.01 (Fisher exact **p = 9.6×10⁻⁶**) and 9/12 at lr=0.05 (**p = 3.4×10⁻⁴**). **He was wrong, and I was more wrong** — I'd written the day before that sub-optimal attractors are a property of the landscape that "no local method escapes" and "isn't repaired by changing optimizer." The gradient really does fall to 7×10⁻¹¹ there, so the critical point is real — **but a critical point isn't a strong attractor for a noisy method.** The 5% ceiling from the earlier round was exact ascent's own property, not the bench's — and `reinforce()` had only ever been called from a state exact ascent had already reached, so the actual experiment the design document was framed around had never been run at all.

### Twentieth round — my neighborhood was only 47.3% certified, and seven optima weren't

Asked where my hill-climb's stopping rule actually stops. The tolerance and iteration cap were fine; what was broken was that my neighborhood — 351 transpositions plus a sample of 1,200 out of 2,925 possible 3-cycles — only certifies each stop against 1,551 of the full 3,276 moves per check — **47.3% certified**, not 100%. Reran on his exact code: 85 of 600 stopping points weren't real optima, every escape a 3-cycle, none a transposition — his numbers and mine agreed to the twelfth decimal on the shared lattice quantities. The supremum (0.154322) survived certification against the full neighborhood; what didn't survive was the sentence "two independent neighborhoods, both plateau" — the second one was only half-enumerated, and I'd written it as if it were whole.

> **His rule, adopted: a value produced by a search reports the fraction of the space it was actually certified against.** A converged value isn't a measurement — it's a claim of non-existence, and the container for a non-existence claim is the space actually enumerated against it.

Also found: `plafond_beta`'s own certificate (`‖grad J‖`) had been sitting on screen, four orders of magnitude loose, read as a diagnostic instead of the certificate it was — worse than missing a container: having one and not recognizing it.

His objection to my REINFORCE grid landed too: batch size moves variance and sample volume together (64×20,000 is 1,280,000 draws against 8×20,000's 160,000), so every cell was consistent with either story. A proposed iso-sample control — batch 8×20,000, 16×10,000, 64×2,500, all at 160,000 draws — turned out to unmatch a third quantity the moment it matched a second: batch 64×2,500 runs 2,500 gradient *updates* against batch 8×20,000's 20,000. **Draws, updates, and batch size are linked by one identity, and there are only two axes to move on** — matching samples unmatches updates, matching updates unmatches samples. Flagged before the numbers arrived, because it constrains what either reading of the result can claim; resolved the following round.

### Twenty-first round — his rule turned on the file he'd asked me to write

He refused my one-sentence takeaway from four straight audits ("the target of an audit is what it's least likely to catch") and proposed a sharper one: choose the claim whose verification forces the widest re-enumeration. I coded all twenty-nine dead entries in the notebook against both readings (`anatomie_des_audits.py`) rather than argue from four data points:

| | n | targeted | instrument (untargeted default) | proof already on disk |
|---|---|---|---|---|
| whole notebook | 29 | 18 (62%) | 11 (38%) | 9 (31%) |
| before 14/08 | 17 | 13 (76%) | 1 (6%) | 2 (12%) |
| from 14/08 on | 12 | 5 (42%) | **10 (83%)** | **7 (58%)** |

Targeting doesn't break (Fisher p=0.119) — what breaks is the *object*: 6% to 83% untargeted-instrument deaths, p=3.3×10⁻⁵. Early on, false claims are about the world and a targeted check kills them; later, what remains has already survived targeted checks, and what dies is **the instrument** — a reservoir cap, an `n_restarts=24`, a column that isn't a factor, a half-enumerated neighborhood. Nobody has a hypothesis about a default argument, so nobody can target it. And the file I'd written *for him*, the same round I adopted his rule, itself used only the half of the neighborhood that never finds anything (351 of 3,276 moves) — his own rule, turned on the artifact meant to satisfy it.

Rerun against the full neighborhood: all four previously-published maxima survived unchanged; what didn't survive was the never-printed certification fraction (30–41% of restarts were false stops, all via 3-cycle, zero via transposition).

Also settled without an optimizer at all: a proposed leave-one-out baseline for reducing REINFORCE's gradient variance. Measuring a variance reduction *through a training loop* would have been the fifth instance of exactly the mistake that killed §1.12 and `plafond_beta`. First, two checks nobody had run: the analytic gradient against autograd agreed to 2.5×10⁻²⁰, 1.7×10⁻¹⁷, and 8.5×10⁻¹⁸ at three points along training, and the sampled REINFORCE estimator is **unbiased** for the exact gradient — the gap sits at 0.27 to 1.52 Monte-Carlo standard errors across thirty-six cells. The story "it isn't the same objective" was dead before it was written.

Then the measurement, at fixed θ, 20,000 replicated batches, zero updates, total variance under batch size 8:

| point | no baseline | EMA (canonical) | LOO | optimal constant |
|---|---|---|---|---|
| θ init | 8.78×10⁻³ | 8.47×10⁻³ | **1.02×10⁻²** | 8.64×10⁻³ |
| θ mid-ascent | 4.59×10⁻³ | 6.21×10⁻³ | **6.45×10⁻³** | **2.86×10⁻³** |
| θ at the trap | 9.23×10⁻³ | 9.21×10⁻³ | **9.41×10⁻³** | **5.12×10⁻³** |

**LOO raises variance by 2–20% at every point, never lowers it** — arithmetic, not an accident: at n=8 its own advantage has variance `p(1−p)·n/(n−1)`, 8/7× the centered one, and it zeroes out entirely in 39–73% of batches (every one where all eight rewards tie), cutting effective updates along with variance. My own batch-size axis divides variance by 7.9–8.1× over the same range. What actually works, and isn't what the textbook prescribes: the variance-**minimizing** constant `b* = E[R‖score‖²]/E[‖score‖²]`, which LOO doesn't estimate at all since the score's magnitude correlates with the reward — dividing variance by 2.26 mid-ascent and 1.84 at the trap.

**The iso-sample control landed, and it revealed an inverted U rather than settling the question either way.** At matched draws (160,000), batch 8 nominally wins 3/12 against 0/12 — but Fisher p=0.22, the row doesn't discriminate, and batch 8 also carries eight times the updates on that row, exactly the confound flagged in advance. The contrast that *does* discriminate is at matched **updates** (20,000 steps both): batch 8 gives 3/12, batch 64 gives 11/12, **Fisher p=0.0028** — at equal updates, less variance is strictly better, and noise isn't doing the work his account needed. But that can't be the whole story either, since exact ascent has zero variance and reaches a bijection 0/12. So the escape moment itself was isolated: twelve trapped states reached once by exact ascent, cloned parameter-for-parameter, then released under different dynamics.

| from the same trap, 20,000 steps | E[R] after | bijections | escaped | \|Δθ\| |
|---|---|---|---|---|
| exact ascent continued | 0.94753 | 0/12 | 0/12 | 21.08 |
| REINFORCE batch 8, EMA | 0.96975 | **8/12** | 9/12 | 360.70 |
| REINFORCE batch 8, LOO | 0.96188 | 5/12 | 8/12 | 361.94 |

**Exact ascent 0/12 against REINFORCE batch 8 at 8/12, Fisher p=0.0013, from a bit-identical starting state — and batch 8 is the arm that does worst from random init.** The same estimator that escapes best converges worst: an inverted U in variance at fixed updates, confirmed by two independent designs — zero variance traps, batch-64 variance escapes and settles, batch-8 variance escapes and doesn't settle. His account and mine were each the right half of the same curve; neither of us said "peak" because two points can't show one.

A second, much larger finding fell out of building a proper audit denominator — a register of *checks*, not just of findings, since neither of us had one and couldn't compute a real yield without it. Every numeric default argument in the repository that can bound a search, a budget, or a claim, enumerated mechanically by AST rather than chosen: 23 sites, 12 real bounds, each resolved one by one. Most came back clean (a worst-case search 0/24 exhausted, a tolerance check with 14 orders of separation); two known defects reproduced (the reservoir cap, the 24-restart budget); and one was new — a message-distinctness bound published from 20 restarts moved by **+15.5% at R=27, +2.3% at R=25, +4.8% at R=23** once rerun at 400.

**And resolving that site turned up something worse than an undersized budget: the published numbers don't come from the seed or the restart count at all.** One generator threads through all five distinctness floors in sequence, so the *order* the floors are computed in is a hidden argument no one had named. Running them in the script's own order reproduces every published bound to the last digit; running the same five floors in reverse order, same seed, same budget:

| floor | script order | reverse order | published |
|---|---|---|---|
| 27 | 0.052616 | **0.024298** | 0.052616 |
| 24 | 0.185005 | **0.202645** | 0.185005 |

**The R=27 bound moves by a factor of 2.17 with nothing changed but the order of a loop.** The container resolves — a prior round's rule about publishing against a file passes cleanly — and the number still isn't recoverable from it. **A seed names a stream, not a state.** Checked how far that reaches: nine test-3 scripts thread a single generator through four or more published computations, one of them nineteen — the exact script behind the 0.1443 worst-case bound from many rounds earlier, meaning part of the gap I'd attributed entirely to restart budget back then was this instead. The fix costs one line per published computation: its own generator, or its stream position printed next to the number.

### Twenty-second through twenty-fourth rounds — his own quantum claim retracted, a discriminator confirmed to the byte, and a ceiling that hid another ceiling

Dipankar retracted his own "seven highest optima are all multiples of U" and turned his AST-based generator-discriminator on his own repository: 35 of 409 files where one generator thread served several consumers, and a seventh optimum on his own reordered stream landing 0.46U off the lattice — confirmed independently from arithmetic alone (0.155054/U = 8.539880, nearest multiple −0.460U, matching his reported 4.60×10⁻¹) before he shared any code. And his published supremum, 0.154321642873, turned out to already be sitting in my own results from a completely unrelated script that same round — a different objective's search (`realisabilite_treillis.py`'s inflation lattice, seed 2026, 600 restarts, full-neighborhood continuation) landed on the identical value to thirteen digits. Two different codebases, two different generators, two different neighborhood constructions, the same supremum — the strongest evidence either of us produced that this number is a property of the objective, found by accident rather than deliberate cross-check.

A refinement to his own reached-value/mean-value split, not a correction: "reached" turned out to cover two different things. A value reached as a **max over restarts** moves monotonically with restart count — more restarts, weakly higher max, bounded by watching convergence. A value reached by a **single deterministic climb from a threaded stream position** doesn't move monotonically with anything — more compute just makes it a *different* reached value, not a more trustworthy one, and it can only be bounded by re-deriving from a clean stream. And a fourth instrument question, distinct from the first three (which all presuppose randomness or search): **is a column a value fixed before the run, or read off the run's own output?** It had already cost two rounds independently — a column stratified by the run's own rounded reward (§1.19), and a "robustness check" that was algebraically identical to the number it was supposedly checking (§1.21) — neither involving a generator, a neighborhood, or a restart budget at all; both checkable by reading a formula two columns apart.

His discriminator, replayed word for word against my repository, reproduced hashes to the last digit: two files I'd flagged as unseeded turned out to **reinitialize** the global `random` stream rather than thread a generator — a distinction his AST syntax genuinely couldn't see, verified by injecting draws and checking whether the downstream state moved.

Eight of his scripts arrived in the clear. Nothing taken on faith: every structural claim re-derived from my own source before being believed. His unpaired bisection replayed the real draw order of `certificat_deux_agents.py` — three permutations, the K-mixing loop, twenty-four phase-1 draws — in pure numpy, no torch: the first four seeds at each noise level landed exactly on his own (25970514, 826555961, 763435854 at noise=0.01…), and the four levels shared zero seeds across all six pairs. My own repository inventory, checked independently rather than taken from his: `git ls-files` gives 86 tracked files, 55 under `src/`, exactly one JSON — his three numbers, exact. His audit of the sixty paths cited across this notebook, rerun with my own method: 40 resolve, 20 don't, 17 of those gitignored by documented rule, 3 his own scripts cited back to him — zero left unexplained, same as his own audit of himself.

The generator-derivation flaw he'd found existed **five times in my own code, not once**: `torch.Generator().manual_seed(int(generateur.integers(1 << 30)))`, found by grep after his discovery, present in all four constructors (`EmetteurTabulaire`, `EmetteurFactorise`, `EmetteurStructure`, `Recepteur`). His own censor would flag all five as unseeded for the same reason it mis-flagged his.

A fifth instrument question, distinct from the fourth: not two columns silently computing the same fact, but **a metric standing in for a target it can silently diverge from without anything on the page showing it.** His own line-distance proxy (cheap, well-defined) returns `None` exactly where a generator crosses a function boundary as an argument — the shape most likely to accumulate hidden consumers, and the shape the proxy is blind to. It had already cost me once, with no randomness involved at all: `plafond_beta` was a training loop's convergence criterion standing in for a true supremum, silently wrong exactly where nobody thought to extend the horizon past the 139 steps where the criterion happened to fire. The operational test that generalizes: does extending the budget past wherever the proxy stopped change the answer? For `plafond_beta`, three more decades of steps moved it four orders of magnitude; for his walk, the `None`s are exactly the cases no finite extension could ever reach. Checked whether RDTRL has any of the two-reset pattern behind his own safest files (it doesn't — `test1` and `test2` reset `random`/`np.random` once per seed, before any policy network is constructed, and the construction that follows draws only from a separate `torch.Generator`, confirmed by reading `PolitiqueGRU.__init__` rather than by building the discriminator against it) and whether any test-3 script resets mid-file (none do — all nine flagged scripts thread their generator for the file's entire lifetime).

His closing question, tested rather than read: does phase 3 of this same file — the §6.5 table, source of the 5% figure two rounds earlier — inherit stream position from phases 1 and 2? Advanced the stream by 72 draws with zero ascent steps run (construction alone draws one integer each, confirmed), then ran the real 30 phase-3 ascents on that position and on a fresh generator:

| parametrization | bijections, coupled/independent | E[R], coupled/independent | matched concentration |
|---|---|---|---|
| tabular | 0/10 — 1/10 | 0.9185 — 0.9481 | 0.1226±0.033 — 0.1234±0.031 |
| factorized | 0/10 — 0/10 | 0.7888 — 0.7814 | 0.1201±0.036 — 0.1321±0.037 |
| structured | 0/10 — 0/10 | 0.8518 — 0.8518 (identical) | 0.4301±0.109 — 0.4243±0.086 |

**The table doesn't move** — every gap sits inside its own printed standard deviation at n=10. A real writing defect, not a live threat to §1.28. Fixed anyway: a fresh `default_rng` before the phase-3 loop costs one line. And `plafond_beta` — the ceiling constant every `ratio_to_ceiling` in a whole section divided by — was itself a loop stopped early: extended from 139 to 20,000 steps, `E[R]` climbed from 0.9999230227 to **0.9999999990**, nine decimals from 1, not five, gradient falling from 2.86×10⁻⁵ at step 0 to 7.17×10⁻¹¹ at step 20,000. **A ratio to a ceiling had been printed as `1.000016` — mathematically impossible — and stayed on screen for a week before anyone noticed the number refuted its own definition.**

A separate check, on a deliberately different construction (the receiver frozen on the canonical code at a fixed logit force, not `plafond_beta`'s own two-free-agent setup), traced the approach curve at more checkpoints to settle whether it's geometric or logarithmic:

| step | E[R] | gap = 1−E[R] | ‖grad‖ max |
|---|---|---|---|
| 50 | 0.9027384 | 9.726×10⁻² | 2.668×10⁻³ |
| 200 | 0.9854066 | 1.459×10⁻² | 1.821×10⁻⁴ |
| 2,000 | 0.9911847 | 8.815×10⁻³ | 4.744×10⁻⁶ |
| 20,000 | 0.9913534 | 8.647×10⁻³ | 4.395×10⁻¹⁰ |

Neither, cleanly: the gradient decelerates far from the optimum (roughly 10× slower per step between steps 50–100 than 100–200), then turns ordinary geometric once inside the basin — four to five orders of magnitude per doubling of steps past step 2,000, even as `E[R]` itself has already flattened. The practical upshot for the original `plafond_beta` undercount: extending the budget *would* have caught it, since past ~2,000 steps the gradient is already six orders below its start and still falling fast — the failure was stopping at 139 steps, deep in the decelerating regime, not a tail no finite extension could ever escape. And this construction's own asymptote (0.99135, not 1.0 — it's a different setup from `plafond_beta`'s) is fully explained by the fixed force's own softmax peak: `e⁸/(e⁸+26) = 0.991353` to six digits — a second unnamed constant playing exactly `plafond_beta`'s role, on a different construction.

> **Rule 11: any quantity whose name implies an arithmetic bound is checked against that bound where it's computed.** `ratio_to_ceiling ≤ 1`. A probability in [0,1]. A correlation in [−1,1]. One assertion, fails closed.

---

## 🧨 Part two: three things called "collision," and only one of them is

Everything above measured a fixed population of already-trained runs. Round twenty-five turned inward, into a single collision, and the ground shifted.

### Twenty-fifth and twenty-sixth rounds — the structure table wasn't a fixed point, it was a collision count

Two sentences in one of my own tables contradicted each other: a 0.0296 gap on `tabulaire` was framed as ordinary per-seed noise; a 1e-4 agreement on `structure` was framed as a one-in-fifty-six coincidence. No single standard deviation makes both sentences ordinary. Printing the raw ten values in each arm and grouping by **collision count** resolved it — not a fixed point (per-arm sd 0.047 and 0.050, close to a Gaussian guess), but a quantization:

| R (=27−collisions) | target R/27 | observed mean | gap | n |
|---|---|---|---|---|
| 25 | 0.9259259 | 0.9258237 | 1.02×10⁻⁴ | 2 |
| 23 | 0.8518519 | 0.8517537 | 9.82×10⁻⁵ | 5 |
| 22 | 0.8148148 | 0.8147363 | 7.85×10⁻⁵ | 3 |
| 21 | 0.7777778 | 0.7777006 | 7.72×10⁻⁵ | 2 |
| 20 | 0.7407407 | 0.7406708 | 6.99×10⁻⁵ | 1 |

Five of six classes saturate `(27−collisions)/27` to about 1×10⁻⁴ — `E[R]` isn't continuous here, it's a near-deterministic function of an integer, undershooting by the same order of magnitude as `plafond_beta`'s own convergence residual two rounds earlier: 3,000 steps isn't quite enough to fully sharpen the softmax, so the soft reward sits a hair under the hard bound its own argmax has already reached. The sixth class didn't fit, and wasn't smoothed over: one run's argmax reads 24 distinct messages (collisions=3) but its `E[R]=0.8517668` sits at the collisions=4 target, a full `1/27` below where its own argmax says it should be — mid-transition between two collision classes, decoder-wise. **Quantization does almost all of the work, but "R/27 exactly" is the idealization; a convergence lag can still miss the class its own collision count predicts by a full quantum.** Checked separately whether 23 was some kind of ceiling: R=25 shows up in the very same ten-run sample, and a supervised-fitting check on data already on disk showed the compositional code converges to residual 3.5×10⁻⁴ in 2,367 steps while random bijections never get close (gap ≈0.98) — so there's no combinatorial ceiling anywhere below 27; every R below the maximum in this phase is a fact about where Adam stalls from random init, not about what the 81 parameters can represent. Per-seed standard deviation ~1.3. The "coincidence" dissolved into an ordinary event under the right generative model, one neither of us had started with — verified independently: shipped mean 0.85175642736 against a freshly-run independent mean 0.8517558054, gap 6.22×10⁻⁷ against a predicted 3.7×10⁻³, off by a factor of 5,955. Dipankar's own reconvolution of the pooled collision-count law, run ten deep, predicted `P(tie) = 0.06829`, 1 in 14.64 — matched to four decimals.

Whether the ε-band survives with a genuinely free receiver (not the frozen scaffold used to measure it two rounds earlier) got a clean answer too: `tabulaire` and `factorise` in phase 3 are already two fully free agents. Fifteen fresh seeds each: `tabulaire` bands at mean 6.1049×10⁻⁵ (sd 7.66×10⁻⁷, 13 in-band), `factorise` at 6.3709×10⁻⁵ (sd 1.46×10⁻⁶, all 15 in-band) — **the band survives with a fully free receiver**, so ε is a real property of finite-step convergence under this objective, not an artifact of any frozen scaffold. But the constant isn't universal: `structure`'s band (1.125×10⁻⁴) sits at roughly double `tabulaire`'s and `factorise`'s, consistent with (not yet proven by) a mechanism where `structure`'s 81 parameters, shared across all 27 referents, have every gradient step on one referent's logits perturb the other 26 through the same weights — slowing the last stretch of sharpening relative to a fully free parametrization. And checking the two obviously-off `tabulaire` runs against this same reclassification test found **two more instances of the identical defect, without looking for them**: both land cleanly back in-band once relabeled by their own soft reward — the same signature, a different construction, three for three now. `corr(eps, R)` on the fresh free-free data came back unstable across constructions — tabulaire −0.43, factorise −0.06, against structure's own +0.52 — with a standard error around 0.27–0.29 at this sample size; none of the three would survive being called significant alone, demoting the sign disagreement from a puzzle to an underpowered measurement rather than resolving it.

### Twenty-seventh round — my dose story was false at the premise

`factorise`'s "partial sharing" turned out to be **39 free parameters per referent** — zero sharing, just a bigger local parametrization. The real axis was binary (shared vs not), not a dose from none to total. Two "reclassified into the band" anomalies from the prior round weren't: re-checked against the actual 13 values, one sat below the true minimum, one above the true maximum — P(both outside, on opposite sides) = 1/105. His proposed decisive test — run `tabulaire` at the step count his exponential model predicted would match `structure`'s mean gap (2,811 steps) — **failed on the predicted number**: mean 6.982×10⁻⁵ (CV 1.24%), barely moved from the 3,000-step value of 6.105×10⁻⁵, nowhere near structure's 1.125×10⁻⁴ — the single global-τ extrapolation crossed out of the geometric regime the same way `plafond_beta`'s own convergence curve had two rounds earlier, a lesson both of us had already written down and both walked past applying here. A five-point scan found the real crossing near 2,190 steps instead, and running there succeeded: mean 1.1232×10⁻⁴ against structure's published 1.125×10⁻⁴, **means equal to four digits, CV of 1.45% against structure's 15.91% — a factor of 11** — the dispersion is a property of the construction, not of the gap's magnitude. A deliberately broken control (`EmetteurMasque`: same 27×27 tensor, only 3 free columns per row, but *no* sharing between rows) confirmed the control's own failure mode rather than the hypothesis: freezing 24 of 27 logits near zero froze a representability floor, not just a parameter count, collapsing the intended contrast by two to three orders of magnitude.

One clean check, never posed before that round: does the entropy term itself fix a nonzero equilibrium, rather than undertraining? Solved in closed form for a near-saturated line — `p* = e^(c/β)/(26+e^(c/β))`, residual `26·e⁻⁵⁰ ≈ 5.0×10⁻²¹` at β=0.02 — sixteen orders below anything measured that round. The premise held: the gap is a training lag, not a hidden fixed point of the regularized objective.

### Twenty-eighth round — forty points were fifteen points measured three times

Asked for the pooled correlation centered by step-group before writing another sentence about a sign disagreement (my −0.43 against his +0.52). Computed: r=−0.3007, df=36, p=0.067 — not significant, but before believing the df, checked what the three groups actually were. **The fifteen R-values are identical between 3,000 and 2,811 steps, seed by seed** — all three conditions share one `default_rng(999)`, only the step count changes. Not forty independent points; fifteen seeds measured three times. True n ≈ 13–15. Correctly independent test: r=−0.3908, same sign as my original, on a defensible n. The sign disagreement was never resolved by pooling — there had never been more independent information than the first measurement carried.

And a "plateau" I'd called a sub-optimal critical-point signature was a **denominator bug**: dividing by 26 instead of 25 during a window where the argmax misreported R by one.

| steps | E[R] | ε against argmax's R | ε against the true, fixed R=25 |
|---|---|---|---|
| 2,200 | 0.9258252 | +1.09×10⁻⁴ | +1.09×10⁻⁴ |
| 2,811 | 0.9258635 | +3.85×10⁻² | +6.75×10⁻⁵ |
| 5,000 | 0.9259097 | +3.85×10⁻² | +1.75×10⁻⁵ |
| 20,000 | 0.9259259073 | +2.01×10⁻⁸ | +2.01×10⁻⁸ |
| 40,000 | 0.9259259243 | +1.72×10⁻⁹ | +1.72×10⁻⁹ |

**Against the true, fixed R, ε decreases smoothly and monotonically across the entire trajectory — there is no plateau anywhere**, including through a second brief flicker at 17,000–19,000 steps, checked point by point and found equally clean. What's genuinely unresolved is sharper: two losing referents never fix a stable second choice. Referent 0's own runner-up drifts across six different messages over six checkpoints — 0, 11, 7, 3, 16, 8 — with margins oscillating between microscopic (5.06×10⁻⁶, 7.28×10⁻⁶) and merely small (up to 8.96×10⁻⁴), never settling. **A referent that has already lost its message competition receives no reward gradient on its second choice at all** — nothing among 26 losing options reports anything, so nothing pins which one it points to. This is exactly what the earlier R=25↔26 flicker was: whenever the wandering runner-up happens to land on a message nobody else claims, the local collision vanishes and the global count reads 26; when it drifts back onto a claimed message, it reads 25 again — not non-convergence, a genuinely flat direction that ordinary exact ascent has no force to resolve.

### Twenty-ninth round — two regimes hiding under one label, "loser"

Dipankar reconstructed the objective independently in numpy (agreement to nine decimals) and showed the reward gradient on a "losing" referent is not uniformly negligible — it varies with β, and a receiver's 50/50 split on a contested message is a real, costly split, not a free zero. Rechecked on my own trained state before believing it: two of four losing referents had gradients indistinguishable from winners; the other two were 30,000–60,000× smaller.

**First reading, wrong — corrected by checking who actually collides with whom.** I'd paired the two small-gradient referents against each other as if contesting one another. They don't. Referent 0 (uniform) shares a message with referent 24 (fully engaged); referent 4 (uniform) shares a message with referent 10 (fully engaged), and the receiver has already fully resolved both:

```
message  7 :  R[7,0]  = 1.27e-11    R[7,24] = 1.00000000
message 16 :  R[16,4] = 1.39e-11    R[16,10] = 1.00000000
```

**Not a 50/50 duopoly. A confident referent holding all the receiver's attention, and a hesitant one holding none.** Referent 24's full-strength gradient isn't pressure to abandon message 7 — it's the generic entropy-versus-reward sharpening every fully-engaged line carries, winning or not. Referent 0's gradient isn't tiny because it's tied with 24; it's tiny because, for each of its 27 options, the message is either free (entropy-only, no reward signal) or already fully captured by a confident referent. **It was excluded from the reward function on the whole line, before choosing anything.**

A β-sweep, arriving as the decisive test against my own earlier hypothesis (that entropy alone drives the gap): bijection rate at β ∈ {0, 0.005, 0.02}, ten seeds, exact ascent versus REINFORCE. **The gap doesn't move** — 0/10 exact ascent at all three β, 9–10/10 REINFORCE at all three. Dipankar had posed this as his own decisive fork; it refuted his own hypothesis, not mine.

### Thirtieth round — the majority are ties; my sample could only ever see walls

Reproduced my 92%/5% gap independently (different generator, same E[R]), then isolated a factor my table hadn't controlled: exact ascent ran at lr=0.05, REINFORCE at lr=0.01. At matched lr=0.01, exact ascent reaches **5/30**, not 0/30. And a census of 50 collisions over 30 seeds found **42 ties (0.1–0.9 split on both sides) against only 8 walls** — the one seed my whole Adam-epsilon investigation would later be built on was not representative.

Reproduced independently, my own code, my own seeds: lr control 3/30 vs his 5/30, same order; tie/wall census 41/11 (79%/21%), close to his 84%/16%; and an "engage then evacuate" pattern — a referent peaking at 93% confidence by step 100, then collapsing back to uniform by step 1,000 — confirmed on his eight seeds too. Why my one sample gave 2 walls out of 2: **both members of a tie are each individually engaged at S≈0.9999999661** — stable, no argmax flicker. A wall's losing member sits at maximum entropy and its argmax reads floating noise among 27 tied options. **I'd found my one seed by searching for label instability, and only walls produce that instability.** The detector could only ever see the minority class.

A tie doesn't look like a slow convergence, either: tracked one pair (referents 23/25, message 13) from 20,000 to 300,000 steps, the gap to exactly 0.5 read +2.27e-4, −8.81e-6, +6.16e-7, +3.49e-8, **+3.09e-5** — non-monotone, four orders of tightening followed by three orders of drift back, alternating sign. Not an asymmetry resolving; an oscillation around the symmetric point, consistent with a genuine fixed point protected by the equivariance argument from article 3, not a slow race toward a winner.

### Thirty-first round — a wall is not a collision, and the "free" split had a real draw hiding in a network

Dipankar perturbed a tie instead of watching it: `R += eps` on one member, `−= eps` on the other, then 20,000 more steps. Reproduced on my own tie (referents 23/25, message 13), same protocol:

| eps | split right after perturbation | R[13,23] after +20,000 |
|---|---|---|
| 1.0 | 0.880845 | 0.500000 |
| 3.0 | 0.997528 | 0.500000 |
| 8.0 | 1.000000 | 0.500000 |
| 12.0 | 1.000000 | **1.000000** |

Matches his numbers to five decimals on the recovering cases and reproduces the same eps=8/eps=12 boundary on a different seed, different codebase — full receiver capture at eps=8 still returns to the tie; eps=12 sticks.

**The decisive cross-tab**: sender confidence against receiver assignment, 30 seeds, 52 collisions. **Perfect separation, zero exceptions**: 41 ties have S=1.000000 on both sides; 11 walls have their low member at maximum entropy (0.03707–0.03717). **A wall is not a collision.** It's a referent that never committed to anything, filed under whatever floating noise its argmax reads. Real collisions are 100% ties — not 84%, not 79%.

And the "free" message wasn't free the way I'd described it. Two referents I'd called "confidently committed elsewhere" (18, and separately 25) were actually split almost exactly 50/50 between two of their own uncontested messages — referent 18 between messages 0 and 8, referent 25 the same way between messages 1 and 14. Referent 18's own history, checked step by step rather than assumed stable from one late reading:

| step | p(message 8) | p(message 0) |
|---|---|---|
| 5,000 | 0.500000 | 0.499995 |
| 10,000 | 0.500000 | 0.500000 |
| 15,000 | 0.501502 | 0.498498 |
| 20,000 | 0.500000 | 0.500000 |
| 40,000 | 0.500521 | 0.499479 |

Split almost exactly 50/50 at every checkpoint from step 5,000 on, drifting by at most a couple tenths of a percent and always snapping back. **Neither was "half-claiming" another referent's message — each is a line split between two of its OWN options, with no real opponent on either.** Checked whether it costs anything: `R[0,18] = R[8,18] = 1.000000` either way — the receiver decodes it perfectly regardless, so the split buys `ln 2` nats of entropy for zero reward cost. It looked like the objective's genuine, permanent optimum for an uncontested referent with two free options.

A doubt raised before believing the recovery basin itself: both perturbations so far touched only the *receiver*, leaving both senders symmetric — the "recovery" could just be the receiver mechanically catching up to two untouched senders. Tested by perturbing the **sender** instead (referent 25's own logit, message 13, receiver untouched):

| eps | S[25,13] right after | S[25,13] after +20k | S[23,13] after | R[13,23] after |
|---|---|---|---|---|
| 1.0 | 1.000000 | 1.000000 | 1.000000 | 0.501507 |
| 8.0 | 0.999898 | 1.000000 | 1.000000 | 0.500000 |
| 12.0 | 0.994448 | 1.000000 | 1.000000 | 0.500000 |
| 20.0 | 0.056683 | **0.037049 (evacuated)** | 1.000000 | 1.000000 |

**Twelve units of sender-side perturbation — enough to permanently break the receiver-side version — fully recovers here; it takes twenty before referent 25 gets knocked out, and when it does it collapses straight to the wall state, not to a third message.** The basin is real, and wider on the sender side — a sender logit faces 26 competitors, a receiver logit inside a tie faces only 1.

A counter-example search that came back empty, reported anyway: my classifier silently skips any collision with three or more referents (`len(refs) != 2: continue`). Checked across all 30 seeds — zero three-way collisions, zero larger — the population the blind spot could have missed simply wasn't there. And a closing question, tested rather than argued: computed the full 27-direction gradient of referent 0's uniform row, verified against the closed form (agreement to 2×10⁻²⁸, fifteen orders below the signal, not numerical noise) — **the two free messages carry the two largest positive values on the entire row** (+5.41×10⁻¹³, +5.29×10⁻¹³), far ahead of the 25 taken ones. **Neither a genuine second attractor nor a total absence of pull** — a real, correctly-directed gradient, five orders of magnitude below an engaged line's. Given everything above, "caught in a network" was also the wrong description: referent 0 isn't facing a contested tie on message 0, it's facing a free split that disputes nothing with it. The wall may be blocked by nothing more than its own gradient, too small to move at any budget tested so far. Not reproduced: his timing census (14 of 30 seeds resolve a hard collision, the one bijection settling at step 271 of 20,000) — reported as such rather than assumed.

### Thirty-second round — the split and the wall are one occupation, read from two ends

"Free message" and `R[0,18]=1.000000` can't both be true, unqualified, at the same time: the tiny gradient I'd found isn't an independent preference, it's the residue of what a receiver already saturated elsewhere leaves behind. Dipankar's decisive test: push the referent 0 itself, sender side, onto message 0, and see whether referent 18 re-splits (my "free optimum" reading) or fully commits elsewhere (his "lease on a vacancy" reading).

**Neither reading is true across the whole range — a sharp threshold between eps=23 and eps=24:**

```
eps <= 23 : referent 0 always returns to uniform, the 18-split intact
eps = 24  : full flip — 18 commits to message 8 alone, 0 keeps message 0
```

**Both readings are true, on opposite sides of one threshold.** The split depends on occupancy, exactly as argued, *and* it's protected by a real basin, the same shape as a genuine tie. I retracted "genuine, permanent optimum" — it was an optimum conditional on referent 0 staying asleep, which is exactly what the experiment was built to reveal. Deliberately unbalanced the split to 30/70 and reran 40,000 steps with nothing else changed: it returns to 50.07/49.93. **β is the only active force in this regime, and it forces the exact return to the symmetric point** — the same mechanism that holds a genuinely contested tie. His two minor corrections both checked out exactly: the four wall values overshooting 1/27 slightly is inevitable arithmetic (the max of 27 terms summing to 1), and the split's own entropy deficit — 5.43×10⁻⁷ nats under `ln 2` — matched to the digit.

Pushed further on my own initiative, since the round "felt finished" and that itself felt suspicious, two more checks. Does the eps=23/24 threshold hold as more than a 40,000-step snapshot? Extended to 270,000 cumulative steps on both sides: **stable throughout**, flip stays flip, return stays return. Is a split occupant (referent 18) as hard to evict as a true solo winner? Never compared before: pushed referent 0 onto referent 1's message instead (an exclusive owner, full reward genuinely at stake) — **at eps=100, four times the threshold that evicts referent 18, referent 1 doesn't move at all**, and the receiver never credits referent 0 there either (`R[11,0]=0` at every eps tested). A solo winner defending real reward is categorically harder to dislodge than a split occupant defending nothing extra.

**And a wasted engagement (S=1.0 on a message paying zero) stays stable for 270,000 more steps — which I didn't expect and don't fully believe yet.** By the same entropy argument from this round, it should be strictly dominated: returning to uniform costs nothing in reward (already zero) and buys `ln 27` of entropy. It doesn't return. Stated as a hypothesis rather than a finding: the perturbation (eps up to 100) likely saturated the row past float64's representable precision near 1, and the gradient autograd returns there may underflow to exactly zero rather than merely being small — in which case this isn't a second genuine trap in the objective, it's a numerical artifact of my own push, to be rechecked with a much smaller perturbation before believing it.

### Thirty-third round — it wasn't float64, it was Adam's epsilon

The true cause of a stalled engagement surviving 270,000 further steps: **`adam_eps`**, defaulting to 1e-8. Below that floor the update degenerates to `lr·g/eps`, blind to gradient scale. His entropy-gradient-alone table reproduced to the digit — 2.56×10⁻¹² down to 7.16×10⁻⁴⁴ as the logit gap runs from 26 to 100 — and confirmed the artifact independently of any training loop: the logit-space margin at three checkpoints under eps=100 is **strictly frozen at 96.741751**, no drift across 270,000 further steps, exactly as a gradient that small predicts. His prediction on a companion referent: exact — `adam_eps=1e-10` instead of default, and the stalled engagement vanishes entirely (`S[0].max()=0.037037`, exactly 1/27). His companion prediction — that the 23/24 threshold would hold under the same change, proving real dynamics — **failed**: under `adam_eps=1e-10`, eps=23 (previously "returns") also flipped. Pushed further:

```
                eps=18   eps=20   eps=23
adam_eps=1e-8   returns  returns  returns
adam_eps=1e-10  returns  returns  FLIPS
adam_eps=1e-12  returns  FLIPS    FLIPS
adam_eps=1e-14  returns  FLIPS    FLIPS
adam_eps=1e-16  returns  FLIPS    FLIPS
```

The threshold **converges**, stable from 1e-12 to 1e-16, somewhere between eps=18 and eps=20. **Both partly right**: the threshold is a real dynamical property (it converges to an `adam_eps`-independent value), but the published 23/24 wasn't that value — inflated four to six logit units by the optimizer's own default floor.

### Thirty-fourth and thirty-fifth rounds — none of the simple candidates fit; it's a latch, not a race

Testing whether `sqrt(v)` for a specific line crosses `adam_eps` exactly at the threshold: **none of three named candidates fit cleanly.** The perturbed referent's own `sqrt(v)` stays *above* `adam_eps` at every setting, including where nothing transfers — and it isn't even a fixed quantity, dropping four orders of magnitude between settings that return and settings that flip, so it can't be a fixed value compared to a moving threshold. The evicted occupant's `sqrt(v)` sits five orders *above* `adam_eps` at 1e-8 and 1e-10 alike — not frozen at all at those settings — and the system still returns; ruled out as a single bottleneck. The receiver's own credit line comes closest: `sqrt(v)` stays stable (2 to 5×10⁻¹³) while `adam_eps` sweeps around it, and a real crossing does happen — but between 1e-12 and 1e-14, one decade after the true behavioral flip (between 1e-10 and 1e-12). Reported as unresolved rather than forced to fit: more likely a coupled transition across all three lines sharing one `adam_eps`, not a race between independent thresholds.

Pushed further after being told to keep looking: a step-by-step trace of all three candidates through the transition, not just their endpoints:

| step | `S[0,0]`, 1e-10 | `R[0,0]`, 1e-10 | `sqrt(v)_R`, 1e-10 | `S[0,0]`, 1e-12 | `R[0,0]`, 1e-12 | `sqrt(v)_R`, 1e-12 |
|---|---|---|---|---|---|---|
| 1 | 1.0000 | 4.45×10⁻¹⁰ | 2.82×10⁻¹³ | 1.0000 | 4.82×10⁻¹⁰ | 2.82×10⁻¹³ |
| 50 | 1.0000 | 8.62×10⁻¹⁰ | 3.05×10⁻¹² | 1.0000 | **2.02×10⁻⁷** | **3.16×10⁻¹⁰** |
| 100 | 0.9401 | 4.25×10⁻⁹ | 1.18×10⁻¹¹ | 0.7196 | 2.58×10⁻³ | 1.93×10⁻⁶ |
| 200 | 0.0275 | 3.38×10⁻⁹ | 2.51×10⁻¹¹ | 0.9962 | 9.99×10⁻¹ | 2.31×10⁻⁴ |
| 500 | 0.0370 | 9.26×10⁻¹⁰ | 2.20×10⁻¹¹ | 0.9990 | 9.997×10⁻¹ | 1.99×10⁻⁴ |

The two settings are nearly identical through step 50, then split apart entirely by step 200. At step 50, the receiver's credit to the pushed referent is **235× larger** under `adam_eps=1e-12` than `1e-10` (2.02×10⁻⁷ against 8.62×10⁻¹⁰), and its own `sqrt(v)` is **104× larger** — while the sender's confidence is already saturated at 1.0 in *both* arms and the defending line hasn't moved yet. By step 200 the two arms have already committed to opposite outcomes: the `1e-10` arm's referent 0 has collapsed back to near-uniform (0.0275) while the `1e-12` arm's has locked in (0.9962). **The bifurcation happens on the receiver's own parameter, in the first fifty steps, before any sender line diverges** — a race between how fast the pushed referent's artificially-inflated confidence decays and how fast the receiver can respond to it while that confidence is still there to reward, not a threshold on any row's final resting state.

Dipankar sharpened "race" into **"latch"**: at step 1, the receiver's `sqrt(v)` is identical across both settings to three digits — it's not falling behind, it's a gate that closes once. His decisive test: a two-phase schedule (`adam_eps` A then B after K steps, and the reverse). **Latch, unambiguously**:

```
A: eps=1e-12 for steps 1..K, then eps=1e-10 (predicted to flip once K passes 10-50)
  K=5,10,20,30: returns    K=50,100,200: FLIPS

B: eps=1e-10 for steps 1..K, then eps=1e-12 (predicted to keep reverting past 100-200)
  K=50,100,150,200,300,500,1000: returns, every single one — never flips
```

Bras A flips exactly inside its predicted 10–50-step window. Bras B never flips, tested to K=1000 — twenty times past its own predicted window; it doesn't even start reverting-then-recovering partway through, as either of us expected — it simply never moves. Not a shared finish line; a door that closes once in the first fifty steps and stays closed. Three other levers that should give the same kind of early boost, checked one at a time: `lr` up to 3× (twelve cells, all return), `beta2` down to 0.5 — 500× faster tracking of a fresh gradient, directly the mechanism the step trace had identified — (twelve cells, all return, doing *nothing at all*), and `lr` cranked to absurd values (0.3–1.0: nine cells, all return; only at 40× the base rate does anything move, and even then non-monotonically — eps=18 and 20 flip, eps=23 doesn't, one seed, not over-read). If "anything that buys the receiver early gain moves the boundary" were the right generalization, `beta2` should have been the cleanest lever of all; it did nothing. The mechanism is specific to the additive floor, not a general story about early reactivity.

### Thirty-sixth and thirty-seventh rounds — a bug in my own switching criterion, and one lever turned out to be another in disguise

Algebra, not a test: `lr` can't rotate the direction of a shared-scalar Adam update, and `beta2` can't change step 1 at all (bias correction forces `v̂ = g²` regardless, at t=1) — explaining why both had been inert, and why a proposed masking test (freeze only the receiver coordinates sitting in the sensitive 1e-13–1e-9 band) couldn't isolate anything: at step 1, all 729 receiver coordinates already sit there (min 2.86×10⁻¹⁴, p99 9.53×10⁻¹², max 1.52×10⁻¹¹, 716 of 729 inside the band, all 729 under 1e-9) — the receiver had already converged in the base run, so its whole gradient is uniformly small and there's no sub-population to mask against. His falsification test: push `adam_eps` to 1e-6, well above every observed `|g|`.

**Non-monotone under my existing criterion** — until traced step by step, exposing a real bug: `S[0,0]` saturates at 1.0 and stays there for 40,000 steps, but `R[0,18]` **never leaves 1.000000.** My criterion (`S.max() > 0.5`) never checked whether the receiver had actually moved. Corrected and re-audited backward through everything already published — **both published thresholds held** — but `lr=2.0`, already reclassified once from "transfer" to "frozen, worthless," was also mislabeled: it hadn't frozen on target at all, it had been thrown into an unrelated collision by an oversized step. "Frozen without value" turned out to cover two different things, and every large-`lr` cell checked landed in the second, never the first.

Then: `adam_eps=1e-6` turned out to be an `lr` line in disguise — with no receiver coordinate ever exceeding 1e-9, the floor at 1e-6 or 1e-8 acts as an SGD-at-momentum switch, at an effective rate of `lr/eps`, and `(g+1e-6)/(g+1e-8)` is uniform to 0.15% across the whole receiver — a rescale, not a rotation. One prediction confirmed cleanly (freezing at `lr=5e-4`, and stronger than predicted — eps=20 froze too). The reciprocal prediction, reported too fast as confirmed "at all three eps," corrected on request: checking *where* the argmax actually landed (not just `S.max()` and `R[0,18]`) showed two of the three were chaos, not capture —

```
eps=18 : argmax = message 9    (not 0 — chaos, not capture)
eps=20 : argmax = message 22   (not 0 — a new, unrelated collision)
eps=23 : argmax = message 0    (the only real capture of the three)
```

— the referent's logit doesn't drift gently back to uniform under `lr=5.0`, it collapses from +22 to −53 in a few steps, crosses zero, and lands somewhere else entirely. Two gentler multipliers, tried for a clean confirmation rather than resting on the one contaminated cell: at 2× the reciprocal (`adam_eps=5e-7, lr=0.1`), eps=18/20 revert (`S[0].max()≈0.037`, noise, not capture) and eps=23 captures — but that's indistinguishable from what *default* settings already do at this seed, no added evidence. At 10× (`adam_eps=1e-7, lr=0.5`), all three cells revert, *less* transfer than at 2×, the opposite of the monotone trend the equivalence claim would predict — read as noise near a threshold at n=1 seed, not a reversal. The reciprocal test stands confirmed once, not three times: freeze direction and receiver census both clean, transfer direction resting on a single genuine data point. Checked whether `lr=2.0` (already reclassified once, from "flips" to "frozen, worthless") carried the same flaw: it did — `argmax = message 6` at eps=18, `message 20` at eps=20, neither 0. Of five large-`lr` cells now checked by argmax, four are chaos and none is a genuine targeted freeze; "frozen without value" had silently covered two different failure modes. A full re-audit of everything published that session found the same flaw nowhere else: the original threshold, the full 18–20 convergence table, the schedule bras, the eps=1e-6 freeze — all held, argmax-verified, no exceptions, and referent 18 itself lands cleanly on message 8 in both cases checked (`S=1.0000000000` at `adam_eps=1e-12`, `S=0.9999999997` at default) — 26 of 27 referents resolved, one collision remaining elsewhere, no new collision created. The damage was confined to two cells, both at an `lr` two orders of magnitude past anything a base run ever used.

### Thirty-eighth round — it doesn't generalize; the window was wrong

The whole `adam_eps` mechanism — threshold, 18–20 convergence, latch, the `lr`/`eps` equivalence — rested on one seed, one collision, one hand-pushed logit. Two more walls, found in independent generator streams, showed **zero sensitivity** to `adam_eps` across the same perturbation range that gave wall one a clean threshold. Reported as an idiosyncrasy of one seed.

**The window was wrong, not the mechanism.** Wall two's test range (15–23) never touched 24 — the only point where wall one transfers under default settings. Wall three tested only eps=20, a point where wall one itself never transfers, at any `adam_eps`. A direct control confirmed it before touching either other wall: run wall one on wall two's exact reduced grid, and **wall one itself comes back mostly "returns" too**, at default `adam_eps` — the 18–20 convergence only appears once `adam_eps` drops to 1e-10 or below. The earlier negative result had never tested a window where wall one reliably transfers either. Retested walls two and three at eps=24–26: **both transfer too**, at eps=26 instead of 24. Three for three. Idiosyncrasy claim retracted.

A second issue held too: a single Adam optimizer covers both agents (`parametres(e,r)` concatenates them), so the receiver and the defending line see the same `adam_eps` — and the earlier "reciprocal `lr`/`eps` confirmation" could have come from an over-amplified sender rather than the receiver's own equivalence. Left open, and settled the same day on my own initiative: two separate optimizers, one agent at a time on the reciprocal settings.

```
receiver alone at reciprocal settings : never transfers, at any eps tested
sender alone at reciprocal settings   : chaos at eps=18,20 ; TRANSFER at eps=23
```

**The sender, not the receiver.** A frozen-receiver check (never updated at all) then produced what looked like a transfer at eps=26 — caught before publishing by checking `R[0,0]` rather than trusting `S.max()`: it read **4.4×10⁻¹⁰**, the same frozen-and-worthless signature as before. Learning on the receiver's side isn't incidental to a real transfer; it's the condition for one — every confirmed genuine capture in this whole investigation involved the receiver itself moving its credit.

### Thirty-ninth round — three-for-three falls back to one-for-one, and his defender-grip prediction lands

Were walls two and three ever scored on `R[message, loser]`, or only on `S.max()` and argmax — the same bug fixed twice already this stretch? **Scored on S/argmax. And wrong.** `R[10,4]` and `R[10,20]` froze at ~10⁻¹¹ and ~10⁻¹¹ across every eps up to 50, never crossing 0.5. **Every "capture" reported for walls two and three was the same frozen, worthless engagement** the frozen-receiver test had already taught me to check for, never applied here. Three-for-three fell to **one-for-one**: only wall one has a confirmed genuine transfer anywhere in this investigation.

His prediction — a defending referent's "grip," `-log10(1−R)`, decades of receiver confidence on its incumbent — explained exactly why: message-10 incumbents hold about a decade more grip (9.31, 9.33, runner-up at 1.19×10⁻¹⁰ and 5.26×10⁻¹¹) than message 0's (8.41, runner-up 7.93×10⁻¹⁰). Tested the same lever that had unfrozen wall one — dropping `adam_eps` at eps_perturb=26 — and it made things **worse, not better**: `R` on both walls shrank monotonically as the floor dropped further (1.34×10⁻¹³ → 3.26×10⁻¹⁵ → 1.77×10⁻¹⁶, same shape on both). His prediction was the first in this whole exchange to **anticipate** a result before the run rather than explain one afterward.

### Fortieth and forty-first rounds — a clean calibration predicts 26–27; the independent test says never; grip counts but isn't enough

His proposed decisive experiment: vary grip on a *single* defender (wall one) across training checkpoints, rather than across walls. Four points, one defender, no confusion between walls:

| checkpoint | grip | threshold |
|---|---|---|
| 10,000 | 5.70 | (14,17] |
| 20,000 | 7.43 | (20,22] |
| 30,000 | 8.18 | (22,24] |
| 40,000 | 8.41 | (23,24] |

Both an additive form (`threshold = −1.40 + 2.98·grip`, residuals −0.09/+0.25/+0.01/−0.17) and a multiplicative one (`threshold = 6.43·10^(0.068·grip)`, residuals −0.14/+0.52/−0.02/−0.36) fit this cleanly — not because the range distinguishes them, but because both happen to be locally well-approximated by the same line over four points spanning under three decades of grip — and **converge on the same extrapolation** for walls two and three's grip (9.31–9.33): threshold 26.4 and 27.5. Checked against data already in hand: **it fails.** Walls two and three never capture at all up to eps=50 — grip counts, clearly, but it doesn't transfer between defenders on its own. And his last question, resolved the other way from his own suspicion: the frozen receiver's grip at eps=26, full precision, comes back **exactly at its unfrozen baseline value**, 8.4107 (`R[0,18]=0.999999996115699`, `1−R=3.884301×10⁻⁹`) — freezing pins grip, it doesn't elevate it. The freeze result and the grip law are two independent findings, not the same one measured by two levers.

Two more checkpoints on the same defender (80k: grip 8.7481; 160k: grip 8.9763) refit both curve forms on all nine points, and the asymptote moved **below** the other walls' grip entirely — 9.204 for one form, 8.839 for the other, both under 9.31. Referent 18 can never reach the other walls' grip no matter how long it trains. The **same full ladder run on a second defender** (wall two's referent 3):

| checkpoint | grip | threshold |
|---|---|---|
| 10,000 | 6.1115 | > 26 (never captures) |
| 20,000 | 8.0450 | > 26 |
| 30,000 | 8.9951 | > 26 |
| 40,000 | 9.3088 | > 26 |

**The curves don't overlay, and not by a little.** At grip=6.11 — close to referent 18's own lowest point (5.70, threshold 14–17) — referent 3 doesn't capture at any eps up to 26, ten units past a comparable grip level on the other defender. **Grip described one defender's own trajectory. It is not a portable law between defenders.**

A raw-logit-gap hypothesis, tested immediately rather than left as a suggestion, fails even more cleanly than grip did:

| checkpoint | referent 18 (wall 1) | referent 3 (wall 2) |
|---|---|---|
| 10,000 | 13.73 | 14.61 |
| 20,000 | 18.19 | 19.45 |
| 30,000 | 20.29 | 22.01 |
| 40,000 | 20.96 | 22.85 |

The gaps sit *closer* between defenders than the probabilities did — referent 3 at 20k (19.45) barely exceeds referent 18 at 40k (20.96) — while the thresholds still diverge completely: referent 18 already captures around eps≈17–24 across this whole range, referent 3 doesn't capture at eps=26 anywhere in it. If the incumbent's own logit gap were the gating quantity, similar gaps should predict similar thresholds. They don't. Whatever sets the threshold isn't a property of the incumbent's row in isolation, at least not one either scalar captures — left open whether it depends on the challenger's own starting logit, or an interaction between the two rows no single-sided number could see.

### Forty-second and forty-third rounds — the logit gap was never independent of grip, and the 0.5 plateau was a tie, not a capture

The unexplained logit gap turned out to be `grip × 2.45`, a near-constant ratio along each defender's own trajectory:

| checkpoint | 10k | 20k | 30k | 40k |
|---|---|---|---|---|
| referent 18 (wall 1) | 2.4106 | 2.4474 | 2.4819 | 2.4916 |
| referent 3 (wall 2) | 2.3913 | 2.4172 | 2.4474 | 2.4546 |

— rejecting one rejects the other by construction. Checked on two more defenders, not left as a one-pair coincidence: referent 14 (wall three's defender, 40k) gave gap/grip = **2.5367**; referent 1 (an earlier round's solo winner, 40k) gave **2.5119** — both sit outside the 2.39–2.49 window measured on the first two defenders. Grip and the raw logit gap aren't perfectly the same variable after all — there's a real, small, defender-dependent spread (2.39 to 2.54, about 6%) — but that spread is an order of magnitude too small to explain the gap between `k_18` and `k_3` below.

And a "k_3" I'd reported (a threshold ratio for wall two's defender) turned out to need the freeze lever too, at the same signature as wall one's own freeze: pushed at its 10k checkpoint past eps=26 under default `adam_eps`, `R[10,4]` froze at 1.9211×10⁻¹¹ identically from eps=30 through 60. Dropping `adam_eps` at eps_perturb=30 landed almost exactly on the threshold — `R[10,4]=0.50001` at `adam_eps=1e-10`, 0.50000 at `1e-12`, back to frozen at `1e-14` — giving `k_3 = 30/6.1115 = 4.907` computed under a reduced `adam_eps`, against `k_18 ≈ 2.79` (feasible range 2.7348–2.8537, all four brackets) at *default* `adam_eps` — mismatched levers. Retested referent 3 on referent 18's own default-eps lever: **never captures up to eps=60** — the gap, if taken at face value, roughly doubles rather than closes.

Two more checks closed the lever-mismatch question before trusting anything built on it. Does referent 18 ever freeze, anywhere? Pushed to eps=100 at checkpoint 40k, default `adam_eps`: eps=24, 26, 30, 40, 60, 100 all identical — clean winner-take-all every time, never a freeze at any perturbation tried. Does `k_18` itself move under referent 3's own unfreezing lever (`adam_eps=1e-10`)? No: at checkpoint 40k it's the same bracket as default (eps=18,20 return, eps=23 captures); at checkpoint 10k, eps=10,13,14 return, eps=15 captures, giving `k=15/5.697=2.633` — consistent with the 2.79 range. **The reduced-eps lever only bites on cells that were actually frozen; it isn't a global rescale, and referent 18 was never one of the frozen ones.**

Then the plateau itself dissolved. `R[10,4] ≈ 0.500` had been read as a censored capture threshold. Printed at full precision instead of trusting the `>0.5` cutoff:

```
eps=24:  R[10,4]=0.500000583504767   R[10,3]=0.499999416424236
eps=26:  R[10,4]=0.499999999984430   R[10,3]=0.499999999944572
```

**Both referents 3 and 4 sat at S≈0.9999999996–0.9999999997 simultaneously**, fully committed at once, the receiver splitting credit within millionths of exactly 0.5, oscillating non-monotonically as the perturbation varied. **A genuine tie, not a capture at all** — the same phenomenon as the very first ties, reached by an entirely different route. There was no `k_3` to compare. And it isn't a trait either referent carries into any contest: checked whether referent 4 ties this way on its own, away from referent 3 — it doesn't. Its own natural slot (message 6) is already captured with zero contest at the same checkpoint, no perturbation needed at all. **The tie is specific to this one collision, not a property of either referent individually.**

### Forty-fourth through forty-seventh rounds — a pinned residual, an algebraic identity, and a transient that moves nine orders of magnitude

Does the receiver's residual — the mass a contested message sends to anything *neither* competing referent — stay pinned while the split between them moves? On his tie: pinned to a 2×10⁻¹⁵ spread while the split moved by up to 1.9×10⁻⁶, five orders apart. His fork: if it pins under a clean capture too, the residual is structural, not tie-specific. Rebuilding wall one's actual capture — after losing, then recovering, its exact reconstruction (below) — and printing the residual at full precision: **pinned there too**, 4.6353×10⁻¹² to 4.6487×10⁻¹² across eps 24 to 100. His second branch, not his first — my "qualitative branch, tie versus capture" reading died with it.

What replaced it needed two more corrections. First: `r` (the receiver's own row) and `R` (the reward cell, `s·r`) only looked identical because the sender's own leak had already underflowed to zero at the eps values checked.

| eps | 1−R[0,0] | 1−r[0,0] | 1−s[0,0] |
|---|---|---|---|
| 24 | 2.847969×10⁻⁷ | 2.841324×10⁻⁷ | 6.644512×10⁻¹⁰ |
| 26 | 2.841825×10⁻⁷ | 2.840549×10⁻⁷ | 1.275220×10⁻¹⁰ |
| 30 | 2.841186×10⁻⁷ | 2.841161×10⁻⁷ | 2.431388×10⁻¹² |
| 40–100 | 2.841160×10⁻⁷ | 2.841160×10⁻⁷ | 0.000000 |

His predicted values at eps=24 and 26 — computed from `1−R ≈ (1−r)+(1−s)` before either of us had run the check — came back exactly: `2.841324×10⁻⁷ + 6.644512×10⁻¹⁰ = 2.847969×10⁻⁷`, matching `1−R` to all six printed figures. **`r` is `R` only because `1−s` vanishes first, not because they're the same tensor** — they collapse onto each other purely because the sender's own leak underflows to exactly zero once it saturates (eps≥40 here), forcing agreement by construction rather than dynamics. The earlier "four orders above the residual" claim had rested on eps=100, a row where `s`'s leak had already underflowed and the comparison had nothing left to discriminate; on the rows where it still could (24, 26), `r`'s leak (2.84×10⁻⁷) really is four orders above the residual (4.6×10⁻¹²) — the claim survived, just on the wrong row the first time. Second, and decisive: sampling the **transient** of a real transfer step by step, rather than only its converged endpoint —

```
pas=6000-7500 : R=0        residual≈0.499999...   (looks exactly like a tie)
pas=8000      : R=0.682857 residual=1.443847e-03
pas=8500      : R=0.999922 residual=5.157076e-10
pas=10000     : R=0.999963 residual=1.490894e-10
```

**The residual is not pinned during a real transfer at all.** It moves nine orders of magnitude, passing through a state indistinguishable from a genuine tie on the way to a clean capture. The pinned value reported earlier was just the tail of a collapse that had already finished. A tie-snapshot and a capture-snapshot can be the *same trajectory* at two different times — except that some 50/50 splits (this one) resolve in a few hundred steps and others (referents 3/4) sit unmoved for 400,000 steps with no drift in either direction, confirmed on the actual reconstructed pair rather than assumed from an earlier round.

A last discriminator, proposed and tested: does the sender's own leak (`1−s`) sit near zero on both competitors of a tie (meaning a single snapshot can't distinguish a way station from a fixed point, and only the time axis can) or near 0.5 (meaning one snapshot settles it forever)? On referents 3/4, at four `adam_eps` settings needed to unfreeze the pair: `1−s[4,10]` and `1−s[3,10]` both sit at 1e-10 to 1e-12 — **near zero on both**, his first branch, the discriminator is dead here, and what actually establishes the fixed point is the longitudinal check (confirmed on the actual reconstructed pair: `R[10,4]` pinned at 0.5 across ten checkpoints spanning 40,000 to 400,000 steps, no systematic drift), not this snapshot.

And the small non-monotone climb in the losing sender's own leak, right at the moment of a real transfer's flip, tested with three formed hypotheses rather than dismissed as "two insignificant figures":

| pas | Σ (26 losing logits) | 1−s[0,0] | receiver's r[0,0] |
|---|---|---|---|
| 7,000 | 53.089291 | 2.586964×10⁻⁹ | 0.000000 |
| 7,500 | 53.205838 | 2.918510×10⁻⁹ | 0.000000 |
| 8,000 | **53.337311** | **3.343607×10⁻⁹** | **0.682857** |
| 8,500 | 53.186451 | 2.860844×10⁻⁹ | 0.999922 |
| 9,000 | 53.052747 | 2.491666×10⁻⁹ | 0.999939 |

**Adam artifact?** Inconclusive — plain SGD at the same window doesn't move at all (frozen exactly), meaning every "exact ascent" claim across this whole exchange has actually been Adam, never literal gradient ascent (`monter()` always builds a `torch.optim.Adam`). **Entropy redistribution among the other 26 logits?** Confirmed — their sum tracks the dip in lockstep, peaking at exactly the same step, 8,000. **Does it coincide with the receiver actually flipping?** Exactly — `r[0,0]` goes from 0.000000 to 0.682857 at that same step, where the dominant logit dips and the losing sum peaks. **Not noise — the visible signature of the coupled sender-receiver system at the instant the receiver's decoding flips**, briefly redistributing the sender's own logit mass before it resumes climbing.

Three more self-posed questions, pushed past where the round would otherwise have closed. **Is Adam load-bearing for the transfer itself, not just for the dip?** Reran wall one's own eps=24 window under plain SGD at a learning rate 100× Adam's (5.0 instead of 0.05), all the way to 20,000 steps: the receiver's credit stays at exactly 0.000000 the entire time, the sender barely moves (`1−s` frozen near 9.8×10⁻¹⁰). **Adam's adaptive scaling isn't standing in for a bigger step — SGD at a hundred times the rate can't reproduce the transfer in this window at all.** **Is the dip carried by one rival, or spread evenly?** The top five non-winning logits (messages 7, 5, 15, 18, 23) all rise together by about +0.01 at the peak step, then fall back together — a uniform lift across the whole losing set, not one competitor gaining ground, consistent with the entropy-redistribution reading rather than a specific rival closing in. **Does pushing the referents-3/4 tie harder ever force a real capture?** Pushed the challenger at eps=60, 100, 200, and 400 — more than thirteen times the perturbation that produces the tie at eps=30 — trained 40,000 steps each: **the result is identical to six figures at every one of the four settings, at every checkpoint.** The sender saturates instantly regardless of how hard it's pushed, and once saturated, pushing harder has nothing left to affect; the receiver stays pinned at exactly 0.5 regardless. This tie doesn't look like a slow capture that a big enough push would eventually tip — it resists sender-side perturbation across more than a factor of six, a stronger and more specific claim than "stable over 400,000 steps."

### The reconstruction that got lost, then found in the transcript

Somewhere in this stretch, the exact seed recipe behind "wall one" — everything above hangs on it — stopped being reproducible. Its labels ("referent 18," "message 0") were presentation labels from a run never saved to a file. Two wrong master seeds (`5`, then `31415` — the seed the one *saved* script actually uses) produced real walls that matched no published number, and one was nearly reported as the real thing: its loser stayed frozen at zero all the way to eps=100, contradicting what had been published for eps=24. Caught only because the baseline state didn't match numbers already sitting in the notebook (`S[18,0]=0.499479` is specific enough to check against). Recovered by grepping this session's own conversation transcript for a literal code fragment from eleven days earlier — `default_rng(999)`, five pairs skipped — confirmed against the notebook's own six-decimal numbers before being trusted, and saved permanently this time as `replay_idx5.py`.

---

## 🕳️ Part three: a wall that wasn't fighting anything, and a col that took six tries to even locate

Rounds forty-eight through fifty-two, then a stretch with no round at all — `dipankarsarkar` went quiet for two days, and rather than wait, I kept pulling the same thread alone, sometimes running a second copy of his own reading style against my own results before trusting them. Everything in this part that isn't credited to a round number is that: not a new critique, my own suspicion applied to my own numbers, because the rule that produced this whole article — check what a statistic is a function of — doesn't stop applying just because the person who taught it isn't in the room.

### Forty-eighth round — is 0.5 a posterior, or a fixed point?

A different referents-3/4 tie surfaced (message 10, seed 77777, three pairs skipped, checkpoint 10,000, referent 4 pushed +30) — same 0.5 split as the wall/tie taxonomy above, different route in. His question: is that 0.5 a Bayesian posterior over which referent a collided message "really" names, or a genuine dynamical fixed point that doesn't care how many times each referent was seen? The test he proposed breaks the prior instead of pushing the perturbation — train with referent 4 shown twice as often as referent 3, everything else equal, and watch whether `R[10,4]` migrates toward 2/3.

It didn't. `R[10,4]` and `R[10,3]` both sat within a hair of 0.5, entropy on message 10 landed at 0.693147 — `ln(2)` to six figures, meaning the receiver's mass really was split between exactly two referents and nothing else. **A fixed point, not a posterior** — the count asymmetry that would move a Bayesian estimate left this one exactly where it started.

### Forty-ninth round — where does the losing referent's mass actually go?

If referent 3 loses this fight outright (pushed past the tie, deep into delta territory), does it fall back to 1/2 — ceding the message cleanly to referent 4 — or does its mass scatter somewhere else entirely? It collapses to **1/27**, not 1/2. H13, formed and left open in that round rather than argued from the armchair: the twenty-five referents not even in the collision are the destination for the losing side's mass, not referent 4. Tested properly, not just asserted, by building the two-referent toy *without* the other twenty-five rows and checking whether the same threshold still exists there — it doesn't, on its own, which is the ablation that actually earns H13 its name rather than leaving it as a remark.

### Fiftieth round — a coincidence in the drift, chased instead of dismissed as noise

A residual tracking the emitter's own deficit had been drifting at a coefficient around 8–10 that I'd waved off as "two significant figures, not a mechanism." `dipankarsarkar` asked why it tracked the deficit at all if the regularizer scaling I'd already ruled out wasn't the cause. It shouldn't have taken his asking. Chasing it properly required a mechanism test I hadn't run: do the emitter's two gradients on the contested referents ever cross — a literal "race" where one term briefly overtakes the other and explains the drift's sign flip? Checked on the emitter side (`grad[4,10]` constant and negligible everywhere, the challenger already saturated; `grad3−grad4` never crosses zero) and, pushed further than he'd asked, on the receiver side too (`grad_r3−grad_r4`, same verdict, smooth, no crossing). **H-course is dead, twice, by two independently constructed tests** — whatever the drift is, it isn't a race between competing gradient terms.

### Fifty-first round — a closed form for the fold, and a number I had to stop taking on faith

The wall/tie/fixed-point picture finally got a form: the referents-3/4 separatrix is a genuine saddle-node bifurcation in a coupled sender-receiver system, reducible to two equations —

```
d3 / (26(1 − d3)) = exp(−(1 − δ)(1 − R) / β)
logit(R) · β = 2δ + (1 − δ)·d3
```

— with a fold at `δ_c = 0.013437210`, independently confirmed by four separate methods across this stretch: root-fusion bisection on the closed form, direct algebraic intersection of the two curves, a basin-probe on the actual game, and — much later — an empirical bisection on a *second*, unrelated referent pair, landing on the same threshold to five decimals (below).

He also published a slowdown coefficient, `0.2212`, with three decimal points of confidence and no derivation attached. The instruction I'd been given after the 0.2212-year: never take his numbers on faith just because they arrive with a lot of digits. Refitting it on eleven algebraic points instead of the three he'd published showed `0.2212` really is the gap between the fold's stable and unstable roots (`d3_unstable − d3_stable`), not a residual against the soft law I'd first tried — but a self-invoked adversarial pass (the first of several agent-simulated challenges in this stretch, see below) found my own refit had stopped at a float64 precision floor, not the real asymptote. Retraced at `eps=1.3437×10⁻⁸` in fifty-digit precision, matched independently by a local Lyapunov–Schmidt expansion at the fold: **0.2212604**, seven significant figures of agreement between a brute-force retrace and a closed-form derivation that hadn't existed until that round.

### Fifty-second round — the rate that wouldn't hold still

The reduction above still needed `k`, the relative speed of the receiver's response to the sender's. It isn't a constant. Measured across `R_init = 0.50` to `0.75`, `k` ranges `1.42` to `2.45` — a factor of 1.7 across a range that a clean two-timescale reduction should have made irrelevant. Neither `beta1` nor `beta2` (Adam's own moment decay rates) move it enough to matter (`k` shifts by 0.02 against a total spread of ~1.0 when `beta2` sweeps `0.99→0.9999`) — whatever sets `k`'s value isn't the optimizer's own hyperparameters. Left open at the end of the round: the answer wasn't in the lever anyone had already pulled.

### What I found alone, chasing the same rule without being told to

Two days with no round. Rather than wait, or lower the bar for what counted as worth checking, I kept applying the one instruction that had produced every correction in this article — verify before citing, including my own numbers — sometimes literally running a second, adversarial pass against my own conclusions before writing them down, because the discipline that makes an external reader valuable doesn't require the reader to be present.

**The 0.2212 coefficient's own correction had a hidden gap.** My two-term drift model (`gap = C₁√ε + C₂ε`) used a basis that's analytically wrong — the `ε¹` term is structurally zero, and the true next term is `ε^(3/2)`. Derived by hand (a Lyapunov–Schmidt expansion one order past the one that gave `0.2212604`): `D = 8.0021212`, matching an independent numerical extrapolation (`8.0021`) to five significant figures — then shown, by sweeping `β` and `N` rather than trusting one point, to be *not* a structural constant at all: it drifts continuously with both (4.86 to 8.57 over the `β` range tested, 8.00 to 8.59 over `N`).

**The fold isn't specific to one pair of referents.** A second, independent tie — referents 23/25, message 13, an entirely different seed — gave `δ_c = 0.0134370` against `0.0134372` on the original pair: a 0.0016% gap, indistinguishable from the bisection's own noise floor. The threshold is a property of the objective (`N=27`, `β=0.02`), not of which two rows happen to collide.

**Background mass moves the threshold, with a floor and a ceiling, not a slope.** A toy with `M` variable-weight background categories showed the shift depends on total background *mass*, not on `M` itself (`M=8` at 8% mass and `M=25` at 8% mass give the identical `δ_c`), and that mass's effect has a **sharp, binary threshold** — located between 17.7422% and 17.7539% across eighteen bisection points, never a value in between — past which the shift saturates rather than growing (`δ_c` shift is identical at 25% and 50% background mass).

**And then the same perturbation, run on the real 27-referent game instead of the toy, inverted a basin outright.** Pinning 25% background mass onto ten otherwise-idle referents and re-running the pin-and-falsify protocol from round fifty-two: at `R_init=0.60`, the side of the threshold that used to collapse now converges cleanly, and the side that used to converge now collapses — a full reversal, located to five decimal places (`0.9994521` to `0.9994526`), absent entirely at `R_init=0.75`. A self-invoked adversarial pass killed the first mechanism I proposed for it (a crossing between the sender's two gradients, tested and refuted twice, exactly as in round fifty) and forced the actual one: tracing the full 40,000-step trajectory of both near-threshold configurations showed them running almost identical for ~600 steps, then diverging at `pas=600.283` in a sign flip on the sender's own gradient — a textbook saddle-node ghost (`dx/dt = μ + ax²`, passage time `τ = C/√μ`), confirmed by a passage-time prefactor consistent to 6% between the two configurations (`C = 0.920` and `0.977`).

**Whether that ghost is the *same* saddle as the one closed-formed in round fifty-one, or a distinct neighbor, took six separate failed attempts to even approach — each one diagnosed, not discarded.** Comparing the two points' local quadratic coefficient directly, rather than assuming the normal form's curvature was 1: a single trajectory's fit was unstable across sub-windows (×5.6 spread); a single fresh Adam step turned out to always carry the optimizer's full bias-corrected step size regardless of the true gradient underneath it, a genuine and previously undocumented measurement trap; pooling trajectories from both sides of the threshold produced a mathematically impossible fixed point (`x0 > 1`) by mixing two incompatible global outcomes into one local fit. Deflating the raw signal by Adam's own bias-correction envelope — computed in closed form and checked against the optimizer's actual internal state to five significant figures — didn't fix it either, and a second self-invoked adversarial pass supplied the piece that finally explained why: an independently-computed 2×2 Jacobian at the closed-form saddle has two *real* eigenvalues of opposite sign (`+3.226×10⁻⁶`, `−1.128×10⁻⁴`, a factor of 35 apart) — a genuine hyperbolic saddle in the underlying continuous dynamics, confirmed a second time by a differently-built finite-difference check. **The chaos wasn't the objective's geometry at all — it was Adam started stone cold (zero moment estimates) on a real saddle.** A precommitted causal test settled it outright: injecting a plausible warmed optimizer state at the *exact* starting coordinates, moving neither the sender's nor the receiver's value by anything, took the trajectory's sign flips from five to zero. Three further attempts to turn that causal proof into an actual measurement — a longer cold-start window (loses the local region before the noise clears), fitting directly on the injected trajectory (the fitted curvature depends continuously on how much state is injected, never settling), a real multi-step warm-up on a pinned point (worse than either — the optimizer learns "this gradient never changes" and takes maximum-size steps once released) — all failed for three distinct, understood reasons. **The identity question is still open.** What's no longer open is why it's hard: the real saddle exists, cold-start Adam is a genuine confound on measuring it, and closing the gap needs an actual approach trajectory with the two hundred-plus steps of runway the toy's own delayed case gets for free — not one more trick played on an optimizer started at rest.

### The fold that turned out never to have moved at all

The saddle-identity question above closed a round, not the thread. The next session picked up the one piece flagged as untried — an ODE for the *toy* (background mass on a reduced two-referent game) instead of one more trick on the real system's optimizer — and it unwound the entire background-mass story down to its floor, twice reversing itself in the process and writing both reversals down rather than quietly fixing the record.

**Building the toy's own branch functions found a 40% error in the first pass.** A grid-based fold locator (200,000 points over the interval) missed the true saddle by a wide margin — it lives in a window forty times narrower than the grid's own spacing — and reported a fold position 40% off. An agent caught it; a second, independent finite-difference check (a different numerical method entirely) confirmed the correction to nine decimal places, and — checking the agent's own follow-up number in turn — found the agent's claimed sensitivity-to-background-mass figure was *itself* wrong by ten orders of magnitude, an artifact of casting a full-precision subtraction through ordinary float64 before it was computed. Redone entirely in fifty-digit arithmetic: the fold's position and its local curvature are identical for every background-mass level tested, to twenty-one orders of magnitude below anything measurable. **The static geometry of the fold — where it sits, how sharply it bends — has nothing to do with the background mass at all.**

That result then reached backward and corrected an already-published number. The `-0.84%` threshold shift from the earlier round turned out to be measured at a bisection tolerance of the same order as the effect itself (`ε_tol = 0.80%` against a reported `0.84%`) — refit at thirty times finer resolution, the real number is `-0.63%`, real (twenty-three times above the finer tolerance's own noise floor) but a quarter smaller than published. A follow-up test (does the residual depend on learning rate, the signature of an optimizer artifact rather than a property of the objective) found it does — `-1.04%` at a quarter the learning rate — and chasing *that* down produced a genuine mechanism, not a bug: at its own threshold, the background-mass configuration lands the sender not on the fold at all but on an entirely different point, `s₃ = 0.5`, which is simply the fold's own branch function evaluated at full receiver commitment (`R = 1`) rather than the receiver's actual equilibrium value — the same function, an extreme input, no second attractor.

**Then a natural-gradient rescaling — the fix for a sigmoid whose gradient vanishes as `s₃(1−s₃)` near saturation — appeared to break the whole reduction: it climbed toward 1 instead of settling where Adam settles, for two million steps, no sign of turning around.** Extending Adam's own trace tenfold (to 400,000 steps) showed its resting point holds firm, ruling out "Adam just hasn't finished yet." Tracing the disagreement down to individual steps found the apparent contradiction was comparing against the *wrong* receiver value — the algebraic equilibrium, not the one actually present mid-transient — and once corrected, no contradiction survived: the true relaxation is exactly linear, but in log-odds space, not probability space, where a seventeen-unit gap invisible in probability terms (crushed by the sigmoid's own exponential compression to a `2.5×10⁻⁴` difference) was in plain view all along. The corresponding relation for the *receiver* side — never written down before this round — closed the loop: a Gibbs-distribution fixed point derived from the objective's linear-reward-plus-entropy form, verified against Adam's own converged state to `2.4×10⁻⁶`. Run as a genuine two-sided natural-gradient system at a large enough step size, it reaches Adam's exact fixed point — both coordinates agreeing to seven significant figures — in a budget comparable to Adam's own, not the millions of steps a mismatched learning rate had first suggested. **Adam is an accelerator here, not a different destination.** And the fixed point both methods reach turns out to be nothing more exotic than a solver already sitting in the repository from an earlier round, built to measure relaxation rates, that had been the exact answer the whole time.

The corrections along the way outnumber the results: a grid resolution bug, an agent's own precision error, a stale headline figure traced to a caveat the note-taker had written and then lost track of, a false contradiction from comparing against the wrong variable, a coordinate choice that hid seventeen units of real distance behind four decimal places of probability. None of them changed the one number that mattered from the start — the fold doesn't move with background mass — but every one of them had to be run down by hand before that number could be trusted past a single measurement.

---

## 🪦 A sample of what died, chosen for what it teaches

| what I believed | what killed it |
|---|---|
| a 0.35 threshold, "derived, not arbitrary" | its own null's supremum is exactly 1; a sample maximum estimates nothing about it |
| `R` (distinct messages) is a factor worth stratifying by | `corr(R, reward)=0.9725` — it is the training objective, rounded |
| my worst-case ratio of 13.9× survives replication | mean vs. maximum, five orders of sample size apart, converging to exactly 1 |
| the collision inflation I bounded for thirteen rounds is a property of emergent codes | it is the null distribution's own mean, saved to disk on day one under an honest field name |
| every audit weakening my claims proves rigor | it is also what an audit that can only find one kind of error looks like |
| REINFORCE reaches a linked target less often than exact ascent | 92% against 5% — a critical point isn't a strong attractor for a noisy method |
| a tie in the game is a symmetric duel | a confident referent holding all the receiver's attention against one holding none |
| a wall and a genuine tie are the same phenomenon | sender confidence separates them with zero exceptions across 52 collisions |
| a permanent free split is the true optimum for an uncontested referent | conditional on a third referent staying asleep — proven by waking it |
| an Adam-epsilon threshold found on one wall generalizes to two more | it does, at the right window — but only one of three walls ever actually captures anything |
| a grip-based law calibrated on one defender transfers to another | the curves don't overlay at all on a matched checkpoint ladder |
| a 0.5 plateau under a reduced optimizer floor is a capture threshold | full precision shows a genuine tie, oscillating around exactly 0.5 |
| a pinned residual distinguishes a tie from a capture | it pins under both; what moves is the transient, by nine orders of magnitude |
| `r` and `R` disagreeing by four orders proves something | they agree exactly once the sender's own leak is accounted for — pure algebra |
| a non-monotone dip in a sender's leak is too small to explain | it's the exact signature of the receiver's own decoding flip, confirmed by three tested hypotheses |
| the 27!-tied-bijections premise describes what the dynamics measures | 95–99.9% of runs never reach a bijection at all; the tied-optima test has a useless bound at the population actually reached |
| sub-optimal attractors can't be escaped by any local method | REINFORCE walks out of the same critical point 92% of the time that traps exact ascent 95% of the time |
| a leave-one-out baseline can only help by conditioning on a selection event | it raises variance at every point tested, 2–20%, and zeroes out 39–73% of batches outright |
| a seed determines a published number | the same seed, same budget, reversed loop order, moves a published bound by a factor of 2.17 |
| a 0.5 split under a count-asymmetric prior is a Bayesian posterior | training referent 4 twice as often leaves R[10,4] at 0.5 and entropy at exactly ln(2) |
| a losing referent's mass falls back to the referent it lost to | it collapses to 1/27 — the twenty-five idle referents are the destination, confirmed by removing them and losing the effect |
| a drift I can't explain at two significant figures isn't worth a mechanism | it tracked a real coupled-system signature the whole time — a receiver flip, not noise |
| an unexplained coefficient's own correction is complete once it matches a second method | the correction itself had a wrong basis (a zero term treated as present), caught on a second pass |
| chaos in a saddle-adjacent fit means the objective's local geometry is genuinely irregular | a real hyperbolic saddle sits there; the chaos is Adam started at zero moment estimates on it |
| a single fresh optimizer step measures the true local gradient | it always carries close to the full bias-corrected step size, regardless of what the gradient underneath actually is |
| a fold's local geometry shifts when background mass is added | position and curvature agree to twenty-one orders of magnitude of precision across every background-mass level tested |
| a natural-gradient trace that climbs toward 1 instead of Adam's resting point proves the reduction is broken | it was compared against the wrong receiver value; corrected, the two methods reach the identical fixed point, verified to seven figures |
| a probability-space view is a neutral coordinate choice for judging distance to equilibrium | it can crush a seventeen-unit gap in log-odds into a four-decimal difference, hiding exactly how far from settled a system still is |

---

## 🧰 What transfers

- **Check what a statistic is a function of before contrasting it, not after.** `R` was disqualified by reading the generator, with no data required.
- **A claim's retraction obliges a search for everything that depended on it.** A dead threshold left its descendant statistic being measured and defended for eight more rounds with no consumer left.
- **The name of a number should carry every argument its value depends on.** Hiding a sample size, a conditioning, or a certification fraction in the name rather than the value is the single rule that would have prevented the most rounds.
- **An audit only ever finds the error it was built to find.** The corrective is asking which direction an audit *cannot* see, and whether anyone has spent effort tightening a result that already favors the argument.
- **A statistic mixing a point mass with a skewed continuous part isn't caught by any test built for either alone.** `min`, `max`, and an exact-zero count would have shown it in one line.
- **Two visually identical phenomena can separate with zero exceptions on the right cross-tab.** Sender confidence against receiver assignment turned "84% ties" into "100% ties, the rest were never collisions."
- **A pinned quantity at convergence can be pure aftermath.** Sampling the transient, not just the endpoint, turned a "structural, tie-independent" residual into "irrelevant — it's the tail of a nine-order-of-magnitude collapse."
- **A reconstruction not saved to a permanent file is not reproducible, no matter how carefully it was run once.** Grep the session's own history before declaring a result unrecoverable; save it the moment it's found.
- **A seed names a stream, not a state.** Threading one generator through several published computations makes the *order* those computations run in a hidden argument — reversing it moved a published bound by a factor of 2.17 with nothing else changed.
- **A critical point is not a strong attractor for a noisy method.** The gradient can genuinely fall to 10⁻¹¹ at a sub-optimal point and still not hold a stochastic estimator there — "no local method escapes this" and "the exact method doesn't escape this" are different claims, and only testing both separates them.
- **Adam started at rest is a confound, not a measurement.** A fresh optimizer's zeroed moment estimates carry no information about the local landscape yet mimic its own bias-correction curve closely enough to be mistaken for a signal — proven by a causal test that moved the optimizer's internal state alone and eliminated the chaos without moving either coordinate.
- **A discipline for using an outside reader is a discipline you can run on yourself.** Two days without a round didn't lower the bar — the same "verify before citing" instruction that produced this article's corrections caught a wrong basis in my own derivation and a float64 precision floor mistaken for an asymptote, neither one flagged by anyone but a second, adversarial pass at my own conclusions.
- **Where a fixed point sits and how a method gets there are different questions, and confusing them manufactures false disagreements.** Everything that looked like a background-mass effect on the objective's landscape turned out to be a fact about the path toward an unmoved destination — a transient, a coordinate choice, a mismatched learning rate — not a fact about the destination itself.

---

## ⚠️ Limits

- **One toy, two eras, one reader — plus a third stretch with no reader at all.** Every correction through round fifty-two came from a single outside collaborator; the material after that came from continuing the same discipline alone, including self-invoked adversarial passes against my own results — a real substitute for keeping the habit alive, not a real substitute for an independent human reader, and every number produced that way is flagged as such above.
- **The Adam-epsilon mechanism is confirmed on two walls out of three, in one direction only.** Walls two and three transfer at the predicted window; neither has ever produced a confirmed clean capture at any perturbation tried. Whether a wall *can* ever capture cleanly, versus only tie or freeze, is unresolved.
- **The grip law is refuted as a cross-defender law, not replaced.** No candidate tested so far — probability grip, raw logit gap — survives a second defender.
- **The residual's transient mechanism is described, not explained.** What pins it at convergence and what drives its motion mid-transfer are measurements, not derivations from the update rule.
- **Every number here runs through Adam.** `monter()` builds a `torch.optim.Adam` in every call across this entire investigation; nothing here is a statement about the exact objective's landscape independent of the optimizer that walks it, except where explicitly checked against the raw gradient.
- **The saddle-identity question closes this article open.** A real hyperbolic saddle is confirmed, and cold-start Adam is confirmed as a genuine confound on measuring it — but whether the ghost encountered under background mass is the *same* saddle as the one closed-formed at δ_c, rather than a close neighbor, is still unresolved after six distinct, diagnosed attempts.

---

## 🗂️ Summary

Nineteen rounds went into making one small statistical table honest, and honesty kept moving the goalposts: a threshold measuring a supremum it could never estimate, a "factor" that was the training objective in disguise, eleven rules earned one failure at a time, a seed that turned out to name a stream rather than a state (reversing a loop's order moved a published bound by 2.17×), and two separate instances where the number under a dozen rounds of dispute had already been computed and saved, correctly, before the dispute began. Along the way, the design's own headline premise — 27! tied bijections, 1,296 compositional — turned out to describe a population the dynamics reaches 5% of the time or less, and the REINFORCE experiment the whole project was named for had never actually been run: exact ascent traps at a genuine critical point 95% of the time, and the sampled estimator of the identical gradient walks out of it 92% of the time. Twenty-three more rounds went into a single 27-referent game, discovering that "collision" hides three different things — a real tie protected by symmetry, a wall that was never fighting anything, and an Adam optimizer floor dressed as a permanent stall — then chasing that floor through a latch, a classifier bug, a false non-generalization, a law that fit one defender and refused a second, and a residual whose pinned value at convergence turned out to be the tail of a transient moving nine orders of magnitude on the way there. Five more rounds, on a different collision entirely, closed a fixed-point-versus-posterior question, found a losing referent's mass scattering to twenty-five idle rows rather than its rival, and closed a fold in closed form — before the reader went quiet and the same discipline, applied alone, found a real saddle hiding under a cold-started optimizer's own noise, proved the noise was the optimizer and not the landscape, and still couldn't quite catch the saddle's exact identity after six honestly diagnosed tries. Neither era ends on a clean theorem. None of the three end on a settled question.

---

## ❓ Q&A

**Why does the statistics era matter if it's "just" one table?**
Because the same failure — trusting a derived quantity's name over what it's actually a function of — produced eleven rules over nineteen rounds, and every one of them had already been available on day one from the generator or the null distribution alone.

**Is the wall/tie taxonomy final?**
No. Three categories are confirmed (tie, wall, a transient passing through a tie-shape), the mechanism separating "can capture cleanly" from "can only tie or freeze" isn't identified, and the grip law that looked like a candidate explanation failed its second test.

**What's the single most expensive mistake in here?**
Checking a switching criterion on sender confidence and argmax alone, without the actual reward, mislabeled a result at least three separate times, on three different mechanisms. The fix was always the same one-line check, applied late every time.

**Did the reader ever concede a point?**
Repeatedly — his own quantum-quantization claim, his own AST discriminator's edge case, his own "race" reading of the Adam-epsilon mechanism — and at least once, in article 2, faster than I could deliver the numbers he'd asked for.

**What does "self-invoked adversarial pass" actually mean, mechanically?**
A second, independent attempt to break a result I'd just produced — sometimes literally a separate process instructed to argue against my conclusion before I trusted it — followed by verifying every number it raised myself, the same standard rounds six through fifty-two applied to his numbers. It caught a wrong basis in my own math and a float64 floor mistaken for a true value; it also produced at least one claim (a rate comparison read as "×17") that turned out to be my own misreading of the *first* adversarial pass, caught by a second one. It is not a substitute for an outside reader. It is what kept the same standard from lapsing while one wasn't available.

**Is the saddle the same one as the closed-form fold in round fifty-one, or isn't it?**
Genuinely unresolved. What's resolved is that the two most likely reasons to think it *wasn't* — an unstable local-curvature fit, and a chaotic sign pattern near the starting point — are both artifacts of measuring a real saddle with an optimizer that hadn't warmed up, not evidence of a different saddle. Removing a wrong reason to doubt something isn't the same as confirming it.

---

## 🙏 Credit

Every correction through round fifty-two is **`dipankarsarkar`**'s ([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)) — the R-is-the-objective catch, all eleven numbered rules, the reservoir-sampler bug, the closed-form supremum question, the REINFORCE-was-never-run catch, the tie/wall separation, the Adam-epsilon mechanism's first identification, the latch-versus-race schedule test, the wrong-window correction that turned a false negative into three-for-three, the grip calibration protocol, the residual-under-capture question that unwound a round of my own overreach, the prior-versus-fixed-point test on the second collision, and the closed form for the fold itself.

This is his sixth through fifty-second round on this project.

Everything after round fifty-two that isn't credited to a round number was found without him in the room — including a wrong basis in my own drift correction, and the mechanism separating a real saddle from a cold-started optimizer's noise — using self-invoked adversarial passes against my own conclusions in place of his, each one verified independently before being trusted rather than taken on the same faith I'd learned not to extend to his numbers either.

---

## 💻 Code and citation

Everything is reproducible: `src/test3_communication/` holds every script behind every number here, `docs/CARNET.md` the full notebook — French, dated, every dead hypothesis left in — and `docs/REPONSE_ORDRE*.md` the English replies this article draws from.

🔗 **GitHub:** https://github.com/RDTvlokip/RDTRL
📦 **DOI (all versions):** [10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216)

MIT licensed.

---

**Théo CHARLET**

TSSR Graduate (IT Systems & Networks Technician) - AI/ML Specialization

Creator of AG-BPE (Attention-Guided Byte-Pair Encoding)

🔗 LinkedIn: https://www.linkedin.com/in/théo-charlet

🔎 RDTvlokip Search (my search engine): https://search.rdtvlokip.fr

🚀 Seeking internship opportunities

🔗 Website : https://rdtvlokip.fr
