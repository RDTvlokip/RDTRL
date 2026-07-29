# 🎲 Teaching a network to write with reward only — it hit 99.9% grammatical without learning a single rule 🇫🇷

---

## 👤 Context — why I even tried this

If you've followed the series, everything so far was **pretraining**: a crawler, a dataset, a custom BPE, a 15M French LLM trained from scratch on a 1080 Ti. Next-token prediction all the way down.

So I asked the obvious question: **does it have to be?**

Humans don't learn to write by predicting the next character a billion times. They write something, get told it's wrong, and adjust. That's reinforcement learning. So: **can a network learn to write from random weights using only a reward signal — no pretraining, no input/output pairs, no language knowledge injected?**

I asked around. The answer I got was: *no, because of sparse reward.* On a 12-character sequence with 27 possible characters you have 10^17 combinations. A random agent never hits anything correct, never gets a positive reward, and therefore has nothing to learn from.

That's a good argument. **I wanted to test it instead of believing it.**

This article is what happened. Short version: it worked twice, and both wins turned out to be fake in interesting ways.

---

## 🖥️ The rig — and a surprise

Same machine as always: GTX 1080 Ti 11GB, Ryzen 5 5600G, 48 GB RAM.

Except the GPU never got used. I benchmarked it first:

| device | batch | ms / update | ms / episode |
|---|---|---|---|
| **cpu** | **1** | **16.4** | **16.4** |
| cuda | 1 | 32.3 | 32.3 |
| cpu | 32 | 23.4 | 0.73 |
| cuda | 32 | 36.1 | 1.13 |
| cpu | 256 | 61.7 | 0.24 |
| cuda | 256 | 34.8 | **0.14** |

At batch size 1 — which is what REINFORCE actually needs — **the CPU is 2× faster than the 1080 Ti**. The model is a GRU with 128 hidden units on a 3-step sequence: ~200 µs of real math, drowned in CUDA kernel launch overhead paid 12 times forward and 12 times backward. The GPU spends its life waiting for orders.

It only wins past batch 256. Worth knowing before you reflexively `.cuda()` a tiny model.

---

## 🧪 Test 1 — copy a fixed sentence

The easiest possible version of the problem. Target: `le chat dort` (12 characters). Vocabulary: only the characters used, 12 tokens. Search space: 12^12 = **8.9 × 10^12**.

- **Policy**: 1-layer GRU, hidden 128, generates one character at a time, no teacher forcing
- **Weights**: random init, zero pretraining, no exceptions
- **Reward**: `(characters in the right position) / length`
- **Algorithm**: REINFORCE with a moving-average baseline, Adam 1e-3, entropy bonus

The target never touches the network. It only enters the reward function, which returns a float.

### It worked

```
before training (random weights) : 'dddddddddddd'
episode  1639 : first reward = 1.0
episode  1846 : first perfect greedy decode
episode  2405 : stable convergence (100-ep mean ≥ 0.99)
```

1,639 episodes against 8.9 × 10^12 sequences — **1.8 × 10^-8 % of the space explored**. Brute force is excluded. Four seeds: 1360 / 1502 / 1639 / 1702, σ = 132. Reproducible, not luck. A control with a *random* target string converged in 1,771 episodes, so nothing French-specific was leaking.

### And it means nothing

I added a control the spec didn't ask for. Same architecture, same algorithm, same seed — only the reward becomes **all-or-nothing** (1.0 if the sentence is exact, 0 otherwise):

```
[all-or-nothing] ep 10000 | mean reward (100) = 0.0000
[all-or-nothing] ep 20000 | mean reward (100) = 0.0000
[all-or-nothing] ep 30000 | mean reward (100) = 0.0000
                 best ever = 'addahr iddno' (r=0.000) | greedy = 'dddddddddddd'
```

30,000 episodes. Reward **exactly zero from start to finish**. Not one success.

Here's why the graded version works: `R = (1/12) Σ 1[a_t = c_t]` is **decomposable**. In the REINFORCE gradient, only the *t*-th indicator depends on the action at position *t*; the other eleven are uncorrelated with it and cancel in expectation. The 12^12 problem factorizes into **12 independent 12-armed bandits**, each solved in a few hundred draws.

The graded reward doesn't guide a search through 8.9 × 10^12. **It deletes the search.**

### A second reward shape, for comparison

I also ran a Levenshtein-based reward (`1 − edit distance / max length`), more tolerant of positional shifts:

| reward | first reward 1.0 | first perfect greedy | stable convergence |
|---|---|---|---|
| per-position | ep 1,639 | ep 1,846 | ep 2,405 |
| **Levenshtein** | **ep 762** | ep 1,585 | ep 2,073 |
| all-or-nothing | **never** (30,000 ep) | never | never |

Levenshtein finds a first perfect sample **twice as fast**, but converges at roughly the same point — the tolerance helps early exploration and stops mattering once positions are locked.

![Reward curves for the three reward shapes on test 1, 100-episode moving average. The per-position and Levenshtein rewards both climb to 1.0 and converge before episode 2,500. The all-or-nothing reward is a perfectly flat line at zero across all 30,000 episodes.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/usFFvasRjU7BgK3s3J0le.png)

### What the policy actually became

Post-training probability heatmap, greedy mode, one cell per position:

```
P(correct character) per position:
[0.9976, 0.9921, 0.9992, 0.9973, 0.9965, 0.9874,
 0.9960, 0.9945, 0.9937, 0.9887, 0.9729, 0.9918]
mean 0.9923   min 0.9729
```

![Heatmap of the learned probabilities after training on test 1: vocabulary characters on the vertical axis, the twelve sequence positions on the horizontal. Exactly one bright yellow cell per column, outlined in red at the target character, everything else dark. A near-deterministic position-to-character table.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/i7cxsEyvztZHstdjqO9Fg.png)

One bright cell per column, everything else at zero. A near-deterministic position→character table. Which is exactly the problem: a 12-entry lookup table is what it learned, and the transfer test below confirms it.

### And it generalized to nothing

Transfer test: retrain on `le chien dort` starting from the trained weights. Result: **×1.74 faster** than from scratch. Encouraging — until I ran the control.

| transfer to | shared positions | speedup |
|---|---|---|
| `le chien dort` | 5 / 13 | ×1.74 |
| `hclt cncir nd` | 0 / 13 | **×0.91** |

With no positional overlap, transfer is **slower than starting over**. The ×1.74 was entirely the literal shared prefix `le ch`. Zero abstract structure.

![Reward curves for transfer to the perturbed target versus retraining from scratch. The transfer run rises earlier, reaching reward 1.0 around episode 1,600 against 2,780 from scratch — a gap that vanishes entirely once the shared prefix is removed.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/OWx5J0wbGGc6f19Y3HsJx.png)

The hidden-state ablation says the same thing. Zeroing `h` after each position leaves 67% of the continuation intact — and the failures are readable:

```
h→zero after position 1  →  'le dort dort'   (0.70)
h→zero after position 2  →  'le dort dort'   (0.67)
h→zero after positions 0,4,6,7,9,10 → 'le chat dort' (1.00)
```

The space appears **twice** in the target (positions 3 and 8), followed by `c` then `d`. Without `h` the model can't tell which space it just wrote and defaults to `d`. So it learned a near-Markovian character→character lookup table, with the hidden state used only to disambiguate repeats.

**Verdict on test 1**: pure RL succeeded, the sparse-reward objection was confirmed rather than refuted, and nothing generalizable was learned.

---

## 📐 Test 2 — a hand-written grammar, no target sentence

The real question left open: **can you give an agent a dense reward without giving it an oracle that already knows the answer?**

So: a formal grammar with a hand-written parser. Zero AI in the loop. The parser never compares to a target sentence — it checks rules, whatever the agent produces. There are dozens of valid sentences, not one.

```python
# token -> (category, gender, number). These features feed the PARSER only.
# The agent never sees them: to it, a token is an index with no structure.
"le":  ("det", "m", "sg"),   "les": ("det", None, "pl"),   # 'les'/'des' are
"la":  ("det", "f", "sg"),   "des": ("det", None, "pl"),   # gender-neutral
"chat": ("nom", "m", "sg"),  "chats": ("nom", "m", "pl"),
"dort": ("verbe", None, "sg"), "dorment": ("verbe", None, "pl"),
```

Two variants:

| grammar | structure | vocab | space | valid | random hit rate |
|---|---|---|---|---|---|
| short | dét nom verbe | 20 | 8,000 | 48 | 0.600 % |
| long | dét adj nom verbe adv | 31 | 28,629,151 | 288 | 0.001 % |

**The short space is enumerable — 8,000 sequences.** That single fact is what makes everything below possible: every number in this article is *exact*, not estimated from samples. I can compute the policy's true distribution, its true entropy, and the true optimum in closed form.

Reward = mean of three sub-scores (structure OK, det-noun agreement, noun-verb agreement).

### 99.9% grammatical, zero rules learned

The agent nails it. Then I forced the antecedent and measured the consequent:

```
P(noun agrees | determiner forced) = 0.333   = 2/6 determiners
P(verb agrees | noun forced)       = 0.500   = 4/8 nouns
```

Not approximately. **Exactly** 2/6 and 4/8, for every seed on the plateau. Those aren't quality scores, they're **counts**. The agent uses two determiners and four nouns — the all-plural subset — and is perfect inside it. Force it onto `le` and it emits a plural noun.

It found a **degenerate sublanguage**: a subset of the output space where the constraint is *vacuously true*. All-plural sentences satisfy number agreement automatically, so the rule carries no learning signal there.

No bug. No cheating. A verifier satisfied at 99.9% by an agent that never learned what it verifies.

> This is the most transferable thing in the whole project. Any constraint-satisfaction reward admits degenerate sublanguages. **A high score on a rule-based reward does not mean the rule was learned.** The diagnostic is cheap: force the antecedent, measure the consequent.

### The full sweep — 8 entropy coefficients × 3 seeds × 20,000 episodes

| β | validity % | effective modes / 48 | seeds covering both families |
|---|---|---|---|
| 0.0 | 100.0 ± 0.0 | 1.0 ± 0.0 | 0/3 |
| 0.01 | 99.9 ± 0.1 | 6.0 ± 2.8 | 0/3 |
| 0.02 | 99.9 ± 0.1 | 14.1 ± 3.2 | 0/3 |
| 0.05 | 96.0 ± 2.6 | 22.5 ± 1.2 | 0/3 |
| 0.08 | 87.6 ± 6.9 | 37.4 ± 9.2 | **2/3** |
| 0.12 | 57.6 ± 1.7 | 45.3 ± 0.6 | 3/3 |
| 0.2 | 21.3 ± 0.9 | 43.6 ± 2.1 | 3/3 |
| 0.35 | 5.6 ± 0.7 | 44.1 ± 1.5 | 3/3 |

The transition is at β ≈ 0.08, and look at the variance there: **±6.9 on validity, ±9.2 on modes**. At that exact coefficient, seed 0 stays single-family (24.4 modes, 94.9% valid) while seeds 1 and 2 span both (43.1 and 44.8 modes, 89.6% and 78.3% valid) *and* learn the real rule — `P(noun agrees|det)` = 0.911 and 0.837, `P(verb agrees|noun)` = 0.977 and 0.916.

**So the rule is learnable by REINFORCE — just unreliably**, and only in the narrow band where both families survive. My first sweep was single-seed, and it picked β=0.08 based on seed 0 — precisely the unlucky one. That's how a single-seed frontier plot can be not just noisy but *wrong in shape*.

![Left panel: grammatical mass and effective modes against the entropy coefficient, mean and standard deviation over three seeds. Validity holds near 100 percent up to β equals 0.05 then collapses, while effective modes climb steadily. Right panel: the validity by diversity frontier, one line per seed, every point a training run. The optimum at 48 uniform solutions is marked with a dotted line that no run reaches.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/uGwDnI-pU2SIAlMvBYfa1.png)

### The long grammar — where the hit rate collapses

| grammar | reward | validity | distinct valid sentences |
|---|---|---|---|
| long (28.6M space) | graded | 6.4 % | 29 / 288 |
| long (28.6M space) | all-or-nothing | **0.0 %** | 0 / 288 |

At a 0.001% random hit rate, all-or-nothing never gets a signal. Graded barely holds on. (Honest caveat: my long grammar changes *two* variables at once — adverbs grow the space, adjectives also add an agreement rule. Fine for the all-or-nothing conclusion, since only the hit rate matters there; confounded for reading the graded run.)

### The all-or-nothing control *succeeds* here

| reward | validity | modes | P(noun agrees\|det) |
|---|---|---|---|
| graded, β=0.08 | 94.87 % | 24.4 | 0.632 |
| **all-or-nothing, β=0.08** | **99.58 %** | 24.0 | 0.334 |

On the short grammar, the sparse signal beats the shaped one. On the long grammar it collapses to 0% while graded holds 6.4%.

So the deciding variable isn't the *shape* of the reward — it's the **random hit rate**:

| setting | random validity | all-or-nothing |
|---|---|---|
| test 1, 12-char copy | 1.1 × 10^-11 % | total failure |
| test 2, short grammar | 0.6 % | **success (99.58%)** |
| test 2, long grammar | 0.001 % | total failure |

![Two side-by-side panels showing the percentage of grammatically valid sentences over 20,000 episodes. Left, the short grammar: graded and all-or-nothing both climb to nearly 100 percent. Right, the long grammar: the graded reward crawls upward while the all-or-nothing curve stays flat at zero throughout.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/VD-nccXZZ73h8zKqIrfdf.png)

**"Sparse reward" is a bad name.** It conflates two independent things: the reward's shape, and the probability of stumbling on a success. The objection I was given was right about the mechanism and imprecise about the word.

---

## 🔬 The spectrum that replaces the word "sparse"

Here's what I think is the real contribution. A reward is a function on a discrete cube, so you can decompose its variance by interaction order (functional ANOVA):

| reward | order 1 (marginals) | order 2 (pairs) | order 3 (triple) |
|---|---|---|---|
| **graded** | **76.1 %** | 23.9 % | 0.0 % |
| **all-or-nothing** | **4.0 %** | 30.5 % | **65.5 %** |

At a uniform policy, the REINFORCE gradient for position *p* sees only `E[R | token at p]` — every interaction is averaged away. **The gradient is initially blind to everything above order 1.**

So reward shaping doesn't "densify" anything. It **moves variance from high orders down to order 1**. That's the operational definition, and it's measurable.

Structural corollary: the graded reward has **exactly zero** order-3 variance, because it's a sum of per-position terms (order 1) and pairwise agreements (order 2). The all-or-nothing indicator is a *product*, so it dumps 65.5% of its variance at order 3 — invisible at initialization.

And there's a curriculum trap in it. The sequence that maximizes the order-1 signal is:

```
'des chat chantent'  ->  INVALID, R = 0.5000
```

**The first signal the agent follows doesn't point at a solution.**

That order-1 signal also decided which sublanguage the agent collapsed into. Singular nouns score 0.2944 marginally vs 0.2778 for plural — +0.0167 for singular. Cause: my lexicon has **4 singular determiners and only 2 plural ones**, so a random determiner agrees in number with a singular noun 4 times out of 6 versus 2 out of 6. An unintended vocabulary imbalance, computable before any training, dictating the outcome.

---

## 🕵️ Isolating the cause of mode collapse — the part you can't do on a real model

The agent doesn't just restrict itself to one sublanguage. It uses **12–24 of the 48 valid sentences**, never both families. Standard read: "mode collapse, that's RL for you."

I wanted the actual cause. With an enumerable space you can walk the entire elimination tree.

**First, is the collapse even suboptimal?** The loss contains `−β·Σ_t H(a_t|a_<t)`, and the sum of per-step conditional entropies *is* the trajectory entropy. So the objective is `E[R] + β·H(s)`, whose optimum is Gibbs: `π*(s) ∝ exp(R(s)/β)`. All 48 valid sentences have `R = 1` **exactly**, so π* gives them all equal probability — 48 effective modes, 50/50 between families, **at every β**.

That's a free certificate. *Whenever several solutions are exactly tied, max-entropy forces them to be equiprobable, so any deviation proves the optimizer failed — without ever computing the optimum's value.*

I verified it the formalism-free way too: compare `J = E[R] + β·H` for the learned policy versus uniform-over-48. On the plateau (β ≤ 0.08), uniform-over-48 wins on **both** terms. Proven suboptimal. (Above β = 0.12 the argument stops working — the learned policy spreads enough mass over invalid sequences to be *more* entropic. I only claim the plateau.)

**Then the elimination — and it does not land where I first thought.**

My initial read, from the β=0.01 runs alone, was "sampling noise cleared, the autoregressive factorization is guilty, no variance reduction will save this." Then the β=0.05 exact-gradient run came in and broke it. Here is what the evidence actually supports:

| suspect | verdict | evidence |
|---|---|---|
| model capacity | **cleared** | supervised fit → 100 %, 48.0 modes, 3/3 seeds |
| the objective itself | **cleared** | tabular parameterization → 48.0 modes, exactly the Gibbs optimum, every β |
| instability of the optimum | **cleared** | started from ideal, stays in a 2-family region |
| autoregressive factorization | **guilty below β≈0.05** | β=0.01–0.02: GRU with a *perfect* gradient gets 12–24 modes where tabular gets 48.0 |
| the sampled procedure | **guilty above β≈0.05** | β=0.05 and 0.08: GRU with a perfect gradient hits **48.0 modes on 3/3 seeds each**; sampled gets 21–45 |

**Two regimes, two different causes, with a sharp transition between β=0.02 and β=0.05.** Below it, the shared-parameter factorization blocks the optimum even when the gradient is exact — no estimator can help. Above it, that blockage vanishes entirely: the exact-gradient GRU reproduces the closed-form Gibbs optimum to two decimals on six independent runs, and everything that still fails is the sampled training procedure.

And I have to flag a confound in my own comparison rather than let it slide. The exact-gradient run optimizes the true objective `E[R] + β·H(p)`. The sampled run uses the standard entropy bonus, which regularizes entropy **at visited states** — a biased estimator of ∇H. **The two runs are not optimizing the same thing.** So the β=0.05 gap conflates sampling noise with estimator bias, and I cannot separate them with what I ran. The clean, surviving claim is the low-β one, where tabular and GRU are compared under an identical objective and an identical exact gradient.

The tabular run is what makes that low-β claim airtight. Same objective, same exact analytic gradient, but one free logit per sequence — no shared parameters:

```
  beta  seed  valid%  modes  unif%   sg%    pl%
  0.01     0  100.00   48.0  100.0  50.0   50.0     <- tabular
  0.02     0   99.98   48.0  100.0  50.0   50.0
  0.05     0   94.60   48.0  100.0  50.0   50.0

  GRU + exact gradient, beta=0.01 : 12.0 modes, 100% singular
```

100.00 / 99.98 / 94.60 versus the analytically computed 100.00 / 99.96 / 94.59. The tabular policy lands **on** the optimum.

**The mechanism**: in a GRU, the six conditionals `P(· | determiner)` run through *shared parameters*. Whichever gets the most gradient early shapes the hidden state, and the others inherit a representation tuned for it. It's rich-get-richer at the level of the **representation**, not the probabilities — a different mechanism from the one usually invoked.

Confirmed by localization. Freeze one position's marginal to its ideal value — the token forced *during* generation so the continuation conditions on it, and that position excluded from the REINFORCE term — and retrain:

```
        frozen   modes   sg%    pl%   P(noun|det)  P(verb|noun)
          none    11.5    0.0  100.0      0.333        0.500
   pos0 (det)     30.3   61.9   38.1      0.999        0.924   <- rule learned, both families
   pos1 (noun)    17.7    0.2   99.8      0.005        0.875
   pos2 (verb)     8.0  100.0    0.0      0.500        0.009
```

**Freezing the determiner marginal alone makes the agent learn the full agreement rule** — `P(noun agrees | det)` goes from 0.333 to **0.999** across all six determiners — and keeps both families alive. The collapse lives in position 0.

The pos1 and pos2 rows are expected and uninformative: forcing the noun to an *independent* draw destroys the det→noun dependency by construction, hence the 0.005. Same for the verb. (One caveat on my own protocol: the validity column for these rows is an artifact — the frozen position is excluded from the gradient during training but left free at evaluation, so it emits garbage. Only the conditionals, normalized inside each determiner, survive that flaw.)

**The practical consequence I wasn't looking for**: RL doesn't *create* a distribution, it *refines* one. Applied after a good initialization it preserves diversity; applied from scratch it destroys it. That's exactly the order RLHF is used in — after pretraining. The usual justification ("pretraining supplies the knowledge") is incomplete: it also supplies **the distribution RL cannot build on its own.**

---

## 😬 The baseline nobody runs

The untrained network has 0.60% valid mass and **47.5 effective modes out of 48** — because a near-uniform policy restricted to the valid set is near-uniform. So sample-and-filter gives 100% validity and ~47.5 modes by construction, at ~167 draws per accepted output.

| method | validity | effective modes |
|---|---|---|
| **rejection sampling from the untrained net** | 100 % | ~47.5 |
| REINFORCE β=0.02, 20,000 episodes | 99.99 % | 18.6 |
| REINFORCE β=0.08, 20,000 episodes | 94.87 % | 24.4 |

Twenty thousand episodes of training, beaten on both axes by a network that learned nothing plus an `if`.

Two honest caveats, because this result is easy to overstate. My diversity metric is *maximized by construction* for rejection sampling — what saves the comparison is that uniform-over-48 is also REINFORCE's own optimum, so this measures optimization failure, not general superiority. And rejection needs the verifier **at inference**, which is exactly what you don't have for real language.

What survives: *when the target is already reachable by random sampling, policy-gradient training does no better than a filter and destroys the diversity the filter preserves for free.*

---

## 🔧 The fix that came out of the diagnosis

The sweep shows two incompatible regimes: low β is grammatical but single-family, high β covers both families but wrecks grammaticality. **Nobody chains them, because you first have to know the family structure exists.**

Geometric annealing of β over 30,000 episodes:

| method | validity | modes / 48 | families |
|---|---|---|---|
| β constant 0.02 | 99.99 % | 18.6 | 1 |
| β constant 0.12 | 57.13 % | 45.9 | 2 |
| **anneal 0.2 → 0.01** | **99.97 %** | **45.3** | **2** |
| **anneal 0.12 → 0.02** | **99.96 %** | **45.3** | **2** |

![Two panels tracking both annealing schedules over 30,000 episodes. Left, grammatical mass climbs from under 40 percent to 99.97 percent, crossing the β constant 0.12 reference line early and reaching the β constant 0.02 line at the end. Right, effective modes stay between 40 and 46 throughout, far above the 18.6 of β constant 0.02, settling exactly on the 45.35 line predicted by a uniform distribution over the six determiners.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/PHEThtDXgr-EmS1FVxG5t.png)

**It dominates both constant regimes simultaneously**, reproduced on two schedules. At high β both families get gradient, so all six conditionals get trained; by the time β drops, the shared representation is already formed for all of them. **The anneal doesn't fight the collapse — it prevents it from forming.**

Why 45.3 and not 48? Both anneals and the from-ideal run converge to a 2/3–1/3 family split. And 2/3–1/3 = 4/6–2/6 is exactly what you get when `P(determiner)` is **uniform over the 6 determiners**:

```
24 sg sentences at (2/3)/24, 24 pl at (1/3)/24
H = ⅔·log₂(36) + ⅓·log₂(72) = 5.5032 bits
2^H = 45.35 effective modes
```

**45.3 — exactly the measured value.** The residual ceiling is a target mismatch: **the per-token entropy bonus aims at uniformity over tokens, not over sequences.** Those coincide only if every prefix has the same number of valid completions — false here, since `les` admits 12 and `le` admits 6. Predicted fix, untested: weight the entropy by valid-completion count, or regularize sequence entropy instead.

---

## 📈 Every number, raw

Nothing hidden. All exact (enumeration), unless marked otherwise.

### Test 1 — anti-cheat controls

| control | result |
|---|---|
| episodes vs space | 1,639 for 8.9 × 10^12 → **1.8 × 10^-8 %** explored |
| 4 seeds | 1360 / 1502 / 1639 / 1702 → mean 1551, **σ = 132** (8.5%) |
| random target ` eaiea innhh` | 1,771 episodes — same order, no French-specific leak |
| baseline leak check | `deque` of 100 past rewards, advantage `.detach()` — no channel to the target outside the reward function |

### Test 1 — hidden-state ablation, position by position

`h` corrupted *after* producing position *t*; figure = accuracy of positions *t+1…11*.

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h→zero | 1.00 | 0.70 | 0.67 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| h→noise | 0.91 | 0.70 | 0.67 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |

Means: **0.6697** (zero) / **0.7069** (noise).

### Test 2 — the Gibbs optimum, and the shaping tax

`π*(s) ∝ exp(R(s)/β)`. Since all 48 valid sentences have `R = 1`, **π\* always has 48 effective modes and a 50/50 family split, at every β.** So the "modes" and "sg/pl" columns of the optimum are constant and I omit them.

| β | π* validity (graded) | achieved (graded) | achieved modes | π* validity (all-or-nothing) |
|---|---|---|---|---|
| 0.01 | 100.00 % | 99.84 % | 9.9 | 100.00 % |
| 0.02 | 99.96 % | 99.99 % | 18.6 | 100.00 % |
| 0.05 | 94.59 % | 92.65 % | 23.8 | 100.00 % |
| 0.08 | **79.12 %** | **94.87 %** | 24.4 | **99.94 %** |
| 0.12 | 52.41 % | 57.13 % | 45.9 | 96.17 % |
| 0.2 | 19.04 % | 20.59 % | 41.2 | 47.25 % |
| 0.35 | 5.51 % | 5.27 % | 43.5 | 9.51 % |
| 0.5 | 2.98 % | 3.01 % | 43.6 | 4.27 % |

![Grammatical mass against the entropy coefficient for three curves: the closed-form optimum of the graded reward, the closed-form optimum of the all-or-nothing reward, and what REINFORCE actually reaches. The all-or-nothing optimum sits above the graded optimum everywhere. Past β equals 0.08 the reached curve rises above the graded optimum it is supposed to be aiming at.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/FM23qKmBY15U4Ar38OJY7.png)

Three things fall out of this table:

1. **The shaping tax is real and large.** At β=0.08 the graded reward's own optimum caps at 79.12% while all-or-nothing's sits at 99.94%. All-or-nothing has a strictly better optimum at every β.
2. **From β=0.08 up, the learned policy is *more grammatical than the optimum of its own objective*** (94.87% vs 79.12%). Not a contradiction — it buys validity by giving up entropy, so it's worse on the total objective. But the consequence is perverse: **if REINFORCE actually succeeded at β=0.08, validity would drop from 95% to 79%.** The optimization failure was masking the shaping tax.
3. **Above β=0.12 the achieved and the optimum nearly coincide** (mode gap falls from 24–38 down to 2–7). So the "cliff" isn't a tradeoff appearing — **it's the optimizer finally succeeding**, and revealing that the optimum at that β is bad.

### Test 2 — the dominance check, verified rather than asserted

Reference: uniform over the 48, `E[R] = 1`, `H = ln(48) = 3.8712` nats.

| β | E[R] | H (nats) | J learned | J uniform-48 | verdict |
|---|---|---|---|---|---|
| 0.0 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | **tie** |
| 0.01 | 0.9993 | 2.2983 | 1.0223 | 1.0387 | suboptimal (proven) |
| 0.02 | 1.0000 | 2.9255 | 1.0585 | 1.0774 | suboptimal (proven) |
| 0.05 | 0.9830 | 3.4430 | 1.1552 | 1.1936 | suboptimal (proven) |
| 0.08 | 0.9679 | 3.5369 | 1.2509 | 1.3097 | suboptimal (proven) |
| 0.12 | 0.8527 | 5.5942 | 1.5240 | 1.4645 | **argument fails** |
| 0.2 | 0.5835 | 7.4569 | 2.0749 | 1.7742 | **argument fails** |
| 0.35 | 0.3254 | 8.6592 | 3.3561 | 2.3549 | **argument fails** |
| 0.5 | 0.2563 | 8.8502 | 4.6814 | 2.9356 | **argument fails** |

I had asserted "dominates strictly on both terms" without computing the entropy term. The learned policy spreads mass over invalid sequences, and **that mass contributes to its entropy**. Above β=0.12 it's *more* entropic than uniform-over-48 and the argument proves nothing. And at β=0 the margin is exactly zero — with no entropy term, collapsing onto one valid sentence *is* the optimum. **Three regimes, not two.**

### Test 2 — the capacity probe, in full

Supervised fit of the same GRU toward uniform-over-48, 3,000 steps, 3 seeds:

```
theoretical optimum of -log P : ln(48) = 3.8712
reached                       : 3.8714  (seeds 0, 1, 2)

before fit : 0.60 % valid mass | 47.5 effective modes
after fit  : 100.00 % valid    | 48.0 modes | 100.0 % uniformity | sg 49.9 / pl 50.1
             P(noun agrees|det) = 1.0000 | P(verb agrees|noun) = 1.0000

P(determiner) reached : des 0.24793  les 0.25324
                        le  0.12499  la  0.12383  un 0.12443  une 0.12557
P(determiner) target  : les/des 0.25 each, le/la/un/une 0.125 each
```

Three decimals from theory, on every seed. **The ideal policy is representable.** (What it does *not* establish: comparable optimization difficulty. The supervised fit sees all 48 targets every step in full-batch. It's a capacity probe, not a difficulty comparison — those are two different things and I initially blurred them.)

### Test 2 — exact-gradient runs

Analytic gradient of `E[R] + β·H`, zero sampling:

| β | seed | validity | modes | sg % | pl % | P(noun agrees\|det) |
|---|---|---|---|---|---|---|
| 0.01 | 0 | 100.00 | 12.0 | 100.0 | 0.0 | 0.333 |
| 0.01 | 1 | 100.00 | 12.0 | 100.0 | 0.0 | 0.333 |
| 0.01 | 2 | 100.00 | 12.0 | 100.0 | 0.0 | 0.333 |
| 0.02 | 0 | 99.95 | 24.0 | 100.0 | 0.0 | 0.666 |
| 0.02 | 1 | 99.93 | 12.0 | 100.0 | 0.0 | 0.334 |
| 0.02 | 2 | 99.94 | 24.0 | 100.0 | 0.0 | 0.667 |
| **0.05** | **0** | **94.60** | **48.0** | **50.0** | **50.0** | **0.943** |
| **0.05** | **1** | **94.60** | **48.0** | **50.0** | **50.0** | **0.943** |
| **0.05** | **2** | **94.59** | **48.0** | **50.1** | **49.9** | **0.943** |
| **0.08** | **0** | **79.12** | **48.0** | **49.9** | **50.1** | **0.823** |
| **0.08** | **1** | **79.13** | **48.0** | **49.9** | **50.1** | **0.823** |
| **0.08** | **2** | **79.10** | **48.0** | **50.2** | **49.8** | **0.823** |

Two different things happen here, and I got the first read wrong.

**At β=0.01 and 0.02, removing the noise doesn't help.** Identical to the decimal across three seeds at β=0.01. It also flips which family wins: sampled REINFORCE sometimes lands on plural, the exact gradient lands on singular **every time**, following the order-1 signal. Noise was the only thing that ever overcame the marginal bias. And the `P(noun agrees|det)` column stays at 0.333 = 2/6 or 0.666 = 4/6 — even with a perfect gradient the agent uses two or four of the six determiners, never all six.

**From β=0.05 up it helps completely, on every seed.** 48.0 effective modes, 100 % uniformity, a 50/50 family split — six runs out of six. And the validity numbers land *on the closed-form optimum*: 94.60 / 94.60 / 94.59 % against a computed Gibbs value of 94.59 %, and 79.12 / 79.13 / 79.10 % against 79.12 %. **The exact-gradient GRU reproduces the analytic optimum to two decimals, six times independently.** It doesn't approach the optimum, it *is* the optimum.

Sampled REINFORCE at those same β gets 21–24 modes at 0.05 (one family, 3/3 seeds) and 24–45 at 0.08.

So the honest version of "is it the noise or the geometry?" is: **there is a sharp transition, and I published the low-β answer as if it were the whole answer.**

| | exact gradient | sampled |
|---|---|---|
| β ≤ 0.02 | collapses (12–24 modes) | collapses (4–19 modes) |
| β ≥ 0.05 | **optimum, 48.0 modes, 3/3 seeds** | still collapsing (21–45 modes) |

Below β≈0.05 the shared-parameter factorization blocks the optimum outright, and no estimator can help. Above it the parameterization stops being an obstacle at all, and everything that remains is the sampled procedure. **Sampling shifts the entropy pressure you need by roughly a factor of 3–5**: the exact gradient needs β≈0.05 to open both families, the sampled version needs β≈0.12 — and by then the objective's own optimum has degraded to 52 % validity.

### Test 2 — the order-1 signal that picks the family

| noun family | marginal `E[R \| noun]` |
|---|---|
| singular nouns | **0.2944** |
| plural nouns | 0.2778 |
| gap | **+0.0167 for singular** |

Arithmetic, verified end to end. Mean det-noun agreement credit: `chat` (sg) gets le 1.0 + la 0.5 + un 1.0 + une 0.5 + les 0.5 + des 0.5 = **0.667**; `chats` (pl) gets 0.5 + 0 + 0.5 + 0 + 1.0 + 1.0 = **0.500**. Gap 0.167 on that sub-score → 0.0556 on the total (mean of 3) → diluted by 6/20, since only sequences with a determiner at position 0 count → **0.0167**. Exactly the measured value.

Root cause: **my lexicon has 4 singular determiners and only 2 plural ones.** An unintended vocabulary imbalance, computable before any training, dictating which sublanguage the agent collapses into.

### Test 2 — the sweep, all 24 raw runs

The synthesis above hides the seed-level story, so here it is unaggregated.

| β | seed | validity % | modes | unif % | sg % | pl % | P(noun\|det) | P(verb\|noun) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0 | 100.00 | 1.0 | 0.0 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.0 | 1 | 99.99 | 1.0 | 0.1 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.0 | 2 | 100.00 | 1.0 | 0.0 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.01 | 0 | 99.84 | 9.9 | 59.1 | 0.0 | 100.0 | 0.333 | 0.500 |
| 0.01 | 1 | 100.00 | 4.0 | 35.8 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.01 | 2 | 99.96 | 4.0 | 35.8 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.02 | 0 | 99.99 | 18.6 | 75.5 | 0.0 | 100.0 | 0.333 | 0.500 |
| 0.02 | 1 | 99.83 | 11.7 | 63.6 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.02 | 2 | 99.97 | 12.0 | 64.2 | 100.0 | 0.0 | 0.333 | 0.500 |
| 0.05 | 0 | 92.65 | 23.8 | 81.9 | 0.0 | 100.0 | 0.334 | 0.500 |
| 0.05 | 1 | 96.43 | 22.8 | 80.8 | 100.0 | 0.0 | **0.649** | 0.504 |
| 0.05 | 2 | 99.02 | 21.0 | 78.6 | 100.0 | 0.0 | **0.663** | 0.500 |
| 0.08 | 0 | 94.87 | 24.4 | 82.5 | 0.8 | 99.2 | 0.632 | 0.608 |
| 0.08 | 1 | 89.56 | **43.1** | 97.2 | 46.8 | 53.2 | **0.911** | **0.977** |
| 0.08 | 2 | 78.27 | **44.8** | 98.2 | 60.8 | 39.2 | **0.837** | **0.916** |
| 0.12 | 0 | 57.13 | 45.9 | 98.9 | 54.8 | 45.2 | 0.684 | 0.885 |
| 0.12 | 1 | 59.79 | 45.5 | 98.6 | 50.1 | 49.9 | 0.709 | 0.880 |
| 0.12 | 2 | 55.81 | 44.5 | 98.0 | 52.8 | 47.2 | 0.650 | 0.864 |
| 0.2 | 0 | 20.59 | 41.2 | 96.0 | 49.0 | 51.0 | 0.481 | 0.546 |
| 0.2 | 1 | 22.57 | 43.4 | 97.4 | 51.6 | 48.4 | 0.498 | 0.582 |
| 0.2 | 2 | 20.78 | 46.3 | 99.1 | 49.3 | 50.7 | 0.459 | 0.569 |
| 0.35 | 0 | 5.27 | 43.5 | 97.4 | 38.9 | 61.1 | 0.283 | 0.350 |
| 0.35 | 1 | 6.49 | 42.7 | 97.0 | 53.6 | 46.4 | 0.343 | 0.351 |
| 0.35 | 2 | 4.99 | 46.1 | 99.0 | 44.9 | 55.1 | 0.272 | 0.367 |

Two things only visible at seed level.

**The 0.649 and 0.663 at β=0.05** aren't partial rule learning — they're **4/6 = 0.667**. Those seeds landed on the *singular* family, which has four determiners; seed 0 landed on plural, which has two, hence 0.333. Same policy quality, different family size. Further proof these numbers are counts, not scores.

**β=0.08 is where seeds diverge qualitatively**: seed 0 stays single-family at 24.4 modes, seeds 1 and 2 span both at 43.1 and 44.8 modes *and* reach P(noun|det) = 0.911 / 0.837. Note that seeds 1–2 at β=0.08 **dominate seed 0's β=0.12 point on both axes** — which is exactly why a single-seed frontier plot can be wrong in shape, not merely noisy.

### Test 2 — separating within-family diversity from between-family coverage

Effective modes conflate two different problems. A determiner can be perfectly exploited *and* almost never emitted. So: conditional entropy `H(noun | determiner)`, per determiner, at β=0.08 seed 0.

| det | policy mass | H (bits) | H max | "saturation" | compatible nouns |
|---|---|---|---|---|---|
| **les** | 0.5049 | 2.027 | 2.000 | 101.4 % | 4 |
| **des** | 0.4488 | 2.017 | 2.000 | 100.9 % | 4 |
| un | 0.0134 | 0.831 | 1.000 | 83.1 % | 2 |
| la | 0.0095 | 2.181 | 1.000 | **218.1 %** | 2 |
| une | 0.0080 | 0.787 | 1.000 | 78.7 % | 2 |
| le | 0.0029 | 0.137 | 1.000 | 13.7 % | 2 |

`les` and `des` carry **95% of the mass** and saturate their 2 bits — inside the plural family the agent is genuinely diverse. The other four are starved.

And my own metric is mislabelled: `saturation` exceeds 100% because H is computed over all 8 nouns while H max uses only the *compatible* ones. Above 100% means **mass leaking onto incompatible nouns** — a failure, not super-saturation. `la` at 218% with a mass of 0.0095 is pure noise: the agent has no idea what follows `la` because it never says it.

### Test 2 — full ANOVA breakdown

| reward | order 1 total | pos0 (det) | pos1 (noun) | pos2 (verb) | order 2 total | 0-1 | 0-2 | 1-2 | order 3 |
|---|---|---|---|---|---|---|---|---|---|
| **graded** | **76.1 %** | 22.3 | 34.2 | 19.6 | 23.9 % | 10.0 | 0.0 | 14.0 | **0.0 %** |
| **all-or-nothing** | **4.0 %** | 1.7 | 0.9 | 1.4 | 30.5 % | 11.9 | 11.4 | 7.1 | **65.5 %** |

Note `pos0-2` = 0.0% for the graded reward: determiner and verb have **no direct agreement constraint** — they only interact through the noun. The decomposition recovers the grammar's dependency structure from the reward alone.

Order-1 greedy sequences: `des chat chantent` (graded, **invalid**, R=0.50) and `des chat chante` (all-or-nothing, **invalid**, R=0.83).

### Test 2 — the generalization tests, in full

**Excluded combination** (`des fleurs` never rewarded; both tokens trained elsewhere):

```
P(fleurs | des) after exclusion  : 0.2248
P(other plural nouns | des)      : chats 0.2802, chiens 0.1792, tables 0.3085  (mean 0.2560)
P(fleurs | des) reference run    : 0.2861
ratio excluded / others          : 0.878
P(des) in the policy             : 0.4326   (prefix is well used, test is not on a rare branch)
```

Reads as compositional generalization — but the spread among *non-excluded* nouns (0.179 to 0.309, i.e. ratios 0.70 to 1.21) is **wider than the effect measured**. One seed. No conclusion available.

**Never-seen token** (`fleurs` fully masked during training), P(correctly-agreeing verb | noun forced):

```
chat    0.0149      chats   0.9922
chien   0.0013      chiens  0.9958
fleur   0.0006      fleurs  0.9966  <- NEVER SEEN
table   0.0003      tables  0.9953

mean over seen nouns : 0.4286      chance : 0.5000
```

The 0.9966 looks like generalization and isn't. Singular nouns give ~0.005 — **the agent emits a plural verb regardless of the noun.** The mean over seen nouns is *below* chance. Confounded by the family collapse.

### Test 2 — the stability trace (starting *from* the ideal policy)

| episode | validity | modes | KL to ideal (bits) | sg / pl |
|---|---|---|---|---|
| 0 (start) | 100.00 % | 48.0 | 0.000 | 49.9 / 50.1 |
| 250 | 99.96 % | 44.0 | 0.126 | 66.7 / 33.3 |
| 2,250 | 99.73 % | 31.7 | 0.597 | 84.4 / 15.6 |
| 4,250 | 99.96 % | 45.1 | 0.089 | 66.7 / 33.3 |
| 6,250 | 99.95 % | 45.4 | 0.082 | 66.7 / 33.3 |
| 8,250 | 99.84 % | 26.7 | 0.844 | 69.9 / 30.1 |
| 12,250 | 99.94 % | 39.9 | 0.265 | 75.4 / 24.6 |
| 16,250 | 99.71 % | 35.5 | 0.437 | 76.1 / 23.9 |
| 18,250 | 99.91 % | 43.0 | 0.159 | 61.6 / 38.4 |

**It leaves the optimum within 250 episodes** and oscillates around the 45.3 attractor, with excursions down to 26.7. So the optimum is *not* a fixed point — the correct statement is: *the optimum is unstable, but the basin it falls into (45.3 modes, both families) is incomparably better than anything reachable from random init (11.5–18.6 modes, one family).* I'd first written "it stays", which was too simple.

### Test 2 — the optimum is a waypoint, not a destination

Same measurement from random init, β=0.02:

```
maximum effective modes : 24.0 at episode 4,750
minimum KL to ideal     : 1.0000 bits at episode 11,500
final state             : 11.5 modes, KL 2.0611 bits
=> early stopping would beat convergence by +12.5 modes
```

Diversity **peaks mid-training and then degrades**. Training past ~episode 4,750 destroys half of it. Caveat before anyone quotes this: single seed, and I observed run-to-run variation at nominally identical settings (11.5 here vs 18.6 in the sweep), most likely torch CPU multithread non-determinism. I have not replicated it and I don't claim it as established.

### Test 2 — the two null results, in full

**PCA on the training trajectory** (81 exact-distribution snapshots × 3 seeds):

```
CP1 36.2 %   CP2 20.3 %   CP3 13.5 %   CP4 7.2 %   CP5 4.8 %   CP6 4.3 %
90 % of the movement -> 8 dimensions
99 %                 -> 21 dimensions
99.9 %               -> 33 dimensions
```

I bet on 2–3 and a drawable phase portrait. Wrong. The CP1-CP2 plane holds 56.5% of the movement, so the plot exists but can't carry an argument alone.

**Except that when I drew it anyway, it showed something I wasn't looking for.**

![Three training trajectories projected onto the first two principal components of exact distribution space. All three start circles sit stacked directly on top of the gold star marking the optimum, then the three lines travel outward to three widely separated square end points. Only 56.5 percent of the movement lives in this plane.](https://cdn-uploads.huggingface.co/production/uploads/663ce1f27e7bc3d3e4e5074e/UH4mz9ziqIQLPon3JsIS9.png)

All three random initializations start **stacked on the optimum**. Measured in that plane:

```
distance to the optimum — start: 0.001    end: 0.212
```

Training moves the policy **200× further from the ideal distribution than where it started.** It's the picture version of the rejection-sampling result: along the dominant directions of training variation, the untrained network *is* the target, and every episode of REINFORCE walks away from it.

Two honest qualifications. This is a 2D projection holding 56.5% of the movement, so it is not the full distance — the untrained net is obviously terrible on *validity*, which is what the remaining 43.5% contains. What the projection isolates is the **diversity** axis, and on that axis training is pure regression. Which is exactly what the effective-modes numbers said numerically (47.5 → 9.9); the plot just makes it geometric.

**Search for conserved functionals** — amplitude of variation of each category × position mass:

| position | category | start | end | amplitude |
|---|---|---|---|---|
| 0 | det ✓ | 0.2893 | 0.9998 | 0.72082 |
| 2 | verb ✓ | 0.3017 | 1.0000 | 0.70015 |
| 1 | noun ✓ | 0.4059 | 1.0000 | 0.59502 |
| 0 | noun ✗ | 0.4079 | 0.0002 | 0.41399 |
| 2 | noun ✗ | 0.4048 | 0.0000 | 0.40624 |
| 0 | verb ✗ | 0.3028 | 0.0000 | 0.30684 |
| 1 | verb ✗ | 0.3020 | 0.0000 | 0.30518 |
| 1 | det ✗ | 0.2921 | 0.0000 | 0.29447 |
| 2 | det ✗ | 0.2935 | 0.0000 | 0.29390 |

**Zero** functionals vary by less than 0.02. No invariant. (Useful by-product: at init, P(det at pos 0) = 0.289 ≈ 6/20 and P(noun at pos 1) = 0.406 ≈ 8/20 — the random network is indeed uniform over tokens, which independently confirms the basis of the rejection-sampling calculation.)

---

## 🪦 Five hypotheses I formed, and what killed them

I kept a lab notebook where refuted hypotheses stay in, with their date of death. A wrong hypothesis you can date tells you which *kind* of reasoning fooled you. A clean conclusion with no history tells you nothing.

### 1. "The plural basin is wider" — killed by seeds

Seeing the agent lock onto the plural family, I produced a cause within a second: `les` and `des` are gender-neutral, so they accept all 4 plural nouns while `le` only accepts 2 masculine singulars. Wider basin → early exploration falls in.

**Killed by**: three extra seeds at the same β. Three out of four went **singular**.

The explanation was compatible with every piece of data I had, mechanically plausible, and produced instantly. *That speed is what should have warned me* — it came from a narrative, not a test. A cause that "fits" almost always fits.

### 2. "The solution set is disconnected in policy space" — killed by the capacity probe

I argued that moving mass from `les …` to `le …` necessarily requires putting mass on invalid mixtures like `le chats dorment`, so covering both families *must* cost grammaticality.

**Killed by**: the same GRU, fitted in supervised mode, reaches 100% valid mass, 48.0 modes and a 49.9/50.1 split in 500 steps, 3 seeds out of 3.

I had reasoned on the **marginal** at position 0 and forgotten the policy is **conditional**. The hidden state carries the emitted determiner; it only needs to encode 6 values in 128 dimensions. A policy can put 25% on `les` and 12.5% on `le` and stay perfectly valid in both.

### 3. "The held-out token test will give chance (0.5)" — killed, and I was right for the wrong reason

I predicted that a noun never seen in training would give P(plural verb) ≈ 0.5, since its embedding is untrained.

**Measured**: 0.9966.

My practical conclusion ("this test measures nothing") held, but the mechanism I'd identified was wrong. The 0.9966 isn't generalization — the agent emits a plural verb **regardless of the noun**, and trained singular nouns give 0.0003 to 0.0149. Mean over seen nouns: 0.4286, *below* chance. The test is confounded by the family collapse, not by the untrained embedding.

Had I only checked the verdict, I'd have kept a false explanation while believing it confirmed.

### 4. "The validity/diversity tradeoff is a property of the task" — killed twice

This is the bad one, because I was about to write it in a verdict. Refuted independently by the Gibbs computation (the optimum has 48 modes and 50/50 at *every* β) and by the dominance argument (uniform-over-48 beats the learned policy on both terms of the objective). It's an optimization failure, not a property of the problem.

### 5. Retroactive correction to test 1

I had written: *"the blockage is on obtaining the signal, never on optimization."*

**False as soon as there are multiple solutions.** True for a single target, where there's nothing to distribute. Test 2 shows an optimization that fails while the signal is perfect.

### And one hypothesis that came back from the dead — twice relocated

The "wider basin" story wasn't wrong, it was **plugged into the wrong variable, with the wrong sign**. The same structural fact produces two opposite effects:

| effect | mechanism | consequence |
|---|---|---|
| on **family choice** | 4 singular determiners vs 2 plural → more partial credit to singular | order-1 signal pushes **singular** |
| on **within-family diversity** | `les`/`des` gender-neutral → 4 nouns each vs 2 | the **plural** family is richer (18.6 modes vs 11.7) |

It took three corrections: refuted by seeds, relocated to within-family diversity, then sign-flipped by the order-1 computation. **A refuted hypothesis isn't always garbage — sometimes it's just wired to the wrong quantity.**

---

## 📊 Received wisdom, run through the enumeration

Rule I set myself here: **report the ones that hold too.** A list containing only refutations is evidence that you went looking for refutations, not that you measured.

### ❌ "A denser reward is better"

Test 1 seemed to establish it. Test 2 contradicts it twice: all-or-nothing gets **99.58%** vs graded's **94.87%** on the short grammar, and its *optimum* is better at every β (99.94% vs 79.12% at β=0.08). Shaping moved the destination, downward.

### ❌ "The entropy bonus prevents mode collapse"

At β=0.01 the agent ends at **9.9 effective modes**. Random initialization has **47.5**. The bonus slows collapse, it doesn't prevent it — and real coverage only arrives at a β where validity has already collapsed.

### ❌ "Training improves the thing you're measuring" — the most striking one

On the diversity metric itself, **the untrained network beats every trained one on the plateau**: 47.5 modes versus 4.0 to 24.4. Training *destroys* diversity while improving validity. Any diversity number reported without the valid-mass number beside it is uninterpretable — and that's exactly the frontier plot I nearly published.

### ⚠️ "Mode collapse is a pathology" — mis-framed rather than false

At β = 0, `J = E[R]` alone, and a policy concentrated on one valid sentence achieves `E[R] = 1` — the **exact optimum**. Collapse there isn't an optimizer defect, it's correct satisfaction of an objective that never asked for diversity. We call "pathology" the fact that the objective didn't contain what we wanted.

### 🗑️ "Sparse reward" — a concept to throw away

Covered above: the deciding variable is the random hit rate, not the reward's shape. `Sparse` isn't a property of a reward, it's a property of the triple reward × space size × initial policy.

### ❌ "A high score proves the rule was learned"

99.9% grammaticality with `P(noun agrees | det) = 0.333`. The single most transferable finding here.

### ✅ What held — and this is what makes the rest credible

- **REINFORCE with a baseline reduces variance** — works, no surprise.
- **An autoregressive GRU can represent the target distribution** — confirmed exactly: `-log P` = 3.8714 against the theoretical ln(48) = 3.8712, with P(determiner) within three decimals of the theoretical 0.25 / 0.25 / 0.125 × 4.
- **Space size governs all-or-nothing failure** — confirmed across three orders of magnitude.
- **Max-entropy optima are Gibbs** — confirmed numerically over 8,000 sequences and 8 values of β.

---

## 🔍 Measurement traps I walked into

**The metric that measures nothing.** "Coverage" — distinct valid sentences among 500 samples — says nothing about their distribution. An agent producing 40 distinct sentences, one at 60% and 39 at 1%, scores the same as a uniform one. Fixed by computing effective modes `2^H` exactly. Then the *next* trap: random init scores 47.5 on that too.

**The number that's identical everywhere.** "First valid sentence at episode 45" appeared in every short-grammar run, every β, all-or-nothing included. Tempting read: "early exploration is efficient." False — those runs share seed 0 and the first 45 episodes barely move the weights. **The number measures the shared initialization, not the agent.**

**The statistic I compared to the wrong thing.** I then wrote that at 0.6% random validity the first success "should" land near episode 167, so the initial network must be non-uniform. Also false — I compared a single draw to the **mean of a geometric distribution**. The median is 116, and P(first success ≤ 45) = 1 − 0.994^45 = **24%**. A perfectly ordinary draw. I had manufactured a phenomenon by confronting an observation with the wrong statistic, then started hunting for its cause.

**My own mislabelled metric.** In the H(noun | determiner) table, the `saturation %` column exceeds 100% (218% for `la`). Cause: H is computed over all 8 nouns while H_max uses only the *compatible* ones. A value above 100% therefore signals **mass leaking onto incompatible nouns** — a failure, not super-saturation. Badly named by me.

**Statistical power I never checked.** The pair-exclusion generalization test gave a ratio of 0.878, read as "compositional generalization". But the spread among *non-excluded* nouns runs from 0.179 (`chiens`) to 0.309 (`tables`) — ratios of 0.70 to 1.21. **The measured effect is smaller than the natural variability.** One seed. No conclusion available.

---

## ❓ What's still open

Things I could not settle, listed so nobody quotes this article as if they were.

- **"Fails" or "is slow"?** Every run is 20,000 episodes. I established that REINFORCE doesn't *reach* the optimum on the plateau; I have no data on 10^6 episodes. Decisive cheap test: one run at β=0.02 over 200,000 episodes, plotting effective modes over time. If the curve plateaus it's a fixed point; if it's still climbing, the whole framing changes.
- **"Pure RL" is not "REINFORCE".** My biggest blind spot. I'm testing a 1992 algorithm and drawing conclusions about "RL". Mode collapse is a pathology **specific** to on-policy methods that maximize expected reward. An objective that samples *proportionally* to reward — GFlowNets are built for exactly this — has no reason to lock a family, and my 48 tied solutions are their canonical use case. **Untested.**
- **The biased entropy estimator.** What I implemented regularizes entropy at *visited states*. True ∇H(trajectory) has an extra term for how the policy changes the prefix distribution. Gibbs-based claims are therefore about the *idealized* objective. The dominance argument doesn't depend on it, but an unbiased implementation is untested.
- **Is order 1 anti-aligned with order 2?** The marginal signal pushes toward singular. Nothing says the singular family is the friendlier one for the pairwise constraints. If the two orders point opposite ways, **early learning actively harms late learning** — a curriculum trap, measurable exactly.
- **Is it Adam rather than REINFORCE?** Adam normalizes per-parameter, amplifying small consistent gradients — it should accelerate rich-get-richer far more than plain SGD. Almost nobody asks whether mode collapse is an *optimizer* artifact. Cheap to test.
- **Does generation order interact with agreement direction?** Agreement flows from the **noun** at position 1. The determiner is generated *before* it, so the agent must commit to gender and number blind; the verb comes after, so its agreement is purely causal. That's exactly the asymmetry measured. Prediction: reordering to `nom dét verbe` makes both agreements causal and should break the lock. One line to change.
- **When does the lock-in happen?** I only measure endpoints. If the family is decided in the first 500 episodes, the entire 20,000-episode budget is settled by a tiny window and any intervention must be early.
- **Does overparameterization help the lock-in?** Six prefixes to distinguish, 128 hidden dimensions. Would a 4-unit GRU collapse *less*? The usual intuition ("more capacity is better") might invert.
- **And the one I turned back on myself: what result would make me abandon the hypothesis?** There's a pattern across both tests — build an environment where a human encodes the answer, then observe that RL finds it. Without a falsification criterion fixed *in advance*, test 3 will produce a third uninterpretable success.

---

## 🧭 A note on how this was actually done

The thing that produced every real result here wasn't a clever algorithm, it was a habit: **question the numbers as hard as the hypotheses, and never stop at the first explanation that fits.**

Concretely, five times in a row: form a hypothesis → find it plausible → notice it was produced too fast → build the control that separates it from an alternative → watch it die. The plural basin, the disconnected solution set, the held-out token, the tradeoff, the scope of the dominance argument. Each one felt right. Each one was wrong.

The second habit: **separate "the model can't" from "the gradient can't find it"**. Those look identical from the outside and have opposite implications. The separator is a capacity probe — fit the model in supervised mode toward the known solution, purely as a diagnostic, never presented as learning. It reframed the entire test 2 verdict.

The third: **audit your own controls for confounds**. My long grammar changes two variables at once. I found that myself, and it's in the limits section rather than quietly omitted.

And the honest meta-point: **the project drifted.** The original question was "can a network learn to write with reward only". Several hours in, I was doing dynamical-systems analysis of REINFORCE on a 20-token toy. That's real work, and some of it transfers — but it is not the question. The drift has a simple engine: each test "succeeds", the success is suspect, digging into the success is more interesting than the next test. Depth optimized at the expense of direction. Worth naming, because it'll happen to you too.

---

## ✅ What worked

- 🎯 **Pure RL from random weights does produce correct output** — perfect copy in 1,639 episodes, 99.9% grammaticality, no data at all
- 🧮 **Exact measurement** — an 8,000-sequence enumerable space means no sampling noise anywhere in the diagnosis
- 🕵️ **Causal isolation of mode collapse** — capacity, the objective and instability all cleared by independent controls; the remaining cause splits by regime (parameterization at low β, the sampled procedure at moderate β)
- 🔧 **A working fix** — β annealing dominates both constant regimes, reproduced on two schedules
- 📐 **A better vocabulary than "sparse"** — the ANOVA spectrum of a reward, measurable in a few lines

## ❌ What didn't (and that's fine — it's research)

- 🌀 **Phase portrait** — I bet the training trajectory would live in 2–3 dimensions and be *drawable*. It needs **8 dimensions for 90%** of the movement, 21 for 99%. Hypothesis dead.
- ⚖️ **Conserved quantities** — I went looking for an invariant along the trajectory. **Zero** functionals vary by less than 0.02. Nothing there.
- 🔁 **The held-out-token test** — measured 0.9966, which looks like generalization and isn't: the agent emits plural verbs regardless of the noun. Confounded by the collapse.
- 🎰 **My first explanation of family selection** — "the plural basin is wider" was wrong three times over. Refuted by seeds, relocated to within-family diversity, then *sign-flipped* by the order-1 computation.

## ⚠️ Limits (read these before quoting any number)

- **20-token toy grammar, 8,000 sequences.** These are exact results *in this setting*, not general laws. Claiming more is exactly what would destroy them.
- **My long grammar changes two variables at once** — adverbs grow the space, adjectives *also* add an agreement rule. Fine for the all-or-nothing control (only the hit rate matters), confounded for interpreting the graded run. Clean version: `dét nom verbe adv adv`.
- **My first sweep was single-seed**, and I picked β on it. At β=0.08 seed 0 is precisely the one that stays single-family. Redone on 3 seeds; downstream tests were not.
- **The implemented entropy bonus** is the standard state-visited regularizer — a biased estimator of ∇H(trajectory). Gibbs-based claims are approximate; the dominance argument doesn't depend on it.
- **20,000-episode budget.** I established REINFORCE is *slow*, not that it's *incapable*.
- **Three bugs of mine**, found and fixed mid-flight: a frozen token applied *after* generation (so the continuation conditioned on a different token), the single-seed β selection above, and an evaluation artifact where a frozen position was left free at eval.

---

## 🧰 What transfers out of the toy

The grammar is 20 tokens. These aren't.

### 1. Tied rewards give you a free optimality certificate

Normally you can't measure distance to the optimum without knowing the optimum. But **whenever several solutions have exactly the same reward, max-entropy forces them to be equiprobable** — so any deviation from equiprobability *proves* the optimizer failed, without ever computing the optimum's value. Mode collapse stops being a qualitative observation and becomes an exact measurement with a known target.

### 2. …and it's practical without enumeration

My certificate uses the exact distribution, which needs enumeration — impossible on a real LM. **But enumeration isn't required.** Take *k* outputs your reward model scores identically, score them under the policy by teacher forcing (*k* forward passes), test for uniformity. O(k), works on any model. You go from a heuristic ("entropy dropped, maybe it collapsed") to a **refutation** ("these must be uniform at the optimum, they aren't").

### 3. Buy diagnosability with reward resolution

That certificate needs *exact* ties. In real RLHF, reward models are continuous — ties have measure zero, so the tool is unavailable. **So manufacture them: quantize the reward model to k levels.** You lose a little resolution and gain an exact optimality certificate at every training step. Nobody treats **diagnosability as something you can buy**, because ties are seen as a defect to avoid rather than a resource to create.

### 4. Compute your shaped reward's optimum before blaming the optimizer

The standard reflex on bad validity is to blame optimization. But your *target* may already be bad. Here the graded reward's own optimum caps at 79.12% at β=0.08. Non-potential-based shaping changing the optimum is a 1999 theorem — what's missing in practice is the arithmetic.

### 5. Separate "can't represent" from "can't find"

Two failures that look identical from outside and have opposite implications. The separator is a **capacity probe**: fit the model in supervised mode toward the known solution, purely as a diagnostic, never presented as learning. It reframed my entire test 2 verdict.

### 6. Force the antecedent, measure the consequent

The only way to tell a learned rule from a degenerate sublanguage. Costs a few forced-decoding passes.

### 7. Never report a diversity number without the validity mass beside it

Random initialization scores 47.5 out of 48 on effective modes. Diversity alone is meaningless.

### 8. The ANOVA spectrum of a reward

Replaces the argument about whether a reward is "sparse" with a measurement of where its variance actually sits, and tells you which constraints the gradient can even perceive at initialization.

### Two ideas I didn't test

- **Masked entropy.** The per-token bonus pushes each conditional toward uniform over the *whole vocabulary* — it can't tell "vary the determiner" from "put a verb in slot 1". Restricting the entropy bonus to the support of actions that have already received positive advantage would target exactly the failure mode measured here.
- **A derived scaling law for mode collapse.** A prefix whose suffix is under-trained is systematically *undervalued*, and the bias is directional, so there should be a critical entropy coefficient β_c(L) growing with the length L of the suffix to relearn. Is it polynomial or exponential in L? If exponential, there's a sequence length past which no practical β works, and max-entropy RL on long sequences is structurally doomed without off-policy correction. My two grammars are a two-point measurement of that curve.

---

## 📋 Quick reference

| Parameter | Value |
|---|---|
| Policy | 1-layer GRU, hidden 128, embedding 32 |
| Algorithm | REINFORCE + moving-average baseline (100), Adam 1e-3 |
| Pretraining | none — random init, no exceptions |
| Data | none — reward only |
| Test 1 target | `le chat dort`, space 12^12 = 8.9 × 10^12 |
| Test 1 result | reward 1.0 at ep 1,639 (4 seeds: 1360–1702, σ=132) |
| Test 1 transfer | ×1.74 with shared prefix, **×0.91 without** |
| Test 2 grammar | dét nom verbe, 20 tokens, 8,000 sequences, 48 valid |
| Test 2 validity | 99.9 % with P(noun agrees\|det) = **0.333 = 2/6** |
| Reward spectrum | graded 76.1 % order-1 / all-or-nothing 4.0 % order-1, 65.5 % order-3 |
| Mode collapse cause | β≤0.02: autoregressive factorization (exact gradient still collapses). β≥0.05: the sampled procedure (exact gradient hits 48.0 modes, 3/3 seeds) |
| Best REINFORCE | β anneal 0.2→0.01: 99.97 % validity, 45.3 / 48 modes |
| Rejection baseline | 100 % validity, ~47.5 modes, ~167 draws per output |
| Hardware | CPU (2× faster than the 1080 Ti at batch 1) |

---

## 💻 The code

Everything is on GitHub: **[RDTvlokip/RDTRL](https://github.com/RDTvlokip/RDTRL)** — MIT
licensed, with the exact commands and runtimes to reproduce every number in this article.

Plain PyTorch, no HuggingFace, no RL library. Fifteen standalone scripts, French comments:

| script | what it does |
|---|---|
| `rl_copie.py` | test 1 — copy a fixed sentence, 3 reward shapes, 4 post-hoc analyses |
| `test4_controle.py` | test 1 — transfer control with zero positional overlap |
| `bench_device.py` | CPU vs GPU per-episode timing |
| `grammaire.py` | the hand-written grammar and parser, exact counts |
| `rl_grammaire.py` | test 2 — policy, sweep, controls, exact diversity measures |
| `balayage_graines.py` | multi-seed sweep |
| `sonde_capacite.py` | capacity probe (supervised fit toward uniform-over-48) |
| `optimum_gibbs.py` | closed-form optimum, shaping tax |
| `verifier_dominance.py` | numeric check of the dominance argument |
| `gradient_exact.py` | order-1 reward analysis + exact-gradient training |
| `parametrisation_et_recuit.py` | tabular vs GRU, β annealing |
| `stabilite_et_trajectoire.py` | stability from the ideal, KL trajectory |
| `trajectoire_et_structure.py` | PCA, ANOVA, conserved-functional search |
| `localisation_effondrement.py` | position freezing (corrected version) |

Plus three markdown files that are part of the method, not decoration: `ANALYSE.md` and `ANALYSE_TEST2.md` for the results, and **`CARNET.md`** — the lab notebook where refuted hypotheses stay in with their date of death. That last one is where most of what you read above actually came from.

---

## 🗂️ Summary

Two experiments testing whether a network can learn to write from random weights with reward only. Test 1 (copy a fixed sentence) succeeded in 1,639 episodes — because the per-position reward factorizes 12^12 into twelve trivial bandits, and transfer to a non-overlapping target was *slower* than starting over. Test 2 (learn a grammar judged by a hand-written parser) reached 99.9% grammaticality while learning **no rule at all**, by hiding in an all-plural sublanguage where agreement is vacuous. The all-or-nothing control succeeds or fails purely as a function of the random hit rate, which kills "sparse reward" as a concept and replaces it with the reward's ANOVA spectrum. Mode collapse was traced by exhaustive elimination on an enumerable space: capacity, the objective and instability are all cleared, and the remaining cause splits by regime — the autoregressive factorization blocks the optimum at low entropy pressure even with an exact gradient, while at β=0.05 the exact gradient reaches the optimum outright and it is the sampled procedure that fails. Annealing the entropy coefficient fixes it. And an untrained network plus a filter beats 20,000 episodes of REINFORCE on both validity and diversity.

---

## 🎯 The question I should have been asking

I started with *"can a network learn to write with reward only?"* Two tests in, that question is badly posed, and I can now say exactly why.

Every reward I used — string equality, then a parser — is a **verifier I wrote by hand**. Code has a natural one (tests pass). Maths has one (the proof checks). Natural language doesn't. So the pattern across both tests was structural, not accidental: **it works the moment you hand it a verifier, and it stops scaling the moment you can't write one.** I was measuring my own specification.

Here's the reformulation that fixes it:

> **Can an RL agent, from random weights and with no human text, discover a general linguistic representation from a reward that doesn't reward sentence validity at all — but communicative efficiency?**

That single change removes the verifier. Reward becomes *"did the receiver understand?"* rather than *"does this match my spec?"*. Language stops being the target and becomes the **instrument**. And task success is checkable without any language model — the receiver picked the right referent or it didn't — which is exactly the "non-model, non-vacuous verifier" I said I most wanted to be shown.

**What it doesn't escape**, and this needs saying before running anything: you don't delete the human specification, you move it from the signal into the **world**. You no longer declare what a valid sentence is, but you do choose the referents and their attribute structure — and that structure is precisely what determines whether a compositional code *can* emerge. Verifier on sentences, out. Oracle on the world, in.

**My prediction, and it's a falsifiable one.** Based on everything above: **high task success, non-compositional code.** The agents will invent a holistic symbol per referent, hit near-perfect accuracy, and generalize to nothing. That is the test 2 degenerate sublanguage one storey up — satisfying the verifier without acquiring the structure. Three findings from this article all point the same way: RL *refines* a distribution rather than creating one; the default attractor is whatever solution family the order-1 signal happens to favour; and the autoregressive factorization actively destroys the diversity a compositional code would need.

**And this is the first version of the question that can actually fail.** "General linguistic representation" is measurable: hold out *combinations* of attributes from training — never the attributes themselves, that was the never-seen-token mistake I made in test 2 — and test zero-shot. A compositional code decodes unseen combinations well above chance; a holistic one drops to chance. Plus topographic similarity between meaning-distance and message-distance, and positional disentanglement.

The falsification threshold goes in **before** the first line of code. That's the one thing tests 1 and 2 never had, and it's why both produced successes that took a day of digging to reveal as empty.

---

## 🏁 Conclusion

The objection I set out to check turns out to be **right on the mechanism and wrong on the name**. What kills pure RL isn't reward "sparseness" — it's the random hit rate, and you can now measure exactly where a reward puts its variance instead of arguing about adjectives.

Along the way, two successes that were both fake on inspection, one fix that actually works, a complete causal isolation of mode collapse that only an enumerable space makes possible, two dead hypotheses, four measurement traps, and three of my own bugs. Reported here in full, because a paper that only lists what worked is telling you what it went looking for.

The most useful thing I take out of it isn't any single number. It's that **a high score on a hand-written verifier tells you almost nothing** — the agent will find the corner of the output space where your constraint is vacuous, and it will look like success from every angle except the one where you force the antecedent and measure the consequent.

Next up: the referential game, with the threshold fixed in advance. If the emergent code generalizes to unseen attribute combinations, my prediction is wrong and there's a real thread to pull. If it doesn't, then "reward alone, no data" has failed at the one design that could have saved it — and that's an answer too. 👇

---

## ❓ Q&A

**— Isn't "the graded reward makes it easy" just reward shaping, which everyone knows changes the optimum?**

The theorem is from 1999 (Ng, Harada & Russell). What's missing in practice isn't the theorem, it's the **computation**. On the short grammar I can compute the shaped reward's Gibbs optimum in closed form: at β=0.08 it caps at **79.12%** validity while the all-or-nothing optimum sits at **99.94%**. Nobody measures that before blaming the optimizer. Even better, the learned policy at β=0.08 reaches 94.87% — *above* the optimum of its own objective — because mode collapse is conservative. The optimization failure was masking the shaping tax.

**— Why not just use PPO / bigger batches / better variance reduction?**

I ran the limit case — the **exact analytic gradient**, zero sampling noise, which no variance-reduction method can beat — and the answer turns out to hinge on the entropy coefficient, with a sharp transition. At β=0.01 and 0.02 it doesn't help at all: the GRU still collapses to 12–24 modes, identically across seeds, so variance reduction is a dead end there. At β=0.05 and 0.08 it helps completely: 48.0 modes and a 50/50 split on 3/3 seeds each, landing on the closed-form optimum to two decimals, where the sampled version gets 21–45.

So: better estimators are useless in the low-entropy regime and decisive in the moderate one. I originally published only the first half of that and had to correct it when the β=0.05 run landed — which is also a good argument for running your sweep to the end before writing your conclusion.

One caveat I can't remove: my exact-gradient run optimizes the true objective, while the sampled run uses the standard visited-state entropy bonus, a biased estimator of ∇H. At β=0.05 those two differences are confounded, so "noise" there might partly be "estimator bias".

**— Is 8,000 sequences too small to mean anything?**

It's small enough to be a toy and that's the point: it's the only reason every number here is exact rather than estimated. I can compute the true distribution, the true optimum, and walk the full elimination tree. The tradeoff is explicit — these are certain results in a small setting, not uncertain results in a big one. Test 3 will need a new arbiter before it can conclude anything.

**— What's the actual takeaway for someone doing RLHF?**

Three, in order of how cheap they are. **One**: a high score on a rule-based reward doesn't mean the rule was learned — force the antecedent, measure the consequent. **Two**: before blaming your optimizer, compute what your shaped reward's optimum actually is. **Three**: if your reward model gives several outputs the same score, they *must* be equiprobable at the optimum — so scoring k tied outputs under your policy (k forward passes, no enumeration) turns mode-collapse detection from a heuristic into a proof.

---

## 💡 Did you know?

Functional ANOVA — decomposing a function on a discrete cube into main effects and interactions — comes from experimental design in 1920s agronomy, not machine learning. Fisher used it to separate the effect of fertilizer from the effect of soil. The same decomposition, applied to a reward function, tells you exactly what a policy gradient can and cannot see at initialization: only the main effects. Every constraint that couples two positions — every agreement rule, every syntactic dependency, every "these two tokens must match" — lives at order 2 and is **invisible to the gradient until the policy is already correlated**. Which is another way of saying the thing you most want it to learn is the last thing it can perceive.

---

**Théo CHARLET**

TSSR Graduate (IT Systems & Networks Technician) - AI/ML Specialization

Creator of AG-BPE (Attention-Guided Byte-Pair Encoding)

🔗 LinkedIn: https://www.linkedin.com/in/théo-charlet

🔎 RDTvlokip Search (my search engine): https://search.rdtvlokip.fr

🚀 Seeking internship opportunities

🔗 Website : https://rdtvlokip.fr
