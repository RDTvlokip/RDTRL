# RDTRL v0.5.0 — test 3, seven questions in a day, and eight dead hypotheses

The version where the third experiment finally ran. It had been designed, argued
about and criticised without ever being executed, and the whole seven-question
investigation programme of [docs/TEST3.md](TEST3.md) went from 0 to 7 in one day.

The full account, with the exchange quoted and every measurement including the
ones I would have cut, is [article 3](https://huggingface.co/blog/RDTvlokip/i-made-my-world-small-enough-to-compute-everything),
mirrored in [docs/ARTICLE3.md](ARTICLE3.md).

## The world

27 referents, 27 messages, a sender and a receiver that are each a 27 × 27
stochastic matrix. Small enough that the optimum set, the null distribution and
the gradient are **computed** rather than estimated.

All 27! perfect codes earn exactly 1. They are tied, so compositionality is
**1 296/27! ≈ 1.19 × 10⁻²⁵** of the optimum set before any training. The
programme exists to ask what, if anything, selects them.

## The headline: the certificate does not survive two agents

The whole calculation imported a result from experiment 2 — *under maximum
entropy, optima with equal reward are equiprobable*. The design document had
flagged this as the one place a published result could collapse, and put it first
in the execution order. It collapsed.

**The certificate requires the tied objects to be the support of the distribution
whose entropy is in the objective.** In experiment 2 they were sequences, and the
entropy was over sequences. Here they are codes, and no distribution over codes
appears in the objective at all. Mixing K codes:

| K | 1 | 2 | 3 | 5 | 10 | 27 |
|---|---|---|---|---|---|---|
| E[R] | 1.0000 | 0.5000 | 0.3416 | 0.2237 | 0.1511 | 0.0713 |

**An equivariance argument replaces it and is stronger.** Relabelling messages
acts transitively on the 27! bijections, so an equivariant procedure makes them
exactly equiprobable — no Gibbs assumption. Verified 8 runs out of 8. It is now a
theorem about the parametrization, valid for **any** equivariant algorithm.

> **The corollary, and it is the one thing here that is checkable by reading an
> architecture rather than running it.** Relabelling *referents* is also
> transitive, so equivariance on either side alone ties every code. **A free
> per-referent embedding table cancels in advance anything the message structure
> could contribute.** Measured: an autoregressive sender with per-referent
> parameters gives a compositional-versus-random gap of 3.3 × 10⁻¹⁶.

## What was measured

| question | result |
|---|---|
| §6.7 certificate | does not survive; equivariance theorem replaces it |
| §6.5 representable / reachable / stable | three different answers; a perfect bijection is reached **once in forty starts** |
| §6.1 which code emerges | z = −0.12, −0.25, **+9.92** for tabular, autoregressive, structured |
| §6.2 does the dynamics sample at random | at 100 seeds, z = −0.01 ± 0.10 and −0.05 ± 0.10; **any residual selection < 0.0087** |
| §6.3 who writes the code | neither — **139 steps in both directions**, same value to eight decimals |
| §6.4 the first-step gradient | prefers nothing; the structured preference appears **between step 10 and step 30** |
| §6.6 the constraint curve | channel noise and population turnover both produce **nothing** |

**A phase diagram in closed form.** The babbling point becomes stable at
**β = 1/N**, confirmed by the Hessian crossing zero at 0.037037037, within
2.4 × 10⁻¹¹ of 1/27. A bisection on the dynamics put it at 0.0381 — it was
measuring **Adam**, whose normalised steps do not slow where the gradient
vanishes. Second threshold at 0.1701, with a bistable region between them.

**Channel noise is the cleanest negative.** It leaves the reward tie **exact** at
every ε — for a bijective code and the optimal decoder, `E[R]*` does not depend on
the code, which takes one line and never mentions 27 — while breaking the
objective's symmetry. So any selection would operate entirely outside the reward.
None occurs: z from −0.44 to +0.38, no run out of 90 past the 99.9th percentile.

> **Breaking the symmetry is necessary and not sufficient**, which refutes the
> unifying hypothesis I had drawn from the equivariance theorem that same morning.

**Conclusion.** Of everything tested, only the parametrization produced
compositionality, and it did so by making the alternatives unrepresentable rather
than by selecting among them. On this bench, compositionality was never
*selected*: it was either impossible or specified.

## Retracted and corrected

Eight dated dead hypotheses, §1.9 to §1.16 of the notebook, **none of them
arithmetic errors**:

| claim | what killed it |
|---|---|
| "the 0.35 threshold is derived, so it's solid" | a sample maximum has no estimand; the null's supremum is the value being excluded |
| "the double-count inflation grows with concentration" | true on the null, false on a structured population at the same level |
| "the equal-optima certificate applies as-is to two agents" | tied objects are not the support of the entropy |
| "the gap from 1/27 comes from the perturbation size" | it was Adam |
| "the top of the concentration scale is safe" | true only for bijections, and the reached codes are not |
| "the structured parametrization prefers compositional from step 1" | z = −0.08 ± 0.24 |
| "channel noise favours compositional codes" | one line of algebra; the gap is 1.1 × 10⁻¹⁶ at every ε |
| "breaking the symmetry is enough" | necessary, not sufficient |

**And the most serious defect is in my own falsification criterion**, recorded
before any data and presented as what made the test sharply falsifiable. It failed
to name the parametrization, which decides the answer; its interpretation clause
said an excess would prove a flaw in the equal-optima reasoning, which does not
follow; and its first half predicted near-perfect bijections, of which there was
one in twenty. Notebook §4.7.

**Plus five protocol defects caught before publishing**: statistics read off the
last seed instead of averaged, twice; nulls not matched to the observation's
orbit, twice; a success threshold arithmetically unreachable in half the
conditions.

## Also in this release

- **`concentration_appariee()`**, one attribute per position, exact Hungarian over
  six assignments. The published `concentration()` is unchanged, so the 20 000-draw
  table still reproduces bit for bit. Both are kept: the unconstrained form is the
  field-standard shape, the matched one is what the position reading uses. The
  decisive argument is that a degenerate code — `m₁ = a₁, m₂ = a₁, m₃ = a₂`, one
  attribute discarded, 9 of 27 messages used — scores a perfect **1.000000** under
  the published form and 0.666667 under the matched one.
- **The null goes from 20 000 to 10 000 000 draws**, with an exact unsampled tail,
  and a matched-null construction on the orbit of each run's fiber profile. That
  correction turned out to move the reference by −0.0001 to +0.0005, which had to
  be checked to be known.
- **A guard on the vectorised path**, which assumed uniform margins — true only for
  bijections — and silently returned 0.110573 instead of 0.108071 otherwise.
- **Metadata brought up to date**: the article 2 URL was missing from the README,
  `CITATION.cff` and `.zenodo.json`, and the Zenodo description had stopped at
  0.3.1 without ever mentioning the reversal test. The acknowledgment said three
  rounds of criticism; it has been five.
- **Article 3** — [published on Hugging Face](https://huggingface.co/blog/RDTvlokip/i-made-my-world-small-enough-to-compute-everything),
  mirrored in [docs/ARTICLE3.md](ARTICLE3.md), including a section publishing
  every measurement I would have cut for readability.

## New code

`loi_nulle_longue.py` · `variabilite_du_maximum.py` · `appariement_vs_distance.py` ·
`certificat_deux_agents.py` · `representable_atteignable_stable.py` ·
`code_emergent.py` · `dynamique_uniforme.py` · `qui_ecrit_le_code.py` ·
`gradient_premier_pas.py` · `courbe_de_contrainte.py`

## Limits, stated plainly

One world, one algorithm, one reward. Exact gradients do not scale, and sampled
REINFORCE gives different results rather than approximate ones. **No literature
review** — the equivariance argument is elementary and someone has very likely
written it down. And the one positive result is close to tautological: the
structured sender cannot write most bijections, so finding that it writes a
structured code is capacity, not emergence.

The uncomfortable finding of the release is that **making the world small enough
to compute everything exactly removed one class of error and left the class that
was actually doing the damage completely untouched.** What caught the eight was an
outside reader who reran the code before speaking, a second independent route to
the same quantity, and predictions written down before measuring.

## Credit

The threshold critique, the power calculation and the matching question are
**Dipankar Sarkar** ([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)),
in his fifth round on this project and the fifth to produce a correction.

## Citation

Concept DOI, always the latest version:
[10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216)

MIT licensed.
