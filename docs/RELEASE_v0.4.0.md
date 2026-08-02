# RDTRL v0.4.0 — the reversal test, and the numeric path

The public-review release. Four rounds of criticism from **Dipankar Sarkar**
([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)) produced a
closed-form bound I had missed, took down four numbers I had published, and
explained an anomaly I had blamed on non-determinism.

The full account, with the exchange quoted verbatim at his request, is
[docs/ARTICLE2.md](ARTICLE2.md).

## The headline: his bound survives a change of grammar

A policy with no determiner→noun coupling has a **product** support, so at full
validity it cannot exceed the largest fully valid product in the sublanguage it
entered. That is a ceiling computable by enumeration before any training.

The obvious objection is that the ceiling comes from *my* lexicon. One experiment
decides it, and **my first version of that experiment tested nothing** — it moved
gender-neutrality from the plural determiners to the singular ones, which, since
the nouns and verbs are already symmetric in number, is exactly the `sg ↔ pl`
relabelling. The two grammars are isomorphic. Seventy seeds would have returned
the mirror image by construction.

> A relabelling can permute, but it cannot change a ratio. A perfectly symmetric
> control is often a perfectly empty one.

The version that tests something uses **three genders**. Both corners still hold
36 valid sentences, but the ceilings become **36 and 12** instead of 12 and 24 — a
ratio of **3 instead of 2**, which no relabelling can produce because the largest
product is an isomorphism invariant.

Ceilings computed and committed **before** the run. Seventy seeds, β = 0.02:

| corner | n | predicted ceiling | max observed | violations | exactly on it |
|---|---|---|---|---|---|
| singular | 33 | **36** | **36.0** | **0** | 2 |
| plural | 37 | **12** | **12.0** | **0** | 7 |

| grammar | ceiling ratio | observed mean ratio |
|---|---|---|
| standard, 2 genders | 2.0 | 1.82 |
| **three genders** | **3.0** | **3.01** |

The mean tracks the ratio, not just the ordering.

## The numeric path

The advantage line existed in two versions across the repository.
`rl_grammaire.py:141` subtracted **in float32**, rounding the baseline before
subtracting; the other sites subtract two doubles and round once.

```
  float32 first : 0.08333331346511841
  float64 first : 0.0833333358168602
```

That fully explains an 18.6-versus-11.50 discrepancy I had attributed to torch
multithread non-determinism — an explanation that never fitted, because **both
numbers were perfectly reproducible** and non-determinism does not reproduce.

`entrainer` now takes an explicit `chemin_avantage`, defaulting to **float64**:
one rounding instead of two, and measured 4× faster on that line (4.57 µs against
19.46), because a Python float is already a double so the subtraction is native
and only one tensor creation remains.

**It is also a better controlled experiment than any seed.** Changing a seed
changes the initialization *and* the trajectory. Changing the rounding changes
only the trajectory — the initial weights are bit-identical. Seventy seeds on
each path:

- **70 of 70 keep the same corner**, although the trajectories disagree on 58 to
  79 percent of the first two thousand steps;
- only **21 of 70 keep the same effective-mode count**, correlation 0.68.

> The initialization decides the corner. The trajectory decides how much of it
> gets filled.

And the conclusions the repository rests on — the fair coin, the ceiling, the null
mutual information — are **identical on both paths**. They were not rounding
artifacts.

⚠️ The tables in 0.3.x were produced on the float32 path. Pass
`chemin_avantage="float32"` to reproduce them exactly.

## Retracted

Four numbers, three of them a single seed and the fourth a single numeric path:

| claim | what killed it |
|---|---|
| "the branch is biased about 2 to 1 toward singular" | 3 seeds pooled across 8 β read as 24 draws. 70 seeds: 37/33, p = 0.72, my claim rejected at p = 0.016 |
| "it hides in an **all-plural** sublanguage" | one seed. Over 70: 37 singular, 33 plural |
| "early stopping would beat convergence by +12.5 modes" | one seed. Over 20: **median +0.03**, 5 of 20 above one mode |
| the corner-conditional version of the above | true on one numeric path, false on the canonical one |

The fourth is the instructive one: it was produced *while fixing* the first
three, with twenty seeds instead of one, and was still wrong because I had
changed the dimension I was sampling and not the dimension that actually varied.

## Also changed

- **The entropy sweep is now 10 seeds per β, 80 runs**, on the canonical path,
  with a new sg/pl column confirming the branch is a coin at every β. Two article-1
  numbers move: the all-or-nothing control 99.58 → 99.91 %, and the long grammar
  **6.4 → 15.8 %**.
- **`I(determiner ; noun)`** replaces a statistic that was an unweighted mean over
  six determiners and therefore measured *determiner coverage*, not agreement.
- **Six provenance collisions fixed**, all the same defect: an artifact whose name
  or caption omitted a dimension the run varied. One of them overwrote 70 saved
  policies an hour after I wrote that they were on disk so nobody would need to
  retrain.
- **Thread pinning centralised** in `rl_grammaire.py`, which fourteen scripts
  import, with `RDTRL_THREADS` to override.

## New code

`sonde_ordre1.py` · `produit_et_saturation.py` · `balayage_70_graines.py` ·
`optimum_produit.py` · `trajectoire_couplage.py` · `chemin_avantage.py` ·
`relancer_float64.py` · `figure_comparaison.py` · `figure_renversement.py`

## Limits, stated plainly

Two grammars, one algorithm, one β for the seed studies. The ceiling binds at
**constant β in the collapse regime**, not universally — annealing β from 0.2 to
0.01 reaches 45.3 modes, above the largest product over the whole valid set. The
mechanism behind the ceiling is localized, not proven. And I have not surveyed the
literature: where something here looks new to me, that is a statement about my
reading.

## Citation

Concept DOI, always the latest version:
[10.5281/zenodo.21726216](https://doi.org/10.5281/zenodo.21726216)

MIT licensed.
