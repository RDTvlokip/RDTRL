# RDTRL v0.3.0 — the product ceiling

Archival release. Everything in this repository is reproducible from the scripts:
the search spaces are small enough to enumerate exhaustively, so validity mass,
effective modes, the Gibbs optimum and the conditional structure are **computed
exactly**, not estimated from samples.

## The question

Can a neural network learn to produce language from a **scalar reward alone**,
starting from random weights, with no pretraining and no input/output pairs?

## What is in this release

**Test 1 — copying a fixed sentence.** The agent succeeds in 1,639 episodes, but
not for the reason that would validate the premise: the graded reward
**decomposes**, factoring a 12^12 space into twelve independent 12-armed bandits.
The all-or-nothing control gets exactly 0.000 across 30,000 episodes. Transfer is
×1.74 with a shared prefix and **×0.91 without**, so there is no reusable
abstract structure.

**Test 2 — a hand-written grammar, judged by a deterministic parser.** 99.9
percent grammaticality **with no learned rule**: the agent hides in a
sublanguage where agreement is vacuously satisfied. The all-or-nothing control
*succeeds* on the short grammar and fails on the long one, so the deciding
variable is the random hit rate, not the shape of the signal.

**New in 0.3.0 — a closed-form bound, and the mechanism behind it.**

A policy with no determiner-to-noun coupling has a **product** support, so at
full validity it cannot exceed the largest fully valid product contained in the
sublanguage it entered. Computed by exhaustive enumeration:

| | valid sentences | largest product |
|---|---|---|
| short grammar, plural corner | 24 | **24** |
| short grammar, singular corner | 24 | **12** |
| long grammar, each corner | 144 | **72** |

Measured against 70 seeds at a single condition:

- **the bound is never exceeded**, 0 of the 37 runs in the corner where it can be
  tested, and the modal outcome **is** the bound (19 runs at exactly 12.0);
- effective modes come out as **integer products**, {2, 4, 6, 8, 12} and
  {6, 8, 12, 16, 18, 24}, so the product structure is visible in the histogram;
- **no run acquires the conditional**: I(det ; noun) has median 0.0000 bits and
  maximum 0.0377 over 70, against the 1.0 bit the coupled solution needs;
- **the branch is a fair coin**: 37 singular against 33 plural, Wilson 95 percent
  [0.413, 0.641], p = 0.72.

And the bound is **a plateau, not a basin**. The exact gradient sits at 12.00
modes with I identically zero for a thousand steps, then escapes to 24.00 at
I = 0.998, so it can be left with no noise at all. What separates the two
procedures is the **depth of the transient collapse**: the exact runs never drop
below 10.7 effective modes, the sampled runs crush the policy down to 1.09 before
rebuilding it. Every trajectory starts at 47.5 modes, the untrained network.

## Corrections to previously published claims

This release exists partly to correct the record, and the corrections are as
load-bearing as the new results.

- **"Complete causal isolation … the autoregressive factorisation is the
  culprit"** is false as stated. The cause splits by regime: at β ≥ 0.05 the
  exact gradient reaches 48.0 modes and a 50/50 split, so the failure is in the
  sampled procedure, not the factorisation.
- **"The agent takes refuge in an all-plural sublanguage"** was true of one seed.
  Over 70 seeds it is 37 singular against 33 plural.
- **P(noun agrees | det) = 0.333** was an *unweighted* mean over six
  determiners, so it equals (determiners emitted)/6 rather than an agreement
  rate. Replaced by the mutual information I(det ; noun).
- **The saturation metric** computed H over all 8 nouns while normalising by the
  count of compatible ones, so it could exceed 100 percent, and a value above 100
  meant mass leaking onto incompatible nouns — a failure that read as a success.
  Split into two properly bounded fields.

Eight hypotheses of mine are dated as refuted in `docs/CARNET.md` section 1,
three of them on the day of this release.

## Reproducing

```bash
pip install -r requirements.txt
cd src/test2_grammar          # scripts import each other, run from their own directory
python grammaire.py           # the parser, and the exact counts
python sonde_ordre1.py        # order-1 marginals, closed form, no training
python produit_et_saturation.py
python balayage_70_graines.py --beta 0.02 --debut 0 --fin 70
python trajectoire_couplage.py
```

## Where things are

| | |
|---|---|
| results, test 1 | `docs/ANALYSE.md` |
| results, test 2 | `docs/ANALYSE_TEST2.md` |
| reasoning, refuted hypotheses, errors | `docs/CARNET.md` |
| test 3 design | `docs/TEST3.md` |
| what to do next | `ROADMAP.md` |
| article | https://huggingface.co/blog/RDTvlokip/teaching-a-network-to-write-with-reward-only |

## Acknowledgement

The product bound at the centre of this release comes from **dipankarsarkar**,
who computed the order-1 marginals independently from the article, then found
that the two degenerate corners hold the same 24 valid sentences but not the same
largest product. Three rounds of their criticism also caught a statistic of mine
that measured determiner coverage rather than agreement, a saturation metric that
could exceed 100 percent, and a sample-size claim that pooled 3 seeds into 24
rows. Each of those is corrected here.

## Limits, stated plainly

One hand-written grammar, one algorithm, and the bound is verified at constant β
in the collapse regime only — annealing β from 0.2 to 0.01 reaches 45.3 modes and
crosses it. The single experiment that would decide whether this is a law or a
coincidence of my lexicon is the **reversal test**: build a vocabulary where the
ordering of the two ceilings flips, record the prediction, check that it flips.
It has not been run. See `docs/CARNET.md` section 7.12.

MIT licensed.
