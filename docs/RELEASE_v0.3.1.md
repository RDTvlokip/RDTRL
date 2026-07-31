# RDTRL v0.3.1 — acknowledgment

Metadata release. **No code and no results have changed** since
[v0.3.0](RELEASE_v0.3.0.md); this exists to put a credit where it belongs before
the record is any more permanent.

## Acknowledgment

**Dipankar Sarkar** — [ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)

The product bound that carries v0.3.0 is his. He derived the order-1 marginals
independently from the published article, then showed that the two degenerate
corners hold the same number of valid sentences but **not the same largest
product**: 24 valid and 24 in one, 24 valid and only 12 in the other, exactly one
bit apart, because a genre-neutral determiner removes one of the two agreement
constraints. That is what makes the ceiling computable before any training, and
it is the difference between a curve that goes up and a number you can predict.

Three rounds of his criticism also corrected, in order:

- a statistic of mine that was an **unweighted mean over six determiners**, so it
  read (determiners emitted)/6 rather than an agreement rate, and gave 0.3333 for
  both of the two collapses it was being asked to distinguish;
- a **saturation metric** that computed entropy over all eight nouns while
  normalising by the count of compatible ones, so it could exceed 100 percent,
  and a value above 100 meant mass leaking onto incompatible nouns — a failure
  that read as a success;
- a **sample-size claim** that pooled 3 seeds across 8 values of β into "24 runs",
  when the branch was set by the seed alone. Rerun as 70 seeds at a single
  condition, the effect I had claimed is rejected at p = 0.016.

He also asked the question that produced the long-grammar measurement, which I
had never run because its low validity had me filing it as a scaling failure
rather than as a collapse worth analysing.

The full exchange, including every error it caught and the two hypotheses of mine
it killed, is in `docs/CARNET.md` sections 7.10 to 7.12. Refuted hypotheses are
dated in section 1.

## What changed in the files

- `README.md` — Acknowledgments section, and the ORCID.
- `CITATION.cff` — acknowledgment in the abstract, version bumped.
- `.zenodo.json` — new, giving Zenodo a structured `contributors` entry with the
  ORCID, which CFF 1.2.0 has no field for.
- `CHANGELOG.md` — 0.3.1 entry.

## Correctness note

`.zenodo.json` takes precedence over `CITATION.cff` on Zenodo, so it carries the
full metadata rather than a fragment: title, creator, contributor, license,
keywords, related identifiers and description.

MIT licensed.
