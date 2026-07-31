# RDTRL v0.3.2 — correct concept DOI

Metadata release. **No code and no results have changed** since
[v0.3.0](RELEASE_v0.3.0.md).

## Why this exists

Zenodo mints three numbers for this repository: one **concept DOI**, which
represents all versions and always resolves to the newest one, and one **version
DOI** per release, frozen on that exact snapshot. They are indistinguishable by
eye.

The archived snapshots of v0.3.0 and v0.3.1 carry a README that presents a
*version* DOI as if it identified the repository. That is wrong in a way that
only shows up later: a reader following that badge after v0.4.0 would land on
v0.3.1 forever, without knowing they were looking at an old state.

| DOI | kind | resolves to |
|---|---|---|
| **10.5281/zenodo.21726216** | **concept** | always the latest version |
| 10.5281/zenodo.21726387 | version | v0.3.1, frozen |
| 10.5281/zenodo.21726217 | version | v0.3.0, frozen |

From this release on, the badge, the BibTeX block and the `doi` field of
`CITATION.cff` carry the concept DOI, both version DOIs are kept as secondary
identifiers, and `CHANGELOG.md` records one frozen DOI per entry. Publishing a
future release therefore requires no edit to the README: the badge follows on its
own.

## Citation

```bibtex
@software{charlet_rdtrl_2026,
  author    = {Charlet, Théo},
  title     = {{RDTRL — Can a network learn to write from reward alone?}},
  year      = {2026},
  version   = {0.3.2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21726216},
  url       = {https://doi.org/10.5281/zenodo.21726216}
}
```

## Acknowledgment

Unchanged from [v0.3.1](RELEASE_v0.3.1.md): the product bound is due to **Dipankar
Sarkar** ([ORCID 0000-0001-5431-6367](https://orcid.org/0000-0001-5431-6367)).

MIT licensed.
