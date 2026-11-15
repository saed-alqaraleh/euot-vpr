# EUOT-VPR: Descriptor Weighting Shapes the Effect of Unbalanced Transport in Occluded Visual Place Recognition

[![Verify archived results](https://github.com/saed-alqaraleh/euot-vpr/actions/workflows/verify.yml/badge.svg)](https://github.com/saed-alqaraleh/euot-vpr/actions/workflows/verify.yml)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)

**Official research implementation and reproducibility materials** for *Descriptor Weighting Shapes the Effect of Unbalanced Transport in Occluded Visual Place Recognition*.

EUOT-VPR investigates how descriptor normalization changes the information retained from unbalanced optimal transport in visual place recognition. The study establishes a precise normalization effect and evaluates whether post-normalization evidence gating restores cluster support under occlusion. The repository provides the executed research implementations, author-fitted heads, frozen per-query evaluation outcomes, statistical analysis scripts, and integrity checks.

## Quick start

**Reproduce the headline MSLS-val results on CPU** using Python 3.11 or 3.12. The following requires no source images, GPU, or upstream SALAD checkpoint.

```bash
git clone https://github.com/saed-alqaraleh/euot-vpr.git
cd euot-vpr
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/verify_release.py
python scripts/reproduce_headline.py --output outputs/headline_msls.json
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The first verification checks the release's SHA-256 inventory, the frozen setting manifests, and all **366 per-query MSLS-val hit matrices**. The second command independently reconstructs the manuscript's **four Recall@1 interactions at 50% query masking and their 95% confidence intervals**, the ungated relaxation contrasts, the mass-only-gate interaction, the global-token-share contrast, and the released SALAD reference. The generated JSON is saved in the Git-ignored `outputs/` directory.

## Method

For a nonzero pooled cluster vector, normalizing that vector removes the **direct positive Sinkhorn column factor**. Unbalanced column relaxation can therefore change the amount of evidence assigned to a cluster without preserving that information in the normalized cluster's length. Changes to row factors may still alter the pooled direction; the result is not a claim that unbalanced transport has no effect.

EUOT-VPR studies a **post-normalization cluster gate** that reintroduces support-sensitive information into the final similarity score. The evaluation crosses balanced and relaxed transport with gated and ungated descriptors, then varies the local-token treatment and the relative weight of the global descriptor. This separates a transport effect from a descriptor-design effect rather than treating them as interchangeable.

## Key results

On **MSLS validation**, the four matched settings produce the following combined-gate difference-in-differences (DiD) at **50% target-query masking**. Values are percentage-point changes in Recall@1; intervals are from paired resampling over the 740 recorded queries.

| Local-token treatment | Global-token share | Combined-gate DiD (pp) | 95% CI (pp) |
|---|---:|---:|---:|
| Unit-normalized | 1/2 | **+2.93** | [+1.17, +4.82] |
| Raw | 1/2 | **+2.34** | [+0.54, +4.28] |
| Unit-normalized | 1/65 | −1.22 | [−3.74, +1.35] |
| Raw | 1/65 | −3.42 | [−5.86, −0.99] |

Here, DiD is the change in the gate's Recall@1 effect when moving from balanced to relaxed transport: `(UOT+gate − UOT) − (balanced+gate − balanced)`.

Across the two controlled and four MSLS-val severe-occlusion factorials, **all six 95% intervals for ungated column relaxation include zero**. In the unit-token, 1/2-share setting, mass-only gating yields **+2.79 pp** (95% CI [+1.13, +4.59]). Reducing global-token share from 1/2 to 1/65 changes the combined-gate interaction by **−4.95 pp** (95% CI [−7.68, −2.32]), averaged across the two token treatments.

These are fitted-head, within-benchmark contrasts. The controlled benchmark and MSLS-val play distinct roles; the reported results do not establish independent-dataset generalization.

## Evaluation protocol

- **MSLS-val:** 18,871 gallery images and 740 official queries; 25 m positive criterion; deterministic target-query masking at 0%, 25%, and 50%.
- **Backbone and training:** a frozen pretrained SALAD DINOv2 trunk; study-specific heads fitted using 150 Gardens Point day-left/night-right pairs, with a separate 50-pair in-sequence development subset.
- **Matched comparisons:** balanced versus relaxed transport, without a gate versus combined gating; unit-normalized versus raw local tokens; global-token shares 1/2 and 1/65.
- **Randomness and uncertainty:** head seeds 42, 108, and 2026; query-level mean across seeds; 2,000 paired-bootstrap replicates and 10,000 two-sided sign flips, using the recorded random generator and correction families.
- **Controlled benchmark:** synthetic 80×80 scenes, with 240 seed-place severe-occlusion realizations at 55% occlusion. These experiments supplement, rather than replace, the MSLS-val evaluation.

The original `U` and `S` results use the frozen v4/v5 hit arrays; the crossed `UW` and `RE` results use v6. Later re-evaluations are retained as separate evidence, not substituted for the original measurements.

## Reproduce the archived evidence

The CPU quick start reproduces the reported headline. The following commands provide the fuller numerical analysis:

```bash
python analysis/recompute_v6_statistics.py --out outputs/v6_statistics.json
python analysis/recompute_msls_statistics.py --setting_u msls_setting_U --setting_s msls_setting_S --controlled controlled_benchmark
python analysis/recompute_gating_energy.py
```

For CPU model-implementation and archived-head checks, add the optional test dependencies:

```bash
python -m pip install -r requirements-cpu-tests.txt
python msls_crossed_and_gates_v6/kit/tests/test_kit_cpu.py
```

| Manuscript evidence | Reproduction input and entry point |
|---|---|
| Four MSLS-val severe-occlusion interactions and confidence intervals | Frozen v4/v5/v6 hit matrices; `scripts/reproduce_headline.py` |
| Ungated relaxation and single-gate decomposition | Archived hit arrays; `analysis/recompute_v6_statistics.py` |
| Grouped resampling and multiplicity adjustments | MSLS per-query arrays and recorded grouping; `analysis/recompute_v6_statistics.py` |
| Controlled severe-occlusion comparisons and scheduling ablation | `controlled_benchmark/` and `analysis/recompute_msls_statistics.py` |
| Cluster mass and energy diagnostics | `analysis/recompute_gating_energy.py` |
| Source integrity, checkpoint consistency, and CPU head behavior | `scripts/verify_release.py` and `msls_crossed_and_gates_v6/kit/tests/test_kit_cpu.py` |

See [the full manuscript-to-evidence map](docs/REPRODUCTION_MAP.md), [dependency scope](docs/DEPENDENCIES.md), and [verification record](docs/VALIDATION.md). Figure-generation inputs and scripts are in `analysis/figure_scripts/`; optional plotting dependencies are listed in `requirements-figures.txt`.

## Repository contents

| Path | Contents |
|---|---|
| `msls_setting_U/` | Original unit-token, 1/2-share evaluation; v4 source, head checkpoints, and hit arrays |
| `msls_setting_S/` | Original raw-token, 1/65-share evaluation; v5 source, head checkpoints, and hit arrays |
| `msls_crossed_and_gates_v6/` | Crossed descriptor settings, gate ablations, 72 fitted heads, per-query hits, diagnostics, and CPU tests |
| `controlled_benchmark/` | Controlled occlusion implementation, saved numerical outcomes, and scheduling ablation |
| `analysis/` | Statistical reproduction, archived aggregate results, and figure-generation scripts |
| `scripts/` | Headline reproduction and release-integrity verification |
| `docs/` | Reproduction mapping, dependency provenance, image-level replay, and third-party rights |
| `SHA256SUMS.txt`, `CITATION.cff` | File integrity inventory and software citation metadata |
| `.github/workflows/verify.yml` | Automated CPU checksum and headline-result verification |

## Full image-level replay

Numerical reproduction from the archived hit matrices **does not require image access**. Generating the same hits from source photographs requires the following external inputs, under their original providers' terms:

1. **MSLS validation images and positives:** the authorized source archive, the exact historical ordered `msls_manifest.json` (not redistributed), 18,871 gallery positions, 740 query positions, and the corresponding 25 m positive-gallery index lists.
2. **Gardens Point day/night images:** the ordered image pairs used to fit and validate the original heads, with their recorded train/development division.
3. **Pretrained SALAD DINOv2 backbone and projection:** the matching historical upstream implementation and model checkpoint; the included `.pt` files are study-fitted heads, **not** the upstream SALAD weights.
4. **Compatible GPU environment:** PyTorch/CUDA and upstream SALAD requirements. A complete historical lockfile for the v6 GPU run was not archived.

Exact replay also depends on preserving query/gallery order, mask generation, and recorded seeds. The photographic input bank used by the controlled raw-image simulator is withheld pending asset-specific redistribution clearance. See [image-level replay instructions](docs/IMAGE_LEVEL_REPLAY.md) for the protocol and [third-party rights](docs/THIRD_PARTY_RIGHTS.md) for access and licensing.

## Integrity, licensing, and citation

`SHA256SUMS.txt` covers the released files; `scripts/verify_release.py` verifies both this inventory and the original setting manifests. Source-level sanitization of non-scientific path defaults and comments is documented in [source provenance](docs/SOURCE_PROVENANCE.json). The study's numerical arrays and results were not altered.

Original research code is under the [MIT License](LICENSE). Third-party datasets, source photographs, upstream implementations, and pretrained weights retain their own terms. The current private research archive includes study-fitted heads and dataset-derived query-group records whose redistribution is still being assessed; see the [asset-level release decisions](docs/THIRD_PARTY_RIGHTS.md). This private repository should not be made public simply by changing visibility, because its earlier commits retain those materials.

For citation, use [CITATION.cff](CITATION.cff) and the final journal article when its bibliographic details are available. To identify a reproducible software version, reference the **exact Git commit** rather than the changing `main` branch.
