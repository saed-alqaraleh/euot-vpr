# EUOT-VPR: Descriptor Weighting Shapes the Effect of Unbalanced Transport in Occluded Visual Place Recognition

**Code, author-fitted heads, frozen per-query evidence, and reproducible statistical analysis.** This repository accompanies the article *Descriptor Weighting Shapes the Effect of Unbalanced Transport in Occluded Visual Place Recognition*. The article manuscript and third-party image datasets are not included.

## Main finding

Intra-cluster normalization cancels the **direct positive column factor** from an unbalanced Sinkhorn transport plan for a nonzero pooled vector; altered row factors may still change its direction. A post-normalization cluster gate restores a mass/feature-length signal. Across the two controlled and four MSLS-val severe-occlusion factorials, the 95% intervals for **ungated column relaxation include zero**. On MSLS-val at 50% target masking, the combined gate's difference-in-differences (DiD) is **+2.93 pp** with unit tokens and a 1/2 global-token share, and **+2.34 pp** with raw tokens and a 1/2 share. Reducing the global share to 1/65 changes the interaction to **−1.22 pp** and **−3.42 pp** for unit and raw tokens, while absolute recall increases. These are **within-benchmark fitted-head contrasts**, not independent-dataset generalization claims.

## CPU quick start — no source images, CUDA, or SALAD download

Use Python 3.11 or 3.12. From the release root:

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -r requirements-cpu.txt
python scripts/verify_release.py
python scripts/reproduce_headline.py --output outputs/headline_msls.json
```

`reproduce_headline.py` reads only the archived, ordered `(740,3)` hit matrices, averages query hits over seeds `42/108/2026`, and recreates the reported severe-occlusion arm recalls, ungated relaxation, combined-gate interaction, mass-only-gate interaction, and the crossed global-share effect. It compares **point estimates and confidence intervals** against the manuscript (tolerance at printed two-decimal precision) and writes its JSON under the gitignored `outputs/` directory.

For the full factorials, grouped uncertainty checks and auxiliary contrasts (slower):

```bash
python analysis/recompute_v6_statistics.py --out outputs/v6_statistics.json
python analysis/recompute_msls_statistics.py --setting_u msls_setting_U --setting_s msls_setting_S --controlled controlled_benchmark
python analysis/recompute_gating_energy.py
```

For the original head equivalence and trained checkpoint smoke tests (CPU PyTorch):

```bash
python -m pip install -r requirements-cpu-tests.txt
python msls_crossed_and_gates_v6/kit/tests/test_kit_cpu.py
```

For figures, install `requirements-figures.txt` and consult `docs/REPRODUCTION_MAP.md`. The controlled image generator requires an **unredistributed external image bank** whose original expected SHA-256 is listed in `docs/THIRD_PARTY_RIGHTS.md`; this numerical-evidence package intentionally excludes the bank pending asset-specific clearance.

## Reproduction map

| Evidence | Path or command | Scope |
|---|---|---|
| Headline severe-occlusion MSLS-val contrasts | `scripts/reproduce_headline.py`; v4/v5 archived and v6 crossed hits | CPU, no images |
| Full MSLS-val factorial, uncertainty and correction | `analysis/recompute_v6_statistics.py` | CPU, hit arrays and grouping records |
| Original two settings and controlled ungated contrast | `analysis/recompute_msls_statistics.py` | CPU |
| Mass/energy diagnostic | `analysis/recompute_gating_energy.py` | CPU, archived diagnostics |
| Controlled 11-mode and scheduler evidence | `controlled_benchmark/*.json/*.npz`, `controlled_benchmark/scheduling_ablation/` | Stored evidence; raw-image benchmark rerun needs authorized bank |
| Fitted MSLS-val heads | `msls_setting_U/checkpoints_v4/`, `msls_setting_S/checkpoints_v5/`, `msls_crossed_and_gates_v6/checkpoints_v6/` | Study-fitted weights only; not upstream SALAD weights |
| Head implementation and equivalence tests | `msls_crossed_and_gates_v6/kit/` | CPU tests; GPU/image replay needs external inputs |
| Image-level repeat | `docs/IMAGE_LEVEL_REPLAY.md` | External licensed data, exact ordered manifest, SALAD trunk, CUDA |
| Integrity verification | `scripts/verify_release.py`, `SHA256SUMS.txt`, three original-setting manifests | SHA-256 and data shapes |

## Archived scientific design

- **MSLS-val:** 18,871 gallery images, 740 official queries, 25 m positive criterion, targets 0%, 25%, 50% masking. Three head seeds: 42, 108, 2026.
- **Training:** frozen released SALAD DINOv2 trunk; heads trained on 150 Gardens Point day-left/night-right pairs (separate in-sequence 50-pair development subset). The four descriptor settings cross **unit/raw pooled tokens** with global-token shares **1/2** and **1/65**.
- **Factorials:** balanced vs relaxed columns × no gate vs combined gate. Gate components: length, mass, mean-norm, reliability. The two original settings `U` and `S` use **archived** v4/v5 hit arrays; `RE` and `UW` use new v6 arrays. Re-evaluations of U/S are stored separately for comparison and are not silently substituted.
- **Uncertainty:** per-query mean over three seeds, 2,000 bootstrap resamples, 10,000 two-sided sign flips, `numpy.default_rng(42)`; specified Holm families and grouped tests in full script. Absolute Recall@k percentages are descriptive, while contrasts are percentage-point differences.
- **Controlled benchmark:** synthetic 80×80 scenes and 240 seed-place test realizations at 55% occlusion for the severe factorial. This simulation is **not** a substitution for MSLS-val.

## Release structure

```text
analysis/                     Original statistical & figure scripts, archived numerical outputs
controlled_benchmark/         Original simulation code, recorded outputs, trained arrays, scheduling ablation
msls_setting_U/               v4 executed model source, 13 checkpoints, 42 per-query hit arrays
msls_setting_S/               v5.1 model source, 12 checkpoints, 36 per-query hit arrays
msls_crossed_and_gates_v6/    v6 72 heads, 288 hit arrays, diagnostics, model/tests, grouping
scripts/                      Release-integrity and headline recomputation checks
docs/                         Data/replay, rights, numerical provenance, release mapping
SHA256SUMS.txt                Full release hash inventory (excluding itself)
```

## Offline Git bundle import

For offline distribution, the accompanying Git bundle contains a complete `main` branch. To inspect the release tree:

```bash
git clone -b main EUOT-VPR_public_candidate.git.bundle euot-vpr
cd euot-vpr
python scripts/verify_release.py
```

For a GitHub release, push to a **private** repository first and review the complete remote tree before changing visibility. Third-party source images and original upstream model weights are not included; see the rights and reproduction documentation for the remaining independent inputs.

## Source and release scope

The included checkpoints and per-query arrays are archived study evidence; the source images, upstream SALAD model/code, training image cache, notebook metadata, internal console logs, staging markers, manuscripts and editorial material are not included. Specific source-code **storage defaults and comments** were sanitized; the full list of original and released SHA-256s is `docs/SOURCE_PROVENANCE.json`. No experimental results were changed, and the complete v4/v5/v6 hit matrices retain their original checksums. `msls_val_query_groups.json` differs only in the two source-file metadata path strings; the numeric grouping is unchanged.

**Known limits:** The exact ordered `msls_manifest.json`, original authorized MSLS and Gardens Point images, and SALAD pretrained trunk are needed for image-level replay. The bundled image bank is withheld pending exact redistribution clearance; the controlled raw-image simulator therefore cannot rerun from the package alone. The archived numerical analysis is independently reproducible on CPU; exact image-level recreation requires the separately licensed inputs documented below. See `docs/THIRD_PARTY_RIGHTS.md` and `docs/IMAGE_LEVEL_REPLAY.md`.

## Attribution and licensing

See `LICENSE` and `docs/THIRD_PARTY_RIGHTS.md`. Copyright in EUOT-VPR model artifacts remains with its respective rightsholders; **no third-party datasets or pretrained weights are relicensed or redistributed**. Original EUOT-VPR code is released under the MIT license; this does not grant rights to external benchmarks, images, upstream weights, or third-party code. For academic use, cite the associated article using the final bibliographic details when available.
