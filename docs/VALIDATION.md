# Release verification record

**Verification date:** 2026-11-15. The uploaded revised manuscript was treated as the scientific record; the supplementary archive supplied the original numerical arrays and code. The manuscript and the original supplementary ZIP are not packaged.

## Passed CPU and integrity checks

Run from the repository root; generated outputs are ignored by Git:

| Command | Observed verification |
| --- | --- |
| `python scripts/verify_release.py` | Works both from extracted ZIP and a fresh Git-bundle clone (ignores Git metadata); every released file is SHA-256-checked against `SHA256SUMS.txt`; all 469 source-manifest entries matched; validated 42 v4, 36 v5 and 288 v6 `(740,3)` binary `uint8` per-query hit matrices; rejected known confidential paths and forbidden file extensions. |
| `python scripts/reproduce_headline.py --output outputs/headline_msls.json` | Recomputed the four severe-occlusion factorials, their paired intervals, ungated contrasts, the mass-only comparison, the crossed global-token-share interaction and the SALAD reference from archived arrays; all checks passed at reported precision. |
| `python msls_crossed_and_gates_v6/kit/tests/test_kit_cpu.py` | 12 tests passed; includes archived-head forward-equivalence and recorded training-loop checks. No GPU images were used. |
| `python analysis/recompute_v6_statistics.py --out outputs/v6_statistics.json` | Complete independently regenerated v6 numerical JSON compared equal as a parsed JSON object to the archived `analysis/v6_statistics_recomputed.json`. |
| `python analysis/recompute_msls_statistics.py --setting_u msls_setting_U --setting_s msls_setting_S --controlled controlled_benchmark` | Original U/S checksum and numerical contrasts, including the stored controlled severe-occlusion comparison, passed. |
| `python analysis/recompute_gating_energy.py` | Reproduced both archived gated-energy summaries, 363 grouped sequences. |

These are saved-evidence tests. **They do not constitute fresh model training, complete image-level replay, or permission to distribute the source images.** Source-code hygiene changes are listed individually with old/new digests in `SOURCE_PROVENANCE.json`; archive hit arrays and author-fitted checkpoint weights were not numerically changed.

## Independent bundle validation

The original review Git bundle was checked with `git bundle verify` from a new empty Git repository, then cloned into a separate directory. The checkout contained 518 tracked files and a clean `main` branch. The release verifier was amended to exclude the clone's `.git` metadata and standard ignored generated directories from its filesystem integrity comparison. All archived numerical arrays and study-fitted checkpoints remained unchanged.

## Exclusion and review policy

The release excludes manuscript, private local and cloud paths, notebook source/metadata, `.log` files, notebook autosaves, internal staging files, editorial communication, credentials, raw image-bank assets of unresolved license status, full benchmark images, and upstream SALAD pretrained weights. The CPU-checkable control-study arrays and recorded numerical outcomes remain present.

## External requirements and scope limits

1. Original EUOT-VPR code is under the MIT license. The license does not apply to unrelated upstream software, images, model weights, or any third-party benchmark content; publication must respect the independent applicable terms.
2. Rights and attribution for all five images in the original controlled-benchmark bank are not fully documented. The image bank is deliberately withheld, so an exact raw-image rerun of that control study is not self-contained.
3. Exact MSLS-val ordered gallery/query manifest and positives, authorized MSLS and Gardens Point images, historical pretrained SALAD backbone/checkpoint and a complete GPU environment lock are not present. Full image-level reproducibility has not been demonstrated.
4. Remote GitHub upload and post-push verification must be performed separately. Do not assume a clean local archive proves that every file has arrived on the remote.

See `IMAGE_LEVEL_REPLAY.md` and `THIRD_PARTY_RIGHTS.md` for required external inputs and permission scope.
