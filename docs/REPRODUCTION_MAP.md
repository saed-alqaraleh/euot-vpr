# Manuscript-to-evidence reproduction map

The manuscript, review correspondence, and internal logs are intentionally absent. The manuscript supplied with the task was used solely to check the release's headline against the archived numerical records.

| Manuscript result | Archived evidence | CPU command |
|---|---|---|
| Severe-occlusion factorials in controlled benchmark (55% target Q4) | `controlled_benchmark/per_query_audit_arrays.npz` | `analysis/recompute_msls_statistics.py` final two lines; numerator arrays per seed/place |
| Four MSLS-val combined-gate crossed interactions at 50% mask | `msls_setting_U/hit_arrays_v4/` for `U`; `msls_setting_S/hit_arrays_v5/` for `S`; `msls_crossed_and_gates_v6/hits_v6/` for `RE`, `UW` | `scripts/reproduce_headline.py` and full `analysis/recompute_v6_statistics.py` |
| Factorial ungated-relaxation intervals | Same `(740,3) uint8` hit arrays | Both scripts above |
| Mass-only and other gate decomposition | v6 gate-variant hit arrays; original `U` and `S` ungated baselines | `scripts/reproduce_headline.py` and `analysis/recompute_v6_statistics.py` |
| Crossed normalization and global-share main effects | Four setting-specific DiD arrays | `analysis/recompute_v6_statistics.py`; Holm Family A |
| Diagnostics of disrupted-cluster energy/mass | `msls_crossed_and_gates_v6/results/diagnostics_arrays.npz` and `diagnostics.json` | `analysis/recompute_gating_energy.py` |
| 363 sequence and 684 positive-connected-component resamples | `msls_val_query_groups.json`, original per-query arrays | `analysis/recompute_v6_statistics.py` |
| Re-evaluation equivalence of archived heads | v6 `reproduction_check.json`; v4/v5 original and v6 rerun hit arrays | `scripts/verify_release.py` and archived `reproduction_check.json` |
| Original controlled benchmark and scheduler | `controlled_benchmark/*results.json`, per-query arrays, `scheduling_ablation/` | Check archived values; simulator rerun requires licensed image bank |
| Mechanistic/figure numeric sources | `analysis/figure_scripts/figure_data.json`, `analysis/v6_statistics_recomputed.json` | `analysis/figure_scripts` (see scripts' command-line help) |

**Important:** The two original MSLS-val settings were analyzed using their initial archived hit arrays, *not* the later re-evaluation. For 72 U/S re-evaluated comparisons, 58 are identical across R@1/R@5/R@10; minimum R@1 agreement is 99.73%, according to the archived `reproduction_check.json`.
