#!/usr/bin/env python3
"""Budget-matched verification-scheduling ablation for the controlled benchmark.

Ablation rationale: the two-stage scheduler (Eq. 8) uses both vacuity and the
top-1/top-2 margin, and vacuity also enters the stage-2 score (Eq. 11-12).
This script isolates those roles at matched verification budgets.

Design (per seed; no training -- every learned quantity is loaded from
`trained_euot_checkpoints.npz`):
  * Stage-1 descriptors, vacuity and reliability of the trained EUOT-VPR head
    are recomputed on the held-out test split (80 places x 5 occlusion levels).
  * The original scheduler assigns a shortlist size k_q in {0, 2..20}.
  * Budget-matched variants re-assign exactly the same multiset of k values to
    queries ranked by a different priority (vacuity only, margin only, random),
    so the number of triggered queries and the total number of verified
    candidates are identical to the original in every seed.
  * `uniform_matched` gives every query the same k (total budget preserved).
  * Score ablations replace vacuity in the stage-2 score by a constant (the mean
    vacuity of the validation queries, so no test statistic is used).

Statistics follow the paper's Implementation I: the unit is the place; pooled
contrasts use 240 seed-place realizations; paired two-sided sign-flip tests
(10,000 flips) and place-bootstrap 95% CIs (2,000 draws), seed 42.

Usage:
    python3 -I scheduling_ablation.py <controlled_benchmark_dir> <out_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

SEEDS = (42, 108, 2026)
OCC_LEVELS = (0.0, 0.10, 0.25, 0.40, 0.55)
N_PLACES = 80
N_RANDOM_DRAWS = 200
EXPECTED_IMPL_SHA256_PREFIX = None  # recorded, not enforced


def load_impl(path: Path):
    spec = importlib.util.spec_from_file_location("euot_impl", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def build_system(impl, ck, seed: int, mode: str):
    vocab = ck[f"seed{seed}_W_sal_vocab"].astype(np.float64)
    s = impl.IndependentlyTrainedVPRSystem(mode, learned_vocab=vocab, d_model=64, M_clusters=64, seed=seed)
    s.rg_W1 = ck[f"seed{seed}_rg_W1"].astype(np.float64)
    s.rg_b1 = ck[f"seed{seed}_rg_b1"].astype(np.float64)
    s.rg_K_dw = ck[f"seed{seed}_rg_K_dw"].astype(np.float64)
    s.rg_b_dw = ck[f"seed{seed}_rg_b_dw"].astype(np.float64)
    s.rg_W2 = ck[f"seed{seed}_rg_W2"].astype(np.float64)
    s.rg_b2 = float(ck[f"seed{seed}_rg_b2"][0])
    s.tau_1 = float(ck[f"seed{seed}_tau_1_learned"][0])
    s.tau_2 = float(ck[f"seed{seed}_tau_2_learned"][0])
    return s


def render_queries(gen, encoder, places, base_seed: int):
    rgbs, masks = [], []
    for q_idx, frac in enumerate(OCC_LEVELS):
        rgb, mask = gen.render_place_rgb_batch(places, view_seed=base_seed + q_idx, occ_frac=frac,
                                               is_database=False, cross_condition_shift=False)
        rgbs.append(rgb)
        masks.append(mask)
    rgb_all = np.concatenate(rgbs, axis=0)
    mask_all = np.concatenate(masks, axis=0)
    return encoder.extract_multistage_tokens_from_rgb(rgb_all), mask_all


def margins_of(sim: np.ndarray) -> np.ndarray:
    top2 = np.sort(np.partition(sim, -2, axis=1)[:, -2:], axis=1)
    return top2[:, 1] - top2[:, 0]


def hits_at1(sim: np.ndarray) -> np.ndarray:
    gt = np.tile(np.arange(N_PLACES), len(OCC_LEVELS))
    return (np.argmax(sim, axis=1) == gt).astype(np.uint8)


def apply_stage2(impl, sim0, out_q, out_db, k_vec, vac_score):
    sim = sim0.copy()
    for i, k in enumerate(k_vec):
        k = int(k)
        if k <= 0:
            continue
        row = sim0[i]
        top_k = np.argsort(-row)[:k]
        sim[i] = impl.exact_stage2_local_rerank_eq11_eq12(
            row, out_q["f_local"][i], out_q["r_pred"][i],
            out_db["f_local"][top_k], out_db["r_pred"][top_k], top_k, float(vac_score[i]),
        )
    return sim


def reassign(k_original: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Give the largest budgets to the highest-priority queries (stable ties)."""
    order = np.argsort(-priority, kind="mergesort")
    k_sorted = np.sort(k_original)[::-1]
    out = np.zeros_like(k_original)
    out[order] = k_sorted
    return out


def original_priority(impl, vac, margin):
    """Monotone surrogate of the original rule; reproduces it under `reassign`."""
    tau_u, tau_m = 0.036, 0.025
    pr = np.empty(len(vac))
    for i, (u, m) in enumerate(zip(vac, margin)):
        if (u < tau_u) and (m > tau_m):
            pr[i] = -1.0
        else:
            u_ex = max(0.0, (u - tau_u) / max(1e-6, 0.050 - tau_u))
            m_def = max(0.0, (tau_m - m) / max(1e-6, tau_m))
            pr[i] = min(1.0, max(0.25, u_ex + 0.5 * m_def))
    return pr


def paired_place_test(a: np.ndarray, b: np.ndarray, seed: int = 42, n_boot: int = 2000, n_flip: int = 10000):
    d = (a.astype(np.float64) - b.astype(np.float64)) * 100.0
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, n, size=(n_boot, n))
    lo, hi = np.percentile(d[bidx].mean(axis=1), [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_flip, n))
    null = np.abs((d[None, :] * signs).mean(axis=1))
    p = float((1 + np.sum(null >= abs(obs) - 1e-12)) / (n_flip + 1))
    return {"diff_pp": round(obs, 2), "ci95_lo_pp": round(float(lo), 2), "ci95_hi_pp": round(float(hi), 2),
            "p_two_sided": round(p, 4), "n_units": int(n), "n_nonzero": int(np.count_nonzero(d))}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for r, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - r) * p))
        adj[k] = round(running, 4)
    return adj


def run_seed(impl, ck, seed: int, log: Callable[[str], None]) -> Dict[str, np.ndarray]:
    t0 = time.time()
    gen = impl.DisjointRGBPlaceBenchmarkGenerator(d_model=64, M_clusters=64, n_total_places=360, seed=seed)
    encoder = impl.StandaloneMultiScaleVisualEncoder(d_model=64)
    val_places, test_places = np.arange(80, 160), np.arange(240, 320)

    sys2 = build_system(impl, ck, seed, "euot_adaptive_2stage")
    sys1 = build_system(impl, ck, seed, "euot_1stage")

    rgb_tdb, _ = gen.render_place_rgb_batch(test_places, view_seed=seed + 20, occ_frac=0.0, is_database=True)
    tdb = encoder.extract_multistage_tokens_from_rgb(rgb_tdb)
    (tq, mask_tq) = render_queries(gen, encoder, test_places, seed + 200)
    (vq, mask_vq) = render_queries(gen, encoder, val_places, seed + 50)
    log(f"seed {seed}: rendered and encoded test/val ({time.time() - t0:.1f}s)")

    out_tdb = sys2.forward_stage1(*tdb)
    out_tq = sys2.forward_stage1(*tq, oracle_occ_mask=mask_tq)
    out_vq = sys2.forward_stage1(*vq, oracle_occ_mask=mask_vq)
    out1_tdb = sys1.forward_stage1(*tdb)
    out1_tq = sys1.forward_stage1(*tq, oracle_occ_mask=mask_tq)

    sim0 = out_tq["global_desc"] @ out_tdb["global_desc"].T
    sim1 = out1_tq["global_desc"] @ out1_tdb["global_desc"].T
    vac = np.asarray(out_tq["vacuity"], dtype=np.float64)
    vac_const = np.full_like(vac, float(np.mean(out_vq["vacuity"])))
    marg = margins_of(sim0)

    k_orig = np.array([impl.compute_adaptive_shortlist_budget_k(float(u), float(m)) for u, m in zip(vac, marg)])
    res: Dict[str, np.ndarray] = {
        "single_stage_1stage_mode": hits_at1(sim1),
        "single_stage": hits_at1(sim0),
        "vacuity": vac,
        "margin": marg,
        "k_original": k_orig,
    }
    res["combined"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_orig, vac))
    k_check = reassign(k_orig, original_priority(impl, vac, marg))
    res["combined_reassigned_check"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_check, vac))
    k_vac = reassign(k_orig, vac)
    k_mar = reassign(k_orig, -marg)
    res["vacuity_only"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_vac, vac))
    res["margin_only"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_mar, vac))
    res["combined_const_vac_score"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_orig, vac_const))
    res["margin_only_const_vac_score"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_mar, vac_const))
    total = int(k_orig.sum())
    base, rem = divmod(total, len(k_orig))
    k_uni = np.full(len(k_orig), base)
    k_uni[np.argsort(-vac, kind="mergesort")[:rem]] += 1  # remainder: deterministic, negligible
    res["uniform_matched"] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_uni, vac))
    res["k_uniform"] = k_uni

    rng = np.random.default_rng(10_000 + seed)
    rand_hits = np.zeros((N_RANDOM_DRAWS, len(k_orig)), dtype=np.float64)
    for r in range(N_RANDOM_DRAWS):
        k_r = reassign(k_orig, rng.random(len(k_orig)))
        rand_hits[r] = hits_at1(apply_stage2(impl, sim0, out_tq, out_tdb, k_r, vac))
    res["random_expected"] = rand_hits.mean(axis=0)
    res["random_draw_q4_r1"] = rand_hits[:, 4 * N_PLACES:].mean(axis=1) * 100.0
    log(f"seed {seed}: done ({time.time() - t0:.1f}s); trigger={np.mean(k_orig > 0) * 100:.1f}% "
        f"mean_k={k_orig.mean():.2f}; Q4 R@1 combined={res['combined'][320:].mean() * 100:.2f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "scheduling_ablation.log", "a", buffering=1)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    impl_path = args.benchmark_dir / "euot_vpr_implementation.py"
    impl_sha = hashlib.sha256(impl_path.read_bytes()).hexdigest()
    impl = load_impl(impl_path)
    ck = np.load(args.benchmark_dir / "trained_euot_checkpoints.npz")
    audit = np.load(args.benchmark_dir / "per_query_audit_arrays.npz")
    log(f"implementation sha256={impl_sha[:12]}...; seeds={SEEDS}")

    per_seed: Dict[int, Dict[str, np.ndarray]] = {}
    for s in SEEDS:
        cache = out / f"seed{s}_arrays.npz"
        if cache.is_file():
            per_seed[s] = dict(np.load(cache))
            log(f"seed {s}: loaded cached arrays")
            continue
        r = run_seed(impl, ck, s, log)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **r)
        os.replace(tmp, cache)
        per_seed[s] = r

    # Reproduction gate against the archived per-query hits.
    repro = {}
    for s in SEEDS:
        for ours, arch in (("combined", "euot_adaptive_2stage"), ("single_stage_1stage_mode", "euot_1stage")):
            a = audit[f"s3_seed{s}_{arch}_test_hits"].astype(np.uint8)
            b = per_seed[s][ours].astype(np.uint8)
            repro[f"seed{s}_{ours}_vs_archived_{arch}"] = {"agree": int((a == b).sum()), "n": int(len(a))}
        va = audit[f"s3_seed{s}_euot_adaptive_2stage_test_vacuity"].astype(np.float64)
        repro[f"seed{s}_vacuity_max_abs_diff"] = float(np.max(np.abs(va - per_seed[s]["vacuity"])))
        repro[f"seed{s}_reassign_check_agree"] = int((per_seed[s]["combined_reassigned_check"] == per_seed[s]["combined"]).sum())

    arms = ["single_stage", "combined", "vacuity_only", "margin_only", "combined_const_vac_score",
            "margin_only_const_vac_score", "uniform_matched", "random_expected"]
    level_names = ["clean", "occ10", "occ25", "occ40", "occ55"]
    summary = {a: {} for a in arms}
    for a in arms:
        for li, ln in enumerate(level_names):
            vals = np.concatenate([per_seed[s][a][li * N_PLACES:(li + 1) * N_PLACES] for s in SEEDS]).astype(np.float64)
            summary[a][ln] = round(float(vals.mean() * 100), 2)
        summary[a]["per_seed_occ55"] = [round(float(per_seed[s][a][320:].mean() * 100), 2) for s in SEEDS]

    budgets = {str(s): {"trigger_pct": round(float(np.mean(per_seed[s]["k_original"] > 0) * 100), 2),
                        "mean_k": round(float(per_seed[s]["k_original"].mean()), 3),
                        "total_k": int(per_seed[s]["k_original"].sum())} for s in SEEDS}

    def q4(a):
        return np.concatenate([per_seed[s][a][320:] for s in SEEDS]).astype(np.float64)

    def occluded(a):
        # mean over the four occluded levels within each place -> 240 seed-place units
        return np.concatenate([per_seed[s][a][80:].astype(np.float64).reshape(4, N_PLACES).mean(axis=0)
                               for s in SEEDS])

    primary_pairs = {
        "combined_minus_margin_only": ("combined", "margin_only"),
        "combined_minus_vacuity_only": ("combined", "vacuity_only"),
        "combined_minus_margin_only_const_vac_score": ("combined", "margin_only_const_vac_score"),
        "combined_minus_uniform_matched": ("combined", "uniform_matched"),
    }
    secondary_pairs = {
        "vacuity_only_minus_margin_only": ("vacuity_only", "margin_only"),
        "combined_minus_combined_const_vac_score": ("combined", "combined_const_vac_score"),
        "combined_minus_random_expected": ("combined", "random_expected"),
        "vacuity_only_minus_random_expected": ("vacuity_only", "random_expected"),
        "margin_only_minus_random_expected": ("margin_only", "random_expected"),
        "combined_minus_single_stage": ("combined", "single_stage"),
    }
    tests = {"occ55_pooled_240_places": {}, "occ10_to_55_place_means_240": {}}
    for fam_name, fam in (("primary", primary_pairs), ("secondary", secondary_pairs)):
        for k, (a, b) in fam.items():
            tests["occ55_pooled_240_places"][k] = paired_place_test(q4(a), q4(b)) | {"family": fam_name}
            tests["occ10_to_55_place_means_240"][k] = paired_place_test(occluded(a), occluded(b)) | {"family": fam_name}
    holm_q4 = holm({k: v["p_two_sided"] for k, v in tests["occ55_pooled_240_places"].items() if v["family"] == "primary"})
    for k, v in holm_q4.items():
        tests["occ55_pooled_240_places"][k]["p_holm_primary"] = v

    rand_q4 = np.concatenate([per_seed[s]["random_draw_q4_r1"][:, None] for s in SEEDS], axis=1).mean(axis=1)
    payload = {
        "schema": "euot_controlled_scheduling_ablation_v1",
        "implementation_sha256": impl_sha,
        "seeds": list(SEEDS),
        "design": "budget-matched reassignment of the original shortlist sizes; no retraining",
        "reproduction_gate": repro,
        "budgets_identical_across_matched_arms": budgets,
        "recall_at_1_pct": summary,
        "random_priority_occ55_r1_pct": {"mean": round(float(rand_q4.mean()), 2),
                                          "p2_5": round(float(np.percentile(rand_q4, 2.5)), 2),
                                          "p97_5": round(float(np.percentile(rand_q4, 97.5)), 2),
                                          "n_draws": N_RANDOM_DRAWS},
        "paired_tests": tests,
        "notes": [
            "Units are places; occ55 pools 240 seed-place realizations (one query per place and seed).",
            "The occ10-55 contrast averages the four occluded queries within each place and seed before testing (240 units).",
            "Stage-2 computation is identical across matched arms because the multiset of shortlist sizes is identical.",
            "Constant-vacuity score uses the mean vacuity of validation queries (P_val), never test statistics.",
        ],
    }
    atomic_json(out / "scheduling_ablation_results.json", payload)
    log("wrote scheduling_ablation_results.json")


if __name__ == "__main__":
    main()
