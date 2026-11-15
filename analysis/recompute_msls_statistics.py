#!/usr/bin/env python3
"""Recompute every MSLS-val statistic reported in the paper from the archived hit arrays,
and the controlled-benchmark relaxation-alone contrast (Table 4) from the per-query arrays.

Usage (from the supplementary root):
    python analysis/recompute_msls_statistics.py \
        --setting_u msls_setting_U --setting_s msls_setting_S --controlled controlled_benchmark

Tests follow the runners: per-query hits are averaged over the three seeds, query identities
are resampled (2,000 bootstrap draws) and a two-sided Monte Carlo sign-flip test with 10,000
flips is used (NumPy default_rng, seed 42). Controlled-benchmark contrasts are place-level.
The script also verifies every file against the SHA-256 manifests before computing anything.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = ["s42", "s108", "s2026"]
ARMS = ["rgtmp_bal", "rgtmp_bal_gate", "euot_no_gate", "euot_1stage"]   # Bal, Bal+G, UOT, UOT+G
OCC = ["occ00", "occ25", "occ50"]


def verify_manifest(folder: Path) -> None:
    man = json.load(open(folder / "SHA256_MANIFEST.json"))
    entries = []  # manifest layout: {subfolder: {filename: {"bytes": int, "sha256": str}}}
    for sub, files in man.items():
        if isinstance(files, dict) and all(isinstance(v, dict) for v in files.values()):
            entries += [(folder / sub / name, rec["sha256"]) for name, rec in files.items()]
    bad = [p.name for p, digest in entries
           if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest]
    print(f"[manifest] {folder.name}: {len(entries)} files checked, {len(bad)} mismatches {bad if bad else ''}")
    if bad:
        raise SystemExit("manifest mismatch")


def load_hits(folder: Path, sub: str, prefix: str):
    return {p.name[len(prefix):-len("_hits.npy")]: np.load(p) for p in (folder / sub).glob("*.npy")}


def test(d, seed=42, n_boot=2000, n_perm=10000):
    n = len(d)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    p = (1 + np.sum(np.abs((d[None] * signs).mean(1)) >= abs(obs) - 1e-12)) / (n_perm + 1)
    return f"{obs:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] p={p:.4f}"


def m3(H, arm, occ, col=0):
    return np.mean([H[f"{arm}_{s}_{occ}"][:, col].astype(float) for s in SEEDS], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting_u", type=Path, default=Path("msls_setting_U"))
    ap.add_argument("--setting_s", type=Path, default=Path("msls_setting_S"))
    ap.add_argument("--controlled", type=Path, default=Path("controlled_benchmark"))
    a = ap.parse_args()

    verify_manifest(a.setting_u)
    verify_manifest(a.setting_s)
    HU = load_hits(a.setting_u, "hit_arrays_v4", "msls_")
    HS = load_hits(a.setting_s, "hit_arrays_v5", "msls_v5_")

    for occ in OCC:
        print(f"\n=== mask {occ[3:]}% ===")
        sal = HU[f"official_pretrained_salad_{occ}"][:, 0].astype(float)
        pca = HU[f"pca128_kmeans_control_{occ}"][:, 0].astype(float)
        print(f"released SALAD R@1 {sal.mean()*100:.2f} | PCA-128 control R@1 {pca.mean()*100:.2f}")
        res = {}
        for tag, H in (("U", HU), ("S", HS)):
            B, BG, U, UG = (m3(H, arm, occ) for arm in ARMS)
            res[tag] = (B, BG, U, UG)
            for col, lab in ((0, "R@1"), (1, "R@5"), (2, "R@10")):
                vals = {arm: [H[f"{arm}_{s}_{occ}"][:, col].mean() * 100 for s in SEEDS] for arm in ARMS}
                print(f" Setting {tag} {lab:4s}: " + " | ".join(
                    f"{arm} {np.mean(v):.2f}±{np.std(v, ddof=1):.2f}" for arm, v in vals.items()))
            print(f" Setting {tag} Relax.      ", test((U - B) * 100))
            print(f" Setting {tag} G|Bal       ", test((BG - B) * 100))
            print(f" Setting {tag} G|UOT       ", test((UG - U) * 100))
            print(f" Setting {tag} UOT+G - Bal ", test((UG - B) * 100))
            print(f" Setting {tag} DiD         ", test(((UG - U) - (BG - B)) * 100))
            for col, lab in ((1, "R@5"), (2, "R@10")):
                b, bg, u, ug = (m3(H, arm, occ, col) for arm in ARMS)
                print(f" Setting {tag} {lab}: DiD {((ug-u)-(bg-b)).mean()*100:+.2f}, UOT+G - Bal {(ug-b).mean()*100:+.2f}")
            per_seed = [(((H[f"euot_1stage_{s}_{occ}"][:, 0] - H[f"euot_no_gate_{s}_{occ}"][:, 0].astype(float))
                          - (H[f"rgtmp_bal_gate_{s}_{occ}"][:, 0] - H[f"rgtmp_bal_{s}_{occ}"][:, 0].astype(float))).mean() * 100)
                        for s in SEEDS]
            print(f" Setting {tag} per-seed DiD", [round(x, 2) for x in per_seed])
            for arm, x in zip(("Bal", "Bal+G", "UOT", "UOT+G"), (B, BG, U, UG)):
                print(f" Setting {tag} {arm:5s} - released SALAD", test((x - sal) * 100))
        (Bu, BGu, Uu, UGu), (Bs, BGs, Us, UGs) = res["U"], res["S"]
        print(" S - U interaction        ", test((((UGs - Us) - (BGs - Bs)) - ((UGu - Uu) - (BGu - Bu))) * 100))
        for arm, xs, xu in zip(("Bal", "Bal+G", "UOT", "UOT+G"), res["S"], res["U"]):
            print(f" S - U same arm {arm:5s}    ", test((xs - xu) * 100))
        print(" S UOT+G - PCA control    ", test((UGs - pca) * 100))

    # Controlled benchmark, Table 4: relaxation alone at Q4 (55%), place level, two-sided
    z = np.load(a.controlled / "per_query_audit_arrays.npz")
    seeds = (42, 108, 2026)

    def q4(mode, s):  # 400 test queries per seed = 5 occlusion levels x 80 places (level-major)
        return z[f"s3_seed{s}_{mode}_test_hits"].reshape(5, 80)[4].astype(float)

    for proto, suf in (("A", ""), ("B", "_indep")):
        B = np.concatenate([q4("rgtmp_bal" + suf, s) for s in seeds])
        U = np.concatenate([q4("euot_no_gate" + suf, s) for s in seeds])
        print(f"\n[controlled] Protocol {proto} relaxation alone at Q4:", test((U - B) * 100))


if __name__ == "__main__":
    main()
