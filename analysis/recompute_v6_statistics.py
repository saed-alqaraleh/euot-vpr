#!/usr/bin/env python3
"""Recompute every MSLS-val statistic of the crossed design, the gate decomposition, and the grouped tests.

Usage (from the supplementary root; CPU, NumPy only, about 10 minutes):
    python analysis/recompute_v6_statistics.py --out analysis/v6_statistics_recomputed.json

Inputs: the archived Setting U_1/2 and R_1/65 hit arrays (msls_setting_U, msls_setting_S) and the hit arrays of the
crossed settings and gate variants (msls_crossed_and_gates_v6). All digests are verified first. As in the paper,
U_1/2 and R_1/65 use their archived arrays; the re-evaluated copies in msls_crossed_and_gates_v6 are used only for the
reproduction check. Tests: three-seed per-query means, 2,000 query-bootstrap draws, two-sided sign-flip tests with
10,000 flips (NumPy default_rng(42)); grouped tests resample the MSLS-val sequences / positive-sharing components.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "msls_crossed_and_gates_v6" / "kit"))
import euot_v6_analysis as A  # noqa: E402


def verify(manifest: Path, base: Path) -> int:
    man = json.load(open(manifest))
    entries = []
    for k, v in man.items():
        if isinstance(v, dict) and all(isinstance(x, dict) for x in v.values()):
            entries += [(base / k / name, rec["sha256"]) for name, rec in v.items()]
        elif isinstance(v, str):
            entries.append((base / k, v))
    bad = [str(p) for p, d in entries if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != d]
    if bad:
        raise SystemExit(f"manifest mismatch in {manifest}: {bad[:5]}")
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "v6_statistics_recomputed.json")
    a = ap.parse_args()
    U, S, V = ROOT / "msls_setting_U", ROOT / "msls_setting_S", ROOT / "msls_crossed_and_gates_v6"
    n = verify(U / "SHA256_MANIFEST.json", U) + verify(S / "SHA256_MANIFEST.json", S) + verify(V / "SHA256_MANIFEST.json", V)
    print(f"[manifest] {n} files verified")
    H = A.HitStore()
    H.add_v6_dir(V / "hits_v6")
    H.add_archived(U / "hit_arrays_v4", "U", "msls_")
    H.add_archived(S / "hit_arrays_v5", "S", "msls_v5_")
    g = json.load(open(V / "results" / "msls_val_query_groups.json"))
    groups = {k: np.asarray(v) for k, v in g["group_ids"].items() if v is not None and k != "city"}
    res = A.analyse(H, groups)
    # same-arm contrasts between settings and against the released SALAD
    sal = {o: np.load(U / "hit_arrays_v4" / f"msls_official_pretrained_salad_{o}_hits.npy")[:, 0].astype(float) for o in A.OCCS}
    arms = [("bal", "none"), ("bal", "combined"), ("uot", "none"), ("uot", "combined")]
    extra = {}
    for o in A.OCCS:
        for x, y, lab in (("UW", "U", "share_at_unit"), ("S", "RE", "share_at_raw"), ("S", "UW", "normalization_at_1_65"), ("RE", "U", "normalization_at_1_2")):
            for t, gt in arms:
                extra[f"{lab}_{t}_{gt}_{o}"] = A.query_test((H.mean3(x, t, gt, o) - H.mean3(y, t, gt, o)) * 100)
        for s in ("U", "RE", "UW", "S"):
            for t, gt in arms:
                extra[f"vs_released_salad_{s}_{t}_{gt}_{o}"] = A.query_test((H.mean3(s, t, gt, o) - sal[o]) * 100)
    res["setting_and_reference_contrasts"] = extra
    rep = json.load(open(V / "results" / "reproduction_check.json"))
    res["reproduction_check_summary"] = {k: rep[k] for k in ("min_r1_agreement", "n_identical", "n_compared")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    a.out.with_suffix(".md").write_text(A.to_markdown(res))
    fa = res["familyA_crossed_design"]["occ50"]
    print("interaction at 50%:", {s: res["per_setting_factorial"][s]["occ50"]["interaction_did"]["diff_pp"] for s in ("U", "RE", "UW", "S")})
    print("share main effect:", fa["global_share_main_effect"], "\nnormalization main effect:", fa["normalization_main_effect"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
