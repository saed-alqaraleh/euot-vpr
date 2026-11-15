#!/usr/bin/env python3
"""EUOT-VPR v6 statistics from per-query hit arrays (NumPy only).

Runs on CPU from the hit arrays the GPU stages write, so the numbers can be
recomputed anywhere. Tests replicate the paper's MSLS protocol exactly: per-query
hits are averaged over the three seeds, query identities are resampled (2,000
bootstrap draws) and a two-sided Monte Carlo sign-flip test (10,000 flips) is
used, NumPy default_rng(42). Optionally, the same contrasts are recomputed with
query groups (sequence or positive-overlap components) as resampling units.

Pre-specified families (Holm within each family):
  A  crossed design at 50% masking: normalization main effect on the
     transport x gate interaction, global-share main effect on it, and their
     interaction (3 tests);
  B  gate variants at 50% masking, separately in Settings U and S: the
     transport x gate interaction for each of the five gates (5 tests each).
Everything else is reported as secondary.

Usage:
    python3 euot_v6_analysis.py --hits_dir <out>/hits_v6 --out_json <file> \
        [--groups <msls_val_query_groups.json>] [--diagnostics <diagnostics.json>]
    # verify against the archived Settings U/S arrays (supplementary layout):
    python3 euot_v6_analysis.py --archived_u msls_setting_U/hit_arrays_v4 \
        --archived_s msls_setting_S/hit_arrays_v5 --out_json check.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SEEDS = (42, 108, 2026)
OCCS = ("occ00", "occ25", "occ50")
SETTINGS = ("U", "S", "UW", "RE")
GATE_VARIANTS = ("combined", "reliability", "length", "mass", "meannorm")
LEGACY = {("bal", "none"): "rgtmp_bal", ("bal", "combined"): "rgtmp_bal_gate",
          ("uot", "none"): "euot_no_gate", ("uot", "combined"): "euot_1stage"}


# ------------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------------

def query_test(d: np.ndarray, seed: int = 42, n_boot: int = 2000, n_perm: int = 10000) -> Dict[str, float]:
    """Identical to recompute_msls_statistics.test (same RNG call order)."""
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    p = float((1 + np.sum(np.abs((d[None] * signs).mean(1)) >= abs(obs) - 1e-12)) / (n_perm + 1))
    return {"diff_pp": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "p": round(p, 4), "n_units": int(n)}


def grouped_test(d: np.ndarray, groups: np.ndarray, seed: int = 42, n_boot: int = 2000, n_perm: int = 10000) -> Dict[str, float]:
    """Cluster bootstrap (resample groups) and cluster sign-flip test (flip each
    group's summed difference). Reduces to query_test when every group is a singleton
    in distribution, not in RNG sequence."""
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    sums = np.bincount(inv, weights=d, minlength=G)
    counts = np.bincount(inv, minlength=G).astype(np.float64)
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, G))
    null = np.abs((signs * sums[None]).sum(1) / n)
    p = float((1 + np.sum(null >= abs(obs) - 1e-12)) / (n_perm + 1))
    return {"diff_pp": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "p": round(p, 4),
            "n_units": int(G), "n_queries": int(n)}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for r, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - r) * p))
        out[k] = round(running, 4)
    return out


# ------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------

class HitStore:
    """Maps (setting, transport, gate, seed, occ) -> (n_query, 3) uint8 hits."""

    def __init__(self) -> None:
        self.h: Dict[Tuple[str, str, str, int, str], np.ndarray] = {}

    def add_v6_dir(self, d: Path) -> int:
        n = 0
        for p in sorted(d.glob("msls_v6_*_hits.npy")):
            stem = p.name[len("msls_v6_"):-len("_hits.npy")]
            parts = stem.split("_")
            if len(parts) != 5:
                continue
            setting, transport, gate, seed_s, occ = parts
            self.h[(setting, transport, gate, int(seed_s[1:]), occ)] = np.load(p)
            n += 1
        return n

    def add_archived(self, d: Path, setting: str, prefix: str) -> int:
        n = 0
        for (transport, gate), mode in LEGACY.items():
            for s in SEEDS:
                for occ in OCCS:
                    p = d / f"{prefix}{mode}_s{s}_{occ}_hits.npy"
                    if p.is_file():
                        self.h[(setting, transport, gate, s, occ)] = np.load(p)
                        n += 1
        return n

    def has(self, setting: str, transport: str, gate: str, occ: str) -> bool:
        return all((setting, transport, gate, s, occ) in self.h for s in SEEDS)

    def mean3(self, setting: str, transport: str, gate: str, occ: str, col: int = 0) -> np.ndarray:
        return np.mean([self.h[(setting, transport, gate, s, occ)][:, col].astype(np.float64) for s in SEEDS], axis=0)

    def per_seed(self, setting: str, transport: str, gate: str, occ: str, col: int = 0) -> List[float]:
        return [float(self.h[(setting, transport, gate, s, occ)][:, col].mean() * 100) for s in SEEDS]


def did(H: HitStore, setting: str, occ: str, gate: str = "combined", col: int = 0) -> np.ndarray:
    B = H.mean3(setting, "bal", "none", occ, col)
    BG = H.mean3(setting, "bal", gate, occ, col)
    U = H.mean3(setting, "uot", "none", occ, col)
    UG = H.mean3(setting, "uot", gate, occ, col)
    return ((UG - U) - (BG - B)) * 100.0


# ------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------

def analyse(H: HitStore, groups: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, object]:
    out: Dict[str, object] = {"protocol": {
        "unit": "query identity; per-query hits averaged over seeds 42/108/2026",
        "bootstrap_draws": 2000, "sign_flips": 10000, "rng": "numpy default_rng(42)", "sidedness": "two-sided"}}

    # Arm-level means for every available cell
    arms = {}
    for (setting, transport, gate, seed, occ) in sorted(H.h):
        if seed != SEEDS[0] or not H.has(setting, transport, gate, occ):
            continue
        key = f"{setting}_{transport}_{gate}"
        arms.setdefault(key, {})[occ] = {
            f"R@{k}": {"mean": round(float(np.mean(H.per_seed(setting, transport, gate, occ, c))), 2),
                       "sd": round(float(np.std(H.per_seed(setting, transport, gate, occ, c), ddof=1)), 2)}
            for c, k in ((0, 1), (1, 5), (2, 10))}
    out["arm_recall_pct"] = arms

    def std_ok(setting: str, occ: str, gate: str = "combined") -> bool:
        return all(H.has(setting, t, g, occ) for t, g in (("bal", "none"), ("bal", gate), ("uot", "none"), ("uot", gate)))

    # Per-setting standard decomposition
    per_setting = {}
    for setting in SETTINGS:
        for occ in OCCS:
            if not std_ok(setting, occ):
                continue
            B, BG = H.mean3(setting, "bal", "none", occ), H.mean3(setting, "bal", "combined", occ)
            U, UG = H.mean3(setting, "uot", "none", occ), H.mean3(setting, "uot", "combined", occ)
            per_setting.setdefault(setting, {})[occ] = {
                "relaxation_alone": query_test((U - B) * 100),
                "gate_under_bal": query_test((BG - B) * 100),
                "gate_under_uot": query_test((UG - U) * 100),
                "uotg_minus_bal": query_test((UG - B) * 100),
                "interaction_did": query_test(did(H, setting, occ)),
                "interaction_did_r5": round(float(did(H, setting, occ, col=1).mean()), 2),
                "interaction_did_r10": round(float(did(H, setting, occ, col=2).mean()), 2),
            }
    out["per_setting_factorial"] = per_setting

    # Family A: crossed design on the interaction
    crossed = {}
    for occ in OCCS:
        if not all(std_ok(s, occ) for s in SETTINGS):
            continue
        dU, dS, dUW, dRE = (did(H, s, occ) for s in ("U", "S", "UW", "RE"))
        c = {
            "normalization_main_effect": query_test(0.5 * ((dS - dUW) + (dRE - dU))),
            "global_share_main_effect": query_test(0.5 * ((dUW - dU) + (dS - dRE))),
            "normalization_x_share": query_test((dS - dUW) - (dRE - dU)),
            "simple_normalization_at_share_1_65": query_test(dS - dUW),
            "simple_normalization_at_share_50": query_test(dRE - dU),
            "simple_share_at_unit_tokens": query_test(dUW - dU),
            "simple_share_at_raw_tokens": query_test(dS - dRE),
            "S_minus_U_both_changed": query_test(dS - dU),
            "sign_convention": "normalization: raw minus unit tokens; share: 1/65 minus 50%",
        }
        if occ == "occ50":
            adj = holm({k: c[k]["p"] for k in ("normalization_main_effect", "global_share_main_effect", "normalization_x_share")})
            for k, v in adj.items():
                c[k]["p_holm_familyA"] = v
        crossed[occ] = c
    out["familyA_crossed_design"] = crossed

    # Family B: gate variants in each setting where they exist
    gates = {}
    for setting in SETTINGS:
        for occ in OCCS:
            avail = [g for g in GATE_VARIANTS if std_ok(setting, occ, g)]
            if len(avail) < 2:
                continue
            rec = {}
            for g in avail:
                B, BG = H.mean3(setting, "bal", "none", occ), H.mean3(setting, "bal", g, occ)
                U, UG = H.mean3(setting, "uot", "none", occ), H.mean3(setting, "uot", g, occ)
                rec[g] = {
                    "interaction_did": query_test(did(H, setting, occ, g)),
                    "gate_under_uot": query_test((UG - U) * 100),
                    "gate_under_bal": query_test((BG - B) * 100),
                    "uotg_minus_bal": query_test((UG - B) * 100),
                    "uotg_r1_pct": round(float(UG.mean() * 100), 2),
                    "balg_r1_pct": round(float(BG.mean() * 100), 2),
                }
            if occ == "occ50":
                adj = holm({g: rec[g]["interaction_did"]["p"] for g in avail})
                for g, v in adj.items():
                    rec[g]["interaction_did"]["p_holm_familyB"] = v
            gates.setdefault(setting, {})[occ] = rec
    out["familyB_gate_variants"] = gates

    # Grouped robustness for the headline contrasts
    if groups:
        grouped = {}
        for gname, g in groups.items():
            rec = {}
            for occ in OCCS:
                r = {}
                for setting in SETTINGS:
                    if std_ok(setting, occ):
                        r[f"{setting}_interaction_did"] = grouped_test(did(H, setting, occ), g)
                        B = H.mean3(setting, "bal", "none", occ)
                        UG = H.mean3(setting, "uot", "combined", occ)
                        r[f"{setting}_uotg_minus_bal"] = grouped_test((UG - B) * 100, g)
                if std_ok("U", occ) and std_ok("S", occ):
                    r["S_minus_U_interaction"] = grouped_test(did(H, "S", occ) - did(H, "U", occ), g)
                if occ in crossed:
                    dU, dS, dUW, dRE = (did(H, s, occ) for s in ("U", "S", "UW", "RE"))
                    r["normalization_main_effect"] = grouped_test(0.5 * ((dS - dUW) + (dRE - dU)), g)
                    r["global_share_main_effect"] = grouped_test(0.5 * ((dUW - dU) + (dS - dRE)), g)
                    r["normalization_x_share"] = grouped_test((dS - dUW) - (dRE - dU), g)
                rec[occ] = r
            grouped[gname] = {"n_groups": int(len(np.unique(g))), "results": rec}
        out["grouped_robustness"] = grouped
    return out


def to_markdown(res: Dict[str, object]) -> str:
    L: List[str] = ["# EUOT-VPR v6 results", ""]

    def fmt(t):
        s = f"{t['diff_pp']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}], p={t['p']:.4f}"
        for k in ("p_holm_familyA", "p_holm_familyB"):
            if k in t:
                s += f", Holm p={t[k]:.4f}"
        return s

    ps = res.get("per_setting_factorial", {})
    if ps:
        L += ["## Transport x gate interaction per setting (pp, Recall@1)", "",
              "| Setting | Mask | Relax. alone | G|Bal | G|UOT | DiD |", "|---|---|---|---|---|---|"]
        for s, occs in ps.items():
            for occ, r in occs.items():
                L.append(f"| {s} | {occ[3:]}% | {fmt(r['relaxation_alone'])} | {fmt(r['gate_under_bal'])} | "
                         f"{fmt(r['gate_under_uot'])} | {fmt(r['interaction_did'])} |")
        L.append("")
    cr = res.get("familyA_crossed_design", {})
    if cr:
        L += ["## Family A: crossed design (effects on the interaction, pp)", "", "| Mask | Contrast | Estimate |", "|---|---|---|"]
        for occ, c in cr.items():
            for k, t in c.items():
                if isinstance(t, dict):
                    L.append(f"| {occ[3:]}% | {k} | {fmt(t)} |")
        L.append("")
    gv = res.get("familyB_gate_variants", {})
    if gv:
        L += ["## Family B: gate variants (pp)", "", "| Setting | Mask | Gate | DiD | G|UOT | G|Bal | UOT+g R@1 |", "|---|---|---|---|---|---|---|"]
        for s, occs in gv.items():
            for occ, rec in occs.items():
                for g, r in rec.items():
                    L.append(f"| {s} | {occ[3:]}% | {g} | {fmt(r['interaction_did'])} | {fmt(r['gate_under_uot'])} | "
                             f"{fmt(r['gate_under_bal'])} | {r['uotg_r1_pct']:.2f} |")
        L.append("")
    gr = res.get("grouped_robustness")
    if gr:
        L += ["## Grouped robustness", ""]
        for gname, rec in gr.items():
            L += [f"### groups = {gname} ({rec['n_groups']} groups)", "", "| Mask | Contrast | Estimate |", "|---|---|---|"]
            for occ, r in rec["results"].items():
                for k, t in r.items():
                    L.append(f"| {occ[3:]}% | {k} | {fmt(t)} |")
            L.append("")
    if "diagnostics" in res:
        L += ["## Absorption diagnostic", "", "```", json.dumps(res["diagnostics"].get("summary", res["diagnostics"]), indent=1)[:20000], "```", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits_dir", type=Path, default=None)
    ap.add_argument("--archived_u", type=Path, default=None, help="folder with archived msls_<mode>_s<seed>_<occ>_hits.npy (Setting U)")
    ap.add_argument("--archived_s", type=Path, default=None, help="folder with archived msls_v5_<mode>_s<seed>_<occ>_hits.npy (Setting S)")
    ap.add_argument("--groups", type=Path, default=None)
    ap.add_argument("--diagnostics", type=Path, default=None)
    ap.add_argument("--out_json", type=Path, required=True)
    a = ap.parse_args()

    H = HitStore()
    n = 0
    if a.archived_u:
        n += H.add_archived(a.archived_u, "U", "msls_")
    if a.archived_s:
        n += H.add_archived(a.archived_s, "S", "msls_v5_")
    if a.hits_dir:
        n += H.add_v6_dir(a.hits_dir)  # v6 arrays take precedence for identical keys
    if n == 0:
        raise SystemExit("no hit arrays found")
    print(f"[analysis] loaded {len(H.h)} hit arrays")

    groups = None
    if a.groups and a.groups.is_file():
        g = json.loads(a.groups.read_text())
        groups = {k: np.asarray(v, dtype=np.int64) for k, v in g.get("group_ids", {}).items() if v is not None}
    res = analyse(H, groups)
    if a.diagnostics and a.diagnostics.is_file():
        res["diagnostics"] = json.loads(a.diagnostics.read_text())
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2))
    os.replace(tmp, a.out_json)
    md = a.out_json.with_suffix(".md")
    md.write_text(to_markdown(res))
    print(f"[analysis] wrote {a.out_json} and {md}")


if __name__ == "__main__":
    main()
