#!/usr/bin/env python3
"""Figure: MSLS-val crossed design (token normalization x global-token share) and gate decomposition.

Usage: python3 make_fig_msls_crossed.py <statistics.json> <out_prefix> [<released_salad_occ50_hits.npy>]
<statistics.json>: output of analysis/recompute_v6_statistics.py (analysis/v6_statistics_recomputed.json).
The released-SALAD line uses msls_setting_U/hit_arrays_v4/msls_official_pretrained_salad_occ50_hits.npy.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"          # global share 1/2 / 1/65
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "hatch.linewidth": 1.0, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42})
MINUS = "−"


def fm(v, d=2):
    return f"{v:+.{d}f}".replace("-", MINUS)


r = json.load(open(sys.argv[1]))
out = sys.argv[2]
ps = r["per_setting_factorial"]
SET = {  # key: (label, color, marker, filled, linestyle)
    "U": ("U$_{1/2}$: unit tokens, share 1/2", BLUE, "o", True, "-"),
    "RE": ("R$_{1/2}$: raw tokens, share 1/2", BLUE, "s", False, "--"),
    "UW": ("U$_{1/65}$: unit tokens, share 1/65", ORANGE, "o", True, "-"),
    "S": ("R$_{1/65}$: raw tokens, share 1/65 (SALAD)", ORANGE, "s", False, "--"),
}
fig, axs = plt.subplots(2, 2, figsize=(10.2, 6.9))
plt.subplots_adjust(wspace=0.34, hspace=0.48, left=0.115, right=0.985, top=0.885, bottom=0.08)

# (a) interaction vs mask fraction
a = axs[0, 0]
x = np.array([0, 25, 50])
for i, (k, (lab, col, mk, fill, ls)) in enumerate(SET.items()):
    d = [ps[k][o]["interaction_did"] for o in ("occ00", "occ25", "occ50")]
    m = np.array([t["diff_pp"] for t in d]); lo = np.array([t["ci95"][0] for t in d]); hi = np.array([t["ci95"][1] for t in d])
    off = (i - 1.5) * 1.3
    a.errorbar(x + off, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=6.5, lw=1.8, ls=ls, capsize=2.5,
               mfc=col if fill else "white", mec=col, mew=1.6, label=lab, zorder=3)
    a.text(x[-1] + off + 2.0, m[-1], fm(m[-1]), color=INK, fontsize=7.6, va="center")
a.axhline(0, color=INK2, lw=0.9, zorder=1)
a.set_xticks(x); a.set_xticklabels(["0%", "25%", "50%"]); a.set_xlim(-6, 62)
a.set_xlabel("Target mask fraction"); a.set_ylabel("Transport × gate interaction (pp)")
a.grid(axis="y", color=GRID, lw=0.7); a.set_axisbelow(True)
a.set_title("(a) Interaction (Recall@1) with 95% CI", fontsize=9, loc="left", color=INK)

# (b) interaction at 50% as a function of the global share (lines = token normalization)
b = axs[0, 1]
xs = np.array([0, 1])
for tok, keys, mk, fill, ls, name in (("unit", ("U", "UW"), "o", True, "-", "unit tokens"), ("raw", ("RE", "S"), "s", False, "--", "raw tokens")):
    d = [ps[k]["occ50"]["interaction_did"] for k in keys]
    m = np.array([t["diff_pp"] for t in d]); lo = np.array([t["ci95"][0] for t in d]); hi = np.array([t["ci95"][1] for t in d])
    off = -0.03 if tok == "unit" else 0.03
    b.plot(xs + off, m, color=INK2, lw=1.2, ls=ls, zorder=2, label=name)
    for j, (col) in enumerate((BLUE, ORANGE)):
        b.errorbar(xs[j] + off, m[j], yerr=[[m[j] - lo[j]], [hi[j] - m[j]]], color=col, marker=mk, ms=7, capsize=2.5, lw=1.6,
                   mfc=col if fill else "white", mec=col, mew=1.6, zorder=3)
        dy = (0.35 if tok == "unit" else -0.35) if j == 0 else 0.0
        b.text(xs[j] + off + (0.07 if j == 1 else -0.07), m[j] + dy, fm(m[j]), ha="left" if j == 1 else "right", va="center", fontsize=7.6, color=INK)
fa = r["familyA_crossed_design"]["occ50"]
b.text(-0.42, -6.8, f"share main effect {fm(fa['global_share_main_effect']['diff_pp'])} pp (Holm p={fa['global_share_main_effect']['p_holm_familyA']:.4f})\n"
                  f"normalization main effect {fm(fa['normalization_main_effect']['diff_pp'])} pp (Holm p={fa['normalization_main_effect']['p_holm_familyA']:.3f})",
       ha="left", va="bottom", fontsize=7.3, color=INK)
b.axhline(0, color=INK2, lw=0.9, zorder=1)
b.set_xticks(xs); b.set_xticklabels(["global share 1/2", "global share 1/65"]); b.set_xlim(-0.45, 1.45); b.set_ylim(-7.2, 6.0)
b.set_ylabel("Interaction at 50% masking (pp)")
b.grid(axis="y", color=GRID, lw=0.7); b.set_axisbelow(True)
b.legend(loc="upper right", fontsize=7.2, frameon=False)
b.set_title("(b) Interaction across global-token shares", fontsize=9, loc="left", color=INK)

# (c) gate decomposition at 50% in U_1/2 and R_1/65
c = axs[1, 0]
gates = [("combined", "combined"), ("length", "length"), ("mass", "mass"), ("meannorm", "mean norm"), ("reliability", "reliability")]
gv = r["familyB_gate_variants"]
y = np.arange(len(gates))[::-1]
for k, col, mk, fill, off in (("U", BLUE, "o", True, 0.14), ("S", ORANGE, "s", False, -0.14)):
    for yi, (g, _) in zip(y, gates):
        t = gv[k]["occ50"][g]["interaction_did"]
        c.errorbar(t["diff_pp"], yi + off, xerr=[[t["diff_pp"] - t["ci95"][0]], [t["ci95"][1] - t["diff_pp"]]], color=col, marker=mk, ms=6.5,
                   capsize=2.5, lw=1.6, mfc=col if fill else "white", mec=col, mew=1.6, zorder=3,
                   label=("U$_{1/2}$" if k == "U" else "R$_{1/65}$ (SALAD)") if g == "combined" else None)
        star = "*" if t.get("p_holm_familyB", 1) < 0.05 else ""
        c.text(t["ci95"][1] + 0.25, yi + off, fm(t["diff_pp"]) + star, va="center", fontsize=7.3, color=INK)
c.axvline(0, color=INK2, lw=0.9, zorder=1)
c.set_yticks(y); c.set_yticklabels([lab for _, lab in gates], fontsize=7.8)
c.set_xlabel("Transport × gate interaction at 50% masking (pp)"); c.set_xlim(-7.5, 7.8)
c.grid(axis="x", color=GRID, lw=0.7); c.set_axisbelow(True)
c.legend(loc="lower right", fontsize=7.2, frameon=False)
c.set_title("(c) Gate decomposition (* Holm p < 0.05)", fontsize=9, loc="left", color=INK)

# (d) Recall@1 at 50% of Bal and UOT+G in the four settings, with released SALAD
d = axs[1, 1]
arms = r["arm_recall_pct"]
order = ["U", "RE", "UW", "S"]
names = ["U$_{1/2}$", "R$_{1/2}$", "U$_{1/65}$", "R$_{1/65}$"]
w = 0.36
for i, k in enumerate(order):
    col = BLUE if k in ("U", "RE") else ORANGE
    for j, (arm, gated) in enumerate((("bal_none", False), ("uot_combined", True))):
        v = arms[f"{k}_{arm}"]["occ50"]["R@1"]["mean"]
        xx = i + (j - 0.5) * (w + 0.03)
        if gated:
            d.bar(xx, v, width=w, color=col, edgecolor=col, zorder=2, label=("UOT+G" if i == 0 else None))
        else:
            d.bar(xx, v, width=w, color="white", edgecolor=col, hatch="////", lw=1.2, zorder=2, label=("Bal" if i == 0 else None))
        d.text(xx, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=7.3, color=INK)
sal = r.get("_salad_r1_occ50", None)
if sal is None and len(sys.argv) > 3:
    sal = round(float(np.load(sys.argv[3])[:, 0].mean() * 100), 2)
if sal is not None:
    d.axhline(sal, color=INK2, ls="--", lw=1.1, zorder=1)
    d.text(-0.45, sal + 1.0, f"released SALAD {sal:.2f}", ha="left", va="bottom", fontsize=7.4, color=INK)
d.set_xticks(range(4)); d.set_xticklabels(names); d.set_ylim(0, 90)
d.set_ylabel("Recall@1 at 50% masking (%)")
d.grid(axis="y", color=GRID, lw=0.7); d.set_axisbelow(True)
d.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=7.2, frameon=False, ncol=2)
d.set_title("(d) Recall@1 across descriptor settings", fontsize=9, loc="left", color=INK)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.6, frameon=False, bbox_to_anchor=(0.53, 0.995))
for ax in axs.ravel():
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
print("wrote", out)
