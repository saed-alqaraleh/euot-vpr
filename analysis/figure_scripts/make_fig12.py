import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root, outdir = sys.argv[1], sys.argv[2]
A = json.load(open(root + "/all_obtained_results.json"))
F = json.load(open(root + "/factorial_results.json"))
BLUE, ORANGE, AQUA, VIOLET, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "hatch.linewidth": 1.0})
def clean_ax(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---------------- Figure 1: controlled benchmark, held-out test places ----------------
M = A["stage3_disjoint_heldout_pixel_occlusion_benchmark"]["data"]["modes"]
levels = ["Clean_0pct_R@1", "Q1_10pct_R@1", "Q2_25pct_R@1", "Q3_40pct_R@1", "Q4_55pct_R@1_Mean"]
xl = ["Clean\n(0%)", "Q1\n(10%)", "Q2\n(25%)", "Q3\n(40%)", "Q4\n(55%)"]
series = [  # key, label, color, ls, marker, filled, lw
    ("salad_style", "SALAD-style", INK2, (0, (5, 2, 1, 2)), "s", True, 1.5),
    ("a2gc_style", "A2GC-style", AQUA, ":", "D", True, 1.6),
    ("folpp_style", "FoL++-style (top-20)", VIOLET, (0, (6, 2)), "P", True, 1.6),
    ("rgtmp_bal", "Bal", ORANGE, "--", "o", False, 1.8),
    ("rgtmp_bal_gate", "Bal+G", ORANGE, "-", "o", True, 1.8),
    ("euot_no_gate", "UOT", BLUE, "--", "^", False, 1.8),
    ("euot_1stage", "UOT+G (EUOT-VPR)", BLUE, "-", "^", True, 2.2),
    ("euot_adaptive_2stage", "EUOT-VPR-2S", BLUE, "-", "*", True, 2.2),
]
fig = plt.figure(figsize=(11.0, 3.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); c = fig.add_subplot(gs[2])
x = np.arange(5)
for k, lab, col, ls, mk, filled, lw in series:
    y = [M[k][l] for l in levels]
    ms = 11 if mk == "*" else 6.5
    a.plot(x, y, color=col, ls=ls, lw=lw, marker=mk, ms=ms, mfc=col if filled else "white", mec=col, mew=1.3, label=lab)
a.set_xticks(x); a.set_xticklabels(xl); a.set_ylabel("Recall@1 (%)"); a.set_ylim(40, 102); clean_ax(a)
a.text(4.12, M["euot_adaptive_2stage"][levels[4]], "EUOT-VPR-2S\n91.25%", fontsize=7.8, color=INK, va="center")
a.text(4.12, M["euot_1stage"][levels[4]], "UOT+G\n76.67%", fontsize=7.8, color=INK, va="center")
a.set_xlim(-0.25, 4.95)
a.legend(loc="lower left", fontsize=7.2, frameon=False, ncol=1, handlelength=2.6)
a.set_title("(a) Recall@1 by occlusion level", fontsize=9.5, loc="left", color=INK)
bars = [("salad_style", "SALAD-\nstyle", INK2, True), ("a2gc_style", "A2GC-\nstyle", AQUA, True),
        ("folpp_style", "FoL++-\nstyle", VIOLET, True), ("rgtmp_bal", "Bal", ORANGE, False),
        ("rgtmp_bal_gate", "Bal+G", ORANGE, True), ("euot_no_gate", "UOT", BLUE, False),
        ("euot_1stage", "UOT+G (EUOT-VPR)", BLUE, True), ("euot_adaptive_2stage", "EUOT-VPR-2S", BLUE, True)]
for ax, key, ttl, fmt in [(b, "AURCC_Joint", "(b) Selective risk, AURCC (lower is better)", "{:.3f}"),
                          (c, "Conformal_SetSize90_Mean", "(c) Mean 90% conformal set size (lower is better)", "{:.2f}")]:
    for i, (k, lab, col, filled) in enumerate(bars):
        v = M[k][key]
        if filled: ax.bar(i, v, width=0.72, color=col, edgecolor=col, zorder=2)
        else: ax.bar(i, v, width=0.72, color="white", edgecolor=col, hatch="////", lw=1.2, zorder=2)
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([t[1].replace("-\n", "-").replace("\n", " ") for t in bars], fontsize=7.4, rotation=40, ha="right", rotation_mode="anchor")
    clean_ax(ax); ax.set_title(ttl, fontsize=9.5, loc="left", color=INK)
b.set_ylim(0, 0.056); c.set_ylim(0, 6.6)
b.set_ylabel("AURCC"); c.set_ylabel("Candidates per query")
fig.savefig(outdir + "/fig1_controlled_benchmark.pdf", bbox_inches="tight")
fig.savefig(outdir + "/fig1_controlled_benchmark.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: sensitivity to rho_2 and factorial interaction plot ----------------
S = A["stage5_hyperparameter_sensitivity_grid"]["data"]["tau2_sensitivity"]
rho2 = [r["rho_2"] for r in S]; q3 = [r["Q3_40pct_R@1"] for r in S]; q4 = [r["Q4_55pct_R@1"] for r in S]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1, 1.05]})
a.plot(rho2, q3, color=INK2, ls="--", lw=1.8, marker="s", ms=6, label="Q3 (40% occlusion)")
a.plot(rho2, q4, color=BLUE, ls="-", lw=2.2, marker="o", ms=6.5, label="Q4 (55% occlusion)")
a.axvline(0.431, color=INK2, lw=0.9, ls=":")
a.text(0.44, 101.2, "learned $\\hat\\rho_2=0.43$", fontsize=8, color=INK, va="top")
a.set_xlabel("Column contraction $\\rho_2=\\tau_2/(\\tau_2+\\epsilon)$  ($\\rho_2\\to1$: balanced columns)")
a.set_ylabel("Validation Recall@1 (%)"); a.set_ylim(60, 102); a.set_xlim(0.18, 1.02); clean_ax(a)
a.legend(loc="lower left", fontsize=8, frameon=False)
a.set_title("(a) Tightening the column constraint (validation places)", fontsize=9.5, loc="left", color=INK)
PA = F["protocol_a_fixed_checkpoint_2x2_intervention"]; PB = F["protocol_b_independently_trained_2x2_factorial"]
cells = {"A": (PA, ["rgtmp_bal", "rgtmp_bal_gate", "euot_no_gate", "euot_1stage"]),
         "B": (PB, ["rgtmp_bal_indep", "rgtmp_bal_gate_indep", "euot_no_gate_indep", "euot_1stage"])}
style = {"A": ("-", "o"), "B": ((0, (6, 2, 1.5, 2)), "s")}
for prot, (P, ks) in cells.items():
    ls, mk = style[prot]
    for (k0, k1), col, nm in [((ks[0], ks[1]), ORANGE, "balanced"), ((ks[2], ks[3]), BLUE, "relaxed")]:
        for s0, s1 in zip(P[k0]["per_seed"], P[k1]["per_seed"]):
            b.plot([0, 1], [s0["Q4_R@1"], s1["Q4_R@1"]], color=col, lw=0.8, alpha=0.28, ls=ls, zorder=1)
        b.plot([0, 1], [P[k0]["Q4_55pct_R@1_Mean"], P[k1]["Q4_55pct_R@1_Mean"]], color=col, lw=2.2, ls=ls,
               marker=mk, ms=7, zorder=3)
b.set_xticks([0, 1]); b.set_xticklabels(["No gate", "With support gate"]); b.set_xlim(-0.25, 1.62)
b.set_ylabel("Q4 Recall@1 (%), held-out places"); b.set_ylim(44, 88); clean_ax(b)
b.text(1.06, 76.67, "relaxed columns\n(UOT+G 76.67%)", fontsize=8, color=INK, va="center")
b.text(1.06, 57.3, "balanced columns\n(A 57.08%, B 57.50%)", fontsize=8, color=INK, va="center")
b.text(0.5, 85.5, "Interaction (DiD): A +20.83 pp, B +20.42 pp (both p = 0.0001)", fontsize=8, color=INK, ha="center")
handles = [Line2D([], [], color=ORANGE, lw=2.2, label="Balanced columns ($\\rho_2=1$)"),
           Line2D([], [], color=BLUE, lw=2.2, label="Relaxed columns ($\\rho_2<1$)"),
           Line2D([], [], color=INK2, lw=1.8, ls="-", marker="o", label="Protocol A (shared checkpoint)"),
           Line2D([], [], color=INK2, lw=1.8, ls=(0, (6, 2, 1.5, 2)), marker="s", label="Protocol B (independent training)"),
           Line2D([], [], color=INK2, lw=0.8, alpha=0.4, label="Individual seeds")]
b.legend(handles=handles, loc="upper left", fontsize=7.4, frameon=False, bbox_to_anchor=(0.0, 0.93))
b.set_title("(b) $2\\times2$ factorial at 55% occlusion", fontsize=9.5, loc="left", color=INK)
fig.tight_layout(w_pad=2.2)
fig.savefig(outdir + "/fig2_relaxation_and_factorial.pdf", bbox_inches="tight")
fig.savefig(outdir + "/fig2_relaxation_and_factorial.png", dpi=200, bbox_inches="tight")
print("rho2", rho2, "q4", q4)
