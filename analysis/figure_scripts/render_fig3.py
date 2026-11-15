import sys, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = sys.argv[1]; out = sys.argv[2]
spec = importlib.util.spec_from_file_location("euot_impl", src + "/euot_vpr_implementation.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
gen = m.DisjointRGBPlaceBenchmarkGenerator(d_model=64, M_clusters=64, seed=42)
pids = np.array([241, 243])
clean, _ = gen.render_place_rgb_batch(pids, view_seed=701, occ_frac=0.0, is_database=True)
q2, _ = gen.render_place_rgb_batch(pids, view_seed=703, occ_frac=0.25, is_database=False)
q4, _ = gen.render_place_rgb_batch(pids, view_seed=704, occ_frac=0.55, is_database=False)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.5))
cols = ["Gallery view (clean)", "Query, 25% occluded", "Query, 55% occluded"]
for r, pid in enumerate(pids):
    for c, img in enumerate([clean[r], q2[r], q4[r]]):
        ax = axes[r, c]; ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.6); s.set_color("#52514e")
        if r == 0: ax.set_title(cols[c], fontsize=9.5, color="#0b0b0b")
    axes[r, 0].set_ylabel(f"Test place {pid}", fontsize=9.5, color="#0b0b0b")
fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("ok")
