"""Recompute the same-head disrupted-cluster energy diagnostic on MSLS-val."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "msls_crossed_and_gates_v6"
sys.path.insert(0, str(ARCHIVE / "kit"))
from euot_v6_analysis import SEEDS, grouped_test


def main():
    results = ARCHIVE / "results"
    metadata = json.loads((results / "msls_val_query_groups.json").read_text())
    sequence = np.asarray(metadata["group_ids"]["sequence"])
    output = {}
    with np.load(results / "diagnostics_arrays.npz", allow_pickle=False) as archive:
        for setting, label in (("U", "U_1/2"), ("S", "R_1/65")):
            per_seed_differences = []
            for seed in SEEDS:
                prefix = f"{setting}_uot_combined_s{seed}_occ50"
                equal_weight_share = archive[f"{prefix}_disrupted_cluster_fraction"]
                gated_share = archive[f"{prefix}_disrupted_weight_share"]
                per_seed_differences.append(gated_share - equal_weight_share)
            difference = np.mean(per_seed_differences, axis=0, dtype=np.float64)
            result = grouped_test(-100 * difference, sequence)
            result.pop("p")
            output[label] = result
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
