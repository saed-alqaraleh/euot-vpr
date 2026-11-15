# Image-level MSLS-val replay — inputs and scope

The CPU checks reproduce statistics **from archived retrieval hits**; they are not fresh image inference. Exact image-level replay needs external, licensed assets and the historical evaluation ordering:

1. **MSLS-val**: authorized `train_val` MSLS archive, exactly **18,871 gallery images** and **740 official queries** for Copenhagen and San Francisco, with the **25 m** positive criterion; see https://github.com/mapillary/mapillary_sls. The original recorded archive was 819,435,520 bytes, SHA-256 `6912c67bc1f65abdd2b88e647eb47a8360d8246e3165a003add6cdfe9985f02c` (this is a provenance check, not a download link).
2. **Exact `msls_manifest.json` (NOT included in the upload)**. JSON object with `database` and `queries` arrays of image paths (legacy runner also supports `db_abs_paths`, `q_abs_paths`) plus `positives`: a length-740 list of lists containing **zero-based indices into the ordered 18,871-image gallery**. Preserve both ordered path lists and gallery-index positions, not just the set of files. Paths may be mapped to local authorized roots **without changing item order**. Dataset-derived `seq_info.csv` and the original positive-sharing relation are required to repeat group-level results, though the archived grouping arrays suffice for CPU verification.
3. **Gardens Point Walking**: original ordered day-left/night-right image pair list, 200 lexicographically frame-aligned pairs: first 150 to fit the heads and last 50 in-sequence development pairs; obtain from https://open.qcr.ai/dataset/day-night-lateral/ under provider terms. The numeric latent cache from the original training is not supplied; reproduce it using the original feature-extraction code and source images.
4. **Frozen pretrained SALAD DINOv2 trunk and projection**: upstream model from https://github.com/serizba/salad (typically `torch.hub.load("serizba/salad", "dinov2_salad")`), with any upstream dependencies and model-version settings. **Not** supplied. The 13+12+72 included `.pt` artifacts are study-fitted heads/controls, not replacement SALAD checkpoints.
5. **Runtime**: NVIDIA CUDA GPU for full-gallery latent extraction, PyTorch ≥ 2.1, torchvision, NumPy, Pillow, scikit-learn, and upstream SALAD requirements (PyTorch Lightning, pytorch-metric-learning, and CUDA-aware components as required). Original v4 evidence specifies Python ≥3.10, PyTorch ≥2.1; no complete pip/conda lock is archived for the v6 GPU job. See `docs/DEPENDENCIES.md` for measured versus recommended versions.

### Evaluation protocol

Resize input to **322×322**. Frozen backbone yields 529 local projected tokens of 128 dimensions, 65 assignment logits per token (64 clusters + dustbin) and a 256-dim global token. Targets are 0%, 25%, and 50% patch-grid masking on a 23×23 grid with deterministic query-index-based RNG and up to five rectangular occluders plus additional patch sampling; the gallery is not occluded. Preserve the original seed, query index, and index offset. Per-seed head training: 12 epochs, AdamW, same recorded heads and hyperparameters across matched arms. Preserve all original head checkpoints and the `U`/`S` original-hit selection documented in README.

### Executable study entrypoints

- `msls_setting_U/EUOT_VPR_MSLS_Matched_Backbone_Final_Diagnostic.py`: original v4.1 head and inference with sanitized CLI defaults.
- `msls_setting_S/EUOT_VPR_MSLS_v5_Retrain_RawTokens_1to64CLS.py`: original v5.1 numerical implementation; release docstring sanitized. It requires the release-vetted v4.1 script in the proper `--v4_script` path.
- `msls_crossed_and_gates_v6/kit/euot_v6_kit.py`: crossed settings `UW` and `RE`, gate decomposition and diagnostics. It imports two local copies of the audited runners (the v5 header is sanitized and its expected digest updated); use explicit arguments, `--help`, and `EUOT_WORKSPACE` rather than personal cloud paths.

**Not asserted:** replay cannot be certified bit-for-bit until the original ordered manifest, dataset versions, SALAD upstream revision/checkpoint and full runtime lock are supplied. Do not claim a reproducible end-to-end fresh GPU experiment from the numerical archive alone.
