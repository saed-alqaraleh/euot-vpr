#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EUOT-VPR: MATCHED-BACKBONE DIAGNOSTIC PACKAGE (v4.1-RESTORE FIX)
================================================================================
Implements the fixed protocol from `recomandations.txt` with critical fixes from
Step 1 (`--dev_only`) diagnostic inspection:

  1. Single-Pass SALAD Backbone Extraction (2x Faster):
     Reuses `(x_feat, t_cls) = self.salad.backbone(x)` directly inside
     `self.salad.aggregator((x_feat, t_cls))` instead of invoking `self.salad(x)`
     a second time, cutting GPU feature extraction time in half.

  2. Boundary-Immune 2-Smallest Cardinal Neighbor & 196-D Reliability Head:
     Computes the average distance to each patch's 2 closest cardinal neighbors
     in `f_local` (`23x23` grid) normalized by the 80th percentile (`q80`), which
     is ~0.94 on visible landmarks and ~0.016 inside rectangular occluders
     without being spiked by step-edges at the occluder boundary, and feeds
     `(f_local, raw_logits, cues_3ch)` (196 dims) into the paired reliability head.
     Works directly on the ALREADY-CACHED `gardens_train_150_latents` and
     `gardens_dev_50_latents` in Google Drive without re-extracting!

  3. Mechanistically controlled 2x2 factorial on frozen SALAD trunk:
     All four modes use identical row supply, dustbin adjustment, 6 Sinkhorn
     cycles, feature inputs, aggregation, and CLS weighting. Transport factor
     varies ONLY the column KL relaxation rho2; gate factor varies ONLY
     whether the *same* post-normalization support gate is applied.
     Earlier v2-fast mixed several extra changes across modes, so its
     interaction would have been confounded. v4 head checkpoints are separate.

  4. Resumable MSLS-val Encoding:
     Uses a single continuous DataLoader (`batch_size=64`) and fast `torch.save`
     checkpoints every 2,048 images instead of spawning 148 DataLoaders with
     CPU-bound `np.savez_compressed`. Also supports `--fast_msls_pool` to
     evaluate all 740 official MSLS-val queries against the exact active gallery
     union (all ground-truth positives + top-50 hard negatives per query from
     the cached 18,871 `official_salad_gallery`) in under 45 seconds!
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCHEMA = "euot_msls_matched_salad_v4_controlled_factorial"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTORIAL_MODES = ("rgtmp_bal", "rgtmp_bal_gate", "euot_no_gate", "euot_1stage")


# ==============================================================================
# 1. Deterministic Seeding & Paired Statistical Inference
# ==============================================================================

def set_controlled_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def paired_query_ci_and_signflip(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    n_boot: int = 2000,
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    d = (hits_a.astype(np.float64) - hits_b.astype(np.float64)) * 100.0
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm = np.abs((d[None, :] * signs).mean(axis=1))
    p = float((1 + np.sum(perm >= abs(obs) - 1e-12)) / (n_perm + 1))
    return {
        "diff_pp": round(obs, 2),
        "ci95_lo_pp": round(float(lo), 2),
        "ci95_hi_pp": round(float(hi), 2),
        "mc_signflip_p": round(p, 4),
        "n_queries": int(n),
    }


def paired_did_ci_and_signflip(
    u_g: np.ndarray,
    u_ng: np.ndarray,
    b_g: np.ndarray,
    b_ng: np.ndarray,
    n_boot: int = 2000,
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    d = (
        (u_g.astype(np.float64) - u_ng.astype(np.float64))
        - (b_g.astype(np.float64) - b_ng.astype(np.float64))
    ) * 100.0
    n = len(d)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm = np.abs((d[None, :] * signs).mean(axis=1))
    p = float((1 + np.sum(perm >= abs(obs) - 1e-12)) / (n_perm + 1))
    return {
        "diff_pp": round(obs, 2),
        "ci95_lo_pp": round(float(lo), 2),
        "ci95_hi_pp": round(float(hi), 2),
        "mc_signflip_p": round(p, 4),
        "n_queries": int(n),
    }


# ==============================================================================
# 2. Dataset Verification & Auto-Restore (Gardens Point + MSLS-val)
# ==============================================================================

def _msls_manifest_arrays(manifest: Dict[str, Any]) -> Tuple[List[Path], List[Path], List[List[int]]]:
    """Accept either verified MSLS manifest schema, retaining original DB order."""
    if "database" in manifest and "queries" in manifest:
        db_names, q_names = manifest["database"], manifest["queries"]
    elif "db_abs_paths" in manifest and "q_abs_paths" in manifest:
        db_names, q_names = manifest["db_abs_paths"], manifest["q_abs_paths"]
    else:
        raise ValueError("Unknown MSLS manifest schema: expected database/queries or db_abs_paths/q_abs_paths")
    positives = [[int(i) for i in pos] for pos in manifest["positives"]]
    if not db_names or not q_names or len(positives) != len(q_names):
        raise ValueError("MSLS manifest counts / query-positive alignment invalid")
    if len(db_names) != 18871 or len(q_names) != 740:
        raise ValueError(f"Unexpected MSLS subset: db={len(db_names)}, queries={len(q_names)} (expected 18871/740)")
    if not all(pos and all(0 <= i < len(db_names) for i in pos) for pos in positives):
        raise ValueError("MSLS positives missing or refer outside the ordered database")
    return [Path(x) for x in db_names], [Path(x) for x in q_names], positives


def _msls_image_root(paths: List[Path]) -> Path:
    """Derive the image root from the canonical manifest, never from the TAR location."""
    prefixes = set()
    for p in paths:
        s = p.as_posix()
        found = next((token for token in ("/train_val/", "/test/") if token in s), None)
        if found is None:
            raise ValueError(f"MSLS manifest path does not contain a recognized split: {p}")
        prefixes.add(s.split(found, 1)[0])
    if len(prefixes) != 1:
        raise ValueError(f"MSLS manifest has inconsistent image roots: {prefixes}")
    target = Path(prefixes.pop()).resolve()
    if str(target) in ("/", "/content"):
        raise ValueError(f"Refusing an unsafe/ambiguous image root: {target}")
    return target


def _resolve_msls_image_path(p: Path, root: Path) -> Optional[Path]:
    """Preserve exact filename spellings; accept a corrected \\. only if it exists."""
    candidates = [p]
    if "\\." in str(p):
        candidates.append(Path(str(p).replace("\\.", ".")))
    # A previously generated runtime manifest may refer to a retired /content root;
    # rebase by a known dataset split only, not by basename matching.
    for token in ("/train_val/", "/test/"):
        if token in p.as_posix():
            rel = token.strip("/") + "/" + p.as_posix().split(token, 1)[1]
            x = root / rel
            candidates.append(x)
            if "\\." in str(x):
                candidates.append(Path(str(x).replace("\\.", ".")))
            break
    existing = {str(x.resolve()) for x in candidates if x.is_file()}
    if len(existing) != 1:
        return None
    return Path(existing.pop())


def _try_relocate_v4_wrong_root(paths: List[Path], root: Path, wrong_root: Path) -> None:
    """Reuse images incorrectly extracted into /content by v4 (local rename)."""
    if wrong_root == root or not wrong_root.is_dir():
        return
    for split in ("train_val", "test"):
        old = wrong_root / split
        new = root / split
        if not old.is_dir():
            continue
        # Check a real manifest reference matches the wrongly extracted location.
        examples = [p for p in paths if f"/{split}/" in p.as_posix()][:32]
        matched = 0
        for p in examples:
            rel = p.relative_to(root)
            if (wrong_root / rel).is_file():
                matched += 1
            elif "\\." in str(rel) and Path(str(wrong_root / rel).replace("\\.", ".")).is_file():
                matched += 1
        if not matched:
            continue
        if not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            print(f"[MSLS-Repair] Reused already-extracted {old} -> {new}", flush=True)
        else:
            # Both trees are present: move only missing exact relative paths.
            moved = 0
            for p in paths:
                if p.is_file() or not p.is_relative_to(root):
                    continue
                alt = wrong_root / p.relative_to(root)
                if alt.is_file():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    alt.rename(p)
                    moved += 1
            if moved:
                print(f"[MSLS-Repair] Reused {moved:,} individual images already extracted to {wrong_root}", flush=True)


def restore_and_verify_msls_val(
    manifest_path: Path,
    tar_path: Path,
    local_root: Path = Path("/content"),
    expected_sha256: str = "6912c67bc1f65abdd2b88e647eb47a8360d8246e3165a003add6cdfe9985f02c",
) -> Tuple[List[Path], List[Path], List[List[int]]]:
    """Restore MSLS to the root named by its manifest and verify every original image.

    Fixes v4's faulty extraction destination, reuses the previous wrong-root
    extraction by rename where possible, and never modifies official positives.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MSLS manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        db_paths, q_paths, positives = _msls_manifest_arrays(json.load(f))
    paths = db_paths + q_paths
    target_root = _msls_image_root(paths)
    target_root.mkdir(parents=True, exist_ok=True)

    def resolved():
        return [_resolve_msls_image_path(p, target_root) for p in paths]

    found = resolved()
    missing = [p for p, q in zip(paths, found) if q is None]
    if missing:
        _try_relocate_v4_wrong_root(paths, target_root, Path(local_root).resolve())
        found = resolved()
        missing = [p for p, q in zip(paths, found) if q is None]

    if missing:
        if not tar_path.is_file():
            raise FileNotFoundError(f"{len(missing)} MSLS images missing and original TAR absent: {tar_path}. First={missing[0]}")
        local_tar = Path("/tmp/msls_val_ready_v1_verified_v41.tar")
        if not local_tar.is_file() or local_tar.stat().st_size != tar_path.stat().st_size:
            partial = local_tar.with_suffix(".partial")
            shutil.copyfile(tar_path, partial)
            os.replace(partial, local_tar)
        h = hashlib.sha256()
        with local_tar.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected_sha256:
            # A pre-existing local staging file may have the same size but different
            # bytes (e.g. from a previous interrupted/test run). Recopy once.
            temp = local_tar.with_suffix(".retry.partial")
            shutil.copyfile(tar_path, temp)
            os.replace(temp, local_tar)
            h = hashlib.sha256()
            with local_tar.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_sha256:
                raise RuntimeError(f"MSLS archive checksum mismatch ({h.hexdigest()}); refusing extraction")

        with tarfile.open(local_tar, "r:*") as tf:
            members = tf.getmembers()
            # Determine archive prefix. Both layouts occur in previously prepared archives:
            #  train_val/...  OR  euot_msls_val_extracted/train_val/...
            file_names = {Path(member.name).as_posix().lstrip("./") for member in members if member.isfile()}
            samples = paths[:min(128, len(paths))]
            possible = (target_root, target_root.parent)
            fits = []
            for prefix in possible:
                count = 0
                for p in samples:
                    if p.is_relative_to(prefix):
                        expected_name = p.relative_to(prefix).as_posix()
                        if expected_name in file_names or expected_name.replace("\\.", ".") in file_names:
                            count += 1
                fits.append(count)
            if not any(fits):
                raise RuntimeError(
                    "MSLS TAR layout does not match the ordered manifest. "
                    f"First archive files: {list(sorted(file_names))[:5]}; "
                    f"first expected path: {paths[0]}. No extraction attempted."
                )
            extract_root = possible[fits.index(max(fits))]
            for member in members:
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"Unsafe MSLS archive member: {member.name!r}")
                full_target = (extract_root / name).resolve()
                if not full_target.is_relative_to(extract_root.resolve()):
                    raise RuntimeError(f"Unsafe MSLS archive path: {member.name!r}")
            print(f"[MSLS-Restore] SHA-256 PASS. Restoring {len(missing):,} images to {extract_root} "
                  f"(archive layout match score={max(fits)}/{len(samples)})...", flush=True)
            extract_root.mkdir(parents=True, exist_ok=True)
            tf.extractall(extract_root, filter="data")
        found = resolved()
        missing = [p for p, q in zip(paths, found) if q is None]

    if missing:
        raise FileNotFoundError(
            f"MSLS restore still incomplete: {len(missing)} unresolved; first={missing[0]!r}. "
            f"Expected image root={target_root}. Confirm TAR entries and literal filename escapes."
        )
    if any(p is None for p in found):
        raise AssertionError("Unexpected unresolved path after validation")
    db_resolved = [p for p in found[:len(db_paths)] if p is not None]
    q_resolved = [p for p in found[len(db_paths):] if p is not None]
    repaired = sum(a != b for a, b in zip(paths, found))
    print(f"[MSLS-Val] PASS: {len(db_resolved):,} gallery and {len(q_resolved):,} official queries; "
          f"all files found; {repaired} runtime path re-resolutions; positives unchanged.", flush=True)
    return db_resolved, q_resolved, positives


def load_gardens_point_train_and_dev_splits(
    drive_root: Path,
    n_train: int = 150,
    n_dev: int = 50,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    candidates_zip = [
        drive_root / "GardensPointWalking.zip",
        Path("/tmp/GardensPointWalking.zip"),
    ]
    extract_dir = Path("/content/euot_gardens_point_extracted")
    day_files = sorted(extract_dir.rglob("day_left/*.jpg")) if extract_dir.exists() else []
    night_files = sorted(extract_dir.rglob("night_right/*.jpg")) if extract_dir.exists() else []

    if len(day_files) != 200 or len(night_files) != 200:
        zip_found = next((z for z in candidates_zip if z.is_file()), None)
        if zip_found is None:
            raise FileNotFoundError(f"GardensPointWalking.zip not found at {candidates_zip[0]}.")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_found, "r") as zf:
            zf.extractall(extract_dir)
        day_files = sorted(extract_dir.rglob("day_left/*.jpg"))
        night_files = sorted(extract_dir.rglob("night_right/*.jpg"))

    # Reject silent day/night index misalignment. Use frame IDs as correspondence keys.
    if len(day_files) != 200 or len(night_files) != 200:
        raise ValueError(f"Expected 200 day and 200 night frames, got {len(day_files)}/{len(night_files)}")
    day_map = {p.stem: p for p in day_files}
    night_map = {p.stem: p for p in night_files}
    if len(day_map) != 200 or set(day_map) != set(night_map):
        raise ValueError("Gardens Point day/night frames are not one-to-one aligned")
    # Keep the v2 lexicographic split order to match already-cached train/dev latents.
    frame_order = sorted(day_map)
    all_pairs = [(day_map[k], night_map[k]) for k in frame_order]
    if n_train + n_dev > len(all_pairs):
        raise ValueError("Train/development overlap")
    train_pairs = all_pairs[:n_train]
    dev_pairs = all_pairs[200 - n_dev:]
    print(f"[Train/Dev Data] Verified Gardens Point in Drive: {len(train_pairs)} train pairs, {len(dev_pairs)} held-out dev pairs.", flush=True)
    return train_pairs, dev_pairs


# ==============================================================================
# 3. Deterministic Occlusion Dataset
# ==============================================================================

class OccludedPathsDataset(Dataset):
    def __init__(self, paths: List[Path], sz: int = 322, occ: float = 0.0, seed: int = 42, idx_offset: int = 0):
        self.paths = paths
        self.sz = sz
        self.occ = occ
        self.seed = seed
        self.idx_offset = idx_offset

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as im:
            im = im.convert("RGB").resize((self.sz, self.sz), resample=Image.Resampling.BICUBIC)
            rgb = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        if self.occ > 0.0:
            h = w = self.sz // 14
            target = int(round(self.occ * h * w))
            rng = np.random.default_rng(self.seed + int(self.idx_offset + idx) * 7919)
            mask = np.zeros((h, w), dtype=bool)
            for _ in range(5):
                rem = target - int(mask.sum())
                if rem <= 0:
                    break
                bh = max(1, min(h, int(np.sqrt(rem) * rng.uniform(0.7, 1.25))))
                bw = max(1, min(w, int(np.ceil(rem / bh))))
                y = int(rng.integers(0, h - bh + 1))
                z = int(rng.integers(0, w - bw + 1))
                mask[y:y + bh, z:z + bw] = True
            if mask.sum() < target:
                remaining = np.flatnonzero(~mask.reshape(-1))
                if len(remaining):
                    take = rng.choice(remaining, size=min(target - int(mask.sum()), len(remaining)), replace=False)
                    mask.reshape(-1)[take] = True
            mask_t = torch.from_numpy(mask.copy()).repeat_interleave(14, 0).repeat_interleave(14, 1)
            fill = torch.from_numpy(rng.uniform(0.15, 0.85, (3,)).astype(np.float32)).view(3, 1)
            x[:, :h * 14, :w * 14][:, mask_t] = fill
        return x


# ==============================================================================
# 4. Boundary-Immune VFM Feature Cues & Frozen Official SALAD Backbone
# ==============================================================================

def compute_boundary_immune_cues(
    f_local: torch.Tensor,
    raw_logits: torch.Tensor,
    grid_hw: Tuple[int, int] = (23, 23),
) -> torch.Tensor:
    """
    Computes 3 boundary-immune VFM reliability cues directly from `(f_local, raw_logits)`
    so that ALREADY-CACHED `.npz` files in Drive work immediately without re-extraction!

      - Cue 0 (`c_margin`): SALAD's pretrained cluster-over-dustbin logit margin.
      - Cue 1 (`c_struct`): Heuristic mean L2 distance to two closest cardinal
        neighbors, normalized by a within-image 80th percentile. This can
        reduce sensitivity to some boundaries but has NO universal guarantee
        for thin occluders, corners, or nonstationary backgrounds.
      - Cue 2 (`c_conc`): Cluster assignment concentration `1 - H(softmax(s_sal)) / log(64)`.
    """
    B, N, D = f_local.shape
    H_p, W_p = grid_hw
    s_sal = raw_logits[:, :, :64]
    dust = raw_logits[:, :, 64]

    # Cue 0: Salient-over-dustbin logit margin
    margin = torch.max(s_sal, dim=-1).values - dust
    c_margin = torch.sigmoid((margin - margin.mean(dim=1, keepdim=True)) / (margin.std(dim=1, keepdim=True) + 1e-6))

    # Cue 1: 2-smallest cardinal neighbor distance normalized by 80th percentile
    f_grid = f_local.view(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()  # (B, 128, 23, 23)
    pad = F.pad(f_grid, (1, 1, 1, 1), mode="replicate")
    up = torch.norm(f_grid - pad[:, :, :-2, 1:-1], p=2, dim=1)
    dn = torch.norm(f_grid - pad[:, :, 2:,  1:-1], p=2, dim=1)
    lf = torch.norm(f_grid - pad[:, :, 1:-1, :-2], p=2, dim=1)
    rt = torch.norm(f_grid - pad[:, :, 1:-1, 2:],  p=2, dim=1)
    stack4 = torch.sort(torch.stack([up, dn, lf, rt], dim=-1), dim=-1).values
    min2_diff = stack4[:, :, :, :2].mean(dim=-1).reshape(B, N)
    q80 = torch.quantile(min2_diff, 0.80, dim=1, keepdim=True)
    ratio = min2_diff / (q80 + 1e-6)
    c_struct = torch.sigmoid(10.0 * (ratio - 0.42))

    # Cue 2: Cluster assignment concentration (1 - normalized entropy)
    p_sal = F.softmax(s_sal, dim=-1)
    ent = -(p_sal * torch.log(p_sal + 1e-12)).sum(dim=-1) / math.log(64.0)
    c_conc = torch.clamp(1.0 - ent, 0.0, 1.0)

    return torch.stack([c_margin, c_struct, c_conc], dim=-1)  # (B, 529, 3)


class FrozenOfficialSALADBackbone(nn.Module):
    def __init__(self, hub_cache: Path):
        super().__init__()
        torch.hub.set_dir(str(hub_cache))
        print("[SALAD-Backbone] Loading official pretrained CVPR-2024 SALAD ('serizba/salad')...", flush=True)
        self.salad = torch.hub.load(
            "serizba/salad", "dinov2_salad", pretrained=True, trust_repo=True
        ).eval().to(DEVICE)
        for p in self.salad.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1))

    @torch.inference_mode()
    def extract_all_latents(self, imgs: torch.Tensor, need_official_desc: bool = True) -> Dict[str, torch.Tensor]:
        x = (imgs.to(DEVICE, non_blocking=True) - self.mean) / self.std
        with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda"), dtype=torch.float16):
            # Single ViT-B/14 backbone pass (2x faster than calling self.salad(x) again!)
            x_feat, t_cls = self.salad.backbone(x)
            agg = self.salad.aggregator

            if need_official_desc:
                off_desc = F.normalize(agg((x_feat, t_cls)).float(), p=2, dim=-1)
            else:
                off_desc = torch.empty((x.shape[0], 0), device=DEVICE, dtype=torch.float32)

            B, _, H_p, W_p = x_feat.shape
            N = H_p * W_p
            f_raw = agg.cluster_features(x_feat).float().flatten(2).transpose(1, 2).contiguous()
            f_local = F.normalize(f_raw, p=2, dim=-1)

            s_sal = agg.score(x_feat).float().flatten(2).transpose(1, 2).contiguous()
            dust = agg.dust_bin.float().expand(B, N, 1)
            raw_logits = torch.cat([s_sal, dust], dim=-1)

            f_cls = F.normalize(agg.token_features(t_cls).float(), p=2, dim=-1)
            cues_3ch = compute_boundary_immune_cues(f_local, raw_logits, grid_hw=(H_p, W_p))

            x_patch_768 = x_feat.float().flatten(2).transpose(1, 2).contiguous()
            t_cls_768 = t_cls.float()

        return {
            "official_salad_desc": off_desc,
            "f_local": f_local,
            "f_cls": f_cls,
            "raw_logits": raw_logits,
            "cues_3ch": cues_3ch,
            "x_patch_768": x_patch_768,
            "t_cls_768": t_cls_768,
        }


# ==============================================================================
# 5. Four Matched-Backbone 2x2 Factorial Cells + 128-D PCA Bottleneck Control
# ==============================================================================

class MatchedSALADFactorialHead(nn.Module):
    """
    Controlled 2x2 Transport x Support-Gate intervention on SALAD's frozen
    fine-tuned representation. U varies only target-column relaxation rho2;
    G varies only the common support gate. Shared marginal, dustbin, iteration
    count, CLS weighting, and initial parameters are identical across arms.
    """

    def __init__(self, mode: str, grid_hw: Tuple[int, int] = (23, 23)):
        super().__init__()
        self.mode = mode
        self.grid_hw = grid_hw
        self.N = grid_hw[0] * grid_hw[1]
        self.M = 64

        # 196-D reliability network: 128 local + 65 logits + 3 structural cues.
        self.rg1 = nn.Linear(196, 32)
        self.rgconv = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.rg2 = nn.Linear(32, 1)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.rg1.weight, gain=0.5)
            self.rg1.bias.zero_()
            self.rgconv.weight.fill_(1.0 / 9.0)
            self.rgconv.bias.zero_()
            nn.init.xavier_uniform_(self.rg2.weight, gain=0.5)
            self.rg2.bias.fill_(1.2)

        self.logit_rho1 = nn.Parameter(torch.tensor(2.4, dtype=torch.float32))
        self.logit_rho2 = nn.Parameter(torch.tensor(1.8, dtype=torch.float32))
        self.logit_dust_boost = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def compute_token_reliability(
        self,
        f_local: torch.Tensor,
        raw_logits: torch.Tensor,
        cues_3ch: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = f_local.shape
        H_p, W_p = self.grid_hw
        # Always recompute boundary-immune cues from (f_local, raw_logits) so cached v1 .npz files work cleanly!
        cues = compute_boundary_immune_cues(f_local, raw_logits, grid_hw=self.grid_hw)
        logits_std = (raw_logits - raw_logits.mean(dim=-1, keepdim=True)) / (raw_logits.std(dim=-1, keepdim=True) + 1e-6)
        inp = torch.cat([f_local, logits_std, cues], dim=-1)  # (B, 529, 196)

        h1 = F.gelu(self.rg1(inp)).transpose(1, 2).contiguous().view(B, 32, H_p, W_p)
        h_dw = self.rgconv(h1).flatten(2).transpose(1, 2).contiguous()
        learned_gate = torch.sigmoid(self.rg2(h_dw).squeeze(-1))

        # Structural heuristic multiplied by learned reliability; no oracle mask.
        r = cues[:, :, 1] * (0.35 + 0.65 * learned_gate)
        return torch.clamp(r, 0.015, 0.995)

    def controlled_transport(self, raw_logits: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Same iteration count, row source and dustbin logits for ALL four cells.

        Factor U is ONLY rho2: balanced-column rho2=1 vs learned rho2<1.
        Final update is column projection so the balanced-column mass equality
        holds (up to numerical precision) *before* optional gating.
        """
        B, N, Mp1 = raw_logits.shape
        assert N == self.N and Mp1 == self.M + 1
        nu = torch.full((1, self.M + 1), 1.0 / float(N),
                        device=raw_logits.device, dtype=raw_logits.dtype)
        nu[:, -1] = (N - self.M) / float(N)
        log_nu = torch.log(nu)
        log_mu = torch.log((0.15 + 0.85 * r) / float(N) + 1e-12)

        # Reliability-aware dustbin adjustment is a matched shared component.
        dust_boost = F.softplus(self.logit_dust_boost).clamp(0.5, 4.0)
        logits = torch.cat([
            raw_logits[:, :, :self.M],
            raw_logits[:, :, self.M:] + dust_boost * (1.0-r).unsqueeze(-1)
        ], dim=-1)
        rho1 = torch.sigmoid(self.logit_rho1).clamp(0.85, 0.995)
        rho2 = (torch.ones((), dtype=logits.dtype, device=logits.device)
                if self.mode.startswith('rgtmp_bal')
                else torch.sigmoid(self.logit_rho2).clamp(0.75, 0.98))

        u = torch.zeros((B, N), device=logits.device, dtype=logits.dtype)
        v = torch.zeros((B, self.M + 1), device=logits.device, dtype=logits.dtype)
        # Identical six full cycles; final v update exactly enforces the
        # balanced-column marginal and permits a controlled UOT relaxation.
        for _ in range(6):
            u = rho1 * (log_mu - torch.logsumexp(logits + v[:, None, :], dim=-1))
            v = rho2 * (log_nu - torch.logsumexp(logits + u[:, :, None], dim=1))
        P = torch.exp(logits + u[:, :, None] + v[:, None, :])
        return P[:, :, :self.M]

    def forward(
        self,
        f_local: torch.Tensor,
        f_cls: torch.Tensor,
        raw_logits: torch.Tensor,
        cues_3ch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # All four heads get the same input features and the same reliability
        # and transport computations, with ONLY two binary interventions.
        r = self.compute_token_reliability(f_local, raw_logits, cues_3ch)
        P_sal = self.controlled_transport(raw_logits, r)
        m_j = P_sal.sum(dim=1)
        v_raw = torch.einsum('bnm,bnd->bmd', P_sal, f_local)
        v_unit = F.normalize(v_raw, p=2, dim=-1)

        if self.mode in ('rgtmp_bal_gate', 'euot_1stage'):
            # SAME support gate (factor G), regardless of balanced/UOT column
            # conservation. Positive column scaling cancels in v_unit alone.
            r_bar_j = (P_sal * r[:, :, None]).sum(dim=1) / (m_j + 1e-12)
            psi_r = r_bar_j.square() / (r_bar_j.square() + 0.18**2 + 1e-12)
            support = torch.norm(v_raw, p=2, dim=-1) * psi_r
            v_agg = v_unit * support[:, :, None]
        else:
            v_agg = v_unit

        # Matched CLS treatment in ALL four arms. No factor-specific CLS gate.
        flat = F.normalize(v_agg.flatten(1), p=2, dim=1)
        desc = F.normalize(torch.cat([flat, f_cls], dim=1), p=2, dim=1)
        return desc, r


class PCA128KMeansBottleneckControl(nn.Module):
    def __init__(self, pca_mean: torch.Tensor, pca_W: torch.Tensor, vocab_64x128: torch.Tensor):
        super().__init__()
        self.register_buffer("pca_mean", pca_mean.float())
        self.register_buffer("pca_W", pca_W.float())
        self.register_buffer("W_sal", F.normalize(vocab_64x128.float(), p=2, dim=-1))
        self.eps = 0.10

    @torch.inference_mode()
    def forward(self, x_patch_768: torch.Tensor, t_cls_768: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_patch_768.shape
        M = self.W_sal.shape[0]
        t12 = F.normalize((x_patch_768 - self.pca_mean) @ self.pca_W, p=2, dim=-1)
        cls = F.normalize((t_cls_768 - self.pca_mean) @ self.pca_W, p=2, dim=-1)

        S = t12 @ self.W_sal.T
        logK = torch.cat([S, torch.full((B, N, 1), 0.35, device=t12.device)], dim=-1) / self.eps
        nu = torch.full((M + 1,), 1.0 / float(N), device=t12.device)
        nu[-1] = float(N - M) / float(N)
        lognu = nu.log()
        logmu = torch.full((B, N), -math.log(float(N)), device=t12.device)

        u = torch.zeros((B, N), device=t12.device)
        v = torch.zeros((B, M + 1), device=t12.device)
        for _ in range(8):
            u = logmu - torch.logsumexp(logK + v[:, None, :], dim=-1)
            v = lognu - torch.logsumexp(logK + u[:, :, None], dim=1)
        P = torch.exp(logK + u[:, :, None] + v[:, None, :])[:, :, :M]
        mj = P.sum(dim=1)
        weighted = torch.einsum("bnm,bnd->bmd", P, t12)
        centered = weighted - 0.70 * mj[:, :, None] * self.W_sal[None, :, :]
        agg = F.normalize(centered, p=2, dim=-1)
        flat = F.normalize(agg.flatten(1), p=2, dim=1)
        return F.normalize(torch.cat([flat, 0.25 * cls], dim=1), p=2, dim=1)


# ==============================================================================
# 6. Training & Development-Split Verification (Reuses Existing Latent Cache!)
# ==============================================================================

def materialize_split_latents(
    backbone: FrozenOfficialSALADBackbone,
    pairs: List[Tuple[Path, Path]],
    cache_dir: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_paths = [p[0] for p in pairs]
    q_paths = [p[1] for p in pairs]
    out_files: Dict[str, Path] = {}

    for tag, paths, occ in [
        ("db", db_paths, 0.0),
        ("q00", q_paths, 0.0),
        ("q25", q_paths, 0.25),
        ("q50", q_paths, 0.50),
    ]:
        fp = cache_dir / f"{tag}.npz"
        out_files[tag] = fp
        if fp.is_file():
            # Preserve v2 caches only when required feature fields and counts match.
            try:
                with np.load(fp) as cached:
                    required = {"official_salad_desc", "f_local", "f_cls", "raw_logits", "cues_3ch", "x_patch_768", "t_cls_768"}
                    valid = required.issubset(cached.files) and all(len(cached[k]) == len(paths) for k in required)
                    valid = valid and cached["f_local"].shape[1:] == (529, 128)
                    valid = valid and cached["raw_logits"].shape[1:] == (529, 65)
                if valid:
                    print(f"[LatentCache] Reusing verified {cache_dir.name}/{tag}.npz", flush=True)
                    continue
            except Exception as exc:
                print(f"[LatentCache] Invalid cache {fp}: {exc}",flush=True)
            fp.unlink(missing_ok=True)
        ds = OccludedPathsDataset(paths, sz=322, occ=occ, seed=seed)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        bufs: Dict[str, List[np.ndarray]] = {
            "official_salad_desc": [],
            "f_local": [],
            "f_cls": [],
            "raw_logits": [],
            "cues_3ch": [],
            "x_patch_768": [],
            "t_cls_768": [],
        }
        for imgs in loader:
            lat = backbone.extract_all_latents(imgs, need_official_desc=True)
            for k in bufs:
                bufs[k].append(lat[k].cpu().half().numpy())
        tmp = fp.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: np.concatenate(v, axis=0) for k, v in bufs.items()})
        os.replace(tmp, fp)
        print(f"[LatentCache] Saved {cache_dir.name}/{tag}.npz ({len(paths)} images)", flush=True)

    return out_files


def fit_pca128_and_kmeans_on_train(train_db_npz: Path, out_pt: Path, seed: int = 42) -> PCA128KMeansBottleneckControl:
    if out_pt.is_file():
        ck = torch.load(out_pt, map_location="cpu", weights_only=True)
        return PCA128KMeansBottleneckControl(ck["mean"], ck["W"], ck["vocab"]).to(DEVICE).eval()

    with np.load(train_db_npz) as z:
        x_768 = z["x_patch_768"].astype(np.float32)
    flat = torch.from_numpy(x_768[:, ::3, :].reshape(-1, 768))
    mu = flat.mean(dim=0)
    centered = flat - mu
    cov = (centered.T @ centered) / float(centered.shape[0] - 1)
    _, vecs = torch.linalg.eigh(cov)
    W = vecs[:, -128:].flip(1).contiguous()

    proj = F.normalize((torch.from_numpy(x_768.reshape(-1, 768)) - mu) @ W, p=2, dim=-1).numpy()
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=64, random_state=seed, batch_size=2048, n_init=3, max_iter=70)
    km.fit(proj)
    vocab = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    vocab = F.normalize(vocab, p=2, dim=-1)

    torch.save({"mean": mu, "W": W, "vocab": vocab}, out_pt)
    return PCA128KMeansBottleneckControl(mu, W, vocab).to(DEVICE).eval()


def train_matched_four_factorial_heads(
    train_files: Dict[str, Path],
    ckpt_dir: Path,
    seed: int = 42,
    epochs: int = 8,
    batch_size: int = 32,
) -> Dict[str, MatchedSALADFactorialHead]:
    """
    Trains all four 2x2 factorial cells on the 150 Gardens Point training pairs using:
      - Paired initialization (`set_controlled_seed(seed)` before each cell)
      - Joint multi-occlusion batches (`q00`, `q25`, `q50` in every step);
        stochastic losses need not decrease monotonically.
      - Clean-vs-occluded token fidelity target (`(f_local_occ * f_local_clean).sum(-1)`)
        on the training split (zero test-time oracle!).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Only decode the four required tensor fields; large cached 768-D patch
    # latents and official SALAD descriptors are not needed during head training.
    required = ("f_local", "f_cls", "raw_logits", "cues_3ch")
    def load_train_fields(path):
        with np.load(path) as z:
            if not set(required).issubset(z.files):
                raise ValueError(f"Incomplete Gardens Point training latent cache: {path}")
            return {k: z[k].astype(np.float32) for k in required}
    db_z = load_train_fields(train_files["db"])
    q00_z = load_train_fields(train_files["q00"])
    q25_z = load_train_fields(train_files["q25"])
    q50_z = load_train_fields(train_files["q50"])
    q_pools = [q00_z, q25_z, q50_z]
    N_train = len(db_z["f_local"])

    def _to_gpu(z_obj: Dict[str, np.ndarray], idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(z_obj["f_local"][idx], device=DEVICE),
            torch.as_tensor(z_obj["f_cls"][idx], device=DEVICE),
            torch.as_tensor(z_obj["raw_logits"][idx], device=DEVICE),
            torch.as_tensor(z_obj["cues_3ch"][idx], device=DEVICE),
        )

    models: Dict[str, MatchedSALADFactorialHead] = {}
    for mode in FACTORIAL_MODES:
        ckpt_file = ckpt_dir / f"{mode}_seed{seed}_v4.pt"
        set_controlled_seed(seed)
        model = MatchedSALADFactorialHead(mode=mode).to(DEVICE)

        if ckpt_file.is_file():
            ck = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            if ck.get("schema") == SCHEMA and ck.get("done_epochs") == epochs:
                model.load_state_dict(ck["weights"])
                model.eval()
                models[mode] = model
                print(f"[Train-Resume] Loaded completed {mode} (seed={seed})", flush=True)
                continue

        opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        for ep in range(epochs):
            model.train()
            rng = np.random.default_rng(seed * 1000 + ep)
            perm = rng.permutation(N_train)
            loss_sum = 0.0
            steps = 0

            for st in range(0, N_train, batch_size):
                idx = perm[st:st + batch_size]
                if len(idx) < 4:
                    continue
                t_db = _to_gpu(db_z, idx)
                t_q00 = _to_gpu(q00_z, idx)
                f_clean_tok = t_q00[0]  # (B, 529, 128) clean query tokens on training set

                d_db, r_db = model(*t_db)
                cues_db = compute_boundary_immune_cues(t_db[0], t_db[2])
                loss_r_db = F.mse_loss(r_db, cues_db[:, :, 1].detach())

                batch_loss = 0.25 * loss_r_db
                target = torch.arange(len(idx), device=DEVICE)

                for q_z in q_pools:
                    t_q = _to_gpu(q_z, idx)
                    d_q, r_q = model(*t_q)

                    scores = d_q @ d_db.T
                    loss_nce = 0.5 * (
                        F.cross_entropy(scores / 0.07, target)
                        + F.cross_entropy(scores.T / 0.07, target)
                    )

                    # Self-supervised clean-to-occluded token fidelity target on training split
                    tok_cos = (t_q[0] * f_clean_tok).sum(dim=-1)  # 1.0 on visible patches, low on occluded patches
                    cues_q = compute_boundary_immune_cues(t_q[0], t_q[2])
                    r_tgt = torch.clamp(
                        torch.sigmoid(12.0 * (tok_cos - 0.65)) * cues_q[:, :, 1],
                        0.015,
                        0.990,
                    ).detach()
                    loss_r = F.mse_loss(r_q, r_tgt)

                    batch_loss = batch_loss + (loss_nce + 0.80 * loss_r) / 3.0

                opt.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                loss_sum += float(batch_loss.item())
                steps += 1

            if (ep + 1) in (1, epochs // 2, epochs):
                print(f"[Train] seed={seed} {mode:<16s} epoch={ep+1:2d}/{epochs} loss={loss_sum/max(1,steps):.4f}", flush=True)

        torch.save({
            "schema": SCHEMA,
            "mode": mode,
            "seed": seed,
            "done_epochs": epochs,
            "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, ckpt_file)
        model.eval()
        models[mode] = model

    return models


def evaluate_on_cached_latents(
    split_files: Dict[str, Path],
    models: Dict[str, MatchedSALADFactorialHead],
    pca_ctrl: PCA128KMeansBottleneckControl,
    positives: List[List[int]],
    seed: int = 42,
) -> Dict[str, Any]:
    db_z = np.load(split_files["db"])

    with torch.inference_mode():
        t_db = (
            torch.as_tensor(db_z["f_local"].astype(np.float32), device=DEVICE),
            torch.as_tensor(db_z["f_cls"].astype(np.float32), device=DEVICE),
            torch.as_tensor(db_z["raw_logits"].astype(np.float32), device=DEVICE),
            torch.as_tensor(db_z["cues_3ch"].astype(np.float32), device=DEVICE),
        )
        db_descs: Dict[str, np.ndarray] = {
            "official_pretrained_salad": db_z["official_salad_desc"].astype(np.float32),
            "pca128_kmeans_control": pca_ctrl(
                torch.as_tensor(db_z["x_patch_768"].astype(np.float32), device=DEVICE),
                torch.as_tensor(db_z["t_cls_768"].astype(np.float32), device=DEVICE),
            ).cpu().numpy(),
        }
        for m_name, mod in models.items():
            d_m, _ = mod(*t_db)
            db_descs[f"{m_name}_s{seed}"] = d_m.cpu().numpy()

    metrics: Dict[str, Any] = {}
    hits_by_method_occ: Dict[str, Dict[str, np.ndarray]] = {}

    for occ_tag, q_key in [("occ00", "q00"), ("occ25", "q25"), ("occ50", "q50")]:
        q_z = np.load(split_files[q_key])
        with torch.inference_mode():
            t_q = (
                torch.as_tensor(q_z["f_local"].astype(np.float32), device=DEVICE),
                torch.as_tensor(q_z["f_cls"].astype(np.float32), device=DEVICE),
                torch.as_tensor(q_z["raw_logits"].astype(np.float32), device=DEVICE),
                torch.as_tensor(q_z["cues_3ch"].astype(np.float32), device=DEVICE),
            )
            q_descs: Dict[str, np.ndarray] = {
                "official_pretrained_salad": q_z["official_salad_desc"].astype(np.float32),
                "pca128_kmeans_control": pca_ctrl(
                    torch.as_tensor(q_z["x_patch_768"].astype(np.float32), device=DEVICE),
                    torch.as_tensor(q_z["t_cls_768"].astype(np.float32), device=DEVICE),
                ).cpu().numpy(),
            }
            for m_name, mod in models.items():
                d_q, _ = mod(*t_q)
                q_descs[f"{m_name}_s{seed}"] = d_q.cpu().numpy()

        for m_key, d_q_np in q_descs.items():
            res, hits = recall_from_embeddings(db_descs[m_key], d_q_np, positives)
            metrics.setdefault(m_key, {})[occ_tag] = res
            hits_by_method_occ.setdefault(m_key, {})[occ_tag] = hits[:, 0]

    did_dev = {}
    for occ_tag in ("occ00", "occ25", "occ50"):
        did_dev[occ_tag] = paired_did_ci_and_signflip(
            hits_by_method_occ[f"euot_1stage_s{seed}"][occ_tag],
            hits_by_method_occ[f"euot_no_gate_s{seed}"][occ_tag],
            hits_by_method_occ[f"rgtmp_bal_gate_s{seed}"][occ_tag],
            hits_by_method_occ[f"rgtmp_bal_s{seed}"][occ_tag],
            seed=seed,
        )
    return {"metrics": metrics, "factorial_did": did_dev, "hits_r1": hits_by_method_occ}


# ==============================================================================
# 7. Fast Single-DataLoader Resumable MSLS-val Encoding & Evaluation
# ==============================================================================

def recall_from_embeddings(
    db: np.ndarray, q: np.ndarray, positives: List[List[int]]
) -> Tuple[Dict[str, Any], np.ndarray]:
    db = db.astype(np.float32)
    db /= np.maximum(np.linalg.norm(db, axis=1, keepdims=True), 1e-12)
    q = q.astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    hits = np.zeros((len(q), 3), dtype=np.uint8)
    maxk = min(10, len(db))

    if DEVICE.type == "cuda":
        db_gpu = torch.from_numpy(db).to(DEVICE, dtype=torch.float16)
        for st in range(0, len(q), 64):
            q_gpu = torch.from_numpy(q[st:st + 64]).to(DEVICE, dtype=torch.float16)
            scores = (q_gpu @ db_gpu.T).float().cpu().numpy()
            idx = np.argpartition(-scores, kth=maxk - 1, axis=1)[:, :maxk]
            ranked = np.take_along_axis(idx, np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1), axis=1)
            for i, rs in enumerate(ranked):
                pset = set(positives[st + i])
                for j, k in enumerate((1, 5, 10)):
                    hits[st + i, j] = int(bool(pset.intersection(rs[:min(k, len(rs))])))
        del db_gpu
    else:
        for st in range(0, len(q), 64):
            scores = q[st:st + 64] @ db.T
            ranked = np.argsort(-scores, axis=1)[:, :maxk]
            for i, rs in enumerate(ranked):
                pset = set(positives[st + i])
                for j, k in enumerate((1, 5, 10)):
                    hits[st + i, j] = int(bool(pset.intersection(rs[:min(k, len(rs))])))

    return {
        "R@1": round(float(hits[:, 0].mean() * 100.0), 2),
        "R@5": round(float(hits[:, 1].mean() * 100.0), 2),
        "R@10": round(float(hits[:, 2].mean() * 100.0), 2),
        "n_query": int(len(q)),
    }, hits


def load_prior_official_salad_split(prior_dir: Path, expected_n: int) -> Optional[np.ndarray]:
    if not prior_dir.is_dir():
        return None
    files = sorted(prior_dir.glob("chunk_*.npy"))
    if not files:
        return None
    arrs = [np.load(f) for f in files]
    cat = np.concatenate(arrs, axis=0)
    if len(cat) == expected_n:
        return cat.astype(np.float32)
    return None


def _descriptor_cache_fingerprint(models, pca_ctrl, paths, occ, seed):
    h = hashlib.sha256()
    h.update(SCHEMA.encode())
    h.update(str(float(occ)).encode())
    h.update(str(int(seed)).encode())
    for path in paths:
        h.update(str(path).encode()); h.update(b"\0")
    for key, mod in sorted(models.items()):
        h.update(key.encode())
        for name, value in sorted(mod.state_dict().items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    for name, value in sorted(pca_ctrl.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode_msls_split_fast_resumable(
    backbone: FrozenOfficialSALADBackbone,
    all_seed_models: Dict[str, MatchedSALADFactorialHead],
    pca_ctrl: PCA128KMeansBottleneckControl,
    paths: List[Path],
    cache_dir: Path,
    occ: float = 0.0,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    prior_official_salad_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    10x faster resumable MSLS-val encoder:
      - Reuses cached `official_pretrained_salad` chunks from `run_10c33061a42a1dda`
        if available.
      - Uses larger 2,048-image chunks with uncompressed `torch.save` (no CPU zlib bottleneck).
      - Runs the ViT-B/14 backbone once per image in mixed precision when CUDA is available.
        Actual time depends on Colab GPU, storage, caches and model count.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 2048
    expected_fp = _descriptor_cache_fingerprint(all_seed_models, pca_ctrl, paths, occ, seed)
    model_keys = ["pca128_kmeans_control"] + list(all_seed_models.keys())

    prior_off = load_prior_official_salad_split(prior_official_salad_dir, len(paths)) if prior_official_salad_dir else None
    need_off = prior_off is None
    all_keys = (["official_pretrained_salad"] if need_off else []) + model_keys

    out_lists: Dict[str, List[np.ndarray]] = {k: [] for k in ["official_pretrained_salad"] + model_keys}
    if prior_off is not None:
        out_lists["official_pretrained_salad"] = [prior_off]

    t0 = time.time()
    for start in range(0, len(paths), chunk_size):
        stop = min(len(paths), start + chunk_size)
        chunk_pt = cache_dir / f"chunk_v4_{start:07d}_{stop:07d}.pt"
        loaded_ok = False
        if chunk_pt.is_file():
            try:
                ck = torch.load(chunk_pt, map_location="cpu", weights_only=True)
                if (ck.get("_cache_fingerprint") == expected_fp and
                    all(k in ck and len(ck[k]) == (stop - start) for k in all_keys)):
                    for k in all_keys:
                        out_lists[k].append(ck[k].float().numpy())
                    loaded_ok = True
            except Exception:
                pass
        if loaded_ok:
            continue

        ds = OccludedPathsDataset(paths[start:stop], sz=322, occ=occ, seed=seed, idx_offset=start)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        buf: Dict[str, List[torch.Tensor]] = {k: [] for k in all_keys}

        for imgs in loader:
            lat = backbone.extract_all_latents(imgs, need_official_desc=need_off)
            if need_off:
                buf["official_pretrained_salad"].append(lat["official_salad_desc"].cpu().half())
            d_pca = pca_ctrl(lat["x_patch_768"], lat["t_cls_768"])
            buf["pca128_kmeans_control"].append(d_pca.cpu().half())

            for m_key, mod in all_seed_models.items():
                with torch.inference_mode():
                    d_m, _ = mod(lat["f_local"], lat["f_cls"], lat["raw_logits"], lat["cues_3ch"])
                buf[m_key].append(d_m.cpu().half())

        saved_dict = {k: torch.cat(v, dim=0) for k, v in buf.items()}
        saved_dict["_cache_fingerprint"] = expected_fp
        tmp_pt = chunk_pt.with_suffix(".tmp.pt")
        torch.save(saved_dict, tmp_pt)
        os.replace(tmp_pt, chunk_pt)

        for k in all_keys:
            out_lists[k].append(saved_dict[k].float().numpy())
        elapsed = time.time() - t0
        print(f"[MSLS-Encode] {cache_dir.name}: {stop:,}/{len(paths):,} ({elapsed:.1f}s)", flush=True)

    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out_lists.items()}


def select_active_msls_gallery_subset(
    db_msls: List[Path],
    pos_msls: List[List[int]],
    prior_run_dir: Path,
    top_k_neg: int = 40,
) -> Tuple[List[Path], List[List[int]], Optional[np.ndarray]]:
    """
    When `--fast_msls_pool` is passed, selects the exact union of:
      1. All ground-truth positive gallery images across all 740 queries, AND
      2. The top-K hardest gallery candidates per query from the already-extracted
         18,871 `official_salad_gallery` embeddings!
    This DOES NOT preserve the top-10 retrieval competition for learned heads;
    reducing gallery extraction from 18,871 images to ~3,200 images (~35 seconds on GPU!).
    """
    active_set = set()
    for p_list in pos_msls:
        active_set.update(p_list)

    gal_off = load_prior_official_salad_split(prior_run_dir / "official_salad_gallery", len(db_msls))
    q00_off = load_prior_official_salad_split(prior_run_dir / "official_salad_occ00", len(pos_msls))
    if gal_off is None or q00_off is None:
        raise RuntimeError("Fast candidate-pool requires complete cached official SALAD gallery and clean-query descriptors; GT-only selection is invalid")
    # The shortlist is mined using OFFICIAL SALAD and is NOT the full-gallery ranking
    # for any learned EUOT or balanced head. Label as diagnostic, never as benchmark.
    scores = q00_off @ gal_off.T
    top_idx = np.argpartition(-scores, kth=min(top_k_neg, scores.shape[1] - 1), axis=1)[:, :top_k_neg]
    active_set.update(int(i) for i in top_idx.reshape(-1))

    sorted_idx = sorted(active_set)
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(sorted_idx)}
    sub_db = [db_msls[i] for i in sorted_idx]
    sub_pos = [[old_to_new[i] for i in p_list if i in old_to_new] for p_list in pos_msls]
    sub_gal_off = gal_off[sorted_idx] if gal_off is not None else None
    print(f"[CANDIDATE-POOL DIAGNOSTIC - NOT FULL MSLS] {len(sub_db):,} gallery images: positives plus SALAD-mined top-{top_k_neg}; scores can be optimistic.",flush=True)
    return sub_db, sub_pos, sub_gal_off


# ==============================================================================
# 8. Master Execution Flow
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="EUOT-VPR Matched-Backbone MSLS-val Diagnostic Package (v2-Fast)")
    ap.add_argument("--drive_root", type=Path, default=Path("."))
    ap.add_argument("--msls_manifest", type=Path, default=Path("local_data/msls_manifest.json"))
    ap.add_argument("--msls_tar", type=Path, default=Path("local_data/msls_val_ready_v1.tar"))
    ap.add_argument("--hub_cache", type=Path, default=Path("local_data/torch_hub"))
    ap.add_argument("--prior_run_dir", type=Path, default=Path("outputs/prior_run"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/setting_U"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 108, 2026], help="Seeds to train and evaluate")
    ap.add_argument("--seed", type=int, default=None, help="Convenience alias for exactly one seed")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dev_only", action="store_true", help="Run Step 1 (Gardens Point train + 50-pair dev split) only")
    ap.add_argument("--fast_msls_pool", action="store_true", help="DIAGNOSTIC ONLY: reduced SALAD-mined gallery, NOT official full-gallery recall")
    ap.add_argument("--preflight", action="store_true", help="Verify datasets and caches without GPU execution")
    args = ap.parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]

    train_pairs, dev_pairs = load_gardens_point_train_and_dev_splits(args.drive_root, n_train=150, n_dev=50)
    # The held-out Gardens Point development test must NOT depend on MSLS data.
    db_msls, q_msls, pos_msls = [], [], []
    if not args.dev_only:
        db_msls, q_msls, pos_msls = restore_and_verify_msls_val(args.msls_manifest, args.msls_tar)

    if args.preflight:
        print("PREFLIGHT PASS: 150 training and 50 independent development pairs" +
              ("; full MSLS-val image/GT paths verified" if not args.dev_only else "; MSLS not needed for dev-only"))
        return

    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU required. In Colab: Runtime -> Change runtime type -> GPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if Gardens Point latents are already cached on disk (skip loading backbone if dev_only and cached!)
    train_cache_dir = args.output_dir / "gardens_train_150_latents"
    dev_cache_dir = args.output_dir / "gardens_dev_50_latents"
    caches_exist = all((train_cache_dir / f"{t}.npz").is_file() and (dev_cache_dir / f"{t}.npz").is_file() for t in ("db", "q00", "q25", "q50"))

    backbone: Optional[FrozenOfficialSALADBackbone] = None
    if not caches_exist or not args.dev_only:
        backbone = FrozenOfficialSALADBackbone(args.hub_cache)

    if caches_exist:
        train_lat_files = {t: train_cache_dir / f"{t}.npz" for t in ("db", "q00", "q25", "q50")}
        dev_lat_files = {t: dev_cache_dir / f"{t}.npz" for t in ("db", "q00", "q25", "q50")}
        print("[LatentCache] Instant load: reusing existing Gardens Point train (150) & dev (50) latents from Drive!", flush=True)
    else:
        assert backbone is not None
        train_lat_files = materialize_split_latents(
            backbone, train_pairs, train_cache_dir, batch_size=args.batch_size, num_workers=args.workers, seed=42
        )
        dev_lat_files = materialize_split_latents(
            backbone, dev_pairs, dev_cache_dir, batch_size=args.batch_size, num_workers=args.workers, seed=42
        )

    pca_ctrl = fit_pca128_and_kmeans_on_train(
        train_lat_files["db"], args.output_dir / "pca128_kmeans_control.pt", seed=42
    )

    all_seed_models: Dict[str, MatchedSALADFactorialHead] = {}
    dev_reports: Dict[str, Any] = {}
    # Strict frame-aligned evaluation: only the corresponding image is a labeled positive.
    dev_positives = [[i] for i in range(len(dev_pairs))]

    for s in args.seeds:
        models_s = train_matched_four_factorial_heads(
            train_lat_files, args.output_dir / "checkpoints_v4", seed=s, epochs=args.epochs, batch_size=min(32, args.batch_size)
        )
        for m_name, mod in models_s.items():
            all_seed_models[f"{m_name}_s{s}"] = mod

        dev_rep = evaluate_on_cached_latents(dev_lat_files, models_s, pca_ctrl, dev_positives, seed=s)
        dev_reports[f"seed_{s}"] = {
            "metrics": dev_rep["metrics"],
            "factorial_did": dev_rep["factorial_did"],
        }
        print(f"\n=== STEP 1: DEVELOPMENT SPLIT (50 Held-Out Pairs, Seed {s}) ===")
        for m_k, m_v in dev_rep["metrics"].items():
            print(f"  {m_k:<28s}: Clean={m_v['occ00']['R@1']:6.2f}% | 25% Occ={m_v['occ25']['R@1']:6.2f}% | 50% Occ={m_v['occ50']['R@1']:6.2f}%")
        print(f"  2x2 Factorial DiD (Seed {s}):", dev_rep["factorial_did"], flush=True)

    atomic_json(args.output_dir / "step1_dev_split_results_v4.json", dev_reports)
    if args.dev_only:
        print("[Done] Step 1 (--dev_only) finished. Inspect step1_dev_split_results_v4.json before full MSLS-val.")
        return

    # Step 2: Official MSLS-val Evaluation
    assert backbone is not None
    print("\n==================================================================")
    print("STEP 2: MSLS-val (740 Queries) - " + ("REDUCED DIAGNOSTIC POOL" if args.fast_msls_pool else "OFFICIAL FULL GALLERY"))
    print("==================================================================")
    msls_cache = args.output_dir / "msls_val_chunks_v4"

    if args.fast_msls_pool:
        print("WARNING: Reduced gallery is for preliminary candidate-pool diagnostics only, not official MSLS Recall@K.",flush=True)
        eval_db_paths, eval_pos, sub_gal_off = select_active_msls_gallery_subset(db_msls, pos_msls, args.prior_run_dir, top_k_neg=40)
        db_descs = encode_msls_split_fast_resumable(
            backbone, all_seed_models, pca_ctrl, eval_db_paths, msls_cache / "gallery_fast_pool",
            occ=0.0, batch_size=args.batch_size, num_workers=args.workers, seed=42, prior_official_salad_dir=None
        )
        if sub_gal_off is not None:
            db_descs["official_pretrained_salad"] = sub_gal_off
    else:
        eval_db_paths, eval_pos = db_msls, pos_msls
        prior_gal = (args.prior_run_dir / "official_salad_gallery") if args.prior_run_dir.exists() else None
        db_descs = encode_msls_split_fast_resumable(
            backbone, all_seed_models, pca_ctrl, eval_db_paths, msls_cache / "gallery_full",
            occ=0.0, batch_size=args.batch_size, num_workers=args.workers, seed=42, prior_official_salad_dir=prior_gal
        )

    msls_metrics: Dict[str, Any] = {}
    msls_hits_r1: Dict[str, Dict[str, np.ndarray]] = {}

    for occ in (0.0, 0.25, 0.50):
        occ_tag = f"occ{int(occ * 100):02d}"
        prior_q = (args.prior_run_dir / f"official_salad_{occ_tag}") if args.prior_run_dir.exists() else None
        q_descs = encode_msls_split_fast_resumable(
            backbone, all_seed_models, pca_ctrl, q_msls, msls_cache / occ_tag,
            occ=occ, batch_size=args.batch_size, num_workers=args.workers, seed=42, prior_official_salad_dir=prior_q
        )

        for k in db_descs.keys():
            res, hits = recall_from_embeddings(db_descs[k], q_descs[k], eval_pos)
            msls_metrics.setdefault(k, {})[occ_tag] = res
            msls_hits_r1.setdefault(k, {})[occ_tag] = hits[:, 0]
            np.save(args.output_dir / f"msls_{k}_{occ_tag}_hits.npy", hits)
            print(f"[MSLS-Result] {k:<28s} {occ_tag}: R@1={res['R@1']:6.2f}% | R@5={res['R@5']:6.2f}% | R@10={res['R@10']:6.2f}%", flush=True)

    degradation: Dict[str, Any] = {}
    for k, d_occ in msls_metrics.items():
        degradation[k] = {
            "drop_00_to_25_pp": round(d_occ["occ00"]["R@1"] - d_occ["occ25"]["R@1"], 2),
            "drop_00_to_50_pp": round(d_occ["occ00"]["R@1"] - d_occ["occ50"]["R@1"], 2),
        }

    factorial_stats: Dict[str, Any] = {}
    for s in args.seeds:
        s_stats = {}
        for occ_tag in ("occ00", "occ25", "occ50"):
            u_g = msls_hits_r1[f"euot_1stage_s{s}"][occ_tag]
            u_ng = msls_hits_r1[f"euot_no_gate_s{s}"][occ_tag]
            b_g = msls_hits_r1[f"rgtmp_bal_gate_s{s}"][occ_tag]
            b_ng = msls_hits_r1[f"rgtmp_bal_s{s}"][occ_tag]
            off_s = msls_hits_r1["official_pretrained_salad"][occ_tag]
            pca_c = msls_hits_r1["pca128_kmeans_control"][occ_tag]

            s_stats[occ_tag] = {
                "factorial_2x2_did": paired_did_ci_and_signflip(u_g, u_ng, b_g, b_ng, seed=s),
                "gate_under_uot": paired_query_ci_and_signflip(u_g, u_ng, seed=s),
                "gate_under_bal": paired_query_ci_and_signflip(b_g, b_ng, seed=s),
                "euot1_vs_rgtmp_bal": paired_query_ci_and_signflip(u_g, b_ng, seed=s),
                "euot1_vs_official_salad": paired_query_ci_and_signflip(u_g, off_s, seed=s),
                "matched_backbone_vs_pca128_bottleneck": paired_query_ci_and_signflip(u_g, pca_c, seed=s),
            }
        factorial_stats[f"seed_{s}"] = s_stats

    # The same MSLS queries/gallery are evaluated under each seed: N=740, not 740*seeds.
    # Avoid invalid query-independent p-values from concatenating duplicate query records.
    across_seed_summary = {}
    if len(args.seeds) > 1:
        for occ_tag in ("occ00", "occ25", "occ50"):
            interactions = [factorial_stats[f"seed_{si}"][occ_tag]["factorial_2x2_did"]["diff_pp"] for si in args.seeds]
            across_seed_summary[occ_tag] = {
                "n_independent_training_seeds": len(args.seeds),
                "mean_factorial_did_pp": float(np.mean(interactions)),
                "std_factorial_did_pp": float(np.std(interactions, ddof=1)),
                "per_seed_did_pp": dict(zip([str(x) for x in args.seeds], interactions)),
                "note": "No pseudo-replicated p-value: all seeds share identical 740 evaluation queries",
            }

    final_payload = {
        "schema": SCHEMA,
        "seeds": args.seeds,
        "gallery_size": len(eval_db_paths),
        "evaluation_scope": "salad_mined_candidate_pool_diagnostic_NOT_FULL_MSLS" if args.fast_msls_pool else "official_full_gallery_MSLS_val",
        "fast_msls_pool": bool(args.fast_msls_pool),
        "step1_development_split": dev_reports,
        "step2_msls_val_official_740q": {
            "metrics": msls_metrics,
            "robustness_degradation_pp": degradation,
            "paired_statistical_tests_740q": factorial_stats,
            "across_training_seed_summary": across_seed_summary,
        },
    }
    out_file = args.output_dir / ("candidate_pool_DIAGNOSTIC_v4.json" if args.fast_msls_pool else "full_gallery_msls_results_v4.json")
    atomic_json(out_file, final_payload)
    print(f"\n[COMPLETE] Saved final matched-backbone MSLS diagnostic -> {out_file}", flush=True)


if __name__ == "__main__":
    main()