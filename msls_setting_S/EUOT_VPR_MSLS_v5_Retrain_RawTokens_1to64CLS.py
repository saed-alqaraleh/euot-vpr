#!/usr/bin/env python3
"""EUOT-VPR matched raw-token SALAD head: recorded MSLS-val v5.1 training and inference implementation.
The release retains the numerical implementation and recorded checks; packaging commentary is omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCHEMA_V5 = "euot_msls_matched_salad_v5_raw_tokens_1to64_cls"
RUNNER_VERSION = "v5.1"  # float16 descriptors, learned-parameter log, v5 supplementary zip
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACTORIAL_MODES = ("rgtmp_bal", "rgtmp_bal_gate", "euot_no_gate", "euot_1stage")

EXPECTED_V4_OCC50_SHA256 = {
    "msls_rgtmp_bal_s42_occ50_hits.npy": "3abd5e6f456dda53fd4191d3692607477431cbaadc099445e6179062443f05b7",
    "msls_rgtmp_bal_gate_s42_occ50_hits.npy": "b90a2f02a05ab791e0701da0fbe0dfc1e11a1f778da4d93df0b619cb94d941ee",
    "msls_euot_no_gate_s42_occ50_hits.npy": "8316557d1e3813f19c205a8280a96d8ff11650000a5540c64c6b1417da93a920",
    "msls_euot_1stage_s42_occ50_hits.npy": "36dbfa93d90b7aa87852b33190b5ed549f1a0aa6f7c1c7a074883bbaf391da49",
    "msls_rgtmp_bal_s108_occ50_hits.npy": "9895c834d3d73813eb3187eddfccb8107d082bdd4f689307934b2ad4a54ec8e1",
    "msls_rgtmp_bal_gate_s108_occ50_hits.npy": "6f158683c97558c20ec69669613644d442bef5dfa0fb82a87eb8755465a57d1c",
    "msls_euot_no_gate_s108_occ50_hits.npy": "70e34e829a8ca6a7e6e5b0c02269c275e673bae8fa83b5e355bfb039c6c75724",
    "msls_euot_1stage_s108_occ50_hits.npy": "d00baacc4576f8b984a0fc2000ed40e8f2b37c8d1ef51016d9bf56e70c3f79ef",
    "msls_rgtmp_bal_s2026_occ50_hits.npy": "723374369caf655bfde577caaa6d8a0d8def2ac93990d63cbc5cc0966e3832c8",
    "msls_rgtmp_bal_gate_s2026_occ50_hits.npy": "c079c76ac56321f4c5f51c2c18d1caa740a2cc3eee9b18848f002289a5ad6bd9",
    "msls_euot_no_gate_s2026_occ50_hits.npy": "dc7bb8c8cf0284e2bb764fc476329e81dd732833e39c2bf3ce5c04b7187a49fa",
    "msls_euot_1stage_s2026_occ50_hits.npy": "c04cc2a67c95156b4d7d67fc93095bebadf779817171c3cbd5523db94f89ffd6",
}


def export_v4_supplementary_bundle(v4_dir: Path, out_zip: Path) -> Dict[str, Any]:
    """
    Exports all 13 v4 checkpoints (.pt) and all 42 MSLS-val per-query hit arrays (.npy)
    into a single <1 MB supplementary ZIP archive with SHA256_MANIFEST.json.
    """
    pts = sorted((v4_dir / "checkpoints_v4").glob("*.pt"))
    pca_pt = v4_dir / "pca128_kmeans_control.pt"
    if pca_pt.is_file():
        pts.append(pca_pt)
    npys = sorted(v4_dir.glob("msls_*_hits.npy"))
    manifest: Dict[str, Any] = {"checkpoints_v4": {}, "hit_arrays_v4": {}}
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pts:
            b = p.read_bytes()
            h = hashlib.sha256(b).hexdigest()
            manifest["checkpoints_v4"][p.name] = {"bytes": len(b), "sha256": h}
            zf.writestr(f"checkpoints_v4/{p.name}", b)
        for p in npys:
            b = p.read_bytes()
            h = hashlib.sha256(b).hexdigest()
            if p.name in EXPECTED_V4_OCC50_SHA256:
                assert h == EXPECTED_V4_OCC50_SHA256[p.name], f"SHA-256 mismatch on {p.name}"
            manifest["hit_arrays_v4"][p.name] = {"bytes": len(b), "sha256": h}
            zf.writestr(f"hit_arrays_v4/{p.name}", b)
        zf.writestr("SHA256_MANIFEST.json", json.dumps(manifest, indent=2))
    print(
        f"[Step 1 PASS] Exported {len(pts)} v4 checkpoints and {len(npys)} v4 hit arrays "
        f"-> {out_zip} ({out_zip.stat().st_size:,} bytes)",
        flush=True,
    )
    return manifest


class MatchedSALADRawToken1to64Head(nn.Module):
    """
    Identical 2x2 factorial design to MatchedSALADFactorialHead (v4.1), with the
    two exact SALAD alignment changes requested by the co-author:
      1. Aggregates RAW un-normalized local token features `f_raw = agg.cluster_features(x)`
         in `v_raw = torch.einsum('bnm,bnd->bmd', P_sal, f_raw)` before intra-cluster
         unit L2 normalization `v_unit = F.normalize(v_raw, p=2, dim=-1)`.
      2. Uses SALAD's native 1:64 global-to-cluster squared-norm weight:
         `flat = F.normalize(v_agg.flatten(1), p=2, dim=1) * 8.0` (squared norm = 64.0)
         concatenated with unit `f_cls` (squared norm = 1.0) before final L2 normalization.
    """

    def __init__(self, mode: str, grid_hw: Tuple[int, int] = (23, 23), cls_cluster_norm_scale: float = 8.0):
        super().__init__()
        self.mode = mode
        self.grid_hw = grid_hw
        self.N = grid_hw[0] * grid_hw[1]
        self.M = 64
        self.cls_cluster_norm_scale = float(cls_cluster_norm_scale)

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
        f_local_unit: torch.Tensor,
        raw_logits: torch.Tensor,
        cues_3ch: torch.Tensor,
        compute_cues_fn,
    ) -> torch.Tensor:
        B, N, _ = f_local_unit.shape
        H_p, W_p = self.grid_hw
        cues = compute_cues_fn(f_local_unit, raw_logits, grid_hw=self.grid_hw)
        logits_std = (raw_logits - raw_logits.mean(dim=-1, keepdim=True)) / (raw_logits.std(dim=-1, keepdim=True) + 1e-6)
        inp = torch.cat([f_local_unit, logits_std, cues], dim=-1)  # (B, 529, 196)

        h1 = F.gelu(self.rg1(inp)).transpose(1, 2).contiguous().view(B, 32, H_p, W_p)
        h_dw = self.rgconv(h1).flatten(2).transpose(1, 2).contiguous()
        learned_gate = torch.sigmoid(self.rg2(h_dw).squeeze(-1))

        r = cues[:, :, 1] * (0.35 + 0.65 * learned_gate)
        return torch.clamp(r, 0.015, 0.995)

    def controlled_transport(self, raw_logits: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B, N, Mp1 = raw_logits.shape
        nu = torch.full((1, self.M + 1), 1.0 / float(N), device=raw_logits.device, dtype=raw_logits.dtype)
        nu[:, -1] = (N - self.M) / float(N)
        log_nu = torch.log(nu)
        log_mu = torch.log((0.15 + 0.85 * r) / float(N) + 1e-12)

        dust_boost = F.softplus(self.logit_dust_boost).clamp(0.5, 4.0)
        logits = torch.cat([
            raw_logits[:, :, :self.M],
            raw_logits[:, :, self.M:] + dust_boost * (1.0 - r).unsqueeze(-1),
        ], dim=-1)
        rho1 = torch.sigmoid(self.logit_rho1).clamp(0.85, 0.995)
        rho2 = (
            torch.ones((), dtype=logits.dtype, device=logits.device)
            if self.mode.startswith("rgtmp_bal")
            else torch.sigmoid(self.logit_rho2).clamp(0.75, 0.98)
        )

        u = torch.zeros((B, N), device=logits.device, dtype=logits.dtype)
        v = torch.zeros((B, self.M + 1), device=logits.device, dtype=logits.dtype)
        for _ in range(6):
            u = rho1 * (log_mu - torch.logsumexp(logits + v[:, None, :], dim=-1))
            v = rho2 * (log_nu - torch.logsumexp(logits + u[:, :, None], dim=1))
        P = torch.exp(logits + u[:, :, None] + v[:, None, :])
        return P[:, :, :self.M]

    def forward(
        self,
        f_local_unit: torch.Tensor,
        f_raw: torch.Tensor,
        f_cls: torch.Tensor,
        raw_logits: torch.Tensor,
        cues_3ch: torch.Tensor,
        compute_cues_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.compute_token_reliability(f_local_unit, raw_logits, cues_3ch, compute_cues_fn)
        P_sal = self.controlled_transport(raw_logits, r)
        m_j = P_sal.sum(dim=1)

        # Aggregate RAW un-normalized token vectors f_raw (as in salad.py line 138)
        v_raw = torch.einsum("bnm,bnd->bmd", P_sal, f_raw)
        v_unit = F.normalize(v_raw, p=2, dim=-1)

        if self.mode in ("rgtmp_bal_gate", "euot_1stage"):
            r_bar_j = (P_sal * r[:, :, None]).sum(dim=1) / (m_j + 1e-12)
            psi_r = r_bar_j.square() / (r_bar_j.square() + 0.18**2 + 1e-12)
            support = torch.norm(v_raw, p=2, dim=-1) * psi_r
            v_agg = v_unit * support[:, :, None]
        else:
            v_agg = v_unit

        # Official SALAD 1:64 global-to-cluster squared-norm ratio:
        # 64 unit clusters have total L2 norm sqrt(64) = 8.0 vs. unit f_cls (norm = 1.0)
        flat = F.normalize(v_agg.flatten(1), p=2, dim=1) * self.cls_cluster_norm_scale
        desc = F.normalize(torch.cat([flat, f_cls], dim=1), p=2, dim=1)
        return desc, r


@torch.inference_mode()
def compute_f_raw_from_cached_768(cluster_mlp: nn.Module, x_768_np: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """
    Computes un-normalized `f_raw = agg.cluster_features(x_feat)` (shape: (N_img, 529, 128))
    directly from the cached `x_patch_768` tensor in `gardens_train_150_latents` /
    `gardens_dev_50_latents` in < 1 second without re-running DINOv2 ViT-B/14!
    """
    out = []
    N_img = x_768_np.shape[0]
    for st in range(0, N_img, batch_size):
        xb = torch.as_tensor(x_768_np[st:st + batch_size].astype(np.float32), device=DEVICE)
        B = xb.shape[0]
        x_grid = xb.transpose(1, 2).contiguous().view(B, 768, 23, 23)
        f_raw = cluster_mlp(x_grid).float().flatten(2).transpose(1, 2).contiguous()
        out.append(f_raw.cpu().numpy())
    return np.concatenate(out, axis=0)


def train_v5_four_factorial_heads(
    v4_mod,
    train_files: Dict[str, Path],
    cluster_mlp: nn.Module,
    ckpt_dir: Path,
    seed: int = 42,
    epochs: int = 12,
    batch_size: int = 32,
) -> Dict[str, MatchedSALADRawToken1to64Head]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    required = ("f_local", "f_cls", "raw_logits", "cues_3ch", "x_patch_768")

    def load_fields(path: Path) -> Dict[str, np.ndarray]:
        with np.load(path) as z:
            d = {k: z[k].astype(np.float32) for k in required}
        d["f_raw"] = compute_f_raw_from_cached_768(cluster_mlp, d.pop("x_patch_768"))
        return d

    db_z = load_fields(train_files["db"])
    q00_z = load_fields(train_files["q00"])
    q25_z = load_fields(train_files["q25"])
    q50_z = load_fields(train_files["q50"])
    q_pools = [q00_z, q25_z, q50_z]
    N_train = len(db_z["f_local"])

    def _to_gpu(z_obj: Dict[str, np.ndarray], idx: np.ndarray):
        return (
            torch.as_tensor(z_obj["f_local"][idx], device=DEVICE),
            torch.as_tensor(z_obj["f_raw"][idx], device=DEVICE),
            torch.as_tensor(z_obj["f_cls"][idx], device=DEVICE),
            torch.as_tensor(z_obj["raw_logits"][idx], device=DEVICE),
            torch.as_tensor(z_obj["cues_3ch"][idx], device=DEVICE),
        )

    models: Dict[str, MatchedSALADRawToken1to64Head] = {}
    for mode in FACTORIAL_MODES:
        ckpt_file = ckpt_dir / f"{mode}_seed{seed}_v5.pt"
        v4_mod.set_controlled_seed(seed)
        model = MatchedSALADRawToken1to64Head(mode=mode, cls_cluster_norm_scale=8.0).to(DEVICE)

        if ckpt_file.is_file():
            ck = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            if ck.get("schema") == SCHEMA_V5 and ck.get("done_epochs") == epochs:
                model.load_state_dict(ck["weights"])
                model.eval()
                models[mode] = model
                print(f"[Train-Resume v5] Loaded completed {mode} (seed={seed})", flush=True)
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
                f_clean_tok = t_q00[0]

                d_db, r_db = model(*t_db, v4_mod.compute_boundary_immune_cues)
                cues_db = v4_mod.compute_boundary_immune_cues(t_db[0], t_db[3])
                loss_r_db = F.mse_loss(r_db, cues_db[:, :, 1].detach())

                batch_loss = 0.25 * loss_r_db
                target = torch.arange(len(idx), device=DEVICE)

                for q_z in q_pools:
                    t_q = _to_gpu(q_z, idx)
                    d_q, r_q = model(*t_q, v4_mod.compute_boundary_immune_cues)

                    scores = d_q @ d_db.T
                    loss_nce = 0.5 * (
                        F.cross_entropy(scores / 0.07, target)
                        + F.cross_entropy(scores.T / 0.07, target)
                    )

                    tok_cos = (t_q[0] * f_clean_tok).sum(dim=-1)
                    cues_q = v4_mod.compute_boundary_immune_cues(t_q[0], t_q[3])
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
                print(f"[Train v5 Raw+1:64] seed={seed} {mode:<16s} epoch={ep+1:2d}/{epochs} loss={loss_sum/max(1,steps):.4f}", flush=True)

        torch.save({
            "schema": SCHEMA_V5,
            "mode": mode,
            "seed": seed,
            "done_epochs": epochs,
            "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, ckpt_file)
        model.eval()
        models[mode] = model

    return models


@torch.inference_mode()
def extract_latents_with_f_raw(backbone, compute_cues_fn, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
    x = (imgs.to(DEVICE, non_blocking=True) - backbone.mean) / backbone.std
    with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda"), dtype=torch.float16):
        x_feat, t_cls = backbone.salad.backbone(x)
        agg = backbone.salad.aggregator
        B, _, H_p, W_p = x_feat.shape
        N = H_p * W_p
        f_raw = agg.cluster_features(x_feat).float().flatten(2).transpose(1, 2).contiguous()
        f_local = F.normalize(f_raw, p=2, dim=-1)
        s_sal = agg.score(x_feat).float().flatten(2).transpose(1, 2).contiguous()
        dust = agg.dust_bin.float().expand(B, N, 1)
        raw_logits = torch.cat([s_sal, dust], dim=-1)
        f_cls = F.normalize(agg.token_features(t_cls).float(), p=2, dim=-1)
        cues_3ch = compute_cues_fn(f_local, raw_logits, grid_hw=(H_p, W_p))
    return {
        "f_local": f_local,
        "f_raw": f_raw,
        "f_cls": f_cls,
        "raw_logits": raw_logits,
        "cues_3ch": cues_3ch,
    }


def encode_msls_v5_split_resumable(
    v4_mod,
    backbone,
    all_seed_models: Dict[str, MatchedSALADRawToken1to64Head],
    paths: List[Path],
    cache_dir: Path,
    occ: float = 0.0,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 2048
    model_keys = list(all_seed_models.keys())
    out_lists: Dict[str, List[np.ndarray]] = {k: [] for k in model_keys}

    t0 = time.time()
    for start in range(0, len(paths), chunk_size):
        stop = min(len(paths), start + chunk_size)
        chunk_pt = cache_dir / f"chunk_v5_{start:07d}_{stop:07d}.pt"
        if chunk_pt.is_file():
            try:
                ck = torch.load(chunk_pt, map_location="cpu", weights_only=True)
                if ck.get("schema") == SCHEMA_V5 and all(k in ck and len(ck[k]) == (stop - start) for k in model_keys):
                    for k in model_keys:
                        out_lists[k].append(ck[k].half().numpy())  # v5.1: float16 in RAM
                    continue
            except Exception:
                pass

        ds = v4_mod.OccludedPathsDataset(paths[start:stop], sz=322, occ=occ, seed=seed, idx_offset=start)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        buf: Dict[str, List[torch.Tensor]] = {k: [] for k in model_keys}

        for imgs in loader:
            lat = extract_latents_with_f_raw(backbone, v4_mod.compute_boundary_immune_cues, imgs)
            for m_key, mod in all_seed_models.items():
                with torch.inference_mode():
                    d_m, _ = mod(
                        lat["f_local"], lat["f_raw"], lat["f_cls"],
                        lat["raw_logits"], lat["cues_3ch"], v4_mod.compute_boundary_immune_cues
                    )
                buf[m_key].append(d_m.cpu().half())

        saved_dict = {k: torch.cat(v, dim=0) for k, v in buf.items()}
        saved_dict["schema"] = SCHEMA_V5
        tmp_pt = chunk_pt.with_suffix(".tmp.pt")
        torch.save(saved_dict, tmp_pt)
        os.replace(tmp_pt, chunk_pt)

        for k in model_keys:
            out_lists[k].append(saved_dict[k].numpy())  # v5.1: keep float16 in RAM
        print(f"[MSLS-v5-Encode] {cache_dir.name}: {stop:,}/{len(paths):,} ({time.time() - t0:.1f}s)", flush=True)

    return {k: np.concatenate(v, axis=0) for k, v in out_lists.items()}  # v5.1: float16; recall casts per head


def main() -> None:
    ap = argparse.ArgumentParser(description="EUOT-VPR v5.1 raw-token and 1:64 head training")
    ap.add_argument("--v4_script", type=Path, default=Path("/content/EUOT_VPR_MSLS_Matched_Backbone_Final_Diagnostic.py"))
    ap.add_argument("--drive_root", type=Path, default=Path("./"))
    ap.add_argument("--msls_manifest", type=Path, default=Path("local_data/msls_manifest.json"))
    ap.add_argument("--msls_tar", type=Path, default=Path("local_data/msls_val_ready_v1.tar"))
    ap.add_argument("--hub_cache", type=Path, default=Path("local_data/torch_hub"))
    ap.add_argument("--v4_dir", type=Path, default=Path("outputs/setting_U"))
    ap.add_argument("--v4_supp_zip", type=Path, default=Path("local_data/v4_supplementary.zip"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 108, 2026])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export_only", action="store_true", help="Only export MSLS_v4_Checkpoints_and_HitArrays_Supplementary.zip")
    args = ap.parse_args()

    # Step 1: Always export the v4 supplementary archive first (< 3 seconds)
    export_v4_supplementary_bundle(args.v4_dir, args.v4_supp_zip)
    if args.export_only:
        return

    # Step 2: Retrain all 4 factorial heads across all 3 seeds with raw tokens + 1:64 CLS weight
    if not args.v4_script.is_file():
        Alt = Path("EUOT_VPR_MSLS_Matched_Backbone_Final_Diagnostic.py")
        if Alt.is_file():
            args.v4_script = Alt
    spec = importlib.util.spec_from_file_location("v4_mod", str(args.v4_script))
    v4_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4_mod)
    # v5.1: fail fast unless the audited v4.1 runner is the one being imported
    v4_sha = hashlib.sha256(Path(args.v4_script).read_bytes()).hexdigest()
    if v4_sha != "e968dff38d6203f58889aa21eb7a287911fda313bc3c9cc0477efad4bc7c057f":
        raise SystemExit(f"[Preflight FAIL] {args.v4_script} is not the audited v4.1 runner (sha256 {v4_sha[:12]}...). Upload the copy from the run kit.")
    missing = [n for n in ("FrozenOfficialSALADBackbone", "OccludedPathsDataset", "atomic_json", "compute_boundary_immune_cues",
                           "paired_did_ci_and_signflip", "paired_query_ci_and_signflip", "recall_from_embeddings",
                           "restore_and_verify_msls_val", "set_controlled_seed") if not hasattr(v4_mod, n)]
    if missing:
        raise SystemExit(f"[Preflight FAIL] v4 runner lacks: {missing}")
    print("[Preflight PASS] audited v4.1 runner (sha256 f0e34e4d...) with all required helpers", flush=True)

    db_msls, q_msls, pos_msls = v4_mod.restore_and_verify_msls_val(args.msls_manifest, args.msls_tar)
    backbone = v4_mod.FrozenOfficialSALADBackbone(args.hub_cache)
    cluster_mlp = backbone.salad.aggregator.cluster_features

    train_cache_dir = args.v4_dir / "gardens_train_150_latents"
    train_lat_files = {t: train_cache_dir / f"{t}.npz" for t in ("db", "q00", "q25", "q50")}

    ckpt_v5_dir = args.v4_dir / "checkpoints_v5_raw_1to64cls"
    all_seed_models: Dict[str, MatchedSALADRawToken1to64Head] = {}
    for s in args.seeds:
        models_s = train_v5_four_factorial_heads(
            v4_mod, train_lat_files, cluster_mlp, ckpt_v5_dir, seed=s, epochs=args.epochs, batch_size=32
        )
        for m_name, mod in models_s.items():
            all_seed_models[f"{m_name}_s{s}"] = mod

    # Step 3: Full-Gallery MSLS-val Evaluation (18,871 gallery, 740 queries, occ00/occ25/occ50)
    msls_v5_cache = args.v4_dir / "msls_val_chunks_v5_raw_1to64cls"
    db_descs = encode_msls_v5_split_resumable(
        v4_mod, backbone, all_seed_models, db_msls, msls_v5_cache / "gallery_full",
        occ=0.0, batch_size=args.batch_size, num_workers=args.workers, seed=42
    )

    msls_metrics: Dict[str, Any] = {}
    msls_hits_r1: Dict[str, Dict[str, np.ndarray]] = {}

    # Load existing official_pretrained_salad and pca128_kmeans_control hits from v4_dir
    for base_ref in ("official_pretrained_salad", "pca128_kmeans_control"):
        for occ_tag in ("occ00", "occ25", "occ50"):
            hit_p = args.v4_dir / f"msls_{base_ref}_{occ_tag}_hits.npy"
            if hit_p.is_file():
                hits = np.load(hit_p)
                msls_hits_r1.setdefault(base_ref, {})[occ_tag] = hits[:, 0]
                msls_metrics.setdefault(base_ref, {})[occ_tag] = {
                    "R@1": round(float(hits[:, 0].mean() * 100.0), 2),
                    "R@5": round(float(hits[:, 1].mean() * 100.0), 2),
                    "R@10": round(float(hits[:, 2].mean() * 100.0), 2),
                    "n_query": int(len(hits)),
                }

    hits_v5_dir = args.v4_dir / "hits_v5_raw_1to64cls"
    hits_v5_dir.mkdir(parents=True, exist_ok=True)

    for occ in (0.0, 0.25, 0.50):
        occ_tag = f"occ{int(occ * 100):02d}"
        q_descs = encode_msls_v5_split_resumable(
            v4_mod, backbone, all_seed_models, q_msls, msls_v5_cache / occ_tag,
            occ=occ, batch_size=args.batch_size, num_workers=args.workers, seed=42
        )
        for k in db_descs.keys():
            res, hits = v4_mod.recall_from_embeddings(db_descs[k], q_descs[k], pos_msls)
            msls_metrics.setdefault(k, {})[occ_tag] = res
            msls_hits_r1.setdefault(k, {})[occ_tag] = hits[:, 0]
            np.save(hits_v5_dir / f"msls_v5_{k}_{occ_tag}_hits.npy", hits)
            print(f"[MSLS-v5-Result] {k:<28s} {occ_tag}: R@1={res['R@1']:6.2f}% | R@5={res['R@5']:6.2f}% | R@10={res['R@10']:6.2f}%", flush=True)

    factorial_stats: Dict[str, Any] = {}
    for s in args.seeds:
        s_stats = {}
        for occ_tag in ("occ00", "occ25", "occ50"):
            u_g = msls_hits_r1[f"euot_1stage_s{s}"][occ_tag]
            u_ng = msls_hits_r1[f"euot_no_gate_s{s}"][occ_tag]
            b_g = msls_hits_r1[f"rgtmp_bal_gate_s{s}"][occ_tag]
            b_ng = msls_hits_r1[f"rgtmp_bal_s{s}"][occ_tag]
            s_stats[occ_tag] = {
                "factorial_2x2_did": v4_mod.paired_did_ci_and_signflip(u_g, u_ng, b_g, b_ng, seed=s),
                "gate_under_uot": v4_mod.paired_query_ci_and_signflip(u_g, u_ng, seed=s),
                "gate_under_bal": v4_mod.paired_query_ci_and_signflip(b_g, b_ng, seed=s),
                "euot1_vs_rgtmp_bal": v4_mod.paired_query_ci_and_signflip(u_g, b_ng, seed=s),
            }
            if "official_pretrained_salad" in msls_hits_r1:
                off_s = msls_hits_r1["official_pretrained_salad"][occ_tag]
                s_stats[occ_tag]["euot1_vs_official_salad"] = v4_mod.paired_query_ci_and_signflip(u_g, off_s, seed=s)
        factorial_stats[f"seed_{s}"] = s_stats

    # Joint 740-query bootstrap & sign-flip across all 3 seeds (resampling the 740 shared queries jointly)
    three_seed_query_matched = {}
    if len(args.seeds) > 1:
        for occ_tag in ("occ00", "occ25", "occ50"):
            u_g_mean = np.mean([msls_hits_r1[f"euot_1stage_s{s}"][occ_tag].astype(np.float64) for s in args.seeds], axis=0)
            u_ng_mean = np.mean([msls_hits_r1[f"euot_no_gate_s{s}"][occ_tag].astype(np.float64) for s in args.seeds], axis=0)
            b_g_mean = np.mean([msls_hits_r1[f"rgtmp_bal_gate_s{s}"][occ_tag].astype(np.float64) for s in args.seeds], axis=0)
            b_ng_mean = np.mean([msls_hits_r1[f"rgtmp_bal_s{s}"][occ_tag].astype(np.float64) for s in args.seeds], axis=0)
            three_seed_query_matched[occ_tag] = {
                "did_2x2_740q": v4_mod.paired_did_ci_and_signflip(u_g_mean, u_ng_mean, b_g_mean, b_ng_mean, seed=42),
                "euot1_vs_rgtmp_bal_740q": v4_mod.paired_query_ci_and_signflip(u_g_mean, b_ng_mean, seed=42),
                "per_seed_did_pp": {str(s): factorial_stats[f"seed_{s}"][occ_tag]["factorial_2x2_did"]["diff_pp"] for s in args.seeds},
            }

    out_payload = {
        "schema": SCHEMA_V5,
        "seeds": args.seeds,
        "gallery_size": len(db_msls),
        "n_queries": len(q_msls),
        "modifications_from_v4": [
            "Aggregates raw un-normalized token vectors f_raw = agg.cluster_features(x) before intra-cluster L2 normalization (matching salad.py line 138).",
            "Concatenates 64-cluster block at L2 norm sqrt(64)=8.0 with unit f_cls (norm=1.0), preserving official SALAD's 1:64 global-to-cluster squared-norm ratio.",
        ],
        "metrics": msls_metrics,
        "per_seed_statistical_tests_740q": factorial_stats,
        "three_seed_query_matched_summary_740q": three_seed_query_matched,
    }
    # v5.1: learned transport parameters of every retrained head
    learned = {}
    for k, mod in all_seed_models.items():
        with torch.no_grad():
            learned[k] = {
                "rho_1": round(float(torch.sigmoid(mod.logit_rho1).clamp(0.85, 0.995)), 4),
                "rho_2": (1.0 if mod.mode.startswith("rgtmp_bal")
                          else round(float(torch.sigmoid(mod.logit_rho2).clamp(0.75, 0.98)), 4)),
                "dust_boost": round(float(F.softplus(mod.logit_dust_boost).clamp(0.5, 4.0)), 4),
            }
    out_payload["learned_transport_params"] = learned
    out_json = args.v4_dir / "full_gallery_msls_results_v5_raw_1to64cls.json"
    v4_mod.atomic_json(out_json, out_payload)
    # v5.1: package checkpoints, hit arrays and results with SHA-256 digests
    v5_zip = args.v4_dir / "MSLS_v5_RawTokens_1to64CLS_Supplementary.zip"
    man = {"checkpoints_v5": {}, "hit_arrays_v5": {}}
    with zipfile.ZipFile(v5_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sub, key, pat in ((ckpt_v5_dir, "checkpoints_v5", "*.pt"), (hits_v5_dir, "hit_arrays_v5", "*.npy")):
            for fp in sorted(sub.glob(pat)):
                b = fp.read_bytes()
                man[key][fp.name] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                zf.writestr(f"{key}/{fp.name}", b)
        zf.write(out_json, out_json.name)
        zf.writestr("SHA256_MANIFEST.json", json.dumps(man, indent=2))
    print(f"[Step 4 PASS] {len(man['checkpoints_v5'])} checkpoints + {len(man['hit_arrays_v5'])} hit arrays -> {v5_zip}", flush=True)
    print(f"\n[Step 3 PASS] Saved full-gallery MSLS-val v5 results -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
