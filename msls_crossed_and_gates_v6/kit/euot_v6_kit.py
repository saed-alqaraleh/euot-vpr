#!/usr/bin/env python3
"""EUOT-VPR v6 crossed-design GPU runner.

The full-gallery MSLS-val protocol extends the archived factorial as follows:

  * Component: crossed design: the transport x gate factorial in two new settings,
               UW (unit tokens, global share 1/65) and RE (raw tokens, global share 50%),
               so that token normalization and global-token weighting are separated.
  * Component: gate variants in Settings U and S: reliability-only, length-only,
               mass-only and mean-norm gates beside the audited combined gate.
  * Component: absorption diagnostic: mass contributed by masked vs. visible tokens to
               every cluster, focusing on clusters whose clean-image support is masked.
  * Component: query-group export (city, MSLS sequence if metadata exist, and
               positive-overlap components) for grouped bootstrap tests, plus an optional
               evaluation of the frozen heads on an additional dataset.

Nothing in the audited runs is modified. The audited v4.1 and v5.1 runners are imported
read-only (SHA-256 checked) and reused for MSLS restore, the frozen backbone, latent
extraction, the occlusion dataset and recall. Archived checkpoints are only read; all new
files go to --out_dir. The 24 archived Setting U/S heads are re-evaluated first and must
reproduce their archived hit arrays before anything else is reported.

Stages (each writes markers/<stage>.json only after validating its outputs; rerunning a
stage skips completed work):
    preflight, train, evaluate, diagnostics, groups, analysis, package, extra, all
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

KIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT_DIR))
import euot_v6_heads as H6  # noqa: E402
import euot_v6_analysis as A6  # noqa: E402

KIT_VERSION = "v6.0"
SCHEMA_HEAD = "euot_msls_v6_generalized_head"
V4_SHA = "e968dff38d6203f58889aa21eb7a287911fda313bc3c9cc0477efad4bc7c057f"
V5_SHA = "3aedd544ac2cdb2f75c858e990eaf7f8053c106faf5b2a3f5807e618c10ea697"
V4_NAME = "EUOT_VPR_MSLS_Matched_Backbone_Final_Diagnostic.py"
V5_NAME = "EUOT_VPR_MSLS_v5_Retrain_RawTokens_1to64CLS.py"
OCC_LEVELS = (0.0, 0.25, 0.50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRIVE = Path(os.environ.get("EUOT_WORKSPACE", ".")).expanduser().resolve()
EUOT = DRIVE


# ------------------------------------------------------------------------------
# utilities
# ------------------------------------------------------------------------------

class Logger:
    def __init__(self, path: Optional[Path]):
        self.f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")


LOG = Logger(None)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def load_module(name: str, path: Path, expected_sha: str):
    if not path.is_file():
        raise SystemExit(f"[Preflight FAIL] missing audited runner {path}")
    got = sha256_file(path)
    if got != expected_sha:
        raise SystemExit(f"[Preflight FAIL] {path.name} sha256 {got[:12]}... != audited {expected_sha[:12]}...")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config_hash(args: argparse.Namespace, keys: Sequence[str]) -> str:
    h = hashlib.sha256(KIT_VERSION.encode())
    for k in keys:
        h.update(f"{k}={getattr(args, k)!r};".encode())
    return h.hexdigest()[:16]


def marker_path(args, stage: str) -> Path:
    return args.out_dir / "markers" / f"{stage}.json"


def marker_valid(args, stage: str, chash: str) -> bool:
    p = marker_path(args, stage)
    if not p.is_file():
        return False
    try:
        m = json.loads(p.read_text())
    except Exception:
        return False
    if m.get("config_hash") != chash or m.get("kit_version") != KIT_VERSION:
        return False
    for rel, digest in m.get("outputs", {}).items():
        f = args.out_dir / rel
        if not f.is_file() or (digest and sha256_file(f) != digest):
            return False
    return True


def write_marker(args, stage: str, chash: str, outputs: Sequence[Path], extra: Optional[Dict] = None) -> None:
    rec = {"stage": stage, "kit_version": KIT_VERSION, "config_hash": chash, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "outputs": {str(p.relative_to(args.out_dir)): sha256_file(p) for p in outputs}}
    if extra:
        rec.update(extra)
    atomic_json(marker_path(args, stage), rec)


# ------------------------------------------------------------------------------
# configuration of heads
# ------------------------------------------------------------------------------

def all_specs(args) -> List[H6.HeadSpec]:
    specs: List[H6.HeadSpec] = []
    for setting in ("U", "S"):
        for (t, g) in H6.LEGACY_MODE:
            for s in args.seeds:
                specs.append(H6.HeadSpec(setting, t, g, s))
    for setting in args.crossed_settings:
        for (t, g) in H6.LEGACY_MODE:
            for s in args.seeds:
                specs.append(H6.HeadSpec(setting, t, g, s))
    for setting in args.gate_variant_settings:
        for g in args.gate_variants:
            for t in ("bal", "uot"):
                for s in args.seeds:
                    specs.append(H6.HeadSpec(setting, t, g, s))
    seen, out = set(), []
    for sp in specs:
        if sp.key not in seen:
            seen.add(sp.key)
            out.append(sp)
    return out


def archived_ckpt_path(args, spec: H6.HeadSpec) -> Path:
    if spec.setting == "U":
        return args.archived_ckpt_u / f"{spec.legacy_mode}_seed{spec.seed}_v4.pt"
    return args.archived_ckpt_s / f"{spec.legacy_mode}_seed{spec.seed}_v5.pt"


def archived_hits_path(args, spec: H6.HeadSpec, occ_tag: str) -> Path:
    if spec.setting == "U":
        return args.archived_hits_u / f"msls_{spec.legacy_mode}_s{spec.seed}_{occ_tag}_hits.npy"
    return args.archived_hits_s / f"msls_v5_{spec.legacy_mode}_s{spec.seed}_{occ_tag}_hits.npy"


def v6_ckpt_path(args, spec: H6.HeadSpec) -> Path:
    return args.out_dir / "checkpoints_v6" / f"{spec.key}_v6.pt"


def hits_path(args, spec: H6.HeadSpec, occ_tag: str) -> Path:
    return args.out_dir / "hits_v6" / f"msls_v6_{spec.key}_{occ_tag}_hits.npy"


def expected_hashes(args) -> Dict[str, Dict[str, str]]:
    p = KIT_DIR / "expected_archive_hashes.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def load_head(args, spec: H6.HeadSpec, v4) -> H6.GeneralizedFactorialHead:
    model = H6.GeneralizedFactorialHead.from_spec(spec, v4.compute_boundary_immune_cues)
    if spec.is_archived:
        p = archived_ckpt_path(args, spec)
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if int(ck.get("done_epochs", -1)) != args.epochs:
            raise SystemExit(f"archived checkpoint {p.name} has done_epochs={ck.get('done_epochs')}")
    else:
        p = v6_ckpt_path(args, spec)
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if ck.get("schema") != SCHEMA_HEAD or ck.get("spec") != asdict(spec) or int(ck.get("done_epochs", -1)) != args.epochs:
            raise SystemExit(f"v6 checkpoint {p.name} does not match its spec; delete it to retrain")
    model.load_state_dict(ck["weights"], strict=True)
    return model.to(DEVICE).eval()


# ------------------------------------------------------------------------------
# stage: preflight
# ------------------------------------------------------------------------------

def stage_preflight(args, mods) -> None:
    v4, v5 = mods
    chash = config_hash(args, ["seeds", "epochs"])
    issues: List[str] = []

    def check(label: str, p: Path) -> bool:
        ok = p.exists()
        LOG(f"  {'OK     ' if ok else 'MISSING'} {label}: {p}")
        if not ok:
            issues.append(f"{label}: {p}")
        return ok

    LOG("[preflight] required inputs")
    check("MSLS manifest", args.msls_manifest)
    check("MSLS tar (needed only if images are not restored yet)", args.msls_tar) or issues.pop()
    check("torch hub cache", args.hub_cache)
    for t in ("db", "q00", "q25", "q50"):
        check(f"Gardens train latents {t}", args.train_latents / f"{t}.npz")
    exp = expected_hashes(args)
    for spec in all_specs(args):
        if spec.is_archived:
            p = archived_ckpt_path(args, spec)
            if check(f"archived ckpt {spec.key}", p):
                want = exp.get("checkpoints", {}).get(p.name)
                if want and sha256_file(p) != want:
                    issues.append(f"sha256 mismatch {p}")
                    LOG(f"  MISMATCH {p.name}")
            for occ in ("occ00", "occ25", "occ50"):
                q = archived_hits_path(args, spec, occ)
                if check(f"archived hits {spec.key} {occ}", q):
                    want = exp.get("hit_arrays", {}).get(q.name)
                    if want and sha256_file(q) != want:
                        issues.append(f"sha256 mismatch {q}")
                        LOG(f"  MISMATCH {q.name}")
    if args.require_gpu and DEVICE.type != "cuda":
        issues.append("CUDA GPU required (Runtime -> Change runtime type -> GPU)")
    if DEVICE.type == "cuda":
        LOG(f"[preflight] GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    if issues:
        for i in issues:
            LOG(f"  ISSUE {i}")
        raise SystemExit(f"[preflight FAIL] {len(issues)} issue(s); nothing was run")

    db, q, pos = v4.restore_and_verify_msls_val(args.msls_manifest, args.msls_tar)
    # Exact token mask reconstruction vs the audited occlusion dataset (pixel level).
    for occ in (0.25, 0.50):
        ds_c = v4.OccludedPathsDataset(q[:8], sz=322, occ=0.0, seed=42, idx_offset=0)
        ds_o = v4.OccludedPathsDataset(q[:8], sz=322, occ=occ, seed=42, idx_offset=0)
        for i in range(8):
            diff = (ds_c[i] != ds_o[i]).any(dim=0).numpy()
            tok = diff.reshape(23, 14, 23, 14).any(axis=(1, 3))
            rec = H6.occluder_token_mask(i, occ, seed=42)
            if not np.array_equal(tok, rec):
                raise SystemExit(f"[preflight FAIL] token-mask reconstruction differs for query {i} at occ={occ}")
    LOG("[preflight] token-mask reconstruction matches the audited occlusion dataset")
    n_new = sum(not s.is_archived for s in all_specs(args))
    LOG(f"[preflight] {len(all_specs(args))} heads: 24 archived (re-evaluated), {n_new} new")
    write_marker(args, "preflight", chash, [], {"n_gallery": len(db), "n_queries": len(q)})
    LOG("[preflight PASS]")


# ------------------------------------------------------------------------------
# stage: train
# ------------------------------------------------------------------------------

def load_train_fields(args, v5, cluster_mlp, need_raw: bool) -> Dict[str, Dict[str, np.ndarray]]:
    fields = {}
    for t in ("db", "q00", "q25", "q50"):
        keys = ("f_local", "f_cls", "raw_logits") + (("x_patch_768",) if need_raw else ())
        with np.load(args.train_latents / f"{t}.npz") as z:
            d = {k: z[k].astype(np.float32) for k in keys}
        if need_raw:
            d["f_raw"] = v5.compute_f_raw_from_cached_768(cluster_mlp, d.pop("x_patch_768"))
        fields[t] = d
    return fields


def stage_train(args, mods) -> None:
    v4, v5 = mods
    chash = config_hash(args, ["seeds", "epochs", "crossed_settings", "gate_variant_settings", "gate_variants"])
    if marker_valid(args, "train", chash):
        LOG("[train] complete (marker valid); skipping")
        return
    new_specs = [s for s in all_specs(args) if not s.is_archived]
    todo = []
    for s in new_specs:
        p = v6_ckpt_path(args, s)
        ok = False
        if p.is_file():
            try:
                ck = torch.load(p, map_location="cpu", weights_only=True)
                ok = ck.get("schema") == SCHEMA_HEAD and ck.get("spec") == asdict(s) and int(ck.get("done_epochs", -1)) == args.epochs
            except Exception:
                ok = False
        if not ok:
            todo.append(s)
    LOG(f"[train] {len(new_specs)} new heads; {len(new_specs) - len(todo)} already trained; {len(todo)} to train")
    replay = {}
    if todo or args.replay_check:
        backbone = v4.FrozenOfficialSALADBackbone(args.hub_cache)
        cluster_mlp = backbone.salad.aggregator.cluster_features
        need_raw = any(s.token_source == "raw" for s in todo) or args.replay_check
        fields = load_train_fields(args, v5, cluster_mlp, need_raw)
        for i, s in enumerate(todo, 1):
            t0 = time.time()
            model = H6.train_head(s, fields, v4.compute_boundary_immune_cues, v4.set_controlled_seed, DEVICE,
                                  epochs=args.epochs, batch_size=32, log=LOG)
            p = v6_ckpt_path(args, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            torch.save({"schema": SCHEMA_HEAD, "kit_version": KIT_VERSION, "spec": asdict(s), "done_epochs": args.epochs,
                        "learned": H6.learned_transport_params(model),
                        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, tmp)
            os.replace(tmp, p)
            LOG(f"[train] {i}/{len(todo)} {s.key} saved ({time.time() - t0:.1f}s) {H6.learned_transport_params(model)}")
        if args.replay_check:
            # Replays the audited training of two archived cells with the v6 loop.
            for s in (H6.HeadSpec("U", "uot", "combined", args.seeds[0]), H6.HeadSpec("S", "uot", "combined", args.seeds[0])):
                m = H6.train_head(s, fields, v4.compute_boundary_immune_cues, v4.set_controlled_seed, DEVICE,
                                  epochs=args.epochs, batch_size=32, log=LOG)
                ck = torch.load(archived_ckpt_path(args, s), map_location="cpu", weights_only=True)["weights"]
                diffs = {k: float((v.detach().cpu() - ck[k]).abs().max()) for k, v in m.state_dict().items()}
                replay[s.key] = {"max_abs_weight_diff": max(diffs.values()), "per_tensor": diffs}
                LOG(f"[train] replay {s.key}: max |w_v6 - w_archived| = {max(diffs.values()):.3e}")
    learned = {}
    for s in new_specs:
        ck = torch.load(v6_ckpt_path(args, s), map_location="cpu", weights_only=True)
        learned[s.key] = ck.get("learned", {})
    out = args.out_dir / "results" / "learned_transport_params_v6.json"
    atomic_json(out, {"learned": learned, "training_replay_check": replay})
    write_marker(args, "train", chash, [out] + [v6_ckpt_path(args, s) for s in new_specs])
    LOG("[train PASS]")


# ------------------------------------------------------------------------------
# stage: evaluate (in-pass encoding as in the audited v5.1 runner, groups of heads)
# ------------------------------------------------------------------------------

def group_fingerprint(args, specs: Sequence[H6.HeadSpec]) -> str:
    h = hashlib.sha256(KIT_VERSION.encode())
    for s in specs:
        h.update(s.key.encode())
        p = archived_ckpt_path(args, s) if s.is_archived else v6_ckpt_path(args, s)
        h.update(sha256_file(p).encode())
    return h.hexdigest()[:16]


def encode_split(v4, v5, backbone, models: Dict[str, H6.GeneralizedFactorialHead], paths: List[Path],
                 cache_dir: Path, occ: float, args, tag: str) -> Dict[str, np.ndarray]:
    """Mirror of v5.1 encode_msls_v5_split_resumable for generalized heads: chunks of 2,048
    images, float16 descriptors, atomic chunk files, resume from valid chunks."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = list(models)
    out: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    t0 = time.time()
    for start in range(0, len(paths), 2048):
        stop = min(len(paths), start + 2048)
        chunk = cache_dir / f"chunk_{start:07d}_{stop:07d}.pt"
        if chunk.is_file():
            try:
                ck = torch.load(chunk, map_location="cpu", weights_only=True)
                if all(k in ck and len(ck[k]) == stop - start for k in keys):
                    for k in keys:
                        out[k].append(ck[k].numpy())
                    continue
            except Exception:
                pass
        ds = v4.OccludedPathsDataset(paths[start:stop], sz=322, occ=occ, seed=42, idx_offset=start)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        buf: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
        for imgs in loader:
            lat = v5.extract_latents_with_f_raw(backbone, v4.compute_boundary_immune_cues, imgs)
            with torch.inference_mode():
                for k, mod in models.items():
                    d, _ = mod(lat["f_local"], lat["f_raw"], lat["f_cls"], lat["raw_logits"])
                    buf[k].append(d.cpu().half())
        saved = {k: torch.cat(v, dim=0) for k, v in buf.items()}
        tmp = chunk.with_name(chunk.name + ".tmp")
        torch.save(saved, tmp)
        os.replace(tmp, chunk)
        for k in keys:
            out[k].append(saved[k].numpy())
        LOG(f"[encode] {tag}: {stop:,}/{len(paths):,} images x {len(keys)} heads ({time.time() - t0:.1f}s)")
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def stage_evaluate(args, mods) -> None:
    v4, v5 = mods
    specs = all_specs(args)
    chash = config_hash(args, ["seeds", "epochs", "crossed_settings", "gate_variant_settings", "gate_variants"])
    if marker_valid(args, "evaluate", chash):
        LOG("[evaluate] complete (marker valid); skipping")
        return
    for s in specs:
        if not s.is_archived and not v6_ckpt_path(args, s).is_file():
            raise SystemExit(f"[evaluate] untrained head {s.key}; run --stage train first")
    groups = [specs[i:i + args.group_size] for i in range(0, len(specs), args.group_size)]
    LOG(f"[evaluate] {len(specs)} heads in {len(groups)} groups of <= {args.group_size}")
    db, q, pos = v4.restore_and_verify_msls_val(args.msls_manifest, args.msls_tar)
    backbone = None
    for gi, grp in enumerate(groups, 1):
        gfp = group_fingerprint(args, grp)
        gmark = args.out_dir / "markers" / f"evaluate_group{gi:02d}_{gfp}.json"
        expected = [hits_path(args, s, f"occ{int(o * 100):02d}") for s in grp for o in OCC_LEVELS]
        if gmark.is_file() and all(p.is_file() for p in expected):
            LOG(f"[evaluate] group {gi}/{len(groups)} complete; skipping")
            continue
        if backbone is None:
            backbone = v4.FrozenOfficialSALADBackbone(args.hub_cache)
        LOG(f"[evaluate] group {gi}/{len(groups)}: {', '.join(s.key for s in grp)}")
        models = {s.key: load_head(args, s, v4) for s in grp}
        cache = args.chunk_dir / f"group{gi:02d}_{gfp}"
        db_desc = encode_split(v4, v5, backbone, models, db, cache / "gallery", 0.0, args, f"group {gi} gallery")
        for occ in OCC_LEVELS:
            tag = f"occ{int(occ * 100):02d}"
            q_desc = encode_split(v4, v5, backbone, models, q, cache / tag, occ, args, f"group {gi} {tag}")
            for s in grp:
                res, hits = v4.recall_from_embeddings(db_desc[s.key], q_desc[s.key], pos)
                atomic_npy(hits_path(args, s, tag), hits)
                LOG(f"[result] {s.key:<26s} {tag}: R@1={res['R@1']:6.2f} R@5={res['R@5']:6.2f} R@10={res['R@10']:6.2f}")
            del q_desc
        del db_desc, models
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
        atomic_json(gmark, {"heads": [s.key for s in grp], "hits": {p.name: sha256_file(p) for p in expected}})
        if not args.keep_chunk_cache:
            shutil.rmtree(cache, ignore_errors=True)  # local, kit-created cache only

    # Reproduction gate: re-evaluated archived heads vs archived hit arrays.
    rep, worst = {}, 1.0
    for s in specs:
        if not s.is_archived:
            continue
        for occ in ("occ00", "occ25", "occ50"):
            a = np.load(archived_hits_path(args, s, occ))
            b = np.load(hits_path(args, s, occ))
            agree = float((a[:, 0] == b[:, 0]).mean())
            worst = min(worst, agree)
            rep[f"{s.key}_{occ}"] = {"r1_agreement": round(agree, 4), "r1_archived": round(float(a[:, 0].mean() * 100), 2),
                                    "r1_v6": round(float(b[:, 0].mean() * 100), 2),
                                    "all_k_identical": bool(np.array_equal(a, b))}
    n_ident = sum(v["all_k_identical"] for v in rep.values())
    rep_path = args.out_dir / "results" / "reproduction_check.json"
    atomic_json(rep_path, {"min_r1_agreement": round(worst, 4), "n_identical": n_ident, "n_compared": len(rep),
                           "threshold": args.repro_min_agreement, "per_head": rep})
    LOG(f"[evaluate] reproduction: {n_ident}/{len(rep)} archived hit arrays identical; min R@1 agreement {worst:.4f}")
    if worst < args.repro_min_agreement:
        raise SystemExit("[evaluate FAIL] archived heads do not reproduce their hit arrays; see results/reproduction_check.json")
    outs = [rep_path] + [hits_path(args, s, f"occ{int(o * 100):02d}") for s in specs for o in OCC_LEVELS]
    write_marker(args, "evaluate", chash, outs)
    LOG("[evaluate PASS]")


# ------------------------------------------------------------------------------
# stage: diagnostics (Concern 6)
# ------------------------------------------------------------------------------

class PairedOccDataset(Dataset):
    def __init__(self, v4, paths: List[Path], occ: float):
        self.c = v4.OccludedPathsDataset(paths, sz=322, occ=0.0, seed=42, idx_offset=0)
        self.o = v4.OccludedPathsDataset(paths, sz=322, occ=occ, seed=42, idx_offset=0)

    def __len__(self) -> int:
        return len(self.c)

    def __getitem__(self, i: int):
        return self.c[i], self.o[i], i


def boot_mean(x: np.ndarray, seed: int = 42, n_boot: int = 2000) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": round(float(x.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(x))}


def stage_diagnostics(args, mods) -> None:
    v4, v5 = mods
    chash = config_hash(args, ["seeds", "epochs", "disrupted_threshold"])
    if marker_valid(args, "diagnostics", chash):
        LOG("[diagnostics] complete (marker valid); skipping")
        return
    db, q, pos = v4.restore_and_verify_msls_val(args.msls_manifest, args.msls_tar)
    backbone = v4.FrozenOfficialSALADBackbone(args.hub_cache)
    specs = [s for s in all_specs(args) if s.is_archived]
    models = {s.key: load_head(args, s, v4) for s in specs}
    N = 529
    arrays: Dict[str, np.ndarray] = {}
    names = ("occluder_share_salient_mass", "retained_salient_mass_ratio", "n_disrupted",
             "disrupted_absorption", "disrupted_retained_mass_ratio", "disrupted_weight_share",
             "disrupted_cluster_fraction", "frac_disrupted_absorbed_ge_half", "prop1_min_slack")
    for occ in (0.25, 0.50):
        tag = f"occ{int(occ * 100):02d}"
        acc = {k: {n: np.full(len(q), np.nan) for n in names} for k in models}
        loader = DataLoader(PairedOccDataset(v4, q, occ), batch_size=args.batch_size, num_workers=args.workers)
        t0 = time.time()
        for clean, occl, idx in loader:
            idx = idx.numpy()
            O = torch.as_tensor(np.stack([H6.occluder_token_mask(int(i), occ, seed=42).reshape(-1) for i in idx]),
                                device=DEVICE, dtype=torch.float32)  # (B, N) 1 = occluder token
            lat_c = v5.extract_latents_with_f_raw(backbone, v4.compute_boundary_immune_cues, clean)
            lat_o = v5.extract_latents_with_f_raw(backbone, v4.compute_boundary_immune_cues, occl)
            with torch.inference_mode():
                for k, mod in models.items():
                    _, _, ac = mod(lat_c["f_local"], lat_c["f_raw"], lat_c["f_cls"], lat_c["raw_logits"], return_aux=True)
                    _, _, ao = mod(lat_o["f_local"], lat_o["f_raw"], lat_o["f_cls"], lat_o["raw_logits"], return_aux=True)
                    Pc, Po = ac["P_sal"], ao["P_sal"]                       # (B, N, M)
                    mc, mo = Pc.sum(1), Po.sum(1)                           # (B, M)
                    occ_mass_o = (Po * O[:, :, None]).sum(1)                # occluder supply per cluster
                    vis_mass_o = mo - occ_mass_o
                    clean_from_masked = (Pc * O[:, :, None]).sum(1) / (mc + 1e-12)
                    disrupted = (clean_from_masked >= args.disrupted_threshold).float()
                    nd = disrupted.sum(1)
                    absorb = occ_mass_o / (mo + 1e-12)
                    w = ao["v_agg"].square().sum(-1)
                    wshare = w / (w.sum(1, keepdim=True) + 1e-12)
                    def dmean(x):
                        return torch.where(nd > 0, (x * disrupted).sum(1) / nd.clamp_min(1), torch.full_like(nd, float("nan")))
                    vals = {
                        "occluder_share_salient_mass": occ_mass_o.sum(1) / (mo.sum(1) + 1e-12),
                        "retained_salient_mass_ratio": mo.sum(1) / (mc.sum(1) + 1e-12),
                        "n_disrupted": nd,
                        "disrupted_absorption": dmean(absorb),
                        "disrupted_retained_mass_ratio": dmean(mo / (mc + 1e-12)),
                        "disrupted_weight_share": (wshare * disrupted).sum(1),
                        "disrupted_cluster_fraction": nd / float(mo.shape[1]),
                        "frac_disrupted_absorbed_ge_half": dmean((absorb >= 0.5).float()),
                        "prop1_min_slack": (occ_mass_o - (1.0 / N - vis_mass_o)).min(1).values,
                    }
                    for n, v in vals.items():
                        acc[k][n][idx] = v.float().cpu().numpy()
        LOG(f"[diagnostics] {tag}: {len(q)} queries x {len(models)} heads ({time.time() - t0:.1f}s)")
        for k in models:
            for n in names:
                arrays[f"{k}_{tag}_{n}"] = acc[k][n]

    # Summaries on three-seed per-query means
    summary: Dict[str, Any] = {"definition": {
        "disrupted_cluster": f"cluster whose clean-image mass comes >= {args.disrupted_threshold:.2f} from tokens that the mask covers",
        "absorption": "share of the occluded-image cluster mass supplied by occluder tokens",
        "weight_share": "share of the squared norm of the (gated) cluster block carried by disrupted clusters",
        "prop1_min_slack": "min_j [occluder supply - (1/N - visible supply)]; >= 0 for balanced columns (Proposition 1)",
    }}
    for setting in ("U", "S"):
        for tag in ("occ25", "occ50"):
            rec = {}
            for (t, g) in H6.LEGACY_MODE:
                keys = [H6.HeadSpec(setting, t, g, s).key for s in args.seeds]
                arm = {}
                for n in names:
                    m = np.nanmean(np.stack([arrays[f"{k}_{tag}_{n}"] for k in keys]), axis=0) if n != "prop1_min_slack" \
                        else np.min(np.stack([arrays[f"{k}_{tag}_{n}"] for k in keys]), axis=0)
                    arm[n] = boot_mean(m) if n != "prop1_min_slack" else {"min": float(np.nanmin(m))}
                rec[f"{t}_{g}"] = arm
            # paired contrasts (same queries): relaxed minus balanced columns, ungated arms
            def m3(t, g, n):
                return np.nanmean(np.stack([arrays[f"{H6.HeadSpec(setting, t, g, s).key}_{tag}_{n}"] for s in args.seeds]), axis=0)
            for n in ("disrupted_absorption", "occluder_share_salient_mass", "disrupted_retained_mass_ratio"):
                d = m3("uot", "none", n) - m3("bal", "none", n)
                rec[f"uot_minus_bal_{n}"] = boot_mean(d)
            d = m3("uot", "combined", "disrupted_weight_share") - m3("uot", "none", "disrupted_weight_share")
            rec["gate_effect_on_disrupted_weight_share_uot"] = boot_mean(d)
            summary[f"{setting}_{tag}"] = rec
    arr_path = args.out_dir / "results" / "diagnostics_arrays.npz"
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = arr_path.with_name(arr_path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, arr_path)
    js = args.out_dir / "results" / "diagnostics.json"
    atomic_json(js, {"summary": summary})
    write_marker(args, "diagnostics", chash, [arr_path, js])
    LOG("[diagnostics PASS]")


# ------------------------------------------------------------------------------
# stage: groups (Concern 4)
# ------------------------------------------------------------------------------

def stage_groups(args, mods) -> None:
    v4, _ = mods
    chash = config_hash(args, ["msls_metadata_dir"])
    if marker_valid(args, "groups", chash):
        LOG("[groups] complete (marker valid); skipping")
        return
    man = json.loads(args.msls_manifest.read_text())
    db_paths, q_paths, positives = v4._msls_manifest_arrays(man)
    nq = len(q_paths)
    cities = []
    for p in q_paths:
        m = re.search(r"/(?:train_val|test)/([^/]+)/", p.as_posix())
        cities.append(m.group(1) if m else "unknown")
    city_names = sorted(set(cities))
    city_ids = [city_names.index(c) for c in cities]

    # union-find over queries sharing at least one positive gallery image
    parent = list(range(nq))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner: Dict[int, int] = {}
    for qi, pl in enumerate(positives):
        for g in pl:
            if g in owner:
                ra, rb = find(qi), find(owner[g])
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[g] = qi
    roots = [find(i) for i in range(nq)]
    rmap = {r: i for i, r in enumerate(sorted(set(roots)))}
    comp = [rmap[r] for r in roots]

    # MSLS sequence keys if seq_info.csv files are available
    seq_ids = None
    cand_dirs = []
    try:
        root = v4._msls_image_root(db_paths + q_paths)
        cand_dirs.append(root / "train_val")
    except Exception:
        pass
    if args.msls_metadata_dir:
        cand_dirs.append(args.msls_metadata_dir)
    key2seq: Dict[str, str] = {}
    found_files = []
    for base in cand_dirs:
        for f in sorted(Path(base).glob("*/query/seq_info.csv")) if Path(base).is_dir() else []:
            found_files.append(str(f))
            lines = f.read_text().splitlines()
            hdr = [h.strip() for h in lines[0].split(",")]
            if "key" in hdr and "sequence_key" in hdr:
                ki, si = hdr.index("key"), hdr.index("sequence_key")
                for ln in lines[1:]:
                    parts = ln.split(",")
                    if len(parts) > max(ki, si):
                        key2seq[parts[ki].strip()] = f"{f.parent.parent.name}/{parts[si].strip()}"
    if key2seq:
        seqs = [key2seq.get(Path(p).stem) for p in q_paths]
        if all(s is not None for s in seqs):
            names = sorted(set(seqs))
            seq_ids = [names.index(s) for s in seqs]
        else:
            LOG(f"[groups] seq_info.csv found but {sum(s is None for s in seqs)} queries lack a sequence key; not used")
    sizes = np.bincount(comp)
    out = {
        "n_queries": nq,
        "cities": city_names,
        "group_ids": {"city": city_ids, "positive_components": comp, "sequence": seq_ids},
        "summary": {
            "n_cities": len(city_names),
            "n_positive_components": int(len(sizes)), "largest_component": int(sizes.max()),
            "median_component": float(np.median(sizes)),
            "n_sequences": None if seq_ids is None else int(len(set(seq_ids))),
            "seq_info_files": found_files,
        },
        "note": "positive_components joins queries that share any ground-truth gallery image (25 m); a conservative proxy for route/location dependence.",
    }
    p = args.out_dir / "results" / "msls_val_query_groups.json"
    atomic_json(p, out)
    write_marker(args, "groups", chash, [p])
    LOG(f"[groups PASS] cities={len(city_names)} components={len(sizes)} (largest {sizes.max()}) "
        f"sequences={'n/a' if seq_ids is None else len(set(seq_ids))}")


# ------------------------------------------------------------------------------
# stage: analysis and package
# ------------------------------------------------------------------------------

def stage_analysis(args, mods) -> None:
    Hs = A6.HitStore()
    n = Hs.add_v6_dir(args.out_dir / "hits_v6")
    LOG(f"[analysis] {n} v6 hit arrays")
    groups = None
    gp = args.out_dir / "results" / "msls_val_query_groups.json"
    if gp.is_file():
        g = json.loads(gp.read_text())
        groups = {k: np.asarray(v, dtype=np.int64) for k, v in g["group_ids"].items() if v is not None and k != "city"}
    res = A6.analyse(Hs, groups)
    dp = args.out_dir / "results" / "diagnostics.json"
    if dp.is_file():
        res["diagnostics"] = json.loads(dp.read_text())
    out = args.out_dir / "results" / "v6_statistics.json"
    atomic_json(out, res)
    (out.with_suffix(".md")).write_text(A6.to_markdown(res), encoding="utf-8")
    LOG(f"[analysis PASS] {out}")


def stage_package(args, mods) -> None:
    zpath = args.out_dir / "EUOT_VPR_v6_Results.zip"
    members = []
    for sub, pat in (("hits_v6", "*.npy"), ("checkpoints_v6", "*.pt"), ("results", "*"), ("markers", "*.json"), ("logs", "*.log"),
                     ("extra", "**/*")):
        base = args.out_dir / sub
        if base.is_dir():
            members += [p for p in sorted(base.glob(pat)) if p.is_file() and not p.name.endswith(".tmp")]
    manifest = {str(p.relative_to(args.out_dir)): sha256_file(p) for p in members}
    tmp = zpath.with_name(zpath.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            zf.write(p, str(p.relative_to(args.out_dir)))
        for kp in (KIT_DIR / "euot_v6_kit.py", KIT_DIR / "euot_v6_heads.py", KIT_DIR / "euot_v6_analysis.py"):
            zf.write(kp, f"kit_source/{kp.name}")
        zf.writestr("SHA256_MANIFEST.json", json.dumps(manifest, indent=2))
    os.replace(tmp, zpath)
    LOG(f"[package PASS] {len(members)} files -> {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


# ------------------------------------------------------------------------------
# stage: extra dataset (optional, Concern 4)
# ------------------------------------------------------------------------------

def extract_with_official(backbone, cues_fn, imgs):
    """v5.1 extract_latents_with_f_raw plus the released SALAD descriptor from the same pass."""
    x = (imgs.to(DEVICE, non_blocking=True) - backbone.mean) / backbone.std
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda"), dtype=torch.float16):
        x_feat, t_cls = backbone.salad.backbone(x)
        agg = backbone.salad.aggregator
        off = F.normalize(agg((x_feat, t_cls)).float(), p=2, dim=-1)
        B, _, H_p, W_p = x_feat.shape
        N = H_p * W_p
        f_raw = agg.cluster_features(x_feat).float().flatten(2).transpose(1, 2).contiguous()
        f_local = F.normalize(f_raw, p=2, dim=-1)
        s_sal = agg.score(x_feat).float().flatten(2).transpose(1, 2).contiguous()
        dust = agg.dust_bin.float().expand(B, N, 1)
        raw_logits = torch.cat([s_sal, dust], dim=-1)
        f_cls = F.normalize(agg.token_features(t_cls).float(), p=2, dim=-1)
    return {"f_local": f_local, "f_raw": f_raw, "f_cls": f_cls, "raw_logits": raw_logits, "official": off}


def stage_extra(args, mods) -> None:
    v4, v5 = mods
    if not args.extra_manifest:
        LOG("[extra] no --extra_manifest given; skipping")
        return
    man = json.loads(Path(args.extra_manifest).read_text())
    name = re.sub(r"[^A-Za-z0-9]+", "", man.get("name", args.extra_name or "extra")) or "extra"
    dbp = [Path(x) for x in man["database"]]
    qp = [Path(x) for x in man["queries"]]
    pos = [[int(i) for i in p] for p in man["positives"]]
    missing = [p for p in dbp + qp if not p.is_file()]
    if missing or len(pos) != len(qp):
        raise SystemExit(f"[extra FAIL] {len(missing)} missing images (first {missing[:1]}) or positives misaligned")
    outd = args.out_dir / "extra" / name
    specs = [s for s in all_specs(args) if s.legacy_mode is not None]  # standard cells of all settings
    backbone = v4.FrozenOfficialSALADBackbone(args.hub_cache)
    LOG(f"[extra] {name}: {len(dbp)} database, {len(qp)} queries, {len(specs)} heads + released SALAD")
    groups = [specs[i:i + args.group_size] for i in range(0, len(specs), args.group_size)]
    for gi, grp in enumerate(groups, 1):
        exp = [outd / f"{name}_v6_{s.key}_occ{int(o * 100):02d}_hits.npy" for s in grp for o in args.extra_occ]
        if all(p.is_file() for p in exp):
            LOG(f"[extra] group {gi} complete; skipping")
            continue
        models = {s.key: load_head(args, s, v4) for s in grp}

        def enc(paths, occ):
            res = {k: [] for k in list(models) + ["official_salad"]}
            ds = v4.OccludedPathsDataset(paths, sz=322, occ=occ, seed=42, idx_offset=0)
            for i, imgs in enumerate(DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)):
                lat = extract_with_official(backbone, v4.compute_boundary_immune_cues, imgs)
                res["official_salad"].append(lat["official"].cpu().half().numpy())
                with torch.inference_mode():
                    for k, mod in models.items():
                        d, _ = mod(lat["f_local"], lat["f_raw"], lat["f_cls"], lat["raw_logits"])
                        res[k].append(d.cpu().half().numpy())
                if i % 20 == 0:
                    LOG(f"[extra] group {gi}: {min(len(paths), (i + 1) * args.batch_size):,}/{len(paths):,}")
            return {k: np.concatenate(v) for k, v in res.items()}

        dbd = enc(dbp, 0.0)
        for occ in args.extra_occ:
            tag = f"occ{int(occ * 100):02d}"
            qd = enc(qp, occ)
            for k in list(models) + ["official_salad"]:
                res, hits = v4.recall_from_embeddings(dbd[k], qd[k], pos)
                atomic_npy(outd / f"{name}_v6_{k}_{tag}_hits.npy", hits)
                LOG(f"[extra-result] {name} {k:<26s} {tag}: R@1={res['R@1']:6.2f}")
    Hs = A6.HitStore()
    for p in sorted(outd.glob(f"{name}_v6_*_hits.npy")):
        stem = p.name[len(f"{name}_v6_"):-len("_hits.npy")].split("_")
        if len(stem) == 5:
            Hs.h[(stem[0], stem[1], stem[2], int(stem[3][1:]), stem[4])] = np.load(p)
    res = A6.analyse(Hs, None)
    sal = {}
    for p in sorted(outd.glob(f"{name}_v6_official_salad_*_hits.npy")):
        sal[p.stem.split("_")[-2]] = round(float(np.load(p)[:, 0].mean() * 100), 2)
    res["released_salad_r1_pct"] = sal
    atomic_json(outd / f"{name}_statistics.json", res)
    LOG(f"[extra PASS] {outd}")


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EUOT-VPR crossed-design runner (GPU)")
    ap.add_argument("--stage", default="all",
                    choices=["preflight", "train", "evaluate", "diagnostics", "groups", "analysis", "package", "extra", "all"])
    ap.add_argument("--msls_manifest", type=Path, default=DRIVE / "VPR_Datasets/msls_manifest.json")
    ap.add_argument("--msls_tar", type=Path, default=DRIVE / "MPEF_VPR_New_Project/protected_data/msls/compact_val_v1/msls_val_ready_v1.tar")
    ap.add_argument("--msls_metadata_dir", type=Path, default=None, help="optional folder with <city>/query/seq_info.csv")
    ap.add_argument("--hub_cache", type=Path, default=EUOT / "torch_hub_shared")
    ap.add_argument("--v4_dir", type=Path, default=EUOT / "euot_msls_matched_salad_diagnostic",
                    help="audited run folder (read only)")
    ap.add_argument("--archived_ckpt_u", type=Path, default=None)
    ap.add_argument("--archived_ckpt_s", type=Path, default=None)
    ap.add_argument("--archived_hits_u", type=Path, default=None)
    ap.add_argument("--archived_hits_s", type=Path, default=None)
    ap.add_argument("--train_latents", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=EUOT / "euot_msls_v6_run")
    ap.add_argument("--chunk_dir", type=Path, default=Path("outputs/euot_v6_chunk_cache"),
                    help="local scratch for descriptor chunks (deleted per group after its hits are saved)")
    ap.add_argument("--keep_chunk_cache", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 108, 2026])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=12)
    ap.add_argument("--crossed_settings", nargs="*", default=["UW", "RE"])
    ap.add_argument("--gate_variant_settings", nargs="*", default=["U", "S"])
    ap.add_argument("--gate_variants", nargs="*", default=["reliability", "length", "mass", "meannorm"])
    ap.add_argument("--disrupted_threshold", type=float, default=0.5)
    ap.add_argument("--repro_min_agreement", type=float, default=0.985)
    ap.add_argument("--replay_check", action="store_true", help="retrain two archived cells and report weight differences")
    ap.add_argument("--require_gpu", action="store_true", default=True)
    ap.add_argument("--extra_manifest", type=Path, default=None)
    ap.add_argument("--extra_name", default=None)
    ap.add_argument("--extra_occ", type=float, nargs="+", default=[0.0])
    return ap


def resolve_defaults(args) -> None:
    a = args
    a.archived_ckpt_u = a.archived_ckpt_u or a.v4_dir / "checkpoints_v4"
    a.archived_ckpt_s = a.archived_ckpt_s or a.v4_dir / "checkpoints_v5_raw_1to64cls"
    a.archived_hits_u = a.archived_hits_u or a.v4_dir
    a.archived_hits_s = a.archived_hits_s or a.v4_dir / "hits_v5_raw_1to64cls"
    a.train_latents = a.train_latents or a.v4_dir / "gardens_train_150_latents"
    for s in a.crossed_settings + a.gate_variant_settings:
        if s not in H6.SETTINGS:
            raise SystemExit(f"unknown setting {s}")
    for g in a.gate_variants:
        if g not in H6.GATES or g in ("none", "combined"):
            raise SystemExit(f"invalid gate variant {g}")
    if a.out_dir.resolve() == a.v4_dir.resolve():
        raise SystemExit("--out_dir must differ from the audited run folder")


def main(argv: Optional[Sequence[str]] = None) -> None:
    global LOG
    args = build_parser().parse_args(argv)
    resolve_defaults(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    LOG = Logger(args.out_dir / "logs" / "kit.log")
    LOG(f"EUOT-VPR crossed-design run {KIT_VERSION} | stage={args.stage} | device={DEVICE} | out={args.out_dir}")
    mods = (load_module("v4_mod", KIT_DIR / V4_NAME, V4_SHA), load_module("v5_mod", KIT_DIR / V5_NAME, V5_SHA))
    LOG("[modules] audited v4.1 and v5.1 runners loaded (sha256 verified)")
    order = ["preflight", "train", "evaluate", "diagnostics", "groups", "analysis", "package"]
    stages = order if args.stage == "all" else [args.stage]
    fns = {"preflight": stage_preflight, "train": stage_train, "evaluate": stage_evaluate, "diagnostics": stage_diagnostics,
           "groups": stage_groups, "analysis": stage_analysis, "package": stage_package, "extra": stage_extra}
    for st in stages:
        LOG(f"===== stage {st} =====")
        fns[st](args, mods)
    if args.stage == "all" and args.extra_manifest:
        stage_extra(args, mods)
        stage_package(args, mods)
    LOG("DONE")


if __name__ == "__main__":
    main()
