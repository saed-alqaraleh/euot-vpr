#!/usr/bin/env python3
"""CPU tests for the EUOT-VPR v6 model suite (no GPU, no MSLS images needed).

    python3 tests/test_kit_cpu.py            # from the kit folder
    EUOT_SUPP_DIR=/path/to/supplementary python3 tests/test_kit_cpu.py   # + archived-checkpoint tests

Covers: equivalence of the generalized head with the audited v4.1/v5.1 heads
(initialization, forward pass with archived checkpoints, training loop), gate
identities, the token-mask replica of the audited occlusion dataset, the
statistics module, and the kit's head plan.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# The equivalence tests compare bit-exact CPU results; hide any GPU from torch and from
# the audited runners (whose DEVICE would otherwise be CUDA in Colab).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import euot_v6_heads as H6  # noqa: E402
import euot_v6_analysis as A6  # noqa: E402

SUPP = Path(os.environ.get("EUOT_SUPP_DIR", KIT.parents[1]))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(KIT / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4 = _load("v4_mod", "EUOT_VPR_MSLS_Matched_Backbone_Final_Diagnostic.py")
V5 = _load("v5_mod", "EUOT_VPR_MSLS_v5_Retrain_RawTokens_1to64CLS.py")
CUES = V4.compute_boundary_immune_cues
MODES = ("rgtmp_bal", "rgtmp_bal_gate", "euot_no_gate", "euot_1stage")
INV = {v: k for k, v in H6.LEGACY_MODE.items()}


def synthetic_latents(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_raw = torch.randn(B, 529, 128, generator=g) * (0.5 + torch.rand(B, 529, 1, generator=g) * 3)
    f_local = F.normalize(f_raw, p=2, dim=-1)
    f_cls = F.normalize(torch.randn(B, 256, generator=g), p=2, dim=-1)
    raw_logits = torch.randn(B, 529, 65, generator=g) * 2.0
    cues = CUES(f_local, raw_logits)
    return f_local, f_raw, f_cls, raw_logits, cues


def spec_for(setting, mode, seed=42):
    t, g = INV[mode]
    return H6.HeadSpec(setting, t, g, seed)


class TestHeadEquivalence(unittest.TestCase):
    def test_initialization_identical(self):
        for mode in MODES:
            for setting, ref_cls in (("U", lambda m: V4.MatchedSALADFactorialHead(mode=m)),
                                     ("S", lambda m: V5.MatchedSALADRawToken1to64Head(mode=m))):
                V4.set_controlled_seed(42)
                ref = ref_cls(mode)
                V4.set_controlled_seed(42)
                new = H6.GeneralizedFactorialHead.from_spec(spec_for(setting, mode), CUES)
                a, b = ref.state_dict(), new.state_dict()
                self.assertEqual(set(a), set(b))
                for k in a:
                    self.assertTrue(torch.equal(a[k], b[k]), f"{setting}/{mode}/{k}")

    def test_forward_identical_random_weights(self):
        f_local, f_raw, f_cls, raw_logits, cues = synthetic_latents()
        for mode in MODES:
            torch.manual_seed(7)
            ref4 = V4.MatchedSALADFactorialHead(mode=mode)
            with torch.no_grad():
                for p in ref4.parameters():
                    p.add_(torch.randn_like(p) * 0.3)
            new = H6.GeneralizedFactorialHead.from_spec(spec_for("U", mode), CUES)
            new.load_state_dict(ref4.state_dict(), strict=True)
            ref5 = V5.MatchedSALADRawToken1to64Head(mode=mode)
            ref5.load_state_dict(ref4.state_dict(), strict=True)
            news = H6.GeneralizedFactorialHead.from_spec(spec_for("S", mode), CUES)
            news.load_state_dict(ref4.state_dict(), strict=True)
            with torch.no_grad():
                d4, r4 = ref4(f_local, f_cls, raw_logits, cues)
                d6, r6 = new(f_local, f_raw, f_cls, raw_logits)
                d5, r5 = ref5(f_local, f_raw, f_cls, raw_logits, cues, CUES)
                d6s, _ = news(f_local, f_raw, f_cls, raw_logits)
            self.assertTrue(torch.equal(d4, d6), f"U/{mode} max diff {(d4 - d6).abs().max()}")
            self.assertTrue(torch.equal(r4, r6))
            self.assertTrue(torch.equal(d5, d6s), f"S/{mode} max diff {(d5 - d6s).abs().max()}")

    @unittest.skipUnless((SUPP / "msls_setting_U" / "checkpoints_v4").is_dir(), "archived checkpoints not available")
    def test_archived_checkpoints_load_and_match(self):
        f_local, f_raw, f_cls, raw_logits, cues = synthetic_latents(B=2, seed=3)
        for s in (42, 108, 2026):
            for mode in MODES:
                for setting, folder, suffix in (("U", "msls_setting_U/checkpoints_v4", "v4"), ("S", "msls_setting_S/checkpoints_v5", "v5")):
                    ck = torch.load(SUPP / folder / f"{mode}_seed{s}_{suffix}.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(ck["done_epochs"], 12)
                    new = H6.GeneralizedFactorialHead.from_spec(spec_for(setting, mode, s), CUES)
                    new.load_state_dict(ck["weights"], strict=True)
                    ref = V4.MatchedSALADFactorialHead(mode=mode) if setting == "U" else V5.MatchedSALADRawToken1to64Head(mode=mode)
                    ref.load_state_dict(ck["weights"], strict=True)
                    with torch.no_grad():
                        d_new, _ = new(f_local, f_raw, f_cls, raw_logits)
                        d_ref, _ = (ref(f_local, f_cls, raw_logits, cues) if setting == "U"
                                    else ref(f_local, f_raw, f_cls, raw_logits, cues, CUES))
                    self.assertTrue(torch.equal(d_new, d_ref), f"{setting}/{mode}/s{s}")

    def test_training_loop_identical(self):
        """v6 train_head reproduces the audited v4.1 and v5.1 training loops weight for weight."""
        torch.manual_seed(0)
        n = 8
        cluster_mlp = nn.Conv2d(768, 128, kernel_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files, fields_u, fields_s = {}, {}, {}
            for i, t in enumerate(("db", "q00", "q25", "q50")):
                g = torch.Generator().manual_seed(100 + i)
                x768 = torch.randn(n, 529, 768, generator=g) * 0.2
                f_raw_eval = V5.compute_f_raw_from_cached_768(cluster_mlp, x768.numpy())
                f_local = F.normalize(torch.as_tensor(f_raw_eval), dim=-1).numpy().astype(np.float16)
                raw_logits = (torch.randn(n, 529, 65, generator=g) * 2).numpy().astype(np.float16)
                f_cls = F.normalize(torch.randn(n, 256, generator=g), dim=-1).numpy().astype(np.float16)
                cues = CUES(torch.as_tensor(f_local.astype(np.float32)), torch.as_tensor(raw_logits.astype(np.float32))).numpy()
                np.savez(tmp / f"{t}.npz", f_local=f_local, f_cls=f_cls, raw_logits=raw_logits, cues_3ch=cues,
                         x_patch_768=x768.numpy().astype(np.float16))
                files[t] = tmp / f"{t}.npz"
                with np.load(files[t]) as z:
                    base = {k: z[k].astype(np.float32) for k in ("f_local", "f_cls", "raw_logits")}
                    fields_u[t] = dict(base)
                    fields_s[t] = dict(base, f_raw=V5.compute_f_raw_from_cached_768(cluster_mlp, z["x_patch_768"].astype(np.float32)))
            ref_u = V4.train_matched_four_factorial_heads(files, tmp / "ck4", seed=42, epochs=2, batch_size=32)
            ref_s = V5.train_v5_four_factorial_heads(V4, files, cluster_mlp, tmp / "ck5", seed=42, epochs=2, batch_size=32)
            for mode in MODES:
                for setting, ref, fields in (("U", ref_u, fields_u), ("S", ref_s, fields_s)):
                    new = H6.train_head(spec_for(setting, mode), fields, CUES, V4.set_controlled_seed, torch.device("cpu"),
                                        epochs=2, batch_size=32, log=lambda m: None)
                    a, b = ref[mode].state_dict(), new.state_dict()
                    for k in a:
                        self.assertTrue(torch.equal(a[k], b[k]), f"{setting}/{mode}/{k} max {(a[k] - b[k]).abs().max()}")


class TestGateIdentities(unittest.TestCase):
    def head(self, transport, gate, token_source="unit", scale=1.0):
        torch.manual_seed(11)
        h = H6.GeneralizedFactorialHead(transport, gate, token_source, scale, CUES)
        with torch.no_grad():
            for p in h.parameters():
                p.add_(torch.randn_like(p) * 0.2)
        return h.eval()

    def test_gate_algebra(self):
        f_local, f_raw, f_cls, raw_logits, _ = synthetic_latents(B=2, seed=5)
        for transport in ("bal", "uot"):
            for src, toks in (("unit", f_local), ("raw", f_raw)):
                auxs = {}
                for g in H6.GATES:
                    h = self.head(transport, g, src)
                    with torch.no_grad():
                        _, _, aux = h(f_local, f_raw, f_cls, raw_logits, return_aux=True)
                    auxs[g] = aux
                P = auxs["none"]["P_sal"]
                v_raw = torch.einsum("bnm,bnd->bmd", P, toks)
                psi = auxs["none"]["psi"]
                m = P.sum(1)
                torch.testing.assert_close(auxs["combined"]["v_agg"], psi[..., None] * v_raw, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(auxs["length"]["v_agg"], v_raw, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(auxs["mass"]["v_agg"], m[..., None] * F.normalize(v_raw, dim=-1), rtol=1e-5, atol=1e-9)
                torch.testing.assert_close(auxs["meannorm"]["v_agg"] * m[..., None], v_raw, rtol=1e-4, atol=1e-7)
                if transport == "bal":
                    torch.testing.assert_close(m, torch.full_like(m, 1.0 / 529), rtol=1e-4, atol=1e-8)

    def test_mass_gate_is_noop_under_balanced_columns(self):
        f_local, f_raw, f_cls, raw_logits, _ = synthetic_latents(B=2, seed=6)
        for src in ("unit", "raw"):
            for scale in (1.0, 8.0):
                with torch.no_grad():
                    d0, _ = self.head("bal", "none", src, scale)(f_local, f_raw, f_cls, raw_logits)
                    dm, _ = self.head("bal", "mass", src, scale)(f_local, f_raw, f_cls, raw_logits)
                    dl, _ = self.head("bal", "length", src, scale)(f_local, f_raw, f_cls, raw_logits)
                    dn, _ = self.head("bal", "meannorm", src, scale)(f_local, f_raw, f_cls, raw_logits)
                torch.testing.assert_close(d0, dm, rtol=1e-4, atol=1e-6)
                torch.testing.assert_close(dl, dn, rtol=1e-4, atol=1e-6)

    def test_global_share(self):
        """The global token carries a share 1/(1+scale^2) of the squared descriptor norm."""
        f_local, f_raw, f_cls, raw_logits, _ = synthetic_latents(B=2, seed=8)
        for scale, share in ((1.0, 0.5), (8.0, 1.0 / 65.0)):
            with torch.no_grad():
                d, _ = self.head("uot", "combined", "unit", scale)(f_local, f_raw, f_cls, raw_logits)
            g = d[:, -256:].square().sum(1)
            torch.testing.assert_close(g, torch.full_like(g, share), rtol=1e-5, atol=1e-7)


class TestMaskReplica(unittest.TestCase):
    def test_token_mask_matches_dataset(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "black.png"
            Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)).save(p)
            for occ in (0.25, 0.5):
                for offset in (0, 2048):
                    ds = V4.OccludedPathsDataset([p] * 12, sz=322, occ=occ, seed=42, idx_offset=offset)
                    for i in range(12):
                        x = ds[i]
                        tok = (x != 0).any(dim=0).numpy().reshape(23, 14, 23, 14).any(axis=(1, 3))
                        rec = H6.occluder_token_mask(offset + i, occ, seed=42)
                        self.assertTrue(np.array_equal(tok, rec), f"occ={occ} idx={offset + i}")
                        self.assertGreaterEqual(int(rec.sum()), int(round(occ * 529)))  # rectangles may overshoot the target


class TestAnalysis(unittest.TestCase):
    def store(self, rng, n=740, effect=0.0):
        H = A6.HitStore()
        for setting in ("U", "S", "UW", "RE"):
            for (t, g) in A6.LEGACY:
                for s in A6.SEEDS:
                    for occ in A6.OCCS:
                        p = 0.7 + (effect if (t, g) == ("uot", "combined") and setting == "S" else 0.0)
                        H.h[(setting, t, g, s, occ)] = (rng.random((n, 3)) < p).astype(np.uint8)
        return H

    def test_did_and_holm(self):
        rng = np.random.default_rng(1)
        H = self.store(rng, effect=0.15)
        res = A6.analyse(H)
        d = res["per_setting_factorial"]["S"]["occ50"]["interaction_did"]["diff_pp"]
        self.assertGreater(d, 8.0)
        fam = res["familyA_crossed_design"]["occ50"]
        self.assertIn("p_holm_familyA", fam["normalization_main_effect"])
        self.assertEqual(A6.holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "c": 0.06, "b": 0.06})

    def test_grouped_singletons_match_estimate(self):
        rng = np.random.default_rng(2)
        d = rng.normal(1.0, 5.0, 500)
        a = A6.query_test(d)
        b = A6.grouped_test(d, np.arange(500))
        self.assertEqual(a["diff_pp"], b["diff_pp"])
        self.assertLess(abs(a["ci95"][0] - b["ci95"][0]), 0.25)
        c = A6.grouped_test(d, np.arange(500) // 10)
        self.assertEqual(c["n_units"], 50)

    @unittest.skipUnless((SUPP / "msls_setting_U" / "hit_arrays_v4").is_dir(), "archived hit arrays not available")
    def test_reproduces_paper_numbers(self):
        H = A6.HitStore()
        H.add_archived(SUPP / "msls_setting_U" / "hit_arrays_v4", "U", "msls_")
        H.add_archived(SUPP / "msls_setting_S" / "hit_arrays_v5", "S", "msls_v5_")
        r = A6.analyse(H)["per_setting_factorial"]
        self.assertEqual(r["U"]["occ50"]["interaction_did"], {"diff_pp": 2.93, "ci95": [1.17, 4.82], "p": 0.0027, "n_units": 740})
        self.assertEqual(r["S"]["occ50"]["interaction_did"], {"diff_pp": -3.42, "ci95": [-5.86, -0.99], "p": 0.0058, "n_units": 740})
        self.assertEqual(r["S"]["occ25"]["interaction_did"]["diff_pp"], -2.03)


class TestKitPlan(unittest.TestCase):
    def test_head_plan(self):
        kit = _load("kit_mod", "euot_v6_kit.py")
        args = kit.build_parser().parse_args(["--out_dir", "/tmp/euot_v6_test_out", "--v4_dir", "/tmp/euot_v6_fake_v4"])
        kit.resolve_defaults(args)
        specs = kit.all_specs(args)
        self.assertEqual(len(specs), 96)
        self.assertEqual(sum(s.is_archived for s in specs), 24)
        self.assertEqual(len({s.key for s in specs}), 96)
        self.assertEqual(kit.V4_SHA[:8], "e968dff3")
        self.assertEqual(kit.V5_SHA[:8], "3aedd544")
        self.assertTrue(kit.archived_ckpt_path(args, specs[0]).name.endswith("_v4.pt"))
        with self.assertRaises(SystemExit):
            bad = kit.build_parser().parse_args(["--out_dir", "/tmp/x", "--v4_dir", "/tmp/x"])
            kit.resolve_defaults(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
