"""EUOT-VPR v6 generalized MSLS factorial head and helpers.

One head class covers every cell of the experiments requested in the review:

* Setting (token normalization x global-token share)
    U  : unit tokens, global share 50%   (== audited v4.1 head)
    S  : raw tokens,  global share 1/65  (== audited v5.1 head)
    UW : unit tokens, global share 1/65  (new, crossed cell)
    RE : raw tokens,  global share 50%   (new, crossed cell)
* Transport: balanced columns (rho2 = 1) or relaxed columns (learned rho2 < 1).
* Gate applied after intra-cluster normalization, v_j = v~_j / ||v~_j||:
    none        : v_j
    combined    : ||v~_j|| psi_j v_j  (= psi_j v~_j; the audited gate, Eq. 13)
    reliability : psi_j v_j
    length      : ||v~_j|| v_j        (= v~_j; intra-normalization removed)
    mass        : m_j v_j             (column mass restored)
    meannorm    : (||v~_j|| / m_j) v_j  (norm of the mass-weighted mean token)
  with m_j = sum_i P_ij and psi_j = rbar_j^2 / (rbar_j^2 + 0.18^2).
  Identities: length = mass * meannorm; combined = reliability * length.
  Under balanced columns m_j = 1/N for every cluster, so `mass` is a no-op and
  `length` equals `meannorm` up to a constant that the final normalization removes.

Every parameter, initialization, transport iteration, reliability network and
descriptor operation is copied from the audited v4.1/v5.1 heads; the state-dict
keys are identical, so archived checkpoints load with strict=True.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SETTINGS: Dict[str, Dict[str, object]] = {
    "U": {"token_source": "unit", "cluster_scale": 1.0, "label": "unit tokens, global share 50% (Setting U)"},
    "S": {"token_source": "raw", "cluster_scale": 8.0, "label": "raw tokens, global share 1/65 (Setting S)"},
    "UW": {"token_source": "unit", "cluster_scale": 8.0, "label": "unit tokens, global share 1/65 (crossed)"},
    "RE": {"token_source": "raw", "cluster_scale": 1.0, "label": "raw tokens, global share 50% (crossed)"},
}
TRANSPORTS = ("bal", "uot")
GATES = ("none", "combined", "reliability", "length", "mass", "meannorm")
LEGACY_MODE = {
    ("bal", "none"): "rgtmp_bal",
    ("bal", "combined"): "rgtmp_bal_gate",
    ("uot", "none"): "euot_no_gate",
    ("uot", "combined"): "euot_1stage",
}
PSI_KAPPA = 0.18


@dataclass(frozen=True)
class HeadSpec:
    setting: str
    transport: str
    gate: str
    seed: int

    def __post_init__(self) -> None:
        if self.setting not in SETTINGS:
            raise ValueError(f"unknown setting {self.setting}")
        if self.transport not in TRANSPORTS:
            raise ValueError(f"unknown transport {self.transport}")
        if self.gate not in GATES:
            raise ValueError(f"unknown gate {self.gate}")

    @property
    def key(self) -> str:
        return f"{self.setting}_{self.transport}_{self.gate}_s{self.seed}"

    @property
    def legacy_mode(self) -> Optional[str]:
        return LEGACY_MODE.get((self.transport, self.gate))

    @property
    def is_archived(self) -> bool:
        """Cells already trained and archived (Settings U and S, standard gates)."""
        return self.setting in ("U", "S") and self.legacy_mode is not None

    @property
    def token_source(self) -> str:
        return str(SETTINGS[self.setting]["token_source"])

    @property
    def cluster_scale(self) -> float:
        return float(SETTINGS[self.setting]["cluster_scale"])


class GeneralizedFactorialHead(nn.Module):
    """Generalization of MatchedSALADFactorialHead (v4.1) and
    MatchedSALADRawToken1to64Head (v5.1). Module creation order, initial values
    and every arithmetic step are kept identical so that (i) archived checkpoints
    load unchanged and (ii) a fresh head consumes the RNG exactly as before."""

    def __init__(
        self,
        transport: str,
        gate: str,
        token_source: str,
        cluster_scale: float,
        cues_fn: Callable,
        grid_hw: Tuple[int, int] = (23, 23),
    ):
        super().__init__()
        if transport not in TRANSPORTS or gate not in GATES or token_source not in ("unit", "raw"):
            raise ValueError("invalid head configuration")
        self.transport = transport
        self.gate = gate
        self.token_source = token_source
        self.cluster_scale = float(cluster_scale)
        self.cues_fn = cues_fn
        self.grid_hw = grid_hw
        self.N = grid_hw[0] * grid_hw[1]
        self.M = 64

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

    @classmethod
    def from_spec(cls, spec: HeadSpec, cues_fn: Callable) -> "GeneralizedFactorialHead":
        return cls(spec.transport, spec.gate, spec.token_source, spec.cluster_scale, cues_fn)

    # -- identical to v4.1 / v5.1 ------------------------------------------------
    def compute_token_reliability(self, f_local: torch.Tensor, raw_logits: torch.Tensor) -> torch.Tensor:
        B, N, _ = f_local.shape
        H_p, W_p = self.grid_hw
        cues = self.cues_fn(f_local, raw_logits, grid_hw=self.grid_hw)
        logits_std = (raw_logits - raw_logits.mean(dim=-1, keepdim=True)) / (raw_logits.std(dim=-1, keepdim=True) + 1e-6)
        inp = torch.cat([f_local, logits_std, cues], dim=-1)
        h1 = F.gelu(self.rg1(inp)).transpose(1, 2).contiguous().view(B, 32, H_p, W_p)
        h_dw = self.rgconv(h1).flatten(2).transpose(1, 2).contiguous()
        learned_gate = torch.sigmoid(self.rg2(h_dw).squeeze(-1))
        r = cues[:, :, 1] * (0.35 + 0.65 * learned_gate)
        return torch.clamp(r, 0.015, 0.995)

    def rho2_value(self) -> torch.Tensor:
        if self.transport == "bal":
            return torch.ones((), dtype=self.logit_rho2.dtype, device=self.logit_rho2.device)
        return torch.sigmoid(self.logit_rho2).clamp(0.75, 0.98)

    def controlled_transport(self, raw_logits: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B, N, Mp1 = raw_logits.shape
        assert N == self.N and Mp1 == self.M + 1
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
        rho2 = (torch.ones((), dtype=logits.dtype, device=logits.device)
                if self.transport == "bal" else torch.sigmoid(self.logit_rho2).clamp(0.75, 0.98))
        u = torch.zeros((B, N), device=logits.device, dtype=logits.dtype)
        v = torch.zeros((B, self.M + 1), device=logits.device, dtype=logits.dtype)
        for _ in range(6):
            u = rho1 * (log_mu - torch.logsumexp(logits + v[:, None, :], dim=-1))
            v = rho2 * (log_nu - torch.logsumexp(logits + u[:, :, None], dim=1))
        P = torch.exp(logits + u[:, :, None] + v[:, None, :])
        return P[:, :, :self.M]

    # -- gate family ---------------------------------------------------------------
    def aggregate(self, P_sal: torch.Tensor, r: torch.Tensor, tokens: torch.Tensor):
        m_j = P_sal.sum(dim=1)
        v_raw = torch.einsum("bnm,bnd->bmd", P_sal, tokens)
        v_unit = F.normalize(v_raw, p=2, dim=-1)
        r_bar_j = (P_sal * r[:, :, None]).sum(dim=1) / (m_j + 1e-12)
        psi_r = r_bar_j.square() / (r_bar_j.square() + PSI_KAPPA ** 2 + 1e-12)
        length = torch.norm(v_raw, p=2, dim=-1)
        if self.gate == "none":
            v_agg = v_unit
            weight = torch.ones_like(m_j)
        else:
            if self.gate == "combined":
                weight = length * psi_r            # same expression order as v4.1/v5.1
            elif self.gate == "reliability":
                weight = psi_r
            elif self.gate == "length":
                weight = length
            elif self.gate == "mass":
                weight = m_j
            elif self.gate == "meannorm":
                weight = length / (m_j + 1e-12)
            else:  # pragma: no cover - guarded in __init__
                raise ValueError(self.gate)
            v_agg = v_unit * weight[:, :, None]
        return v_agg, {"m_j": m_j, "psi": psi_r, "length": length, "weight": weight, "r_bar": r_bar_j}

    def forward(
        self,
        f_local: torch.Tensor,
        f_raw: Optional[torch.Tensor],
        f_cls: torch.Tensor,
        raw_logits: torch.Tensor,
        return_aux: bool = False,
    ):
        r = self.compute_token_reliability(f_local, raw_logits)
        P_sal = self.controlled_transport(raw_logits, r)
        if self.token_source == "raw":
            if f_raw is None:
                raise ValueError("raw-token head needs f_raw")
            tokens = f_raw
        else:
            tokens = f_local
        v_agg, aux = self.aggregate(P_sal, r, tokens)
        flat = F.normalize(v_agg.flatten(1), p=2, dim=1)
        if self.cluster_scale != 1.0:
            flat = flat * self.cluster_scale
        desc = F.normalize(torch.cat([flat, f_cls], dim=1), p=2, dim=1)
        if return_aux:
            aux["P_sal"] = P_sal
            aux["r"] = r
            aux["v_agg"] = v_agg
            return desc, r, aux
        return desc, r


# ------------------------------------------------------------------------------
# Occluder mask: exact replica of OccludedPathsDataset (v4.1) at token resolution
# ------------------------------------------------------------------------------

def occluder_token_mask(global_index: int, occ: float, seed: int = 42, sz: int = 322) -> np.ndarray:
    """Token-level (23 x 23) mask drawn by OccludedPathsDataset for image
    `global_index` of a split encoded with `idx_offset` chunks. The occluder is
    painted in whole 14 x 14 patches, so the mask is exact at token resolution."""
    h = w = sz // 14
    mask = np.zeros((h, w), dtype=bool)
    if occ <= 0.0:
        return mask
    target = int(round(occ * h * w))
    rng = np.random.default_rng(seed + int(global_index) * 7919)
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
    return mask


# ------------------------------------------------------------------------------
# Training: identical loop to v4.1 / v5.1 (12 epochs, AdamW 3e-3, InfoNCE 0.07)
# ------------------------------------------------------------------------------

def train_head(
    spec: HeadSpec,
    fields: Dict[str, Dict[str, np.ndarray]],
    cues_fn: Callable,
    set_seed_fn: Callable[[int], None],
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 32,
    log: Callable[[str], None] = print,
) -> GeneralizedFactorialHead:
    """`fields[t]` for t in db, q00, q25, q50 holds float32 arrays f_local, f_cls,
    raw_logits and (raw settings) f_raw, exactly as the audited runners load them."""
    db_z, q00_z = fields["db"], fields["q00"]
    q_pools = [fields["q00"], fields["q25"], fields["q50"]]
    n_train = len(db_z["f_local"])
    raw = spec.token_source == "raw"

    def to_dev(z, idx):
        return (
            torch.as_tensor(z["f_local"][idx], device=device),
            torch.as_tensor(z["f_raw"][idx], device=device) if raw else None,
            torch.as_tensor(z["f_cls"][idx], device=device),
            torch.as_tensor(z["raw_logits"][idx], device=device),
        )

    set_seed_fn(spec.seed)
    model = GeneralizedFactorialHead.from_spec(spec, cues_fn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(spec.seed * 1000 + ep)
        perm = rng.permutation(n_train)
        loss_sum, steps = 0.0, 0
        for st in range(0, n_train, batch_size):
            idx = perm[st:st + batch_size]
            if len(idx) < 4:
                continue
            t_db = to_dev(db_z, idx)
            f_clean_tok = to_dev(q00_z, idx)[0]
            d_db, r_db = model(*t_db)
            cues_db = cues_fn(t_db[0], t_db[3])
            batch_loss = 0.25 * F.mse_loss(r_db, cues_db[:, :, 1].detach())
            target = torch.arange(len(idx), device=device)
            for q_z in q_pools:
                t_q = to_dev(q_z, idx)
                d_q, r_q = model(*t_q)
                scores = d_q @ d_db.T
                loss_nce = 0.5 * (F.cross_entropy(scores / 0.07, target) + F.cross_entropy(scores.T / 0.07, target))
                tok_cos = (t_q[0] * f_clean_tok).sum(dim=-1)
                cues_q = cues_fn(t_q[0], t_q[3])
                r_tgt = torch.clamp(torch.sigmoid(12.0 * (tok_cos - 0.65)) * cues_q[:, :, 1], 0.015, 0.990).detach()
                loss_r = F.mse_loss(r_q, r_tgt)
                batch_loss = batch_loss + (loss_nce + 0.80 * loss_r) / 3.0
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += float(batch_loss.item())
            steps += 1
        if (ep + 1) in (1, epochs // 2, epochs):
            log(f"[Train v6] {spec.key:<26s} epoch={ep + 1:2d}/{epochs} loss={loss_sum / max(1, steps):.4f}")
    model.eval()
    return model


def learned_transport_params(model: GeneralizedFactorialHead) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "rho_1": round(float(torch.sigmoid(model.logit_rho1).clamp(0.85, 0.995)), 4),
            "rho_2": 1.0 if model.transport == "bal" else round(float(torch.sigmoid(model.logit_rho2).clamp(0.75, 0.98)), 4),
            "dust_boost": round(float(F.softplus(model.logit_dust_boost).clamp(0.5, 4.0)), 4),
        }
