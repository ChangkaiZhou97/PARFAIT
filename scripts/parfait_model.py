"""
PARFAIT
==================================

Extends v8's 1D abstraction path z(t) to a 2D scale grid z(t, s):

    t ∈ [0, 1] : abstraction scale  (0 = molecular detail, 1 = topographic)
    s ∈ [0, 1] : spatial-smoothing scale (0 = raw spot, 1 = max graph-smoothed)

Endpoints
---------
    z^0_i = encoder(X_i, log_lib)            # micro/molecular   (same as v8)
    z^1_i = f_psi(r_i)                        # macro/topographic (same as v8)

NEW: spatial-smoothing axis applied to the micro endpoint
---------------------------------------------------------
    z^0(s) = (I - α L)^k  z^0    where α, k come from sigmoid(s)

This is graph heat diffusion on the spatial Laplacian L: s = 0 means no
smoothing (raw z^0), s ≈ 1 means strong neighbor-aggregation. The polynomial
form is the Chebyshev approximation of the heat kernel exp(-σ L); we use
a fixed propagation depth K_diff and a learnable per-step coefficient
α(s) = α_max · sigmoid(... ).

2D path
-------
    z(t, s) = z^0(s) + t (z^1 - z^0(s)) + t(1-t) MLP([z^0(s), z^1, t, s])

so z(0, s) = z^0(s), z(1, s) = z^1, and at s=0 we recover v8's path.

Per-embedding learned (t_m, s_m)
--------------------------------
Each pretrained embedding observes z(t_m, s_m) at its OWN scale coordinates:

    E_i^{(m)} ~ StudentT_{nu_m}(A_m z(t_m, s_m) + b_m, sigma_m^2)

Beta prior on each, init from Rayleigh quotient (for t) and from a
"smoothness deviation" statistic (for s) — see compute_smoothness_init.

Generative model
----------------
    z^0_i ~ N(0, I)     [encoder amortized in guide]
    X_i ~ NB(softmax(W_X z^0_i + b_X), theta_g)
    z^1_i = f_psi(r_i)
    z(t, s) as above
    E_i^{(m)} ~ StudentT(A_m z(t_m, s_m) + b_m, sigma_m^2)
    t_m, s_m ~ Beta(...)        [via pyro.factor logit-prior]

Notes
-----
- 1D path latents are recovered when s_m = 0 for all m; the 2D formulation
  adds an explicit spatial-smoothing axis.
- Propagation matrix exp(-σL) is approximated by (I - α L)^K_diff with
  α = α_max · sigmoid(.). This gives O(K_diff) sparse-matrix products per
  forward pass — cheap on a kNN-sparse L.
- For 5000-spot single-slice DLPFC, K_diff = 8 is plenty.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


# =============================================================================
# Reused encoders from v8 (with same architecture)
# =============================================================================
class CountsEncoder(nn.Module):
    """q(z^0 | X, log_lib). Same as v8."""
    def __init__(self, n_genes: int, K: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_genes + 1, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_mu = nn.Linear(hidden, K)
        self.head_logvar = nn.Linear(hidden, K)

    def forward(self, X_raw, log_lib):
        x = torch.log1p(X_raw)
        x = torch.cat([x, log_lib.unsqueeze(-1)], dim=-1)
        h = self.trunk(x)
        return self.head_mu(h), self.head_logvar(h).clamp(-8.0, 4.0)


class LocationEncoder(nn.Module):
    """f_psi: standardized (x, y) -> z^1. Same as v8."""
    def __init__(self, K: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, K),
        )

    def forward(self, coords_std):
        return self.net(coords_std)


# =============================================================================
# NEW: 2D path network (z^0_smooth, z^1, t, s) -> z(t, s)
# =============================================================================
class PathNetwork2D(nn.Module):
    """g_theta(z^0(s), z^1, t, s) returning z(t, s).

    Construction:
        z(t, s) = z^0(s) + t (z^1 - z^0(s)) + t(1-t) MLP([z^0(s), z^1, t, s])

    Endpoint constraints:
        z(0, s) = z^0(s)   for any s
        z(1, s) = z^1      for any s
    """
    def __init__(self, K: int, hidden: int = 64):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(2 * K + 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, K),
        )
        # init final layer to small so initial path is approximately linear
        nn.init.normal_(self.mlp[-1].weight, std=0.5)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z0_smooth, z1, t, s):
        """
        z0_smooth, z1 : (N, K)
        t, s          : scalar tensors
        """
        N = z0_smooth.shape[0]
        if t.dim() == 0:
            t_vec = t.expand(N).unsqueeze(-1)
        else:
            t_vec = t.unsqueeze(-1)
        if s.dim() == 0:
            s_vec = s.expand(N).unsqueeze(-1)
        else:
            s_vec = s.unsqueeze(-1)
        x = torch.cat([z0_smooth, z1, t_vec, s_vec], dim=-1)
        delta = self.mlp(x)
        gate = (t_vec * (1.0 - t_vec))
        linear = z0_smooth + t_vec * (z1 - z0_smooth)
        return linear + gate * delta


# =============================================================================
# Graph heat diffusion: smooth z^0 by s ∈ [0, 1]
# =============================================================================
class GraphSmoother(nn.Module):
    """
    Approximates exp(-σ L) z by Chebyshev/polynomial expansion:
        z(s) = (I - α(s) L_norm)^K_diff  z

    where α(s) = α_max * sigmoid(logit_α_max + 6(s - 0.5))   ∈ (0, α_max),
    so:
        s = 0   -> α ≈ 0     -> z(s) ≈ z (no smoothing)
        s = 1   -> α ≈ α_max -> strong smoothing

    L_norm is symmetric-normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    α_max kept < 2 / λ_max(L_norm) for numerical stability (λ_max(L_norm) ≤ 2).
    """
    def __init__(self, K_diff: int = 8, alpha_max: float = 0.5):
        super().__init__()
        self.K_diff = K_diff
        self.alpha_max = alpha_max

    @torch.no_grad()
    def normalize_laplacian(self, L_sparse: torch.sparse.Tensor,
                           N: int) -> torch.sparse.Tensor:
        """Build symmetric normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}."""
        # L is sparse coo with structure: -W on off-diagonal, D on diagonal
        # We need: L_norm[i, j] = L[i, j] / sqrt(D[i] D[j])
        L = L_sparse.coalesce()
        idx = L.indices()
        vals = L.values()
        # Extract diagonal degrees
        is_diag = (idx[0] == idx[1])
        diag_idx = idx[0][is_diag]
        diag_vals = vals[is_diag]
        D = torch.zeros(N, dtype=vals.dtype, device=vals.device)
        D[diag_idx] = diag_vals
        D_inv_sqrt = 1.0 / torch.sqrt(D.clamp_min(1e-6))
        new_vals = vals * D_inv_sqrt[idx[0]] * D_inv_sqrt[idx[1]]
        return torch.sparse_coo_tensor(idx, new_vals, L.size()).coalesce()

    def forward(self, z: torch.Tensor, s: torch.Tensor,
                L_norm: torch.sparse.Tensor) -> torch.Tensor:
        """
        z: (N, K)
        s: scalar in [0, 1]
        L_norm: sparse symmetric-normalized Laplacian (N, N)
        """
        # α(s) ∈ (0, alpha_max), with α(0) ≈ 0
        # use shifted sigmoid so that s=0 is ~ no smoothing
        # alpha = self.alpha_max * torch.sigmoid(8.0 * (s - 0.05))
        alpha = self.alpha_max * (s ** 2)
        # z_{k+1} = z_k - α L_norm z_k
        out = z
        for _ in range(self.K_diff):
            out = out - alpha * torch.sparse.mm(L_norm, out)
        return out


# =============================================================================
# Config
# =============================================================================
@dataclass
class PARFAITConfig:
    n_genes: int
    emb_names: Sequence[str]
    emb_dims: Sequence[int]

    K: int = 128
    encoder_hidden: int = 256
    encoder_dropout: float = 0.1
    loc_hidden: int = 64
    path_hidden: int = 64

    # Graph diffusion settings
    K_diff: int = 8
    alpha_max: float = 0.5

    # Spatial regularization
    lambda_spatial: float = 0.0     # GMRF on z^1; can set 0 (often redundant)
    lambda_path: float = 0.0

    # Beta prior strengths
    beta_strength_t: float = 4.0
    beta_strength_s: float = 4.0

    # Per-embedding init for (t, s). If None, defaults set in builder.
    t_init: Optional[Dict[str, float]] = None
    s_init: Optional[Dict[str, float]] = None

    kl_warmup_steps: int = 1000
    kl_ramp_steps: int = 2000
    kl_floor: float = 1e-3


# =============================================================================
# Spatial Laplacian (single slice) -- same as v8
# =============================================================================
def build_laplacian_single_slice(coords: np.ndarray, k: int = 6) -> torch.sparse.Tensor:
    from sklearn.neighbors import NearestNeighbors
    n = coords.shape[0]
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn_.kneighbors(coords)
    rs = np.repeat(np.arange(n), k)
    cs = idx[:, 1:].reshape(-1)
    pairs = np.unique(np.sort(np.stack([rs, cs], axis=1), axis=1), axis=0)
    r = np.concatenate([pairs[:, 0], pairs[:, 1]])
    c = np.concatenate([pairs[:, 1], pairs[:, 0]])
    w = np.ones_like(r, dtype=np.float32)
    deg = np.bincount(r, weights=w, minlength=n).astype(np.float32)
    diag = np.arange(n)
    rows_L = np.concatenate([r, diag])
    cols_L = np.concatenate([c, diag])
    vals_L = np.concatenate([-w, deg]).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows_L, cols_L], axis=0).astype(np.int64)),
        torch.from_numpy(vals_L),
        size=(n, n),
    ).coalesce()


# =============================================================================
# Data-driven inits for both t and s
# =============================================================================
def rayleigh_quotient(Z: np.ndarray, L_sparse: torch.sparse.Tensor) -> float:
    """R(Z) = tr(Z^T L Z) / tr(Z^T Z)."""
    Zt = torch.from_numpy(Z.astype(np.float32))
    LZ = torch.sparse.mm(L_sparse.cpu(), Zt)
    num = (Zt * LZ).sum().item()
    den = (Zt * Zt).sum().item() + 1e-12
    return float(num / den)


def rayleigh_to_t_init(
    embeddings: Dict[str, np.ndarray],
    L_sparse: torch.sparse.Tensor,
    t_low: float = 0.1, t_high: float = 0.9,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    R = {n: rayleigh_quotient(E, L_sparse) for n, E in embeddings.items()}
    Rmin, Rmax = min(R.values()), max(R.values())
    span = Rmax - Rmin if Rmax > Rmin else 1.0
    t_init = {n: t_high - (t_high - t_low) * (R[n] - Rmin) / span for n in R}
    return t_init, R


def smoothness_to_s_init(
    embeddings: Dict[str, np.ndarray],
    L_sparse: torch.sparse.Tensor,
    s_low: float = 0.1, s_high: float = 0.9,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Initialize s_m by an "implicit smoothing index". Idea:

        For each embedding E^{(m)}, compute Rayleigh of E vs Rayleigh of
        a smoothed-version E_smoothed. The ratio
            S_m = R(E_smoothed) / R(E)
        measures how much E was already smooth (if S_m close to 1, E was
        already smooth so additional smoothing has limited effect; if S_m
        close to 0, smoothing dramatically lowers Rayleigh — meaning E was
        rough and benefits from smoothing).

    But for initialization we use a simpler proxy: HIGH Rayleigh (rough)
    embeddings get LOW s_init (because they should not be heavily smoothed
    a priori — let the model decide). LOW Rayleigh (already smooth)
    embeddings get HIGH s_init (consistent with their pretraining smoothing).

    This is intentionally coarse — it's just an initialization. The model
    learns the actual s_m from data.

    Note: an alternative is uniform init s = 0.3 for all. We provide the
    Rayleigh-based heuristic as primary, uniform as fallback.
    """
    R = {n: rayleigh_quotient(E, L_sparse) for n, E in embeddings.items()}
    Rmin, Rmax = min(R.values()), max(R.values())
    span = Rmax - Rmin if Rmax > Rmin else 1.0
    # smaller R (smoother) -> larger s_init
    s_init = {n: s_high - (s_high - s_low) * (R[n] - Rmin) / span for n in R}
    return s_init, R

def effective_rank_pca(E: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy effective rank from singular values.
    E: (N, d), assumed z-scored per feature.
    """
    X = E - E.mean(axis=0, keepdims=True)
    # singular values of centered matrix
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    p = lam / (lam.sum() + eps)
    H = -(p * np.log(p + eps)).sum()
    return float(np.exp(H))


def participation_ratio(E: np.ndarray, eps: float = 1e-12) -> float:
    """
    Participation-ratio intrinsic dimension.
    """
    X = E - E.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    return float((lam.sum() ** 2) / (np.sum(lam ** 2) + eps))

def intrinsic_dim_to_t_init(
    embeddings: Dict[str, np.ndarray],
    t_low: float = 0.1,
    t_high: float = 0.9,
    method: str = "effective_rank",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if method == "effective_rank":
        ID = {n: effective_rank_pca(E) for n, E in embeddings.items()}
    elif method == "participation_ratio":
        ID = {n: participation_ratio(E) for n, E in embeddings.items()}
    else:
        raise ValueError(method)

    vals = np.array([np.log(ID[n] + 1e-12) for n in embeddings])
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin if vmax > vmin else 1.0

    t_init = {}
    for n in embeddings:
        u = (np.log(ID[n] + 1e-12) - vmin) / span
        # larger ID -> lower t
        t_init[n] = float(t_high - (t_high - t_low) * u)
    return t_init, ID
# =============================================================================
# Main model
# =============================================================================
class PARFAIT:
    def __init__(
        self,
        cfg: PARFAITConfig,
        L_sparse: torch.sparse.Tensor,
        encoder: CountsEncoder,
        loc_encoder: LocationEncoder,
        path_net: PathNetwork2D,
        smoother: GraphSmoother,
    ):
        self.cfg = cfg
        self.L = L_sparse.coalesce()
        self.L_norm = None             # populated lazily on right device
        self.encoder = encoder
        self.loc_encoder = loc_encoder
        self.path_net = path_net
        self.smoother = smoother
        self._svi = None
        self._guide_global = None
        self._cache: Dict = {}
        self._kl_weight = 1.0

    # ------------------------------------------------------------------
    def _ensure_on_device(self, device: torch.device):
        if self.L.device != device:
            self.L = self.L.to(device).coalesce()
        if self.L_norm is None or self.L_norm.device != device:
            N = self.L.size(0)
            self.L_norm = self.smoother.normalize_laplacian(self.L, N).to(device)
        for m in [self.encoder, self.loc_encoder, self.path_net, self.smoother]:
            p = next(m.parameters(), None)
            if p is not None and p.device != device:
                m.to(device)

    # ------------------------------------------------------------------
    def model(self, X_raw, emb_list, coords_std, log_lib=None):
        cfg = self.cfg
        N = X_raw.shape[0]
        device = X_raw.device
        dtype = X_raw.dtype if X_raw.is_floating_point() else torch.float32
        self._ensure_on_device(device)

        pyro.module("parfait9_loc_encoder", self.loc_encoder)
        pyro.module("parfait9_path_net", self.path_net)

        # --------- hyperpriors ---------
        theta_g = pyro.sample(
            "theta_gene",
            dist.Gamma(
                torch.full((cfg.n_genes,), 2.0, device=device, dtype=dtype),
                torch.full((cfg.n_genes,), 1.0, device=device, dtype=dtype),
            ).to_event(1),
        )

        sigma_emb_list = []
        nu_emb_list = []
        for name in cfg.emb_names:
            s_m = pyro.sample(
                f"sigma_{name}",
                dist.HalfNormal(torch.tensor(1.0, device=device, dtype=dtype)),
            )
            xi_m = pyro.sample(
                f"xi_{name}",
                dist.Normal(
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(1.0, device=device, dtype=dtype),
                ),
            )
            nu_m = 2.0 + F.softplus(xi_m)
            sigma_emb_list.append(s_m)
            nu_emb_list.append(nu_m)

        # --------- decoder weights ---------
        W_X = pyro.sample(
            "W_X",
            dist.Normal(
                torch.zeros(cfg.n_genes, cfg.K, device=device, dtype=dtype),
                (1.0 / float(cfg.K) ** 0.5)
                * torch.ones(cfg.n_genes, cfg.K, device=device, dtype=dtype),
            ).to_event(2),
        )
        b_X = pyro.sample(
            "b_X",
            dist.Normal(
                torch.zeros(cfg.n_genes, device=device, dtype=dtype),
                torch.ones(cfg.n_genes, device=device, dtype=dtype),
            ).to_event(1),
        )

        A_list, b_list = [], []
        for name, d_m in zip(cfg.emb_names, cfg.emb_dims):
            A_m = pyro.sample(
                f"A_{name}",
                dist.Normal(
                    torch.zeros(d_m, cfg.K, device=device, dtype=dtype),
                    (1.0 / float(cfg.K) ** 0.5)
                    * torch.ones(d_m, cfg.K, device=device, dtype=dtype),
                ).to_event(2),
            )
            b_m_dec = pyro.sample(
                f"b_{name}",
                dist.Normal(
                    torch.zeros(d_m, device=device, dtype=dtype),
                    torch.ones(d_m, device=device, dtype=dtype),
                ).to_event(1),
            )
            A_list.append(A_m); b_list.append(b_m_dec)

        # --------- learnable (t_m, s_m) per embedding ---------
        t_init = cfg.t_init or {n: 0.5 for n in cfg.emb_names}
        s_init = cfg.s_init or {n: 0.3 for n in cfg.emb_names}

        t_m_list, s_m_list = [], []
        for name in cfg.emb_names:
            t0 = float(np.clip(t_init.get(name, 0.5), 1e-3, 1 - 1e-3))
            s0 = float(np.clip(s_init.get(name, 0.3), 1e-3, 1 - 1e-3))

            alpha_t = pyro.param(
                f"alpha_t_{name}",
                torch.tensor(np.log(t0 / (1 - t0)), device=device, dtype=dtype),
            )
            alpha_s = pyro.param(
                f"alpha_s_{name}",
                torch.tensor(np.log(s0 / (1 - s0)), device=device, dtype=dtype),
            )
            t_m = torch.sigmoid(alpha_t)
            s_m = torch.sigmoid(alpha_s)
            t_m_list.append(t_m)
            s_m_list.append(s_m)

            # Beta priors (mean = init, total strength = beta_strength_*)
            for var_name, val, init_val, strength in [
                (f"t_{name}", t_m, t0, cfg.beta_strength_t),
                (f"s_{name}", s_m, s0, cfg.beta_strength_s),
            ]:
                a = strength * init_val
                b = strength * (1.0 - init_val)
                vc = val.clamp(1e-5, 1 - 1e-5)
                log_prior = (a - 1.0) * torch.log(vc) \
                            + (b - 1.0) * torch.log1p(-vc)
                pyro.factor(f"prior_{var_name}", log_prior)

        # --------- latents ---------
        z1 = self.loc_encoder(coords_std)

        with pyro.plate("spots_z0", N):
            z0 = pyro.sample(
                "z_0",
                dist.Normal(
                    torch.zeros(N, cfg.K, device=device, dtype=dtype),
                    torch.ones(N, cfg.K, device=device, dtype=dtype),
                ).to_event(1),
            )

        # GMRF on z^1 (often redundant since z^1 is already a smooth function
        # of coords — still useful to guard against pathological f_psi)
        if cfg.lambda_spatial > 0:
            Lz1 = torch.sparse.mm(self.L, z1)
            pyro.factor("z1_gmrf", -0.5 * cfg.lambda_spatial * (z1 * Lz1).sum())

        if cfg.lambda_path > 0:
            pyro.factor("path_reg",
                        -0.5 * cfg.lambda_path * ((z1 - z0) ** 2).sum())

        # --------- counts likelihood ---------
        if log_lib is None:
            log_lib = torch.log(X_raw.sum(-1).clamp_min(1.0))
        lib = log_lib.exp().unsqueeze(-1)
        gene_logits = z0 @ W_X.t() + b_X
        gene_props = torch.softmax(gene_logits, dim=-1)
        mu_nb = lib * gene_props
        theta_b = theta_g.unsqueeze(0).expand(N, -1).clamp_min(1e-3)
        probs = mu_nb / (mu_nb + theta_b + 1e-8)
        with pyro.plate("spots_X", N):
            pyro.sample("X_raw",
                        dist.NegativeBinomial(total_count=theta_b, probs=probs).to_event(1),
                        obs=X_raw)

        # --------- embedding likelihoods at z(t_m, s_m) ---------
        for m, (name, d_m) in enumerate(zip(cfg.emb_names, cfg.emb_dims)):
            t_m = t_m_list[m]
            s_m = s_m_list[m]
            # smooth z^0 by s_m, then path
            z0_smooth = self.smoother(z0, s_m, self.L_norm)
            z_ts = self.path_net(z0_smooth, z1, t_m, s_m)
            mu_E = z_ts @ A_list[m].t() + b_list[m]
            scale_E = sigma_emb_list[m].expand_as(mu_E)
            with poutine.scale(scale=1.0 / float(d_m)):
                with pyro.plate(f"spots_{name}", N):
                    pyro.sample(
                        f"E_{name}",
                        dist.StudentT(df=nu_emb_list[m],
                                      loc=mu_E, scale=scale_E).to_event(1),
                        obs=emb_list[m],
                    )

    # ------------------------------------------------------------------
    def guide(self, X_raw, emb_list, coords_std, log_lib=None):
        cfg = self.cfg
        N = X_raw.shape[0]
        device = X_raw.device
        self._ensure_on_device(device)
        pyro.module("parfait9_encoder", self.encoder)

        if self._guide_global is None:
            self._guide_global = AutoNormal(
                poutine.block(self.model, hide=["z_0"]),
                init_scale=0.05,
            )
        self._guide_global(X_raw, emb_list, coords_std, log_lib)

        if log_lib is None:
            log_lib = torch.log(X_raw.sum(-1).clamp_min(1.0))
        mu0, logvar0 = self.encoder(X_raw, log_lib)

        with pyro.plate("spots_z0", N):
            with poutine.scale(scale=self._kl_weight):
                pyro.sample("z_0",
                            dist.Normal(mu0, (0.5 * logvar0).exp()).to_event(1))

    # ------------------------------------------------------------------
    def _kl_schedule(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.kl_warmup_steps:
            return cfg.kl_floor
        elif step < cfg.kl_warmup_steps + cfg.kl_ramp_steps:
            frac = (step - cfg.kl_warmup_steps) / float(cfg.kl_ramp_steps)
            return cfg.kl_floor + (1.0 - cfg.kl_floor) * frac
        return 1.0

    # ------------------------------------------------------------------
    def fit(self, X_raw, emb_list, coords_std, log_lib=None,
            n_steps: int = 6000, lr: float = 1e-3,
            log_every: int = 200, seed: int = 0):
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        self._ensure_on_device(X_raw.device)
        self._cache = dict(X_raw=X_raw, emb_list=emb_list,
                           coords_std=coords_std, log_lib=log_lib)
        self._guide_global = None

        self._svi = SVI(self.model, self.guide, Adam({"lr": lr}), loss=Trace_ELBO())
        losses = []
        for step in range(n_steps):
            self._kl_weight = self._kl_schedule(step)
            loss = self._svi.step(X_raw, emb_list, coords_std, log_lib)
            losses.append(loss)
            if step % log_every == 0 or step == n_steps - 1:
                t_str = ", ".join(
                    f"{n}=({torch.sigmoid(pyro.param(f'alpha_t_{n}')).item():.2f},"
                    f"{torch.sigmoid(pyro.param(f'alpha_s_{n}')).item():.2f})"
                    for n in self.cfg.emb_names
                )
                print(f"[step {step:>5d}] -ELBO={loss:.1f}  klw={self._kl_weight:.3f}")
                print(f"            (t_m, s_m): {t_str}")
        return losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_t_s(self) -> Dict[str, Tuple[float, float]]:
        out = {}
        for n in self.cfg.emb_names:
            t = float(torch.sigmoid(pyro.param(f"alpha_t_{n}")).item())
            s = float(torch.sigmoid(pyro.param(f"alpha_s_{n}")).item())
            out[n] = (t, s)
        return out

    @torch.no_grad()
    def posterior_z(self, n_samples: int = 30) -> Dict:
        c = self._cache
        device = c["X_raw"].device
        z0_samples = []
        for _ in range(n_samples):
            tr = poutine.trace(self.guide).get_trace(
                c["X_raw"], c["emb_list"], c["coords_std"], c["log_lib"]
            )
            z0_samples.append(tr.nodes["z_0"]["value"].detach().cpu())
        z0_stk = torch.stack(z0_samples, 0)
        z0_mean = z0_stk.mean(0)
        z1 = self.loc_encoder(c["coords_std"]).detach().cpu()
        return {"z0_mean": z0_mean, "z1": z1}

    @torch.no_grad()
    def z_at_grid(
        self, t_grid: Sequence[float], s_grid: Sequence[float],
    ) -> Dict[Tuple[float, float], np.ndarray]:
        """Compute z(t, s) on a 2D grid of (t, s) values."""
        post = self.posterior_z(n_samples=30)
        device = self._cache["X_raw"].device
        z0 = post["z0_mean"].to(device)
        z1 = post["z1"].to(device)
        out: Dict[Tuple[float, float], np.ndarray] = {}
        self._ensure_on_device(device)
        for s in s_grid:
            s_t = torch.tensor(float(s), device=device, dtype=torch.float32)
            z0_smooth = self.smoother(z0, s_t, self.L_norm)
            for t in t_grid:
                t_t = torch.tensor(float(t), device=device, dtype=torch.float32)
                z = self.path_net(z0_smooth, z1, t_t, s_t).detach().cpu().numpy()
                out[(float(t), float(s))] = z
        return out

    @torch.no_grad()
    def z_at_learned(self) -> Dict[str, np.ndarray]:
        """For each embedding, compute z(t_m, s_m) at its learned coordinates."""
        post = self.posterior_z(n_samples=30)
        device = self._cache["X_raw"].device
        z0 = post["z0_mean"].to(device)
        z1 = post["z1"].to(device)
        out = {}
        self._ensure_on_device(device)
        for n in self.cfg.emb_names:
            t_m = torch.sigmoid(pyro.param(f"alpha_t_{n}")).detach()
            s_m = torch.sigmoid(pyro.param(f"alpha_s_{n}")).detach()
            z0_smooth = self.smoother(z0, s_m, self.L_norm)
            z = self.path_net(z0_smooth, z1, t_m, s_m).detach().cpu().numpy()
            out[n] = z
        return out


# =============================================================================
# Builder
# =============================================================================
def build_parfait(
    cfg: PARFAITConfig,
    coords: np.ndarray,
    n_neighbors: int = 6,
) -> Tuple[PARFAIT, np.ndarray]:
    L = build_laplacian_single_slice(coords, k=n_neighbors)
    encoder = CountsEncoder(cfg.n_genes, cfg.K,
                            hidden=cfg.encoder_hidden, dropout=cfg.encoder_dropout)
    loc_enc = LocationEncoder(cfg.K, hidden=cfg.loc_hidden)
    path_net = PathNetwork2D(cfg.K, hidden=cfg.path_hidden)
    smoother = GraphSmoother(K_diff=cfg.K_diff, alpha_max=cfg.alpha_max)

    coords_std = (coords - coords.mean(0, keepdims=True)) \
                 / (coords.std(0, keepdims=True) + 1e-6)
    coords_std = coords_std.astype(np.float32)

    model = PARFAIT(cfg, L, encoder, loc_enc, path_net, smoother)
    return model, coords_std
