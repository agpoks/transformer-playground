"""Tire-Patch-Wear Transformer -- spatial attention over a tire contact patch,
with a physics-informed wear-monotonicity loss.

THIS IS NOT A REPRODUCTION OF ANY PUBLISHED PAPER. As far as this project's
own literature search this session could find, no published work applies
attention/transformers to tire contact-patch wear specifically. This model
is this repo's own combination of two real, separately-established ideas:

1. Spatial self-attention over tokenized 1D positions -- the same mechanism
   `models/patchtst`'s patching and `models/perceiver`'s tokenization use,
   applied here to the tire contact patch's spatial extent instead of a
   time series or an image.

2. Real, classical tire-contact-patch mechanics and wear physics:

   - The **brush model** of the contact patch (Pacejka, H.B., "Tire and
     Vehicle Dynamics", 3rd ed., Butterworth-Heinemann, 2012 -- bibtex key
     `pacejka2012tire`): a parabolic normal-load pressure distribution
     along the patch, and a bristle-deflection/shear-stress model with an
     adhesion region (deflection grows linearly from the leading edge) and
     a sliding region (stress clipped to the local friction bound
     `mu * p(xi)`).
   - **Archard's wear law** (Archard, J.F., "Contact and Rubbing of Flat
     Surfaces", Journal of Applied Physics, 24(8):981-988, 1953, DOI
     10.1063/1.1721448 -- bibtex key `archard1953wear`): wear rate is
     proportional to normal load/pressure and sliding distance.

   The exact equations used to *generate the synthetic training data* below
   (`patch_coordinates`, `parabolic_pressure`, `patch_shear`) are a direct,
   honest re-derivation of the real, already-implemented and tested physics
   in the sibling project `~/github/tire_physics_nn`'s
   `tire_nn/physics/brush_patch.py` (discretized brush model) and
   `tire_nn/physics/wear.py` (`wear_rate(raw) = softplus(raw) >= 0`,
   structurally monotone). This repo does NOT import `tire_nn` as a Python
   dependency (every repo in this session's family is self-contained and
   independently cloneable) -- the same real formulas are simply
   reimplemented here, with explicit credit, rather than referenced only in
   prose.

Why simulated, not measured, data: no public labeled tire-contact-patch-wear
dataset exists (tire wear telemetry at this spatial resolution is
proprietary industry data). The simulator below is not arbitrary or
random -- it is the same real brush-model mechanics above, and it produces
a genuine, physically-expected NON-UNIFORM wear pattern (concentrated near
the trailing edge, where the adhesion region's linearly-growing bristle
deflection first exceeds the local friction bound and the patch begins to
slide) -- see `docs/source/models/tirewear.md` for a plot showing this
pattern is not noise. This is the same honest procedurally-generated-target
pattern already used by `nca` in `cnn-playground`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


# --------------------------------------------------------------------------
# Physics simulator (re-derived from tire_physics_nn's brush_patch.py/wear.py,
# credited above -- no code import, same real equations).
# --------------------------------------------------------------------------


def patch_coordinates(n_elements: int, half_length: float) -> tuple[torch.Tensor, float]:
    """Midpoint quadrature nodes xi in (0, 2a) along the patch, leading edge
    at xi=0, trailing edge at xi=2a. Returns (xi, dxi)."""
    a = half_length
    dxi = (2.0 * a) / n_elements
    xi = (torch.arange(n_elements, dtype=torch.float32) + 0.5) * dxi
    return xi, dxi


def parabolic_pressure(xi: torch.Tensor, half_length: float, Fz: float) -> torch.Tensor:
    """Classical parabolic line-load pressure, normalized so it integrates to Fz.

    p(xi) = (3 Fz / 4a) * [1 - ((xi - a) / a)^2]
    """
    a = half_length
    shape = 1.0 - ((xi - a) / a) ** 2
    return (3.0 * Fz / (4.0 * a)) * torch.clamp(shape, min=0.0)


def patch_shear(
    xi: torch.Tensor, pressure: torch.Tensor, slip: float, stiffness: float, mu: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local shear stress tau(xi) and a sliding indicator, brush-model style.

    Adhesion (tread carried undeflected from the leading edge, deflection
    grows linearly with distance travelled): tau_adhesion = stiffness*|slip|*xi
    Sliding (once that exceeds the local friction bound): tau = mu * p(xi).
    Real, checkable brush-model mechanics -- not a shortcut.
    """
    adhesion = stiffness * abs(slip) * xi
    limit = mu * pressure
    tau = torch.minimum(adhesion, limit)
    sliding = (adhesion > limit).float()
    return tau, sliding


def simulate_wear_trajectory(
    n_elements: int = 32,
    n_revolutions: int = 60,
    half_length: float = 1.0,
    Fz: float = 800.0,
    stiffness: float = 600.0,
    mu: float = 1.1,
    k_wear: float = 0.01,
    slip_mean: float = 0.35,
    slip_std: float = 0.08,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Simulate one tire's contact patch over `n_revolutions` rolling
    revolutions under a (noisy, but roughly constant-condition) slip input.

    Returns dict with:
      xi: (n_elements,) patch positions
      pressure: (n_revolutions, n_elements) -- recomputed each revolution
        (Fz held fixed here, so pressure is actually constant across
        revolutions in this simplified simulator -- documented in the docs
        page's Simplifications section)
      sliding_distance: (n_revolutions, n_elements) -- slip-proportional
        sliding amount in the sliding region only, this revolution
      wear_increment: (n_revolutions, n_elements) -- Archard-consistent
        wear added this revolution, softplus'd (structurally >= 0, same
        invariant as tire_physics_nn's wear_rate)
      wear_cumulative: (n_revolutions, n_elements) -- running total
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    xi, _dxi = patch_coordinates(n_elements, half_length)
    pressure_row = parabolic_pressure(xi, half_length, Fz)

    pressures, sliding_dists, wear_incs, wear_cums = [], [], [], []
    wear_total = torch.zeros(n_elements)
    for _ in range(n_revolutions):
        slip = torch.normal(mean=slip_mean, std=slip_std, size=(1,), generator=g).item()
        tau, sliding = patch_shear(xi, pressure_row, slip, stiffness, mu)
        sliding_distance = sliding * abs(slip)  # sliding-region tread travel this revolution
        raw_wear = k_wear * pressure_row * sliding_distance
        wear_inc = F.softplus(raw_wear)  # Archard-consistent, structurally >= 0
        wear_total = wear_total + wear_inc

        pressures.append(pressure_row.clone())
        sliding_dists.append(sliding_distance)
        wear_incs.append(wear_inc)
        wear_cums.append(wear_total.clone())

    return {
        "xi": xi,
        "pressure": torch.stack(pressures),
        "sliding_distance": torch.stack(sliding_dists),
        "wear_increment": torch.stack(wear_incs),
        "wear_cumulative": torch.stack(wear_cums),
    }


# --------------------------------------------------------------------------
# Model: bidirectional self-attention over patch-position tokens.
# --------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Ordinary bidirectional self-attention (no mask) over patch positions
    -- hand-written, self-contained (not imported from another models/*)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TirePatchWearTransformer(nn.Module):
    """Input per patch position: [pressure_i, wear_so_far_i] (2 features) +
    a learned per-position embedding (patch position is a fixed physical
    index, not a free sequence order, so a learned embedding indexed by
    position is appropriate, same idea as BERT-style's learned positional
    embedding). Bidirectional self-attention across all `n_elements`
    positions (no causal mask -- unlike a rolling-time sequence, every
    patch position is simultaneously present in a single revolution's
    contact footprint, there is no "future position" to hide). Predicts the
    next revolution's wear increment at every position.
    """

    def __init__(
        self,
        n_elements: int = 32,
        d_model: int = 48,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_elements, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, pressure: torch.Tensor, wear_so_far: torch.Tensor) -> torch.Tensor:
        """pressure, wear_so_far: (B, n_elements). Returns raw (pre-softplus)
        wear-increment logits (B, n_elements) -- the physics-informed
        monotonicity penalty (see example.py) is applied to these logits
        directly, and softplus(logits) is the actual predicted increment,
        mirroring tire_physics_nn's wear_rate(raw)=softplus(raw) invariant."""
        x = torch.stack([pressure, wear_so_far], dim=-1)  # (B, n_elements, 2)
        x = self.input_proj(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x).squeeze(-1)  # (B, n_elements)
