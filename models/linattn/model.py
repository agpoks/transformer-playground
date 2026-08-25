"""Performer-style linear attention: causal, O(T) instead of O(T^2).

Reference: Choromanski, Likhosherstov, Dohan, Song, Gane, Sarlos, Hawkins,
Davis, Mohiuddin, Kaiser, Belanger, Colwell, Weller, "Rethinking Attention
with Performers", 2020. arXiv:2009.14794 (FAVOR+). See papers/README.md
(bibtex key `choromanski2021performer`).

Contrast with models/gpt/model.py, which this model otherwise mirrors
exactly (same GPTBlock-style pre-norm architecture, same Tiny Shakespeare
causal-LM task, same learned positional embedding) so that the ONLY
difference in a benchmark run is the attention mechanism itself:

Ordinary softmax attention computes an explicit (T x T) score matrix
softmax(QK^T/sqrt(d_k)) -- O(T^2) time and memory in the sequence length.
FAVOR+ instead approximates the softmax *kernel* itself with an explicit,
always-positive random feature map phi(x) such that

    softmax(q . k) ~= phi(q) . phi(k)

for phi(x) = (1/sqrt(m)) * exp(w_i . x - ||x||^2 / 2), i=1..m, with
w_1..w_m ~ N(0, I) FIXED random Gaussian projections (a buffer, never
trained -- this repo does not implement the paper's optional periodic
resampling / orthogonal-random-feature refinements, a real simplification,
stated once here). Because this factorizes attention into a plain dot
product against a feature vector instead of an explicit T x T score
matrix, causal (autoregressive) attention can be computed EXACTLY via
running cumulative sums over the sequence dimension -- no windowing hack,
no approximation of the causal mask itself is needed (only the softmax
kernel is approximated):

    S_i = sum_{j<=i} phi(k_j) (x) v_j            (a (m, d_v) matrix, running)
    Z_i = sum_{j<=i} phi(k_j)                     (an m-vector, running)
    out_i = [phi(q_i) . S_i] / [phi(q_i) . Z_i]

This is O(T * m * d) instead of O(T^2 * d) -- LINEAR in sequence length T,
which is the entire point of including this model: a genuine, real
compute-cost comparison against models/gpt's quadratic attention at
increasing context length (see docs/source/models/linattn.md and the
benchmark for the actual measured MMACs/latency numbers at two sequence
lengths).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_random_features(d_k: int, m: int) -> torch.Tensor:
    """Fixed (never trained) random projection directions w_1..w_m ~ N(0, I)."""
    return torch.randn(m, d_k)


def phi(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FAVOR+ positive random feature map. x: (B, h, T, d_k), w: (m, d_k).
    Returns (B, h, T, m), always > 0."""
    m = w.shape[0]
    wx = torch.einsum("bhtd,md->bhtm", x, w)  # (B, h, T, m)
    x_sq = (x**2).sum(dim=-1, keepdim=True) / 2  # (B, h, T, 1)
    stabilizer = wx.max(dim=-1, keepdim=True).values  # per-position max, for exp() stability
    return (torch.exp(wx - x_sq - stabilizer) + eps) / math.sqrt(m)


def causal_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """q, k, v: (B, h, T, d_k)/(B, h, T, d_v). w: (m, d_k) fixed random features.
    Returns (B, h, T, d_v), causal (position i uses only j <= i)."""
    phi_q, phi_k = phi(q, w), phi(k, w)  # (B, h, T, m)
    kv = torch.einsum("bhtm,bhtd->bhtmd", phi_k, v)  # (B, h, T, m, d_v), per-position outer product
    kv_cumsum = kv.cumsum(dim=2)  # S_i = sum_{j<=i} phi(k_j) (x) v_j
    k_cumsum = phi_k.cumsum(dim=2)  # Z_i = sum_{j<=i} phi(k_j), (B, h, T, m)

    numerator = torch.einsum("bhtm,bhtmd->bhtd", phi_q, kv_cumsum)
    denominator = torch.einsum("bhtm,bhtm->bht", phi_q, k_cumsum).unsqueeze(-1)
    return numerator / denominator.clamp_min(1e-6)


class CausalLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_features: int = 64):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.register_buffer("w_features", make_random_features(self.d_k, n_features))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d_model = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        out = causal_linear_attention(q, k, v, self.w_features)
        out = out.transpose(1, 2).contiguous().view(b, t, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinAttnBlock(nn.Module):
    """Pre-norm, identical layout to models/gpt's GPTBlock -- only the
    attention sublayer differs (FAVOR+ linear attention instead of
    softmax attention)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_features: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalLinearAttention(d_model, n_heads, n_features)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class LinAttnModel(nn.Module):
    """Same overall shape as models/gpt's GPTModel (learned positional
    embedding, pre-norm blocks, causal LM head) so a benchmark comparison
    isolates the attention mechanism itself."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        n_features: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [LinAttnBlock(d_model, n_heads, d_ff, n_features, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_embed(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x)
