"""Perceiver: latent-bottleneck cross-attention, modality-agnostic.

Reference: Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira, "Perceiver:
General Perception with Iterative Attention", ICML 2021, arXiv:2103.03206.
See papers/README.md (bibtex key `jaegle2021perceiver`).

Core idea: plain self-attention over the raw input costs O(input_len^2),
which is fine for a few hundred tokens but not for e.g. every raw pixel of
an image (32*32*3 = 3072 "bytes" here, flattened, no convolutional
patchification -- that is deliberately not used, since the whole point of
this model is not needing any modality-specific input structure). Instead:

1. Keep a small, FIXED-size learned latent array (`n_latents` vectors),
   completely independent of input size.
2. CROSS-attend: latents as queries, the (large) raw input array as
   keys/values. Cost is O(n_latents * input_len) -- LINEAR in input size,
   because n_latents is fixed and small (here 32, vs. 3072 input tokens).
   This is the mechanism that decouples compute from input size.
3. SELF-attend among the latents only (cheap: O(n_latents^2), tiny since
   n_latents is small).
4. Repeat (2)+(3) a few times, refining the latent array's summary of the
   input each time.
5. Read out a classification from the final latent array (mean-pool + a
   linear head).

Simplification vs. the paper, stated explicitly: the paper repeats the
cross-attend+self-attend block many times (typically 8) with the SAME
weights shared across iterations after the first (a weight-tying scheme
that lets it scale to very deep effective computation with few
parameters). This repo uses UNSHARED weights per iteration (n_layers
independent blocks) for simplicity and because the model is already small
-- documented here rather than silently deviating from the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Queries from the latent array, keys/values from the (large) input
    array -- this is the step whose cost is linear in input length."""

    def __init__(self, d_latent: int, d_input: int, n_heads: int):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_latent // n_heads
        self.w_q = nn.Linear(d_latent, d_latent)
        self.w_k = nn.Linear(d_input, d_latent)
        self.w_v = nn.Linear(d_input, d_latent)
        self.w_o = nn.Linear(d_latent, d_latent)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        b, n_lat, d_latent = latents.shape
        q = self._split_heads(self.w_q(latents))
        k = self._split_heads(self.w_k(inputs))
        v = self._split_heads(self.w_v(inputs))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, n_lat, d_latent)
        return self.w_o(out)


class SelfAttention(nn.Module):
    """Ordinary self-attention among the (small) latent array only."""

    def __init__(self, d_latent: int, n_heads: int):
        super().__init__()
        assert d_latent % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_latent // n_heads
        self.w_q = nn.Linear(d_latent, d_latent)
        self.w_k = nn.Linear(d_latent, d_latent)
        self.w_v = nn.Linear(d_latent, d_latent)
        self.w_o = nn.Linear(d_latent, d_latent)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d_latent = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d_latent)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    """One cross-attend (input -> latents) + self-attend (latents) round,
    each pre-norm with a residual FFN, matching this repo's other models."""

    def __init__(self, d_latent: int, d_input: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_norm_lat = nn.LayerNorm(d_latent)
        self.cross_norm_in = nn.LayerNorm(d_input)
        self.cross_attn = CrossAttention(d_latent, d_input, n_heads)
        self.cross_ff_norm = nn.LayerNorm(d_latent)
        self.cross_ff = FeedForward(d_latent, d_ff)

        self.self_norm = nn.LayerNorm(d_latent)
        self.self_attn = SelfAttention(d_latent, n_heads)
        self.self_ff_norm = nn.LayerNorm(d_latent)
        self.self_ff = FeedForward(d_latent, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        latents = latents + self.drop(
            self.cross_attn(self.cross_norm_lat(latents), self.cross_norm_in(inputs))
        )
        latents = latents + self.drop(self.cross_ff(self.cross_ff_norm(latents)))
        latents = latents + self.drop(self.self_attn(self.self_norm(latents)))
        latents = latents + self.drop(self.self_ff(self.self_ff_norm(latents)))
        return latents


class PerceiverModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_latents: int = 32,
        d_latent: int = 64,
        d_ff: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)  # placeholder identity-ish proj, kept explicit for clarity
        self.latents = nn.Parameter(torch.zeros(1, n_latents, d_latent))
        nn.init.normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            [PerceiverBlock(d_latent, input_dim, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_latent)
        self.head = nn.Linear(d_latent, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_len, input_dim) -- the flattened "byte array"
        (e.g. raw pixels + position embedding, already prepared by the
        caller). Returns (B, n_classes) logits."""
        b = x.shape[0]
        latents = self.latents.expand(b, -1, -1)
        inputs = self.input_proj(x)
        for block in self.blocks:
            latents = block(latents, inputs)
        latents = self.norm_f(latents)
        return self.head(latents.mean(dim=1))
