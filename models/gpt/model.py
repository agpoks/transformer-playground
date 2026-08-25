"""GPT-style: causal self-attention, decoder-only, next-token prediction.

References: Radford, Narasimhan, Salimans, Sutskever, "Improving Language
Understanding by Generative Pre-Training" (GPT), 2018; Radford, Wu, Child,
Luan, Amodei, Sutskever, "Language Models are Unsupervised Multitask
Learners" (GPT-2), 2019. See papers/README.md (bibtex keys `radford2018gpt`,
`radford2019gpt2`).

Contrast with models/transformer/model.py and models/bert/model.py:
  - No encoder, no cross-attention -- only causal self-attention (token i
    attends to tokens <= i only), trained to predict token i+1 from tokens
    <= i (ordinary autoregressive language modeling, not BERT's masked
    in-fill objective).
  - PRE-norm residual blocks (LayerNorm *before* each sublayer, not after)
    -- GPT-2's actual choice, contrasted with the original Transformer's
    post-norm (models/transformer). Pre-norm trains more stably at depth,
    which is why it became the default for every transformer after GPT-2.

Positional encoding is a constructor flag, `pos_encoding`:
  - "learned" (GPT-2's actual choice): a plain learned embedding table
    indexed by absolute position, added to the token embedding once at the
    input, exactly like models/bert/model.py's positional embedding.
  - "rope" (Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", arXiv:2104.09864 -- NOT part of the original
    GPT-2 paper, added here as the modern alternative every current LLM
    actually uses): instead of adding anything to the input, each
    attention head's query/key vectors are *rotated* by an angle
    proportional to their position, applied fresh inside every attention
    layer (not just once at the input). Splitting a d_k-dim vector x into
    two halves x1, x2 and rotating each (x1_i, x2_i) pair by angle
    pos * theta_i, theta_i = 10000^(-2i/d_k), the standard "rotate-half"
    form (as used in GPT-NeoX/LLaMA) is:

        rotate_half(x) = concat(-x2, x1)
        RoPE(x, pos)   = x * cos(pos * theta) + rotate_half(x) * sin(pos * theta)

    applied to q and k (never v) before the dot product -- the relative
    position pos_i - pos_j falls directly out of q_i . k_j after rotation,
    which is RoPE's actual selling point (relative position from an
    absolute-looking operation, no learned table needed, generalizes to
    sequence lengths never seen during training).

Both modes live in ONE `GPTModel` class via the flag, sharing everything
else, so the only difference in a benchmark run is this one choice.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(t: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_cos_sin(t: int, d_k: int, device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each (t, d_k), ready to multiply elementwise
    against a (..., t, d_k) tensor, per the rotate-half RoPE formulation."""
    half = d_k // 2
    theta = base ** (-torch.arange(0, half, dtype=torch.float32, device=device) / half)  # (half,)
    pos = torch.arange(t, dtype=torch.float32, device=device)  # (t,)
    freqs = torch.outer(pos, theta)  # (t, half)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (t, d_k)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, h, T, d_k). cos/sin: (T, d_k)."""
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pos_encoding: str):
        super().__init__()
        assert d_model % n_heads == 0
        assert pos_encoding in ("learned", "rope")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pos_encoding = pos_encoding
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d_model = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        if self.pos_encoding == "rope":
            cos, sin = rope_cos_sin(t, self.d_k, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal_mask(t, x.device), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTBlock(nn.Module):
    """PRE-norm: LayerNorm before each sublayer, residual add after."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, pos_encoding: str, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, pos_encoding)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        pos_encoding: str = "learned",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert pos_encoding in ("learned", "rope")
        self.pos_encoding = pos_encoding
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        if pos_encoding == "learned":
            self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads, d_ff, pos_encoding, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids. Returns (B, T, vocab_size) logits."""
        b, t = idx.shape
        x = self.tok_embed(idx)
        if self.pos_encoding == "learned":
            pos = torch.arange(t, device=idx.device)
            x = x + self.pos_embed(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x)
