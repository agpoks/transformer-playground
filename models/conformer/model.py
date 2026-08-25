"""Conformer: convolution-augmented Transformer, for speech/audio.

Reference: Gulati, Qin, Chiu, Parmar, Zhang, Yu, Han, Wang, Zhang, Wu, Pang,
"Conformer: Convolution-augmented Transformer for Speech Recognition",
2020, arXiv:2005.08100. See papers/README.md (bibtex key `gulati2020conformer`).

Conformer's actual contribution isn't "add a conv layer somewhere" -- it's a
specific block structure (the "macaron" design, from the original Macaron
Net paper Conformer borrows the idea from): a normal Transformer block has
one FFN sublayer; a Conformer block has TWO half-strength FFN sublayers,
one on either side of a self-attention module and a convolution module:

    x = x + 1/2 * FFN(LN(x))          # first half-step FFN
    x = x + MHSA(LN(x))               # self-attention (bidirectional, this
                                       # is used as an ASR *encoder*, not a
                                       # causal decoder)
    x = x + Conv(x)                   # the conv module (has its own LN
                                       # inside, see ConvModule below)
    x = x + 1/2 * FFN(LN(x))          # second half-step FFN
    x = LN(x)                         # final norm

The macaron idea (splitting the FFN into two half-weighted copies around
the "main" sublayer) is a real, specific, checkable detail from the paper
-- not a normal FFN with a bug in front of it.

Simplification vs. the paper: the original uses Transformer-XL-style
*relative* positional encoding inside self-attention. This repo uses a
simpler learned absolute positional embedding instead, added once at the
input (same idea as models/bert's positional embedding) -- stated here
explicitly rather than silently. The conv module's pointwise -> GLU ->
depthwise -> BatchNorm -> Swish -> pointwise sequence, and the macaron FFN
structure, are both implemented literally/faithfully.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    """Plain 2-layer FFN with Swish (SiLU) -- the *caller* applies the 0.5
    residual weight (see ConformerBlock), this module just computes f(x)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Bidirectional (no causal mask) multi-head self-attention -- Conformer
    is used as an ASR *encoder* over a whole utterance, unlike GPT's causal
    decoder self-attention. Self-contained (no import from models/gpt)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        b, t, d_model = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d_model)
        return self.drop(self.w_o(out))


class ConvModule(nn.Module):
    """The real Conformer conv sequence: LN -> pointwise conv (2x channels)
    -> GLU -> depthwise conv -> BatchNorm -> Swish -> pointwise conv -> drop.
    Operates on (B, T, C); internally transposes to (B, C, T) for Conv1d."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pointwise1(x)  # (B, 2C, T)
        x = F.glu(x, dim=1)  # (B, C, T) -- GLU gates half the channels with the other half
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        x = self.drop(x)
        return x.transpose(1, 2)  # (B, T, C)


class ConformerBlock(nn.Module):
    """The macaron block: half-FFN, self-attn, conv, half-FFN, final norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, conv_kernel: int, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm_final(x)


class ConformerModel(nn.Module):
    """Input projection (n_mels -> d_model) + learned positional embedding +
    a stack of ConformerBlocks + mean-pool + linear classification head."""

    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        conv_kernel: int = 15,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, n_heads, d_ff, conv_kernel, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_mels) log-mel spectrogram frames. Returns (B, num_classes) logits."""
        b, t, _ = x.shape
        x = self.input_proj(x)
        pos = torch.arange(t, device=x.device)
        x = self.drop(x + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # mean-pool over time
        return self.head(x)
