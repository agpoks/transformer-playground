"""PatchTST-style: attention over time-series patches, channel-independent.

Reference: Nie, Nguyen, Sinthong, Kalagnanam, "A Time Series is Worth 64
Words: Long-term Forecasting with Transformers", ICLR 2023, arXiv:2211.14730.
See papers/README.md (bibtex key `nie2023patchtst`).

Two ideas from the paper, both implemented literally (not just referenced):

1. PATCHING: instead of one token per timestep (as a naive time-series
   transformer would do), each univariate series is split into overlapping
   patches of length `patch_len` (stride `stride` apart) along the time
   axis, and each patch is linearly projected to one token -- exactly like
   ViT patchifying a 2D image into spatial patches, but 1D. This shortens
   the attention sequence length from `seq_len` to roughly `seq_len /
   stride`, and lets each token carry local sub-sequence shape information
   the way a single-timestep token cannot.

2. CHANNEL INDEPENDENCE: for a multivariate series (ETTh1 has 7 channels
   here), every channel is patch-embedded, encoded, and forecast by
   *the exact same* transformer weights, run independently per channel --
   channels are never mixed together inside the model (no channel-mixing
   attention, no channel dimension in any embedding). This is implemented
   here by folding the channel dimension into the batch dimension before
   the patch embedding and un-folding it back afterward, so it is not an
   approximation -- the weights are structurally shared, channels literally
   cannot attend to each other.

Bidirectional (encoder-only) self-attention over patches, no causal mask --
forecasting looks at the whole lookback window at once, there is no
"future patch" to hide from a patch token the way there is for
autoregressive next-token prediction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Bidirectional (no mask) multi-head self-attention, hand-written."""

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
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d_model = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm: LayerNorm before each sublayer, residual add after."""

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


class PatchEmbedding(nn.Module):
    """Unfold a (B, L) univariate series into overlapping patches, project
    each patch to d_model, add a learned per-patch-index positional
    embedding."""

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len) -> (B, n_patches, d_model)."""
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
        return self.proj(patches) + self.pos_embed


class PatchTSTModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.patch_embed.n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, seq_len) -- C channels, each forecast independently by
        the SAME weights (channel-independence): fold C into the batch
        dimension, run the shared encoder, un-fold back to (B, C, pred_len).
        """
        b, c, seq_len = x.shape
        x = x.reshape(b * c, seq_len)  # channels never mix -- literally not present as a dim
        h = self.patch_embed(x)  # (B*C, n_patches, d_model)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        h = h.reshape(b * c, -1)  # flatten patches
        out = self.head(h)  # (B*C, pred_len)
        return out.reshape(b, c, self.pred_len)
