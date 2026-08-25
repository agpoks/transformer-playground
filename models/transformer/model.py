"""Transformer: encoder-decoder, self-attention + cross-attention.

Reference: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
Polosukhin, "Attention Is All You Need", NeurIPS 2017. arXiv:1706.03762.
See papers/README.md (bibtex key `vaswani2017transformer`).

The core idea: instead of processing a sequence step-by-step (RNN) or with
a local sliding window (CNN), let every position attend directly to every
other position in one matrix multiply. A single "head" of attention is

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

-- a weighted average of value vectors V, where the weights are how well
each query Q matches each key K. Multi-head attention runs this h times in
parallel on different learned linear projections of Q/K/V, so the model
can attend to different kinds of relationships at once, then concatenates
and projects the result back down.

This model uses attention in three distinct ways:
  - encoder self-attention: every source-sentence token attends to every
    other source-sentence token (no mask -- bidirectional).
  - decoder self-attention: every target-sentence token attends only to
    itself and *earlier* target tokens (causal mask -- autoregressive,
    since at generation time later tokens don't exist yet).
  - decoder cross-attention: every target token's query attends to the
    *encoder's* output as keys/values -- this is how information flows
    from the source sentence into the translation being generated.

Positional encoding is FIXED (not learned), a deliberate choice this paper
makes (contrast with BERT's learned positional embeddings, see
models/bert/model.py):

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

This repo uses POST-norm (residual add, then LayerNorm), matching the
original paper exactly -- most modern transformers (GPT-2 onward) use
pre-norm instead (LayerNorm before the sublayer) because it trains more
stably at large depth; post-norm is kept here for historical fidelity
since this model is deliberately small/shallow, where the stability
difference doesn't matter much in practice.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """q, k, v: (B, h, T, d_k). mask: (T_q, T_k) or (B, 1, T_q, T_k) bool,
    True = masked OUT (blocked). Returns (B, h, T_q, d_k)."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


class MultiHeadAttention(nn.Module):
    """h parallel attention heads via nn.Linear projections -- no
    nn.MultiheadAttention anywhere in this repo."""

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

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t_q, d_model = q_in.shape
        q = self._split_heads(self.w_q(q_in))
        k = self._split_heads(self.w_k(k_in))
        v = self._split_heads(self.w_v(v_in))
        out = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, h, T_q, d_k)
        out = out.transpose(1, 2).contiguous().view(b, t_q, d_model)
        return self.w_o(out)


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)


def causal_mask(t: int, device) -> torch.Tensor:
    """True above the diagonal (future positions) -> masked out."""
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))  # post-norm
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class TransformerModel(nn.Module):
    """Full encoder-decoder translation model."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        pe = sinusoidal_positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor | None) -> torch.Tensor:
        t = src.shape[1]
        x = self.drop(self.src_embed(src) * math.sqrt(self.d_model) + self.pe[:t])
        for layer in self.encoder:
            x = layer(x, src_pad_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        t = tgt.shape[1]
        x = self.drop(self.tgt_embed(tgt) * math.sqrt(self.d_model) + self.pe[:t])
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_pad_mask)
        return self.out_proj(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src: (B, T_src), tgt: (B, T_tgt) token ids. Returns (B, T_tgt, tgt_vocab)."""
        enc_out = self.encode(src, src_pad_mask=None)
        tgt_mask = causal_mask(tgt.shape[1], tgt.device)
        return self.decode(tgt, enc_out, tgt_mask, src_pad_mask=None)
