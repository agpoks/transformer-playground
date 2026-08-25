"""BERT-style: bidirectional self-attention, encoder-only, masked language modeling.

Reference: Devlin, Chang, Lee, Toutanova, "BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding", NAACL 2019.
arXiv:1810.04805. See papers/README.md (bibtex key `devlin2019bert`).

Contrast with models/transformer/model.py (the original Transformer):
  - No decoder, no cross-attention, no causal mask anywhere -- every
    encoder layer's self-attention is fully bidirectional (mask=None
    always): every token can see every other token, both before and
    after it in the sequence. This is what "bidirectional" means here,
    and it's the whole point of BERT vs. a left-to-right (GPT-style)
    decoder-only model.
  - Positional information is a LEARNED embedding table here, not the
    original Transformer's fixed sinusoidal function -- one of BERT's
    actual design choices (Sec. 3.1 of the paper), not an architectural
    necessity.
  - Trained with the masked-language-modeling (MLM) objective, not
    next-token prediction: given a sentence with ~15% of its tokens
    hidden, predict the original token at each hidden position. Because
    a model that only ever sees a literal [MASK] token during training
    would never see one at fine-tuning/inference time, the real BERT
    recipe (this repo implements it exactly) replaces each of the chosen
    15% of tokens as follows:
        80% of the time -> replace with [MASK]
        10% of the time -> replace with a random vocabulary token
        10% of the time -> leave the original token unchanged
    and always trains the model to predict the ORIGINAL token at every
    one of these chosen positions (not just the ones literally replaced
    with [MASK]).

This repo's BERT is a from-scratch, single-sentence MLM pretraining model
only -- BERT's other pretraining task (Next Sentence Prediction, over a
pair of sentences with a [SEP] token and segment embeddings) and any
downstream fine-tuning head are both omitted; MLM alone is enough to
demonstrate what bidirectional self-attention buys you over a causal mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """q, k, v: (B, h, T, d_k). Same formula as models/transformer/model.py
    (duplicated here deliberately -- each model dir in this repo is
    self-contained, the same convention used e.g. between odenet/liquidode
    in cnn-playground or fno/pino in sciml-playground)."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
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

    def forward(self, q_in, k_in, v_in, mask=None) -> torch.Tensor:
        b, t_q, d_model = q_in.shape
        q = self._split_heads(self.w_q(q_in))
        k = self._split_heads(self.w_k(k_in))
        v = self._split_heads(self.w_v(v_in))
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, t_q, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + FFN, both with a residual + LayerNorm
    (post-norm, matching BERT's original implementation)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask=None)))  # no mask -> bidirectional
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class BERTModel(nn.Module):
    """(B, T) token ids -> (B, T, vocab_size) logits, one prediction per
    position (only the masked positions' logits are used in the MLM loss,
    computed in example.py via ignore_index on the unmasked labels)."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)  # LEARNED, unlike models/transformer's sinusoidal PE
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        x = self.drop(self.token_embed(tokens) + self.pos_embed(positions))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.mlm_head(x)
