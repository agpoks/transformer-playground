# Transformer

**Paper:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
Polosukhin, *"Attention Is All You Need"*, NeurIPS 2017 —
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Instead of processing a sequence step-by-step (RNN) or with a local
sliding window (CNN), let every position attend directly to every other
position in one matrix multiply: `Attention(Q,K,V) = softmax(QK^T /
sqrt(d_k)) V`. Multi-head attention runs this in parallel on several
learned linear projections of `Q`/`K`/`V`. This model uses attention three
ways: encoder self-attention (bidirectional, no mask), decoder
self-attention (causal mask — token `i` only sees `<= i`), and decoder
cross-attention (`Q` from the decoder, `K`/`V` from the encoder's output —
this is how the source sentence's information reaches the translation
being generated). Positional information is a fixed sinusoidal function of
position, not learned (contrast with `models/bert`).

## Files

- `model.py` — `scaled_dot_product_attention`, `MultiHeadAttention`,
  sinusoidal positional encoding, `EncoderLayer`/`DecoderLayer`,
  `TransformerModel` (full encoder-decoder).
- `example.py` — trains on real English→French sentence pairs
  (`--device {auto,cpu,cuda,mps}`), from-scratch word-level tokenizer.
- `example.ipynb` — same walkthrough with a loss plot.

## Run it

```bash
pip install -e .
python models/transformer/example.py --device auto
# or open models/transformer/example.ipynb
```
