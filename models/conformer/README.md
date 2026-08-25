# Conformer

**Paper:** Gulati, Qin, Chiu, Parmar, Zhang, Yu, Han, Wang, Zhang, Wu, Pang,
*"Conformer: Convolution-augmented Transformer for Speech Recognition"*,
2020 — [arXiv:2005.08100](https://arxiv.org/abs/2005.08100). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Attention doesn't have to be pure. Each Conformer block is a "macaron"
sandwich: a half-weighted feed-forward, then bidirectional self-attention
(global context), then a real convolution module (pointwise conv -> GLU ->
depthwise conv -> BatchNorm -> Swish -> pointwise conv, giving local
inductive bias), then a second half-weighted feed-forward, then a final
LayerNorm — residual connections around every one of the four sublayers.
Attention captures long-range structure, the conv module captures local
patterns a fixed-size kernel is naturally suited to — genuinely combining
the two rather than picking one.

## Files

- `model.py` — `FeedForwardModule`, `MultiHeadSelfAttention` (bidirectional,
  no causal mask — this is an ASR *encoder*), `ConvModule` (the real
  pointwise/GLU/depthwise/BN/Swish/pointwise sequence), `ConformerBlock`
  (the macaron ordering), `ConformerModel`.
- `example.py` — trains on real Google Speech Commands (core 10 words: yes,
  no, up, down, left, right, on, off, stop, go), log-mel spectrogram
  features via `torchaudio.transforms`, `--device {auto,cpu,cuda,mps}`.
- `example.ipynb` — same walkthrough with a training-curve plot.

Simplification vs. the paper: learned absolute positional embedding
instead of the original's Transformer-XL-style relative positional
encoding — stated explicitly, see `model.py`'s docstring and the docs page.

## Run it

```bash
pip install -e .
python models/conformer/example.py --device auto
# or open models/conformer/example.ipynb
```
