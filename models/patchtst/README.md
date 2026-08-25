# PatchTST-style

**Paper:** Nie, Nguyen, Sinthong, Kalagnanam, *"A Time Series is Worth 64
Words: Long-term Forecasting with Transformers"*, ICLR 2023 —
[arXiv:2211.14730](https://arxiv.org/abs/2211.14730). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Attention generalizes to time series the same way ViT generalized it to
images: split each univariate series into overlapping **patches** along
the time axis, linearly project each patch to one token, and run an
ordinary bidirectional (encoder-only) transformer over the patch tokens
instead of one token per timestep. For a multivariate series, every
channel is patch-embedded and encoded by the **exact same weights**,
independently — **channel independence** — rather than mixed together, a
specific real design choice from the paper (not an approximation:
channels are folded into the batch dimension, so they structurally cannot
attend to each other).

## Files

- `model.py` — `PatchEmbedding` (unfold + linear projection + learned
  positional embedding), `EncoderBlock` (bidirectional self-attention +
  FFN), `PatchTSTModel` (channel-independent forecasting head).
- `example.py` — trains on real ETTh1 (Electricity Transformer
  Temperature, hourly), `--seq-len`/`--pred-len`/`--device`.
- `example.ipynb` — same walkthrough with a forecast plot.

## Run it

```bash
pip install -e .
python models/patchtst/example.py --device auto
# or open models/patchtst/example.ipynb
```
