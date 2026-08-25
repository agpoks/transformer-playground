# Perceiver

**Paper:** Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira, *"Perceiver:
General Perception with Iterative Attention"*, ICML 2021 —
[arXiv:2103.03206](https://arxiv.org/abs/2103.03206). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Self-attention over the raw input directly costs O(input_len²) — fine for
a few hundred language tokens, not for e.g. every raw pixel of an image.
Perceiver instead keeps a small, **fixed-size learned latent array** and
repeatedly (1) **cross-attends** from the latents (as queries) to the
(much larger) raw input array (as keys/values) — cost is linear in input
size, since the latent array's size never changes — then (2) runs ordinary
self-attention among the latents only (cheap, since the array is small).
This decouples compute from input size and needs no modality-specific
input structure at all (no convolutional patchification here — CIFAR-10 is
fed in as 1024 raw pixel tokens + a 2D position feature, nothing else).

## Files

- `model.py` — `CrossAttention` (latents=Q, input=K/V),
  `SelfAttention` (among latents), `PerceiverModel`.
- `example.py` — trains on real CIFAR-10 (raw flattened pixels, no
  convolution), `--n-latents`/`--device`.
- `example.ipynb` — same walkthrough.

## Run it

```bash
pip install -e .
python models/perceiver/example.py --device auto
# or open models/perceiver/example.ipynb
```
