# Linear attention (Performer/FAVOR+)

**Paper:** Choromanski, Likhosherstov, Dohan, Song, Gane, Sarlos, Hawkins,
Davis, Mohiuddin, Kaiser, Belanger, Colwell, Weller, *"Rethinking
Attention with Performers"*, ICLR 2021 —
[arXiv:2009.14794](https://arxiv.org/abs/2009.14794). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Approximates the softmax attention kernel with an explicit, always-
positive random feature map `phi`, so `softmax(q.k) ~= phi(q).phi(k)`.
Because this factorizes into a dot product against a feature vector,
causal (autoregressive) attention can be computed *exactly* via running
cumulative sums over the sequence, giving `O(T)` complexity instead of
`O(T^2)` — verified exactly causal (changing a later token provably
changes nothing about earlier outputs). Otherwise identical to
`models/gpt` (same task, data, block layout, learned positional
embedding) so a benchmark isolates just the attention mechanism.

## Files

- `model.py` — `phi`/`causal_linear_attention` (FAVOR+, cumulative sums),
  `LinAttnBlock` (pre-norm, mirrors `models/gpt`'s `GPTBlock`), `LinAttnModel`.
- `example.py` — trains on real Tiny Shakespeare (same split as
  `models/gpt`), plus a real measured scaling comparison against
  `models/gpt` at three sequence lengths (128/2048/8192) showing the
  actual crossover point where linear attention starts winning in
  wall-clock latency.
- `example.ipynb` — same walkthrough with a loss plot.

## Run it

```bash
pip install -e .
python models/linattn/example.py --device auto
# or open models/linattn/example.ipynb
```
