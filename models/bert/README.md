# BERT-style

**Paper:** Devlin, Chang, Lee, Toutanova, *"BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding"*, NAACL 2019 —
[arXiv:1810.04805](https://arxiv.org/abs/1810.04805). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Encoder-only, no causal mask anywhere: every token attends to every other
token, both before and after it — genuinely bidirectional, unlike a
left-to-right decoder. Positional information is a *learned* embedding
table (contrast with `models/transformer`'s fixed sinusoidal PE). Trained
with masked-language-modeling: for ~15% of positions, the real BERT recipe
is followed exactly — 80% replaced with `[MASK]`, 10% replaced with a
random vocabulary token, 10% left unchanged — and the model always
predicts the *original* token at every one of these positions. This
repo's BERT is MLM-pretraining only (no Next Sentence Prediction, no
fine-tuning head) — enough to demonstrate what bidirectional attention
buys you that a causal mask can't.

## Files

- `model.py` — self-contained `MultiHeadAttention`/`FeedForward`
  (duplicated from `models/transformer`, not imported — each model dir is
  self-contained), `EncoderLayer` (bidirectional, no mask), `BERTModel`
  (learned positional embeddings + MLM head).
- `example.py` — trains on real WikiText-2 with the exact 80/10/10 masking
  recipe (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with a loss plot.

## Run it

```bash
pip install -e .
python models/bert/example.py --device auto
# or open models/bert/example.ipynb
```
