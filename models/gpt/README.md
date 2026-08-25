# GPT-style

**Papers:** Radford, Narasimhan, Salimans, Sutskever, *"Improving Language
Understanding by Generative Pre-Training"*, 2018; Radford, Wu, Child, Luan,
Amodei, Sutskever, *"Language Models are Unsupervised Multitask
Learners"*, 2019 (GPT-2). Rotary positional variant: Su, Lu, Pan, Wen, Liu,
*"RoFormer: Enhanced Transformer with Rotary Position Embedding"*,
2021 — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Decoder-only: every position attends only to itself and earlier positions
(a causal mask), trained to predict the next token given everything
before it — no encoder, no cross-attention. Uses PRE-norm residual blocks
(LayerNorm before each sublayer), GPT-2's actual choice, contrasted with
the original Transformer's post-norm. Positional information is a
constructor flag: `"learned"` (GPT-2's real choice, a plain learned
embedding table) or `"rope"` (rotary position embedding — rotates each
attention head's query/key vectors by a position-dependent angle, applied
fresh inside every layer, giving relative-position-aware attention with no
learned table, the modern default in most current LLMs).

## Files

- `model.py` — `CausalSelfAttention` (with both `"learned"`/`"rope"`
  modes), `rope_cos_sin`/`apply_rope`, `GPTBlock` (pre-norm), `GPTModel`.
- `example.py` — trains on real Tiny Shakespeare (character-level),
  `--pos-encoding {learned,rope}`, `--device {auto,cpu,cuda,mps}`.
- `example.ipynb` — same walkthrough with a loss plot.

## Run it

```bash
pip install -e .
python models/gpt/example.py --device auto --pos-encoding learned
python models/gpt/example.py --device auto --pos-encoding rope
# or open models/gpt/example.ipynb
```
