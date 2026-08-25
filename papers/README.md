# Papers

For each model in `models/`, the paper picked here is the one that
introduces the architectural idea in its original form (or, for models
that are this repo's own assembled combination, an explicit statement that
no single paper does exactly that). BibTeX for all references is in
[`references.bib`](references.bib).

| Model | Paper | Year | Link |
|---|---|---|---|
| [Transformer](../models/transformer) | Attention Is All You Need | NeurIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| [BERT-style](../models/bert) | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | NAACL 2019 | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |
| [GPT-style](../models/gpt) | Improving Language Understanding by Generative Pre-Training / Language Models are Unsupervised Multitask Learners | 2018 / 2019 | OpenAI preprints, no arXiv id |
| [GPT-style, RoPE variant](../models/gpt) | RoFormer: Enhanced Transformer with Rotary Position Embedding | 2021 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| [Linear attention](../models/linattn) | Rethinking Attention with Performers | ICLR 2021 | [arXiv:2009.14794](https://arxiv.org/abs/2009.14794) |

## Why these nine, and not others

Each model isolates one structurally different idea about what attention
can be used for or how it can be masked/arranged, rather than being a
size/dataset variant of another model already here:

- **Transformer** is the origin: self-attention (bidirectional, in the
  encoder), causal self-attention (in the decoder), and cross-attention
  (decoder queries against encoder keys/values) all appear here for the
  first time, in one model.
- **BERT-style** removes the decoder entirely and removes the causal mask
  from the remaining encoder -- every position sees every other position,
  trained with a masked-prediction objective instead of next-token
  prediction. It's the natural "what if attention is only ever
  bidirectional" question.
- **GPT-style** is the other half of that split: causal self-attention
  only, no encoder, next-token prediction instead of masked in-fill. Its
  `pos_encoding` flag also demonstrates the field's actual historical
  progression in positional encoding -- fixed sinusoidal ({doc}`Transformer <../models/transformer>`)
  to learned ({doc}`BERT-style <../models/bert>` and GPT-2's real choice)
  to rotary/relative (RoPE, the modern default) -- inside one model
  instead of three separate ones.
- **Linear attention** answers the compute-cost question directly:
  identical task/architecture to GPT-style, differing only in the
  attention mechanism (FAVOR+ random features + cumulative sums instead
  of an explicit score matrix), with a real measured latency comparison
  at increasing sequence length as the payoff.

**ViT (Vision Transformer) is deliberately not duplicated here** -- it
already lives in
[`cnn-playground`](https://github.com/agpoks/cnn-playground/tree/main/models/vit)
as the repo's non-convolutional contrast baseline; this repo links to it
rather than rebuilding it.

_(remaining sections filled in as PatchTST-style, Conformer, Perceiver,
Decision Transformer, and the Tire-Patch-Wear Transformer are added.)_
