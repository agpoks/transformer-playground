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

**ViT (Vision Transformer) is deliberately not duplicated here** -- it
already lives in
[`cnn-playground`](https://github.com/agpoks/cnn-playground/tree/main/models/vit)
as the repo's non-convolutional contrast baseline; this repo links to it
rather than rebuilding it.

_(remaining sections filled in as GPT-style, PatchTST-style, Conformer,
Performer/Linformer, Perceiver, Decision Transformer, and the
Tire-Patch-Wear Transformer are added.)_
