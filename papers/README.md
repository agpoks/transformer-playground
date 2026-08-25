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
| [PatchTST-style](../models/patchtst) | A Time Series is Worth 64 Words: Long-term Forecasting with Transformers | ICLR 2023 | [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) |
| [Perceiver](../models/perceiver) | Perceiver: General Perception with Iterative Attention | ICML 2021 | [arXiv:2103.03206](https://arxiv.org/abs/2103.03206) |
| [Conformer](../models/conformer) | Conformer: Convolution-augmented Transformer for Speech Recognition | 2020 | [arXiv:2005.08100](https://arxiv.org/abs/2005.08100) |
| [Decision Transformer](../models/decisiontransformer) | Decision Transformer: Reinforcement Learning via Sequence Modeling | NeurIPS 2021 | [arXiv:2106.01345](https://arxiv.org/abs/2106.01345) |
| [Tire-Patch-Wear Transformer](../models/tirewear) | *(this repo's own combination, not one paper)* -- brush-model contact-patch mechanics + Archard's wear law | 2012 / 1953 | Pacejka, *Tire and Vehicle Dynamics* (no arXiv id); [Archard, DOI:10.1063/1.1721448](https://pubs.aip.org/aip/jap/article/24/8/981/160178) |

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

- **PatchTST-style** shows attention generalizing to a completely
  different modality (time series) via **patching** -- exactly ViT's
  "split into patches, project each to a token" idea, but 1D and over
  time instead of 2D and over space. Its **channel independence** design
  (every variate patch-embedded and encoded by the *same* weights,
  never mixed) is a real, checkable choice worth calling out on its own:
  it is not the same idea as patching, just paired with it in this paper.
- **Perceiver** shows attention generalizing to *any* modality at all, by
  a different mechanism than patching: a fixed-size learned latent array
  cross-attends into an arbitrarily large raw input array (here, literally
  every raw CIFAR-10 pixel, no convolutional patchification), making
  compute linear in input size instead of quadratic. Complementary to,
  not a restatement of, PatchTST-style's patching idea -- patching reduces
  sequence length by tokenizing groups of input elements together;
  Perceiver keeps every input element as its own token and instead bounds
  the *query* side of attention.

- **Conformer** shows attention doesn't have to be pure: each block
  sandwiches bidirectional self-attention (global context) between real
  convolution (a depthwise conv with a GLU gate, giving local inductive
  bias a fixed-size kernel captures naturally) and two half-weighted
  feed-forwards (the "macaron" structure). This is the same CNN+attention
  hybrid theme as CoAtNet, applied here to audio instead of vision -- a
  genuinely different combination than any other model in this repo, none
  of which mix a real learned convolution into the attention block itself.

- **Decision Transformer** applies attention to *control*, not perception:
  the entire sequence is a trajectory (return-to-go, state, action)
  rather than a token stream, and the mechanism that steers behavior is
  return-conditioning, not a target label. The closely related
  Trajectory Transformer (Janner et al. 2021,
  [arXiv:2106.02039](https://arxiv.org/abs/2106.02039) -- discretizes
  every scalar dimension and repurposes beam search as a planner) was
  deliberately **not** built here: Decision
  Transformer is simpler and more canonical, needing no discretization or
  search machinery, while still demonstrating the same "control as
  sequence modeling" idea. This repo has no per-agent RL-labeled dataset,
  so it reinterprets the real NGSIM traffic field (already used in
  `sciml-playground`) as an offline-imitation control dataset -- see
  {doc}`the model's docs page <../models/decisiontransformer>` for the
  full, explicit adaptation.

- **Tire-Patch-Wear Transformer** is this repo's own combination, stated
  explicitly as such (no published paper does this) -- spatial
  self-attention over tire contact-patch positions, trained on
  physics-simulated data generated from real brush-model contact
  mechanics (Pacejka 2012) and Archard's classical wear law (1953). The
  physics simulator itself is an honest re-derivation of the real,
  already-validated `brush_patch.py`/`wear.py` modules in the sibling
  [`tire_physics_nn`](https://github.com/agpoks/tire_physics_nn) project
  (credited, not imported -- this repo stays self-contained). It is also
  the only model here whose token is neither a linguistic unit, an image
  patch, a time step, nor a trajectory step, but a fixed *spatial*
  position on a physical object -- a fourth kind of "what is a token"
  answer, alongside PatchTST-style's time patches, Perceiver's raw
  pixels, and Decision Transformer's trajectory steps.

**ViT (Vision Transformer) is deliberately not duplicated here** -- it
already lives in
[`cnn-playground`](https://github.com/agpoks/cnn-playground/tree/main/models/vit)
as the repo's non-convolutional contrast baseline; this repo links to it
rather than rebuilding it.

## The whole lineup, in one sentence each

Attention generalizes across: **masking pattern** (Transformer's
bidirectional-encoder/causal-decoder split, BERT's pure bidirectionality,
GPT's pure causality), **positional scheme** (Transformer's fixed
sinusoidal, BERT/GPT-2's learned, RoFormer's rotary), **efficiency
mechanism** (Performer's linear-time random-feature attention vs. GPT's
quadratic exact attention), **tokenization strategy** (PatchTST's time
patches, Perceiver's raw untouched elements bounded on the query side
instead, Tire-Patch-Wear's fixed spatial positions), **hybrid
composition** (Conformer folding real convolution into the attention
block itself), and **task framing** (Decision Transformer treating an
entire control trajectory, not a token stream, as the sequence). Nine
models, nine different axes of the same underlying mechanism -- not nine
sizes of the same idea.
