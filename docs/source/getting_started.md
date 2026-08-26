# Getting started

## What every model here is built from

Every model in this repo is hand-written from scratch: scaled dot-product
attention, multi-head projection, causal masking, cross-attention, and
positional encoding (sinusoidal, learned, or rotary/RoPE) are all literal
Python in `models/*/model.py` -- never `torch.nn.MultiheadAttention` or a
pre-built architecture import. The building blocks every model *does*
reuse are `nn.Linear` and plain tensor ops (`torch.matmul`, `softmax`,
`torch.autograd`). See [`liquid-nn-playground`'s getting-started
page](https://github.com/agpoks/liquid-nn-playground/blob/main/docs/source/getting_started.md)
and [`cnn-playground`'s](https://github.com/agpoks/cnn-playground/blob/main/docs/source/getting_started.md)
for the same from-scratch philosophy applied to recurrent cells and
convolutions respectively.

## Attention and architecture, in detail

Two dedicated pages cover this properly, separately from any one model:

- {doc}`attention_explained` -- what attention computes, why the
  $\sqrt{d_k}$ scaling exists, multi-head attention, and the three
  masking patterns (none / causal / cross) that every model here is
  built from, each with a diagram and verbatim code.
- {doc}`encoder_decoder` -- how those masking patterns combine into
  encoder-only, decoder-only, and encoder-decoder models, with a
  side-by-side diagram and the real `EncoderLayer`/`DecoderLayer` code.

Building your own model on top of these conventions? See
{doc}`adding_a_model`.

## Install and run

```bash
git clone https://github.com/agpoks/transformer-playground.git
cd transformer-playground
pip install -e ".[notebooks]"
python models/gpt/example.py --device auto
```

Every example script takes `--device {auto,cpu,cuda,mps}`. Every model also
has a matching `example.ipynb`.
