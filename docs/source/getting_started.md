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

## Scaled dot-product attention, from scratch

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

where $M$ is an additive mask (all zeros for full/bidirectional attention;
$-\infty$ above the diagonal for causal/decoder attention). In code, this
is nothing more than:

```python
def attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

Multi-head attention just runs this in parallel on `h` learned linear
projections of `Q`, `K`, `V`, then concatenates and projects the result back
down -- see any model's `model.py` for the exact `nn.Linear`-only
implementation, and its docs page for the architecture diagram.

## Install and run

```bash
git clone https://github.com/agpoks/transformer-playground.git
cd transformer-playground
pip install -e ".[notebooks]"
python models/gpt/example.py --device auto
```

Every example script takes `--device {auto,cpu,cuda,mps}`. Every model also
has a matching `example.ipynb`.
