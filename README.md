# transformer-playground

A playground for **Transformer architectures and attention**: implement,
run, and benchmark nine attention-based models spanning the field's core
ideas -- language, vision, audio, time series, control/planning, and a
mechanical-engineering application -- side by side on real datasets.
Fourth companion project, after
[`liquid-nn-playground`](https://github.com/agpoks/liquid-nn-playground),
[`sciml-playground`](https://github.com/agpoks/sciml-playground), and
[`cnn-playground`](https://github.com/agpoks/cnn-playground), same layout
and philosophy: every attention mechanism is hand-written from
`nn.Linear`/`torch.autograd` primitives -- never `nn.MultiheadAttention`,
never a pre-built architecture import.

| Idea | Model | Folder |
|---|---|---|
| The original: encoder-decoder, self- and cross-attention | **Transformer** | [`models/transformer`](models/transformer) |
| Bidirectional self-attention, encoder-only | **BERT-style** | [`models/bert`](models/bert) |
| Causal self-attention, decoder-only (+ RoPE variant) | **GPT-style** | [`models/gpt`](models/gpt) |
| Attention over time-series patches | **PatchTST-style** | [`models/patchtst`](models/patchtst) |
| Convolution-augmented attention, audio | **Conformer** | [`models/conformer`](models/conformer) |
| Linear-time attention, direct cost comparison to GPT | **Performer/Linformer** | [`models/linattn`](models/linattn) |
| Latent-bottleneck cross-attention, modality-agnostic | **Perceiver** | [`models/perceiver`](models/perceiver) |
| Control as sequence modeling | **Decision Transformer** | [`models/decisiontransformer`](models/decisiontransformer) |
| Spatial attention over a tire contact patch + physics-informed wear loss (this repo's own combination) | **Tire-Patch-Wear Transformer** | [`models/tirewear`](models/tirewear) |

Full paper references and why each one was picked: [`papers/README.md`](papers/README.md).
Docs: see [`docs/`](docs) (built on Read the Docs).

## Layout

```
transformer-playground/
├── models/<name>/    model.py, example.py, example.ipynb, README.md  (one per architecture)
├── transformer_playground/  shared package: device (cpu/gpu/mps) resolution, real dataset loaders
├── datasets/          dataset docs
├── benchmarks/        YAML suites, grouped by which models share a dataset/task
├── papers/            reference list, BibTeX
└── docs/              Sphinx / Read the Docs source
```

## Install

```bash
git clone https://github.com/agpoks/transformer-playground.git
cd transformer-playground
pip install -e ".[notebooks]"
```

## Run a model

```bash
python models/gpt/example.py --device auto
```

Every example script takes `--device {auto,cpu,cuda,mps}`. Every model also
has a matching `example.ipynb`.

## Status

- [ ] Transformer (Vaswani et al. 2017)
- [ ] BERT-style encoder
- [ ] GPT-style decoder (+ RoPE)
- [ ] PatchTST-style time-series transformer
- [ ] Conformer
- [ ] Performer/Linformer (linear attention)
- [ ] Perceiver
- [ ] Decision Transformer
- [ ] Tire-Patch-Wear Transformer
