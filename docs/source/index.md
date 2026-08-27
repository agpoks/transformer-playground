# transformer-playground

A playground for **attention and Transformer architectures** — implement,
run, and benchmark nine attention-based models spanning language, vision,
audio, time series, control/planning, and mechanical engineering, all
hand-written from `nn.Linear`/`torch.autograd` primitives (never
`nn.MultiheadAttention`) on real datasets. Fourth companion project, after
`liquid-nn-playground`, `sciml-playground`, and `cnn-playground`.

## How to read this documentation

| | |
|---|---|
| **1. Start here** | {doc}`Getting started <getting_started>` — install and run your first model. |
| **2. What attention is** | {doc}`Attention, from scratch <attention_explained>` — the equation, why the $\sqrt{d_k}$ scaling exists, multi-head attention, and the three masking patterns every model here is built from. |
| **3. How it's arranged** | {doc}`Encoder vs. decoder, from scratch <encoder_decoder>` — the same attention block arranged three ways (encoder-only, decoder-only, encoder-decoder), with a side-by-side diagram. |
| **4. The models, one at a time** | Nine pages, one per model, equation → code → real results: {doc}`Transformer <models/transformer>`, {doc}`BERT-style <models/bert>`, {doc}`GPT-style <models/gpt>`, {doc}`PatchTST-style <models/patchtst>`, {doc}`Conformer <models/conformer>`, {doc}`Performer/linattn <models/linattn>`, {doc}`Perceiver <models/perceiver>`, {doc}`Decision Transformer <models/decisiontransformer>`, {doc}`Tire-Patch-Wear Transformer <models/tirewear>`. |
| **5. How they compare** | {doc}`Model comparison <model_comparison>` and {doc}`Benchmark results <benchmark_results>` — what changes structurally between models, and the real, measured params/size/train-time/accuracy/inference-latency numbers. |
| **6. Add your own** | {doc}`Adding a model <adding_a_model>` — this repo's own conventions, as a template. |

Every model ships with a runnable Python example and a Jupyter notebook.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start here

getting_started
attention_explained
encoder_decoder
adding_a_model
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Models

models/transformer
models/bert
models/gpt
models/linattn
models/patchtst
models/perceiver
models/conformer
models/decisiontransformer
models/tirewear
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Evidence

model_comparison
benchmark_results
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

datasets
benchmarks
papers
```
