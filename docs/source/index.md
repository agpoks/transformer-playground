# transformer-playground

A playground for **attention and Transformer architectures** -- implement,
run, and benchmark nine attention-based models spanning language, vision,
audio, time series, control/planning, and mechanical engineering, all
hand-written from `nn.Linear`/`torch.autograd` primitives (never
`nn.MultiheadAttention`) on real datasets. Fourth companion project, after
`liquid-nn-playground`, `sciml-playground`, and `cnn-playground`.

Every model ships with a runnable Python example and a Jupyter notebook.

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
model_comparison
benchmark_results
models/transformer
models/bert
models/gpt
models/linattn
models/patchtst
models/perceiver
models/conformer
models/decisiontransformer
datasets
benchmarks
papers
```
