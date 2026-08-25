# Benchmarks

Like `cnn-playground`/`sciml-playground`, these nine models don't share one
input/output shape or task (translation vs. masked LM vs. causal LM vs.
time-series forecasting vs. audio vs. image classification vs. control vs.
tire-wear regression), so benchmarking is grouped by **task cluster**, most
of which hold a single model (kept in a consistent format rather than
forced into an unfair comparison), except where two models share a task by
design (GPT vs. its linear-attention variant, for a direct compute-cost
comparison).

Every `models/*/example.py` prints one final line in a common format:

```
RESULT: model=<name> metric_name=<name> metric=<value> params=<n> train_time_s=<value>
```

`run_cluster.py` runs every model in a cluster back-to-back with the same
`--device`/`--epochs`, parses that line, and prints a comparison table.

## Clusters

- `translation` -- Transformer, solo, real English-French pairs.
- `mlm` -- BERT-style, solo, real WikiText-2.
- `language` -- GPT-style and Linear attention (linattn), real Tiny
  Shakespeare, identical task/split for both -- the one genuine two-model
  comparison cluster, isolating the attention mechanism itself.

- `timeseries` -- PatchTST-style, solo, real ETTh1.
- `vision` -- Perceiver, solo, real CIFAR-10.
- `audio` -- Conformer, solo, real Google Speech Commands (core 10 words).

- `control` -- Decision Transformer, solo, real NGSIM traffic field
  reinterpreted as an offline-imitation control dataset (see
  `datasets/README.md` and the model's docs page for the honest
  adaptation).

_(remaining cluster filled in as the Tire-Patch-Wear Transformer is added.)_
