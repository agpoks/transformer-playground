# Adding your own model

A practical checklist for adding a new model to this repo, following the
same conventions every one of the nine existing models already uses --
useful whether you're extending this repo or just want a template for
your own from-scratch attention project.

## 1. Files every model needs

```
models/<name>/
├── model.py         the architecture, hand-written (nn.Linear/nn.Conv only)
├── example.py       argparse CLI, trains + evaluates, prints one RESULT: line
├── example.ipynb    same walkthrough, as a notebook
└── README.md        idea in one paragraph, file list, how to run it
```

`model.py` should be **self-contained**: write your attention mechanism
again in this file rather than importing it from another model's
`model.py`, even if it's nearly identical to one that already exists --
{doc}`models/gpt` and {doc}`models/linattn` each write their own causal
attention rather than sharing one, the same convention `sciml-playground`
uses between `models/fno`/`models/pino` and `cnn-playground` uses between
`models/odenet`/`models/liquidode` -- a small amount of duplication buys
each model directory independence and makes every file readable on its
own.

## 2. The `RESULT:` line

Every `example.py` must print exactly one line in this format at the end
of its run:

```
RESULT: model=<name> metric_name=<name> metric=<value> params=<n> train_time_s=<value>
```

`benchmarks/run_cluster.py` regex-parses this line to build comparison
tables across a cluster of models that share a task -- see
`benchmarks/run_cluster.py`'s `RESULT_RE` if you want the exact pattern.
Pick whatever `metric_name` is natural for your task (`test_acc`,
`val_loss`, `test_mse`, ...); the script doesn't require it to match other
models' metric names, only the format above.

## 3. Benchmark clusters

Add a `benchmarks/configs/<cluster>_suite.yaml`:

```yaml
# One-line description of what this cluster tests and why these models
# are grouped together.
models:
  - your_model_name
epochs: 20
```

If your model shares a real, fair comparison with an existing one (same
task, same data, same budget -- like {doc}`models/gpt` and
{doc}`models/linattn` sharing `language_suite.yaml`), add it to that
existing cluster instead of making a new one. Otherwise, a solo cluster
(most models here are solo -- different tasks aren't fairly comparable)
is the norm, not an exception.

## 4. Dataset loaders

Add a `load_<dataset>()` function to
`transformer_playground/data/datasets.py`, export it from
`transformer_playground/data/__init__.py`, and import it in your
`example.py` as `from transformer_playground.data import load_<dataset>`.
Real data only -- if no public dataset exists for what you're modeling
(as with {doc}`models/tirewear`), a physics-simulated or procedurally
generated target is an acceptable, clearly-labeled substitute; a
synthetic placeholder standing in for data that *does* exist publicly is
not. Every loader downloads with a `User-Agent` header (a real lesson
from this repo's own history -- `manythings.org` returns HTTP 406
without one) and caches into `data_cache/` (gitignored).

## 5. Docs page

Add `docs/source/models/<name>.md`, matching the structure every existing
page uses: intro citing the real paper (or stating plainly that this is
an assembled combination, if it is -- see {doc}`models/tirewear` for that
honesty pattern), `## The equation` (the real math, in LaTeX), `## How
it's built` (a verbatim code excerpt, not paraphrased), a pre-rendered PNG
diagram plus an identical live `.. plot::` block using
`transformer_playground.utils.diagrams`, an explicit `## Simplifications
vs. the paper` section, `## Try it`, `## References`. Render the diagram
via a throwaway script and look at the PNG before committing -- don't
trust an unrendered plotting script.

Then wire it in:
- `docs/source/index.md` -- add `models/<name>` to the toctree.
- `docs/source/model_comparison.md` -- add a row.
- `papers/references.bib` + `papers/README.md` -- add the citation.
- root `README.md` -- add to the model table, check it off in Status.
- `datasets/README.md`, `benchmarks/README.md` -- document the dataset/cluster.

## 6. Verify before committing

- Smoke test: random input through `forward()` and `.backward()`, check
  the output shape and that every parameter has a non-`None` `.grad`.
- A real training run (or an honestly-labeled partial/subset run if the
  machine is under heavy load -- state the exact numbers and why in the
  commit message and docs, never fabricate a number).
- `python3 -m sphinx -b html docs/source docs/_build_test -q` builds
  clean (a few pre-existing cosmetic warnings -- duplicate citations on
  the combined `papers.md` page, `myst.xref_missing` on plain relative
  links -- are expected and fine; a *new* warning or error is not), then
  delete `docs/_build_test`.

## 7. Try it

```bash
python models/<name>/example.py --device auto
python benchmarks/run_cluster.py --cluster <cluster> --device auto
```
