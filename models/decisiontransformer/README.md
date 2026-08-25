# Decision Transformer

## Idea in one paragraph

Decision Transformer ({cite}`chen2021decisiontransformer` in the docs) casts
control as sequence modeling: a causal transformer sees an interleaved
sequence of (return-to-go, state, action) tokens and is trained to predict
the next action, conditioned on a target return supplied up front, rather
than fitting a value function or a policy gradient. **Honesty note**: this
repo has no per-vehicle RL-labeled dataset, so the real NGSIM traffic field
(already used in `sciml-playground`) is reinterpreted as an offline-
imitation control dataset -- one spatial bin's time series = one
trajectory, action = observed speed change, reward = closeness to a
free-flow-speed target. See `model.py`'s docstring for the full, explicit
data-adaptation note. The transformer mechanism itself is implemented
exactly as the paper defines it.

## Files

- `model.py` -- `DecisionTransformerModel` (separate R/s/a embeddings +
  shared timestep embedding, causal self-attention, action readout at
  every state token) and `build_control_trajectories` (the honest NGSIM ->
  control-trajectory adaptation).
- `example.py` -- trains on the real NGSIM-derived trajectories.
- `example.ipynb` -- same, as a notebook.

## Run it

```bash
python models/decisiontransformer/example.py --device auto --epochs 10
```
