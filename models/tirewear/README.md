# Tire-Patch-Wear Transformer

**This is this repo's own combination, not a reproduction of a published
paper** -- as far as this session's literature search could find, no
published work applies attention/transformers to tire contact-patch wear.
Combines spatial self-attention over patch-position tokens (the same
mechanism `models/patchtst`/`models/perceiver` use) with real brush-model
contact-patch mechanics (Pacejka, *Tire and Vehicle Dynamics*, 2012) and
Archard's wear law (Archard, *Contact and Rubbing of Flat Surfaces*, 1953).
See [`papers/README.md`](../../papers/README.md).

The physics-simulator equations here are an honest re-derivation of the
real, already-implemented brush-model/wear-rate code in the sibling
project [`tire_physics_nn`](https://github.com/agpoks/tire_physics_nn)
(`tire_nn/physics/brush_patch.py`, `tire_nn/physics/wear.py`) -- credited,
not imported (every repo in this family is self-contained).

## Idea in one paragraph

No public tire-contact-patch-wear dataset exists at this spatial
resolution, so training data comes from a small, real (not arbitrary)
physics simulator: a parabolic contact-patch pressure distribution, a
brush-model adhesion/sliding split (bristle deflection grows linearly from
the leading edge until it hits the local friction bound, then the tread
slides), and an Archard-consistent wear accumulation (wear rate
proportional to pressure x sliding distance, enforced non-negative via
`softplus` -- structurally, not as a loss penalty, mirroring
`tire_physics_nn`'s own `wear_rate` invariant). The transformer tokenizes
the patch by position, attends bidirectionally across positions (every
position is present simultaneously in one revolution's footprint -- there
is no "future position" to mask), and predicts next-revolution wear
increments.

## Files

- `model.py` -- the physics simulator (`simulate_wear_trajectory` and its
  brush-model helpers) AND `TirePatchWearTransformer` (hand-written
  bidirectional self-attention over patch-position tokens).
- `example.py` -- builds a training/test set from many simulated tires
  (varying slip), trains with a physics-encoded (softplus) monotonicity
  guarantee on predicted wear increments, `--device`.
- `example.ipynb` -- same walkthrough with a ground-truth-vs-predicted wear
  pattern plot.

## Run it

```bash
pip install -e .
python models/tirewear/example.py --device auto
# or open models/tirewear/example.ipynb
```
