"""Decision Transformer: control as return-conditioned sequence modeling.

Reference: Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas,
Mordatch, "Decision Transformer: Reinforcement Learning via Sequence
Modeling", NeurIPS 2021 (arXiv:2106.01345). See papers/README.md (bibtex
key `chen2021decisiontransformer`).

Core idea: instead of fitting a value function or a policy gradient, treat
an entire (return-to-go, state, action) trajectory as one sequence and
train an ordinary causal (GPT-style) transformer to predict the next
action -- conditioned on a *target* return-to-go supplied up front, so at
generation time you steer behavior by asking for a return, not by
re-optimizing anything.

=== HONEST DATA-ADAPTATION NOTE (read this before anything else) ===

This repo has no per-vehicle (state, action, reward)-labeled control
dataset. What it *does* have is the real NGSIM US-101 macroscopic traffic
field already validated in `sciml-playground` (density-like and speed-like
channels, binned from real vehicle observations) -- see
`transformer_playground/data/datasets.py`'s `load_ngsim_traffic_field`
(downloaded independently here, not a cross-repo import, so this repo has
no dependency on sciml-playground being installed).

`build_control_trajectories` below is this repo's OWN, EXPLICIT
reinterpretation of that aggregate field as an offline-imitation control
dataset, honestly documented as an adaptation, not as literally
reward-labeled RL data:

  - one "trajectory" = one spatial bin's time series (a fixed point along
    the road, observed over time)
  - state_t     = (density_t, speed_t) at that bin
  - action_t    = speed_{t+1} - speed_t -- the OBSERVED change in speed,
                  used as a proxy control input (imitation-style: "what
                  change in speed did this bin's traffic exhibit")
  - reward_t    = -(v_free - speed_t)^2, v_free = the 95th percentile of
                  speed over all bins/timesteps with an actual vehicle
                  observation -- a free-flow-speed target, so reward is
                  highest when traffic moves near free-flow speed and
                  penalized under congestion
  - return-to-go_t = discounted sum of reward_t..reward_{T-2} (gamma=0.99)

Timesteps/bins with NO vehicle observation (density == 0) have a
meaningless speed value (0, not "stopped traffic") and are excluded: only
context windows where every timestep in the window has density > 0 at
both t and t+1 are used for training/eval. Real coverage is uneven across
time (checked empirically: splitting by time left only ~11 valid training
windows vs. ~100 test windows at a useful context length), so
`example.py` splits by SPATIAL BIN instead -- the first ~80% of spatial
bins' trajectories are train, the rest are test, each using its full real
time range -- which is a defensible choice here since each spatial bin is
already treated as an independent "trajectory" (episode), not a single
trajectory being forecast forward in time.

The MECHANISM (return-conditioned causal sequence modeling over
interleaved (R, s, a) tokens, exactly as the paper defines it) is
faithfully implemented below. What's adapted is only the *data*, and that
adaptation is stated here plainly rather than silently.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def build_control_trajectories(
    field: torch.Tensor, context_len: int = 15, gamma: float = 0.99
):
    """field: (2, T, S) density/speed, e.g. from load_ngsim_traffic_field.

    Returns (states, actions, returns_to_go, valid_starts):
      - states:  (T-1, S, 2)  state_t = (density_t, speed_t)
      - actions: (T-1, S)     action_t = speed_{t+1} - speed_t
      - returns_to_go: (T-1, S)  discounted sum of future reward from t
      - valid_starts: list of (t0, s) pairs such that every one of the
        context_len consecutive timesteps starting at t0, at spatial bin
        s, has a real vehicle observation (density > 0) at both t and
        t+1 -- i.e. a window with NO synthesized/missing-data steps.
    """
    density, speed = field[0], field[1]  # (T, S)
    T, S = density.shape

    observed = (density > 0).numpy()
    v_free_samples = speed[density > 0].numpy()
    v_free = float(np.percentile(v_free_samples, 95)) if v_free_samples.size else 1.0

    states = torch.stack([density[:-1], speed[:-1]], dim=-1)  # (T-1, S, 2)
    actions = speed[1:] - speed[:-1]  # (T-1, S)
    rewards = -((v_free - speed[:-1]) ** 2)  # (T-1, S)

    step_valid = observed[:-1] & observed[1:]  # (T-1, S)
    rewards_masked = (rewards.numpy() * step_valid).astype(np.float32)

    returns_to_go = np.zeros_like(rewards_masked)
    running = np.zeros(S, dtype=np.float32)
    for t in range(T - 2, -1, -1):
        running = rewards_masked[t] + gamma * running
        returns_to_go[t] = running
    returns_to_go = torch.tensor(returns_to_go)

    valid_starts = []
    for s in range(S):
        col = step_valid[:, s]
        run = 0
        for t in range(T - 1):
            run = run + 1 if col[t] else 0
            if run >= context_len:
                valid_starts.append((t - context_len + 1, s))

    return states, actions, returns_to_go, valid_starts


def causal_mask(n: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


class CausalSelfAttention(nn.Module):
    """Self-contained (not imported from models/gpt) causal multi-head
    attention over the flattened (R, s, a) x context_len token sequence."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d_model = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal_mask(t, x.device), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DTBlock(nn.Module):
    """Pre-norm, same choice as models/gpt."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecisionTransformerModel(nn.Module):
    """Separate nn.Linear embeddings for return-to-go, state, action (the
    paper's real, specific design choice -- NOT a shared embedding table),
    plus one shared learned per-timestep embedding added to all three
    tokens at that timestep. The three token streams are interleaved as
    (R_0, s_0, a_0, R_1, s_1, a_1, ...) before the causal transformer, and
    the action-prediction head reads out at every *state* token's
    position -- the paper's actual readout point.
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        max_context_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_return = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(action_dim, d_model)
        self.embed_timestep = nn.Embedding(max_context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(
        self, returns_to_go: torch.Tensor, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """returns_to_go, actions: (B, L, 1). states: (B, L, state_dim).
        Returns predicted actions (B, L, action_dim), one per state-token
        readout position.
        """
        b, length, _ = states.shape
        timesteps = torch.arange(length, device=states.device)
        t_embed = self.embed_timestep(timesteps)  # (L, d_model)

        r = self.embed_return(returns_to_go) + t_embed
        s = self.embed_state(states) + t_embed
        a = self.embed_action(actions) + t_embed

        # interleave (R_0, s_0, a_0, R_1, s_1, a_1, ...) -> (B, 3L, d_model)
        tokens = torch.stack([r, s, a], dim=2).reshape(b, 3 * length, self.d_model)
        x = self.drop(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        # readout at every state token's position: indices 1, 4, 7, ... (0-indexed: 3*i+1)
        state_positions = x[:, 1::3, :]  # (B, L, d_model)
        return self.action_head(state_positions)
