"""Train Decision Transformer on real NGSIM traffic data, reinterpreted as
an offline-imitation control dataset (see model.py's honest data-adaptation
note for exactly how, and why this is not literally reward-labeled RL data).

    python models/decisiontransformer/example.py --device auto --epochs 10

See model.py for the return-conditioned causal-attention architecture and
papers/README.md for the reference (Chen et al. 2021).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_ngsim_traffic_field  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import DecisionTransformerModel, build_control_trajectories  # noqa: E402


def make_batch(states, actions, rtg, starts, context_len, batch_size, device):
    idx = torch.randint(0, len(starts), (batch_size,))
    S_list, A_list, R_list, Y_list = [], [], [], []
    for i in idx:
        t0, s = starts[i]
        S_list.append(states[t0 : t0 + context_len, s, :])
        A_list.append(actions[t0 : t0 + context_len, s].unsqueeze(-1))
        R_list.append(rtg[t0 : t0 + context_len, s].unsqueeze(-1))
        Y_list.append(actions[t0 : t0 + context_len, s].unsqueeze(-1))
    return (
        torch.stack(R_list).to(device),
        torch.stack(S_list).to(device),
        torch.stack(A_list).to(device),
        torch.stack(Y_list).to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Each epoch = --steps-per-epoch batches")
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--space-bins", type=int, default=64)
    parser.add_argument("--time-bins", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    field = load_ngsim_traffic_field(space_bins=args.space_bins, time_bins=args.time_bins)
    states, actions, rtg, valid_starts = build_control_trajectories(field, context_len=args.context_len)
    S = states.shape[1]
    s_train_end = int(0.8 * S)
    train_starts = [(t0, s) for (t0, s) in valid_starts if s < s_train_end]
    test_starts = [(t0, s) for (t0, s) in valid_starts if s >= s_train_end]
    print(
        f"real NGSIM-derived control trajectories: {len(train_starts)} train windows "
        f"({s_train_end} spatial bins), {len(test_starts)} test windows ({S - s_train_end} spatial bins), "
        f"context_len={args.context_len}"
    )
    if len(train_starts) < args.batch_size or len(test_starts) < 4:
        raise SystemExit(
            "Too few valid (fully-observed) windows for this space_bins/time_bins/context_len "
            "combination -- try a smaller --context-len or different --space-bins/--time-bins."
        )

    model = DecisionTransformerModel(
        state_dim=2,
        action_dim=1,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_context_len=args.context_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    @torch.no_grad()
    def eval_loss(n_batches: int = 10) -> float:
        model.eval()
        total = 0.0
        for _ in range(n_batches):
            R, S_, A, Y = make_batch(states, actions, rtg, test_starts, args.context_len, args.batch_size, device)
            pred = model(R, S_, A)
            total += loss_fn(pred, Y).item()
        model.train()
        return total / n_batches

    t0 = time.perf_counter()
    val_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _ in range(args.steps_per_epoch):
            R, S_, A, Y = make_batch(states, actions, rtg, train_starts, args.context_len, args.batch_size, device)
            pred = model(R, S_, A)
            loss = loss_fn(pred, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = eval_loss()
        print(f"epoch {epoch:3d} | train_loss {loss.item():.4f} | val_action_mse {val_loss:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=decisiontransformer metric_name=val_action_mse metric={val_loss:.4f} "
        f"params={n_params} train_time_s={train_time:.2f}"
    )


if __name__ == "__main__":
    main()
