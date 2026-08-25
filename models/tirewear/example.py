"""Train the Tire-Patch-Wear Transformer on physics-simulated contact-patch
wear trajectories (no public labeled dataset exists at this spatial
resolution -- see model.py's module docstring for the honest explanation and
the real brush-model/Archard's-law physics behind the simulator).

    python models/tirewear/example.py --device auto --epochs 30

See model.py for the architecture/simulator and papers/README.md for the
references (this is this repo's own combination, not a reproduction of any
single paper).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import TirePatchWearTransformer, simulate_wear_trajectory  # noqa: E402


def build_dataset(n_trajectories: int, n_revolutions: int, n_elements: int, seed_offset: int = 0):
    """Simulate `n_trajectories` independent tires (varying slip_mean per
    tire) and turn every (pressure_r, wear_cumulative_{r-1}) -> wear at
    revolution r into one supervised example. Returns (pressure, wear_so_far,
    target_increment), each (N, n_elements)."""
    g = torch.Generator().manual_seed(1000 + seed_offset)
    pressures, wears_so_far, targets = [], [], []
    for i in range(n_trajectories):
        slip_mean = 0.20 + 0.30 * torch.rand(1, generator=g).item()
        traj = simulate_wear_trajectory(
            n_elements=n_elements, n_revolutions=n_revolutions, slip_mean=slip_mean, seed=2000 + seed_offset + i
        )
        pressure = traj["pressure"]  # (R, n_elements)
        wear_cum = traj["wear_cumulative"]  # (R, n_elements)
        wear_prev = torch.cat([torch.zeros(1, n_elements), wear_cum[:-1]], dim=0)
        wear_inc = traj["wear_increment"]  # (R, n_elements), the real target
        pressures.append(pressure)
        wears_so_far.append(wear_prev)
        targets.append(wear_inc)
    return torch.cat(pressures), torch.cat(wears_so_far), torch.cat(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-elements", type=int, default=32)
    parser.add_argument("--n-revolutions", type=int, default=60)
    parser.add_argument("--n-train-trajectories", type=int, default=40)
    parser.add_argument("--n-test-trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    p_train, w_train, y_train = build_dataset(args.n_train_trajectories, args.n_revolutions, args.n_elements, seed_offset=0)
    p_test, w_test, y_test = build_dataset(args.n_test_trajectories, args.n_revolutions, args.n_elements, seed_offset=10_000)
    print(
        f"physics-simulated tire-patch-wear data: {p_train.shape[0]} train / {p_test.shape[0]} test "
        f"revolution-samples, {args.n_elements} patch positions each (see model.py's docstring for the "
        "brush-model/Archard's-law physics generating this, and datasets/README.md for why it's simulated)"
    )

    # Normalize pressure (wear starts at/near zero, left as-is so the model
    # sees the real physical scale it must predict).
    p_mean, p_std = p_train.mean(), p_train.std()
    p_train_n = (p_train - p_mean) / p_std
    p_test_n = (p_test - p_mean) / p_std

    model = TirePatchWearTransformer(n_elements=args.n_elements).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def predicted_increment(pressure_n: torch.Tensor, wear_so_far: torch.Tensor) -> torch.Tensor:
        """softplus(raw logits) -- structurally >= 0 for any logits, exactly
        mirroring tire_physics_nn's wear_rate(raw)=softplus(raw) invariant.
        This is a physics-ENCODED monotonicity guarantee (built into the
        forward computation), not a soft loss penalty that could be
        violated -- the stronger of the two options mentioned in the
        original plan, and consistent with the real precedent."""
        logits = model(pressure_n, wear_so_far)
        return F.softplus(logits)

    def eval_test() -> float:
        model.eval()
        with torch.no_grad():
            pred = predicted_increment(p_test_n.to(device), w_test.to(device))
            loss = nn.functional.mse_loss(pred, y_test.to(device)).item()
        model.train()
        return loss

    t0 = time.perf_counter()
    n = p_train_n.shape[0]
    test_mse = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            pb, wb, yb = p_train_n[idx].to(device), w_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            pred = predicted_increment(pb, wb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            test_mse = eval_test()
            print(f"epoch {epoch:3d} | train_mse {total_loss / n_batches:.5f} | test_mse {test_mse:.5f}")

    if test_mse is None:
        test_mse = eval_test()
    train_time_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=tirewear metric_name=test_mse metric={test_mse:.6f} "
        f"params={n_params} train_time_s={train_time_s:.2f}"
    )


if __name__ == "__main__":
    main()
