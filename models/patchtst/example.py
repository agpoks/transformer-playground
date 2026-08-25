"""Train PatchTST-style on real ETTh1 (Electricity Transformer Temperature).

    python models/patchtst/example.py --device auto --epochs 10

See model.py for the patching/channel-independence architecture and
papers/README.md for the reference (Nie et al. 2023).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_etth1  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import PatchTSTModel  # noqa: E402


def make_windows(data: np.ndarray, seq_len: int, pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """data: (T, C). Returns (N, C, seq_len) inputs and (N, C, pred_len) targets,
    time-ordered sliding windows (no shuffling across the time axis itself --
    only the resulting window *indices* are shuffled at batch time)."""
    T, C = data.shape
    n = T - seq_len - pred_len + 1
    xs = np.stack([data[i : i + seq_len] for i in range(n)])  # (n, seq_len, C)
    ys = np.stack([data[i + seq_len : i + seq_len + pred_len] for i in range(n)])  # (n, pred_len, C)
    x = torch.from_numpy(xs).permute(0, 2, 1).float()  # (n, C, seq_len)
    y = torch.from_numpy(ys).permute(0, 2, 1).float()  # (n, C, pred_len)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    raw = load_etth1()  # (T, 7), real hourly readings
    n_total = raw.shape[0]
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    train_raw = raw[:n_train]
    test_raw = raw[n_train + n_val :]

    mean, std = train_raw.mean(axis=0, keepdims=True), train_raw.std(axis=0, keepdims=True)
    train_norm = (train_raw - mean) / std
    test_norm = (test_raw - mean) / std
    print(f"real ETTh1: {n_train} train hours, {len(test_norm)} test hours, 7 channels")

    x_train, y_train = make_windows(train_norm, args.seq_len, args.pred_len)
    x_test, y_test = make_windows(test_norm, args.seq_len, args.pred_len)

    model = PatchTSTModel(seq_len=args.seq_len, pred_len=args.pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def eval_test() -> float:
        model.eval()
        with torch.no_grad():
            preds = model(x_test.to(device))
            loss = loss_fn(preds, y_test.to(device)).item()
        model.train()
        return loss

    t0 = time.perf_counter()
    test_mse = None
    n = x_train.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        test_mse = eval_test()
        print(f"epoch {epoch:3d} | train_mse {total_loss / n_batches:.4f} | test_mse {test_mse:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=patchtst metric_name=test_mse metric={test_mse:.4f} "
        f"params={n_params} train_time_s={train_time:.2f}"
    )


if __name__ == "__main__":
    main()
