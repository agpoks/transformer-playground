"""Train Perceiver on real CIFAR-10 (raw flattened pixels, no convolution).

    python models/perceiver/example.py --device auto --epochs 5

See model.py for the latent-bottleneck cross-attention architecture and
papers/README.md for the reference (Jaegle et al. 2021).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_cifar10  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import PerceiverModel  # noqa: E402


def to_byte_array(imgs: torch.Tensor) -> torch.Tensor:
    """imgs: (B, 3, 32, 32) -> (B, 1024, 5): 3 raw RGB channels + a 2D
    (row, col) position embedding per pixel, normalized to [-1, 1] --
    exactly the Perceiver paper's "flatten to a byte array, no
    convolutional patchification" input, just with a raw-coordinate
    positional feature instead of the paper's Fourier positional features
    (documented simplification -- see docs/source/models/perceiver.md)."""
    b, c, h, w = imgs.shape
    pixels = imgs.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, 1024, 3)
    rows = torch.linspace(-1, 1, h).view(h, 1).expand(h, w).reshape(-1)
    cols = torch.linspace(-1, 1, w).view(1, w).expand(h, w).reshape(-1)
    pos = torch.stack([rows, cols], dim=-1).unsqueeze(0).expand(b, -1, -1).to(imgs.device)  # (B, 1024, 2)
    return torch.cat([pixels, pos], dim=-1)  # (B, 1024, 5)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            x = to_byte_array(imgs)
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-latents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train", type=int, default=None,
        help="Use only the first N real training images (CPU-time-budget escape hatch; full CIFAR-10 by default).",
    )
    parser.add_argument(
        "--max-test", type=int, default=None,
        help="Use only the first N real test images (same reason as --max-train).",
    )
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_set = load_cifar10(train=True)
    test_set = load_cifar10(train=False)
    if args.max_train is not None:
        train_set = torch.utils.data.Subset(train_set, range(min(args.max_train, len(train_set))))
    if args.max_test is not None:
        test_set = torch.utils.data.Subset(test_set, range(min(args.max_test, len(test_set))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(f"real CIFAR-10: {len(train_set)} train, {len(test_set)} test images")

    model = PerceiverModel(input_dim=5, n_classes=10, n_latents=args.n_latents).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    test_acc = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = None
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            x = to_byte_array(imgs)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:3d} | train_loss {last_loss:.4f} | test_acc {test_acc:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=perceiver metric_name=test_acc metric={test_acc:.4f} "
        f"params={n_params} train_time_s={train_time:.2f}"
    )


if __name__ == "__main__":
    main()
