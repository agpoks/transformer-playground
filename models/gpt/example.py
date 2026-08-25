"""Train a small decoder-only GPT-style model on real Tiny Shakespeare.

    python models/gpt/example.py --device auto --epochs 5
    python models/gpt/example.py --device auto --pos-encoding rope

See model.py for the causal self-attention/RoPE architecture and
papers/README.md for the references (Radford et al. 2018/2019, Su et al.
2021 for RoPE).

Character-level tokenization: vocab = the corpus's unique characters, no
external tokenizer library.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_tiny_shakespeare  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import GPTModel  # noqa: E402


def make_batches(data: torch.Tensor, block_size: int, batch_size: int, device):
    n = data.shape[0] - block_size - 1
    idx = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx]).to(device)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training 'epochs' (each = --steps-per-epoch batches)")
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--pos-encoding", choices=["learned", "rope"], default="learned")
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}, pos_encoding: {args.pos_encoding}")

    text = load_tiny_shakespeare()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = len(data) // 10
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"real Tiny Shakespeare: {len(train_data)} train chars, {len(val_data)} val chars, vocab {len(chars)}")

    model = GPTModel(
        vocab_size=len(chars),
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_len=args.block_size,
        pos_encoding=args.pos_encoding,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def eval_loss(n_batches: int = 20) -> float:
        model.eval()
        total = 0.0
        for _ in range(n_batches):
            x, y = make_batches(val_data, args.block_size, args.batch_size, device)
            logits = model(x)
            total += loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
        model.train()
        return total / n_batches

    t0 = time.perf_counter()
    val_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _ in range(args.steps_per_epoch):
            x, y = make_batches(train_data, args.block_size, args.batch_size, device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = eval_loss()
        print(f"epoch {epoch:3d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=gpt metric_name=val_loss metric={val_loss:.4f} "
        f"params={n_params} train_time_s={train_time:.2f}"
    )


if __name__ == "__main__":
    main()
