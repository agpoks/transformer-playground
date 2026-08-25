"""Train a small Conformer on real Google Speech Commands (core 10 words).

    python models/conformer/example.py --device auto --epochs 10

See model.py for the macaron conv+attention block and papers/README.md for
the reference (Gulati et al. 2020).

Log-mel spectrogram features (via torchaudio.transforms, pure tensor ops --
no audio-codec dependency) turn each 1s waveform into a (T, n_mels)
sequence of frames, which is what the Conformer's self-attention/conv
modules treat as the sequence dimension.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_speech_commands  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import ConformerModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--max-per-class-train", type=int, default=200)
    parser.add_argument("--max-per-class-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    (train_wavs, train_labels), (test_wavs, test_labels), words = load_speech_commands(
        max_per_class_train=args.max_per_class_train, max_per_class_test=args.max_per_class_test, seed=args.seed
    )
    print(f"real Speech Commands (core {len(words)} words): {len(train_wavs)} train, {len(test_wavs)} test clips")

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=args.n_mels)
    to_db = torchaudio.transforms.AmplitudeToDB()

    def to_features(wavs: torch.Tensor) -> torch.Tensor:
        # (N, 16000) -> (N, n_mels, T) -> (N, T, n_mels)
        return to_db(mel(wavs)).transpose(1, 2)

    train_x = to_features(torch.from_numpy(train_wavs)).to(device)
    train_y = torch.from_numpy(train_labels).to(device)
    test_x = to_features(torch.from_numpy(test_wavs)).to(device)
    test_y = torch.from_numpy(test_labels).to(device)
    print(f"feature shape: {tuple(train_x.shape[1:])} (T, n_mels)")

    model = ConformerModel(n_mels=args.n_mels, num_classes=len(words), max_len=train_x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def iterate_batches(x, y, batch_size, shuffle):
        n = x.shape[0]
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            yield x[b], y[b]

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        correct, total = 0, 0
        for xb, yb in iterate_batches(test_x, test_y, args.batch_size, shuffle=False):
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        model.train()
        return correct / total

    t0 = time.perf_counter()
    test_acc = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = None
        for xb, yb in iterate_batches(train_x, train_y, args.batch_size, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        test_acc = evaluate()
        print(f"epoch {epoch:3d} | train_loss {last_loss:.4f} | test_acc {test_acc:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=conformer metric_name=test_acc metric={test_acc:.4f} "
        f"params={n_params} train_time_s={train_time:.2f}"
    )


if __name__ == "__main__":
    main()
