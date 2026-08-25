"""Train a small encoder-decoder Transformer on real English->French pairs.

    python models/transformer/example.py --device auto --epochs 20

See model.py for the attention/encoder-decoder architecture and
papers/README.md for the reference (Vaswani et al. 2017).

Tokenization here is a simple from-scratch word-level vocabulary (no
external tokenizer library) -- good enough to demonstrate the architecture
on a small real corpus, not a production BPE/subword tokenizer.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_translation_pairs  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import TransformerModel  # noqa: E402

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


class Vocab:
    def __init__(self, sentences: list[str], max_size: int = 4000):
        counter = Counter(tok for s in sentences for tok in s.split())
        specials = [PAD, SOS, EOS, UNK]
        most_common = [w for w, _ in counter.most_common(max_size - len(specials))]
        self.itos = specials + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, sentence: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, self.stoi[UNK]) for tok in sentence.split()]
        ids = [self.stoi[SOS]] + ids[: max_len - 2] + [self.stoi[EOS]]
        ids = ids + [self.stoi[PAD]] * (max_len - len(ids))
        return ids


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        eng, fra = self.pairs[idx]
        src = torch.tensor(self.src_vocab.encode(eng, self.max_len), dtype=torch.long)
        tgt = torch.tensor(self.tgt_vocab.encode(fra, self.max_len), dtype=torch.long)
        return src, tgt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    pairs = load_translation_pairs(max_pairs=args.max_pairs)
    n_val = max(1, len(pairs) // 20)
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]
    print(f"real English-French pairs: {len(train_pairs)} train, {len(val_pairs)} val")

    src_vocab = Vocab([e for e, _ in train_pairs])
    tgt_vocab = Vocab([f for _, f in train_pairs])
    print(f"src vocab {len(src_vocab)}, tgt vocab {len(tgt_vocab)}")

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, args.max_len)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=args.max_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD])

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n_batches = 0.0, 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.set_grad_enabled(train):
                logits = model(src, tgt_in)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    t0 = time.perf_counter()
    val_loss = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        if epoch % 2 == 0 or epoch == args.epochs:
            val_loss = run_epoch(val_loader, train=False)
            print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
    train_time = time.perf_counter() - t0

    if val_loss is None:
        val_loss = run_epoch(val_loader, train=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RESULT: model=transformer metric_name=val_loss metric={val_loss:.4f} params={n_params} train_time_s={train_time:.2f}")


if __name__ == "__main__":
    main()
