"""Train a small BERT-style encoder with masked-language-modeling on real
WikiText-2 text.

    python models/bert/example.py --device auto --epochs 20

See model.py for the bidirectional-attention architecture and
papers/README.md for the reference (Devlin et al. 2018).

Tokenization is a from-scratch word-level vocabulary (no external
tokenizer library), same convention as models/transformer/example.py.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformer_playground.data import load_wikitext2  # noqa: E402
from transformer_playground.device import add_device_arg, resolve_device  # noqa: E402
from transformer_playground.utils.seed import set_seed  # noqa: E402
from model import BERTModel  # noqa: E402

PAD, MASK, UNK = "<pad>", "<mask>", "<unk>"


class Vocab:
    def __init__(self, tokens: list[str], max_size: int = 8000):
        counter = Counter(tokens)
        specials = [PAD, MASK, UNK]
        most_common = [w for w, _ in counter.most_common(max_size - len(specials))]
        self.itos = specials + most_common
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)


def _load_lines(split: str, min_tokens: int = 4) -> list[list[str]]:
    text = load_wikitext2(split)
    lines = []
    for raw in text.splitlines():
        toks = raw.strip().split()
        if len(toks) >= min_tokens:
            lines.append(toks)
    return lines


class MLMDataset(Dataset):
    """Real BERT masking recipe: for each of ~15% chosen positions,
    80% -> [MASK], 10% -> random vocab token, 10% -> unchanged; label is
    always the ORIGINAL token id at every chosen position (ignore_index
    elsewhere)."""

    def __init__(self, lines: list[list[str]], vocab: Vocab, max_len: int, mask_prob: float = 0.15):
        self.lines = lines
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx):
        toks = self.lines[idx][: self.max_len]
        ids = [self.vocab.stoi.get(t, self.vocab.stoi[UNK]) for t in toks]
        input_ids = list(ids)
        labels = [-100] * len(ids)  # -100 = ignore_index, matches unmasked positions

        for i in range(len(ids)):
            if random.random() < self.mask_prob:
                labels[i] = ids[i]
                r = random.random()
                if r < 0.8:
                    input_ids[i] = self.vocab.stoi[MASK]
                elif r < 0.9:
                    input_ids[i] = random.randrange(len(self.vocab))
                # else: 10% unchanged, input_ids[i] already == ids[i]

        pad_len = self.max_len - len(ids)
        input_ids = input_ids + [self.vocab.stoi[PAD]] * pad_len
        labels = labels + [-100] * pad_len
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--max-lines", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_lines = _load_lines("train")[: args.max_lines]
    val_lines = _load_lines("valid")[: max(1, args.max_lines // 10)]
    print(f"real WikiText-2 lines: {len(train_lines)} train, {len(val_lines)} val")

    vocab = Vocab([t for line in train_lines for t in line])
    print(f"vocab size {len(vocab)}")

    train_ds = MLMDataset(train_lines, vocab, args.max_len)
    val_ds = MLMDataset(val_lines, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = BERTModel(vocab_size=len(vocab), max_len=args.max_len, d_model=128, n_heads=4, n_layers=2, d_ff=256).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n_batches = 0.0, 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.set_grad_enabled(train):
                logits = model(input_ids)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
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
            print(f"epoch {epoch:3d} | train_mlm_loss {train_loss:.4f} | val_mlm_loss {val_loss:.4f}")
    train_time = time.perf_counter() - t0

    if val_loss is None:
        val_loss = run_epoch(val_loader, train=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RESULT: model=bert metric_name=mlm_loss metric={val_loss:.4f} params={n_params} train_time_s={train_time:.2f}")


if __name__ == "__main__":
    main()
