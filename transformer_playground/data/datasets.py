"""Real dataset loaders shared across models/*/example.py scripts.

Each `load_*` function downloads (if needed, into ~/.cache or a local
data_cache/ dir) and returns real data -- no synthetic placeholders except
where a model's own docs explicitly says otherwise (e.g. a physics-simulated
target where no public labeled dataset exists).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; transformer-playground/0.1)"}


def load_translation_pairs(max_pairs: int = 20000) -> list[tuple[str, str]]:
    """English-French sentence pairs, real data, from the well-known
    manythings.org/anki Tatoeba-derived export (used in many from-scratch
    seq2seq tutorials, e.g. the official PyTorch seq2seq translation
    tutorial). Each line of the underlying file is "eng<TAB>fra<TAB>attrib".
    Returns a list of (english, french) string pairs, lowercased.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    txt_path = CACHE_DIR / "fra.txt"
    if not txt_path.exists():
        zip_path = CACHE_DIR / "fra-eng.zip"
        if not zip_path.exists():
            resp = requests.get("https://www.manythings.org/anki/fra-eng.zip", timeout=60, headers=_HEADERS)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("fra.txt") as f:
                txt_path.write_bytes(f.read())

    pairs = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng, fra = parts[0].strip().lower(), parts[1].strip().lower()
            if eng and fra:
                pairs.append((eng, fra))
            if len(pairs) >= max_pairs:
                break
    return pairs


def load_wikitext2(split: str = "train") -> str:
    """WikiText-2 (Merity et al. 2016), real Wikipedia-derived text corpus,
    the word-tokenized (non-raw) release mirrored as plain text by the
    official PyTorch examples repo. Returns the full text of one split as
    a single string (already whitespace-tokenized, one sentence/line-ish
    unit per line, <unk> tokens present from the original release).
    """
    assert split in ("train", "valid", "test")
    CACHE_DIR.mkdir(exist_ok=True)
    txt_path = CACHE_DIR / f"wikitext2_{split}.txt"
    if not txt_path.exists():
        url = (
            "https://raw.githubusercontent.com/pytorch/examples/main/"
            f"word_language_model/data/wikitext-2/{split}.txt"
        )
        resp = requests.get(url, timeout=60, headers=_HEADERS)
        resp.raise_for_status()
        txt_path.write_bytes(resp.content)
    return txt_path.read_text(encoding="utf-8")


def load_tiny_shakespeare() -> str:
    """Tiny Shakespeare (~1.1MB), the well-known character-level corpus from
    Andrej Karpathy's char-rnn repo, real text (a concatenation of
    Shakespeare's plays), standard for from-scratch character-level LM
    tutorials. Returns the full text as one string.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    txt_path = CACHE_DIR / "tinyshakespeare.txt"
    if not txt_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        resp = requests.get(url, timeout=60, headers=_HEADERS)
        resp.raise_for_status()
        txt_path.write_bytes(resp.content)
    return txt_path.read_text(encoding="utf-8")


def load_etth1() -> np.ndarray:
    """ETTh1 (Electricity Transformer Temperature, hourly), Zhou et al.
    2021's real public benchmark dataset (the same one PatchTST's own paper
    forecasts on) -- 7 real sensor channels (6 load features + oil
    temperature "OT") over ~17,420 hourly readings from two Chinese
    electricity transformer stations. Returns a (T, 7) float32 array, time-
    ordered (no shuffling -- a forecasting split must respect time order).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    csv_path = CACHE_DIR / "ETTh1.csv"
    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        resp = requests.get(url, timeout=60, headers=_HEADERS)
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)
    import csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(v) for v in row[1:]])  # drop the date column
    return np.array(rows, dtype=np.float32)


def load_cifar10(train: bool = True):
    """Real CIFAR-10 (Krizhevsky) -- (3, 32, 32) RGB, 10 classes. Auto-
    downloads via torchvision on first call. Powers **Perceiver**, for
    cross-repo comparability with cnn-playground's cifar_suite models."""
    import torchvision
    from torchvision import transforms

    CACHE_DIR.mkdir(exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(root=str(CACHE_DIR), train=train, download=True, transform=tfm)
