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
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; transformer-playground/0.1)"}

NGSIM_RESOURCE_URL = "https://data.transportation.gov/resource/8ect-6jqj.csv"
_FEET_TO_M = 0.3048


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


def load_speech_commands(
    words: tuple[str, ...] = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"),
    max_per_class_train: int = 200,
    max_per_class_test: int = 50,
    seed: int = 0,
):
    """Real Google Speech Commands v0.02 (Warden 2018), the standard "core
    ten" spoken-word classes. Powers **Conformer**.

    Auto-downloads via `torchaudio.datasets.SPEECHCOMMANDS`'s constructor
    (for the download+extraction side effect only -- this environment's
    torchaudio build routes `.load()`/`__getitem__` through TorchCodec,
    which in turn needs system ffmpeg shared libraries (`libavdevice.so.58`)
    not present here, so `__getitem__` is never called; instead this
    function reads the extracted `.wav` files directly with Python's
    built-in `wave` module (they're plain 16kHz mono 16-bit PCM, no codec
    needed) and returns raw waveforms as float32 numpy arrays in [-1, 1].

    The real dataset itself is large (~2.4GB, ~4000+ clips/class across 35
    words); `max_per_class_train`/`max_per_class_test` are honest, CPU-
    training-speed-motivated subset caps (same pattern as Perceiver's CIFAR-
    10 subset) -- documented, not silent. Uses the dataset's own real
    `testing_list.txt` for the test split (never touched during training),
    and a random (seeded) sample of the remaining files for train.

    Returns ((train_wavs, train_labels), (test_wavs, test_labels), words)
    -- `*_wavs` are (N, 16000) float32 arrays (1s @ 16kHz, zero-padded/
    truncated), `*_labels` are (N,) int arrays indexing into `words`.
    """
    import random
    import wave

    import torchaudio

    CACHE_DIR.mkdir(exist_ok=True)
    torchaudio.datasets.SPEECHCOMMANDS(root=str(CACHE_DIR), download=True)  # download+extract only
    root = CACHE_DIR / "SpeechCommands" / "speech_commands_v0.02"
    test_set = set((root / "testing_list.txt").read_text().split())

    rng = random.Random(seed)
    train_paths, train_labels, test_paths, test_labels = [], [], [], []
    for label_idx, word in enumerate(words):
        files = sorted(f.name for f in (root / word).glob("*.wav"))
        test_files = [f for f in files if f"{word}/{f}" in test_set]
        train_files = [f for f in files if f"{word}/{f}" not in test_set]
        rng.shuffle(train_files)
        rng.shuffle(test_files)
        for f in train_files[:max_per_class_train]:
            train_paths.append(root / word / f)
            train_labels.append(label_idx)
        for f in test_files[:max_per_class_test]:
            test_paths.append(root / word / f)
            test_labels.append(label_idx)

    def _load_wav(path) -> np.ndarray:
        with wave.open(str(path), "rb") as w:
            frames = w.readframes(w.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.shape[0] < 16000:
            arr = np.pad(arr, (0, 16000 - arr.shape[0]))
        return arr[:16000]

    train_wavs = np.stack([_load_wav(p) for p in train_paths])
    test_wavs = np.stack([_load_wav(p) for p in test_paths])
    return (train_wavs, np.array(train_labels, dtype=np.int64)), (test_wavs, np.array(test_labels, dtype=np.int64)), words


def load_ngsim_traffic_field(
    location: str = "us-101", space_bins: int = 64, time_bins: int = 200, v_class: int = 2
) -> "torch.Tensor":
    """Real macroscopic NGSIM US-101 traffic field (US DOT, public domain,
    via the Socrata SODA API -- no login), the same real data source and
    histogram-binning methodology already used by `sciml-playground`'s
    PDE-Net/PINO/FNO/GNO models -- downloaded independently here (not a
    cross-repo import), so this repo has no dependency on sciml-playground
    being installed alongside it. Powers **Decision Transformer** (see that
    model's honest data-adaptation note in `models/decisiontransformer/model.py`
    for exactly how this aggregate field is turned into control
    trajectories -- it is NOT itself an RL/control dataset).

    Returns (2, time_bins, space_bins): channel 0 = density-like (raw
    observation count / bin width, vehicles/meter), channel 1 = speed-like
    (mean vehicle speed, m/s) -- both zero in any (t, s) bin with no
    vehicle observation, which is "no data", not "vehicle stopped."
    """
    import torch

    CACHE_DIR.mkdir(exist_ok=True)
    raw_path = CACHE_DIR / f"ngsim_{location}_traffic_field.csv"
    if not raw_path.exists():
        where = f"location='{location}' AND v_class={v_class}"
        resp = requests.get(
            NGSIM_RESOURCE_URL,
            params={"$where": where, "$limit": 500_000, "$order": "vehicle_id,frame_id"},
            timeout=180,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        raw_path.write_bytes(resp.content)

    df = pd.read_csv(raw_path)
    df.columns = [c.lower() for c in df.columns]
    df["x_m"] = df["local_y"] * _FEET_TO_M
    t0 = df["global_time"].min()
    df["t_s"] = (df["global_time"] - t0) / 1000.0

    x_edges = np.linspace(df["x_m"].min(), df["x_m"].max(), space_bins + 1)
    t_edges = np.linspace(df["t_s"].min(), df["t_s"].max(), time_bins + 1)

    count_field, _, _ = np.histogram2d(df["t_s"], df["x_m"], bins=[t_edges, x_edges])
    speed_sum, _, _ = np.histogram2d(
        df["t_s"], df["x_m"], bins=[t_edges, x_edges], weights=df["v_vel"] * _FEET_TO_M
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_field = np.where(count_field > 0, speed_sum / np.maximum(count_field, 1), 0.0)
    density_field = count_field / (x_edges[1] - x_edges[0])

    field = np.stack([density_field, speed_field], axis=0)
    return torch.tensor(field, dtype=torch.float32)


def load_cifar10(train: bool = True):
    """Real CIFAR-10 (Krizhevsky) -- (3, 32, 32) RGB, 10 classes. Auto-
    downloads via torchvision on first call. Powers **Perceiver**, for
    cross-repo comparability with cnn-playground's cifar_suite models."""
    import torchvision
    from torchvision import transforms

    CACHE_DIR.mkdir(exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(root=str(CACHE_DIR), train=train, download=True, transform=tfm)
