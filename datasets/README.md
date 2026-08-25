# Datasets

Real, standard datasets across language, vision, audio, time-series,
control, and (one honestly-labeled exception) physics-simulated tire data
where no public labeled dataset exists. All auto-download on first use and
cache in `data_cache/` (gitignored) -- no accounts, no manual steps, unless
noted otherwise per dataset below.

## English-French sentence pairs

Real Tatoeba-derived translation pairs from the well-known
[manythings.org/anki](https://www.manythings.org/anki/fra-eng.zip) export
(the same source used by many from-scratch seq2seq tutorials, including
the official PyTorch translation tutorial). Powers **Transformer**.
`transformer_playground.data.load_translation_pairs`.

## WikiText-2

Merity et al.'s real Wikipedia-derived language-modeling corpus, mirrored
as plain text by the PyTorch examples repo (word-tokenized release, not
the `-raw` variant). Powers **BERT-style**.
`transformer_playground.data.load_wikitext2`.

## Tiny Shakespeare

Andrej Karpathy's well-known ~1.1MB character-level corpus (a
concatenation of Shakespeare's plays), real text, the standard dataset for
from-scratch character-level language modeling tutorials. Powers
**GPT-style** and **Linear attention** (identical task/split for both, for
a direct comparison). `transformer_playground.data.load_tiny_shakespeare`.

## ETTh1

Zhou et al.'s real Electricity Transformer Temperature dataset (hourly
split), 7 real sensor channels from two Chinese electricity-transformer
stations -- the same benchmark PatchTST's own paper forecasts on. Powers
**PatchTST-style**. `transformer_playground.data.load_etth1`.

## CIFAR-10

Krizhevsky's real 32x32 RGB image classification dataset, via
`torchvision`. Powers **Perceiver** (fed in as 1024 raw pixel tokens, no
convolution) -- cross-repo comparable in spirit to `cnn-playground`'s
`cifar_suite` models. `transformer_playground.data.load_cifar10`.

## Google Speech Commands (core 10 words)

Warden's real spoken-word dataset (v0.02), 16kHz mono 1-second clips --
this repo uses the standard "core 10" command words (yes, no, up, down,
left, right, on, off, stop, go), reading `.wav` files directly via
Python's `wave` module (this environment's `torchaudio.load()` needs
system ffmpeg libraries not present here -- see
`load_speech_commands`'s docstring). An honest, CPU-training-speed subset
cap is applied per class (documented in the loader); the real, official
`testing_list.txt` split is used for test regardless. Powers **Conformer**.
`transformer_playground.data.load_speech_commands`.

## NGSIM traffic field (US-101), reinterpreted as offline-imitation control data

The real NGSIM US-101 macroscopic traffic field (US DOT, public domain,
Socrata SODA API -- no login), downloaded independently here with the same
real data source and histogram-binning methodology already validated in
[`sciml-playground`](https://github.com/agpoks/sciml-playground) (not a
cross-repo import or dependency). This aggregate density/speed field is
**not** itself an RL-labeled control dataset -- **Decision Transformer**
reinterprets each spatial bin's time series as one imitation trajectory
(state = density/speed, action = observed speed change, reward = closeness
to a free-flow-speed target); see that model's docs page for the full,
explicit adaptation. `transformer_playground.data.load_ngsim_traffic_field`.

_(remaining dataset filled in as the Tire-Patch-Wear Transformer is added.)_
