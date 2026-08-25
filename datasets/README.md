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

_(remaining datasets filled in as PatchTST-style, Conformer, Perceiver,
Decision Transformer, and the Tire-Patch-Wear Transformer are added.)_
