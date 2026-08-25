"""Real dataset loaders for transformer-playground. One `load_*` function per
dataset in `datasets.py`, added incrementally as each model needs one.
"""

from transformer_playground.data.datasets import (
    load_tiny_shakespeare,
    load_translation_pairs,
    load_wikitext2,
)

__all__ = ["load_translation_pairs", "load_wikitext2", "load_tiny_shakespeare"]
