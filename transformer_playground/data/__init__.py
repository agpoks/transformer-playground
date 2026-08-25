"""Real dataset loaders for transformer-playground. One `load_*` function per
dataset in `datasets.py`, added incrementally as each model needs one.
"""

from transformer_playground.data.datasets import (
    load_cifar10,
    load_etth1,
    load_ngsim_traffic_field,
    load_speech_commands,
    load_tiny_shakespeare,
    load_translation_pairs,
    load_wikitext2,
)

__all__ = [
    "load_translation_pairs",
    "load_wikitext2",
    "load_tiny_shakespeare",
    "load_etth1",
    "load_cifar10",
    "load_speech_commands",
    "load_ngsim_traffic_field",
]
