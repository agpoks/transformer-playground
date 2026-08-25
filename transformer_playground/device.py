"""Single place every example/benchmark goes through to pick cpu vs. gpu.

Usage in a script:

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    device = resolve_device(args.device)
"""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a user-facing device string into a torch.device.

    "auto" picks CUDA if available, then Apple MPS, then falls back to CPU.
    Anything else ("cpu", "cuda", "cuda:1", "mps") is passed straight through,
    but we raise early with a clear message if it isn't actually usable.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {name} requested but CUDA is not available on this machine. "
            "Use --device cpu or --device auto."
        )
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            f"--device {name} requested but MPS is not available on this machine. "
            "Use --device cpu or --device auto."
        )
    return device


def add_device_arg(parser) -> None:
    """Attach the standard --device flag to an argparse.ArgumentParser."""
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. 'auto' (default) uses CUDA/MPS if available, else CPU.",
    )
