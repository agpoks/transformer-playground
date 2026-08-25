"""Tiny matplotlib block-diagram primitives, shared by the docs' architecture
diagrams (docs/source/models/*.md). Not used by the models themselves --
this is purely so every "how it's built inside" diagram in the docs shares
one visual language instead of nine ad-hoc drawings.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INPUT = "#0f172a"      # slate: raw input / output tensors
LINEAR = "#0891b2"     # teal: an nn.Linear (projection) affine map
NONLIN = "#4f46e5"     # indigo: an elementwise nonlinearity / gate / softmax
STATE = "#be123c"      # rose: the recurrent/hidden state itself
OTHER = "#64748b"      # slate-grey: norm, pooling, positional encoding, misc
ATTN = "#b45309"       # amber: an attention block specifically (Q/K/V mixing)


def new_ax(figsize=(7.5, 4.2), xlim=(0, 10), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, color=LINEAR, fontsize=9.5, textcolor="white"):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", color=textcolor, fontsize=fontsize, wrap=True)
    return (cx, cy, w, h)


def arrow(ax, start, end, color="#334155", dashed=False, curve=0.0, label=None, lw=1.6):
    style = "->"
    connectionstyle = f"arc3,rad={curve}" if curve else None
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle="dashed" if dashed else "solid",
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        ),
    )
    if label:
        mx = (start[0] + end[0]) / 2 + (0.6 if curve else 0)
        my = (start[1] + end[1]) / 2 + (0.5 if curve else 0)
        ax.text(mx, my, label, ha="center", va="center", fontsize=8, color=color)
