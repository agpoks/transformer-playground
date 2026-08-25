# How the nine models differ

Every model here is "attention + residual/FFN blocks" at its core; what
differs is the masking pattern, what plays the role of a token, and what
(if anything) gets added on top of vanilla attention.

| Model | Core idea | What breaks if you remove it |
|---|---|---|
| [Transformer](models/transformer) | encoder self-attn (bidirectional) + decoder causal self-attn + cross-attn | no way for the decoder to condition on the source sentence at all (remove cross-attn), or the decoder could see future tokens (remove causal mask) |
| [BERT-style](models/bert) | encoder-only, bidirectional self-attn, masked-LM | remove the mask-free bidirectionality and it degenerates into a causal decoder (GPT-style) with the wrong training objective |
| [GPT-style](models/gpt) | decoder-only, causal self-attn, pre-norm, learned OR RoPE positional encoding | remove the causal mask and it becomes BERT-style with the wrong objective; RoPE removed just falls back to learned positional embeddings |
| [Linear attention](models/linattn) | causal attention via FAVOR+ random features + cumulative sums, O(T) not O(T^2) | remove the random-feature kernel approximation and this is just GPT-style again -- the whole point is identical task/architecture, different attention complexity |

_(remaining rows filled in as PatchTST-style, Conformer, Perceiver,
Decision Transformer, and the Tire-Patch-Wear Transformer are added.)_
