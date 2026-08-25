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
| [PatchTST-style](models/patchtst) | bidirectional attention over 1D time-series patches, channel-independent shared weights | remove patching and it's one token per timestep (much longer, costlier sequences); remove channel-independence and channels could leak into each other's forecast |
| [Perceiver](models/perceiver) | latents cross-attend into a large raw input array, then self-attend among themselves -- compute linear in input size | remove the latent bottleneck and this is full self-attention over every raw input element, O(input_len^2), infeasible at CIFAR-10-pixel scale |
| [Conformer](models/conformer) | macaron block: half-FFN + bidirectional self-attn + real conv module (pointwise/GLU/depthwise/BN/Swish/pointwise) + half-FFN | remove the conv module and it's a plain (bidirectional) Transformer encoder over spectrogram frames -- no local inductive bias, just global attention |
| [Decision Transformer](models/decisiontransformer) | causal self-attn over interleaved (return, state, action) tokens; separate embeddings per token type + return-conditioning | remove return-conditioning (feed a fixed/zero return) and it degenerates into plain behavior cloning, unable to be steered toward higher- or lower-return behavior at inference |
| [Tire-Patch-Wear Transformer](models/tirewear) | bidirectional attention over tire-contact-patch positions (a token = a spatial position, not a time/language/image unit); predicted wear increment passed through `softplus` -- a physics-*encoded*, not merely penalized, non-negativity guarantee | remove the `softplus` and the model could predict physically-impossible negative wear; remove attention across positions and each patch position would have to guess its wear increment from its own pressure/wear-so-far alone, blind to the rest of the footprint (e.g. to where the adhesion-to-sliding transition actually falls) |
