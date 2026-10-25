# Looped transformer: ByteDance Ouro-1.4B

This folder holds a recorded forward pass of a looped language model,
[Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)
(Zhu et al. 2025, "Scaling Latent Reasoning via Looped Language Models",
[arXiv:2510.25741](https://arxiv.org/abs/2510.25741)), over two short prompts.

- Revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; weights `model.safetensors` SHA-256
  `58872a72…dfa9885af` (the Hugging Face LFS object id), 1,434,652,673 parameters (bfloat16).
- The checkpoint ships custom code. SHA-256 of the files `trust_remote_code=True` would run:
  `modeling_ouro.py` `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`,
  `configuration_ouro.py` `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`.
- Apache-2.0 (see `../licenses/Apache-2.0.txt`). The recording holds activations, probabilities
  and parameter counts only, never weights.

It was captured on October 25, 2026 at 18:08 UTC on CPU in float32, Python 3.12.12, PyTorch
2.14.1, Transformers 4.57.6, safetensors 0.8.0 and tokenizers 0.22.2, by
`scripts/capture-looped.py` (its SHA-256, the package versions and the SHA-256 of the config and
tokenizer files are in the manifest).

| File            |   Bytes | Contents                                                  |
| --------------- | ------: | --------------------------------------------------------- |
| `manifest.json` |  31,489 | model, gate, per-loop predictions, validation, provenance |
| `monday.f32`    | 153,692 | `If today is Monday, tomorrow is`: block 1 of every loop  |
| `water.f32`     | 104,660 | `Water freezes at zero degrees`: the same tensors         |

The `.f32` files are little-endian float32; the manifest lists every tensor's offset and shape and
each file's byte size and SHA-256, which the page checks before use.

## The model, as its code runs it

Read from the pinned `modeling_ouro.py` and `config.json`, not from the model card:

- **One stack of 24 decoder blocks, applied 4 times** (`total_ut_steps: 4`) with the same weights.
  Each block is Llama-style with sandwich RMSNorm: x ← x + RMSNorm(Attention(RMSNorm(x))), then
  x ← x + RMSNorm(SwiGLU(RMSNorm(x))). Multi-head attention (16 heads of 128, no GQA), RoPE
  θ = 1,000,000, SwiGLU 2048 → 5632 → 2048, vocabulary 49,152, untied LM head.
- **How the loops feed each other.** After block 24 the final RMSNorm (`model.norm`) is applied, and
  its output is what the next loop starts from. The same tensor is what the exit gate and the LM
  head read. There is no per-loop embedding, no re-injection of the token embeddings and no
  per-loop norm. Loop 1 starts from the token embeddings.
- **Exit gate.** One `Linear(2048 → 1)` with a bias (2,049 parameters), shared by all loops:
  λᵣ = σ(w · hᵣ + b) is the probability of stopping after loop r given loop r was reached. The exit
  distribution is pᵣ = λᵣ Πⱼ<ᵣ(1 − λⱼ) for r < 4, and p₄ takes the remaining mass (λ₄ is computed
  but unused).
- **Inference with the release config** (`early_exit_threshold: 1.0`, no `early_exit_step`): each
  position reads its logits from the first loop whose cumulative exit probability reaches 1.0, or
  from loop 4. The forward always computes all 4 loops; the gate only selects. In this recording
  every position reads loop 4.
- **KV cache.** Each loop has its own entries (slot = loop × 24 + layer), so attention in loop r
  reads keys and values computed in loop r: 96 entries per token.

## How it was recorded

The remote code needs no CUDA kernels, but the capture does not execute it. The forward is a
plain-PyTorch reimplementation of `modeling_ouro.py` over the checkpoint's safetensors, read as
data. Every loop's logits and gate logits are checked against a second implementation assembled
from transformers' own `LlamaRMSNorm`, `LlamaAttention` (eager), `LlamaMLP` and
`LlamaRotaryEmbedding`, loaded with the same weights and composed in Ouro's loop order: the max
difference is 0.0 on both prompts. `scripts/capture-looped.py --remote-code-check` additionally
runs the pinned remote code (after checking the SHA-256 above) and stores how far its default-path
logits, per-loop states and gate logits are from this recording. That check was **not run** for
this recording (`validation.remoteCode.run` is `false`).

The prompts are tokenized without a BOS token, as the model card's example does (`If today is
Monday, tomorrow is` is 7 tokens, `Water freezes at zero degrees` 5). They were chosen among a
handful tried because the top next token changes from loop 1 to loop 2 (Monday: ` Wednesday`
0.75% → ` Tuesday` 48%) or is already right after loop 1 (water: ` Celsius` 45% → 85%).

## What was recorded

Per prompt, in its `.f32` file, every tensor has a leading loop axis of 4:

- **Block 1 of every loop**, as the Dense UI needs it: the block input (`input`: the embedding in
  loop 1, the previous loop's normalized state after that), RMSNorm 1, Q, K, V for all 16 heads
  before and after RoPE, each head's context, the raw `QKᵀ` scores (4 × 16 × n × n), the causal
  attention probabilities, the attention output, its sandwich RMSNorm, the residual, RMSNorm 2, the
  SwiGLU gate, up projection, SiLU(gate), their product, the down projection, its sandwich RMSNorm
  and the block output. Vectors are stored as their first 8 coordinates (`name`) and the norm over
  all coordinates (`nameNorm`: 2,048, 5,632 or 128).
- **Blocks 1–24 of every loop** as per-position norms: block input, attention output before and
  after its sandwich norm, MLP output before and after its sandwich norm, and block output.
- **Each loop's ends:** the output of block 24 (`loopOutput`) and the normalized state
  (`loopState`), previews and norms; the gate's logit, λ and exit probability.

In the manifest, per prompt: `gate` (logits, λ, exit probabilities, expected loops Σ r · pᵣ and the
loop the release config reads), `predictions` (the top 10 next tokens, id and probability over the
full vocabulary, when stopping after each loop: the LM head on that loop's state, i.e.
`exit_at_step = r − 1` in `modeling_ouro.py`), `finalArgmax` and its probability and rank after
every loop, `stateChange` (‖hᵣ − hᵣ₋₁‖ / ‖hᵣ₋₁‖ per position) and each loop's log-sum-exp.
`model` carries the parameter counts the page's weights card shows, counted from the checkpoint:
51,388,416 parameters per block, 1.23B in the 24-block stack, 5.13B for 96 distinct blocks, and
matrix-multiply FLOPs per token (2 × weights used; attention's sequence-length terms and the norms
ignored) of 10.07 G for 96 block passes against 2.67 G for one pass, a ratio of 3.77.

## Validation

All in `manifest.json` → `validation`, as max absolute errors over both prompts and all 4 loops:

| Check                                                                  | Max error |
| ---------------------------------------------------------------------- | --------: |
| Logits of every loop, reimplementation vs transformers Llama modules   |       0.0 |
| Gate logits of every loop, same comparison                             |       0.0 |
| Release-config logits rebuilt from the recorded states and the LM head |       0.0 |
| Loop r + 1 input vs loop r's normalized state                          |       0.0 |
| Rerun without recording, logits                                        |       0.0 |
| RMSNorms (all four in block 1), float64 recompute                      |   3.0e-06 |
| Q/K/V projections, float64 recompute                                   |   2.4e-05 |
| RoPE                                                                   |       0.0 |
| Attention probabilities from the recorded scores                       |   9.2e-07 |
| Attention context, output projection                                   |   4.8e-05 |
| SwiGLU, down projection, residuals, block output                       |   3.9e-05 |
| Final RMSNorm, gate logit, LM head                                     |   6.6e-05 |
| Weight on future positions                                             |       0.0 |

`tests/looped-trace.test.mjs` re-checks the file hashes, the loop chaining, the causal softmax,
the gate arithmetic (λ = σ(logit), the exit distribution, expected loops, the release rule) and
the per-loop predictions from the stored values.
