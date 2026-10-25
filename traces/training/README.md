# One training step of GPT-2 (124M)

This folder holds one real training step of the pretrained
[GPT-2 small](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)
checkpoint, pinned to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (MIT; see `LICENSE-gpt2.txt`).
The weights file has SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. GPT-2 small has 12
blocks, width 768, 12 heads of 64, a 3072-wide MLP with tanh GELU, learned position embeddings, and an LM head tied
to the token embeddings: 124,439,808 parameters.

It was captured on October 25, 2026 at 15:05 UTC on CPU in float32, with deterministic algorithms and seed 0, using
Python 3.12.12, PyTorch 2.14.1, Transformers 4.57.6 and NumPy 2.5.4. `manifest.json` records the package versions
and the SHA-256 of the capture script and of the installed `modeling_gpt2.py`.

## The step

The batch is one sequence: **“To be, or not to be”** (William Shakespeare, _Hamlet_, Act 3, Scene 1; public domain),
7 GPT-2 tokens. Positions 1–6 each predict the next token, so there are 6 targets. The loss is the mean of their
cross-entropies; the last token has no target.

1. **Forward.** GPT-2 op for op, with the checkpoint's parameters: eager attention, tanh GELU, Conv1D as x·W + b.
   Its logits, loss and every parameter gradient equal `GPT2LMHeadModel`'s exactly (largest differences 0, 2.4e-7
   and 0). Dropout is off (p = 0), so the step is deterministic; GPT-2 was pretrained with p = 0.1.
2. **Backward.** One `loss.backward()`; `retain_grad` keeps ∂L/∂x on every intermediate.
3. **Update.** One `torch.optim.AdamW` step: one parameter group with lr 5e-5 (Hugging Face Trainer's default),
   betas (0.9, 0.999), eps 1e-8 and weight decay 0.01 (PyTorch's defaults), applied to every parameter. At step 1,
   m = 0.1·g and v = 0.001·g², so after bias correction m̂ = g and v̂ = g², and the Adam step is
   −lr·g / (|g| + eps): ±lr for every coordinate whose |g| is far above eps, less where it is not. Weight decay
   subtracts lr·0.01·θ on top.
4. **Second forward** with the updated weights.

| Per-token loss (nats) | `To`→` be` | ` be`→`,` | `,`→` or` | ` or`→` not` | ` not`→` to` | ` to`→` be` |      Mean |
| --------------------- | ---------: | --------: | --------: | -----------: | -----------: | ----------: | --------: |
| Before                |      4.161 |     6.869 |     6.027 |        1.343 |        1.071 |       0.039 | **3.252** |
| After one step        |      2.887 |     4.422 |     3.090 |        0.085 |        0.131 |       0.011 | **1.771** |

The drop is measured on the very batch the step was computed from. On the rest of the line, “That is the
question.”, which was never trained on, the mean loss goes from 3.301 to 3.185.

## Files

Lossless little-endian float32 tensors; the manifest lists each tensor's offset and shape and each file's size and
SHA-256.

| File            | Contents                                                                                                                                                                                                                                                                                                                                                      |    Size |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------: |
| `manifest.json` | Model, tokens, targets, optimizer, per-token losses before and after, top ∂L/∂logits per position, gradient and update norms for all 12 blocks, the embeddings and the final LayerNorm, block 1's weight statistics, finite differences, validation, decoded vocabulary                                                                                       | 0.51 MB |
| `forward.f32`   | Block 1 at full resolution: token and position embeddings, block input, LayerNorm 1, Q/K/V per head, scaled scores, attention, A·V, attention output, residual, LayerNorm 2, MLP up (3072), GELU, MLP down, block output; final-block output and final LayerNorm                                                                                              | 0.50 MB |
| `backward.f32`  | ∂L/∂ of every one of those block-1 tensors, and of the final-block output and final LayerNorm                                                                                                                                                                                                                                                                 | 0.46 MB |
| `logits.f32`    | All 50,257 logits at all 7 positions, before and after the update                                                                                                                                                                                                                                                                                             | 2.81 MB |
| `weights.f32`   | Block 1's weights: for c_attn, c_proj, c_fc and the MLP's c_proj, the gradient as RMS over 32 × 32 blocks, and exact 16 × 16 slices of θ, g, m, v, the Adam step, the decay term and θ after (for c_attn, the first 16 columns of each of Q, K and V); for its LayerNorm gains and biases and its four biases, the whole vectors of the same seven quantities | 0.35 MB |

What is reduced: the binned gradient maps (RMS over 32 × 32 blocks) and the 16 × 16 slices. Everything else in the
`.f32` files is a full tensor. Apart from those slices and block 1's LayerNorm and bias vectors, no parameters are
stored.

## Validation

From `manifest.json`. “Rule” rows re-derive each backward rule shown in the explainer in float64 from the recorded
forward tensors, at all 12 blocks, and compare with autograd (largest difference relative to the largest gradient
of that tensor).

| Check                                                                                                                                         |              Result |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------: |
| Explicit forward vs `GPT2LMHeadModel`: logits / loss / every parameter gradient                                                               |      0 / 2.4e-7 / 0 |
| Rules vs float64 autograd: cross-entropy, tied head, LayerNorm, GELU, matmul, A·V, softmax, scaled scores, residual fan-in, embedding scatter |           ≤ 1.1e-14 |
| The same rules vs the float32 recording (block ops and embedding / cross-entropy and tied head, sums over 50,257 logits near −100)            | ≤ 5.5e-6 / ≤ 2.9e-4 |
| Central finite differences in float64 (h = 1e-5) vs float64 autograd, 14 weights and 12 activations                                           |            ≤ 7.0e-7 |
| The same finite differences vs the recorded float32 gradients                                                                                 |            ≤ 4.7e-4 |
| AdamW formula vs `torch.optim.AdamW`: θ (in float32 roundings of \|θ\| + \|Δθ\|) / m / v (relative)                                           | 1.9 / 7e-8 / 1.4e-7 |
| Second forward, explicit vs model loss                                                                                                        |              1.2e-7 |
| Position-embedding rows 7–1023 receive no gradient                                                                                            |           exactly 0 |

Two details the numbers show:

- **The key bias gets no gradient.** It adds q·b_K to every score in a row, and softmax ignores a constant added to a
  whole row, so the exact gradient is zero (float64: 4e-18). Float32 rounding leaves |g| ≤ 4.5e-9, which is below
  eps, so Adam moves those 768 coordinates by less than lr. That is why only 66% of c_attn's bias steps are within
  0.1% of lr.
- **The step size ignores the gradient's scale.** Block gradient norms range from 3.0 to 10.5, but every block's update
  norm is 0.133 ≈ lr·√7.08M, because nearly every coordinate moves by lr. The token-embedding matrix is the
  exception: most of its 50,257 rows get only the LM head's tiny gradient (|g| near eps), so only 2.4% of its
  coordinates move by a full lr.

```sh
uv venv /private/tmp/training-trace-venv --python 3.12
uv pip install --python /private/tmp/training-trace-venv/bin/python torch==2.14.1 transformers==4.57.6 \
  huggingface-hub==0.36.2 numpy==2.5.4 safetensors
TOKENIZERS_PARALLELISM=false /private/tmp/training-trace-venv/bin/python scripts/capture-training.py --cache-dir .hf-cache
python3 scripts/capture-training.py --validate   # NumPy only: re-checks the stored arithmetic
node --test tests/training-full.test.mjs
```

The first run downloads about 550 MB.
