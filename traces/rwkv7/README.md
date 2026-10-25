# Recorded RWKV-7 0.1B trace

`manifest.json`, `model.f32` and `block-00.f32` … `block-11.f32` are one forward pass of the pretrained
[RWKV-7 0.1B g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1/tree/7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8)
checkpoint (`fla-hub/rwkv7-0.1B-g1`), pinned to revision `7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8`.
The weights are Apache-2.0 licensed (see `../licenses/Apache-2.0.txt`). The trace holds activations, plus
four rows of the output matrix (`classifierWeights`) so the top logits can be checked. The checkpoint itself
is not included.

The capture ran on October 24, 2026 at 18:38:54 UTC, on CPU in float32, with Python 3.12.12, PyTorch 2.14.1,
NumPy 2.5.4 and safetensors 0.8.0. The bf16 checkpoint tensors are upcast to float32, which is exact.
`model.safetensors` matches its Git LFS object id,
`051b3684ffa896aa842522e5bbc2f355ac00783e12e373a8eb7e4907dd5ead70`.

| Prompt                     | Tokens                                         | Next token           |
| -------------------------- | ---------------------------------------------- | -------------------- |
| `The capital of France is` | `The`, ` capital`, ` of`, ` France`, ` is` (5) | ` Paris`, p = 0.3204 |

The model has 12 blocks, width 768, 12 heads of size 64, a 3072-wide channel mix and 65,536 vocabulary
entries (RWKV World tokenizer).

## How it was run

The checkpoint's own Transformers code imports flash-linear-attention's Triton kernels, which do not run
on macOS. `scripts/capture-linear-attention.py` therefore runs no checkpoint code. It reads the
safetensors weights and runs RWKV-7 as an explicit per-token PyTorch recurrence:

1. Block 1 applies an extra LayerNorm (`ln0`) to the embedding. Every block then applies `ln1`.
2. Token shift: each of r, w, k, v, a and g reads `x_t + (x_{t-1} − x_t) ⊙ μ`, with its own μ.
3. The projections are r, k, v = W·x̂. Decay is `w = exp(−e^{−0.5} · σ(w₀ + tanh(x̂_w A_w) B_w))`, so it
   lies in (0.545, 1). The in-context rate is `a = σ(a₀ + x̂_a A_a B_a)` and the gate is `g = σ(x̂_g A_g) B_g`.
4. The delta-rule key is `κ̂ = normalize(k ⊙ k_k)` per head. The written key is `k̃ = k ⊙ (1 + (a − 1) ⊙ k_a)`.
5. v-first: in blocks 2–12, `v = v + (v₁ − v) ⊙ σ(v₀ + x̂_v A_v B_v)`, where v₁ is block 1's value.
6. The state update, per head: `S_t = S_{t−1} (diag(w_t) − κ̂_t (κ̂_t ⊙ a_t)ᵀ) + v_t k̃_tᵀ`, and the readout is
   `o_t = S_t r_t`. S is 64×64 for every token. There is no key-value cache.
7. Each head goes through GroupNorm (ε = 64·10⁻⁵). The bonus `(r · (k̃ ⊙ r_k)) v` is added, the result is
   multiplied by g, and W_o maps it back to 768. Then comes the residual.
8. Channel mix: `ln2`, its own token shift, then `W_v · relu(W_k x̂)²`, then the residual.
9. The final LayerNorm is applied, then the output matrix (not tied to the embedding).

## Validation (stored in `manifest.json` → `validation`)

| Check                                                                                     | Max abs error |
| ----------------------------------------------------------------------------------------- | ------------: |
| Independent implementation: all 5 × 65,536 logits (logits reach 31 in magnitude)          |       4.67e-5 |
| Independent implementation: final 12 × 64 × 64 state of every block (values reach 52)     |       4.20e-5 |
| Independent implementation: both token-shift states of every block                        |       1.44e-5 |
| Every stored state rebuilt from the previous state and the stored w, κ̂, a, k̃, v (float64) |       4.73e-6 |
| `o = A · V` with the token-mixing matrices below, against the recurrence                  |       7.65e-6 |
| `torch.nn.functional.group_norm` against the recurrence's GroupNorm                       |       3.81e-6 |

The independent implementation is BlinkDL's `rwkv` package,
[`rwkv==0.8.32`](https://pypi.org/project/rwkv/0.8.32/) (wheel SHA-256
`d3f0594810a4a7e91569f177003ec22294c3eabc91c5f8d0ff3511702d4b67c1`). It ran its `RWKV_x070` CPU fp32 path
(`RWKV_JIT_ON=0`, `RWKV_CUDA_ON=0`). It received the same tensors, renamed from fla's layout to BlinkDL's
original key layout, with the LoRA matrices transposed and no values changed. Its own tokenizer produces
the same five token ids. The model's code SHA-256 and vocabulary SHA-256 are in `capture.reference`.
It uses 0.606531 for e^−0.5, and float32 summation order differs, which accounts for the small differences.

## What the tensors mean

Each `block-NN.f32` is little-endian float32. Offsets and shapes are in the manifest. The first axis is the token
unless noted.

- `input` is the block's residual input. Block 1's input is `ln0(embedding)`, stored as `preNorm` in `model.f32`.
- `attnNorm` is `ln1(input)`. `shifted` is the previous token's `attnNorm` (zeros for the first token).
- `xr`, `xw`, `xk`, `xv`, `xa`, `xg` are the six token-shifted inputs.
- `r`, `kRaw` (W_k x̂), `k` (k̃), `kk` (κ̂), `vRaw` (W_v x̂), `vMix` (blocks 2–12), `v`, `a` and `g` are the
  projections. `wLora` is the decay pre-activation and `decay` is w.
- `state` is `[token][head][value index][key index]`: the full 64×64 state of every head after every token.
- `wkv` is `o = S r` before GroupNorm. `groupNorm`, `bonus`, `gated` (`(groupNorm + bonus) ⊙ g`) and
  `timeMix` (W_o · gated) follow it.
- `mid` is `input + timeMix`. `ffnNorm` is `ln2(mid)`, `ffnShifted` is the previous token's `ffnNorm`,
  `ffnMixed` is the shifted channel-mix input, `ffnKeyRaw` is W_k x̂ (3072 wide), `ffnKey` is its relu², and
  `channelMix` is W_v · ffnKey. `output` is `mid + channelMix`.
- `mixing` is `[head][t][i]`: `A[t,i] = k̃_iᵀ G_{i+1} ⋯ G_t r_t` with `G_j = diag(w_j) − κ̂_j (κ̂_j ⊙ a_j)ᵀ`.
  These are the token-to-token weights the recurrence applies, so `o_t = Σ_i A[t,i] v_i`. They are computed in
  float64 from the recorded tensors. The upper triangle is 0 because no token reads a later one.
- `scores` (`k̃_i · r_t` for every pair) and `decayOnly` (`k̃_iᵀ diag(w_{i+1} ⋯ w_t) r_t`, the delta-rule
  term left out) are comparisons. The model does not compute them.

`model.f32` holds `embedding`, `preNorm`, `finalHidden` (after the final LayerNorm), the last token's
65,536 `logits`, and the top-4 `classifierWeights` with their `classifierTokenIds`.

## Reproduce

```sh
python3 -m venv .rwkv7-venv
.rwkv7-venv/bin/pip install torch==2.14.1 numpy safetensors
mkdir -p .rwkv-reference && cd .rwkv-reference && curl -LO https://files.pythonhosted.org/packages/5b/61/d7c764f01da87145342a8cdc3a86690e03cda4dfc2cf20377bdf5dc86b05/rwkv-0.8.32-py3-none-any.whl && unzip -o rwkv-0.8.32-py3-none-any.whl && cd ..
.rwkv7-venv/bin/python -I scripts/capture-linear-attention.py --reference-site .rwkv-reference
node --test tests/full-rwkv7.test.mjs
```

The first run downloads 382 MB of weights into `/private/tmp/rwkv7-capture-cache` (change it with `--cache-dir`).
