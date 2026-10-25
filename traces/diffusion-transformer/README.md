# Recorded DiT-XL/2 sampling run

This folder holds one real class-conditional image generation by
[facebook/DiT-XL-2-256](https://huggingface.co/facebook/DiT-XL-2-256/tree/eab87f77abd5aef071a632f08807fbaab0b704d0),
pinned to revision `eab87f77abd5aef071a632f08807fbaab0b704d0`, run through diffusers' own
`DiTPipeline`. The DiT tab replays it; no model runs in the browser.

- Class 207 (golden retriever), seed 0 with a CPU `torch.Generator`, 50 DDIM steps (η = 0),
  classifier-free guidance scale 4: the pipeline's default settings.
- Captured on Apple MPS in float32 with PyTorch 2.14.1 and diffusers 0.42.0. `manifest.json` lists
  the exact package versions, the SHA-256 of both weight files and of the capture script, and every
  validation number.
- The model is DiT-XL/2 (Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)),
  with the bundled `stabilityai/sd-vae-ft-ema` VAE. Its weights are licensed
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The recorded activations and
  images are outputs of that model and are shared here for non-commercial, explanatory use with
  this attribution. No weights are distributed.

## What is recorded

DiT-XL/2 turns the 4 × 32 × 32 latent into 256 patch tokens (2 × 2 patches) of width 1152. It runs
28 blocks with 16 heads of 72 dimensions, and conditions every block with adaLN-Zero: one
conditioning vector c = MLP(sinusoid(t)) + E[class] is projected per block to six 1152D vectors
(shift β, scale γ and gate α, for attention and for the MLP). The final layer predicts 32 numbers
per patch, which unpatchify into 4 noise channels ε̂ and 4 variance channels Σ.

| File                                           | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `trajectory.f32`                               | All 50 steps: the latent x_t (51 frames, x_T to x_0), the class-207 and null-class noise predictions, the guided ε̂ the scheduler used, the scheduler's x̂₀, the 256-value sinusoid, the timestep embedding, c for both passes, both class embeddings, and the RMS of every block's six adaLN vectors for both passes. First 8 coordinates and full norms of the fixed position embedding.                                                                     |
| `step-980.f32`, `step-500.f32`, `step-100.f32` | Three timesteps in detail, class-207 pass. Block 1's complete Q, K, V and the model's own attention output before Wout, for all 16 heads and 256 patches. For every vector the diagram draws, the first 8 coordinates and the full L2 norm for all 256 patches. Block 1's full adaLN vectors, every block's adaLN previews, RMS and mean, the final adaLN vectors, the final layer's 32 outputs per patch, and both passes' complete model output (ε and Σ). |
| `final.png`                                    | The pipeline's own 256 px output image.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `decoded-xt-*.png`                             | The VAE decode of x_t at the three detailed steps, 256 px; the patch thumbnails are cut from these.                                                                                                                                                                                                                                                                                                                                                          |
| `trajectory-xt.jpg`, `trajectory-x0.jpg`       | 128 px JPEG decodes of x_t (51 frames) and of x̂₀ (50 frames), 10 frames per row.                                                                                                                                                                                                                                                                                                                                                                             |

Float32 files are little-endian; `manifest.json` gives each tensor's element offset, length and
shape, and each file's byte length and SHA-256, in the same format as the ViT recording.

### Downsampling, and what is not stored

The folder is about 20 MB. To keep it there:

- Only block 1 is recorded in full, at three of the 50 steps. Blocks 2–28 appear through their
  adaLN values. The diagram's block arrows are hidden for this reason.
- Activation previews keep the first 8 of 1152 (or 4608) coordinates, as in the other tabs; norms
  use every coordinate.
- The block detail is the class-207 pass. The null-class pass of classifier-free guidance is
  recorded through its class embedding, c, every block's adaLN RMS and previews, and its complete
  noise prediction at every step.
- Attention probabilities are not stored. The model's fused attention never materialises them. The
  page computes softmax(QKᵀ/√72) over all 256 keys from the recorded Q and K, in float64. The
  capture checks that these probabilities times V reproduce the model's own attention output.
- Image previews are display-only JPEGs.

Block 1's attention is nearly uniform: the mean row entropy is about 5.5 nats against
ln 256 ≈ 5.55, and the largest weight is 0.035 at t = 980, 0.067 at t = 500 and 0.08 at t = 100,
against 1/256 ≈ 0.0039. The diagram's matrix uses a 0–1 scale, so it looks almost white. The
attention map under the patch picker scales to the row's maximum and says so.

## Validation

Hooks only observe module inputs and outputs, and a wrapper observes `scheduler.step`; neither
changes a tensor. A second, unhooked pipeline call with the same seed gives the identical image
(maximum difference 0), and a second capture run reproduced every tensor file byte for byte. Reconstructions are computed in float64 from the captured float32 tensors:

| Check                                                      |   Max abs error |
| ---------------------------------------------------------- | --------------: |
| Hooked vs unhooked final image                             |               0 |
| Guidance mix ε*∅ + 4(ε_207 − ε*∅) vs the scheduler's input |          2.4e-7 |
| DDIM x\_{t−1} from x_t, ε̂ and ᾱ                            |          6.7e-7 |
| DDIM x̂₀                                                    |          2.7e-5 |
| Sinusoid recomputed from t                                 |          6.2e-5 |
| Timestep MLP, c = t_emb + y                                |  1.8e-7, 2.2e-7 |
| Patchify as a 16 → 1152 matrix product, + position         |  5.4e-6, 1.9e-6 |
| adaLN projection SiLU(c)·W_ada for all 28 blocks           |          1.0e-6 |
| adaLN modulation before attention / before the MLP         | 6.4e-7 / 3.3e-7 |
| softmax(QKᵀ/√72)·V vs the model's attention output         |          1.7e-6 |
| Gated residuals in block 1 (attention / MLP)               | 1.5e-5 / 8.1e-6 |
| Gated residuals in all 28 blocks                           |          9.3e-4 |
| GELU (tanh)                                                |          4.7e-7 |
| Final adaLN, unpatchify                                    |       4.1e-7, 0 |
| Final decode vs the pipeline's image                       |          1.7e-6 |

The 28-block residual error is float32 summation order on residual-stream vectors whose norms reach
about 6,000 by block 28, a relative error of a few parts per million. The capture also asserts that
every block's embedder copy is identical, so c is the same in all 28 blocks. The diffusers
conversion keeps 28 copies, which is why it counts 750M parameters against the paper's 675M.

`tests/dit-full.test.mjs` re-checks the files independently in JavaScript: checksums, attention
against the stored context, unpatchify, guidance, DDIM at every step, c, the position embedding and
the previews.

## Reproduce

diffusers 0.42 needs `huggingface-hub` 1.32 or later, and imports `transformers` when it is
installed, whose 4.x releases need hub < 1.0. Use an environment without `transformers`:

```sh
uv venv --python 3.12 /tmp/dit-trace-venv
uv pip install --python /tmp/dit-trace-venv/bin/python torch==2.14.1 diffusers==0.42.0 safetensors numpy==2.5.4 Pillow
HF_HOME=/tmp/hf-cache /tmp/dit-trace-venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/DiT-XL-2-256', revision='eab87f77abd5aef071a632f08807fbaab0b704d0')"
HF_HOME=/tmp/hf-cache HF_HUB_OFFLINE=1 /tmp/dit-trace-venv/bin/python scripts/capture-diffusion-transformer.py
node --test tests/dit-full.test.mjs
```

The download is about 3.3 GB. The capture takes a few minutes on Apple MPS; pass `--device cpu` to
run on the CPU (about 10 s per step). Another device or library version can change the floating
point slightly; the pinned revision, the environment record and the checks make that inspectable.
