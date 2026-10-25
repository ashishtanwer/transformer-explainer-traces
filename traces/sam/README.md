# Recorded SAM ViT-B promptable segmentation

`manifest.json` and its `.f32` and `.png` files record the pretrained
[facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base/tree/70c1a07f894ebb5b307fd9eaaee97b9dfc16068f)
checkpoint, revision `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` (`model.safetensors` SHA-256
`892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579`), on one photo and two prompts.
The model is Apache-2.0 licensed; see `LICENSE-Apache-2.0.txt`. The files hold activations and six
small parameter tensors (`shared.f32`, below), never the complete checkpoint.

The capture ran on CPU in float32 with Python 3.12.12, PyTorch 2.14.1, Transformers 4.57.6 and the
slow `SamImageProcessor`. The manifest records the exact package versions, the capture-script hash, the
config hashes and every file's SHA-256, and the browser checks each `.f32` file against its hash before
using it.

## Inputs

| File                           | Source                                                                             | License       |
| ------------------------------ | ---------------------------------------------------------------------------------- | ------------- |
| `architectures/sam-beagle.jpg` | [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg), Jon Sullivan | Public domain |

`python scripts/capture-sam.py --prepare-image` downloads the 1000 × 750 Commons original, checks its
SHA-256 (`8c7bb1b3…df90b2a`) and saves it unmodified. The processor resizes it so the longest side is
1024 (1024 × 768), normalizes each channel, and pads the bottom with zeros to 1024 × 1024. In pixels
those zeros are the normalization mean, the grey-brown band the page draws under the photo. The
manifest records the SHA-256 of the resulting `pixel_values`.

The two prompts, in the photo's pixel coordinates:

- **Click on the tongue**: one foreground point at (610, 505). The processor scales it to (624.64,
  517.12) in the 1024-wide input, in patch row 32, column 39. With points and no box, SAM adds a
  padding point (label −1), so the decoder gets two prompt tokens.
- **Box around the dog**: (265, 95) to (1000, 750), scaled to (271.36, 97.28) and (1024, 768). The
  two corners are the two prompt tokens, and there is no padding point.

Both prompts reuse one image embedding: the image encoder runs once, and the mask decoder runs once
per prompt.

## Image encoder: what is stored and how it is reduced

The image encoder is a ViT-B/16 on the 1024 × 1024 input: a 64 × 64 grid of 4,096 patch tokens, 12
blocks of 12 heads (64 coordinates each), learned absolute positions, and decomposed relative
positions inside every attention. Blocks 3, 6, 9 and 12 (`globalAttnIndexes` 2, 5, 8, 11) attend
globally. The other eight pad the grid with zeros to 70 × 70, cut it into 25 windows of 14 × 14, and
attend only within a window. A neck (1 × 1 convolution 768 → 256, LayerNorm, 3 × 3 convolution,
LayerNorm) gives the 64 × 64 × 256 image embedding.

A global block has a 4,096 × 4,096 attention matrix per head, too large to store or draw. The
recording keeps every head's **complete attention row** for 18 query patches, in three strips of six
(`strips`):

| Strip             | Patches (row, column) | Why                                                                                                                           |
| ----------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Across the tongue | (32, 39)–(32, 44)     | Starts at the click's patch and crosses the window boundary at column 42                                                      |
| Across the eye    | (16, 25)–(16, 30)     | Crosses the window boundary at column 28                                                                                      |
| Into the padding  | (45, 60)–(50, 60)     | Runs from the bottom of the photo into the processor's padding, in the windows whose last six columns are the 70 × 70 padding |

- `strip-<s>.f32`
  - `windowAttention`, shape `[8 windowed blocks][12 heads][6 queries][196]`: the query's whole row
    over its own 14 × 14 window, row by row within the window, including any of the 70 × 70 grid's
    zero padding. The page draws a windowed block's rows on the window; the padding keys land in the
    hatched margin beyond the input.
  - `globalAttention`, shape `[4 global blocks][12][6][4096]`: the whole row over the 64 × 64 grid.
  - Both are the model's post-softmax weights, lossless float32. Every row sums to one.
- The page's head cards show the six strip patches as both queries and keys. They are 6 of 196 or 6
  of 4,096 keys, so each circle is shaded by its weight over the **largest weight in the query's whole
  row**. A key in another window is greyed: it is not attended in that block. The maps on the image
  shade each key by √(weight ÷ largest), so weights spread over thousands of keys stay visible; the
  page says so.
- `encoder.f32`, for the 18 patches in strip order:
  - `patch` (the 16 × 16 convolution), `position` (the learned absolute position) and `embedding`
    (their sum, the input to block 1).
  - Per block, `input`, `norm1`, `attentionOutput` (after the output projection, back on the grid),
    `residualAfterAttention`, `norm2`, `up` (first 8 of 3,072), `activation` (exact GELU), `down` and
    `output`, each `[12][18][8]`: the first 8 coordinates. The matching `…Norm` tensors hold the
    complete vector's L2 norm.
  - The neck: `neckConv1`, `neckNorm1`, `neckConv2`, `imageEmbedding` (first 8 of 256 plus norms),
    `imagePositional` (the decoder's fixed sin/cos encoding of the position) and `decoderImageInput`
    (the image embedding plus the learned no-mask vector). `imageEmbeddingNormMap` is the L2 norm of all
    4,096 image-embedding vectors, `[64][64]`.

## Prompt encoder and mask decoder

- `shared.f32` keeps six small parameter tensors whole: `gaussian` (the 2 × 128 matrix of the random
  Fourier positional encoding, a fixed buffer), `pointEmbed` (the four point-type rows: background,
  foreground, box corner 1, box corner 2), `notAPointEmbed`, `noMaskEmbed`, `iouToken` and
  `maskTokens`. With them a point's token re-derives exactly:
  c = ((x, y) + ½) / 1024, f = 2π (2c − 1) G, token = [sin f ‖ cos f] + pointEmbed[label]; the padding
  point's token is `notAPointEmbed`.
- `prompt-<p>.f32`, one per prompt. The decoder has 7 tokens: the IoU token, mask tokens 0–3, and the
  two prompt tokens.
  - `positional`: the model's own encoding of each prompt point before its type embedding (the padding
    point is encoded at (0, 0) and then replaced). `sparse`: the prompt tokens. `tokens`: all 7.
  - Per two-way layer (`[2][7][256]`, whole): `selfOutput`, `norm1`, `tokenToImageOutput`, `norm2`,
    `down`, `norm3`, and `up` / `activation` (first 8 of 2,048, ReLU, plus norms). The image side keeps
    the 18 strip patches: `imageToTokenOutput` `[2][18][8]` and `imageKeys` `[3][18][8]` (decoder input,
    after layer 1, after layer 2), plus norms. `finalOutput` and `finalNorm` are the final token → image
    attention and its LayerNorm.
  - Attention, post-softmax and complete: `selfAttention` `[2][8][7][7]`, `tokenToImageAttention`
    `[3][8][7][4096]` (layers 1 and 2, then the final attention), and `imageToTokenAttention`
    `[2][8][4096][7]`. Decoder self-attention has 8 heads of 32; cross-attention narrows 256 to 128,
    8 heads of 16.
  - `hyper` `[4][32]`: each mask token after its own hypernetwork MLP. `iou` `[4]`: the IoU head's
    predictions for mask tokens 0–3. `lowResMasks` `[4][256][256]`: the mask logits, the hypernetwork
    output dotted with the upscaled image embedding at every pixel.
- `prompt-<p>-mask-<m>.png`: mask token `m`'s mask at the photo's own 1000 × 750 size, white with
  alpha = mask. It is the processor's `post_process_masks` of the stored logits (bilinear to 1024,
  crop to 1024 × 768, bilinear to 1000 × 750, then > 0). Tokens 1–3 are the three candidates SAM
  returns with `multimask_output=True`; token 0 is the single-mask output it returns otherwise.

| Prompt              | Token 0 (single) | Mask 1 | Mask 2 | Mask 3 | Highest candidate |
| ------------------- | ---------------: | -----: | -----: | -----: | ----------------- |
| Click on the tongue |            0.870 |  0.743 |  0.884 |  0.934 | mask 3            |
| Box around the dog  |            0.988 |  0.981 |  0.976 |  0.836 | mask 1            |

The click is ambiguous. Its three candidates nest from the whole dog (44.0% of the photo) through the
tongue and nose (3.0%) to the tongue alone (1.9%). The IoU head ranks them; it is the model's own
estimate of each mask's quality, not a measured overlap.

## Validation

Forward hooks only observe; their outputs are cloned because SAM edits some tensors in place
afterwards. The capture asserts the following and stores the maximum errors in `validation`:

- The full model on the pixels, once per prompt, gives the same masks and IoU predictions as the image
  encoder run once plus the decoder on its embedding (0). The hooked and unhooked image embeddings are
  identical (0).
- Every encoder block's attention is reconstructed from its own Q, K and relative positions as
  softmax(QKᵀ/8 + rel) (0), then A · V and the output projection (0). Both residual additions, exact
  GELU and the neck hold (0). Each stored windowed row is recomputed by cutting the patch's window
  out of the zero-padded 70 × 70 grid by hand and running that block's attention on it alone
  (1.3 × 10⁻⁶).
- The prompt tokens re-derive from the stored Gaussian matrix and point embeddings in float64
  (1.3 × 10⁻⁶).
- Every decoder attention is reconstructed from its own projections (0), A · V and the output
  projection (1.8 × 10⁻⁵ in float32), every residual-plus-LayerNorm (0), ReLU (0), the upscaling and the
  hypernetwork MLPs (0).
- The mask logits rebuilt as hypernetwork · upscaled embedding equal the model's own low-resolution
  masks, for the three candidates and the single-mask output (0). The IoU head rebuilt from the final
  IoU token equals the model's IoU predictions (0).
- Post-processing the stored float32 logits gives the same masks, pixel for pixel, as post-processing
  the model's own (0 differing pixels).

`tests/sam-full.test.mjs` re-checks the hashes, that every stored attention row sums to one, the
block-to-block activation chain, the prompt-token derivation from `shared.f32`, and the IoU values.

## Reproduce

```sh
python scripts/capture-sam.py --prepare-image
HF_HOME=… python scripts/capture-sam.py
```

The first run needs `facebook/sam-vit-base` at the pinned revision in the Hugging Face cache (about
375 MB).
