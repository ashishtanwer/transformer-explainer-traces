# Recorded CLIP ViT-B/32 dual encoder

`manifest.json` and its `.f32` files are one complete forward pass of the pretrained
[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)
checkpoint, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (weights SHA-256
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`). The Hugging Face
repository names no license. The weights come from [openai/CLIP](https://github.com/openai/CLIP),
which is MIT-licensed; see `LICENSE-openai-CLIP.txt`. The files hold activations and four
output rows of each projection matrix, never the complete checkpoint.

The capture ran on CPU in float32 with PyTorch 2.14.1 and Transformers 4.57.6. The manifest
records the exact package versions, the capture-script hash and every file's SHA-256, and the
browser checks each file against its hash before using it.

## Inputs

Four images and five captions go through one unmodified `CLIPModel` forward pass.

| Image          | File                               | Source                                                                                                                             | License          |
| -------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Colored shapes | `architectures/vit-shapes.png`     | The ViT tab's synthetic test image                                                                                                 | Repository image |
| Cat            | `architectures/clip-cat.jpg`       | [Closeup photo of a cat](https://commons.wikimedia.org/wiki/File:Closeup_photo_of_a_cat.jpg), RobotBlanket                         | CC0 1.0          |
| Beagle         | `architectures/clip-dog.jpg`       | [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg), Jon Sullivan                                                 | Public domain    |
| Astronaut      | `architectures/clip-astronaut.jpg` | [Bruce McCandless II during EVA in 1984](https://commons.wikimedia.org/wiki/File:Bruce_McCandless_II_during_EVA_in_1984.jpg), NASA | Public domain    |

`python scripts/capture-clip.py --prepare-images` downloads the 500-pixel Commons thumbnails,
checks their SHA-256, and saves the CLIP processor's own 224 × 224 view: shortest side resized to
224 with bicubic resampling, then a centre crop. The capture then reads these saved files, so they
are exactly what the model saw. The processor normalizes each channel with CLIP's mean and
standard deviation; the manifest records the SHA-256 of each resulting `pixel_values` tensor.

The captions are `a drawing of colored shapes`, `a photo of a cat`, `a photo of a dog`,
`a photo of an astronaut` and `a photo of a beagle`. Each is 7 tokens: `<|startoftext|>`, five
words and `<|endoftext|>`, so the batch has no padding. `matches` lists which captions describe
each image. These are human labels, not model output: the beagle photo matches both
"a photo of a dog" and "a photo of a beagle".

## What the numbers mean

- `image-<i>.f32`, one per image, 50 positions: `[CLS]` and 49 patches of 32 × 32 pixels in a
  7 × 7 grid, read row by row.
  - `attention`, shape `[12 blocks][12 heads][50][50]`: the post-softmax attention returned by the
    model, lossless. Every row sums to one over all 50 keys; the image encoder is bidirectional.
  - `start`: the learned `[CLS]` vector in row 0 and the patch convolution output (3,072 pixels →
    768, no bias) in rows 1–49. `position` is the learned position row, `embeddingSum` their sum,
    and `embedding` the result of CLIP's pre-LayerNorm, which is the input to block 1.
  - Per block, `input`, `norm1`, `attentionOutput` (after the output projection),
    `residualAfterAttention`, `norm2`, `up` (first 8 of 3,072), `activation` (QuickGELU,
    x · σ(1.702x)), `down` and `output`, each with shape `[12][50][8]`: the first 8 coordinates.
    The matching `…Norm` tensors hold the L2 norm of the complete vector.
  - `pooled`: `[CLS]` after block 12 (768). `pooledNorm`: after the post-LayerNorm. `projected`:
    `visual_projection` into the shared 512-D space, no bias. `normalized`: divided by its L2 norm.
- `caption-<j>.f32`, one per caption, 7 tokens. It has the same block tensors, with 8 heads,
  512 coordinates and 2,048 MLP coordinates. Attention is causal: every entry above the diagonal
  is exactly zero. `token` is the token-embedding row, and `embedding` adds the learned position
  (the text side has no pre-LayerNorm). `finalNorm` is the final LayerNorm over all tokens.
  `pooled` and `pooledNorm` are the `<|endoftext|>` position before and after it, `projected`
  is `text_projection` (512 → 512, no bias), and `normalized` is that vector over its L2 norm.
- `shared.f32`: the post-LayerNorm and final-LayerNorm weights and biases, and the first four
  output rows of `visual_projection` and `text_projection` with every input coordinate.
- `similarity`: `cosine[i][j]` is image `i` · caption `j` of the normalized vectors, and `logits`
  is the model's own `logits_per_image`, cosine × exp(`logitScaleRaw`) = 100.0. `probabilities`
  is the softmax of each image's row over the five captions, the zero-shot answer to "which
  caption?". `probabilitiesPerCaption` is the softmax of each caption's column over the four
  images. The training loss uses both directions. Softmax runs in float64 on the model's float32
  logits.

The recording does not train, fine-tune or prompt-engineer anything. The probabilities are what
the pretrained model gives for these five captions only. Another set of captions gives another
softmax.

## Validation

Forward hooks only observe. The capture reruns the model without hooks and asserts the following,
storing the maximum errors in `validation`:

- Hooked and unhooked logits are identical (0).
- Attention is reconstructed from the captured Q and K as softmax(QKᵀ/8 + mask) (0), and the
  head outputs as A · V (0). Every text attention entry above the diagonal is 0.
- Both residual additions and QuickGELU hold in every block (0). The embedding sums and hidden-state
  chaining hold (0).
- The post-LayerNorm and final LayerNorm, both projections and the L2 normalization match the
  model's `image_embeds` and `text_embeds` (≤ 1.2e-7).
- scale · cosine reproduces the model's `logits_per_image` to 2.5e-5 in float32 and 1.6e-5 in float64,
  and `logits_per_text` is exactly its transpose.
- Each image and each caption run alone give the batch's projected features to 5.7e-6, so no
  input reads another.

`node --test tests/clip-full.test.mjs` re-derives the residuals, QuickGELU, the pooled LayerNorm,
four coordinates of each projection, the normalization, the similarity logits and the softmax from
these files.

## Reproduce

```sh
HF_HOME=.hf-cache python scripts/capture-clip.py --prepare-images
HF_HOME=.hf-cache python scripts/capture-clip.py
```

The first run needs the pinned checkpoint in the Hugging Face cache (about 600 MB for
`pytorch_model.bin`), for example from
`huggingface_hub.snapshot_download('openai/clip-vit-base-patch32', revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')`.
