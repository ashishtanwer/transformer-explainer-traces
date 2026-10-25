# Recorded Gemma 3 270M · local and global attention

`manifest.json`, `boscombe.f32`, `caucus.f32` and `shared.f32` are forward passes of the pretrained
[unsloth/gemma-3-270m](https://huggingface.co/unsloth/gemma-3-270m/tree/edcf1fe394ee1168e7c5538afd104cfb1f7947e6)
checkpoint, revision `edcf1fe394ee1168e7c5538afd104cfb1f7947e6`. It is an ungated mirror of
[google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m/tree/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1):
both repositories list the same Git LFS SHA-256 for `model.safetensors`,
`abb58eb73aece5163624090d5383c59cbcdb9e9539ec84cb864f74386c4b21d5`, and for `tokenizer.model`,
`1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`. The mirror's `config.json` adds
`unsloth_fixed`; its `tokenizer.json` differs by two bytes, and the capture checks that the
SentencePiece tokenizer built from `tokenizer.model` gives the same token ids for both passages.

Gemma is provided under and subject to the Gemma Terms of Use found at
[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms); see `NOTICE.txt`. The files hold
activations, the norm scales of blocks 1 and 6 and the final norm, and the embedding rows of each
passage's 20 most likely next tokens. They never hold the complete checkpoint.

The capture ran on October 25, 2026 at 15:04:49 UTC, on CPU in float32 with eager attention, with
Python 3.12.12, PyTorch 2.14.1, Transformers 4.57.6 and NumPy 2.5.4. The bfloat16 weights are upcast
to float32, which is exact. The manifest records the package versions, the capture-script hash and
every file's SHA-256, and the browser checks each file against its hash before using it.

## The model

18 blocks of width 640. Blocks 6, 12 and 18 are global (full causal attention); the other 15 are
local, with a causal sliding window of 512 tokens: query `i` sees keys `i − 511 … i`. Local blocks
rotate Q and K with RoPE base 10,000 and global blocks with base 1,000,000. Each block has 4 query
heads and 1 shared key/value head of size 256 (multi-query attention). Q and K pass an RMSNorm of
their own (QK-norm) before RoPE, and scores are scaled by 1/√256, Gemma's `query_pre_attn_scalar`.
Every block normalizes its input and also its attention and MLP outputs before adding them back
(pre- and post-norms); Gemma's RMSNorm scales by (1 + γ). The MLP is GeGLU,
`W_down (GELU_tanh(W_gate x) ⊙ W_up x)`, 640 → 2 × 2,048 → 640. The token embedding is
multiplied by √640 and is tied to the output layer; the vocabulary has 262,144 tokens.

## Inputs

Each passage is the opening of a public-domain chapter from Project Gutenberg, with its hard line
wraps joined (blank lines stay paragraph breaks), cut after a fixed number of tokens. The cut text is
stored as `<id>.txt` and re-tokenizes to the same ids.

| Passage    | Source                                                                                                | Tokens | Ends with  | Next token     |
| ---------- | ----------------------------------------------------------------------------------------------------- | -----: | ---------- | -------------- |
| `boscombe` | Arthur Conan Doyle, _The Adventures of Sherlock Holmes_, IV. The Boscombe Valley Mystery (ebook 1661) |    663 | `“Bos`     | `combe`, 0.990 |
| `caucus`   | Lewis Carroll, _Alice’s Adventures in Wonderland_, III. A Caucus-Race and a Long Tale (ebook 11)      |    774 | `a Caucus` | `-`, 0.489     |

Both are longer than the window. At the last query the local blocks see only keys 151–662
(Boscombe) or 262–773 (Caucus). The name being completed appeared once before, outside that window:
"Boscombe Valley" at positions 68–70 and the title's "Caucus-Race" at 5–7. Global block 6, head 3
puts 0.914 and 0.987 of the last query's attention on keys beyond the window other than `<bos>`
(`farAttention`). The bfloat16 model gives the same top token, at 0.991 and 0.488.

## What the numbers mean

- Ten shown rows per passage (`rows`): `<bos>`, the earlier mention, the last key outside and the
  first key inside the last query's window, the local blocks' substitute sink (`localSink`, see
  below), and the last tokens. The last row is the prediction row.
- `rows`, shape `[18 blocks][4 heads][10 rows][T keys]`: the post-softmax attention returned by the
  model for the shown query rows, lossless. Every row sums to one over its visible keys.
- `maps`, shape `[18][4][B][B]` with `B = ⌈T / 8⌉`: all queries binned into 8 × 8-token blocks.
  Each value is the attention summed over the keys of a block and averaged over its queries, so every
  binned row still sums to one. It is a downsampled drawing aid; `rows` holds the exact values.
- `maskLow`, shape `[2][T]`: the first key each query may attend to under the 4-D masks the model
  received, local in row 0 and global in row 1. Each query sees one contiguous run of keys that ends
  at itself; the capture checks every entry of both masks against the window rule.
- Per block at the shown rows, `input`, `norm1`, `attentionOutput` (after `o_proj`), `postAttention`,
  `residualAfterAttention`, `norm2`, `down`, `postFeedForward` and `output`, shape `[18][10][8]`:
  the first 8 of 640 coordinates. The matching `…Norm` tensors are full-vector L2 norms. The
  residual stream reaches about 1e5 at a few coordinates.
- `b0.*` (block 1, local) and `b5.*` (block 6, global): whole vectors at the shown rows. `q`,
  `qNorm`, `qRope` are `[10][4][256]`; `k`, `kNorm`, `kRope`, `v` are `[10][256]` (one shared head);
  `context` is the 4 heads' outputs concatenated (1,024); `gate`, `up` and `activation`
  (GELU_tanh(gate) ⊙ up) are 2,048 wide; the rest are 640 wide.
- `embedding`: the scaled token embedding at the shown rows (640). `finalHidden`: the last token
  after the final norm. `candidateEmbeddings`: the embedding rows of the 20 `candidates`, so their
  logits are `finalHidden · row`.
- `candidates`: the 20 largest logits with float64 softmax over all 262,144; `logSumExp` is the
  normalizer.
- `kvCache`: the DynamicCache the model builds over the passage. Local blocks keep the last 511 keys
  and values (the next query adds its own to make 512); global blocks keep all T. At one key/value
  head of size 256, each cached token costs 2 × 256 × 2 bytes = 1 KiB per block in bfloat16.
- `localSink`: the key that the last query's 15 local blocks attend to most, on average over heads.
  Queries before position 512 use `<bos>` as an attention sink in every block. Later queries cannot
  see `<bos>` in local blocks, and they settle on another token: the "." at 509 (0.357 average
  attention) and the "’" at 523 (0.206).
- `shared.f32`: the RMSNorm scales γ of blocks 1 and 6 (input, QK, post-attention, pre- and
  post-feed-forward) and of the final norm.

## Validation

Forward hooks only observe. The capture reruns each passage without hooks and asserts the following,
storing the maximum errors in `validation`:

- Hooked and unhooked logits are identical (0). The SentencePiece and fast tokenizers agree (0
  mismatches). Every mask entry follows the window rule (0 mismatches).
- QK-norm (1.4e-5), RoPE with float32 angles as the model computes them (7.4e-6), and softmax of the
  masked scores (3.0e-6) reproduce the recorded attention. Attention outside a local window and
  after the query is exactly 0. A · V gives the recorded context (0), and GeGLU the recorded MLP
  activation (5.5e-6).
- Each block re-implemented in float64 from its recorded input reproduces its output to 8.9e-7 of
  the largest value (0.024 absolute against values near 1e5).
- The re-implementation chained from the embedding through all 18 blocks reproduces the model's
  logits to 5.3e-5, and the recorded final hidden state times the embedding to 2.2e-5.
- Every binned row sums to one (3.6e-7). Every cache length is 511 or T as expected (0 mismatches).

`node --test tests/local-global-full.test.mjs` re-derives the window masks, QK-norm, RoPE, attention
row sums, the binned maps, the residual additions, GeGLU and the candidate logits from these files.

## Reproduce

```sh
HF_HOME=.hf-cache python scripts/capture-local-global.py --prepare-text
HF_HOME=.hf-cache python scripts/capture-local-global.py
```

The first run needs the pinned checkpoint in the Hugging Face cache (536 MB), for example from
`huggingface_hub.snapshot_download('unsloth/gemma-3-270m', revision='edcf1fe394ee1168e7c5538afd104cfb1f7947e6')`.
If Python cannot download from Project Gutenberg, fetch `pg1661.txt` and `pg11.txt` another way and
pass `--source-dir`; their SHA-256 is checked either way.
