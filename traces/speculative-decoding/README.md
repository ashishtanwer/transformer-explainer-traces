# Speculative decoding: Qwen2.5-0.5B-Instruct drafts for Qwen2.5-1.5B-Instruct

This folder holds two recorded generations of one chat answer with standard speculative
decoding (Leviathan et al. 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192); Chen et
al. 2023, [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)):

- **Draft:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775),
  revision `7ae557604adf67be50417f59c2c2f167def9a775`, weights SHA-256 `fdf756fa…288fb7fe`,
  494,032,768 parameters.
- **Target:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306),
  revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, weights SHA-256 `dd924a11…df483c6d3ee`,
  1,543,714,304 parameters.

Both checkpoints are Apache-2.0 licensed (see `../licenses/Apache-2.0.txt`). The recording holds
activations and probabilities only, never weights. The two models share one tokenizer: the
capture checks that `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` are
byte-identical, so draft and target probabilities refer to the same 151,936 token ids.

It was captured on October 25, 2026 at 15:22 UTC on CPU in float32 with eager attention, Python
3.12.12, PyTorch 2.14.1, Transformers 4.57.6, huggingface-hub 0.36.0 and tokenizers 0.22.2. The
manifest records the package versions, both weight hashes, the tokenizer hashes and the capture
script's SHA-256.

| File                         |   Bytes | Contents                                             |
| ---------------------------- | ------: | ---------------------------------------------------- |
| `manifest.json`              | 138,247 | prompt, both runs round by round, counts, validation |
| `verify-greedy-round-1.f32`  | 291,104 | block 1 of the greedy run's first verification pass  |
| `verify-sampled-round-1.f32` | 291,104 | block 1 of the sampled run's first verification pass |

The `.f32` files are little-endian float32; the manifest lists every tensor's offset and shape and
each file's byte size and SHA-256, which the page checks before use.

## What was recorded

The prompt is Qwen's chat template around `Why is the sky blue? Answer in one sentence.`, with
the template's default system message: 40 tokens. Both runs use **k = 4** and both models' KV
caches. Each round the draft proposes 4 tokens, one forward pass each; the target then runs
**one** forward pass over the new positions and scores 5 of them: one distribution for each
drafted token and one for the position after the last draft. Round 1's target pass starts with an
empty cache, so it is a single pass over the prompt and the 4 drafts (44 positions); later passes
take the 5 new positions against the cache. Rejected drafts are cropped from both caches.

- **Greedy, temperature 0.** A draft is kept while it equals the target's argmax at its position.
  At the first mismatch the target's argmax is emitted (the correction) and the later drafts are
  discarded; if all 4 match, the target's argmax after them is emitted for free (the bonus). The
  answer, `The sky appears blue because the Earth's atmosphere scatters sunlight in all
directions, with blue light being scattered more than other colors due to its shorter
wavelength.` plus `<|im_end|>`, is 32 tokens from **9 target passes** (and 36 draft passes);
  target-only greedy decoding takes 32 passes. 23 of 36 drafts were kept (64%), 3.56 tokens per
  target pass. Five rounds kept all 4 drafts and gained a bonus; four ended in a correction.
- **Sampled, temperature 1, seed 0.** The draft samples d ~ q. The target keeps it if a uniform
  draw u < min(1, p(d)/q(d)); the first rejected position is resampled from max(0, p − q),
  renormalized, and the later drafts are discarded; a fully kept round would sample a bonus from
  the target's next distribution. All draws come from one `torch.Generator` seeded with 0, and
  every u the rule compared is recorded. The answer is 40 tokens from **20 target passes** (80 draft passes); 21 of
  80 drafts were kept (26%). Plain decoding would take 40 target passes.

Per round (`greedy.rounds`, `sampled.rounds`): `drafted` (4 token ids), `accepted` (how many were
kept), `rejectedAt`, `next` and `nextKind` (`correction`, `bonus`, or `eos` when a kept draft
ended the answer), and for each of the 5 scored `positions`: the target's top 10 (`target`, id
and probability over the full vocabulary at temperature 1), its argmax and top-1 logit margin and
the log-sum-exp of its logits; for the 4 drafted positions also the draft's top 10 (`draft`), and
`p` and `q`, the target's and the draft's probability of the drafted token. The sampled run adds
`u`, `acceptProbability` = min(1, p/q) and, at a rejected position, `residualMass` = Σ max(0, p − q)
and the residual's top 10. Probabilities are float64 softmaxes of the float32 logits. Greedy
decoding is plain argmax: the checkpoints' `generation_config.json` (repetition penalty 1.1,
top-k 20, top-p 0.8) is not applied to either run or to the target-only reference.

`verify-*-round-1.f32` record block 1 of round 1's verification pass for every one of the 44
positions, as the Dense UI needs it: the embedding lookup, RMSNorm 1, Q (12 heads) and K, V (2
key/value heads, grouped-query attention) before and after RoPE, each head's output, the raw
`QKᵀ` scores, the model's own causal attention probabilities (12 × 44 × 44, from eager
attention), the attention output, the residual, RMSNorm 2, the SwiGLU gate, up projection,
SiLU(gate), their product, the down projection and the block output, plus the final RMSNorm
output. Vectors are stored as their first 8 coordinates (`name`) and the norm over all
coordinates (`nameNorm`: 1,536, 8,960 or 128). Blocks 2–28 are recorded only as per-position
norms of the block input, attention output, MLP output and block output.

## Validation

Stored in `validation`; the capture asserts every limit.

- **Greedy output = target-only greedy output**, token for token (`greedyEqualsTargetOnly`). The
  smallest gap between the target's two largest logits at any emitted token is 0.30, so no
  emitted token is a near tie. Each round's 5 rows were rescored without a cache: the argmax at
  every row matches and the log-sum-exp differs by at most 5.0e-8. The verification passes'
  logits differ from target-only decoding's logits at the same answer positions by at most
  2.2e-4 (float32 arithmetic in a different order).
- **The rejection-sampling rule preserves p.** At every drafted position of the sampled run, the
  rule's output distribution q·min(1, p/q) + (1 − Σ min(p, q))·r was computed in float64 over the
  whole vocabulary and differs from p by at most 7.0e-13; Σ max(0, p − q) equals the total
  variation distance between p and q within 3.6e-13. At round 1, position 3 (draft ` is`, where
  the target prefers ` appears`), 1,000,000 draws of the rule kept 540,293 drafts (expected
  0.5391) and land at total variation 1.1e-4 from p, against 9.3e-5 for 1,000,000 draws taken from
  p directly.
- **Hooks only copy tensors.** Rerunning round 1's target pass without hooks gives identical
  logits (max error 0), and rerunning both generations without hooks gives the same rounds.
- **Block 1 reconstructs exactly** from the captured tensors in float32: the embedding lookup,
  RMSNorm 1, the Q/K/V projections with their biases, RoPE (the recomputed keys equal the keys in
  the model's KV cache), the attention probabilities from the scores with the causal mask (rows
  sum to 1 within 2.4e-7; every weight on a later position is exactly 0), attention × V, the
  output projection, both residuals, SiLU and the SwiGLU product; block outputs chain into the next
  block's inputs and the final RMSNorm output times the tied embedding gives the logits. Apart
  from the row sums, all of these errors are 0.

`tests/speculative-decoding-trace.test.mjs` checks the artifact independently: hashes and shapes,
both acceptance rules round by round (the greedy argmax rule, and u < min(1, p/q) with the
recorded u, p and q), that the committed tokens spell the recorded answers and the greedy answer
equals the target-only answer, the counts, the probability lists, and every attention row of both
recorded passes against the recorded scores.

## Reproduce

Downloads about 4.1 GB of weights; needs about 9 GB of free memory.

```sh
HF_HOME=/private/tmp/explainer-hf-cache python scripts/capture-speculative-decoding.py \
  --cache-dir /private/tmp/explainer-hf-cache/hub
```

Use Python 3.12 with torch 2.14.1, transformers 4.57.6 and huggingface-hub 0.36.0. The models
must already be in the cache at the pinned revisions (`snapshot_download(..., revision=...)`);
the script refuses other weights.
