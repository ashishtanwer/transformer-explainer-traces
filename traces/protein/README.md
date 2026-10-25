# Protein recording · ESM-2 35M on human ubiquitin

This directory holds three complete forward passes of
[ESM-2 35M](https://huggingface.co/facebook/esm2_t12_35M_UR50D/tree/6fbf070e65b0b7291e7bbcd451118c216cff79d8)
(`facebook/esm2_t12_35M_UR50D`), pinned to revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`
(MIT, weights SHA-256 `e3564781…d753150fc0`). It was captured on October 25, 2026 at 15:10 UTC on CPU
in float32, with Python 3.12.12, PyTorch 2.14.1, Transformers 4.57.6 and eager attention.
`manifest.json` records the package versions and the SHA-256 of the capture script, the config and
the installed `modeling_esm.py`.

ESM-2 is a BERT-style encoder over amino acids: 12 pre-LayerNorm blocks, 480 wide, 20 heads of 24,
an exact-GELU MLP of 1,920 and RoPE (no position embedding). Its vocabulary has 33 tokens. The input
is the 76-residue sequence of human ubiquitin (UniProt P0CG48, chain A of PDB 1UBQ), which becomes
78 tokens with `<cls>` and `<eos>`.

| Pass        | Input                           | Used for                                            |
| ----------- | ------------------------------- | --------------------------------------------------- |
| `contacts`  | the unmasked sequence           | the model's contact head (`predict_contacts`)       |
| `i44` · I44 | residue 44 replaced by `<mask>` | masked-residue prediction: I 66.0%, L 18.2%, V 9.3% |
| `k48` · K48 | residue 48 replaced by `<mask>` | masked-residue prediction: K 53.1%, R 22.2%, Q 7.8% |

I44 is the centre of ubiquitin's hydrophobic patch; K48 links the polyubiquitin chains that mark a
protein for degradation. In both passes the native residue is the model's top prediction.

## Files

All `.f32` files are lossless little-endian float32 tensors whose names, offsets and shapes are listed
in the manifest, which also checks every file's size and SHA-256.

- `contacts.f32` (unmasked pass): `attention` [12 blocks, 20 heads, 78, 78], the contact head's output
  `probability` [76, 76] and its pre-sigmoid `logit` [76, 76].
- `<example>/model.f32` (what the page loads): `attention` [12, 20, 78, 78]; `stream` [13, 78, 8], the first
  8 coordinates of the residual stream entering each block and leaving block 12, with full-vector
  `streamNorm`; and at the `<mask>` position the block-12 output, the final LayerNorm, the LM head's dense
  and LayerNorm outputs and all 33 `logits`.
- `<example>/block-00.f32` (for validation; the page does not load it): block 1 whole, at every position
  and head: `input` (x₀), `norm1`, `query`/`key`/`value` (the projections split into heads),
  `queryRotated`/`keyRotated` (recorded from the installed rotary module; the query is multiplied by
  1/√24 before rotation, as ESM does), `scores` (rotated Q·Kᵀ), `attention`, `context` (A·V),
  `attentionOutput` (after Wo), `residualAfterAttention`, `norm2`, `up`, `gelu`, `down` and `output`.
- `head.f32` (for validation): the final LayerNorm, the LM head (dense, LayerNorm, bias; its decoder is
  tied to the 33 × 480 token embeddings, also stored), the contact head's 240 weights and bias, and the
  checkpoint's 12 RoPE frequencies.
- `LICENSE-ESM.txt`: `head.f32`, and the 240 contact weights the manifest repeats, copy parameters
  unmodified from the checkpoint, so they carry the ESM MIT license (copied from facebookresearch/esm at
  commit `2b369911`). The other tensors are activations.

Two details of the real model show up in the files:

- **Token dropout.** ESM-2 gives `<mask>` no vector: its embedding row is set to zero, and every row is
  multiplied by 0.88 / (1 − m/L), where 0.88 = 1 − 0.15 · 0.8 is the training share left unmasked and m/L
  = 1/78 here, so the scale is 0.8914. The recorded `<mask>` row of x₀ is exactly zero.
- **RoPE frequencies.** The checkpoint stores its own 12 inverse frequencies, which round 10000^(−2i/24) by
  up to 4.8e-5. The model uses the stored ones, and so does every re-derivation here.

## Contact prediction

The contact head (`EsmContactPredictionHead`) drops the `<cls>` and `<eos>` rows and columns of all
240 attention maps (12 blocks × 20 heads), symmetrizes each one (S = A + Aᵀ), applies the average
product correction F = S − (S·1)(1ᵀS)/(1ᵀS1), then a logistic regression with one weight per map and a
sigmoid: p_ij = σ(b + Σ w F_ij). 179 of the 240 weights are exactly zero; b = −4.762.

The true contacts come from the 1UBQ crystal structure (X-ray, 1.8 Å; Vijay-Kumar, Bugg and Cook,
1987), downloaded from [RCSB](https://files.rcsb.org/download/1UBQ.pdb) (SHA-256
`d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161`, latest revision 14-FEB-24). PDB data
is in the public domain (CC0 1.0). Two residues touch when their Cβ atoms (Cα for the six glycines) are
closer than 8 Å; the manifest stores those 76 coordinates. With |i − j| ≥ 6 there are 139 contacts.

Precision of the model's most probable pairs:

| Separation     | Top L (76) | Top L/2 | Top L/5 |
| -------------- | ---------: | ------: | ------: |
| \|i − j\| ≥ 6  |      65.8% |   76.3% |   66.7% |
| \|i − j\| ≥ 12 |      56.6% |   71.1% |   73.3% |
| \|i − j\| ≥ 24 |      44.7% |   65.8% |   73.3% |

`contribution` in the manifest is each map's mean w·F over the top-L pairs; with the bias they add up to
those pairs' mean logit. Block 12 head 14 contributes most, and its F alone ranks pairs nearly as well as
the whole contact head: 64.5% of its top 76 pairs touch (`headPrecision`), against 65.8%. Masking one residue changes the contact
probabilities by at most 0.17 (I44) and 0.15 (K48), so the page reads contacts from the unmasked pass,
as the model's `predict_contacts` is meant to be used.

## Validation

The capture compares each hooked pass with an unhooked one (identical logits) and with the default SDPA
attention (largest logit difference 3.2e-5). At every block of every pass it re-derives, in float64:
the token-dropout embedding (≤ 9.0e-8, and the `<mask>` row exactly 0), the 1/√24 query scale (≤ 1.1e-7),
RoPE from the stored frequencies (≤ 1.1e-6), the softmax of the rotated scores (≤ 4.1e-7), A·V
(≤ 2.9e-6), both LayerNorms (≤ 1.7e-6), exact GELU (≤ 4.7e-7) and both residual additions (≤ 7.6e-6, float32
rounding on large coordinates). The LM head re-derived from the recorded final hidden state matches the
model's logits to 2.1e-5. The contact head applied to the recorded attention gives exactly the model's
`predict_contacts` output; the float64 reimplementation matches its logits to 2.8e-6 and its probabilities
to 2.7e-7. The maxima are stored in the manifest, and `tests/protein-full.test.mjs` repeats the contact
head, the true contacts, block 1 and the LM head from these files in JavaScript.

```sh
HF_HOME=.hf-cache .trace-venv/bin/python scripts/capture-protein.py   # downloads 136 MB and 1UBQ.pdb
.trace-venv/bin/python scripts/capture-protein.py --validate          # NumPy only: re-checks the files
```
