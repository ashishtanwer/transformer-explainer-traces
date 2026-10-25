# Transformer Explainer recordings

Recorded forward passes that the hosted Transformer Explainer loads for its MLA,
encoder–decoder, speech, Mamba hybrid, vision–language, ViT, DiT, encoder-only,
SAM, protein and local–global tabs. The files are too large for the site's static
host, so the hosted build fetches them from a pinned commit of this repository
through `raw.githubusercontent.com`. The browser checks every file against the
SHA-256 recorded in its manifest before using it.

This repository holds data only. The application code lives elsewhere.

| Path | Checkpoint |
| --- | --- |
| `traces/deepseek-mla.json`, `traces/*-operations-deepseek-fact.json*` | [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) |
| `traces/t5-full.json` | [T5-small](https://huggingface.co/google-t5/t5-small) |
| `traces/speech/` | [Whisper base](https://huggingface.co/openai/whisper-base); audio: LibriSpeech dev-clean 1272-135031-0012 (CC BY 4.0) |
| `traces/hybrid/` | [Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B) |
| `traces/vlm/` | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| `traces/vit/` | [DeiT-tiny-patch16-224](https://huggingface.co/facebook/deit-tiny-patch16-224) |
| `traces/speculative-decoding/` | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) drafting for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (activations and probabilities only, no weight excerpts) |
| `traces/rwkv7/` | [RWKV-7 0.1B (g1)](https://huggingface.co/fla-hub/rwkv7-0.1B-g1) |
| `traces/diffusion-transformer/` | [DiT-XL-2-256](https://huggingface.co/facebook/DiT-XL-2-256), CC BY-NC 4.0; see its README |
| `traces/encoder-only/` | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) |
| `traces/sam/` | [SAM ViT-B](https://huggingface.co/facebook/sam-vit-base) |
| `traces/protein/` | [ESM-2 35M](https://huggingface.co/facebook/esm2_t12_35M_UR50D) on PDB [1UBQ](https://www.rcsb.org/structure/1UBQ) |
| `traces/local-global/` | [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m), read from the ungated mirror [unsloth/gemma-3-270m](https://huggingface.co/unsloth/gemma-3-270m) with the same weights; Gemma Terms of Use, see `traces/local-global/NOTICE.txt` |

Each manifest records the checkpoint revision, file hashes, capture script hash
and library versions. The recordings contain activations and small excerpts of
weight rows used to show arithmetic, never complete checkpoints. Each excerpt
remains under its checkpoint's license; see the model pages above and
`traces/licenses/`.

Files larger than the host's per-file limit are stored as `chunked-json` parts:
the original file is a manifest that lists its `.bin` parts and their hashes, and
the browser reassembles and verifies the exact bytes.
