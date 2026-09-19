# Model Architecture Comparison (from GGUF header metadata)

Extracted directly from each model's GGUF header (key-value metadata + tensor list) using a pure-Python GGUF parser (`scripts/gguf_header_dump.py`) run against the first 64MB of each file (comfortably covers the full metadata/tensor-info section without transferring the whole file). This is the same information Netron displays for a `.gguf` file — GGUF has no separate computation-graph representation, only tensors plus hyperparameter metadata, so a header dump and a Netron view are informationally equivalent here.

## Hyperparameters

| | Qwen2.5-0.5B-Instruct | SmolLM2-360M-Instruct | Pythia-410M |
|---|---|---|---|
| `general.architecture` | `qwen2` | `llama` | `gptneox` |
| Layers (`block_count`) | 24 | 32 | 24 |
| Hidden size (`embedding_length`) | 896 | 960 | 1024 |
| Attention heads | 14 | 15 | 16 |
| KV heads (`head_count_kv`) | 2 | 5 | *(none — full MHA)* |
| Head dim (hidden/heads) | 64 | 64 | 64 |
| GQA ratio (Q:KV heads) | 7:1 | 3:1 | 1:1 (no GQA) |
| FFN size (`feed_forward_length`) | 4864 (5.4x hidden) | 2560 (2.7x hidden) | 4096 (4x hidden) |
| Context length | 32,768 | 8,192 | 2,048 |
| Vocab size | 151,936 | 49,152 | 50,304 |
| RoPE dimension count | (full head dim, 64) | (full head dim, 64) | **16** (partial rotary, 25% of the 64-dim head) |
| `use_parallel_residual` | — | — | **1** (parallel attn+MLP) |

## Tensor-naming-derived structural differences

**Qwen2.5-0.5B** (`blk.N.*`, per layer):
- `attn_q/k/v.weight` **and** `attn_q/k/v.bias` — separate Q/K/V projections, each with a bias term (a distinctive Qwen2 choice; most Llama-family models omit attention bias).
- `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` — three FFN matrices, no bias → **gated SwiGLU MLP** (`down(gate(x) * up(x))`).
- `attn_norm.weight` / `ffn_norm.weight` only, **no bias** on norms → **RMSNorm** (confirmed by presence of `layer_norm_rms_epsilon` in metadata).

**SmolLM2-360M** (`blk.N.*`, per layer):
- `attn_q/k/v.weight` with **no bias** — plain Llama-style linear projections.
- Same `ffn_gate/up/down.weight` pattern, no bias → SwiGLU MLP, same family as Qwen.
- Norms are weight-only → RMSNorm, same as Qwen. Structurally, SmolLM2 is architecturally identical in *shape* to Qwen2.5 (both Llama-family transformer blocks), just smaller and without the Qwen2-specific QKV bias.

**Pythia-410M** (`blk.N.*`, per layer):
- **`attn_qkv.weight`** (single fused `[1024, 3072]` matrix) instead of separate Q/K/V — classic GPT-NeoX fused-QKV projection, plus a bias term (`attn_qkv.bias`).
- No `head_count_kv` key at all → standard **multi-head attention**, every head has its own K/V (no GQA).
- `ffn_up.weight`/`ffn_down.weight` **only** (two matrices, not three) — a plain (non-gated) MLP, and **with bias terms** on every weight (`attn_output.bias`, `ffn_down.bias`, `ffn_up.bias`) — a much more "classic transformer" style than the bias-free SwiGLU blocks in the other two models.
- `attn_norm.weight` **and** `attn_norm.bias` present → confirms **LayerNorm** (LayerNorm has a learnable bias; RMSNorm does not) — directly distinguishes it from Qwen/SmolLM2's RMSNorm.
- `use_parallel_residual = 1` in the metadata confirms GPT-NeoX's parallel block design: attention and the MLP both read the *same* normalized input and their outputs are summed directly, rather than being applied sequentially (attn output feeding into a second, separately-normalized MLP) as in Qwen/SmolLM2.
- Partial rotary embeddings (`rope.dimension_count = 16` out of a 64-dim head, i.e. only 25% of each head's dimensions get positional rotation) — a known Pythia/GPT-NeoX design choice, versus full-head rotary in Qwen2.5/SmolLM2.

## Quantization confirmation (from tensor `type` field)

Across all three models, the large weight matrices (`attn_q/k/v/output.weight`, `ffn_*.weight`, `token_embd.weight`, `output.weight`) are tensor type 8 = **Q8_0** (matches the `Q8_0` files downloaded), while every norm weight/bias and every attention/FFN bias vector is type 0 = **F32** (full precision). This is a direct, file-level confirmation of the general quantization principle described in the lab report: bulk weight matrices are aggressively quantized, while small, precision-sensitive parameters (norms, biases) are kept at full precision.

## Summary

Qwen2.5-0.5B and SmolLM2-360M are structurally the same family (Llama-style: RMSNorm, SwiGLU, GQA, rotary-full), differing mainly in scale (24 vs 32 layers, 896 vs 960 hidden, 7:1 vs 3:1 GQA ratio) and Qwen2's extra QKV bias. Pythia-410M is a genuinely different architecture at every level checked: fused QKV, full MHA (no GQA), plain non-gated MLP, LayerNorm (not RMSNorm), parallel residual blocks, and partial rotary embeddings — consistent with its much older (2023, EleutherAI GPT-NeoX-derived) design lineage versus the newer Llama-family models.
