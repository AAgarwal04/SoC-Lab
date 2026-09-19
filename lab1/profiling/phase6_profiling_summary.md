# Phase 6 — Profiling Results (Qwen2.5-0.5B-Instruct Q8_0, single A53 core @ ~300MHz)

## Run metadata
| Field | Value |
|---|---|
| Binary | `llama-cli` — `-pg` profiling build (`llama.cpp/build-profile/bin/`), same source as normal build, build `b10851-67672dc5b` |
| Model | `qwen2.5-0.5b-instruct-q8_0.gguf` (same file used for baseline in Phase 4) |
| Command | `./llama-cli -m qwen2.5-0.5b-instruct-q8_0.gguf -t 1 -tb 1 -st -p "Describe a system-on-chip in two sentences." < /dev/null` |
| CPU configuration | CPU 0 only online (cpus 1-3 disabled via `chcpu -d 1-3`), CPU 0 at 299999 kHz (~300MHz) via `scaling_setspeed` |
| Board | Ultra96 (`pynq`), kernel 5.15.19-xilinx-v2022.1 aarch64 |
| Timestamp | 2026-09-13 12:30:19 (start) |
| Reported inference speed | Prompt: 0.6 tok/s | Generation: 0.4 tok/s (vs. 9.72 / 5.79 tok/s at 4 cores/1.2GHz baseline — ~14-16x slower, consistent with 4x fewer cores x 4x lower clock) |
| Raw log | `logs/raw/profile_run_20260913_123019.log` |
| `gmon.out` | `profiling/gmon.out` (4,365,712 bytes), preserved alongside the exact profiling binary copy in `profiling/llama-cli` |
| gprof used | native aarch64 `gprof` 2.38 on the board (no cross-toolchain needed) |
| Full gprof output | `profiling/llama.perf` (18,393 lines: flat profile + call graph) |
| Total profiled runtime (gprof accounting) | 231.31 s |

## Flat profile — top functions by self time

| Function | % time | Self (s) | Calls | Self s/call | Total s/call |
|---|---|---|---|---|---|
| `ggml_vec_dot_q8_0_q8_0` | **91.53%** | 211.72 | 41,840,384 | 0.0000 | 0.0000 |
| `ggml_compute_forward_mul_mat` | 1.34% | 3.11 | 66 (+11,154 via call graph) | 0.05 | 3.35 |
| `ggml_compute_forward_flash_attn_ext` | 1.30% | 3.01 | 1,584 | 0.00 | 0.00 |
| `llama_vocab::impl::load(...)` | 0.65% | 1.51 | 3 | 0.50 | 2.09 |
| `ggml_vec_swiglu_f32` | 0.48% | 1.12 | 2,504 | 0.00 | 0.00 |
| `ggml_vec_dot_f16` | 0.41% | 0.94 | 1,767,024 | 0.00 | 0.00 |
| `std::_Hashtable<...>::find(...)` (BPE tokenizer map) | 0.37% | 0.85 | 3,087,507 | 0.00 | 0.00 |
| `std::__detail::_Map_base<...>::operator[]` (BPE tokenizer map) | 0.25% | 0.58 | 455,808 | 0.00 | 0.00 |
| `gguf_get_arr_str` | 0.17% | 0.39 | 1,819,938 | 0.00 | 0.00 |
| `quantize_row_q8_0` | 0.16% | 0.36 | 17,754 | 0.00 | 0.00 |

(Full 18,393-line output in `profiling/llama.perf`; everything past the top ~10 entries is <0.2% each.)

## Call graph — total-contribution hot path

```
llama_decode                                   95.8% total (221.70s / 66 calls)
 └─ llama_context::decode
     └─ llama_context::process_ubatch          95.7% total (221.47s)
         └─ llama_context::graph_compute
             └─ ggml_backend_sched_compute_splits
                 └─ ggml_backend_cpu_graph_compute
                     └─ ggml_graph_compute
                         └─ ggml_compute_forward_mul_mat   95.7% total (3.11s self + 218.22s children, 66+11,154 calls)
                             ├─ ggml_vec_dot_q8_0_q8_0      211.72s  (~97% of mul_mat's total, 41,840,384 calls)
                             ├─ ggml_compute_forward_flash_attn_ext   3.01s self + 1.23s children
                             ├─ ggml_compute_forward_glu (SwiGLU)     1.12s children
                             ├─ quantize_row_q8_0                     0.36s
                             └─ (everything else under mul_mat)       ~1.6s combined
```

## Bottleneck identification (based on total contribution, not just self time)

1. **The single dominant bottleneck is matrix multiplication against Q8_0-quantized weights**, specifically the `ggml_vec_dot_q8_0_q8_0` kernel invoked from `ggml_compute_forward_mul_mat`. This one function is 91.53% of *total* self time and, via the call graph, essentially *all* (≈97%) of `mul_mat`'s total (self+children) time — and `mul_mat` itself is 95.7% of the entire program's runtime. Every other function is a rounding error by comparison.
2. This is architecturally expected: LLM inference is dominated by GEMM/GEMV operations between the model's weight matrices and activations, and at Q8_0 the reference (non-repacked/non-SIMD-optimized-for-this-path) dot-product kernel is doing scalar or lightly-vectorized block-wise multiply-accumulate over 8-bit values plus dequantization scaling — with 41.8M calls at 1 core / 300MHz, this saturates the entire prompt+generation pass.
3. Secondary contributors — flash attention (1.30%), SwiGLU activation (0.48%), and BPE tokenizer hashtable lookups (~0.6% combined across the `std::_Hashtable`/`_Map_base` entries) — are an order of magnitude smaller and not meaningful optimization targets by comparison.
4. Model loading (`llama_vocab::impl::load`, 0.65%, 1.51s total) is negligible relative to the compute-bound decode loop under this configuration.
5. **Implication for the "single A53 @ 300MHz" target platform**: any optimization effort should focus on the Q8_0 matmul kernel itself — e.g., NEON-vectorized dot-product implementations, alternative quantization formats with cheaper unpacking, or blocking/tiling strategies to improve cache reuse — since that is where >90% of wall-clock time is spent.
