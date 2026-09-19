# AArch64 Optimization Features (Lab PDF §4f)

Board: Ultra96 (`pynq`), 4x Cortex-A53 (ARMv8.0-A), 1.2GHz, all cores online (normal/unmodified config — not the profiling 300MHz/1-core setup). Normal `llama-cli` build (`b10851-67672dc5b`, not `-pg`). Model: `qwen2.5-0.5b-instruct-q8_0.gguf`. Same fixed prompt for every run: `"Describe a system-on-chip in two sentences."` (`-st`, single-turn, `< /dev/null`), one run at a time.

## Features identified (from `llama-cli --help` and `ggml/src/ggml-cpu/arch/arm/` source)

1. **Thread/core parallelism** (`-t`/`-tb`) — distributes matmul work across the SoC's 4 Cortex-A53 cores.
2. **Weight repacking** (`--repack`/`--no-repack`, default: enabled) — reorganizes quantized weight blocks into an ARM-NEON-optimized memory layout for faster matmul (`ggml-cpu/arch/arm/repack.cpp`).
3. **Flash Attention** (`-fa on|off|auto`) — memory-efficient attention kernel (`ggml_compute_forward_flash_attn_ext`, seen at 1.30% of total runtime in the Phase 6 gprof profile).

Note: Cortex-A53 is ARMv8.0-A and does **not** support the `dotprod`/`i8mm`/`sve` ISA extensions (confirmed via `ggml-cpu/arch/arm/cpu-feats.cpp`, which explicitly checks `af.has_dotprod`/`has_i8mm`/`has_sve`). Several of the fastest repack kernel paths in `repack.cpp` are compiled only when `__ARM_FEATURE_MATMUL_INT8` or `__ARM_FEATURE_DOTPROD` is defined — neither is available on this board, which turns out to matter (see below).

## Results

| Feature | Setting | Prompt tok/s | Generation tok/s |
|---|---|---|---|
| Threads | `-t 1 -tb 1` | 2.5 | 1.6 |
| Threads | `-t 2 -tb 2` | 5.0 | 3.1 |
| Threads | `-t 4 -tb 4` | 9.8 | 5.6 |
| Weight repacking | default (enabled) | 9.8 | 5.6 |
| Weight repacking | `--no-repack` | 9.8 | 5.6 |
| Flash Attention | `-fa on` | 9.8 | 5.6 |
| Flash Attention | `-fa off` | 9.5 | 5.4 |

Raw logs: `logs/raw/opt_threads{1,2,4}_*.log`, `logs/raw/opt_repack_{on,off}_*.log`, `logs/raw/opt_fa_{on,off}_*.log`.

## Analysis

- **Thread scaling is the dominant, high-impact optimization on this platform**: near-perfectly linear from 1→2→4 threads (roughly 2x speedup per doubling: 1.6 → 3.1 → 5.6 tok/s generation). This is expected — the workload is compute-bound matmul with minimal cross-thread synchronization overhead at this model size, and the SoC's 4 A53 cores are the single biggest lever available for this platform (directly relevant to the Phase 6 finding that >90% of time is spent in the matmul dot-product kernel: more cores directly parallelizes that exact hot path).
- **Weight repacking had zero measurable effect** (identical 9.8/5.6 tok/s on vs off, and no repack-related messages appeared in either log at load time). This is a genuine architectural finding, not a measurement artifact: Q8_0 is not one of the block formats repacking targets, and even if it were, Cortex-A53's lack of `dotprod`/`i8mm` means the accelerated repack kernels in `repack.cpp` are gated out at compile time for this target. The `--repack` flag is a real optimization on newer ARM cores (e.g. those with `dotprod`/`i8mm`), but inert on this specific A53-based board with this quantization.
- **Flash Attention gave a small (~3-4%) speedup** (9.8 vs 9.5 prompt tok/s, 5.6 vs 5.4 generation tok/s). This is modest because the test prompt/response is short — FA's main advantage is avoiding O(n²) memory growth in the attention computation, which matters much more at longer context lengths. This is consistent with Phase 6 profiling, where `ggml_compute_forward_flash_attn_ext` was only 1.30% of total runtime for this same short-context workload — attention simply isn't the bottleneck here (matmul is), so FA's headline benefit doesn't show up much in a short single-turn test.

## Takeaway for the Ultra96/A53 target

Given the Phase 6 profiling result (matmul dot-product = ~91.5% of runtime), the highest-leverage optimization on this exact hardware is **exploiting all 4 cores** (thread count), since that directly parallelizes the dominant bottleneck. Repacking is not useful on this specific CPU (A53 lacks the required ISA extensions), and Flash Attention's benefit would grow with longer contexts but is minor for short single-turn prompts like the one tested here.
