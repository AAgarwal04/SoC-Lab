# Phase 5 — Cross-Model / Quantization Comparison

Board: Ultra96 (`pynq`), 4x Cortex-A53 @ 1.2GHz (all cores online, unmodified — no profiling config applied), 1.9GiB RAM. Normal (non-`-pg`) `llama-server` build `b10851-67672dc5b`. Identical benchmark settings for every run: `benchmark.py` defaults (20 HellaSwag, 20 ARC-Easy, 10 IFEval, seed=42, threads=4/max).

## Performance

| Model | File size | Load (s) | Peak RSS (MB) | Prefill (tok/s) | Decode (tok/s) | TTFT median (ms) | Wall clock (s) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 (baseline) | 675.7 MB | 9.67 | 1035.5 | 9.72 | 5.79 | 3191.7 | 524.8 |
| Qwen2.5-0.5B Q5_K_M | 522.2 MB | 28.91 | 1006.5 | 7.53 | 5.09 | 4092.6 | 645.2 |
| Qwen2.5-0.5B Q4_K_M | 491.4 MB | 30.46 | 971.6 | 7.80 | 5.17 | 3962.2 | 635.2 |
| SmolLM2-360M-Instruct Q8_0 | 386.4 MB | 19.79 | 937.6 | 10.71 | 6.97 | 2982.9 | 495.7 |
| Pythia-410M Q8_0 | 433.4 MB | 21.30 | 1246.3 | 11.25 | 6.18 | 3814.8 | 835.8 |

## Quality

| Model | HellaSwag % (argmax) | ARC-Easy % (argmax) | IFEval % (argmax) | **Total %** | Score |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 (baseline) | 34.9% (55.0%) | 63.7% (75.0%) | 60.0% (40.0%) | **51.4%** | 25.71/50 |
| Qwen2.5-0.5B Q5_K_M | 33.0% (45.0%) | 62.2% (75.0%) | 70.0% (50.0%) | **52.1%** | 26.04/50 |
| Qwen2.5-0.5B Q4_K_M | 32.4% (50.0%) | 61.0% (65.0%) | 85.0% (70.0%) | **54.3%** | 27.16/50 |
| SmolLM2-360M-Instruct Q8_0 | 7.9% (30.0%) | 14.1% (45.0%) | 70.0% (50.0%) | **22.8%** | 11.39/50 |
| Pythia-410M Q8_0 | 19.3% (25.0%) | 18.3% (30.0%) | 25.0% (10.0%) | **20.1%** | 10.03/50 |

## Observations (from measured data only)

- **Quantization (Q8_0 → Q5_K_M → Q4_K_M) barely changed quality** on this 50-item sample (51.4% → 52.1% → 54.3%) — within noise for a 50-item suite, not evidence that lower quantization improves quality in general.
- **Decode speed decreased slightly** going to lower quantization (5.79 → 5.09 → 5.17 tok/s) despite smaller files — on this CPU, K-quant dequantization overhead is not fully offset by reduced memory bandwidth at this model size.
- **SmolLM2-360M was the fastest decoder** (6.97 tok/s) but by far the weakest on quality (22.8%) — expected, given it's a much smaller model and this suite (ARC-Easy/HellaSwag ranking, IFEval) is tuned toward instruction-following capability that a 360M model largely lacks.
- **Pythia-410M had the highest peak RSS** (1246.3 MB) of all five runs despite being architecturally smaller than Qwen-0.5B — plausible explanation: GPT-NeoX uses full multi-head attention (no GQA, unlike Qwen), and this run generated far more tokens (2600 vs ~750-860 for the others), growing the KV cache more.
- **Pythia scored lowest on quality** (20.1%, with negative raw Brier skill on HellaSwag/ARC-Easy, i.e. worse than random guessing) — consistent with it being a small base (non-instruct-tuned) model evaluated through a chat-formatted benchmark harness it wasn't trained for.
