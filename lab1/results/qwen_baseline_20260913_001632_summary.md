# Qwen2.5-0.5B-Instruct Q8_0 — Baseline Benchmark

## Run metadata
| Field | Value |
|---|---|
| Model file | `qwen2.5-0.5b-instruct-q8_0.gguf` |
| File size | 675,710,816 bytes (~676 MB) |
| SHA256 | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| Source | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf |
| Binary | `llama-server` — normal build (`llama.cpp/build/bin/`, not `-pg`), build `b10851-67672dc5b` |
| Command | `python3 benchmark.py --llama-server ../bin/llama-server --model ../models/qwen2.5-0.5b-instruct-q8_0.gguf --team lab1_baseline --report --out ../results/qwen_baseline_20260913_001632.json` |
| Server launch flags (auto) | `-t 4 -tb 4 --host 127.0.0.1 --port 8080 --no-warmup` |
| CPU configuration | All 4 Cortex-A53 cores online, 1.2 GHz (default/unmodified — no profiling config applied) |
| Board | Ultra96 (`pynq`), kernel 5.15.19-xilinx-v2022.1 aarch64, 1.9 GiB RAM |
| Timestamp | 2026-09-13 00:16:32 (start) |
| Wall clock | 524.8 s |
| Raw log | `logs/raw/qwen_baseline_bench_20260913_001632.log` |
| Raw JSON | `results/qwen_baseline_20260913_001632.json` |

## Performance
| Metric | Value |
|---|---|
| Load time | 9.67 s |
| Peak RSS | 1035.5 MB |
| Prefill (prompt) | 9.72 tok/s |
| Decode (generation) | 5.79 tok/s |
| TTFT median | 3191.7 ms |
| TTFT p95 | 3745.2 ms |
| Prompt tokens (total) | 3525 |
| Generated tokens (total) | 862 |

## Quality
| Section | n | Score % | Raw (Brier skill) | Argmax accuracy |
|---|---|---|---|---|
| HellaSwag | 20 | 34.9% | +0.089 | 55.0% |
| ARC-Easy | 20 | 63.7% | +0.329 | 75.0% |
| IFEval | 10 | 60.0% | +0.600 | 40.0% |
| **TOTAL** | 50 | **51.4%** | — | 60.0% |

Score: 25.71 / 50 points.
