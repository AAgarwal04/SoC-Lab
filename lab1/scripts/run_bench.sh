#!/usr/bin/env bash
# Repeatable benchmark runner for Lab 1 Phase 5.
# Runs benchmark.py with identical settings against a single model, writing
# separate timestamped raw log + JSON per invocation (never overwrites).
#
# Usage (run on the Ultra96 board, from /home/macoto/lab1):
#   ./scripts/run_bench.sh <model_path_relative_to_llama-bench_dir> <tag>
#
# Example:
#   ./scripts/run_bench.sh ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf qwen_q4_k_m
set -euo pipefail

MODEL="$1"
TAG="$2"

LAB1_ROOT="/home/macoto/lab1"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LAB1_ROOT}/logs/bench_${TAG}_${TS}.log"
OUT="${LAB1_ROOT}/results/bench_${TAG}_${TS}.json"

cd "${LAB1_ROOT}/llama-bench"

echo "=== ${TAG} : $(date) ===" | tee -a "${LOG}"
echo "model: ${MODEL}" | tee -a "${LOG}"

# Identical settings across all models: default suite sizes (20 HellaSwag,
# 20 ARC-Easy, 10 IFEval), default seed=42, default threads (all online cores).
python3 benchmark.py \
    --llama-server ../bin/llama-server \
    --model "${MODEL}" \
    --team "${TAG}" \
    --report \
    --out "${OUT}" \
    >> "${LOG}" 2>&1

echo "EXIT:$?" | tee -a "${LOG}"
echo "raw log: ${LOG}"
echo "json:    ${OUT}"
