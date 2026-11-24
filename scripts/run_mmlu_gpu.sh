#!/usr/bin/env bash
# MMLU zero-shot calibration baseline for the two Qwen3.5-4B variants on one GPU.
# Pick the GPU with CUDA_VISIBLE_DEVICES (default 0). Logs go to logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs

run() {  # name model out
  local log=$PWD/logs/$1.log
  echo "$(date -Is) start $1 on GPU $CUDA_VISIBLE_DEVICES (log: $log)"
  uv run python scripts/mmlu_baseline.py --model "$2" --out "$3" > "$log" 2>&1
  local rc=$?
  echo "$(date -Is) end $1 (exit $rc)"
  return $rc
}

run mmlu-qwen3.5-4b      Qwen/Qwen3.5-4B      results/mmlu_zeroshot_qwen3.5-4b
run mmlu-qwen3.5-4b-base Qwen/Qwen3.5-4B-Base results/mmlu_zeroshot_qwen3.5-4b-base
