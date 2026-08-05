#!/bin/bash
# Reward / judge server — vLLM backend (FIXED to vLLM).
# Also serves the extract / summary role used by VisitTool + crop_and_search.
# Point VLLM_EXTRACT_URL and JUDGE_BASE_URL at this endpoint, and set
# EXTRACT_MODEL == JUDGE_MODEL == MODEL_NAME.
#
# Env overrides:
#   MODEL_PATH      local path or HF id of the VL model
#   MODEL_NAME      served-model-name
#   PORT            server port (default 8002)
#   GPUS            CUDA_VISIBLE_DEVICES value (default 0,1,2,3,4,5,6,7)
#   TP_SIZE         tensor-parallel-size (default 8)
#   GPU_MEM_UTIL    gpu-memory-utilization (default 0.8)
#   MAX_MODEL_LEN   max_model_len (default 160000)

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-VL-30B-A3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-30B-A3B-Instruct}"
PORT="${PORT:-8002}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-160000}"

CUDA_VISIBLE_DEVICES="$GPUS" vllm serve \
  "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --served-model-name "$MODEL_NAME" \
  --max_model_len "$MAX_MODEL_LEN" \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching
