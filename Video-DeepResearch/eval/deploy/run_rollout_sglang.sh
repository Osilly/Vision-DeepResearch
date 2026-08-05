#!/bin/bash
# Rollout (inference / policy) server — SGLang backend.
# Produces tool calls and final answers. Point TONGYI_BASE_URL at this endpoint.
#
# Pair with `run_reward_vllm.sh` (vLLM) for the reward / judge role.
#
# Env overrides:
#   MODEL_PATH      local path or HF id of the VL model
#   MODEL_NAME      served-model-name (must match TONGYI_MODEL_NAME on the client)
#   PORT            server port (default 8001)
#   GPUS            CUDA_VISIBLE_DEVICES value (default 0,1,2,3,4,5,6,7)
#   TP_SIZE         tensor-parallel-size (default 8)
#   GPU_MEM_FRAC    mem-fraction-static (default 0.8)
#   MAX_MODEL_LEN   context-length (default 160000)
#   CHAT_TEMPLATE   chat-template name or path (default qwen3-vl)

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-VL-30B-A3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-30B-A3B-Instruct}"
PORT="${PORT:-8001}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-160000}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3-vl}"

CUDA_VISIBLE_DEVICES="$GPUS" python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$GPU_MEM_FRAC" \
  --context-length "$MAX_MODEL_LEN" \
  --chat-template "$CHAT_TEMPLATE" \
  --disable-radix-cache
