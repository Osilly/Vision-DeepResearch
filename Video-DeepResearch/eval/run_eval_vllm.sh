#!/bin/bash
# VideoDR evaluation via vLLM backend (OpenAI-compatible /v1 endpoint).
#
# Also supports openai/claude backends via extra env vars — vLLM-served
# closed-source proxies work identically to OpenAI/Claude endpoints.
#
# Usage:
#   bash run_eval_vllm.sh [inference_url] [reward_url] [mode] [model_name] [reward_backend] [reward_model]
#
#   inference_url:   vLLM base URL (http://host:8000/v1) or Claude Bedrock invoke URL
#   reward_url:      reward endpoint
#                      - sglang  => /generate URL
#                      - vllm    => base URL like http://host:8000/v1
#                    (default: sglang @ localhost:13141)
#   mode:            "tool" | "direct" | "both"   (default: both)
#   model_name:      tag for output subdir        (default: Qwen-3.5-35B-A3B)
#   reward_backend:  "sglang" | "vllm" | "openai" (default: sglang)
#   reward_model:    model id for vllm/openai reward (defaults to HF_CHECKPOINT)
#
# Env overrides:
#   CSV, FRAMES_DIR, OUTPUT_DIR, CONFIG, HF_CHECKPOINT, REWARD_API_KEY
#   BACKEND               "vllm" (default) | "openai" | "claude"
#   CLAUDE_TOKEN          required when BACKEND=claude
#   CLAUDE_THINKING       "1" enables extended thinking
#   CLAUDE_THINKING_BUDGET  budget tokens (default 4000)
#   CLAUDE_MAX_IMAGE_DIM  max image side
#   CLAUDE_MAX_IMAGE_BYTES max bytes per image
#   CLAUDE_TIMEOUT        HTTP timeout seconds

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Paths (override via env or edit here) -------------------------------
CSV="${CSV:-/path/to/VideoDR.csv}"
FRAMES_DIR="${FRAMES_DIR:-/path/to/frames}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/output/results}"
CONFIG="${CONFIG:-${EVAL_DIR}/config.yaml}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/model}"

INFERENCE_URL=${1:-"http://localhost:8000/v1"}
REWARD_URL=${2:-"http://localhost:13141"}
MODE=${3:-"both"}
MODEL_NAME=${4:-"Qwen-3.5-35B-A3B"}
REWARD_BACKEND=${5:-"sglang"}
REWARD_MODEL=${6:-"${HF_CHECKPOINT}"}
REWARD_API_KEY="${REWARD_API_KEY:-EMPTY}"

BACKEND="${BACKEND:-vllm}"

if [ "${BACKEND}" = "claude" ] && [ -z "${CLAUDE_TOKEN:-}" ]; then
    echo "[error] BACKEND=claude requires CLAUDE_TOKEN env var" >&2
    exit 1
fi

echo "============================================"
echo " VideoDR ${BACKEND} Evaluation"
echo " inference_url : ${INFERENCE_URL}"
echo " reward_backend: ${REWARD_BACKEND}"
echo " reward_url    : ${REWARD_URL}"
echo " mode          : ${MODE}"
echo " model_name    : ${MODEL_NAME}"
echo " output_dir    : ${OUTPUT_DIR}/${MODEL_NAME}/"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"
cd "${EVAL_DIR}"

run_mode() {
    local m="$1"
    echo ""
    echo "[Eval] Running — model: ${MODEL_NAME}  mode: ${m} ..."

    local PROMPT_FILE
    if [ "${m}" = "direct" ]; then
        PROMPT_FILE="${EVAL_DIR}/prompts/direct_system_prompt.txt"
    else
        PROMPT_FILE="${EVAL_DIR}/prompts/eval_system_prompt.txt"
    fi

    local BACKEND_ARGS=()
    if [ "${BACKEND}" = "vllm" ]; then
        BACKEND_ARGS=(
            --backend       vllm
            --vllm-url      "${INFERENCE_URL}"
            --vllm-model    "${HF_CHECKPOINT}"
        )
    elif [ "${BACKEND}" = "openai" ]; then
        BACKEND_ARGS=(
            --backend           openai
            --openai-base-url   "${INFERENCE_URL}"
            --openai-model      "${MODEL_NAME}"
        )
    elif [ "${BACKEND}" = "claude" ]; then
        BACKEND_ARGS=(
            --backend       claude
            --claude-url    "${INFERENCE_URL}"
            --claude-token  "${CLAUDE_TOKEN}"
        )
        if [ "${CLAUDE_THINKING:-0}" = "1" ]; then
            BACKEND_ARGS+=(--claude-thinking)
            if [ -n "${CLAUDE_THINKING_BUDGET:-}" ]; then
                BACKEND_ARGS+=(--claude-thinking-budget "${CLAUDE_THINKING_BUDGET}")
            fi
        fi
        [ -n "${CLAUDE_MAX_IMAGE_DIM:-}" ]   && BACKEND_ARGS+=(--claude-max-image-dim   "${CLAUDE_MAX_IMAGE_DIM}")
        [ -n "${CLAUDE_MAX_IMAGE_BYTES:-}" ] && BACKEND_ARGS+=(--claude-max-image-bytes "${CLAUDE_MAX_IMAGE_BYTES}")
        [ -n "${CLAUDE_TIMEOUT:-}" ]         && BACKEND_ARGS+=(--claude-timeout         "${CLAUDE_TIMEOUT}")
    else
        echo "[error] unknown BACKEND=${BACKEND} (expected vllm|openai|claude)" >&2
        exit 1
    fi

    local REWARD_ARGS=(
        --reward-backend     "${REWARD_BACKEND}"
        --reward-model-url   "${REWARD_URL}"
    )
    if [ "${REWARD_BACKEND}" = "vllm" ] || [ "${REWARD_BACKEND}" = "openai" ]; then
        REWARD_ARGS+=(
            --reward-model     "${REWARD_MODEL}"
            --reward-api-key   "${REWARD_API_KEY}"
        )
    fi

    python3 "${EVAL_DIR}/run_eval.py" \
        --csv                   "${CSV}"            \
        --frames-dir            "${FRAMES_DIR}"     \
        --config                "${CONFIG}"         \
        --system-prompt-file    "${PROMPT_FILE}"    \
        --hf-checkpoint         "${HF_CHECKPOINT}"  \
        --output-dir            "${OUTPUT_DIR}"     \
        --model-name            "${MODEL_NAME}"     \
        --mode                  "${m}"              \
        --max-async-samples     1                   \
        --max-turns             20                  \
        --max-new-tokens        8192                \
        --log-level             INFO                \
        "${BACKEND_ARGS[@]}"                        \
        "${REWARD_ARGS[@]}"

    echo "[Eval] Done — results in ${OUTPUT_DIR}/${MODEL_NAME}/${m}/"
}

if [ "${MODE}" = "both" ]; then
    run_mode "tool"
    run_mode "direct"
else
    run_mode "${MODE}"
fi
