#!/bin/bash
# VideoDR evaluation via SGLang backend.
#
# Usage:
#   bash run_eval_sglang.sh [inference_url] [reward_url] [mode] [model_name]
#
#   inference_url:  SGLang /generate URL, e.g. http://10.x.x.x:13141
#   reward_url:     SGLang /generate URL for judge, defaults to inference_url
#   mode:           "tool" | "direct" | "both"    (default: both)
#   model_name:     tag for output subdir         (default: Qwen-3.5-35B-A3B)
#
# Env overrides:
#   CSV, FRAMES_DIR, OUTPUT_DIR, CONFIG, HF_CHECKPOINT
#
# Output layout: {OUTPUT_DIR}/{model_name}/{mode}/

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Paths (override via env or edit here) -------------------------------
CSV="${CSV:-/path/to/VideoDR.csv}"
FRAMES_DIR="${FRAMES_DIR:-/path/to/frames}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/output/results}"
CONFIG="${CONFIG:-${EVAL_DIR}/config.yaml}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/model}"

INFERENCE_URL=${1:-"http://localhost:13141"}
REWARD_URL=${2:-"${INFERENCE_URL}"}
MODE=${3:-"both"}
MODEL_NAME=${4:-"Qwen-3.5-35B-A3B"}

echo "============================================"
echo " VideoDR SGLang Evaluation"
echo " inference_url : ${INFERENCE_URL}"
echo " reward_url    : ${REWARD_URL}"
echo " mode          : ${MODE}"
echo " model_name    : ${MODEL_NAME}"
echo " frames_dir    : ${FRAMES_DIR}"
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
        --backend               sglang              \
        --sglang-url            "${INFERENCE_URL}"  \
        --reward-backend        sglang              \
        --reward-model-url      "${REWARD_URL}"

    echo "[Eval] Done — results in ${OUTPUT_DIR}/${MODEL_NAME}/${m}/"
}

if [ "${MODE}" = "both" ]; then
    run_mode "tool"
    run_mode "direct"
else
    run_mode "${MODE}"
fi
