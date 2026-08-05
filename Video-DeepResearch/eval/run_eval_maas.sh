#!/bin/bash
# VideoDR evaluation via OpenAI-compatible API (MaaS / any OpenAI endpoint).
#
# Usage:
#   bash run_eval_maas.sh [api_key] [base_url] [reward_url] [mode] [model_name]
#
#   api_key:    API key for the inference endpoint
#   base_url:   Base URL, e.g. https://<maas-host>/v1
#   reward_url: SGLang /generate URL for the judge model, e.g. http://10.x.x.x:13141
#               (leave empty "" to reuse the MaaS API as judge)
#   mode:       "tool"   — multi-turn deep research (search/visit/select_crop_search)
#               "direct" — single-turn answer from keyframes only
#               "both"   — run both modes sequentially
#   model_name: tag for output subdir (defaults to MODEL)
#
# Output layout: {OUTPUT_DIR}/{model_name}/{mode}/
#
# Examples:
#   bash run_eval_maas.sh "sk-xxx" "https://<maas-host>/v1" "http://10.x.x.x:13141" direct qwen3.5-35b
#   bash run_eval_maas.sh "sk-xxx" "https://<maas-host>/v1" ""                       both   my-model-tag

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Paths (override via env or edit here) -------------------------------
CSV="${CSV:-/path/to/VideoDR.csv}"
FRAMES_DIR="${FRAMES_DIR:-/path/to/frames}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/output/results}"

# ---- MaaS Defaults -------------------------------------------------------
DEFAULT_API_KEY="<YOUR_MAAS_API_KEY>"
DEFAULT_BASE_URL="https://<maas-host>/v1"
MODEL="qwen3.5-35b-a3b"

# ---- Args ----------------------------------------------------------------
API_KEY=${1:-"${DEFAULT_API_KEY}"}
BASE_URL=${2:-"${DEFAULT_BASE_URL}"}
REWARD_URL=${3:-"http://localhost:13141"}
MODE=${4:-"direct"}
MODEL_NAME=${5:-"${MODEL}"}

# ---- Extra HTTP headers for MaaS -----------------------------------------
MAAS_EMAIL="${MAAS_EMAIL:-<your-email@example.com>}"
MAAS_APP_ID="${MAAS_APP_ID:-qs-api}"

echo "============================================"
echo " VideoDR MaaS Evaluation"
echo " base_url     : ${BASE_URL}"
echo " model        : ${MODEL}"
echo " reward_url   : ${REWARD_URL}"
echo " mode         : ${MODE}"
echo " model_name   : ${MODEL_NAME}"
echo " frames_dir   : ${FRAMES_DIR}"
echo " output_dir   : ${OUTPUT_DIR}/${MODEL_NAME}/"
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

    local REWARD_ARGS=()
    if [ -n "${REWARD_URL}" ]; then
        REWARD_ARGS=(--reward-url "${REWARD_URL}")
    fi

    python3 "${EVAL_DIR}/run_eval_maas.py" \
        --csv                   "${CSV}"            \
        --frames-dir            "${FRAMES_DIR}"     \
        --api-key               "${API_KEY}"        \
        --base-url              "${BASE_URL}"       \
        --model                 "${MODEL}"          \
        --header                "x-maas-user-email=${MAAS_EMAIL}" \
        --header                "x-maas-app-id=${MAAS_APP_ID}"    \
        --system-prompt-file    "${PROMPT_FILE}"    \
        --output-dir            "${OUTPUT_DIR}"     \
        --model-name            "${MODEL_NAME}"     \
        --mode                  "${m}"              \
        --max-async-samples     4                  \
        --max-new-tokens        4096                \
        --log-level             INFO                \
        "${REWARD_ARGS[@]}"

    echo "[Eval] Done — results in ${OUTPUT_DIR}/${MODEL_NAME}/${m}/"
}

if [ "${MODE}" = "both" ]; then
    run_mode "tool"
    run_mode "direct"
else
    run_mode "${MODE}"
fi
