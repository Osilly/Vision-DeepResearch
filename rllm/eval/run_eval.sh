#!/bin/bash
# Run DeepResearch evaluation.
# Mirrors the CLI of the reference run_eval11.sh (rllm-free).
#
# Required env vars:
#   TONGYI_BASE_URL    e.g. http://localhost:8001/v1
#   TONGYI_MODEL_NAME  e.g. Qwen3-VL-30B-A3B-Instruct
#   PARQUET            either a single .parquet file, OR a directory containing
#                      .parquet files (each will be evaluated in turn)
#
# Optional env vars:
#   TONGYI_API_KEY (default EMPTY), PARALLEL_TASKS, MAX_SAMPLES,
#   TEMPERATURE, TOP_P, MAX_TOKENS, OUTPUT_DIR,
#   JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY

set -euo pipefail
shopt -s nullglob

BASE_URL="${TONGYI_BASE_URL:-}"
MODEL="${TONGYI_MODEL_NAME:-}"
API_KEY="${TONGYI_API_KEY:-EMPTY}"
PARQUET="${PARQUET:-}"
PARALLEL_TASKS="${PARALLEL_TASKS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

if [ -z "$BASE_URL" ]; then
    echo "[ERROR] TONGYI_BASE_URL is required" >&2
    exit 1
fi
if [ -z "$MODEL" ]; then
    echo "[ERROR] TONGYI_MODEL_NAME is required" >&2
    exit 1
fi
if [ -z "$PARQUET" ]; then
    echo "[ERROR] PARQUET is required (single .parquet file or directory)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Resolve PARQUET into a list of parquet files
parquet_files=()
if [ -d "$PARQUET" ]; then
    parquet_files=("$PARQUET"/*.parquet)
    if [ ${#parquet_files[@]} -eq 0 ]; then
        echo "[ERROR] No .parquet files found under directory: $PARQUET" >&2
        exit 1
    fi
elif [ -f "$PARQUET" ]; then
    parquet_files=("$PARQUET")
else
    echo "[ERROR] PARQUET path does not exist: $PARQUET" >&2
    exit 1
fi

build_extra_args() {
    local -n out=$1
    out=()
    [ -n "${MAX_SAMPLES:-}" ]    && out+=(--max-samples "$MAX_SAMPLES")
    [ -n "${TEMPERATURE:-}" ]    && out+=(--temperature "$TEMPERATURE")
    [ -n "${TOP_P:-}" ]          && out+=(--top-p "$TOP_P")
    [ -n "${MAX_TOKENS:-}" ]     && out+=(--max-tokens "$MAX_TOKENS")
    [ -n "${JUDGE_MODEL:-}" ]    && out+=(--judge-model "$JUDGE_MODEL")
    [ -n "${JUDGE_BASE_URL:-}" ] && out+=(--judge-base-url "$JUDGE_BASE_URL")
    [ -n "${JUDGE_API_KEY:-}" ]  && out+=(--judge-api-key "$JUDGE_API_KEY")
}

EXTRA_ARGS=()
build_extra_args EXTRA_ARGS

for pqt in "${parquet_files[@]}"; do
    name=$(basename "$pqt" .parquet)
    sub_output="${OUTPUT_DIR}/${name}"
    mkdir -p "$sub_output"
    echo "=================================================="
    echo "[EVAL] dataset = $name"
    echo "       parquet = $pqt"
    echo "       output  = $sub_output"
    echo "=================================================="

    python3 eval_runner.py \
        --parquet        "$pqt" \
        --base-url       "$BASE_URL" \
        --model          "$MODEL" \
        --api-key        "$API_KEY" \
        --parallel-tasks "$PARALLEL_TASKS" \
        --output-dir     "$sub_output" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${sub_output}/eval.log"
done

echo ""
echo "All datasets evaluated. Results under: $OUTPUT_DIR"
