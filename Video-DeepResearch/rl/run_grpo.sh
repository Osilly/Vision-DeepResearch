#!/bin/bash
# VideoDR GRPO training via slime + megatron + sglang rollout.
# Model: Qwen3.5-35B-A3B (MoE, 256 experts / top-8).
# Reference: run-qwen3.5-35B-A3B-videoDR-16gpus-bs64_len8k-grpo.sh (upstream slime).
#
# Cluster: 2 nodes × 8 × 80 GiB (world = TP × CP × PP × DP = 1 × 1 × 2 × 8 = 16).
# Experts are sharded along DP via EP=8.
#
# Prereqs:
#   - Ray cluster up (external head at RAY_JOB_ADDR, or set USE_EXTERNAL_RAY=0 to start locally)
#   - judge server (vLLM OpenAI-compatible) reachable at JUDGE_IP:JUDGE_PORT/v1/models
#
# Usage:
#   # external ray (default)
#   export SLIME_SCRIPT_EXTERNAL_RAY=1
#   export SLIME_SCRIPT_NUM_NODES=2
#   export SLIME_SCRIPT_GPUS_PER_NODE=8
#   export SLIME_SCRIPT_RAY_JOB_ADDR="http://127.0.0.1:8265"
#   bash rl/run_grpo.sh
#
# Env overrides:
#   SLIME_SCRIPT_EXP_NAME / MODEL_NAME / TRAIN_DATA / HF_MODEL_PATH / CUSTOM_CONFIG / SAVE_PATH
#   SLIME_SCRIPT_JUDGE_IP / JUDGE_PORT
#   NEG_ADV_KEEP_PROB   (default 0.2)
#   WANDB_KEY           (default: uses existing wandb login)

set -ex

# Upgrade wandb (matches launcher convention).
python3 -m pip install --quiet --upgrade wandb

if [ -n "${WANDB_KEY:-}" ]; then
    wandb login "${WANDB_KEY}"
fi

EXP_NAME=${SLIME_SCRIPT_EXP_NAME:-"qwen3.5-35B-A3B_videoDR_16gpus_bs64_len8k_grpo"}
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3.5-35B-A3B"}

# ===== cluster =====
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-2}
GPUS_PER_NODE=${SLIME_SCRIPT_GPUS_PER_NODE:-8}
USE_EXTERNAL_RAY=${SLIME_SCRIPT_EXTERNAL_RAY:-1}
RAY_JOB_ADDR=${SLIME_SCRIPT_RAY_JOB_ADDR:-"http://127.0.0.1:8265"}

# ===== paths — this dir mirrors slime-2.4 root: train.py + slime/ + examples/ + scripts/ =====
RL_DIR="$(cd "$(dirname "$0")" && pwd)"
SLIME_DIR="${RL_DIR}"

# Default TRAIN_DATA_RAW points at a non-existent path so users must set SLIME_SCRIPT_TRAIN_DATA.
TRAIN_DATA_RAW=${SLIME_SCRIPT_TRAIN_DATA:-"/path/to/rollout.jsonl"}
TRAIN_DATA="${TRAIN_DATA_RAW%.jsonl}_eval_style.jsonl"
HF_MODEL_PATH=${SLIME_SCRIPT_HF_MODEL_PATH:-"/path/to/${MODEL_NAME}"}
CUSTOM_CONFIG_PATH=${SLIME_SCRIPT_CUSTOM_CONFIG:-"${SLIME_DIR}/examples/vision_deepresearch/config.yaml"}

# ===== judge server (vLLM OpenAI-compatible) =====
JUDGE_IP=${SLIME_SCRIPT_JUDGE_IP:-"127.0.0.1"}
JUDGE_PORT=${SLIME_SCRIPT_JUDGE_PORT:-8001}

# ===== cleanup =====
pkill -9 sglang || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    ray stop --force || true
    pkill -9 ray || true
fi
pkill -9 slime || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    pkill -9 ray || true
fi
pkill -9 slime || true
pkill -9 redis || true

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ===== judge health check =====
curl -sf http://$JUDGE_IP:$JUDGE_PORT/v1/models > /dev/null
echo "Judge model server (vLLM) is up at $JUDGE_IP:$JUDGE_PORT"

cd "${SLIME_DIR}"

# ===== preprocess to eval-aligned messages format =====
if [ ! -f "${TRAIN_DATA}" ] || [ "${TRAIN_DATA_RAW}" -nt "${TRAIN_DATA}" ]; then
    echo "[preprocess] rebuilding ${TRAIN_DATA}"
    PYTHONPATH="${SLIME_DIR}:${PYTHONPATH:-}" \
        python3 examples/vision_deepresearch/preprocess_rollout_eval_style.py \
        --input  "${TRAIN_DATA_RAW}" \
        --output "${TRAIN_DATA}"
else
    echo "[preprocess] reusing ${TRAIN_DATA} (newer than raw)"
fi

SAVE_PATH=${SLIME_SCRIPT_SAVE_PATH:-"${SLIME_DIR}/checkpoints/${EXP_NAME}"}

CKPT_ARGS=(
    --hf-checkpoint ${HF_MODEL_PATH}
    --save ${SAVE_PATH}
    --save-interval 20
)

# Mirror swift's `--enable_thinking false` so Qwen3 chat template skips thinking
# mode during rollout (otherwise the model burns response budget on <think>...</think>).
APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key messages
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
    --rollout-shuffle
    --num-epoch 1
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --rollout-max-response-len 64000
    --rollout-temperature 1.0
    --global-batch-size 512
    --balance-data
    --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

# Reward is owned by the Gym env (examples/vision_deepresearch/env.py);
# --judge-url is read by env to know where to POST.
RM_ARGS=(
    --judge-url http://$JUDGE_IP:$JUDGE_PORT
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    # Down-sample negative-advantage trajectories (format-violating / repetitive loops).
    --negative-advantage-keep-prob ${NEG_ADV_KEEP_PROB:-0.2}
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.5
    --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
    # --use-wandb
    --wandb-project slime-videoDR
    --wandb-group ${EXP_NAME}
)

MISC_ARGS=(
    --colocate
)

# MoE parallelism: TP=1, CP=1, PP=2, DP=8, EP=8.
BACKEND_ARGS=(
    --train-backend megatron
    --load ${HF_MODEL_PATH}
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 2
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-expert-capacity-factor 2
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 64000
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --multimodal-load-workers 128
    --megatron-to-hf-mode bridge
)

MODEL_ARGS_FILE="qwen3.5-35B-A3B"
source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head \
        --node-ip-address ${MASTER_ADDR} \
        --num-gpus ${GPUS_PER_NODE} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"JUDGE_URL\": \"http://$JUDGE_IP:$JUDGE_PORT\"
  }
}"

ray job submit --address="${RAY_JOB_ADDR}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes ${NUM_NODES} \
    --actor-num-gpus-per-node ${GPUS_PER_NODE} \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${BACKEND_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${RM_ARGS[@]}
