#!/bin/bash
# VideoDR SFT — MegaTron backend via ms-swift, Qwen3-VL-30B-A3B-Instruct (MoE).
# Cluster: 4 nodes × 8 × 80 GiB (adjust NPROC_PER_NODE / TP / EP / CP for other topologies).
#
# Usage:
#   # single-node quick check
#   bash sft/run_video_dr_sft.sh
#
#   # multi-node (per-node invocation; launcher sets WORLD_SIZE / RANK)
#   WORLD_SIZE=4 RANK=$NODE_RANK bash sft/run_video_dr_sft.sh
#
# Env overrides:
#   MODEL_PATH      base model path (HF safetensors)          — default: Qwen3-VL-30B-A3B-Instruct
#   DATASET_PATH    space-separated jsonl paths (bash array)  — default: two internal paths, edit as needed
#   SAVE_PATH       checkpoint output dir                     — default: ./checkpoints/video_dr_w_text
#   WANDB_KEY       W&B API key (leave empty to disable)      — default: uses existing wandb login

set -e

# Upgrade wandb to avoid 0.27.0 filestream 409 fatal (matches launcher convention).
python3 -m pip install --quiet --upgrade wandb

if [ -n "${WANDB_KEY:-}" ]; then
    wandb login "${WANDB_KEY}"
fi

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-VL-30B-A3B-Instruct}"
# DATASET_PATH may hold one or more space-separated jsonl paths (bash array).
# Example: DATASET_PATH="/data/set1.jsonl /data/set2.jsonl" bash sft/run_video_dr_sft.sh
DEFAULT_DATASETS=(
    "/path/to/sft_train.jsonl"
)
if [ -n "${DATASET_PATH:-}" ]; then
    read -r -a DATASET_ARGS <<< "${DATASET_PATH}"
else
    DATASET_ARGS=("${DEFAULT_DATASETS[@]}")
fi
SAVE_PATH="${SAVE_PATH:-./checkpoints/video_dr_w_text}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=32 \
NNODES=${WORLD_SIZE:-1} \
NODE_RANK=${RANK:-0} \
NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
megatron sft \
    --model "${MODEL_PATH}" \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "${DATASET_ARGS[@]}" \
    --load_from_cache_file true \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --save "${SAVE_PATH}" \
    --eval_interval 500000 \
    --save_interval 500 \
    --max_length 80000 \
    --packing false \
    --num_workers 8 \
    --dataset_num_proc 128 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --context_parallel_size 2 \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --report_to wandb
