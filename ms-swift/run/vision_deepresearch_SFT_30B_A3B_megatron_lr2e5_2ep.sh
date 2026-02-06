# 4 * 8 * 80GiB
wandb login `Your wandb key`

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=32 \
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'data/Vision-DeepResearch-Toy-SFT-Data.jsonl' \
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
    --max_epochs 2 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 2e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --save checkpoints/Vision-DeepResearch-SFT-30B-A3B-lr2e5-2ep \
    --eval_interval 50000000 \
    --save_interval 500 \
    --max_length 64000 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 128 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --context_parallel_size 2 \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --report_to wandb