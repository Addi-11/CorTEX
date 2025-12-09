#!/bin/bash
# Launch GRPO training with 6 GPUs

# Set number of GPUs
NUM_GPUS=6

# Use specific GPUs (0-5)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Increase NCCL timeout to 30 minutes (1800 seconds)
export NCCL_TIMEOUT=1800

# Helps with memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Launching GRPO Training with $NUM_GPUS GPUs"
echo "=============================================="

# Launch with torchrun for distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    /mnt/workspace/CorTEX/RL/grpo_trainer.py

echo "=============================================="
echo "Training completed!"
echo "=============================================="

