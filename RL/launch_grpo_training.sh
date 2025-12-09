#!/bin/bash
# Launch GRPO training with 6 GPUs (FAST MODE, skipping bad GPU 0)

# Set number of GPUs
NUM_GPUS=6
# Use GPUs 1-6 (skip GPU 0 which is bad)
# PyTorch will see these as devices 0-5
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

# Increase NCCL timeout
export NCCL_TIMEOUT=1800

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Speed optimizations
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "=============================================="
echo "FAST GRPO Training with $NUM_GPUS GPUs"
echo "=============================================="

# Launch with torchrun for distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    /home/azureuser/localfiles/cortex-project/RL/grpo_trainer.py

echo "=============================================="
echo "Training completed!"
echo "=============================================="

