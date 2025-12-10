# Launch GRPO training with 6 GPUs (FAST MODE, skipping bad GPU 0)

NUM_GPUS=6

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

export NCCL_TIMEOUT=1800

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "=============================================="
echo "FAST GRPO Training with $NUM_GPUS GPUs"
echo "=============================================="

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    /home/azureuser/localfiles/cortex-project/RL/grpo_trainer.py

echo "=============================================="
echo "Training completed!"
echo "=============================================="

