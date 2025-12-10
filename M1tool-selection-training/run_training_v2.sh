#!/bin/bash
# Run LLaVA-Med Tool Selection Training V2
# Enhanced training with better hyperparameters

set -e

# Set CUDA device(s) - use 6 GPUs (excluding GPU 0 with ECC errors)
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6"

# Activate conda environment
source /anaconda/etc/profile.d/conda.sh
conda activate azureml_py310_sdkv2

# Set environment variables
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false

cd /home/azureuser/localfiles/cortex-project

echo "=========================================="
echo "LLaVA-Med Tool Selection Training V2"
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo ""

# Run training with optimized hyperparameters (resume from latest checkpoint)
python train_tool_selection_v2.py \
    --model_name microsoft/llava-med-v1.5-mistral-7b \
    --output_dir checkpoints/llava-med-tool-selection-v2 \
    --data_dir datasets/finetuning \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --max_length 4096 \
    --bf16 \
    --gradient_checkpointing \
    --resume_from_checkpoint latest

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "End: $(date)"
