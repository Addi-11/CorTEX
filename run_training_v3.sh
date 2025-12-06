#!/bin/bash
# Run LLaVA-Med Tool Selection Training V3
# Uses CLEANED dataset

# Use GPUs 1-6 (excluding faulty GPU 0)
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate azureml_py310_sdkv2

cd /home/azureuser/localfiles/cortex-project

echo "=============================================="
echo "LLaVA-Med Tool Selection Training V3"
echo "Using CLEANED dataset (no CallAgent, valid tools only)"
echo "=============================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Run training
python train_tool_selection_v3.py \
    --model_name microsoft/llava-med-v1.5-mistral-7b \
    --output_dir checkpoints/llava-med-tool-selection-v3 \
    --data_dir datasets/finetuning/cleaned \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --lora_r 128 \
    --lora_alpha 256 \
    2>&1 | tee logs/training_v3.log

echo ""
echo "End time: $(date)"
echo "Training complete!"
