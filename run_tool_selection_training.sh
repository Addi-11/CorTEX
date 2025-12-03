set -e

export CUDA_VISIBLE_DEVICES="0,1"
export HF_HOME="/mnt/huggingface_cache"
export TRANSFORMERS_CACHE="/mnt/huggingface_cache"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /home/azureuser/localfiles/cortex-project

echo "=============================================="
echo "LLaVA-Med Tool Selection Training"
echo "=============================================="
echo "Start: $(date)"
echo ""

# First, combine all datasets into one file
echo "Combining datasets..."
cat datasets/finetuning/model1_tool_selection*.jsonl > datasets/finetuning/model1_combined.jsonl
wc -l datasets/finetuning/model1_combined.jsonl

# Activate environment
source /anaconda/etc/profile.d/conda.sh
conda activate azureml_py310_sdkv2

# Install required packages if needed
pip install -q peft accelerate bitsandbytes scikit-learn

# Run training
echo ""
echo "Starting training..."
python train_tool_selection.py \
    --model_name "microsoft/llava-med-v1.5-mistral-7b" \
    --output_dir "checkpoints/llava-med-tool-selection" \
    --data_dir "datasets/finetuning" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --max_length 2048 \
    --bf16 \
    --gradient_checkpointing \
    --max_samples 5 \
    2>&1 | tee logs/training_tool_selection.log

# Run evaluation on test set
echo ""
echo "Starting evaluation..."
python evaluate_tool_selection.py \
    --model_path "checkpoints/llava-med-tool-selection/final" \
    --base_model "microsoft/llava-med-v1.5-mistral-7b" \
    --data_dir "datasets/finetuning" \
    --num_test 5 \
    --output_file "evaluation_results.json" \
    2>&1 | tee logs/evaluation_tool_selection.log

echo ""
echo "=============================================="
echo "Training complete!"
echo "End: $(date)"
echo "=============================================="
