set -e

export CUDA_VISIBLE_DEVICES="6"
export HF_HOME="/mnt/huggingface_cache"
export TRANSFORMERS_CACHE="/mnt/huggingface_cache"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /home/azureuser/localfiles/cortex-project

echo "=============================================="
echo "LLaVA-Med Tool Selection Training"
echo "=============================================="
echo "Start: $(date)"
echo ""

echo "Combining datasets..."
cat datasets/finetuning/model1_tool_selection*.jsonl > datasets/finetuning/model1_combined.jsonl
wc -l datasets/finetuning/model1_combined.jsonl

source /anaconda/etc/profile.d/conda.sh
conda activate azureml_py310_sdkv2

pip install -q peft accelerate bitsandbytes scikit-learn

echo ""
echo "Starting training..."
PYTHONPATH="/home/azureuser/localfiles/cortex-project/LLaVA-Med:$PYTHONPATH" \
python train_tool_selection.py \
    --model_name "microsoft/llava-med-v1.5-mistral-7b" \
    --output_dir "checkpoints/llava-med-tool-selection" \
    --data_dir "datasets/finetuning" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --max_length 2048 \
    --bf16 \
    --gradient_checkpointing \
    2>&1 | tee logs/training_tool_selection.log

echo ""
echo "Starting evaluation..."
PYTHONPATH="/home/azureuser/localfiles/cortex-project/LLaVA-Med:$PYTHONPATH" \
python evaluate_tool_selection.py \
    --model_path "checkpoints/llava-med-tool-selection/final" \
    --base_model "microsoft/llava-med-v1.5-mistral-7b" \
    --data_dir "datasets/finetuning" \
    --num_test 100 \
    --output_file "evaluation_results.json" \
    2>&1 | tee logs/evaluation_tool_selection.log

echo ""
echo "=============================================="
echo "Training complete!"
echo "End: $(date)"
echo "=============================================="
