# Multi-GPU Training Script for LLaVA-Med Finetuning
# GPUs with torchrun (DDP)

NUM_GPUS=${1:-8}

DATA_PATH=${2:-"datasets/MedInstQA/MedQa_train.json"}
OUTPUT_DIR=${3:-"./llava-med-finetuned"}
NUM_EPOCHS=${4:-3}
BATCH_SIZE_PER_GPU=${5:-2}
GRAD_ACCUM=${6:-4}

echo "========================================"
echo "LLaVA-Med Multi-GPU Finetuning"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM))"
echo "========================================"

source /anaconda/etc/profile.d/conda.sh
conda activate azureml_py310_sdkv2


torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 finetune_llava_med.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --bf16 True \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 256 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ddp_find_unused_parameters False

echo "Training complete!"
