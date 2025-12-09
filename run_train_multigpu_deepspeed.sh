#!/usr/bin/env bash

set -e

# Adjust these paths
MODEL_PATH="microsoft/llava-med-v1.5-mistral-7b"
DATA_PATH="datasets/MedS-Ins"        # text-only MedInst/MedS-Ins json/jsonl
OUTPUT_DIR="outputs/llava_med_text"

python finetune_llava_med_v2.py \
  --model_path "$MODEL_PATH" \
  --use_lora True \
  --vision_finetune False \
  --freeze_vision_tower True \
  \
  --data_path "$DATA_PATH" \
  --max_length 2048 \
  --max_samples 0 \
  --conv_mode "mistral_instruct" \
  \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --bf16 True \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --report_to "none"


#!/usr/bin/env bash

set -e

# Adjust these paths
MODEL_PATH="microsoft/llava-med-v1.5-mistral-7b"
DATA_PATH="datasets/VisualQA-PMCArticle-Dataset/train/llava_med_alignment_500k.json"       # multimodal json/jsonl with "image" fields
IMAGE_ROOT="datasets/VisualQA-PMCArticle-Dataset/train/images"
OUTPUT_DIR="outputs/llava_med_vision"

python finetune_llava_med_v2.py \
  --model_path "$MODEL_PATH" \
  --use_lora True \
  --vision_finetune True \
  --freeze_vision_tower True \
  \
  --data_path "$DATA_PATH" \
  --image_root "$IMAGE_ROOT" \
  --max_length 2048 \
  --max_samples 0 \
  --conv_mode "mistral_instruct" \
  \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --bf16 True \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --report_to "none"
