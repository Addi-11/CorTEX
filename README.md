# CorTEX

# Steps:

## 1. Benchmarking LLava-Med Model

### Basic LLava-Med Testing

1. Run the [llava-med-basic-test.ipynb](./llava-med-basic-test.ipynb) to download the model weights.

### VQA Dataset (PMC article 60K-IM)
1. Run the [download_images.py](./download_images.py) to download (PMC-15M dataset). Pass the urls to get the images.
2. Datasetfor visual QA added in [VisualQA-Dataset](.datasets/VisualQA-Dataset/)
3. Generate predictions using this script:
    1. [predictions](./llava_med_vqa_predictions.jsonl) for the VQAdataset.
        Generated predictions: [custom_prediction_generation](./llava_med_vqa_predictions.jsonl)
    2. OR Using the LLava-Med prediction script
        Generated predictions: [predictions_using_llava_script](./llava_med_answers.jsonl)
    ```
    python llava/eval/model_vqa.py     --conv-mode mistral_instruct     --model-path /home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2     --question-file data/eval/llava_med_eval_qa50_qa.jsonl     --image-folder data/data/images     --answers-file ../llava_med_answers.jsonl     --temperature 0.0
    ```
4. Evaluations:
Use the llava-med Chat GPT Scaring Eval script
```
python llava/eval/eval_multimodal_chat_gpt_score.py \
    --answers-file ../llava_med_vqa_predictions.jsonl \
    --question-file ../datasets/VisualQA-Dataset/llava_med_eval_qa50_qa.jsonl \
    --scores-file ../llava_med_vqapred_scores.jsonl
```
To summarize the scores:
```
python llava/eval/summarize_gpt_review.py \
    --scores-file ../llava_med_vqa_scores.jsonl
```
|   |   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|
|   |conversation|  detailed_description|  chest_xray|        mri|  histology|      gross|    ct_scan|     overall|
|gpt4_score |              9.132867|              8.500000|    8.783784|   9.131579|   8.931818|   8.852941|   9.125000|    8.968912|
|pred_score             |  5.132867             |3.140000|    4.945946 |  3.789474 |  4.954545 |  4.764706 |  4.600000|    4.616580|
|pred_relative_score    | 59.254079             |37.325397|   56.790004|  42.544904|  56.401515|  61.715686|  51.041667|   53.573073|
|data_size              |143.000000             |50.000000|   37.000000|  38.000000|  44.000000|  34.000000|  40.000000|  193.000000|

### MedINST Dataset (Multiple Choice Question Answering)
(“Meta Dataset of Biomedical Instructions”) — A large-scale multi-domain/multi-task instruction dataset spanning ~7 million instruction-samples across 133 biomedical NLP task
1. Dataset used [](./datasets/MedInst/MedQa_test.json)
2. Run [predict_medinst](./predict_medinst.py) to generate predictions
```
python predict_medinst.py \
    --model-path /home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2 \
    --input-file datasets/MedInst32/MedQa_test.json \
    --output-file llava_med_medinst_predictions.jsonl
```
Total samples: 1273
Correct: 418
Accuracy: 32.84%

### MedS-Ins Dataset
Dataset link - https://huggingface.co/datasets/Henrychur/MedS-Ins
1. Download dataset using [MedS-Ins data download](./download_meds_ins.py). 136 tasks (4k examples each)

## 2. Finetuning the LLava-Med Model on MedINST dataset
We use SFT using LoRA-PEFT

1. Use [finetune-script for medinst](./finetune_llava_med.py)
For testing on a small batch
```
CUDA_VISIBLE_DEVICES=0 python finetune_llava_med.py \
    --data_path datasets/MedInstQA/MedQa_train.json \
    --max_samples 100 \
    --output_dir ./llava-med-finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --logging_steps 50 \
    --save_strategy "epoch" \
    --use_lora True
```

Using DeepSeed - Faster training
```
./train_multigpu.sh 4 datasets/MedInstQA/MedQa_train.json ./llava-med-finetuned 3 2 4
```

2. Generate predictions and evaluate the finetuned model [evaluation script](./predict_medinst_finetuned.py)
```
python predict_medinst_finetuned.py \
    --lora-adapter-path llava-med-finetuned \
    --input-file datasets/MedInstQA/MedQa_test.json \
    --output-file llava_med_finetuned_predictions.jsonl \
    --max-samples 20 \
```
Summary - 
  "total_samples": 1273,
  "correct": 661,
  "accuracy": 51.92,


  ## 3. Tool RAG
  Selects suitable candidate from Tool Universe based on description
  https://huggingface.co/mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B

  - Discovering - add suitable tools for the MedInst Dataset [tool_rag model](./tool_rag.ipynb)
  - Script to run inference on MedInstdataset [run_txagent_inference](./run_txagent_inference.py)
  - Run to run all the generation script on all GPUS, with different data portions - [launch-gpu](./launch_all_gpus.sh)

  ## 4. Train MedINST to produce tools and parameters for input - Training Tool Selection
  - [Training model for tool selection](./train_tool_selection.py)
  - [Run Training script](./run_tool_selection_training.sh)
  - [Evaluation trained model](./evaluate_tool_selection.py)
  - [Inference on trained model](./inference_tool_selection.py)
  