# CorTEX

## Steps:
1. Run the [llava-med-basic-test.ipynb](./llava-med-basic-test.ipynb) to download the model weights and image-urls data.

### VQA Dataset
2. Run the [download_images.py](./download_images.py) to download (PMC-15M dataset). Pass the urls to get the images.
3. Datasetfor visual QA added in [VisualQA-Dataset](./VisualQA-Dataset/)
4. Generate predictions using this script:
    1. [predictions](./llava_med_vqa_predictions.jsonl) for the VQAdataset.
        Generated predictions: [custom_prediction_generation](./llava_med_vqa_predictions.jsonl)
    2. OR Using the LLava-Med prediction script
        Generated predictions: [predictions_using_llava_script](./llava_med_answers.jsonl)
    ```
    python llava/eval/model_vqa.py     --conv-mode mistral_instruct     --model-path /home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2     --question-file data/eval/llava_med_eval_qa50_qa.jsonl     --image-folder data/data/images     --answers-file ../llava_med_answers.jsonl     --temperature 0.0
    ```
5. Evaluations:
Use the llava-med Chat GPT Scaring Eval script
```
python llava/eval/eval_multimodal_chat_gpt_score.py \
    --answers-file ../llava_med_answers.jsonl \
    --question-file ../VisualQA-Dataset/llava_med_eval_qa50_qa.jsonl \
    --scores-file ../llava_med_vqa_scores.jsonl
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