"""
LLaVA-Med Prediction Script for Medical VQA Dataset
Generates predictions on the llava_med_eval_qa50_qa.jsonl dataset
"""

import json
import os
from PIL import Image
import torch
from tqdm import tqdm
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

DATA_DIR = "VisualQA-Dataset"
IMAGE_DIR = "VisualQA-Dataset/images"
INPUT_FILE = os.path.join(DATA_DIR, "llava_med_eval_qa50_qa.jsonl")
OUTPUT_FILE = "llava_med_vqa_predictions.jsonl"

CONV_MODE = "mistral_instruct"

def load_dataset(input_file):
    """Load the evaluation dataset from JSONL file."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def generate_prediction(image_path, question, tokenizer, model, image_processor):
    """Generate a prediction for a single image-question pair."""
    
    image = Image.open(image_path).convert("RGB")
    conv = conv_templates[CONV_MODE].copy()
    
    clean_question = question.replace("<image>", "").replace("\n", " ").strip()
    inp = f"{DEFAULT_IMAGE_TOKEN}\n{clean_question}"
    
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to("cuda")
    
    image_tensor = process_images([image], image_processor, model)
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size[::-1]],
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
    
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output

def run_predictions(tokenizer, model, image_processor):

    print(f"Loading dataset from {INPUT_FILE}...")
    dataset = load_dataset(INPUT_FILE)
    print(f"Loaded {len(dataset)} samples")
    
    results = []
    skipped = 0
    
    for sample in tqdm(dataset, desc="Generating predictions"):
        question_id = sample["question_id"]
        image_name = sample["image"]
        question = sample["text"]
        ground_truth = sample["gpt4_answer"]
        
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            skipped += 1
            continue
        
        try:
            prediction = generate_prediction(
                image_path, question, tokenizer, model, image_processor
            )
            
            result = {
                "question_id": question_id,
                "image": image_name,
                "question": question.replace("<image>", "").replace("\n", " ").strip(),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "domain": sample.get("domain", {}),
                "type": sample.get("type", "")
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            skipped += 1
            continue
    
    print(f"\nSaving {len(results)} predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Done! Processed {len(results)} samples, skipped {skipped}")
    return results
