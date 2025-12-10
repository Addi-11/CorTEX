"""
LLaVA-Med Prediction Script for Finetuned LoRA Model
Loads the base model + LoRA adapter and generates predictions
"""

import json
import os
import torch
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLaVA-Med'))

from transformers import AutoTokenizer
from peft import PeftModel
from llava.model import LlavaMistralForCausalLM
from llava.conversation import conv_templates


BASE_MODEL_PATH = "/home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"


def load_finetuned_model(base_model_path, lora_adapter_path):
    """Load base model and apply LoRA adapter."""
    
    print(f"Loading base model from {base_model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    model = model.cuda()
    model.eval()
    
    print("Model loaded successfully!")
    return tokenizer, model


def load_medinst_dataset(input_file):
    """Load the MedInst dataset from JSON file (one JSON object per line)."""
    data = []
    with open(input_file, 'r') as f:
        content = f.read().strip()
        
        if content.startswith('['):
            data = json.loads(content)
        else:
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def format_prompt(sample):
    """Format the MedInst sample into a prompt for LLaVA-Med."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    
    if instruction and input_text:
        prompt = f"{instruction}\n\n{input_text}"
    elif input_text:
        prompt = input_text
    else:
        prompt = instruction
    
    return prompt


def generate_prediction(prompt, tokenizer, model, conv_mode="mistral_instruct", max_new_tokens=256):
    """Generate a prediction for a text-only prompt."""
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    input_ids = tokenizer(full_prompt, return_tensors='pt').input_ids.to("cuda")
    input_length = input_ids.shape[1]
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    if output_ids.shape[1] > input_length:
        new_tokens = output_ids[0, input_length:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    else:
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "[/INST]" in full_output:
            output = full_output.split("[/INST]")[-1].strip()
        else:
            output = full_output.strip()
    
    return output


def normalize_answer(answer):
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    for prefix in ["the answer is", "answer:", "the correct answer is"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    answer = answer.rstrip('.,;:')
    return answer


def check_accuracy(prediction, ground_truth):
    """Check if prediction matches ground truth."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    if gt_norm in pred_norm:
        return True
    
    if pred_norm.startswith(gt_norm):
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Run predictions with finetuned LLaVA-Med LoRA model")
    parser.add_argument("--base-model-path", type=str, default=BASE_MODEL_PATH,
                        help="Path to base LLaVA-Med model")
    parser.add_argument("--lora-adapter-path", type=str, required=True,
                        help="Path to LoRA adapter (e.g., llava-med-finetuned-test)")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to input JSON file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct",
                        help="Conversation mode (default: mistral_instruct)")
    
    args = parser.parse_args()
    
    tokenizer, model = load_finetuned_model(args.base_model_path, args.lora_adapter_path)
    
    print(f"Loading dataset from {args.input_file}...")
    data = load_medinst_dataset(args.input_file)
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"Processing {len(data)} samples...")
    
    correct = 0
    total = 0
    results = []
    
    with open(args.output_file, 'w') as f:
        for sample in tqdm(data, desc="Generating predictions"):
            prompt = format_prompt(sample)
            prediction = generate_prediction(prompt, tokenizer, model, args.conv_mode)
            
            ground_truth = sample.get("output", "")
            is_correct = check_accuracy(prediction, ground_truth)
            
            if is_correct:
                correct += 1
            total += 1
            
            result = {
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
            }
            
            f.write(json.dumps(result) + "\n")
            results.append(result)
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_file}")
    
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "base_model_path": args.base_model_path,
        "lora_adapter_path": args.lora_adapter_path,
        "input_file": args.input_file,
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
