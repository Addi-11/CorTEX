"""
LLaVA-Med Prediction Script for MedInst32 Dataset (Text-Only Medical QA)
Generates predictions on the MedQa_test.json dataset
"""

import json
import os
import torch
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLaVA-Med'))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates


def load_medinst_dataset(input_file):
    """Load the MedInst dataset from JSON file (one JSON object per line)."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
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
    
    history = sample.get("history", [])
    if history:
        examples = "\n\nHere are some examples:\n"
        for i, (q, a) in enumerate(history, 1):
            examples += f"\nExample {i}:\n{q}\nAnswer: {a}\n"
        prompt = examples + "\nNow answer the following:\n" + prompt
    
    return prompt


def generate_prediction(prompt, tokenizer, model, conv_mode="mistral_instruct", max_new_tokens=256):
    """Generate a prediction for a text-only prompt."""
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    input_ids = tokenizer(full_prompt, return_tensors='pt').input_ids.to("cuda")
    input_length = input_ids.shape[1]
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    output = ""
    
    if output_ids.shape[1] > input_length:
        new_tokens = output_ids[0, input_length:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    if not output and "[/INST]" in full_output:
        parts = full_output.split("[/INST]")
        if len(parts) > 1:
            output = parts[-1].strip()
    
    if not output:
        if len(full_output) < len(full_prompt):
            output = full_output.strip()
        else:
            for marker in ["Answer:", "The answer is", "The correct answer"]:
                if marker.lower() in full_output.lower():
                    idx = full_output.lower().find(marker.lower())
                    output = full_output[idx:].strip()
                    break
    
    if not output:
        output = full_output.strip()
    
    return output


def extract_answer(prediction, options):
    """
    Try to extract a clean answer from the prediction.
    Match against the provided options.
    """
    prediction_lower = prediction.lower().strip()
    
    option_list = [opt.strip() for opt in options.split('/')]
    
    for option in option_list:
        if option.lower() in prediction_lower:
            return option
    
    return prediction


def run_predictions(
    model_path,
    input_file,
    output_file,
    conv_mode="mistral_instruct",
    max_samples=None
):
    """Run predictions on the MedInst dataset."""
    
    print(f"Loading model from {model_path}...")
    model_name = get_model_name_from_path(model_path)
    print(f"Model name: {model_name}")
    
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, 
        model_base=None, 
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device="cuda"
    )
    model.eval()
    
    print(f"Loading dataset from {input_file}...")
    dataset = load_medinst_dataset(input_file)
    print(f"Loaded {len(dataset)} samples")
    
    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Using first {max_samples} samples")
    
    results = []
    correct = 0
    total = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="Generating predictions")):
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        ground_truth = sample.get("output", "")
        
        prompt = format_prompt(sample)
        
        try:
            prediction = generate_prediction(
                prompt, tokenizer, model, conv_mode
            )
            
            if "Options:" in input_text:
                options_str = input_text.split("Options:")[-1].strip()
            else:
                options_str = ""
            
            extracted_answer = extract_answer(prediction, options_str) if options_str else prediction
            
            is_correct = ground_truth.lower().strip() in prediction.lower()
            if is_correct:
                correct += 1
            total += 1
            
            result = {
                "id": idx,
                "instruction": instruction,
                "input": input_text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct
            }
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"\nProgress: {idx + 1}/{len(dataset)} | Accuracy: {correct}/{total} ({100*correct/total:.2f}%)")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            results.append({
                "id": idx,
                "instruction": instruction,
                "input": input_text,
                "ground_truth": ground_truth,
                "prediction": f"ERROR: {str(e)}",
                "extracted_answer": "",
                "is_correct": False
            })
            continue
    
    print(f"\nSaving {len(results)} predictions to {output_file}...")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA-Med predictions on MedInst dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2",
        help="Path to the LLaVA-Med model"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="datasets/MedInst32/MedQa_test.json",
        help="Path to the MedInst dataset file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="medinst_predictions.jsonl",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="mistral_instruct",
        help="Conversation template mode"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    run_predictions(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        conv_mode=args.conv_mode,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
