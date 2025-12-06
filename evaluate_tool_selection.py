#!/usr/bin/env python3
"""
Evaluation script for LLaVA-Med Tool Selection Model.
Tests the fine-tuned model on held-out data and computes accuracy metrics.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from collections import defaultdict

# Add LLaVA-Med to path
LLAVA_PATH = "/home/azureuser/localfiles/cortex-project/LLaVA-Med"
if LLAVA_PATH not in sys.path:
    sys.path.insert(0, LLAVA_PATH)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tool selection model")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/llava-med-tool-selection/final",
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/llava-med-v1.5-mistral-7b",
                        help="Base model name")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Test JSONL file (if None, uses split from training data)")
    parser.add_argument("--data_dir", type=str, default="datasets/finetuning",
                        help="Directory with training data for split")
    parser.add_argument("--num_test", type=int, default=10,
                        help="Number of test samples to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="Output file for detailed results")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    return parser.parse_args()


def load_model(model_path, base_model):
    """Load the fine-tuned model with LoRA adapter."""
    from transformers import AutoTokenizer
    from peft import PeftModel
    from llava.model import LlavaMistralForCausalLM
    
    print(f"Loading base model: {base_model}")
    
    # Load tokenizer from adapter path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base LLaVA-Med model
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer


def load_test_data(data_dir, test_file=None, num_test=10):
    """Load test data - either from file or split from training."""
    import glob
    from sklearn.model_selection import train_test_split
    
    if test_file and os.path.exists(test_file):
        print(f"Loading test data from: {test_file}")
        with open(test_file, 'r') as f:
            data = [json.loads(line) for line in f]
        return data[:num_test]
    
    # Otherwise, load all data and split
    print(f"Loading data from {data_dir} and creating test split...")
    all_data = []
    pattern = os.path.join(data_dir, "model1_tool_selection*.jsonl")
    files = glob.glob(pattern)
    
    for f in sorted(files):
        with open(f, 'r') as file:
            samples = [json.loads(line) for line in file]
            all_data.extend(samples)
    
    print(f"Total samples: {len(all_data)}")
    
    # Use same split as training (seed 42)
    _, test_data = train_test_split(all_data, test_size=0.1, random_state=42)
    
    print(f"Test set size: {len(test_data)}")
    return test_data[:num_test]


def create_prompt(question):
    """Create inference prompt matching training format."""
    instruction = """You are a biomedical tool selector. Given a medical question, identify the appropriate biomedical database tools and their parameters needed to answer it.

Output each tool call on a new line in the format:
tool_name: {'parameter1': 'value1', 'parameter2': 'value2'}

If multiple tools are needed, list each on a separate line."""
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{question}

### Response:
"""
    return prompt


def generate_prediction(model, tokenizer, question, max_new_tokens=128):
    """Generate tool prediction for a question using manual generation loop."""
    prompt = create_prompt(question)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # Manual generation loop
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / 0.1
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract just the response
    if "### Response:" in full_output:
        response = full_output.split("### Response:")[-1].strip()
    else:
        response = full_output[len(prompt):].strip()
    
    # Clean up - take only first line or until newline
    response = response.split('\n')[0].strip()
    
    return response


def extract_tools_and_drugs(output_str):
    """Extract tool name and drugs from output like 'get_drug_indications: methotrexate, prednisone'."""
    tools = set()
    drugs = set()
    
    if not output_str:
        return tools, drugs
    
    # Handle multiple lines
    for line in output_str.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            tool_name = parts[0].strip().lower()
            tools.add(tool_name)
            
            if len(parts) > 1:
                drug_str = parts[1].strip()
                for drug in drug_str.split(','):
                    drug = drug.strip().lower()
                    if drug:
                        drugs.add(drug)
    
    return tools, drugs


def compute_metrics(predictions, ground_truths):
    """Compute accuracy metrics."""
    metrics = {
        'exact_match': 0,
        'tool_match': 0,
        'drug_precision': [],
        'drug_recall': [],
        'drug_f1': [],
        'partial_match': 0,
    }
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tools, pred_drugs = extract_tools_and_drugs(pred)
        gt_tools, gt_drugs = extract_tools_and_drugs(gt)
        
        # Exact match
        if pred.strip().lower() == gt.strip().lower():
            metrics['exact_match'] += 1
        
        # Tool match (correct tool name)
        if pred_tools == gt_tools:
            metrics['tool_match'] += 1
        elif pred_tools & gt_tools:  # Any overlap
            metrics['partial_match'] += 1
        
        # Drug precision/recall
        if pred_drugs:
            precision = len(pred_drugs & gt_drugs) / len(pred_drugs)
            metrics['drug_precision'].append(precision)
        
        if gt_drugs:
            recall = len(pred_drugs & gt_drugs) / len(gt_drugs)
            metrics['drug_recall'].append(recall)
        
        if pred_drugs and gt_drugs:
            precision = len(pred_drugs & gt_drugs) / len(pred_drugs)
            recall = len(pred_drugs & gt_drugs) / len(gt_drugs)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics['drug_f1'].append(f1)
    
    # Compute averages
    n = len(predictions)
    results = {
        'num_samples': n,
        'exact_match_accuracy': metrics['exact_match'] / n * 100,
        'tool_match_accuracy': metrics['tool_match'] / n * 100,
        'partial_match': metrics['partial_match'] / n * 100,
        'avg_drug_precision': sum(metrics['drug_precision']) / len(metrics['drug_precision']) * 100 if metrics['drug_precision'] else 0,
        'avg_drug_recall': sum(metrics['drug_recall']) / len(metrics['drug_recall']) * 100 if metrics['drug_recall'] else 0,
        'avg_drug_f1': sum(metrics['drug_f1']) / len(metrics['drug_f1']) * 100 if metrics['drug_f1'] else 0,
    }
    
    return results


def main():
    args = parse_args()
    
    print("="*60)
    print("LLaVA-Med Tool Selection Evaluation")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Model: {args.model_path}")
    print(f"Num test samples: {args.num_test}")
    print()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Load test data
    test_data = load_test_data(args.data_dir, args.test_file, args.num_test)
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    print("-"*60)
    
    predictions = []
    ground_truths = []
    detailed_results = []
    
    for i, sample in enumerate(test_data):
        question = sample['input']
        ground_truth = sample['output']
        
        # Generate prediction
        prediction = generate_prediction(model, tokenizer, question, args.max_new_tokens)
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        
        # Check if correct
        is_correct = prediction.strip().lower() == ground_truth.strip().lower()
        
        detailed_results.append({
            'index': i,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': is_correct
        })
        
        # Print progress
        status = "✓" if is_correct else "✗"
        print(f"[{i+1}/{len(test_data)}] {status}")
        print(f"  GT:   {ground_truth}")
        print(f"  Pred: {prediction}")
        print()
    
    # Compute metrics
    print("-"*60)
    print("RESULTS")
    print("-"*60)
    
    metrics = compute_metrics(predictions, ground_truths)
    
    print(f"Samples evaluated:     {metrics['num_samples']}")
    print(f"Exact Match Accuracy:  {metrics['exact_match_accuracy']:.1f}%")
    print(f"Tool Match Accuracy:   {metrics['tool_match_accuracy']:.1f}%")
    print(f"Partial Match:         {metrics['partial_match']:.1f}%")
    print(f"Avg Drug Precision:    {metrics['avg_drug_precision']:.1f}%")
    print(f"Avg Drug Recall:       {metrics['avg_drug_recall']:.1f}%")
    print(f"Avg Drug F1:           {metrics['avg_drug_f1']:.1f}%")
    
    # Save detailed results
    output = {
        'timestamp': str(datetime.now()),
        'model_path': args.model_path,
        'metrics': metrics,
        'detailed_results': detailed_results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output_file}")
    print(f"Completion time: {datetime.now()}")


if __name__ == "__main__":
    main()
