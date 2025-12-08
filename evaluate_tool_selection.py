#!/usr/bin/env python3
"""
Evaluate Tool Selection V3 Model on Test Data

Metrics:
- Exact Match: Predicted tool == Ground truth tool (first tool)
- Top-K Accuracy: Correct tool in top K predictions
- Precision/Recall: For multi-tool predictions
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import warnings
warnings.filterwarnings("ignore")

# Add llava to path
sys.path.insert(0, '/home/azureuser/localfiles/LLaVA-Med')

def parse_tools_from_output(output_text):
    """Extract tool names from model output."""
    tools = []
    for line in output_text.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            tool_name = line.split(':')[0].strip()
            if tool_name and not tool_name.startswith('#'):
                tools.append(tool_name)
    return tools

def load_test_data(test_dir):
    """Load all test data from directory."""
    all_samples = []
    for filename in os.listdir(test_dir):
        if filename.startswith('model1_tool_selection') and filename.endswith('.jsonl'):
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    all_samples.append(sample)
    return all_samples

def load_model(base_model_path, adapter_path, device="cuda"):
    """Load base model with LoRA adapter."""
    from llava.model import LlavaMistralForCausalLM
    
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model (LLaVA-Med)...")
    # Use 4-bit quantization for faster inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_prediction(model, tokenizer, input_text, max_new_tokens=512):
    """Generate model prediction for input."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=3584)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    with torch.no_grad():
        # Manual generation loop to avoid transformers version incompatibility
        # LLaVA-Med doesn't support newer transformers args like cache_position
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                use_cache=False,
            )
            
            # Get next token (greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    # Decode only the generated part
    input_length = input_ids.shape[1]
    generated_tokens = generated_ids[0][input_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return prediction.strip()

def evaluate(model, tokenizer, test_samples, output_file=None):
    """Run evaluation and compute metrics."""
    results = []
    
    # Metrics
    exact_match = 0
    first_tool_match = 0
    any_tool_match = 0
    total_gt_tools = 0
    total_pred_tools = 0
    correct_tools = 0
    
    # Per-tool stats
    tool_correct = Counter()
    tool_total = Counter()
    tool_predicted = Counter()
    
    print(f"\nEvaluating on {len(test_samples)} samples...")
    
    for sample in tqdm(test_samples):
        input_text = sample['input']
        ground_truth = sample['output']
        
        # Generate prediction
        prediction = generate_prediction(model, tokenizer, input_text)
        
        # Parse tools
        gt_tools = parse_tools_from_output(ground_truth)
        pred_tools = parse_tools_from_output(prediction)
        
        # Compute metrics
        gt_set = set(gt_tools)
        pred_set = set(pred_tools)
        
        # Exact match (all tools in order)
        if gt_tools == pred_tools:
            exact_match += 1
        
        # First tool match
        if gt_tools and pred_tools and gt_tools[0] == pred_tools[0]:
            first_tool_match += 1
        
        # Any tool match
        if gt_set & pred_set:
            any_tool_match += 1
        
        # Precision/Recall stats
        total_gt_tools += len(gt_set)
        total_pred_tools += len(pred_set)
        correct_tools += len(gt_set & pred_set)
        
        # Per-tool stats
        for tool in gt_tools:
            tool_total[tool] += 1
            if tool in pred_set:
                tool_correct[tool] += 1
        for tool in pred_tools:
            tool_predicted[tool] += 1
        
        results.append({
            'input': input_text[:200] + '...',
            'ground_truth': ground_truth,
            'prediction': prediction,
            'gt_tools': gt_tools,
            'pred_tools': pred_tools,
            'exact_match': gt_tools == pred_tools,
            'first_tool_match': gt_tools and pred_tools and gt_tools[0] == pred_tools[0],
        })
    
    # Compute final metrics
    n = len(test_samples)
    precision = correct_tools / total_pred_tools if total_pred_tools > 0 else 0
    recall = correct_tools / total_gt_tools if total_gt_tools > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'total_samples': n,
        'exact_match': exact_match,
        'exact_match_pct': 100 * exact_match / n,
        'first_tool_match': first_tool_match,
        'first_tool_match_pct': 100 * first_tool_match / n,
        'any_tool_match': any_tool_match,
        'any_tool_match_pct': 100 * any_tool_match / n,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_gt_tools': total_gt_tools,
        'total_pred_tools': total_pred_tools,
        'correct_tools': correct_tools,
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS - Tool Selection V3")
    print("="*60)
    print(f"Total samples: {n}")
    print(f"\n--- Accuracy Metrics ---")
    print(f"Exact Match:      {exact_match:4d} / {n} ({100*exact_match/n:.2f}%)")
    print(f"First Tool Match: {first_tool_match:4d} / {n} ({100*first_tool_match/n:.2f}%)")
    print(f"Any Tool Match:   {any_tool_match:4d} / {n} ({100*any_tool_match/n:.2f}%)")
    print(f"\n--- Precision/Recall ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Top predicted tools
    print(f"\n--- Top 10 Predicted Tools ---")
    for tool, count in tool_predicted.most_common(10):
        correct = tool_correct.get(tool, 0)
        total = tool_total.get(tool, 0)
        print(f"  {tool}: {count} predicted, {correct}/{total} correct")
    
    # Save results
    if output_file:
        output_data = {
            'metrics': metrics,
            'results': results[:100],  # Save first 100 for inspection
            'tool_stats': {
                'predicted': dict(tool_predicted.most_common(50)),
                'correct': dict(tool_correct.most_common(50)),
                'total': dict(tool_total.most_common(50)),
            }
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Tool Selection V3')
    parser.add_argument('--base-model', type=str, 
                       default='microsoft/llava-med-v1.5-mistral-7b',
                       help='Base model path')
    parser.add_argument('--adapter-path', type=str,
                       default='checkpoints/llava-med-tool-selection-v3/checkpoint-600',
                       help='LoRA adapter path')
    parser.add_argument('--test-dir', type=str,
                       default='datasets/finetuning/test',
                       help='Test data directory')
    parser.add_argument('--output-file', type=str,
                       default='evaluation_results/tool_selection_v3_eval.json',
                       help='Output file for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples to evaluate (for quick testing)')
    parser.add_argument('--gpu', type=int, default=1,
                       help='GPU to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    test_samples = load_test_data(args.test_dir)
    print(f"Loaded {len(test_samples)} test samples")
    
    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    
    # Run evaluation
    metrics, results = evaluate(model, tokenizer, test_samples, args.output_file)
    
    return metrics

if __name__ == '__main__':
    main()
