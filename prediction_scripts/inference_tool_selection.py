#!/usr/bin/env python3
"""
Inference script for the trained LLaVA-Med Tool Selection model.
Uses the fine-tuned model to predict tools and arguments for medical questions.
"""

import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained tool selection model")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/llava-med-tool-selection/final",
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/llava-med-v1.5-mistral-7b",
                        help="Base model name")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to process")
    parser.add_argument("--input_file", type=str, default=None,
                        help="JSONL file with questions to process")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for predictions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature")
    return parser.parse_args()


def load_model(model_path, base_model):
    """Load the fine-tuned model."""
    print(f"Loading base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer


def create_prompt(question):
    """Create inference prompt."""
    instruction = "Identify the biomedical database tools and drug names needed to answer this medical question. Output format: TOOL_NAME: drug1, drug2, ..."
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{question}

### Response:
"""
    return prompt


def generate_tool_calls(model, tokenizer, question, max_new_tokens=256, temperature=0.1):
    """Generate tool calls for a question."""
    prompt = create_prompt(question)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in full_output:
        response = full_output.split("### Response:")[-1].strip()
    else:
        response = full_output
    
    return response


def parse_tool_output(output):
    """Parse the model output into structured tool calls."""
    tool_calls = []
    
    for line in output.strip().split('\n'):
        if ':' not in line:
            continue
        
        parts = line.split(':', 1)
        tool_name = parts[0].strip()
        args_str = parts[1].strip() if len(parts) > 1 else ""
        
        # Parse arguments (comma-separated)
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        
        for arg in args:
            tool_calls.append({
                'name': tool_name,
                'arguments': {'drug_name': arg, 'limit': 5}
            })
    
    return tool_calls


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.question:
        # Single question mode
        print("\n" + "="*60)
        print("Question:")
        print(args.question[:200] + "..." if len(args.question) > 200 else args.question)
        print("\nGenerated Tool Calls:")
        print("="*60)
        
        response = generate_tool_calls(
            model, tokenizer, args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(response)
        
        # Parse into structured format
        tool_calls = parse_tool_output(response)
        print("\nParsed Tool Calls:")
        for tc in tool_calls:
            print(f"  {tc}")
    
    elif args.input_file:
        # Batch mode
        print(f"Processing questions from: {args.input_file}")
        
        results = []
        with open(args.input_file, 'r') as f:
            questions = [json.loads(line) for line in f]
        
        for i, q in enumerate(questions):
            question = q.get('input', q.get('question', ''))
            
            response = generate_tool_calls(
                model, tokenizer, question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            tool_calls = parse_tool_output(response)
            
            results.append({
                'index': i,
                'question': question,
                'predicted_tools': response,
                'parsed_tool_calls': tool_calls,
                'ground_truth': q.get('output', '')
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(questions)} questions")
        
        # Save results
        output_file = args.output_file or "predictions.jsonl"
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        print(f"\nResults saved to: {output_file}")
    
    else:
        # Interactive mode
        print("\nInteractive mode. Enter questions (Ctrl+C to exit):")
        print("="*60)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if not question:
                    continue
                
                response = generate_tool_calls(
                    model, tokenizer, question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                
                print("\nTool Calls:")
                print(response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
