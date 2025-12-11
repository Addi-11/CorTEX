"""
Evaluation script for M1 (Tool Calling) and M2 (Reasoning) models.

Usage:
    python evaluate_M1_M2.py \
        --base_model_path /path/to/base/model \
        --test_file /path/to/test.json \
        --M1_lora_weights_path /path/to/M1/lora (optional) \
        --M2_lora_weights_path /path/to/M2/lora (optional)
"""

import argparse
import json
import sys
import torch
from typing import Dict, Any, Optional
from peft import PeftModel

# Add LLaVA-Med to path
sys.path.insert(0, '/mnt/workspace/LLaVA-Med')

from llava.model import LlavaMistralForCausalLM
from transformers import AutoTokenizer

# Add TxAgent to path for ToolUniverse SDK
sys.path.insert(0, '/mnt/workspace/TxAgent')


# System prompt for M1 (Tool Calling)
M1_SYSTEM_PROMPT = """You are a medical tool selector. Given a question, output ONLY the tool name and arguments to call. Do NOT answer the question directly.

Format:
ToolName: {"param": "value"}

Available Tools:
- FDA_get_mechanism_of_action_by_drug_name: {"drug_name": "X"} - HOW a drug works
- FDA_get_indications_by_drug_name: {"drug_name": "X"} - WHAT a drug treats
- FDA_get_adverse_reactions_by_drug_name: {"drug_name": "X"} - SIDE EFFECTS
- FDA_get_drug_interactions_by_drug_name: {"drug_name": "X"} - drug INTERACTIONS
- FDA_get_contraindications_by_drug_name: {"drug_name": "X"} - when NOT to use
- FDA_get_drug_names_by_indication: {"indication": "X", "limit": 5} - find drugs for a condition
- FDA_get_boxed_warning_info_by_drug_name: {"drug_name": "X"} - BLACK BOX warnings
- FDA_get_pregnancy_or_breastfeeding_info_by_drug_name: {"drug_name": "X"} - pregnancy/breastfeeding info
- OpenTargets_get_disease_ids_by_name: {"disease_name": "X"} - get disease IDs
- OpenTargets_get_associated_drugs_by_disease_efoId: {"efo_id": "X"} - drugs for disease by ID
- DiseaseAnalyzerAgent: {"disease_name": "X"} - analyze disease info
- TRIP_Database_Guidelines_Search: {"query": "X"} - search clinical guidelines
- PubMed_search_articles: {"query": "X", "max_results": 5} - search PubMed
- FAERS_search_adverse_event_reports: {"drug_name": "X"} - search FDA adverse events

Question: {question}

Tool Call:"""


# System prompt for M2 (Reasoning)
M2_SYSTEM_PROMPT = """You are a medical assistant. Answer the question using the provided tool result.

Question: {question}

Tool Called: {tool_name}
Tool Result: {tool_result}

Answer:"""


def load_model(base_model_path: str, lora_path: Optional[str] = None, device: str = "cuda"):
    """Load base model with optional LoRA weights"""
    
    print(f"Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    
    if lora_path:
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_tool_call(output: str) -> tuple[Optional[str], Optional[Dict]]:
    """Parse tool name and params from M1 output"""
    lines = output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue
        
        parts = line.split(':', 1)
        tool_name = parts[0].strip()
        params_str = parts[1].strip() if len(parts) > 1 else ""
        
        # Try to parse params as JSON
        try:
            params_str = params_str.replace("'", '"')
            params = json.loads(params_str)
            return tool_name, params
        except:
            continue
    
    return None, None


def call_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """Call tool using ToolUniverse SDK"""
    try:
        from tooluniverse import ToolUniverse
        tu = ToolUniverse(tool_files=["/mnt/workspace/TxAgent/data/new_tool.json"])
        tool = tu.get_tool_by_name(tool_name)
        if tool:
            result = tool.run(**params)
            return str(result)
    except Exception as e:
        print(f"ToolUniverse SDK error: {e}")
    
    # Fallback mock response
    return f"Tool {tool_name} called with {params}"


def evaluate_answer(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Simple keyword-based evaluation"""
    pred_lower = prediction.lower()
    truth_lower = ground_truth.lower()
    
    # Extract keywords (words longer than 3 chars)
    keywords = [w for w in truth_lower.split() if len(w) > 3]
    if not keywords:
        return {"match": False, "score": 0.0}
    
    matches = sum(1 for kw in keywords if kw in pred_lower)
    score = matches / len(keywords)
    
    return {"match": score > 0.4, "score": score}


def run_evaluation(
    base_model_path: str,
    test_file: str,
    M1_lora_path: Optional[str] = None,
    M2_lora_path: Optional[str] = None,
    device: str = "cuda"
):
    """Run evaluation pipeline"""
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Load M1 model (Tool Calling)
    print("\n=== Loading M1 (Tool Calling) ===")
    M1_model, M1_tokenizer = load_model(base_model_path, M1_lora_path, device)
    
    # Load M2 model (Reasoning)
    # If M2 lora is different from M1, load separately. Otherwise reuse base.
    if M2_lora_path and M2_lora_path != M1_lora_path:
        print("\n=== Loading M2 (Reasoning) ===")
        M2_model, M2_tokenizer = load_model(base_model_path, M2_lora_path, device)
    elif M2_lora_path is None and M1_lora_path is None:
        # Both use base model
        print("\n=== Using same base model for M2 ===")
        M2_model, M2_tokenizer = M1_model, M1_tokenizer
    else:
        print("\n=== Loading M2 (Reasoning) ===")
        M2_model, M2_tokenizer = load_model(base_model_path, M2_lora_path, device)
    
    # Run evaluation
    results = []
    correct = 0
    
    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(test_data)} samples")
    print(f"{'='*60}\n")
    
    for i, sample in enumerate(test_data):
        question = sample["input"]
        ground_truth = sample["output"]
        
        print(f"\n--- Sample {i+1}/{len(test_data)} ---")
        print(f"Q: {question[:80]}...")
        
        # Step 1: M1 generates tool call
        M1_prompt = M1_SYSTEM_PROMPT.format(question=question)
        M1_output = generate(M1_model, M1_tokenizer, M1_prompt, max_new_tokens=100)
        print(f"M1 Output: {M1_output}")
        
        tool_name, tool_params = parse_tool_call(M1_output)
        
        if tool_name and tool_params:
            print(f"Tool: {tool_name} | Params: {tool_params}")
            
            # Step 2: Call tool
            tool_result = call_tool(tool_name, tool_params)
            print(f"Tool Result: {tool_result[:100]}...")
            
            # Step 3: M2 generates final answer
            M2_prompt = M2_SYSTEM_PROMPT.format(
                question=question,
                tool_name=tool_name,
                tool_result=tool_result
            )
            M2_output = generate(M2_model, M2_tokenizer, M2_prompt, max_new_tokens=200)
        else:
            print("No valid tool call found, using M1 output as answer")
            tool_result = None
            M2_output = M1_output
        
        print(f"M2 Answer: {M2_output[:100]}...")
        
        # Step 4: Evaluate
        eval_result = evaluate_answer(M2_output, ground_truth)
        if eval_result["match"]:
            correct += 1
            print(f"✓ Match (score: {eval_result['score']:.2f})")
        else:
            print(f"✗ No match (score: {eval_result['score']:.2f})")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "tool_result": tool_result,
            "prediction": M2_output,
            "evaluation": eval_result
        })
    
    # Summary
    accuracy = correct / len(test_data) if test_data else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{len(test_data)} correct ({accuracy:.1%})")
    print(f"{'='*60}")
    
    return results, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate M1 (Tool) and M2 (Reasoning) models")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model weights")
    parser.add_argument("--M1_lora_weights_path", type=str, default=None, help="Path to M1 LoRA weights (optional)")
    parser.add_argument("--M2_lora_weights_path", type=str, default=None, help="Path to M2 LoRA weights (optional)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    results, accuracy = run_evaluation(
        base_model_path=args.base_model_path,
        test_file=args.test_file,
        M1_lora_path=args.M1_lora_weights_path,
        M2_lora_path=args.M2_lora_weights_path,
        device=args.device
    )
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
