import json
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from tqdm import tqdm
from collections import defaultdict


def load_model(base_model_path, lora_model_path):
    """Load the fine-tuned model with LoRA adapter."""
    from transformers import AutoTokenizer
    from peft import PeftModel
    from llava.model import LlavaMistralForCausalLM
    
    print(f"Loading base model: {base_model_path}")
    
    # Load tokenizer from adapter path
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base LLaVA-Med model
    model = LlavaMistralForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    print(f"Loading LoRA adapter from: {lora_model_path}")
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.eval()
    
    return model, tokenizer


# Load model and tokenizer
base_model_path = "/mnt/workspace/CorTEX/.models/llava-med-v1.5-mistral-7b"  # Update this
lora_model_path = "/mnt/workspace/CorTEX/checkpoints/llava-med-tool-selection-v3/checkpoint-600"  # Update this

model_name = 'llava-med-v1.5-mistral-7b'
tokenizer, model = load_model(base_model_path, lora_model_path)

TOOL_DEFINITIONS = """
1) DiseaseAnalyzerAgent  
Description: Analyze diseases, symptoms, and diagnostic criteria for medical conditions.
Parameters: context (str) - Clinical context; disease_name (str) - Name of disease to analyze

2) FDA_get_mechanism_of_action_by_drug_name
Description: Get mechanism of action from FDA drug labels.
Parameters: drug_name (str); limit (int)

3) FDA_get_drug_names_by_indication
Description: Find drugs approved for a specific indication.
Parameters: indication (str); limit (int)

4) FDA_get_drug_interactions_by_drug_name
Description: Get drug interaction information from FDA labels.
Parameters: drug_name (str); limit (int)

5) FDA_get_adverse_reactions_by_drug_name
Description: Get adverse reactions listed in FDA drug labels.
Parameters: drug_name (str); limit (int)

6) FDA_get_overdosage_info_by_drug_name
Description: Get overdose information from FDA labels.
Parameters: drug_name (str); limit (int)

7) WikiPathways_search
Description: Search WikiPathways for biological pathways.
Parameters: query (str); organism (str); limit (int)

8) euhealthinfo_search_causes_of_death
Description: Search EU health statistics on causes of death.
Parameters: country (str); language (str); term_override (str); method (str); limit (int)

9) euhealthinfo_search_mental_health
Description: Search EU mental health data.
Parameters: country (str); condition (str); limit (int)
"""

def parse_output_format(output_str):
    """
    Parse output string into list of (tool_name, arguments) tuples
    Format: TOOL_NAME: arg1, arg2, ...
            TOOL_NAME: arg1, arg2, ...
    
    Returns: [(tool_name, [args]), ...]
    """
    tool_calls = []
    lines = output_str.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ':' in line:
            parts = line.split(':', 1)
            tool_name = parts[0].strip()
            args_str = parts[1].strip()
            
            # Parse arguments (comma-separated)
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            
            tool_calls.append((tool_name, args))
    
    return tool_calls

def evaluate_tool_and_args(dataset_path):
    """
    Evaluate tool selection AND argument correctness
    
    Expected dataset format (JSONL):
    {
        "input": "Question: ...",
        "output": "FDA_get_mechanism_of_action_by_drug_name: metoprolol\nFDA_get_clinical_pharmacology_by_drug_name: metoprolol"
    }
    """
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    total = len(dataset)
    exact_match = 0  # Both tool and all args correct
    tool_correct = 0  # Tool selected but args may differ
    tool_recall = 0  # Correct tools found (regardless of extra tools)
    
    results = []
    error_analysis = defaultdict(int)
    
    print(f"Evaluating on {total} samples...")
    
    for sample in tqdm(dataset):
        user_input = sample.get("input", "")
        expected_output = sample.get("output", "")
        
        # Parse expected output
        expected_calls = parse_output_format(expected_output)
        expected_tools = set([call[0] for call in expected_calls])
        expected_dict = {call[0]: set(call[1]) for call in expected_calls}  # tool -> args set
        
        system_prompt = f"""SYSTEM: You MUST output ONLY tool calls. Nothing else. No explanations, no reasoning, no text.

TOOLS:
{TOOL_DEFINITIONS}

STRICT OUTPUT RULES:
1. Output format: ToolName: arg1, arg2
2. One tool per line
3. NO extra text before or after
4. NO explanations
5. NO "ARGUMENTS TO EXTRACT:" 
6. NO "OUTPUT:"
7. ONLY tool calls
8. MULTIPLE TOOLS: If multiple tools are needed, output each on a separate line

EXAMPLES (EXACT FORMAT TO FOLLOW):

Input: What is the mechanism of action for metoprolol?
Output:
FDA_get_mechanism_of_action_by_drug_name: metoprolol

Input: Find FDA approved drugs for hypertension
Output:
FDA_get_drug_names_by_indication: hypertension

Input: What are the side effects and interactions of aspirin?
Output:
FDA_get_adverse_reactions_by_drug_name: aspirin
FDA_get_drug_interactions_by_drug_name: aspirin

Input: Analyze symptoms of diabetes
Output:
DiseaseAnalyzerAgent: diabetes

Input: What are the side effects, interactions, and clinical pharmacology of metoprolol?
Output:
FDA_get_adverse_reactions_by_drug_name: metoprolol
FDA_get_drug_interactions_by_drug_name: metoprolol
FDA_get_clinical_pharmacology_by_drug_name: metoprolol

Remember: ONLY output the tool calls. Nothing else. Start directly with the tool name."""
        
        prompt = f"{system_prompt}\n\nInput: {user_input}\nOutput:"
        
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        input_ids = input_ids.to(device=model.device, dtype=torch.long)
    
        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids,
                images = None,
                max_new_tokens=2000,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                use_cache=True,
                num_beams=1  # Reduce to greedy search for less memory
            )
                
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response.strip()
        print(f"Response: {response}")
        # Parse predicted output
        predicted_calls = parse_output_format(response)
        predicted_tools = set([call[0] for call in predicted_calls])
        predicted_dict = {call[0]: set(call[1]) for call in predicted_calls}
        
        # Evaluation metrics
        is_exact_match = expected_calls == predicted_calls
        if is_exact_match:
            exact_match += 1
        
        # Tool-level correctness
        tools_match = expected_tools == predicted_tools
        if tools_match:
            tool_correct += 1
            # Check if args also match
            args_match = all(
                expected_dict.get(tool) == predicted_dict.get(tool)
                for tool in expected_tools
            )
            if not args_match:
                error_analysis["tool_correct_args_wrong"] += 1
        else:
            # Some tools correct but not all
            correct_tool_count = len(expected_tools & predicted_tools)
            missing_tools = expected_tools - predicted_tools
            extra_tools = predicted_tools - expected_tools
            
            if correct_tool_count > 0:
                tool_recall += 1
                error_analysis["partial_tools"] += 1
            
            if missing_tools:
                error_analysis["missing_tools"] += 1
            if extra_tools:
                error_analysis["extra_tools"] += 1
        
        # Detailed result
        result = {
            "input_summary": user_input[:100] + "...",
            "expected": [(t, list(a)) for t, a in expected_calls],
            "predicted": [(t, a) for t, a in predicted_calls],
            "exact_match": is_exact_match,
            "tools_correct": tools_match,
            "expected_tools": list(expected_tools),
            "predicted_tools": list(predicted_tools),
            "response": response
        }
        results.append(result)
    
    # Calculate metrics
    exact_match_rate = (exact_match / total) * 100
    tool_match_rate = (tool_correct / total) * 100
    
    print(f"\n{'='*70}")
    print(f"Evaluation Results:")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Exact Match (Tool + All Args): {exact_match}/{total} ({exact_match_rate:.2f}%)")
    print(f"Tool Set Correct: {tool_correct}/{total} ({tool_match_rate:.2f}%)")
    print(f"\nError Analysis:")
    for error_type, count in sorted(error_analysis.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {error_type}: {count}")
    print(f"{'='*70}\n")
    
    # Show sample errors
    incorrect = [r for r in results if not r["exact_match"]]
    if incorrect:
        print(f"Sample Errors (first 3):\n")
        for i, error in enumerate(incorrect[:9]):
            print(f"{i+1}. Query: {error['input_summary']}")
            print(f"   Expected: {error['expected']}")
            print(f"   Got Response: {error['response']}")
            print()
    
    return {
        "exact_match_rate": exact_match_rate,
        "tool_match_rate": tool_match_rate,
        "total": total,
        "exact_matches": exact_match,
        "tool_matches": tool_correct,
        "error_analysis": dict(error_analysis),
        "results": results
    }

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    dataset_path = "/mnt/workspace/CorTEX/datasets/finetuning/cleaned/model1_tool_selection_val_filtered_9.jsonl"  # Your dataset path
    eval_results = evaluate_tool_and_args(dataset_path)
    