"""
TxAgent Inference Script for MedQA Dataset
Used for tool Calling Data
"""

import os
import json
import sys
import io
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_USE_V1"] = "0"

print(f"Starting TxAgent inference at {datetime.now()}")
print("="*80)


print("Loading TxAgent model...")
from txagent import TxAgent

model_name = 'mims-harvard/TxAgent-T1-Llama-3.1-8B'
rag_model_name = 'mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B'

agent = TxAgent(model_name, rag_model_name, enable_summary=True)
agent.init_model()
print("TxAgent loaded")


data_path = "datasets/MedInstQA/MedQa_train.json"
print(f"\nLoading dataset from {data_path}...")

data = []
with open(data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
print(f"Loaded {len(df)} samples")


def parse_tool_calls_from_output(output_text):
    """Parse tool calls and results from TxAgent's printed output."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_text = ansi_escape.sub('', output_text)
    
    tool_calls_with_results = []
    sections = clean_text.split('Tool Call:')
    
    for section in sections[1:]:
        try:
            name_match = re.search(r"'name':\s*'([^']+)'", section)
            args_match = re.search(r"'arguments':\s*(\{[^}]+\})", section)
            
            if not name_match:
                continue
                
            tool_name = name_match.group(1)
            
            if tool_name in ['Finish', 'Tool_RAG']:
                continue
            
            arguments = None
            if args_match:
                try:
                    args_str = args_match.group(1).replace("'", '"')
                    arguments = json.loads(args_str)
                except:
                    arguments = args_match.group(1)
            
            result_match = re.search(r'Tool Call Result:\s*(.+?)(?=Tool Call:|Summarized Tool Result:|$)', section, re.DOTALL)
            
            result = None
            if result_match:
                result_str = result_match.group(1).strip()
                if result_str.startswith('{') or result_str.startswith('['):
                    try:
                        result = json.loads(result_str.split('\n')[0])
                    except:
                        result = result_str[:1000] + '...' if len(result_str) > 1000 else result_str
                else:
                    result = result_str[:1000] + '...' if len(result_str) > 1000 else result_str
            
            tool_calls_with_results.append({
                'name': tool_name,
                'arguments': arguments,
                'result': result
            })
            
        except Exception as e:
            continue
    
    return tool_calls_with_results


print("\n" + "="*80)
print("STARTING INFERENCE")
print("="*80)

os.makedirs("results", exist_ok=True)
os.makedirs("datasets/finetuning", exist_ok=True)

multiagent = False
max_round = 20

results_file = "results/medqa_txagent_results.jsonl"
model1_file = "datasets/finetuning/model1_tool_selection.jsonl"
model2_file = "datasets/finetuning/model2_reasoning.jsonl"

for f in [results_file, model1_file, model2_file]:
    open(f, 'w').close()

print(f"📝 Writing results incrementally to:")
print(f"   - {results_file}")
print(f"   - {model1_file}")
print(f"   - {model2_file}")

def create_model1_sample(question, tool_calls, ground_truth):
    """Create Model 1 (tool selection) training sample."""
    tool_groups = defaultdict(list)
    for tc in tool_calls:
        name = tc.get('name', '')
        if name and name not in ['Finish', 'Tool_RAG']:
            args = tc.get('arguments', {})
            if isinstance(args, dict):
                drug = args.get('drug_name', args.get('name', ''))
                if drug:
                    tool_groups[name].append(drug)
                else:
                    tool_groups[name].append(str(args))
            else:
                tool_groups[name].append(str(args))
    
    if not tool_groups:
        return None
    
    output_lines = []
    for tool_name, args_list in tool_groups.items():
        unique_args = list(dict.fromkeys(args_list))
        output_lines.append(f"{tool_name}: {', '.join(unique_args)}")
    
    return {
        'instruction': "Identify the biomedical database tools and drug names needed to answer this medical question. Output format: TOOL_NAME: drug1, drug2, ...",
        'input': question,
        'output': "\n".join(output_lines),
        'ground_truth_answer': ground_truth
    }

def create_model2_sample(question, tool_calls, final_response, ground_truth):
    """Create Model 2 (reasoning) training sample."""
    tool_results_text = []
    for tc in tool_calls:
        name = tc.get('name', '')
        if name and name not in ['Finish', 'Tool_RAG']:
            args = tc.get('arguments', {})
            result = tc.get('result', '')
            
            if isinstance(args, dict):
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            else:
                args_str = str(args)
            
            if isinstance(result, dict):
                result = json.dumps(result)
            result_str = str(result)[:1500] if len(str(result)) > 1500 else str(result)
            
            tool_results_text.append(f"[{name}({args_str})]\n{result_str}")
    
    if not tool_results_text or not final_response:
        return None
    
    model2_input = f"QUESTION:\n{question}\n\nTOOL RESULTS:\n" + "\n".join(tool_results_text)
    
    return {
        'instruction': "Based on the medical question and tool results, select the best answer option and explain your reasoning.",
        'input': model2_input,
        'output': final_response,
        'ground_truth_answer': ground_truth
    }

completed = 0
errors = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running TxAgent"):
    query = row['input']
    ground_truth = row['output']
    
    try:
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        response = agent.run_multistep_agent(
            query,
            temperature=0.3,
            max_new_tokens=1024,
            max_token=90240,
            call_agent=multiagent,
            max_round=max_round
        )
        
        sys.stdout = sys.__stdout__
        output_text = captured_output.getvalue()
        
        tool_calls_with_results = parse_tool_calls_from_output(output_text)
        
        result = {
            'index': idx,
            'instruction': row.get('instruction', ''),
            'input': query,
            'output': ground_truth,
            'tool_calls': tool_calls_with_results,
            'final_response': response
        }
        
        with open(results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        # Model 1 (Tool Calling) sample
        if tool_calls_with_results:
            m1_sample = create_model1_sample(query, tool_calls_with_results, ground_truth)
            if m1_sample:
                with open(model1_file, 'a') as f:
                    f.write(json.dumps(m1_sample) + '\n')
            
            # Model 2 (Reasoning) sample
            m2_sample = create_model2_sample(query, tool_calls_with_results, response, ground_truth)
            if m2_sample:
                with open(model2_file, 'a') as f:
                    f.write(json.dumps(m2_sample) + '\n')
        
        completed += 1
        print(f"[{idx+1}/{len(df)}] Tool calls: {len(tool_calls_with_results)} | Completed: {completed}")
        
    except Exception as e:
        sys.stdout = sys.__stdout__
        

        error_result = {
            'index': idx,
            'instruction': row.get('instruction', ''),
            'input': query,
            'output': ground_truth,
            'tool_calls': None,
            'final_response': None,
            'error': str(e)
        }
        with open(results_file, 'a') as f:
            f.write(json.dumps(error_result) + '\n')
        
        errors += 1


print("COMPLETED")
print(f"Total samples: {len(df)}")
print(f"Completed: {completed}")
print(f"Errors: {errors}")
print(f"\nOutput files:")
print(f"   - {results_file}")
print(f"   - {model1_file}")
print(f"   - {model2_file}")


for f in [results_file, model1_file, model2_file]:
    with open(f, 'r') as file:
        lines = sum(1 for _ in file)
    print(f"   {f}: {lines} samples")