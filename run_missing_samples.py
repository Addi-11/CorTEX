#!/usr/bin/env python3
"""
TxAgent Inference Script - Process ONLY missing samples
This script identifies which samples haven't been generated yet and processes them.
"""

import os
import json
import sys
import io
import re
import glob
import argparse
from tqdm import tqdm
from datetime import datetime

def get_missing_indices(source_path, output_dir):
    """Find indices of samples that haven't been processed yet."""
    # Load source data
    source_data = []
    with open(source_path, 'r') as f:
        for line in f:
            source_data.append(json.loads(line.strip()))
    
    print(f"Total source samples: {len(source_data)}")
    
    # Load all generated data and extract inputs
    generated_inputs = set()
    for fpath in glob.glob(os.path.join(output_dir, 'model1_tool_selection*.jsonl')):
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    # Use first 200 chars of input as key
                    generated_inputs.add(d['input'][:200])
                except:
                    continue
    
    print(f"Total generated samples: {len(generated_inputs)}")
    
    # Find missing indices
    missing_indices = []
    for i, sample in enumerate(source_data):
        input_key = sample['input'][:200]
        if input_key not in generated_inputs:
            missing_indices.append(i)
    
    print(f"Missing samples: {len(missing_indices)}")
    return missing_indices, source_data


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
            
            tool_calls_with_results.append({
                'tool_name': tool_name,
                'arguments': arguments
            })
        except Exception as e:
            continue
    
    return tool_calls_with_results


def format_tool_calls_for_training(tool_calls):
    """Format tool calls into training output format."""
    if not tool_calls:
        return ""
    
    output_lines = []
    for tc in tool_calls:
        tool_name = tc['tool_name']
        args = tc['arguments']
        if args:
            if isinstance(args, dict):
                output_lines.append(f"{tool_name}: {args}")
            else:
                output_lines.append(f"{tool_name}: {args}")
        else:
            output_lines.append(f"{tool_name}: None")
    
    return '\n'.join(output_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True, help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of samples to process')
    parser.add_argument('--start_offset', type=int, default=0, help='Start from this offset in missing indices')
    args = parser.parse_args()
    
    GPU_ID = args.gpu
    BATCH_SIZE = args.batch_size
    START_OFFSET = args.start_offset
    
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["VLLM_USE_V1"] = "0"
    
    print(f"="*80)
    print(f"TxAgent Missing Samples Processing - GPU {GPU_ID}")
    print(f"Start time: {datetime.now()}")
    print(f"="*80)
    
    # Get missing indices
    source_path = "datasets/MedInstQA/MedQa_train.json"
    output_dir = "datasets/finetuning"
    
    missing_indices, source_data = get_missing_indices(source_path, output_dir)
    
    if not missing_indices:
        print("No missing samples! All done.")
        return
    
    # Select batch to process
    batch_indices = missing_indices[START_OFFSET:START_OFFSET + BATCH_SIZE]
    print(f"\nProcessing {len(batch_indices)} samples (offset {START_OFFSET} to {START_OFFSET + len(batch_indices)})")
    print(f"Sample indices: {batch_indices[0]} to {batch_indices[-1]}")
    
    # Load TxAgent
    print("\nLoading TxAgent model...")
    from txagent import TxAgent
    
    model_name = 'mims-harvard/TxAgent-T1-Llama-3.1-8B'
    rag_model_name = 'mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B'
    
    agent = TxAgent(model_name, rag_model_name, enable_summary=True)
    agent.init_model()
    print("✅ TxAgent loaded")
    
    # Output file
    output_file = f"{output_dir}/model1_tool_selection_missing_gpu{GPU_ID}.jsonl"
    print(f"\nOutput file: {output_file}")
    
    # Process samples
    instruction = "Identify the biomedical database tools and drug names needed to answer this medical question. Output format: TOOL_NAME: drug1, drug2, ..."
    
    processed = 0
    failed = 0
    
    with open(output_file, 'a') as outf:
        for idx in tqdm(batch_indices, desc=f"GPU {GPU_ID}"):
            sample = source_data[idx]
            question = sample['input']
            
            try:
                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                # Run TxAgent - use run_multistep_agent (not run_gradio_chat)
                response = agent.run_multistep_agent(
                    question,
                    temperature=0.3,
                    max_new_tokens=1024,
                    max_token=90240,
                    call_agent=True,
                    max_round=5
                )
                
                # Get captured output
                sys.stdout = old_stdout
                output_text = captured_output.getvalue()
                
                # Parse tool calls
                tool_calls = parse_tool_calls_from_output(output_text)
                
                if tool_calls:
                    output_str = format_tool_calls_for_training(tool_calls)
                    
                    training_sample = {
                        "instruction": instruction,
                        "input": question,
                        "output": output_str,
                        "ground_truth_answer": sample.get('output', ''),
                        "source_idx": idx
                    }
                    
                    outf.write(json.dumps(training_sample) + '\n')
                    outf.flush()
                    processed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                # Make sure stdout is restored
                sys.stdout = sys.__stdout__
                print(f"Error processing index {idx}: {e}")
                failed += 1
                continue
    
    print(f"\n{'='*80}")
    print(f"Completed at {datetime.now()}")
    print(f"Processed: {processed}, Failed: {failed}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
