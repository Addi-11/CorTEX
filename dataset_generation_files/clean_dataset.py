"""
Post-processing script to clean TxAgent-generated datasets.
Run this after inference completes to create a clean training dataset.

Fixes:
1. Filters to only drug-related FDA tools
2. Removes meta-tools (agents, tool finders)
3. Ensures consistent output format: TOOL_NAME: drug1, drug2, ...
4. Removes samples with no valid tool calls
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


VALID_TOOLS = {
    # FDA Drug Tools
    'FDA_get_indications_by_drug_name',
    'FDA_get_contraindications_by_drug_name', 
    'FDA_get_dosage_info_by_drug_name',
    'FDA_get_warnings_and_cautions_by_drug_name',
    'FDA_get_adverse_reactions_by_drug_name',
    'FDA_get_drug_interactions_by_drug_name',
    'FDA_get_pregnancy_or_breastfeeding_info_by_drug_name',
    'FDA_get_pregnancy_effects_info_by_drug_name',
    'FDA_get_population_use_info_by_drug_name',
    'FDA_get_general_precautions_by_drug_name',
    'FDA_get_overdosage_info_by_drug_name',
    'FDA_get_clinical_pharmacology_by_drug_name',
    'FDA_get_mechanism_of_action_by_drug_name',
    'FDA_get_pharmacodynamics_by_drug_name',
    'FDA_get_pharmacokinetics_by_drug_name',
    'FDA_get_teratogenic_effects_by_drug_name',
    'FDA_get_boxed_warning_by_drug_name',
    'FDA_get_pediatric_use_info_by_drug_name',
    'FDA_get_geriatric_use_info_by_drug_name',
    # DrugBank
    'DrugBank_get_drug_interactions',
    'DrugBank_search_drugs',
    # RxNorm
    'RxNorm_get_drug_info',
    'RxNorm_get_interactions',
    # OpenFDA
    'OpenFDA_search_drug_events',
    'OpenFDA_get_drug_label',
}

# EXCLUDE (meta-tools, agents)
EXCLUDED_TOOLS = {
    'DiseaseAnalyzerAgent',
    'BiomarkerDiscoveryWorkflow', 
    'MedicalLiteratureReviewer',
    'LiteratureSynthesisAgent',
    'ClinicalTrialDesignAgent',
    'Tool_Finder_Keyword',
    'Tool_Finder_LLM',
    'ToolDiscover',
    'ToolDescriptionOptimizer',
    'Finish',
    'Tool_RAG',
}

def extract_drug_names_from_question(question):
    """Extract drug names from the Options line in the question."""
    match = re.search(r'Options?:\s*(.+?)(?:\n|$)', question, re.IGNORECASE)
    if match:
        options = match.group(1)
        drugs = [d.strip() for d in options.split('/')]
        return [d for d in drugs if d and len(d) > 1]
    return []

def is_drug_related_question(question):
    """Check if question is about drugs/medications."""
    drug_keywords = [
        'drug', 'medication', 'treatment', 'antibiotic', 'therapy',
        'dose', 'dosage', 'prescribe', 'administer', 'contraindicated',
        'pregnancy', 'pregnant', 'breastfeeding', 'side effect',
        'adverse', 'interaction', 'mechanism', 'pharmacology'
    ]
    question_lower = question.lower()
    
    has_keyword = any(kw in question_lower for kw in drug_keywords)
    
    drugs = extract_drug_names_from_question(question)
    has_drug_options = len(drugs) >= 3
    
    return has_keyword or has_drug_options

def parse_and_clean_output(output_str):
    """
    Parse the output and return cleaned format.
    Returns dict: {tool_name: [arg1, arg2, ...]}
    """
    tool_args = defaultdict(list)
    
    for line in output_str.strip().split('\n'):
        if ':' not in line:
            continue
            
        tool_name = line.split(':')[0].strip()
        args_part = ':'.join(line.split(':')[1:]).strip()
        
        if tool_name in EXCLUDED_TOOLS:
            continue

        if args_part.startswith('{'):
            drug_match = re.search(r"'drug_name':\s*'([^']+)'", args_part)
            if drug_match:
                tool_args[tool_name].append(drug_match.group(1))
            else:
                matches = re.findall(r"'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)'", args_part)
                for m in matches:
                    if len(m) > 2 and m not in ['True', 'False', 'None']:
                        tool_args[tool_name].append(m)
        else:
            args = [a.strip() for a in args_part.split(',')]
            for arg in args:
                if arg and len(arg) > 1:
                    tool_args[tool_name].append(arg)
    
    return dict(tool_args)

def format_clean_output(tool_args):
    """Format tool_args dict into clean output string."""
    lines = []
    for tool_name, args in tool_args.items():
        unique_args = list(dict.fromkeys(args))
        if unique_args:
            lines.append(f"{tool_name}: {', '.join(unique_args)}")
    return '\n'.join(lines)

def clean_dataset(input_file, output_file, mode='all'):
    """
    Clean the dataset.
    
    mode options:
    - 'all': Keep all samples with valid tool calls
    - 'fda_only': Only keep FDA tool calls
    - 'drug_questions': Only drug-related questions
    """
    print(f"Cleaning {input_file} -> {output_file}")
    print(f"Mode: {mode}")
    
    cleaned_samples = []
    stats = {
        'total': 0,
        'no_tools': 0,
        'excluded_tools_only': 0,
        'non_drug_question': 0,
        'kept': 0,
    }
    tool_counts = Counter()
    
    with open(input_file, 'r') as f:
        for line in f:
            stats['total'] += 1
            try:
                sample = json.loads(line.strip())
            except:
                continue
            
            question = sample.get('input', '')
            output = sample.get('output', '')
            ground_truth = sample.get('ground_truth_answer', '')

            tool_args = parse_and_clean_output(output)
            
            if mode == 'fda_only':
                tool_args = {k: v for k, v in tool_args.items() if k in VALID_TOOLS}
            
            if not tool_args:
                stats['no_tools'] += 1
                continue
            
            if mode == 'drug_questions' and not is_drug_related_question(question):
                stats['non_drug_question'] += 1
                continue
            
            clean_output = format_clean_output(tool_args)
            
            if not clean_output:
                stats['excluded_tools_only'] += 1
                continue
            
            for tool in tool_args.keys():
                tool_counts[tool] += 1
            
            cleaned_sample = {
                'instruction': "Given a medical question, identify the biomedical database tools and parameters needed to answer it. Format: TOOL_NAME: param1, param2, ...",
                'input': question,
                'output': clean_output,
                'ground_truth_answer': ground_truth
            }
            cleaned_samples.append(cleaned_sample)
            stats['kept'] += 1
    
    with open(output_file, 'w') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n=== Cleaning Stats ===")
    print(f"Total input samples: {stats['total']}")
    print(f"Removed (no valid tools): {stats['no_tools']}")
    print(f"Removed (excluded tools only): {stats['excluded_tools_only']}")
    if mode == 'drug_questions':
        print(f"Removed (non-drug questions): {stats['non_drug_question']}")
    print(f"Kept: {stats['kept']} ({100*stats['kept']/max(1,stats['total']):.1f}%)")
    
    print(f"\n=== Tool Distribution (top 15) ===")
    for tool, count in tool_counts.most_common(15):
        print(f"  {count:4d} | {tool}")
    print(f"\nUnique tools: {len(tool_counts)}")
    
    return cleaned_samples

def create_balanced_dataset(input_file, output_file, samples_per_tool=50):
    """Create a balanced dataset with similar representation of tools."""
    print(f"\nCreating balanced dataset...")
    
    samples_by_tool = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                output = sample.get('output', '')
                if '\n' in output:
                    primary_tool = output.split('\n')[0].split(':')[0].strip()
                elif ':' in output:
                    primary_tool = output.split(':')[0].strip()
                else:
                    continue
                samples_by_tool[primary_tool].append(sample)
            except:
                continue
    
    balanced = []
    for tool, samples in samples_by_tool.items():
        n = min(len(samples), samples_per_tool)
        balanced.extend(samples[:n])
    
    import random
    random.shuffle(balanced)
    
    with open(output_file, 'w') as f:
        for sample in balanced:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Balanced dataset: {len(balanced)} samples")
    return balanced


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean TxAgent dataset')
    parser.add_argument('--mode', choices=['all', 'fda_only', 'drug_questions'], 
                        default='all', help='Cleaning mode')
    parser.add_argument('--balance', action='store_true', 
                        help='Create balanced dataset')
    args = parser.parse_args()
    
    input_file = 'datasets/finetuning/model1_tool_selection.jsonl'
    
    print("="*60)
    print("MODE: all (keep all valid tools)")
    print("="*60)
    clean_dataset(
        input_file,
        'datasets/finetuning/model1_cleaned_all.jsonl',
        mode='all'
    )
    
    print("\n" + "="*60)
    print("MODE: fda_only (only FDA drug tools)")
    print("="*60)
    clean_dataset(
        input_file,
        'datasets/finetuning/model1_cleaned_fda.jsonl',
        mode='fda_only'
    )
    
    print("\n" + "="*60)
    print("MODE: drug_questions (drug-related questions only)")
    print("="*60)  
    clean_dataset(
        input_file,
        'datasets/finetuning/model1_cleaned_drug_questions.jsonl',
        mode='drug_questions'
    )
    
    if args.balance:
        print("\n" + "="*60)
        print("Creating balanced dataset")
        print("="*60)
        create_balanced_dataset(
            'datasets/finetuning/model1_cleaned_all.jsonl',
            'datasets/finetuning/model1_balanced.jsonl'
        )
    
    print("  - model1_cleaned_all.jsonl (all valid tools)")
    print("  - model1_cleaned_fda.jsonl (FDA tools only)")
    print("  - model1_cleaned_drug_questions.jsonl (drug questions)")
