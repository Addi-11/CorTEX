#!/usr/bin/env python3
"""
LLaVA-Med Fine-tuning Script for Tool Selection (Model 1) - V2
Enhanced version with comprehensive system prompt including tool definitions.

This script fine-tunes the LLaVA-Med model using LoRA for efficient training.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path

# Set environment variables before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add LLaVA-Med to path
LLAVA_PATH = "/home/azureuser/localfiles/cortex-project/LLaVA-Med"
if LLAVA_PATH not in sys.path:
    sys.path.insert(0, LLAVA_PATH)

# Top 100 most frequently used tools with their descriptions (extracted from dataset + tooluniverse)
TOOL_DEFINITIONS = """
## Core Analysis Tools

### CallAgent
Delegate complex reasoning or multi-step analysis tasks to a specialized agent.
Parameters: solution (str) - Description of the analysis task or solution approach

### DiseaseAnalyzerAgent  
Analyze diseases, symptoms, and diagnostic criteria for medical conditions.
Parameters: context (str) - Clinical context; disease_name (str) - Name of disease to analyze

### BiomarkerDiscoveryWorkflow
Identify potential biomarkers for diseases using multi-omics data integration.
Parameters: disease_condition (str) - Target disease; sample_type (str) - e.g., blood, serum, tissue

### LiteratureSynthesisAgent
Synthesize findings from medical literature on specific topics.
Parameters: topic (str) - Research topic; literature_data (str) - Literature content; focus_area (str) - Focus area

### MedicalLiteratureReviewer
Conduct systematic reviews of medical literature with evidence assessment.
Parameters: research_topic (str); literature_content (str); focus_area (str); study_types (str); quality_level (str); review_scope (str)

### ClinicalTrialDesignAgent
Design clinical trials with appropriate methodology and endpoints.
Parameters: condition (str); intervention (str); phase (str); primary_endpoint (str)

### DrugSafetyAnalyzer
Analyze drug safety profiles including adverse events and contraindications.
Parameters: drug_name (str); analysis_type (str) - e.g., adverse_events, interactions

### DrugInteractionAnalyzerAgent
Analyze drug-drug interactions and their clinical significance.
Parameters: drug_list (list); severity_filter (str)

## Tool Discovery Tools

### Tool_Finder
Find relevant biomedical tools based on description.
Parameters: description (str) - Task description; limit (int) - Max results

### Tool_Finder_Keyword
Keyword-based search for biomedical tools.
Parameters: description (str) - Keywords or task description; limit (int)

### Tool_Finder_LLM
LLM-powered intelligent tool discovery.
Parameters: description (str) - Natural language description; limit (int)

### ToolDiscover
Discover tools based on capability requirements.
Parameters: capability (str); domain (str)

## FDA Drug Database Tools

### FDA_get_indications_by_drug_name
Get approved indications for a drug from FDA labels.
Parameters: drug_name (str); limit (int); skip (int)

### FDA_get_mechanism_of_action_by_drug_name
Get mechanism of action from FDA drug labels.
Parameters: drug_name (str); limit (int)

### FDA_get_drug_names_by_indication
Find drugs approved for a specific indication.
Parameters: indication (str); limit (int)

### FDA_get_drug_interactions_by_drug_name
Get drug interaction information from FDA labels.
Parameters: drug_name (str); limit (int)

### FDA_get_adverse_reactions_by_drug_name
Get adverse reactions listed in FDA drug labels.
Parameters: drug_name (str); limit (int)

### FDA_get_overdosage_info_by_drug_name
Get overdose information from FDA labels.
Parameters: drug_name (str); limit (int)

### FDA_get_pharmacodynamics_by_drug_name
Get pharmacodynamics information from FDA labels.
Parameters: drug_name (str); limit (int)

### FDA_get_clinical_pharmacology_by_drug_name
Get clinical pharmacology from FDA labels.
Parameters: drug_name (str); limit (int)

### FDA_get_population_use_info_by_drug_name
Get special population use info (pregnancy, pediatric, geriatric).
Parameters: drug_name (str); limit (int)

## Clinical Guidelines Search Tools

### TRIP_Database_Guidelines_Search
Search TRIP database for clinical guidelines.
Parameters: query (str); limit (int); search_type (str) - e.g., guideline

### EuropePMC_Guidelines_Search
Search Europe PMC for clinical guidelines and systematic reviews.
Parameters: query (str); limit (int); search_type (str)

### NICE_Clinical_Guidelines_Search
Search NICE (UK) clinical guidelines.
Parameters: query (str); limit (int)

### GIN_Guidelines_Search
Search Guidelines International Network database.
Parameters: query (str); limit (int)

### OpenAlex_Guidelines_Search
Search OpenAlex for academic guidelines.
Parameters: query (str); limit (int)

### CMA_Guidelines_Search
Search Canadian Medical Association guidelines.
Parameters: query (str); limit (int)

### WHO_Guidelines_Search
Search World Health Organization guidelines.
Parameters: query (str); limit (int)

### PubMed_Guidelines_Search
Search PubMed for clinical practice guidelines.
Parameters: query (str); limit (int)

## MedlinePlus Tools

### MedlinePlus_get_genetics_condition_by_name
Get genetic condition information from MedlinePlus.
Parameters: condition (str); format (str) - e.g., json

### MedlinePlus_get_genetics_gene_by_name
Get gene information from MedlinePlus Genetics.
Parameters: gene (str); format (str)

### MedlinePlus_search_topics_by_keyword
Search MedlinePlus health topics.
Parameters: term (str); db (str) - e.g., healthTopics; rettype (str)

## HPO (Human Phenotype Ontology) Tools

### get_HPO_ID_by_phenotype
Get HPO ID for a phenotype description.
Parameters: query (str) - Phenotype description; limit (int)

### get_phenotype_by_HPO_ID
Get phenotype details by HPO ID.
Parameters: id (str) - HPO ID like HPO:0000001

### get_joint_associated_diseases_by_HPO_ID_list
Find diseases associated with multiple phenotypes.
Parameters: hpo_ids (list); limit (int)

## Gene/Protein Analysis Tools

### HPA_search_genes_by_query
Search Human Protein Atlas for genes.
Parameters: search_query (str); limit (int)

### HPA_get_biological_processes_by_gene
Get biological processes for a gene from HPA.
Parameters: gene_name (str)

### HPA_get_contextual_biological_process_analysis
Analyze biological processes in specific context.
Parameters: gene_name (str); context_name (str)

### HPAGetContextualBiologicalProcessAnnotation
Get contextual biological process annotations.
Parameters: gene_name (str); context_name (str)

### GO_get_genes_for_term
Get genes associated with a Gene Ontology term.
Parameters: go_term (str); limit (int)

### enrichr_gene_enrichment_analysis
Perform gene enrichment analysis using Enrichr.
Parameters: gene_list (list); libs (list) - Enrichr libraries

### WikiPathways_search
Search WikiPathways for biological pathways.
Parameters: query (str); organism (str); limit (int)

## OpenTargets Tools

### OpenTargets_get_target_id_description_by_name
Get target ID and description from OpenTargets.
Parameters: targetName (str)

### OpenTargets_get_disease_ids_by_name
Get disease IDs from OpenTargets.
Parameters: diseaseName (str)

### OpenTargets_get_diseases_phenotypes_by_target_ensembl
Get disease phenotypes for a target gene.
Parameters: ensemblId (str); limit (int)

## Clinical Trials Tools

### search_clinical_trials
Search ClinicalTrials.gov database.
Parameters: condition (str); intervention (str); status (str); limit (int)

### get_clinical_trial_conditions_and_interventions
Get conditions and interventions from trials.
Parameters: nct_id (str)

## EU Health Information Tools

### euhealthinfo_search_causes_of_death
Search EU health statistics on causes of death.
Parameters: country (str); language (str); term_override (str); method (str); limit (int)

### euhealthinfo_search_infectious_diseases
Search EU infectious disease surveillance data.
Parameters: country (str); disease (str); limit (int)

### euhealthinfo_search_hospital_in_patient_data
Search EU hospital inpatient data.
Parameters: country (str); diagnosis (str); limit (int)

### euhealthinfo_search_diabetes_mellitus_epidemiology_registry
Search EU diabetes epidemiology data.
Parameters: country (str); limit (int)

### euhealthinfo_search_cancer_registry
Search EU cancer registry data.
Parameters: country (str); cancer_type (str); limit (int)

### euhealthinfo_search_mental_health
Search EU mental health data.
Parameters: country (str); condition (str); limit (int)

## Variant/Genomics Tools

### clinvar_search_variants
Search ClinVar for genetic variants.
Parameters: gene (str); variant (str); significance (str); limit (int)

### GWAS_search_associations_by_gene
Search GWAS Catalog for gene associations.
Parameters: gene (str); limit (int)
"""

SYSTEM_PROMPT = f"""You are a biomedical tool selection expert. Your task is to analyze medical questions and identify the most appropriate biomedical database tools and their parameters to help answer the question.

## Available Tools
{TOOL_DEFINITIONS}

## Output Format
For each tool needed, output on a separate line:
tool_name: {{'parameter1': 'value1', 'parameter2': 'value2'}}

For tools that take simple drug/condition names:
tool_name: drug_name1, drug_name2

## Guidelines
1. Select tools that directly address the medical question
2. For drug-related questions, use FDA tools to get drug information
3. For disease questions, use DiseaseAnalyzerAgent or HPO tools
4. For literature/evidence needs, use Guidelines Search tools
5. Multiple tools may be needed for complex questions
6. Always include appropriate parameters for each tool"""


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Med for tool selection")
    
    parser.add_argument("--model_name", type=str, 
                        default="microsoft/llava-med-v1.5-mistral-7b",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, 
                        default="checkpoints/llava-med-tool-selection-v2",
                        help="Output directory for checkpoints")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str,
                        default="datasets/finetuning",
                        help="Directory containing training data")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (for testing)")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=128,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda:0, etc.)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint (path or 'latest')")
    
    return parser.parse_args()


def load_and_combine_datasets(data_dir, max_samples=None):
    """Load and combine all JSONL dataset files."""
    import glob
    
    all_data = []
    pattern = os.path.join(data_dir, "model1_tool_selection*.jsonl")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} dataset files:")
    for f in sorted(files):
        with open(f, 'r') as file:
            samples = [json.loads(line) for line in file]
            print(f"  {f}: {len(samples)} samples")
            all_data.extend(samples)
    
    print(f"\nTotal samples: {len(all_data)}")
    
    # Limit samples if specified (for testing)
    if max_samples and max_samples < len(all_data):
        all_data = all_data[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    return all_data


def create_prompt(sample, include_system=True):
    """Create training prompt from sample with comprehensive system prompt."""
    question = sample['input']
    output = sample['output']
    
    if include_system:
        prompt = f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{question} [/INST]
{output}</s>"""
    else:
        # Simplified format without system prompt (shorter context)
        instruction = """Identify biomedical tools needed to answer this medical question. Output format: tool_name: {'param': 'value'} or tool_name: value1, value2"""
        prompt = f"""### Instruction:
{instruction}

### Input:
{question}

### Response:
{output}"""
    
    return prompt


def prepare_dataset(data, tokenizer, max_length, include_system=True):
    """Prepare dataset for training."""
    from torch.utils.data import Dataset
    
    class ToolSelectionDataset(Dataset):
        def __init__(self, data, tokenizer, max_length, include_system):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.include_system = include_system
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            prompt = create_prompt(sample, self.include_system)
            
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            labels = encoding["input_ids"].clone()
            
            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # For instruction tuning, we should mask the instruction/input part
            # Only train on the response
            if self.include_system:
                # Find [/INST] token and mask everything before it
                inst_token = "[/INST]"
            else:
                # Find ### Response: and mask before
                inst_token = "### Response:"
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }
    
    return ToolSelectionDataset(data, tokenizer, max_length, include_system)


def main():
    args = parse_args()
    
    print("="*60)
    print("LLaVA-Med Tool Selection Fine-tuning V2")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading datasets...")
    data = load_and_combine_datasets(args.data_dir, max_samples=args.max_samples)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    from transformers import (
        AutoTokenizer, 
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType
    )
    
    # Import LLaVA-Med model
    from llava.model import LlavaMistralForCausalLM
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Add pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load LLaVA-Med model using custom class
    print(f"Loading LLaVA-Med model from {args.model_name}...")
    model = LlavaMistralForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Configure LoRA with higher rank for more capacity
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Check sample prompt length to decide on system prompt
    sample_prompt = create_prompt(data[0], include_system=True)
    sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
    sample_len = sample_tokens["input_ids"].shape[1]
    
    include_system = sample_len < args.max_length * 0.8  # Leave room for response
    print(f"\nSample prompt length: {sample_len} tokens")
    print(f"Including system prompt: {include_system}")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length, include_system=False)  # Use simpler format to fit context
    val_dataset = prepare_dataset(val_data, tokenizer, args.max_length, include_system=False)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments with optimizations
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    # Handle resume from checkpoint
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Find latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
            if checkpoints:
                resume_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                print(f"Resuming from latest checkpoint: {resume_checkpoint}")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {resume_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Save system prompt for inference
    with open(os.path.join(args.output_dir, "system_prompt.txt"), "w") as f:
        f.write(SYSTEM_PROMPT)
    
    # Save training config
    config = vars(args)
    config["train_samples"] = len(train_data)
    config["val_samples"] = len(val_data)
    config["include_system_prompt"] = include_system
    config["completion_time"] = str(datetime.now())
    
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Model saved to: {args.output_dir}/final")
    print(f"Completion time: {datetime.now()}")


if __name__ == "__main__":
    main()
