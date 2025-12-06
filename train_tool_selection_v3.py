#!/usr/bin/env python3
"""
LLaVA-Med Tool Selection Fine-tuning V3
- Uses CLEANED dataset (no CallAgent, no invalid tools)
- Only 487 valid biomedical API tools
- Multi-tool output format
"""

import os
import json
import torch
import glob
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# System prompt with tool selection guidance
SYSTEM_PROMPT = """You are CorTEX, a biomedical AI assistant specialized in selecting the right tools to answer medical questions.

Given a medical question, you must select ONE OR MORE tools from the available biomedical tool library. Output each tool call on a separate line in the format:
ToolName: {arguments}

## Available Tool Categories:

### FDA Drug Information Tools
- FDA_get_indications_by_drug_name: Get approved indications for a drug
- FDA_get_mechanism_of_action_by_drug_name: Get drug mechanism of action
- FDA_get_drug_interactions_by_drug_name: Get drug-drug interactions
- FDA_get_adverse_reactions_by_drug_name: Get adverse reactions
- FDA_get_contraindications_by_drug_name: Get contraindications
- FDA_get_drug_names_by_indication: Find drugs for a given indication
- FDA_get_boxed_warning_info_by_drug_name: Get black box warnings
- FDA_get_pregnancy_or_breastfeeding_info_by_drug_name: Pregnancy safety info
- FDA_get_pediatric_use_info_by_drug_name: Pediatric usage information
- FDA_get_geriatric_use_info_by_drug_name: Geriatric usage information
- FDA_get_overdosage_info_by_drug_name: Overdose information
- FDA_get_pharmacokinetics_by_drug_name: Pharmacokinetic data
- FDA_get_pharmacodynamics_by_drug_name: Pharmacodynamic data
- FDA_get_clinical_pharmacology_by_drug_name: Clinical pharmacology
- FDA_get_dosage_and_storage_information_by_drug_name: Dosage information

### Clinical Guidelines Tools
- TRIP_Database_Guidelines_Search: Search TRIP database for clinical guidelines
- NICE_Clinical_Guidelines_Search: Search NICE clinical guidelines
- GIN_Guidelines_Search: Search Guidelines International Network
- EuropePMC_Guidelines_Search: Search Europe PMC for guidelines
- OpenAlex_Guidelines_Search: Search OpenAlex for guidelines
- WHO_Guidelines_Search: Search WHO guidelines
- CMA_Guidelines_Search: Search CMA guidelines
- PubMed_Guidelines_Search: Search PubMed for guidelines

### Human Phenotype Ontology (HPO) Tools
- get_HPO_ID_by_phenotype: Convert phenotype description to HPO ID
- get_phenotype_by_HPO_ID: Get phenotype details from HPO ID
- get_joint_associated_diseases_by_HPO_ID_list: Find diseases matching multiple phenotypes

### OpenTargets Tools
- OpenTargets_get_disease_ids_by_name: Get disease EFO IDs by name
- OpenTargets_get_target_id_description_by_name: Get target info by gene name
- OpenTargets_get_associated_targets_by_disease_efoId: Find drug targets for disease
- OpenTargets_get_associated_drugs_by_disease_efoId: Find drugs for disease
- OpenTargets_get_associated_drugs_by_target_ensemblID: Find drugs targeting a gene
- OpenTargets_get_drug_mechanisms_of_action_by_chemblId: Get drug MOA
- OpenTargets_get_disease_description_by_efoId: Get disease description

### Gene/Protein Tools
- HPA_search_genes_by_query: Search Human Protein Atlas for genes
- HPA_get_biological_processes_by_gene: Get biological processes for gene
- HPA_get_protein_interactions_by_gene: Get protein interactions
- GO_get_genes_for_term: Get genes for GO term
- GO_get_term_details: Get GO term details
- UniProt_search: Search UniProt database
- UniProt_get_function_by_accession: Get protein function
- ensembl_lookup_gene: Look up gene in Ensembl

### Literature Search Tools
- PubMed_search_articles: Search PubMed
- EuropePMC_search_articles: Search Europe PMC
- SemanticScholar_search_papers: Search Semantic Scholar
- DOAJ_search_articles: Search open access journals
- openalex_literature_search: Search OpenAlex

### Clinical Trials Tools
- search_clinical_trials: Search clinical trials
- get_clinical_trial_conditions_and_interventions: Get trial details
- get_clinical_trial_eligibility_criteria: Get eligibility criteria
- get_clinical_trial_outcome_measures: Get outcome measures

### Drug Safety (FAERS) Tools  
- FAERS_search_adverse_event_reports: Search FDA adverse events
- FAERS_count_reactions_by_drug_event: Count reactions by drug
- FAERS_search_serious_reports_by_drug: Search serious adverse events

### MedlinePlus Tools
- MedlinePlus_search_topics_by_keyword: Search health topics
- MedlinePlus_get_genetics_condition_by_name: Get genetic condition info
- MedlinePlus_get_genetics_gene_by_name: Get gene information

### EU Health Info Tools
- euhealthinfo_search_causes_of_death: Search mortality data
- euhealthinfo_search_infectious_diseases: Search infectious disease data
- euhealthinfo_search_hospital_in_patient_data: Search hospital data
- euhealthinfo_search_cancer_registry: Search cancer registry

### Variant/Genetic Tools
- clinvar_search_variants: Search ClinVar for variants
- clinvar_get_clinical_significance: Get variant significance
- GWAS_search_associations_by_gene: Search GWAS by gene
- dbsnp_search_by_gene: Search dbSNP by gene

### DrugBank Tools
- drugbank_get_indications_by_drug_name_or_drugbank_id: Get drug indications
- drugbank_get_drug_interactions_by_drug_name_or_drugbank_id: Get interactions
- drugbank_get_targets_by_drug_name_or_drugbank_id: Get drug targets
- drugbank_get_pharmacology_by_drug_name_or_drugbank_id: Get pharmacology

### Pathway Tools
- kegg_search_pathway: Search KEGG pathways
- WikiPathways_search: Search WikiPathways
- Reactome_get_pathway_reactions: Get pathway reactions

### Structure Tools
- alphafold_get_prediction: Get AlphaFold structure prediction
- PDB_search_similar_structures: Search similar protein structures

## Instructions:
1. Analyze the medical question carefully
2. Identify what information is needed (drugs, diseases, genes, guidelines, etc.)
3. Select the most appropriate tools that can provide the needed information
4. Output tool calls with appropriate arguments
5. Use multiple tools when the question requires diverse information sources

Remember: Output ONLY the tool calls, one per line, in the format: ToolName: {arguments}"""


def load_cleaned_data(data_dir: str = "datasets/finetuning/cleaned"):
    """Load the cleaned tool selection dataset."""
    train_file = os.path.join(data_dir, "model1_tool_selection_train_cleaned.jsonl")
    val_file = os.path.join(data_dir, "model1_tool_selection_val_cleaned.jsonl")
    
    train_data = []
    val_data = []
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
    
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            val_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    
    return train_data, val_data


def format_sample(sample: dict) -> dict:
    """Format a sample for training."""
    question = sample['input']
    tools_output = sample['output']
    
    # Create the full prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nTool Calls:"
    
    return {
        'input': full_prompt,
        'output': tools_output
    }


def prepare_dataset(data: list, tokenizer, max_length: int = 4096):
    """Prepare dataset for training with batch tokenization."""
    from tqdm import tqdm
    
    print(f"Formatting {len(data)} samples...")
    formatted_samples = [format_sample(sample) for sample in tqdm(data, desc="Formatting")]
    
    # Prepare texts for batch tokenization
    input_texts = [s['input'] for s in formatted_samples]
    full_texts = [s['input'] + "\n" + s['output'] for s in formatted_samples]
    
    print("Batch tokenizing inputs...")
    input_encodings = tokenizer(
        input_texts,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        padding=False
    )
    
    print("Batch tokenizing full texts...")
    full_encodings = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        padding=False
    )
    
    print("Creating training examples...")
    formatted_data = []
    for i in tqdm(range(len(data)), desc="Processing"):
        input_ids = input_encodings['input_ids'][i]
        full_ids = full_encodings['input_ids'][i]
        
        # Create labels (mask input portion with -100)
        labels = [-100] * len(input_ids) + full_ids[len(input_ids):]
        labels = labels[:max_length]
        full_ids = full_ids[:max_length]
        
        # Pad labels if needed
        if len(labels) < len(full_ids):
            labels = labels + [-100] * (len(full_ids) - len(labels))
        
        formatted_data.append({
            'input_ids': full_ids,
            'attention_mask': [1] * len(full_ids),
            'labels': labels
        })
    
    return Dataset.from_list(formatted_data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--output_dir", type=str, default="checkpoints/llava-med-tool-selection-v3")
    parser.add_argument("--data_dir", type=str, default="datasets/finetuning/cleaned")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLaVA-Med Tool Selection Fine-tuning V3")
    print("Using CLEANED dataset (no meta tools, no invalid tools)")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print(f"\nLoading cleaned data from {args.data_dir}...")
    train_data, val_data = load_cleaned_data(args.data_dir)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    val_dataset = prepare_dataset(val_data, tokenizer, args.max_length)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Load model with quantization
    print(f"\nLoading model from {args.model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # Import custom model
    from llava.model import LlavaMistralForCausalLM
    
    model = LlavaMistralForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n✅ Training complete! Model saved to {final_dir}")


if __name__ == "__main__":
    main()
