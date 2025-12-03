#!/usr/bin/env python3
"""
LLaVA-Med Fine-tuning Script for Tool Selection (Model 1)
Trains LLaVA-Med to generate biomedical tool calls from medical questions.

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

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-Med for tool selection")
    
    parser.add_argument("--model_name", type=str, 
                        default="microsoft/llava-med-v1.5-mistral-7b",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, 
                        default="checkpoints/llava-med-tool-selection",
                        help="Output directory for checkpoints")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str,
                        default="datasets/finetuning",
                        help="Directory containing training data")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (for testing)")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
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


def create_prompt(sample):
    """Create training prompt from sample."""
    instruction = sample.get('instruction', 
        "Identify the biomedical database tools and drug names needed to answer this medical question.")
    question = sample['input']
    output = sample['output']
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{question}

### Response:
{output}"""
    
    return prompt


def prepare_dataset(data, tokenizer, max_length):
    """Prepare dataset for training."""
    from torch.utils.data import Dataset
    
    class ToolSelectionDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            prompt = create_prompt(sample)
            
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            labels = encoding["input_ids"].clone()
            
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }
    
    return ToolSelectionDataset(data, tokenizer, max_length)


def main():
    args = parse_args()
    
    print("="*60)
    print("LLaVA-Med Tool Selection Fine-tuning")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading datasets...")
    data = load_and_combine_datasets(args.data_dir, max_samples=args.max_samples)
    
    # Split into train/val (use smaller val set for small datasets)
    from sklearn.model_selection import train_test_split
    val_size = min(0.1, max(1, len(data) // 5)) if len(data) < 20 else 0.1
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=42)
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
        prepare_model_for_kbit_training,
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
    
    # Configure LoRA
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
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    val_dataset = prepare_dataset(val_data, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=False,  # Disabled due to peft compatibility issue
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        report_to="none",  # Disable wandb
        remove_unused_columns=False,
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
    
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Save training config
    config = vars(args)
    config["train_samples"] = len(train_data)
    config["val_samples"] = len(val_data)
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
