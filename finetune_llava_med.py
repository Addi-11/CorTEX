"""
LLaVA-Med Finetuning Script for MedInst/MedS-Ins Dataset
Fine-tunes the language model on medical instruction-following data

This script supports:
- Multi-GPU training with DeepSpeed
- LoRA (Low-Rank Adaptation) for efficient finetuning
- Full finetuning option
- Mixed precision training
"""

import os
import sys
import json
import torch

os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
import argparse
import transformers
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLaVA-Med'))

from llava.model import LlavaMistralForCausalLM
from llava.conversation import conv_templates


@dataclass
class ModelArguments:
    model_path: str = field(
        default="/home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2",
        metadata={"help": "Path to the pretrained LLaVA-Med model"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for efficient finetuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the vision tower during training"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="datasets/MedS-Ins",
        metadata={"help": "Path to the training data folder"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (for debugging)"}
    )


class MedInstDataset(Dataset):
    """Dataset for medical instruction tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        conv_mode: str = "mistral_instruct"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conv_mode = conv_mode
        self.data = []
        
        print(f"Loading data from {data_path}...")
        
        if os.path.isfile(data_path):
            self._load_file(data_path)
        else:
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith('.json') and not filename.startswith('_'):
                    filepath = os.path.join(data_path, filename)
                    self._load_file(filepath)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} samples")
    
    def _load_file(self, filepath: str):
        """Load samples from a JSON/JSONL file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if content.startswith('['):
                    samples = json.loads(content)
                    self.data.extend(samples)
                else:
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                self.data.append(sample)
                            except json.JSONDecodeError:
                                continue
            print(f"  Loaded {filepath}: {len(self.data)} total samples")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        
        if instruction and input_text:
            user_message = f"{instruction}\n\n{input_text}"
        elif input_text:
            user_message = input_text
        else:
            user_message = instruction
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], output_text)
        conversation = conv.get_prompt()
        
        tokenized = self.tokenizer(
            conversation,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        
        labels = input_ids.copy()
        
        conversation_parts = conversation.split("[/INST]")
        if len(conversation_parts) > 1:
            prompt_part = conversation_parts[0] + "[/INST]"
            prompt_tokens = self.tokenizer(
                prompt_part,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            
            for i in range(min(len(prompt_tokens), len(labels))):
                labels[i] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }


def load_model_and_tokenizer(model_args: ModelArguments, local_rank: int = -1):
    """Load the LLaVA-Med model and tokenizer."""
    
    print(f"Loading model from {model_args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_path,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlavaMistralForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,  # Use bf16 for A100
        low_cpu_mem_usage=True,
    )
    
    # For DDP, move to the correct device
    if local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
    else:
        model = model.cuda()
    
    if model_args.freeze_vision_tower:
        print("Freezing vision tower...")
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            for param in vision_tower.parameters():
                param.requires_grad = False
    
    if model_args.use_lora:
        print("Applying LoRA...")
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Get local rank for DDP
    local_rank = training_args.local_rank if hasattr(training_args, 'local_rank') else -1
    
    model, tokenizer = load_model_and_tokenizer(model_args, local_rank)
    
    train_dataset = MedInstDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        max_samples=data_args.max_samples,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=data_args.max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {training_args.output_dir}...")
    
    if model_args.use_lora:
        model.save_pretrained(training_args.output_dir)
        print("Saved LoRA adapter weights")
    else:
        trainer.save_model()
    
    tokenizer.save_pretrained(training_args.output_dir)
    
    import json
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w") as f:
        json.dump({
            "model_path": model_args.model_path,
            "use_lora": model_args.use_lora,
            "lora_r": model_args.lora_r,
            "lora_alpha": model_args.lora_alpha,
            "data_path": data_args.data_path,
            "max_length": data_args.max_length,
        }, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
