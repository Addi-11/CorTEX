"""
LLaVA-Med Finetuning Script for MedInst/MedS-Ins Dataset
Fine-tunes the language model on medical instruction-following data.

This script supports:
- Multi-GPU training with DeepSpeed
- LoRA (Low-Rank Adaptation) for efficient finetuning
- Full finetuning option
- Mixed precision training
- Text-only and vision-language finetuning (single-image per sample)
"""

import os
import sys
import json
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
import argparse
import transformers
from dataclasses import dataclass, field
from typing import Optional, Any
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image

# Add LLaVA-Med repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LLaVA-Med"))

from llava.model import LlavaMistralForCausalLM
from llava.conversation import conv_templates


# -----------------------------
# Args
# -----------------------------


@dataclass
class ModelArguments:
    model_path: str = field(
        default="/home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2",
        metadata={"help": "Path to the pretrained LLaVA-Med model (or HF hub id)"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for efficient finetuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (higher = more capacity)"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha parameter (typically 2x lora_r)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the vision tower during training"}
    )
    vision_finetune: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, expect image inputs and pass them to the model. "
                "If False, run text-only finetuning."
            )
        }
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="datasets/M2training/text_data.jsonl",
        metadata={"help": "Path to the training data folder (json/jsonl files)"}
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root folder where images referenced in the dataset live"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use (for debugging)"}
    )
    conv_mode: str = field(
        default="mistral_instruct",
        metadata={"help": "Conversation template key from llava.conversation.conv_templates"}
    )


# -----------------------------
# Dataset
# -----------------------------


class MedInstDataset(Dataset):
    """
    Dataset for medical instruction tuning.

    This version supports:
    - pure text samples (no image)
    - multimodal samples with a single image per example

    Expected sample schema (examples):
    {
        "instruction": "...",
        "input": "...",           # optional
        "output": "...",
        "image": "file.jpg"       # optional, relative to image_root, or absolute path
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        conv_mode: str = "mistral_instruct",
        image_root: Optional[str] = None,
        image_processor: Optional[Any] = None,
        vision_finetune: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conv_mode = conv_mode
        self.image_root = image_root
        self.image_processor = image_processor
        self.vision_finetune = vision_finetune
        self.data = []

        print(f"Loading data from {data_path}...")

        if os.path.isfile(data_path):
            self._load_file(data_path)
        else:
            for filename in sorted(os.listdir(data_path)):
                if (filename.endswith(".json") or filename.endswith(".jsonl") or filename.endswith(".csv")) and not filename.startswith("_"):
                    filepath = os.path.join(data_path, filename)
                    self._load_file(filepath)

        if max_samples:
            self.data = self.data[:max_samples]

        print(f"Loaded {len(self.data)} samples")

    def _load_file(self, filepath: str):
        """Load samples from a JSON/JSONL/CSV file."""
        before = len(self.data)
        try:
            # Handle CSV files
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    sample = {
                        'instruction': str(row.get('instruction', '')) if pd.notna(row.get('instruction')) else '',
                        'input': str(row.get('input', '')) if pd.notna(row.get('input')) else '',
                        'output': str(row.get('output', '')) if pd.notna(row.get('output')) else '',
                    }
                    # Add image field if present
                    if 'image' in row and pd.notna(row.get('image')):
                        sample['image'] = str(row['image'])
                    self.data.append(sample)
                print(
                    f"  Loaded {filepath}: {len(self.data) - before} samples "
                    f"(total {len(self.data)})"
                )
                return
            
            # Handle JSON/JSONL files
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

                if not content:
                    return

                if content[0] == "[":
                    samples = json.loads(content)
                    self.data.extend(samples)
                else:
                    for line in content.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                            self.data.append(sample)
                        except json.JSONDecodeError:
                            continue
            print(
                f"  Loaded {filepath}: {len(self.data) - before} samples "
                f"(total {len(self.data)})"
            )
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    def __len__(self):
        return len(self.data)

    def _build_conversation(self, instruction: str, input_text: str, output_text: str) -> str:
        if instruction and input_text:
            user_message = f"{instruction}\n\n{input_text}"
        elif input_text:
            user_message = input_text
        else:
            user_message = instruction

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], output_text)
        return conv.get_prompt()

    def __getitem__(self, idx):
        sample = self.data[idx]

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        conversation = self._build_conversation(instruction, input_text, output_text)

        tokenized = self.tokenizer(
            conversation,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # Mask out the prompt part (up to [/INST]) from the loss
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

        example = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }

        # Optionally load and process image
        if self.vision_finetune and self.image_processor is not None:
            image_field = sample.get("image", None)
            if image_field:
                image_path = image_field
                if self.image_root is not None and not os.path.isabs(image_path):
                    image_path = os.path.join(self.image_root, image_path)
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = self.image_processor(
                        image, return_tensors="pt"
                    )["pixel_values"][0]
                    example["pixel_values"] = image_tensor
                except Exception as e:
                    # If image loading fails, we simply drop the image and treat as text-only
                    print(f"  Warning: failed to load image {image_path}: {e}")

        return example


# -----------------------------
# Collator (text + optional vision)
# -----------------------------


class VisionDataCollatorForSeq2Seq:
    """
    Data collator that:
    - pads text inputs with tokenizer.pad
    - stacks pixel_values if present (for vision finetuning)

    It also tolerates mixed batches where some samples don't have images by
    filling missing pixel_values with zeros of the same shape as the first found one.
    """

    def __init__(self, tokenizer, max_length: int = 2048, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        has_any_pixel = any("pixel_values" in f for f in features)
        pixel_values_list = []

        if has_any_pixel:
            for f in features:
                if "pixel_values" in f:
                    pixel_values_list.append(f.pop("pixel_values"))
                else:
                    pixel_values_list.append(None)

        # Separate labels before padding (need special handling)
        labels_list = [f.pop("labels") for f in features]
        
        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Pad labels separately with -100 (ignore index)
        max_label_length = batch["input_ids"].shape[1]
        padded_labels = []
        for labels in labels_list:
            padding_length = max_label_length - len(labels)
            if padding_length > 0:
                # Pad with -100 so they're ignored in loss
                padded_labels.append(labels + [-100] * padding_length)
            else:
                padded_labels.append(labels[:max_label_length])
        batch["labels"] = torch.tensor(padded_labels)

        if has_any_pixel:
            first = next(p for p in pixel_values_list if p is not None)
            filled = [
                p if p is not None else torch.zeros_like(first) for p in pixel_values_list
            ]
            batch["pixel_values"] = torch.stack(filled, dim=0)

        return batch


# -----------------------------
# Model / tokenizer / processor
# -----------------------------


def load_model_and_tokenizer(model_args: ModelArguments, local_rank: int = -1):
    """Load the LLaVA-Med model, tokenizer, and (optionally) image processor."""

    print(f"Loading model from {model_args.model_path}...")

    # For vision finetuning, we try to load AutoProcessor (LLaVA-Med provides one).
    if model_args.vision_finetune:
        processor = AutoProcessor.from_pretrained(model_args.model_path)
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_path,
                padding_side="right",
                use_fast=False,
            )
        image_processor = getattr(processor, "image_processor", None)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_path,
            padding_side="right",
            use_fast=False,
        )
        image_processor = None

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

        # Enable gradient checkpointing for memory efficiency
        model.enable_input_require_grads()

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

    return model, tokenizer, image_processor


# -----------------------------
# Main
# -----------------------------


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Get local rank for DDP
    local_rank = training_args.local_rank if hasattr(training_args, "local_rank") else -1

    model, tokenizer, image_processor = load_model_and_tokenizer(model_args, local_rank)

    train_dataset = MedInstDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        max_samples=data_args.max_samples,
        conv_mode=data_args.conv_mode,
        image_root=data_args.image_root,
        image_processor=image_processor,
        vision_finetune=model_args.vision_finetune,
    )

    data_collator = VisionDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        pad_to_multiple_of=8,
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

    with open(os.path.join(training_args.output_dir, "training_config.json"), "w") as f:
        json.dump(
            {
                "model_path": model_args.model_path,
                "use_lora": model_args.use_lora,
                "lora_r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "data_path": data_args.data_path,
                "max_length": data_args.max_length,
                "vision_finetune": model_args.vision_finetune,
                "image_root": data_args.image_root,
            },
            f,
            indent=2,
        )

    # Plot training progress
    plot_training_progress(trainer, training_args.output_dir)

    print("Training complete!")


def plot_training_progress(trainer, output_dir: str):
    """Plot and save training loss curve from trainer's log history."""
    log_history = trainer.state.log_history
    
    steps = []
    losses = []
    learning_rates = []
    grad_norms = []
    epochs = []
    
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry["loss"])
            epochs.append(entry.get("epoch", 0))
            if "learning_rate" in entry:
                learning_rates.append(entry["learning_rate"])
            if "grad_norm" in entry:
                grad_norms.append(entry["grad_norm"])
    
    if not steps:
        print("No training logs found to plot")
        return
    
    # Create figure with subplots
    n_plots = 1 + (1 if learning_rates else 0) + (1 if grad_norms else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot training loss
    axes[plot_idx].plot(steps, losses, 'b-', linewidth=1.5, label='Training Loss')
    axes[plot_idx].set_xlabel('Steps')
    axes[plot_idx].set_ylabel('Loss')
    axes[plot_idx].set_title('Training Loss over Steps')
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].legend()
    
    # Add smoothed loss
    if len(losses) > 10:
        window = min(50, len(losses) // 5)
        smoothed = pd.Series(losses).rolling(window=window, center=True).mean()
        axes[plot_idx].plot(steps, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
        axes[plot_idx].legend()
    
    plot_idx += 1
    
    # Plot learning rate
    if learning_rates and plot_idx < len(axes):
        axes[plot_idx].plot(steps[:len(learning_rates)], learning_rates, 'g-', linewidth=1.5)
        axes[plot_idx].set_xlabel('Steps')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plot_idx += 1
    
    # Plot gradient norm
    if grad_norms and plot_idx < len(axes):
        axes[plot_idx].plot(steps[:len(grad_norms)], grad_norms, 'm-', linewidth=1.5)
        axes[plot_idx].set_xlabel('Steps')
        axes[plot_idx].set_ylabel('Gradient Norm')
        axes[plot_idx].set_title('Gradient Norm over Steps')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {plot_path}")
    
    # Also save training stats
    stats = {
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "total_steps": steps[-1] if steps else 0,
        "total_epochs": epochs[-1] if epochs else 0,
    }
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Training stats saved to {stats_path}")


if __name__ == "__main__":
    main()
