import torch
import json
import os
import sys
import csv
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist

from rewards import compute_reward
sys.path.insert(0, '/home/azureuser/localfiles/cortex-project/LLaVA-Med')
from llava.model import LlavaMistralForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


MODEL_ID = "/home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"

DATASET_PATH = "/home/azureuser/localfiles/cortex-project/datasets/VisualQA-PMCArticle-Dataset/test/llava_med_eval_qa50_qa.jsonl"
IMAGE_BASE_DIR = "/home/azureuser/localfiles/cortex-project/datasets/VisualQA-PMCArticle-Dataset/test/images"
OUTPUT_DIR = "/home/azureuser/localfiles/cortex-project/RL/grpo_output"

CONFIG = {
    "num_generations": 2,  
    "temperature": 1.0,        
    "max_new_tokens": 64,      
    "learning_rate": 5e-6,     
    "num_epochs": 1,      
    "batch_size": 1,               
    "gradient_accumulation_steps": 1,  
    "logging_steps": 10,
}

class GRPODataset(Dataset):
    """Dataset for GRPO training with multimodal support.
    
    Supports two formats:
    1. QA format: {"text": ..., "gpt4_answer": ..., "image": ...}
    2. Conversation format: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}], "image": ...}
    """
    
    def __init__(self, data, image_base_dir):
        self.data = data
        self.image_base_dir = image_base_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        

        image_field = example.get("image", "")
        if image_field.startswith("datasets/"):

            image_path = os.path.join(self.image_base_dir, image_field)
        elif "/" in image_field:
            image_path = image_field
        else:
            image_path = os.path.join(self.image_base_dir, image_field)
        
        image = Image.open(image_path).convert("RGB")
        
        if "conversations" in example:
            # M2 conversation format
            conversations = example["conversations"]
            question = ""
            reference_answer = ""
            
            for turn in conversations:
                if turn["from"] == "human":
                    question = turn["value"].replace("<image>", "").strip()
                elif turn["from"] == "gpt":
                    reference_answer = turn["value"]
            
            prompt = f"<image>\n{question}"
            
        else:
            # QA format (original)
            question = example.get("text", "")
            question = question.replace("<image>", "").strip()
            
            context_parts = []
            
            if "fig_caption" in example and example["fig_caption"]:
                context_parts.append(f"Figure caption: {example['fig_caption']}")
            
            if "in_text_mention" in example and example["in_text_mention"]:
                mention_tokens = []
                for mention in example["in_text_mention"]:
                    if "tokens" in mention:
                        mention_tokens.append(mention["tokens"])
                if mention_tokens:
                    context_parts.append("Related text: " + " ".join(mention_tokens))
            
            if context_parts:
                context_str = "\n".join(context_parts)
                question = f"{context_str}\n\nQuestion: {question}"
            
            prompt = f"<image>\n{question}"
            reference_answer = example.get("gpt4_answer", "")
        
        return {
            "prompt": prompt,
            "image": image,
            "reference_answer": reference_answer,
            "question_id": example.get("question_id", example.get("id", idx))
        }


def load_model(local_rank):
    """Load LLaVA-Med model with LoRA and image processor."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device(f"cuda:{local_rank}")
    model = LlavaMistralForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        device_map={"": device}
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    lora_config = LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    return model, tokenizer, image_processor


def generate_responses(model, tokenizer, image_processor, prompt, image, device, num_gen, temp, max_tokens):
    """Generate multiple responses for a prompt with image.
    
    Uses manual generation loop to avoid transformers version incompatibility
    (LLaVA-Med doesn't support newer transformers args like cache_position).
    """

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    
    responses = []
    for _ in range(num_gen):
        with torch.no_grad():
            generated_ids = input_ids.clone()
            
            for _ in range(max_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    images=image_tensor,
                    attention_mask=torch.ones_like(generated_ids),
                    use_cache=False,
                )
                
                next_token_logits = outputs.logits[:, -1, :] / temp
                
                # Sample from distribution (do_sample=True equivalent)
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check  EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            generated_tokens = generated_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(text)
    
    return responses


def compute_grpo_loss(model, tokenizer, image_processor, prompt, image, responses, rewards, device):
    """Compute GRPO loss with group-relative advantages and multimodal input.
    
    GRPO maximizes: E[advantage * log_prob(response)]
    Which translates to minimizing: -E[advantage * log_prob(response)]
    
    For responses with positive advantage (better than average), we want to increase their probability.
    For responses with negative advantage (worse than average), we want to decrease their probability.
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    if rewards_t.std() > 0:
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    else:
        advantages = rewards_t - rewards_t.mean()
    
    advantages = torch.clamp(advantages, -2.0, 2.0)
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0
    
    for response, advantage in zip(responses, advantages):
        resp_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if resp_ids.shape[1] == 0:
            continue
        
        full_ids = torch.cat([input_ids, resp_ids], dim=1)
        labels = full_ids.clone()
        labels[:, :input_ids.shape[1]] = -100
        
        outputs = model(
            input_ids=full_ids,
            images=image_tensor,
            attention_mask=torch.ones_like(full_ids),
            labels=labels,
        )
        
        # GRPO loss: -advantage * log_prob = advantage * CE_loss (since CE = -log_prob)
        # Positive advantage: DECREASE the loss (increase prob) -> multiply by negative
        # Negative advantage: NCREASE the loss (decrease prob) -> multiply by negative  
        # So: loss = -advantage * CE_loss
        policy_loss = -advantage * outputs.loss
        total_loss = total_loss + policy_loss
        count += 1
    
    return total_loss / max(count, 1)



def train():
    # distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    is_main = local_rank == 0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if is_main:
        print("Loading model...")
    
    model, tokenizer, image_processor = load_model(local_rank)
    
    # DDP for distributed training
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
    
    if is_main:
        print("Loading dataset...")
    
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    
    dataset = GRPODataset(data, IMAGE_BASE_DIR)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=lambda x: x[0])
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    if is_main:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'loss', 'avg_reward'])
    
    if is_main:
        print(f"\nStarting GRPO training...")
        print(f"  Samples: {len(dataset)}")
        print(f"  Generations per prompt: {CONFIG['num_generations']}")
        print(f"  World size: {world_size}")
        print(f"  Device: {device}\n")
    
    base_model = model.module if world_size > 1 else model
    
    model.train()
    step = 0
    accumulated_loss = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not is_main)
        epoch_rewards = []
        
        for batch_idx, batch in enumerate(progress):
            responses = generate_responses(
                base_model, tokenizer, image_processor, 
                batch['prompt'], batch['image'], device,
                CONFIG['num_generations'], CONFIG['temperature'], CONFIG['max_new_tokens']
            )
            
            rewards = [compute_reward(r, batch['reference_answer']) for r in responses]
            epoch_rewards.extend(rewards)
            
            if is_main and batch_idx % 10 == 0:
                print(f"\n[Sample {batch_idx}] Rewards: {[f'{r:.3f}' for r in rewards]}")
                print(f"  Response samples: {[r[:50] + '...' if len(r) > 50 else r for r in responses[:2]]}")
            
            loss = compute_grpo_loss(
                base_model, tokenizer, image_processor, 
                batch['prompt'], batch['image'], responses, rewards, device
            )
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                if is_main and step % CONFIG['logging_steps'] == 0:
                    avg_reward = np.mean(epoch_rewards[-CONFIG['num_generations']*CONFIG['gradient_accumulation_steps']:])
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([step, accumulated_loss, avg_reward])
                    accumulated_loss = 0.0
            
            if is_main:
                progress.set_postfix(loss=f"{loss.item():.4f}", reward=f"{np.mean(rewards):.2f}")
        
        if is_main:
            print(f"\nEpoch {epoch+1} - Avg Reward: {np.mean(epoch_rewards):.4f}")
            base_model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}"))
    
    if is_main:
        base_model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
        print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}/final")
        
        plot_training_curves(csv_path, OUTPUT_DIR)
    
    if world_size > 1:
        dist.destroy_process_group()


def plot_training_curves(csv_path, output_dir):
    """Plot and save training loss and reward curves."""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) < 2:
            print("Not enough data points to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Loss
        axes[0].plot(df['step'], df['loss'], 'b-', linewidth=1.5, alpha=0.7)
        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('GRPO Training Loss', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        if len(df) > 5:
            window = min(5, len(df) // 2)
            smoothed_loss = df['loss'].rolling(window=window, center=True).mean()
            axes[0].plot(df['step'], smoothed_loss, 'b-', linewidth=2.5, label='Smoothed')
            axes[0].legend()
        
        # Plot Reward
        axes[1].plot(df['step'], df['avg_reward'], 'g-', linewidth=1.5, alpha=0.7)
        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('Average Reward', fontsize=12)
        axes[1].set_title('GRPO Training Reward', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        if len(df) > 5:
            window = min(5, len(df) // 2)
            smoothed_reward = df['avg_reward'].rolling(window=window, center=True).mean()
            axes[1].plot(df['step'], smoothed_reward, 'g-', linewidth=2.5, label='Smoothed')
            axes[1].legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {plot_path}")
        
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")


if __name__ == "__main__":
    train()
