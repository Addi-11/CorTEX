"""
LLaVA-Med PPO (Proximal Policy Optimization) Training Script
Fine-tunes vision-language model using PPO with clipped surrogate objective.

Key differences from GRPO:
- Uses a learned value network for advantage estimation
- Clipped surrogate objective for stable policy updates
- Multiple optimization epochs per batch of experiences
- GAE (Generalized Advantage Estimation) for variance reduction
"""

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
import json
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
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import rewards script from grpo
sys.path.insert(0, '/home/azureuser/localfiles/cortex-project/GRPO')
from rewards import compute_reward
sys.path.insert(0, '/home/azureuser/localfiles/cortex-project/LLaVA-Med')
from llava.model import LlavaMistralForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


MODEL_ID = "/home/azureuser/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"
DATASET_PATH = "/home/azureuser/localfiles/cortex-project/datasets/VisualQA-PMCArticle-Dataset/test/llava_med_eval_qa50_qa.jsonl"
IMAGE_BASE_DIR = "/home/azureuser/localfiles/cortex-project/datasets/VisualQA-PMCArticle-Dataset/test/images"
OUTPUT_DIR = "/home/azureuser/localfiles/cortex-project/PPO/ppo_output"


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    num_generations: int = 2          
    temperature: float = 1.0
    max_new_tokens: int = 48
    
    clip_epsilon: float = 0.2        
    value_clip_epsilon: float = 0.2   
    gamma: float = 0.99              
    gae_lambda: float = 0.95          
    
    learning_rate: float = 1e-5
    value_lr: float = 1e-4           
    num_epochs: int = 3              
    ppo_epochs: int = 2               
    batch_size: int = 1
    gradient_accumulation_steps: int = 4  
    max_grad_norm: float = 0.5
    
    # Entropy & KL
    entropy_coef: float = 0.01       
    vf_coef: float = 0.5            
    target_kl: float = 0.02         
    # Logging
    logging_steps: int = 2          

CONFIG = PPOConfig()


class ValueHead(nn.Module):
    """Value network head for PPO.
    
    Takes the last hidden state from the LM and predicts a scalar value.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.value_out = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states[:, -1, :]
        x = self.dropout(torch.tanh(self.dense(x)))
        return self.value_out(x).squeeze(-1)


class PPODataset(Dataset):
    """Dataset for PPO training with multimodal support."""
    
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
            question = example.get("text", "").replace("<image>", "").strip()
            
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


@dataclass
class Experience:
    """Single experience tuple for PPO."""
    prompt: str
    image: Image.Image
    response: str
    response_ids: torch.Tensor
    log_prob: float
    value: float
    reward: float
    advantage: float = 0.0
    returns: float = 0.0


def load_model(local_rank: int):
    """Load LLaVA-Med model with LoRA and value head."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device(f"cuda:{local_rank}")
    model = LlavaMistralForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,  # Memory optimization
    )
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    # LoRA config - smaller for memory efficiency
    lora_config = LoraConfig(
        r=4,                   
        lora_alpha=16,          
        lora_dropout=0.0,       # No dropout saves memory
        target_modules=["q_proj", "v_proj"],  
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    model.gradient_checkpointing_enable()
    
    hidden_size = model.config.hidden_size
    value_head = ValueHead(hidden_size).to(device, dtype=torch.float16)
    
    if local_rank == 0:
        model.print_trainable_parameters()
        print(f"Value head params: {sum(p.numel() for p in value_head.parameters()):,}")
    
    return model, value_head, tokenizer, image_processor


def get_log_probs_and_values(
    model: nn.Module,
    value_head: nn.Module,
    tokenizer,
    image_processor,
    prompt: str,
    response: str,
    image: Image.Image,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute log probabilities and values for a response."""
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    
    resp_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = torch.cat([input_ids, resp_ids], dim=1)
    
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            images=image_tensor,
            attention_mask=torch.ones_like(full_ids),
            output_hidden_states=True,
        )
    
    logits = outputs.logits[:, input_ids.shape[1]-1:-1, :]  
    log_probs = torch.log_softmax(logits, dim=-1)
    
    token_log_probs = log_probs.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)
    total_log_prob = token_log_probs.sum()
    
    hidden_states = outputs.hidden_states[-1]
    value = value_head(hidden_states)
    
    return total_log_prob, value, resp_ids


def generate_and_collect_experiences(
    model: nn.Module,
    value_head: nn.Module,
    tokenizer,
    image_processor,
    batch: dict,
    device: torch.device,
    config: PPOConfig
) -> List[Experience]:
    """Generate responses and collect experiences for PPO."""
    
    prompt = batch['prompt']
    image = batch['image']
    reference = batch['reference_answer']
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    
    experiences = []
    
    for _ in range(config.num_generations):
        with torch.no_grad():
            generated_ids = input_ids.clone()
            log_probs_list = []
            
            for _ in range(config.max_new_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    images=image_tensor,
                    attention_mask=torch.ones_like(generated_ids),
                    output_hidden_states=True,
                    use_cache=False,
                )
                
                next_token_logits = outputs.logits[:, -1, :] / config.temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                token_log_prob = log_probs.gather(1, next_token).item()
                log_probs_list.append(token_log_prob)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            hidden_states = outputs.hidden_states[-1]
            value = value_head(hidden_states).item()
            
            response_ids = generated_ids[0, input_ids.shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            reward = compute_reward(response_text, reference)
            total_log_prob = sum(log_probs_list)
            
            experiences.append(Experience(
                prompt=prompt,
                image=image,
                response=response_text,
                response_ids=response_ids.cpu(),
                log_prob=total_log_prob,
                value=value,
                reward=reward,
            ))
    
    return experiences


def compute_gae(
    experiences: List[Experience],
    gamma: float,
    gae_lambda: float
) -> List[Experience]:
    """Compute Generalized Advantage Estimation."""
    
    # For single-step episode, GAE simplifies to:
    # advantage = reward - value
    # returns = reward
    
    for exp in experiences:
        exp.returns = exp.reward
        exp.advantage = exp.reward - exp.value
    
    advantages = [exp.advantage for exp in experiences]
    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages) + 1e-8
    
    for exp in experiences:
        exp.advantage = (exp.advantage - mean_adv) / std_adv
    
    return experiences


def compute_ppo_loss(
    model: nn.Module,
    value_head: nn.Module,
    tokenizer,
    image_processor,
    experiences: List[Experience],
    device: torch.device,
    config: PPOConfig
) -> Tuple[torch.Tensor, dict]:
    """Compute PPO loss with clipped surrogate objective."""
    
    total_policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_value_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_entropy = torch.tensor(0.0, device=device)
    
    approx_kl = 0.0
    clip_fracs = []
    
    for exp in experiences:
        input_ids = tokenizer_image_token(exp.prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device)
        
        resp_ids = exp.response_ids.unsqueeze(0).to(device)
        if resp_ids.shape[1] == 0:
            continue
            
        full_ids = torch.cat([input_ids, resp_ids], dim=1)
        
        image_tensor = process_images([exp.image], image_processor, model.config)
        image_tensor = image_tensor.to(device, dtype=torch.float16)
        
        outputs = model(
            input_ids=full_ids,
            images=image_tensor,
            attention_mask=torch.ones_like(full_ids),
            output_hidden_states=True,
        )
        
        logits = outputs.logits[:, input_ids.shape[1]-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, resp_ids.unsqueeze(-1)).squeeze(-1)
        new_log_prob = token_log_probs.sum()
        
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        total_entropy = total_entropy + entropy
        
        hidden_states = outputs.hidden_states[-1]
        new_value = value_head(hidden_states)
        
        old_log_prob = torch.tensor(exp.log_prob, device=device)
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        advantage = torch.tensor(exp.advantage, device=device, dtype=torch.float32)
        
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2)
        
        old_value = torch.tensor(exp.value, device=device, dtype=torch.float32)
        returns = torch.tensor(exp.returns, device=device, dtype=torch.float32)
        
        value_pred = new_value.float()
        value_clipped = old_value + torch.clamp(value_pred - old_value, -config.value_clip_epsilon, config.value_clip_epsilon)
        value_loss1 = (value_pred - returns) ** 2
        value_loss2 = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2)
        
        total_policy_loss = total_policy_loss + policy_loss
        total_value_loss = total_value_loss + value_loss
        
        approx_kl += (old_log_prob - new_log_prob).item()
        clip_fracs.append(((ratio - 1).abs() > config.clip_epsilon).float().item())
    
    n = len(experiences)
    total_policy_loss = total_policy_loss / max(n, 1)
    total_value_loss = total_value_loss / max(n, 1)
    total_entropy = total_entropy / max(n, 1)
    
    loss = total_policy_loss + config.vf_coef * total_value_loss - config.entropy_coef * total_entropy
    
    stats = {
        "policy_loss": total_policy_loss.item(),
        "value_loss": total_value_loss.item(),
        "entropy": total_entropy.item(),
        "approx_kl": approx_kl / max(n, 1),
        "clip_frac": np.mean(clip_fracs) if clip_fracs else 0.0,
    }
    
    return loss, stats


def train():
    """Main PPO training loop."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    is_main = local_rank == 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if is_main:
        print("=" * 60)
        print("PPO Training for LLaVA-Med")
        print("=" * 60)
        print(f"\nConfig:")
        for k, v in vars(CONFIG).items():
            print(f"  {k}: {v}")
        print()
    
    if is_main:
        print("Loading model...")
    
    model, value_head, tokenizer, image_processor = load_model(local_rank)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
        value_head = torch.nn.parallel.DistributedDataParallel(
            value_head, device_ids=[local_rank], output_device=local_rank
        )
    
    if is_main:
        print("Loading dataset...")
    
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    
    dataset = PPODataset(data, IMAGE_BASE_DIR)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=lambda x: x[0])
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    policy_optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=CONFIG.value_lr)
    
    csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    if is_main:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'step', 'policy_loss', 'value_loss', 'entropy', 
                'approx_kl', 'clip_frac', 'avg_reward'
            ])
    
    if is_main:
        print(f"\nStarting PPO training...")
        print(f"  Samples: {len(dataset)}")
        print(f"  Generations per prompt: {CONFIG.num_generations}")
        print(f"  PPO epochs per batch: {CONFIG.ppo_epochs}")
        print(f"  World size: {world_size}\n")
    
    base_model = model.module if world_size > 1 else model
    base_value_head = value_head.module if world_size > 1 else value_head
    
    model.train()
    value_head.train()
    
    global_step = 0
    
    for epoch in range(CONFIG.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not is_main)
        epoch_rewards = []
        accumulated_stats = {k: 0.0 for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]}
        
        for batch_idx, batch in enumerate(progress):
            # Clear cache before each batch
            torch.cuda.empty_cache()
            
            experiences = generate_and_collect_experiences(
                base_model, base_value_head, tokenizer, image_processor,
                batch, device, CONFIG
            )
            
            # Compute advantages with GAE
            experiences = compute_gae(experiences, CONFIG.gamma, CONFIG.gae_lambda)
            
            rewards = [exp.reward for exp in experiences]
            epoch_rewards.extend(rewards)
            
            if is_main and batch_idx % 10 == 0:
                print(f"\n[Sample {batch_idx}] Rewards: {[f'{r:.3f}' for r in rewards]}")
                print(f"  Advantages: {[f'{exp.advantage:.3f}' for exp in experiences]}")
            
            for ppo_epoch in range(CONFIG.ppo_epochs):
                loss, stats = compute_ppo_loss(
                    base_model, base_value_head, tokenizer, image_processor,
                    experiences, device, CONFIG
                )
                
                if stats["approx_kl"] > CONFIG.target_kl:
                    if is_main and ppo_epoch > 0:
                        print(f"  Early stopping at PPO epoch {ppo_epoch} due to KL={stats['approx_kl']:.4f}")
                    break
                
                loss = loss / CONFIG.gradient_accumulation_steps
                loss.backward()
                
                for k, v in stats.items():
                    accumulated_stats[k] += v / CONFIG.ppo_epochs
            
            if (batch_idx + 1) % CONFIG.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(value_head.parameters(), CONFIG.max_grad_norm)
                
                policy_optimizer.step()
                value_optimizer.step()
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()
                
                global_step += 1
                
                if is_main and global_step % CONFIG.logging_steps == 0:
                    avg_reward = np.mean(epoch_rewards[-CONFIG.num_generations * CONFIG.gradient_accumulation_steps:])
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            global_step,
                            accumulated_stats["policy_loss"] / CONFIG.gradient_accumulation_steps,
                            accumulated_stats["value_loss"] / CONFIG.gradient_accumulation_steps,
                            accumulated_stats["entropy"] / CONFIG.gradient_accumulation_steps,
                            accumulated_stats["approx_kl"] / CONFIG.gradient_accumulation_steps,
                            accumulated_stats["clip_frac"] / CONFIG.gradient_accumulation_steps,
                            avg_reward
                        ])
                    accumulated_stats = {k: 0.0 for k in accumulated_stats}
            
            if is_main:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    reward=f"{np.mean(rewards):.2f}",
                    kl=f"{stats['approx_kl']:.4f}"
                )
        
        if is_main:
            print(f"\nEpoch {epoch+1} - Avg Reward: {np.mean(epoch_rewards):.4f}")
            
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            base_model.save_pretrained(ckpt_dir)
            torch.save(base_value_head.state_dict(), os.path.join(ckpt_dir, "value_head.pt"))
    
    if is_main:
        final_dir = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)
        base_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        torch.save(base_value_head.state_dict(), os.path.join(final_dir, "value_head.pt"))
        
        print(f"\n✓ Training complete! Model saved to {final_dir}")
        
        plot_training_curves(csv_path, OUTPUT_DIR)
    
    if world_size > 1:
        dist.destroy_process_group()


def plot_training_curves(csv_path: str, output_dir: str):
    """Plot PPO training curves."""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) < 2:
            print("Not enough data points to plot.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = [
            ('policy_loss', 'Policy Loss', 'blue'),
            ('value_loss', 'Value Loss', 'orange'),
            ('entropy', 'Entropy', 'green'),
            ('approx_kl', 'Approx KL', 'red'),
            ('clip_frac', 'Clip Fraction', 'purple'),
            ('avg_reward', 'Average Reward', 'cyan'),
        ]
        
        for ax, (col, title, color) in zip(axes.flat, metrics):
            ax.plot(df['step'], df[col], f'{color[0]}-', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            if len(df) > 5:
                window = min(5, len(df) // 2)
                smoothed = df[col].rolling(window=window, center=True).mean()
                ax.plot(df['step'], smoothed, f'{color[0]}-', linewidth=2.5, label='Smoothed')
                ax.legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "ppo_training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
        
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")


if __name__ == "__main__":
    train()
