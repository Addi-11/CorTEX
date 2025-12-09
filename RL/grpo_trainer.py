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
import numpy as np
import torch.distributed as dist

from rewards import compute_reward
sys.path.insert(0, '/mnt/workspace/LLaVA-Med')
from llava.model import LlavaMistralForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "/mnt/workspace/CorTEX/.models/llava-med-v1.5-mistral-7b"
DATASET_PATH = "/mnt/workspace/CorTEX/RL/test/llava_med_eval_qa50_qa.jsonl"
IMAGE_DIR = "/mnt/workspace/CorTEX/RL/test/images"
OUTPUT_DIR = "/mnt/workspace/CorTEX/RL/grpo_output"

CONFIG = {
    "num_generations": 3,           # Generate 3 responses per prompt
    "temperature": 0.7,
    "max_new_tokens": 64,           # Max tokens per response
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "batch_size": 2,                # Per GPU batch size (conservative with 4 generations)
    "gradient_accumulation_steps": 2,  # Effective batch = 2 * 2 * 6 GPUs = 24
    "logging_steps": 5,
}

class GRPODataset(Dataset):
    """Dataset for GRPO training with multimodal support"""
    
    def __init__(self, data, image_dir):
        self.data = data
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, example["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Build prompt
        question = example["text"]
        
        # Remove any existing <image> token from the question
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
        
        # Add single <image> token at the beginning for LLaVA-Med
        prompt = f"<image>\n{question}"
        
        return {
            "prompt": prompt,
            "image": image,
            "reference_answer": example.get("gpt4_answer", ""),
            "question_id": example.get("question_id", idx)
        }

# ============================================================================
# Core Functions
# ============================================================================
def load_model(local_rank):
    """Load LLaVA-Med model with LoRA and image processor."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model directly to the correct GPU
    device = torch.device(f"cuda:{local_rank}")
    model = LlavaMistralForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    
    # Get vision tower and image processor
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    
    # Add LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    return model, tokenizer, image_processor


def generate_responses(model, tokenizer, image_processor, prompt, image, device, num_gen, temp, max_tokens):
    """Generate multiple responses for a prompt with image."""
    # Tokenize with image token handling
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Process image - process_images returns [N, C, H, W] tensor
    image_tensor = process_images([image], image_processor, model.config)
    # Ensure it's on the right device and dtype
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    
    responses = []
    for _ in range(num_gen):
        with torch.no_grad():
            output = model.generate(
                inputs=input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(text)
    
    return responses


def compute_grpo_loss(model, tokenizer, image_processor, prompt, image, responses, rewards, device):
    """Compute GRPO loss with group-relative advantages and multimodal input."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Group-relative normalization (GRPO key insight)
    if rewards_t.std() > 0:
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    else:
        advantages = rewards_t - rewards_t.mean()
    
    # Tokenize prompt with image token
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device)
    
    # Process image - process_images returns [N, C, H, W] tensor
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
        
        # Weight loss by advantage
        weighted_loss = outputs.loss * (1.0 - advantage * 0.1)
        total_loss = total_loss + weighted_loss
        count += 1
    
    return total_loss / max(count, 1)


# ============================================================================
# Training Loop
# ============================================================================
def train():
    # Setup distributed training
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
    
    # Wrap model with DDP for distributed training
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
    
    if is_main:
        print("Loading dataset...")
    
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    
    dataset = GRPODataset(data, IMAGE_DIR)
    
    # Use DistributedSampler for multi-GPU
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=lambda x: x[0])
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # CSV logging (only on main process)
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
    
    # Get base model for generate (DDP wrapper doesn't have generate)
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
            # Generate responses with image (use base model for generate)
            responses = generate_responses(
                base_model, tokenizer, image_processor, 
                batch['prompt'], batch['image'], device,
                CONFIG['num_generations'], CONFIG['temperature'], CONFIG['max_new_tokens']
            )
            
            # Compute rewards
            rewards = [compute_reward(r, batch['reference_answer']) for r in responses]
            epoch_rewards.extend(rewards)
            
            # Compute loss with image
            loss = compute_grpo_loss(
                base_model, tokenizer, image_processor, 
                batch['prompt'], batch['image'], responses, rewards, device
            )
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights
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
        
        # Save checkpoint (only on main process)
        if is_main:
            print(f"\nEpoch {epoch+1} - Avg Reward: {np.mean(epoch_rewards):.4f}")
            base_model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}"))
    
    # Save final model (only on main process)
    if is_main:
        base_model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
        print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}/final")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
