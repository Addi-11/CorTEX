import torch
import json
import os
import sys
import csv
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig
from PIL import Image
from tqdm import tqdm
import numpy as np


from rewards import compute_reward
sys.path.insert(0, '/mnt/workspace/LLaVA-Med')
from llava.model import LlavaMistralForCausalLM

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "/mnt/workspace/CorTEX/.models/llava-med-v1.5-mistral-7b"
DATASET_PATH = "/mnt/workspace/CorTEX/RL/test/llava_med_eval_qa50_qa.jsonl"
IMAGE_DIR = "/mnt/workspace/CorTEX/RL/test/images"
OUTPUT_DIR = "/mnt/workspace/CorTEX/RL/grpo_output"

CONFIG = {
    "num_generations": 2,
    "temperature": 0.7,
    "max_new_tokens": 64,
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "logging_steps": 5,
}


# ============================================================================
# Dataset
# ============================================================================
class GRPODataset(Dataset):
    def __init__(self, data, image_dir):
        self.data = data
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, example["image"])).convert("RGB")
        return {
            "prompt": example["text"],
            "image": image,
            "reference": example.get("gpt4_answer", ""),
        }


# ============================================================================
# Core Functions
# ============================================================================
def load_model():
    """Load LLaVA-Med model with LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlavaMistralForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def generate_responses(model, tokenizer, prompt, device, num_gen, temp, max_tokens):
    """Generate multiple responses for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    responses = []
    for _ in range(num_gen):
        with torch.no_grad():
            # LLaVA-Med expects 'inputs' not 'input_ids'
            output = model.generate(
                inputs=input_ids,
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


def compute_grpo_loss(model, tokenizer, prompt, responses, rewards, device):
    """Compute GRPO loss with group-relative advantages."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Group-relative normalization (GRPO key insight)
    if rewards_t.std() > 0:
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    else:
        advantages = rewards_t - rewards_t.mean()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading model...")
    model, tokenizer = load_model()
    model = model.to(device)
    
    print("Loading dataset...")
    with open(DATASET_PATH) as f:
        data = [json.loads(line) for line in f]
    
    dataset = GRPODataset(data, IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                           collate_fn=lambda x: x[0])  # Single item batch
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # CSV logging
    csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'loss', 'avg_reward'])
    
    print(f"\nStarting GRPO training...")
    print(f"  Samples: {len(dataset)}")
    print(f"  Generations per prompt: {CONFIG['num_generations']}")
    print(f"  Device: {device}\n")
    
    model.train()
    step = 0
    accumulated_loss = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_rewards = []
        
        for batch_idx, batch in enumerate(progress):
            # Generate responses
            responses = generate_responses(
                model, tokenizer, batch['prompt'], device,
                CONFIG['num_generations'], CONFIG['temperature'], CONFIG['max_new_tokens']
            )
            
            # Compute rewards
            rewards = [compute_reward(r, batch['reference']) for r in responses]
            epoch_rewards.extend(rewards)
            
            # Compute loss
            loss = compute_grpo_loss(model, tokenizer, batch['prompt'], responses, rewards, device)
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                if step % CONFIG['logging_steps'] == 0:
                    avg_reward = np.mean(epoch_rewards[-CONFIG['num_generations']*CONFIG['gradient_accumulation_steps']:])
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([step, accumulated_loss, avg_reward])
                    accumulated_loss = 0.0
            
            progress.set_postfix(loss=f"{loss.item():.4f}", reward=f"{np.mean(rewards):.2f}")
        
        # Save checkpoint
        print(f"\nEpoch {epoch+1} - Avg Reward: {np.mean(epoch_rewards):.4f}")
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}"))
    
    # Save final model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train()
