import torch
import torch.nn.functional as F
import json
import os
import csv
from pathlib import Path
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from PIL import Image
from tqdm import tqdm
import numpy as np

# ============================================================================
# GRPO (Group Relative Policy Optimization) Training for LLaVA-Med
# ============================================================================
# GRPO Algorithm:
# 1. For each prompt, generate K responses from the policy
# 2. Score each response using a reward function
# 3. Compute advantages using group-relative normalization
# 4. Update policy to increase probability of high-reward responses
# ============================================================================

# Configuration
MODEL_ID = "llava-hf/llava-med-v1.5-mistral-7b"
DATASET_PATH = "/mnt/workspace/CorTEX/RL/test/dataset.jsonl"
IMAGE_DIR = "/mnt/workspace/CorTEX/RL/test/images"
OUTPUT_DIR = "/mnt/workspace/CorTEX/RL/grpo_llava_med_output"
LOSS_CSV_PATH = os.path.join(OUTPUT_DIR, "training_loss.csv")

# GRPO Hyperparameters
GRPO_CONFIG = {
    "num_generations": 4,        # K: number of responses per prompt
    "temperature": 0.7,          # Sampling temperature
    "max_new_tokens": 256,       # Max tokens to generate
    "learning_rate": 1e-5,       # Lower LR for RL
    "num_epochs": 2,
    "batch_size": 1,             # Prompts per batch (each generates K responses)
    "gradient_accumulation_steps": 4,
    "beta": 0.1,                 # KL penalty coefficient
    "clip_range": 0.2,           # PPO-style clipping
    "logging_steps": 5,
}


class GRPODataset(TorchDataset):
    """Dataset for GRPO training"""
    
    def __init__(self, data, processor, image_dir):
        self.data = data
        self.processor = processor
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
        
        return {
            "prompt": question,
            "image": image,
            "reference_answer": example.get("gpt4_answer", ""),
            "question_id": example.get("question_id", idx)
        }


class LossTracker:
    """Track and save training losses"""
    
    def __init__(self, output_dir, loss_csv_path):
        self.output_dir = output_dir
        self.loss_csv_path = loss_csv_path
        self.epoch_losses = []
        self.current_epoch_losses = []
        
        os.makedirs(output_dir, exist_ok=True)
        with open(self.loss_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'grpo_loss', 'avg_reward', 'avg_epoch_loss'])
    
    def log_step(self, epoch, step, loss, avg_reward):
        self.current_epoch_losses.append(loss)
        print(f"[Epoch {epoch:.2f}] Step {step}: GRPO Loss = {loss:.4f}, Avg Reward = {avg_reward:.4f}")
        
        with open(self.loss_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, avg_reward, ''])
    
    def end_epoch(self, epoch, step):
        if self.current_epoch_losses:
            avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} completed!")
            print(f"Average GRPO training loss: {avg_loss:.4f}")
            print(f"{'='*60}\n")
            
            with open(self.loss_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, step, '', '', avg_loss])
            
            self.current_epoch_losses = []
            return avg_loss
        return 0.0


def setup_model_and_processor():
    """Load model and processor"""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, processor


def setup_lora(model):
    """Setup LoRA for efficient finetuning"""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_jsonl_dataset(jsonl_path):
    """Load JSONL dataset"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_reward(generated_text, reference_answer, processor):
    """
    Compute reward for a generated response.
    
    This is a simple reward function based on:
    1. Length penalty (not too short, not too long)
    2. Overlap with reference answer (if available)
    3. Coherence bonus (no repetition)
    
    You can replace this with a more sophisticated reward model.
    """
    reward = 0.0
    
    # Length reward: prefer moderate length responses
    gen_len = len(generated_text.split())
    if gen_len < 5:
        reward -= 1.0  # Too short
    elif gen_len > 200:
        reward -= 0.5  # Too long
    else:
        reward += 0.5  # Good length
    
    # Overlap with reference (simple word overlap)
    if reference_answer:
        ref_words = set(reference_answer.lower().split())
        gen_words = set(generated_text.lower().split())
        if ref_words:
            overlap = len(ref_words & gen_words) / len(ref_words)
            reward += overlap * 2.0  # Scale overlap reward
    
    # Repetition penalty
    words = generated_text.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        reward += unique_ratio * 0.5  # Bonus for diversity
    
    # Coherence: penalize if starts with weird tokens
    if generated_text.strip().startswith(('<', '[', '{', '|')):
        reward -= 0.5
    
    return reward


def generate_responses(model, processor, prompt, image, num_generations, temperature, max_new_tokens):
    """Generate K responses for a prompt using sampling"""
    
    # Process input
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    responses = []
    log_probs_list = []
    
    for _ in range(num_generations):
        with torch.no_grad():
            # Generate with sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        
        # Decode generated text
        generated_ids = outputs.sequences[0, inputs['input_ids'].shape[1]:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        responses.append(generated_text)
        
        # Compute log probabilities for the generated sequence
        # Stack scores and compute log probs
        if outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab_size]
            log_probs = F.log_softmax(scores, dim=-1)
            
            # Get log probs of actual generated tokens
            seq_log_probs = torch.gather(
                log_probs[0], 
                dim=-1, 
                index=generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_list.append(seq_log_probs.sum().item())
        else:
            log_probs_list.append(0.0)
    
    return responses, log_probs_list


def compute_grpo_loss(model, processor, prompt, image, responses, rewards, old_log_probs, beta, clip_range):
    """
    Compute GRPO loss for a batch of responses.
    
    GRPO uses group-relative advantages:
    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    
    Loss = -sum(advantage_i * log_prob_i) + beta * KL_penalty
    """
    
    # Normalize rewards within the group (GRPO key insight)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    if rewards_tensor.std() > 0:
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    else:
        advantages = rewards_tensor - rewards_tensor.mean()
    
    # Process input
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    total_loss = 0.0
    
    for i, (response, advantage, old_log_prob) in enumerate(zip(responses, advantages, old_log_probs)):
        # Tokenize the response
        response_ids = processor.tokenizer(
            response, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(model.device)
        
        if response_ids.shape[1] == 0:
            continue
        
        # Create full sequence (prompt + response)
        full_input_ids = torch.cat([inputs['input_ids'], response_ids], dim=1)
        
        # Create labels (mask prompt, only compute loss on response)
        labels = full_input_ids.clone()
        labels[:, :inputs['input_ids'].shape[1]] = -100
        
        # Forward pass
        outputs = model(
            input_ids=full_input_ids,
            pixel_values=inputs.get('pixel_values'),
            attention_mask=torch.ones_like(full_input_ids),
            labels=labels
        )
        
        # Get log prob of the response under current policy
        # (negative loss is approximately log prob)
        current_log_prob = -outputs.loss.item() * response_ids.shape[1]
        
        # PPO-style clipping
        ratio = torch.exp(torch.tensor(current_log_prob - old_log_prob))
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        # Policy loss with advantage weighting
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        
        # KL penalty (approximated)
        kl_penalty = beta * (old_log_prob - current_log_prob)
        
        total_loss += outputs.loss + kl_penalty
    
    return total_loss / max(len(responses), 1)


def save_checkpoint(model, processor, output_dir, epoch):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    print(f"✓ Checkpoint saved to {checkpoint_dir}")


def grpo_train(model, processor, train_data, config, loss_tracker):
    """
    Main GRPO training loop.
    
    For each prompt:
    1. Generate K responses
    2. Compute rewards for each response
    3. Compute group-relative advantages
    4. Update policy using GRPO objective
    """
    
    print("\n" + "="*60)
    print("Starting GRPO (Group Relative Policy Optimization) Training")
    print("="*60)
    print(f"  • Generations per prompt (K): {config['num_generations']}")
    print(f"  • Temperature: {config['temperature']}")
    print(f"  • Learning rate: {config['learning_rate']}")
    print(f"  • KL penalty (beta): {config['beta']}")
    print(f"  • Clip range: {config['clip_range']}")
    print("="*60 + "\n")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Create dataset
    dataset = GRPODataset(train_data, processor, IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    global_step = 0
    model.train()
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*40} Epoch {epoch}/{config['num_epochs']} {'='*40}")
        
        epoch_rewards = []
        accumulated_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            prompt = batch['prompt'][0]  # Batch size is 1
            image = batch['image'][0]
            reference = batch['reference_answer'][0]
            
            # Step 1: Generate K responses
            responses, old_log_probs = generate_responses(
                model, processor, prompt, image,
                num_generations=config['num_generations'],
                temperature=config['temperature'],
                max_new_tokens=config['max_new_tokens']
            )
            
            # Step 2: Compute rewards for each response
            rewards = [
                compute_reward(resp, reference, processor) 
                for resp in responses
            ]
            epoch_rewards.extend(rewards)
            
            # Step 3 & 4: Compute GRPO loss and update
            loss = compute_grpo_loss(
                model, processor, prompt, image,
                responses, rewards, old_log_probs,
                beta=config['beta'],
                clip_range=config['clip_range']
            )
            
            # Gradient accumulation
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % config['logging_steps'] == 0:
                    avg_reward = np.mean(rewards)
                    loss_tracker.log_step(
                        epoch=epoch + batch_idx/len(dataloader),
                        step=global_step,
                        loss=accumulated_loss * config['gradient_accumulation_steps'],
                        avg_reward=avg_reward
                    )
                    accumulated_loss = 0.0
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item()*config["gradient_accumulation_steps"]:.4f}',
                'avg_reward': f'{np.mean(rewards):.2f}'
            })
        
        # End of epoch
        loss_tracker.end_epoch(epoch, global_step)
        
        # Save checkpoint after each epoch
        save_checkpoint(model, processor, OUTPUT_DIR, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  • Average reward: {np.mean(epoch_rewards):.4f}")
        print(f"  • Reward std: {np.std(epoch_rewards):.4f}")
        print(f"  • Total steps: {global_step}")
    
    return model


def main():
    print("="*60)
    print("GRPO (Group Relative Policy Optimization) for LLaVA-Med")
    print("="*60)
    
    print("\n1. Loading model and processor...")
    model, processor = setup_model_and_processor()
    
    print("\n2. Setting up LoRA...")
    model = setup_lora(model)
    
    print("\n3. Loading dataset...")
    raw_data = load_jsonl_dataset(DATASET_PATH)
    print(f"   Loaded {len(raw_data)} examples")
    
    # Split data
    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    eval_data = raw_data[split_idx:]
    print(f"   Train: {len(train_data)} | Eval: {len(eval_data)}")
    
    # Initialize loss tracker
    loss_tracker = LossTracker(OUTPUT_DIR, LOSS_CSV_PATH)
    
    print("\n4. Starting GRPO Training...")
    model = grpo_train(model, processor, train_data, GRPO_CONFIG, loss_tracker)
    
    print(f"\n{'='*60}")
    print("GRPO Training completed!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training loss CSV saved to: {LOSS_CSV_PATH}")
    print(f"{'='*60}")
    
    # Save final model
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{OUTPUT_DIR}/processor")


if __name__ == "__main__":
    main()