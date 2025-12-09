import torch
import torch.nn.functional as F
import json
import os
import sys
import csv
from pathlib import Path
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig
from PIL import Image
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed

# Standard NLP metrics for reward computation
# Using simple implementations to avoid NumPy binary incompatibility issues
ROUGE_AVAILABLE = False
BLEU_AVAILABLE = False
BERTSCORE_AVAILABLE = False

def simple_rouge_l(candidate: str, reference: str) -> float:
    """Simple ROUGE-L implementation using longest common subsequence."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        # Use space-efficient LCS
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev, curr = curr, prev
        return prev[n]
    
    # Tokenize by splitting on whitespace
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(cand_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    lcs_len = lcs_length(cand_tokens, ref_tokens)
    precision = lcs_len / len(cand_tokens) if len(cand_tokens) > 0 else 0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def simple_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Simple BLEU score implementation with smoothing."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, len(cand_tokens) + 1)):
        # Get n-grams
        cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        
        if len(cand_ngrams) == 0:
            continue
            
        # Count matches
        ref_ngram_counts = {}
        for ng in ref_ngrams:
            ref_ngram_counts[ng] = ref_ngram_counts.get(ng, 0) + 1
        
        matches = 0
        for ng in cand_ngrams:
            if ref_ngram_counts.get(ng, 0) > 0:
                matches += 1
                ref_ngram_counts[ng] -= 1
        
        # Add smoothing (add-1)
        precision = (matches + 1) / (len(cand_ngrams) + 1)
        precisions.append(precision)
    
    if len(precisions) == 0:
        return 0.0
    
    # Geometric mean of precisions
    import math
    log_precision = sum(math.log(p) for p in precisions) / len(precisions)
    
    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(cand_tokens))
    
    return bp * math.exp(log_precision)

# Add LLaVA-Med to path
sys.path.insert(0, '/mnt/workspace/LLaVA-Med')
from llava.model import LlavaMistralForCausalLM

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
MODEL_ID = "/mnt/workspace/CorTEX/.models/llava-med-v1.5-mistral-7b"
DATASET_PATH = "/mnt/workspace/CorTEX/RL/test/llava_med_eval_qa50_qa.jsonl"
IMAGE_DIR = "/mnt/workspace/CorTEX/RL/test/images"
OUTPUT_DIR = "/mnt/workspace/CorTEX/RL/grpo_llava_med_output"
LOSS_CSV_PATH = os.path.join(OUTPUT_DIR, "training_loss.csv")

# GRPO Hyperparameters
GRPO_CONFIG = {
    "num_generations": 4,        # K: number of responses per prompt
    "temperature": 0.7,          # Sampling temperature
    "max_new_tokens": 128,       # Max tokens to generate (reduced for faster/more stable multi-GPU)
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
    
    def __init__(self, data, tokenizer, image_dir):
        self.data = data
        self.tokenizer = tokenizer
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


def setup_model_and_tokenizer(accelerator):
    """Load model and tokenizer using LLaVA-Med SDK"""
    if accelerator.is_main_process:
        print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    if accelerator.is_main_process:
        print(f"Loading LLaVA-Med model (fp16)...")
    # Use LLaVA-Med SDK - don't use device_map with accelerate
    model = LlavaMistralForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer


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


class RewardCalculator:
    """
    Comprehensive reward calculator using standard NLP metrics.
    
    Combines multiple signals:
    - ROUGE scores (recall-oriented, good for summarization/QA)
    - BLEU scores (precision-oriented, good for translation-like tasks)
    - Length penalties
    - Repetition penalties
    - Format quality checks
    """
    
    def __init__(self, use_rouge=True, use_bleu=True, use_bertscore=False):
        # Use simple implementations (no external dependencies)
        self.use_rouge = use_rouge
        self.use_bleu = use_bleu
        self.use_bertscore = False  # Not implemented
    
    def compute_rouge(self, generated: str, reference: str) -> dict:
        """Compute ROUGE-L score using simple LCS implementation"""
        if not self.use_rouge or not reference:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        rouge_l = simple_rouge_l(generated, reference)
        # Approximate rouge1 and rouge2 from rougeL (simplified)
        return {
            'rouge1': min(1.0, rouge_l * 1.1),  # Unigram typically higher
            'rouge2': max(0.0, rouge_l * 0.9),  # Bigram typically lower
            'rougeL': rouge_l,
        }
    
    def compute_bleu(self, generated: str, reference: str) -> float:
        """Compute BLEU score using simple n-gram implementation"""
        if not self.use_bleu or not reference:
            return 0.0
        
        return simple_bleu(generated, reference)
    
    def compute_length_penalty(self, generated: str, min_len=10, max_len=200) -> float:
        """
        Length penalty based on DeepMind's approach.
        Penalizes too short or too long responses.
        """
        gen_len = len(generated.split())
        
        if gen_len < min_len:
            # Penalize short responses more harshly
            return -1.0 * (1 - gen_len / min_len)
        elif gen_len > max_len:
            # Gentle penalty for long responses
            return -0.5 * min(1.0, (gen_len - max_len) / max_len)
        else:
            # Optimal length range
            return 0.2
    
    def compute_repetition_penalty(self, generated: str) -> float:
        """
        Penalize repetitive text (common failure mode in RL).
        Based on distinct n-gram ratio.
        """
        words = generated.lower().split()
        if len(words) == 0:
            return -1.0
        
        # Unigram diversity
        unigram_ratio = len(set(words)) / len(words)
        
        # Bigram diversity
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_ratio = len(set(bigrams)) / max(len(bigrams), 1)
        
        # Combined diversity score
        diversity = 0.5 * unigram_ratio + 0.5 * bigram_ratio
        
        # Convert to penalty (diversity of 1.0 = no penalty)
        return (diversity - 0.5) * 1.0  # Range: [-0.5, 0.5]
    
    def compute_format_quality(self, generated: str) -> float:
        """
        Check for format issues common in LLM outputs.
        """
        penalty = 0.0
        
        # Penalize empty or whitespace-only
        if not generated.strip():
            return -2.0
        
        # Penalize if starts with special tokens
        if generated.strip().startswith(('<', '[', '{', '|', '#')):
            penalty -= 0.3
        
        # Penalize excessive special characters
        special_ratio = sum(1 for c in generated if not c.isalnum() and c not in ' .,!?') / max(len(generated), 1)
        if special_ratio > 0.3:
            penalty -= 0.3
        
        # Reward proper sentence structure (starts with capital, ends with punctuation)
        if generated[0].isupper():
            penalty += 0.1
        if generated.rstrip()[-1] in '.!?':
            penalty += 0.1
        
        return penalty
    
    def __call__(self, generated: str, reference: str = None) -> tuple:
        """
        Compute comprehensive reward.
        
        Returns:
            total_reward: float - combined reward score
            reward_details: dict - breakdown of individual components
        """
        details = {}
        total = 0.0
        
        # 1. ROUGE scores (weight: 1.5)
        if reference:
            rouge_scores = self.compute_rouge(generated, reference)
            rouge_reward = (
                0.3 * rouge_scores['rouge1'] + 
                0.3 * rouge_scores['rouge2'] + 
                0.4 * rouge_scores['rougeL']
            ) * 1.5
            details['rouge'] = rouge_scores
            details['rouge_reward'] = rouge_reward
            total += rouge_reward
        
        # 2. BLEU score (weight: 1.0)
        if reference:
            bleu = self.compute_bleu(generated, reference)
            bleu_reward = bleu * 1.0
            details['bleu'] = bleu
            details['bleu_reward'] = bleu_reward
            total += bleu_reward
        
        # 3. Length penalty (weight: 0.5)
        length_penalty = self.compute_length_penalty(generated)
        details['length_penalty'] = length_penalty
        total += length_penalty * 0.5
        
        # 4. Repetition penalty (weight: 0.5)
        rep_penalty = self.compute_repetition_penalty(generated)
        details['repetition_penalty'] = rep_penalty
        total += rep_penalty * 0.5
        
        # 5. Format quality (weight: 0.3)
        format_quality = self.compute_format_quality(generated)
        details['format_quality'] = format_quality
        total += format_quality * 0.3
        
        details['total'] = total
        return total, details


# Global reward calculator instance
reward_calculator = None


def get_reward_calculator():
    """Get or create global reward calculator"""
    global reward_calculator
    if reward_calculator is None:
        reward_calculator = RewardCalculator(
            use_rouge=True,
            use_bleu=True,
            use_bertscore=False  # Disabled by default (slow)
        )
    return reward_calculator


def compute_reward(generated_text, reference_answer, tokenizer=None):
    """
    Compute reward for a generated response using standard NLP metrics.
    
    Uses:
    - ROUGE (rouge-score library)
    - BLEU (nltk library)
    - Length and repetition penalties
    - Format quality checks
    """
    calculator = get_reward_calculator()
    reward, details = calculator(generated_text, reference_answer)
    return reward


def generate_responses(model, tokenizer, prompt, image, num_generations, temperature, max_new_tokens):
    """Generate K responses for a prompt using sampling (LLaVA-Med compatible)"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    responses = []
    log_probs_list = []
    
    for _ in range(num_generations):
        with torch.no_grad():
            # Manual generation loop for LLaVA-Med compatibility
            generated_ids = input_ids.clone()
            seq_log_probs = []
            
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    use_cache=False,
                )
                
                # Get logits for next token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Get log prob of sampled token
                log_prob = F.log_softmax(next_token_logits, dim=-1)
                token_log_prob = log_prob.gather(-1, next_token).squeeze(-1)
                seq_log_probs.append(token_log_prob.item())
                
                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Decode only the generated part
        generated_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(generated_text)
        
        # Sum log probs for this sequence
        log_probs_list.append(sum(seq_log_probs) if seq_log_probs else 0.0)
    
    return responses, log_probs_list


def compute_grpo_loss(model, tokenizer, prompt, image, responses, rewards, old_log_probs, beta, clip_range):
    """
    Compute GRPO loss for a batch of responses.
    
    GRPO uses group-relative advantages:
    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    
    Loss = -sum(advantage_i * log_prob_i) + beta * KL_penalty
    """
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Normalize rewards within the group (GRPO key insight)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    if rewards_tensor.std() > 0:
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    else:
        advantages = rewards_tensor - rewards_tensor.mean()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    
    # Initialize total_loss as a tensor on the correct device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_valid_responses = 0
    
    for i, (response, advantage, old_log_prob) in enumerate(zip(responses, advantages, old_log_probs)):
        # Tokenize the response
        response_ids = tokenizer(
            response, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(device)
        
        if response_ids.shape[1] == 0:
            continue
        
        # Create full sequence (prompt + response)
        full_input_ids = torch.cat([input_ids, response_ids], dim=1)
        
        # Create labels (mask prompt, only compute loss on response)
        labels = full_input_ids.clone()
        labels[:, :input_ids.shape[1]] = -100
        
        # Forward pass
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=torch.ones_like(full_input_ids),
            labels=labels,
            use_cache=False,
        )
        
        # The loss from the model is the negative log likelihood
        # Weight it by the advantage (GRPO core idea)
        # Higher advantage = we want to increase this response's probability
        # So we weight the loss inversely (or use the loss directly for gradient descent)
        
        # Simple GRPO: weight the cross-entropy loss by negative advantage
        # This increases probability of high-reward responses
        weighted_loss = outputs.loss * (1.0 - advantage * 0.1)  # Scale advantage contribution
        
        # KL penalty to prevent divergence from reference policy
        # Approximated using the difference in log probs
        response_len = response_ids.shape[1]
        current_avg_log_prob = -outputs.loss  # Average log prob per token
        old_avg_log_prob = old_log_prob / max(response_len, 1)
        kl_penalty = beta * torch.abs(current_avg_log_prob - old_avg_log_prob)
        
        total_loss = total_loss + weighted_loss + kl_penalty
        num_valid_responses += 1
    
    if num_valid_responses == 0:
        # Return a zero tensor that still has gradients
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / num_valid_responses


def save_checkpoint(model, tokenizer, output_dir, epoch):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"✓ Checkpoint saved to {checkpoint_dir}")


def grpo_train(model, tokenizer, train_data, config, loss_tracker, accelerator):
    """
    Main GRPO training loop with multi-GPU support via accelerate.
    
    For each prompt:
    1. Generate K responses
    2. Compute rewards for each response
    3. Compute group-relative advantages
    4. Update policy using GRPO objective
    """
    
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Starting GRPO (Group Relative Policy Optimization) Training")
        print("="*60)
        print(f"  • Generations per prompt (K): {config['num_generations']}")
        print(f"  • Temperature: {config['temperature']}")
        print(f"  • Learning rate: {config['learning_rate']}")
        print(f"  • KL penalty (beta): {config['beta']}")
        print(f"  • Clip range: {config['clip_range']}")
        print(f"  • Number of GPUs: {accelerator.num_processes}")
        print("="*60 + "\n")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Custom collate function to handle PIL Images
    def custom_collate(batch):
        """Custom collate that keeps PIL Images as lists instead of stacking"""
        return {
            'prompt': [item['prompt'] for item in batch],
            'image': [item['image'] for item in batch],  # Keep as list of PIL Images
            'reference_answer': [item['reference_answer'] for item in batch],
            'question_id': [item['question_id'] for item in batch],
        }
    
    # Create dataset and dataloader
    dataset = GRPODataset(train_data, tokenizer, IMAGE_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate
    )
    
    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    global_step = 0
    model.train()
    
    for epoch in range(1, config['num_epochs'] + 1):
        if accelerator.is_main_process:
            print(f"\n{'='*40} Epoch {epoch}/{config['num_epochs']} {'='*40}")
        
        epoch_rewards = []
        accumulated_loss = 0.0
        
        # Only show progress bar on main process
        if accelerator.is_main_process:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            progress_bar = dataloader
        
        for batch_idx, batch in enumerate(progress_bar):
            prompt = batch['prompt'][0]  # Batch size is 1
            image = batch['image'][0]
            reference = batch['reference_answer'][0]
            
            # Step 1: Generate K responses (use unwrapped model for generation)
            unwrapped_model = accelerator.unwrap_model(model)
            responses, old_log_probs = generate_responses(
                unwrapped_model, tokenizer, prompt, image,
                num_generations=config['num_generations'],
                temperature=config['temperature'],
                max_new_tokens=config['max_new_tokens']
            )
            
            # Step 2: Compute rewards for each response
            rewards = [
                compute_reward(resp, reference, tokenizer) 
                for resp in responses
            ]
            epoch_rewards.extend(rewards)
            
            # Step 3 & 4: Compute GRPO loss and update
            loss = compute_grpo_loss(
                model, tokenizer, prompt, image,
                responses, rewards, old_log_probs,
                beta=config['beta'],
                clip_range=config['clip_range']
            )
            
            # Gradient accumulation with accelerate
            loss = loss / config['gradient_accumulation_steps']
            accelerator.backward(loss)
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging (only on main process)
                if accelerator.is_main_process and global_step % config['logging_steps'] == 0:
                    avg_reward = np.mean(rewards)
                    loss_tracker.log_step(
                        epoch=epoch + batch_idx/len(dataloader),
                        step=global_step,
                        loss=accumulated_loss * config['gradient_accumulation_steps'],
                        avg_reward=avg_reward
                    )
                    accumulated_loss = 0.0
            
            # Update progress bar (only on main process)
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    'loss': f'{loss.item()*config["gradient_accumulation_steps"]:.4f}',
                    'avg_reward': f'{np.mean(rewards):.2f}'
                })
        
        # Synchronize before epoch end
        accelerator.wait_for_everyone()
        
        # End of epoch (only on main process)
        if accelerator.is_main_process:
            loss_tracker.end_epoch(epoch, global_step)
            
            # Save checkpoint after each epoch
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped_model, tokenizer, OUTPUT_DIR, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  • Average reward: {np.mean(epoch_rewards):.4f}")
            print(f"  • Reward std: {np.std(epoch_rewards):.4f}")
            print(f"  • Total steps: {global_step}")
    
    return model


def main():
    # Initialize accelerator for multi-GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=GRPO_CONFIG['gradient_accumulation_steps'],
        mixed_precision='fp16',
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    if accelerator.is_main_process:
        print("="*60)
        print("GRPO (Group Relative Policy Optimization) for LLaVA-Med")
        print("="*60)
        print(f"\n🚀 Using {accelerator.num_processes} GPUs for training!")
    
    if accelerator.is_main_process:
        print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(accelerator)
    
    if accelerator.is_main_process:
        print("\n2. Setting up LoRA...")
    model = setup_lora(model)
    
    if accelerator.is_main_process:
        print("\n3. Loading dataset...")
    raw_data = load_jsonl_dataset(DATASET_PATH)
    if accelerator.is_main_process:
        print(f"   Loaded {len(raw_data)} examples")
    
    # Split data
    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    eval_data = raw_data[split_idx:]
    if accelerator.is_main_process:
        print(f"   Train: {len(train_data)} | Eval: {len(eval_data)}")
    
    # Initialize loss tracker (only on main process)
    loss_tracker = LossTracker(OUTPUT_DIR, LOSS_CSV_PATH) if accelerator.is_main_process else None
    
    if accelerator.is_main_process:
        print("\n4. Starting GRPO Training...")
    model = grpo_train(model, tokenizer, train_data, GRPO_CONFIG, loss_tracker, accelerator)
    
    # Save final model (only on main process)
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("GRPO Training completed!")
        print(f"Model saved to: {OUTPUT_DIR}")
        print(f"Training loss CSV saved to: {LOSS_CSV_PATH}")
        print(f"{'='*60}")
        
        # Save final model
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(f"{OUTPUT_DIR}/final_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")


if __name__ == "__main__":
    main()