"""
Reward calculation utilities for GRPO training.
Simple implementations - no external dependencies.
"""

import math
from typing import Dict, Tuple


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Compute ROUGE-L score using longest common subsequence."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
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
    
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(cand_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    lcs_len = lcs_length(cand_tokens, ref_tokens)
    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Compute BLEU score with smoothing."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    precisions = []
    for n in range(1, min(max_n + 1, len(cand_tokens) + 1)):
        cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        
        if len(cand_ngrams) == 0:
            continue
        
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        
        matches = 0
        for ng in cand_ngrams:
            if ref_counts.get(ng, 0) > 0:
                matches += 1
                ref_counts[ng] -= 1
        
        # Add-1 smoothing
        precision = (matches + 1) / (len(cand_ngrams) + 1)
        precisions.append(precision)
    
    if len(precisions) == 0:
        return 0.0
    
    log_precision = sum(math.log(p) for p in precisions) / len(precisions)
    
    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(cand_tokens))
    
    return bp * math.exp(log_precision)


def compute_reward(generated: str, reference: str) -> float:
    """
    Compute reward combining ROUGE-L, BLEU, length and format penalties.
    
    Weights:
    - ROUGE-L: 1.0
    - BLEU: 0.5
    - Length: 0.3
    - Format: 0.2
    """
    if not generated or not generated.strip() or not reference:
        return -1.0
    
    total = 0.0
    
    # ROUGE-L (weight: 1.0)
    total += compute_rouge_l(generated, reference)
    
    # BLEU (weight: 0.5)
    total += compute_bleu(generated, reference) * 0.5
    
    # Length penalty (weight: 0.3)
    gen_len = len(generated.split())
    if gen_len < 10:
        total -= 0.3
    elif gen_len > 200:
        total -= 0.15
    else:
        total += 0.06
    
    # Format quality (weight: 0.2)
    gen = generated.strip()
    if gen[0].isupper():
        total += 0.02
    if gen[-1] in '.!?':
        total += 0.02
    
    return total
