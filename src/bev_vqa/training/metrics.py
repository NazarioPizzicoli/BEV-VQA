"""
Metrics and evaluation utilities for BEV-VQA.
"""

import re
import string
import logging
from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """
    Normalizza la stringa per una comparazione equa (Normalizes string for comparison).
    - Lowercase
    - Remove punctuation
    - Remove articles
    - Strip whitespace
    """
    if not isinstance(s, str):
        return ""
        
    s = s.lower().strip()
    
    # Rimuovi punteggiatura
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    # Rimuovi articoli (the, a, an) - basic English normalization
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    # Rimuovi spazi extra
    s = ' '.join(s.split())
    
    return s

def compute_nlg_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Calcola metriche NLG (BLEU, METEOR, ROUGE-L, CIDEr).
    Placeholder implementation - requires nlg-eval or pycocoevalcap packages.
    """
    logger.warning("compute_nlg_metrics is a placeholder. Install pycocoevalcap for actual metrics.")
    return {
        "BLEU-4": 0.0,
        "METEOR": 0.0,
        "ROUGE-L": 0.0,
        "CIDEr": 0.0
    }

@torch.no_grad()
def shuffle_test(
    model: torch.nn.Module, 
    dataset: Any, 
    tokenizer: Any, 
    device: torch.device, 
    n_samples: int = 100
) -> Dict[str, float]:
    """
    Esegue il test di shuffle spaziale per verificare che il modello non ignori l'input visivo.
    (Spatial shuffle test to verify model doesn't ignore visual input)
    """
    model.eval()
    logger.info(f"Esecuzione Shuffle Test su {n_samples} campioni (Running Shuffle Test)")
    
    # Create subset loader for n_samples
    subset = torch.utils.data.Subset(dataset, range(min(n_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=4, collate_fn=dataset.collate_fn)
    
    correct_base = 0
    correct_shuffled = 0
    total = 0
    
    for batch in loader:
        bev = batch["bev"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        answers = batch["answers"]
        
        # Base generation
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            base_out = model.generate(
                bev=bev, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Shuffle BEV features spatially
        # bev: [B, C, H, W]
        B, C, H, W = bev.shape
        bev_flat = bev.view(B, C, -1)
        # Random permutation for each batch element
        for i in range(B):
            idx = torch.randperm(H * W, device=device)
            bev_flat[i] = bev_flat[i, :, idx]
        bev_shuffled = bev_flat.view(B, C, H, W)
        
        # Shuffled generation
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            shuff_out = model.generate(
                bev=bev_shuffled, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        base_preds = tokenizer.batch_decode(base_out, skip_special_tokens=True)
        shuff_preds = tokenizer.batch_decode(shuff_out, skip_special_tokens=True)
        
        for bp, sp, gt in zip(base_preds, shuff_preds, answers):
            n_gt = normalize_answer(gt)
            if normalize_answer(bp) == n_gt:
                correct_base += 1
            if normalize_answer(sp) == n_gt:
                correct_shuffled += 1
            total += 1
            
    base_acc = correct_base / total if total > 0 else 0
    shuff_acc = correct_shuffled / total if total > 0 else 0
    gap = base_acc - shuff_acc
    
    logger.info(f"Shuffle Test Risultati (Results):")
    logger.info(f"  Base Acc:     {base_acc:.4f}")
    logger.info(f"  Shuffled Acc: {shuff_acc:.4f}")
    logger.info(f"  Gap:          {gap:.4f}")
    
    return {
        "correct_acc": base_acc,
        "shuffled_acc": shuff_acc,
        "gap": gap
    }
