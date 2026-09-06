"""
Training utilities for BEV-VQA.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bev_vqa.training.metrics import normalize_answer

logger = logging.getLogger(__name__)

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum: int = 1,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """
    Esegue un'epoca di training (Trains for one epoch).
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        bev = batch["bev"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(bev=bev, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            
        scaler.scale(loss).backward()
        
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        loss_val = loss.item() * grad_accum
        total_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        
    avg_loss = total_loss / len(loader)
    elapsed = time.time() - start_time
    logger.info(f"Train Epoch: avg_loss={avg_loss:.4f}, time={elapsed:.2f}s")
    
    return avg_loss, elapsed


@torch.no_grad()
def val_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> float:
    """
    Calcola la loss di validazione (Computes validation loss).
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    
    pbar = tqdm(loader, desc="Validation")
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
            
        bev = batch["bev"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(bev=bev, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        total_loss += loss.item()
        batches += 1
        pbar.set_postfix({"loss": f"{total_loss/batches:.4f}"})
        
    avg_loss = total_loss / batches if batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    
    return avg_loss


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 32,
    max_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Valuta l'accuratezza (Evaluates exact match accuracy and per-template breakdown).
    """
    model.eval()
    correct = 0
    total = 0
    
    # Per-template accuracy tracking
    template_correct = {}
    template_total = {}
    
    samples = []
    
    pbar = tqdm(loader, desc="Evaluating Accuracy")
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
            
        bev = batch["bev"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        # Actual answers
        answers = batch.get("answers", [])
        template_types = batch.get("template_type", ["unknown"] * len(bev))
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            generated_ids = model.generate(
                bev=bev,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for pred, gt, t_type in zip(decoded_preds, answers, template_types):
            norm_pred = normalize_answer(pred)
            norm_gt = normalize_answer(gt)
            
            is_correct = (norm_pred == norm_gt)
            if is_correct:
                correct += 1
            total += 1
            
            template_correct[t_type] = template_correct.get(t_type, 0) + (1 if is_correct else 0)
            template_total[t_type] = template_total.get(t_type, 0) + 1
            
            samples.append({
                "pred": pred,
                "norm_pred": norm_pred,
                "gt": gt,
                "norm_gt": norm_gt,
                "template_type": t_type,
                "correct": is_correct
            })
            
        pbar.set_postfix({"acc": f"{correct/total:.4f}"})
        
    overall_acc = correct / total if total > 0 else 0.0
    
    breakdown = {k: template_correct[k] / template_total[k] for k in template_total}
    
    logger.info(f"Evaluation Accuracy: {overall_acc:.4f}")
    for k, v in breakdown.items():
        logger.info(f"  {k}: {v:.4f}")
        
    return {
        "accuracy": overall_acc,
        "breakdown": breakdown,
        "samples": samples
    }
