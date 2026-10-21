"""Collate functions for BEV-VQA data loading."""

from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


def _batch_bev_tensors(batch: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    """Helper to unpack and batch BEV tensors.
    
    Args:
        batch (List[Dict[str, Any]]): A list of data samples.
        
    Returns:
        Optional[torch.Tensor]: The batched BEV tensors, or None if empty.
    """
    bevs = []
    for item in batch:
        b = item.get("bev")
        if b is not None:
            if b.dim() == 3:
                b = b.unsqueeze(0)
            bevs.append(b)
    return torch.cat(bevs, dim=0) if bevs else None


def collate_train(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    """Collate function for training. Right-pads the sequences.
    
    Args:
        batch (List[Dict[str, Any]]): List of samples from the dataset.
        pad_id (int): Padding token ID.
        
    Returns:
        Dict[str, Any]: Batched tensors.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Right-pad
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    bev_tensor = _batch_bev_tensors(batch)

    res = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "bev": bev_tensor,
    }
    if batch and "template_type" in batch[0]:
        res["template_types"] = [item.get("template_type", "") for item in batch]
    if batch and "answer_text" in batch[0]:
        res["answers"] = [item.get("answer_text", "") for item in batch]

    return res

def collate_eval(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    """Collate function for evaluation. Left-pads the sequences for generation.
    
    Args:
        batch (List[Dict[str, Any]]): List of samples from the dataset.
        pad_id (int): Padding token ID.
        
    Returns:
        Dict[str, Any]: Batched tensors and metadata.
    """
    prompt_ids = [item["prompt_ids"].flip(dims=[0]) for item in batch]
    attention_mask = [item["attention_mask"].flip(dims=[0]) for item in batch]
    
    # Left-pad by right-padding reversed sequences then reversing again
    prompt_ids_padded = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_id).flip(dims=[1])
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0).flip(dims=[1])
    
    bev_tensor = _batch_bev_tensors(batch)
    
    return {
        "prompt_ids": prompt_ids_padded,
        "attention_mask": attention_mask_padded,
        "bev": bev_tensor,
        "questions": [item.get("questions", "") for item in batch],
        "answers": [item.get("answers", "") for item in batch],
        "template_types": [item.get("template_type", "") for item in batch],
        "sample_tokens": [item.get("sample_token", "") for item in batch],
        "source_datasets": [item.get("source_dataset", "") for item in batch],
    }
