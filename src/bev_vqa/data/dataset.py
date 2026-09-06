import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

PRETRAIN_SYSTEM_PROMPT = "You are given a bird's-eye-view feature map of the driving scene. Describe what you see."
VQA_SYSTEM_PROMPT = "You are a driving assistant. You are given a bird's-eye-view feature map of the scene around the ego-vehicle. Answer the question about the scene concisely."
BEV_TOKEN = "<|bev|>"

def build_tokenizer(llm_name: str) -> PreTrainedTokenizer:
    """
    Builds the tokenizer and adds the special BEV token.
    
    Args:
        llm_name: Name or path of the LLM tokenizer.
        
    Returns:
        The configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Aggiungi il token speciale per le feature BEV
    tokenizer.add_special_tokens({"additional_special_tokens": [BEV_TOKEN]})
    return tokenizer

def normalize_answer(text: str) -> str:
    """
    Normalizes the answer string for evaluation.
    
    Args:
        text: The string to normalize.
        
    Returns:
        The normalized string.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class BaseBEVDataset(Dataset):
    """Base dataset per BEV-VQA."""
    
    def __init__(
        self,
        json_path: Union[str, Path],
        bev_dir: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        num_bev_tokens: int = 1,
        fraction: float = 1.0,
        load_bev: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.json_path = Path(json_path)
        self.bev_dir = Path(bev_dir) / split
        self.tokenizer = tokenizer
        self.split = split
        self.num_bev_tokens = num_bev_tokens
        self.fraction = fraction
        self.load_bev = load_bev
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.json_path.parent) / ".cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
        
    def __len__(self) -> int:
        return len(self.data)
        
    def _load_bev_features(self, sample_token: str) -> Optional[torch.Tensor]:
        if not self.load_bev:
            return None
            
        bev_path = self.bev_dir / f"{sample_token}.pt"
        if not bev_path.exists():
            logger.warning(f"BEV feature not found: {bev_path}")
            return torch.zeros((1, 128, 200, 200), dtype=torch.float16)
            
        data = torch.load(bev_path, map_location="cpu", weights_only=True)
        return data["features_fused"]

    def _prepare_sample(self, system_prompt: str, user_text: str, answer_text: str) -> Dict[str, Any]:
        bev_placeholder = BEV_TOKEN * self.num_bev_tokens if self.load_bev else ""
        
        # Struttura prompt con chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{bev_placeholder}\n{user_text}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        result = {
            "prompt_ids": tokenized_prompt.input_ids[0],
            "attention_mask": tokenized_prompt.attention_mask[0],
        }

        if answer_text:
            full_text = prompt + answer_text + self.tokenizer.eos_token
            tokenized_full = self.tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=220,
            )
            input_ids = tokenized_full.input_ids[0]
            attention_mask = tokenized_full.attention_mask[0]
            labels = input_ids.clone()
            prompt_len = min(tokenized_prompt.input_ids.shape[1], input_ids.shape[0])
            labels[:prompt_len] = -100

            result.update({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        return result


class BEVPretrainDataset(BaseBEVDataset):
    """Dataset per lo Stage 1 (Allineamento BEV-testo)."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        cache_file = self.cache_dir / f"pretrain_{self.split}_{self.fraction}.pkl"
        if cache_file.exists():
            logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
                
        logger.info(f"Loading JSON data from {self.json_path}")
        with open(self.json_path, "r") as f:
            full_data = json.load(f)
            
        descriptions = full_data.get("descriptions", [])
        # Filtra solo campioni con file BEV esistente
        valid_descriptions = [d for d in descriptions if (self.bev_dir / f"{d['sample_token']}.pt").exists()]
        logger.info(f"Filtro BEV: {len(valid_descriptions)}/{len(descriptions)} campioni validi.")
        descriptions = valid_descriptions

        if self.fraction < 1.0:
            num_samples = int(len(descriptions) * self.fraction)
            descriptions = random.sample(descriptions, num_samples)

        logger.info(f"Loaded {len(descriptions)} descriptions.")

        with open(cache_file, "wb") as f:
            pickle.dump(descriptions, f)

        return descriptions
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        sample_token = item["sample_token"]
        description = item["description"]
        
        bev_features = self._load_bev_features(sample_token)
        
        sample = self._prepare_sample(
            system_prompt=PRETRAIN_SYSTEM_PROMPT,
            user_text="",
            answer_text=description
        )
        
        sample["bev"] = bev_features
        if self.split != "train":
            sample.update({
                "sample_token": sample_token,
                "answers": [description],
            })
            
        return sample


class BEVQADataset(BaseBEVDataset):
    """Dataset per lo Stage 2 (VQA fine-tuning)."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        cache_file = self.cache_dir / f"vqa_{self.json_path.stem}_{self.split}_{self.fraction}.pkl"
        if cache_file.exists():
            logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
                
        logger.info(f"Loading JSON data from {self.json_path}")
        with open(self.json_path, "r") as f:
            full_data = json.load(f)
            
        questions = full_data.get("questions", [])
        questions = [q for q in questions if q.get("split", self.split) == self.split]
        valid_questions = [q for q in questions if (self.bev_dir / f"{q['sample_token']}.pt").exists()]
        logger.info(f"Filtro BEV: {len(valid_questions)}/{len(questions)} domande con BEV valido.")
        questions = valid_questions

        if self.fraction < 1.0:
            num_samples = int(len(questions) * self.fraction)
            questions = random.sample(questions, num_samples)

        logger.info(f"Loaded {len(questions)} questions.")

        with open(cache_file, "wb") as f:
            pickle.dump(questions, f)

        return questions
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        sample_token = item["sample_token"]
        question = item["question"]
        answer = item["answer"]
        
        bev_features = self._load_bev_features(sample_token)
        
        sample = self._prepare_sample(
            system_prompt=VQA_SYSTEM_PROMPT,
            user_text=question,
            answer_text=answer
        )
        
        sample["bev"] = bev_features
        if self.split != "train":
            sample.update({
                "sample_token": sample_token,
                "questions": question,
                "answers": answer,
                "template_type": item.get("template_type", ""),
                "source_dataset": item.get("source_dataset", ""),
            })
            
        return sample


class BEVMixedDataset(Dataset):
    """Combines multiple BEVQADatasets with a specific ratio."""
    
    def __init__(self, datasets: List[BEVQADataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        self.total_len = self.cumulative_lengths[-1]
        
    def __len__(self) -> int:
        return self.total_len
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                local_idx = idx if i == 0 else idx - self.cumulative_lengths[i-1]
                return self.datasets[dataset_idx][local_idx]
        raise IndexError("Index out of bounds")
