"""Dataset classes for BEV-VQA."""

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from bev_vqa.constants import BEV_TOKEN, VQA_SYSTEM_PROMPT
from bev_vqa.training.metrics import normalize_answer

logger = logging.getLogger(__name__)

PRETRAIN_SYSTEM_PROMPT = "You are given a bird's-eye-view feature map of the driving scene. Describe what you see."

def build_tokenizer(llm_name: str) -> PreTrainedTokenizer:
    """Builds the tokenizer and adds the special BEV token.
    
    Args:
        llm_name (str): Name or path of the LLM tokenizer.
        
    Returns:
        PreTrainedTokenizer: The configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Add the special token for BEV features
    tokenizer.add_special_tokens({"additional_special_tokens": [BEV_TOKEN]})
    return tokenizer

class BaseBEVDataset(Dataset):
    """Base dataset for BEV-VQA."""
    
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
        system_prompt: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize the BaseBEVDataset.

        Args:
            json_path (Union[str, Path]): Path to the JSON data file.
            bev_dir (Union[str, Path]): Path to the BEV features directory.
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            split (str, optional): Dataset split. Defaults to "train".
            num_bev_tokens (int, optional): Number of BEV tokens. Defaults to 1.
            fraction (float, optional): Fraction of dataset to use. Defaults to 1.0.
            load_bev (bool, optional): Whether to load BEV features. Defaults to True.
            cache_dir (Optional[Union[str, Path]], optional): Directory for caching. Defaults to None.
            system_prompt (Optional[str], optional): System prompt. Defaults to None.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
        """
        self.json_path = Path(json_path)
        bev_path = Path(bev_dir)
        self.raw_bev_dir = bev_path
        if (bev_path / split).exists():
            self.bev_dir = bev_path / split
        else:
            self.bev_dir = bev_path
        self.tokenizer = tokenizer
        self.split = split
        self.num_bev_tokens = num_bev_tokens
        self.fraction = fraction
        self.load_bev = load_bev
        self.system_prompt = system_prompt or VQA_SYSTEM_PROMPT
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.json_path.parent) / ".cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from the source. Must be implemented by subclasses.
        
        Returns:
            List[Dict[str, Any]]: A list of loaded data samples.
        """
        raise NotImplementedError
        
    def __len__(self) -> int:
        """Get the length of the dataset.
        
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)
        
    def _find_bev_path(self, sample_token: str) -> Optional[Path]:
        """Resolves BEV feature path across direct, train, val, and parent paths.
        
        Args:
            sample_token (str): The sample token to find.
            
        Returns:
            Optional[Path]: The path to the BEV feature if found, else None.
        """
        candidates = [
            self.bev_dir / f"{sample_token}.pt",
            self.bev_dir / "train" / f"{sample_token}.pt",
            self.bev_dir / "val" / f"{sample_token}.pt",
            self.bev_dir.parent / f"{sample_token}.pt",
            self.bev_dir.parent / "train" / f"{sample_token}.pt",
            self.bev_dir.parent / "val" / f"{sample_token}.pt",
            self.raw_bev_dir / f"{sample_token}.pt",
            self.raw_bev_dir / "train" / f"{sample_token}.pt",
            self.raw_bev_dir / "val" / f"{sample_token}.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _get_available_bev_tokens(self) -> set:
        """Finds all available sample tokens with .pt files in candidate directories."""
        if not self.load_bev:
            return set()
        candidate_dirs = {
            self.bev_dir,
            self.bev_dir / "train",
            self.bev_dir / "val",
            self.bev_dir.parent / "train",
            self.bev_dir.parent / "val",
            self.raw_bev_dir,
            self.raw_bev_dir / "train",
            self.raw_bev_dir / "val",
        }
        tokens = set()
        for d in candidate_dirs:
            if d.is_dir():
                tokens.update(p.stem for p in d.glob("*.pt"))
        return tokens


    def _load_bev(self, sample_token: str) -> Optional[torch.Tensor]:
        """Loads the BEV feature for a given sample token.
        
        Args:
            sample_token (str): The sample token.
            
        Returns:
            Optional[torch.Tensor]: The loaded BEV feature tensor, or zeros if not found.
        """
        if not self.load_bev:
            return None
            
        bev_path = self._find_bev_path(sample_token)
        if bev_path is None or not bev_path.exists():
            logger.warning(f"BEV feature not found for token: {sample_token} (searched around {self.bev_dir})")
            return torch.zeros((128, 200, 200), dtype=torch.float16)
            
        data = torch.load(bev_path, map_location="cpu", weights_only=True)
        feat = data["features_fused"]
        if feat.dim() == 4 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        return feat

    _load_bev_features = _load_bev

    def _prepare_sample(self, system_prompt: str, user_text: str, answer_text: Optional[str]) -> Dict[str, Any]:
        """Prepares a single sample for the model.
        
        Args:
            system_prompt (str): The system prompt.
            user_text (str): The user input text.
            answer_text (Optional[str]): The target answer text.
            
        Returns:
            Dict[str, Any]: A dictionary containing the prepared tokenized inputs.
        """
        bev_placeholder = BEV_TOKEN * self.num_bev_tokens if self.load_bev else ""
        
        # Structure prompt with chat template
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

        if answer_text is not None:
            ans_str = str(answer_text)
            full_text = prompt + ans_str + self.tokenizer.eos_token
            tokenized_full = self.tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
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
    """Dataset for Stage 1 (BEV-text alignment)."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads and filters the pretraining data.
        
        Returns:
            List[Dict[str, Any]]: A list of valid descriptions for pretraining.
        """
        cache_file = self.cache_dir / f"pretrain_{self.split}_{self.fraction}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                    if len(cached) > 0:
                        logger.info(f"Loading data from cache: {cache_file} ({len(cached)} samples)")
                        return cached
                    logger.warning(f"Cache file {cache_file} was empty, recomputing.")
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}, recomputing.")
                
        logger.info(f"Loading JSON data from {self.json_path}")
        with open(self.json_path, "r") as f:
            full_data = json.load(f)
            
        descriptions = full_data.get("descriptions", [])
        if self.load_bev:
            existing_tokens = self._get_available_bev_tokens()
            if existing_tokens:
                valid_descriptions = [d for d in descriptions if d.get("sample_token") in existing_tokens]
                logger.info(f"BEV filter: {len(valid_descriptions)}/{len(descriptions)} valid samples.")
                if len(valid_descriptions) > 0:
                    descriptions = valid_descriptions
            else:
                logger.warning(f"No BEV files found around {self.bev_dir}, keeping all descriptions.")


        if self.fraction < 1.0:
            num_samples = int(len(descriptions) * self.fraction)
            descriptions = random.sample(descriptions, num_samples)

        logger.info(f"Loaded {len(descriptions)} descriptions.")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(descriptions, f)
        except Exception as e:
            logger.warning(f"Could not save cache to {cache_file}: {e}")

        return descriptions
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets a single pretraining sample.
        
        Args:
            idx (int): The index of the sample.
            
        Returns:
            Dict[str, Any]: The loaded and processed sample.
        """
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
    """Dataset for Stage 2 (VQA fine-tuning)."""

    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads and filters the VQA data.
        
        Returns:
            List[Dict[str, Any]]: A list of valid questions for VQA.
        """
        cache_file = self.cache_dir / f"vqa_{self.json_path.stem}_{self.split}_{self.fraction}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                    if len(cached) > 0:
                        logger.info(f"Loading data from cache: {cache_file} ({len(cached)} samples)")
                        return cached
                    logger.warning(f"Cache file {cache_file} was empty, recomputing.")
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_file}: {e}, recomputing.")
                
        logger.info(f"Loading JSON data from {self.json_path}")
        with open(self.json_path, "r") as f:
            full_data = json.load(f)
            
        questions = full_data.get("questions", [])
        questions_with_split = [q for q in questions if "split" in q]
        if questions_with_split:
            filtered = [q for q in questions if q.get("split") == self.split]
            if len(filtered) > 0:
                questions = filtered
            else:
                logger.warning(
                    f"No question with split='{self.split}' found (out of {len(questions)} questions). "
                    f"Keeping all questions from the file."
                )

        if self.load_bev:
            existing_tokens = self._get_available_bev_tokens()
            if existing_tokens:
                valid_questions = [q for q in questions if q.get("sample_token") in existing_tokens]
                logger.info(f"BEV filter: {len(valid_questions)}/{len(questions)} questions with valid BEV.")
                if len(valid_questions) > 0:
                    questions = valid_questions
                else:
                    logger.warning("No question has a token matching the found BEV files; keeping all questions.")
            else:
                logger.warning(f"No BEV files (.pt) found around {self.bev_dir}. Keeping all {len(questions)} questions.")


        if self.fraction < 1.0:
            num_samples = int(len(questions) * self.fraction)
            questions = random.sample(questions, num_samples)

        logger.info(f"Loaded {len(questions)} questions.")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(questions, f)
        except Exception as e:
            logger.warning(f"Could not save cache to {cache_file}: {e}")

        return questions
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Gets a single VQA sample.
        
        Args:
            idx (int): The index of the sample.
            
        Returns:
            Dict[str, Any]: The loaded and processed sample.
        """
        item = self.data[idx]
        sample_token = item["sample_token"]
        question = item["question"]
        answer = item["answer"]
        
        bev_features = self._load_bev_features(sample_token)
        
        sample = self._prepare_sample(
            system_prompt=self.system_prompt,
            user_text=question,
            answer_text=answer
        )
        
        sample["bev"] = bev_features
        sample["template_type"] = item.get("template_type", "")
        sample["answer_text"] = answer
        if self.split != "train":
            sample.update({
                "sample_token": sample_token,
                "questions": question,
                "answers": answer,
                "source_dataset": item.get("source_dataset", ""),
            })
            
        return sample
