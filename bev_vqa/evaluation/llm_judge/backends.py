"""
Execution backends for LLM-as-a-Judge: Local HuggingFace models and local vLLM / REST endpoints.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import time
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


class BaseJudgeBackend(ABC):
    """Abstract interface for LLM Judge backends."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generates a text response given system and user prompts."""
        pass


class LocalHFBackend(BaseJudgeBackend):
    """Executes judge inference locally using HuggingFace Transformers."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 384,
    ) -> None:
        super().__init__(model_name=model_name_or_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_new_tokens = max_new_tokens

        logger.info(f"Initializing LocalHFBackend with model '{model_name_or_path}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic greedy decoding for reproducible judging
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


class APIJudgeBackend(BaseJudgeBackend):
    """Executes judge inference via local vLLM or standard REST endpoint."""

    def __init__(
        self,
        model_name: str = "local-judge",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    logger.error(f"API Judge error {response.status_code}: {response.text}")
                    break
            except Exception as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                time.sleep(1.0)

        raise RuntimeError(f"Failed to get response from API Judge ({self.model_name}) after 3 attempts.")
