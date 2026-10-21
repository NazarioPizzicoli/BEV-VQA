"""
Deterministic disk caching for LLM-as-a-Judge evaluations to prevent redundant API/GPU calls.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DiskJudgeCache:
    """Manages persistent caching of LLM Judge verdicts using SHA-256 keys."""

    def __init__(self, cache_file: str = "outputs/llm_judge_cache.json") -> None:
        self.cache_path = Path(cache_file)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache):,} judge cached evaluations from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Could not read judge cache from {self.cache_path}: {e}. Initializing fresh cache.")
                self._cache = {}

    def save(self) -> None:
        """Persists cache to disk."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save judge cache to {self.cache_path}: {e}")

    @staticmethod
    def compute_key(question: str, ground_truth: str, prediction: str, judge_model: str, benchmark: str) -> str:
        """Computes a unique SHA-256 hash for the evaluation instance."""
        payload = f"{benchmark}::{judge_model}::{question.strip()}::{ground_truth.strip()}::{prediction.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)
