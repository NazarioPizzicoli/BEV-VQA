"""
Category-Balanced Sampler for NuScenes-QA Training (Intervention 2).

Ensures balanced representation across question categories:
- count: 30%
- status: 30%
- object: 20%
- exist: 10%
- comparison: 10%
"""

from collections import defaultdict
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY_RATIOS: Dict[str, float] = {
    "count": 0.30,
    "status": 0.30,
    "object": 0.20,
    "exist": 0.10,
    "comparison": 0.10,
}

DRIVELM_CATEGORY_RATIOS: Dict[str, float] = {
    "perception": 0.35,
    "prediction": 0.30,
    "planning": 0.30,
    "behavior": 0.05,
}


class CategoryBalancedSampler(Sampler[int]):
    """Category-Balanced Sampler that samples items from the dataset according
    to predefined target category proportions.

    Attributes:
        target_ratios (Dict[str, float]): Dict mapping category name to target probability.
        seed (int): Random seed.
        num_samples (int): Total number of samples to yield per epoch.
    """

    def __init__(
        self,
        dataset_data: Union[List[Dict[str, Any]], Any],
        target_ratios: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the CategoryBalancedSampler.

        Args:
            dataset_data (Union[List[Dict[str, Any]], Any]): The dataset data.
            target_ratios (Optional[Dict[str, float]], optional): The target category ratios. Defaults to None.
            num_samples (Optional[int], optional): Total number of samples to yield. Defaults to None.
            seed (int, optional): Random seed. Defaults to 42.
        """
        items = dataset_data.data if hasattr(dataset_data, "data") else dataset_data
        self.target_ratios = target_ratios or DEFAULT_CATEGORY_RATIOS
        self.seed = seed
        self.num_samples = num_samples if num_samples is not None else len(items)

        # Group indices by category
        self.cat_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(items):
            cat = (
                item.get("template_type")
                or item.get("question_type")
                or item.get("category")
                or item.get("type")
                or "unknown"
            )
            cat = str(cat).lower().strip()
            self.cat_indices[cat].append(idx)

        # Normalize known categories with respect to target_ratios (case-insensitive)
        total_weight = sum(self.target_ratios.values())
        self.norm_ratios = {k.lower().strip(): v / total_weight for k, v in self.target_ratios.items()}

        # Calculate weights per sample (inverse frequency weighting per category)
        self.sample_weights = torch.zeros(len(items), dtype=torch.double)
        assigned_count = 0

        for cat, target_prob in self.norm_ratios.items():
            indices = self.cat_indices.get(cat, [])
            if len(indices) > 0:
                # Each sample in this category receives weight = target_prob / count
                w = target_prob / len(indices)
                self.sample_weights[indices] = w
                assigned_count += len(indices)
            else:
                logger.warning(f"Category '{cat}' requested in target ratios but missing in dataset!")

        # Any remaining / unknown categories
        remaining_indices = [
            idx for idx, w in enumerate(self.sample_weights) if w == 0.0
        ]
        if remaining_indices:
            # Assign a minimal residual distributed weight
            residual_prob = 0.02
            w = residual_prob / len(remaining_indices)
            self.sample_weights[remaining_indices] = w
            logger.info(f"Assigned residual weight to {len(remaining_indices)} samples outside main categories.")

        # Normalize overall weights
        sum_w = self.sample_weights.sum()
        if sum_w > 0:
            self.sample_weights /= sum_w

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Statistics for logging
        stats = {cat: len(idxs) for cat, idxs in self.cat_indices.items()}
        logger.info(f"CategoryBalancedSampler initialized: {stats} | Target ratios: {self.norm_ratios}")

    def set_epoch(self, epoch: int) -> None:
        """Allows variability of the generator between different epochs.
        
        Args:
            epoch (int): The current epoch number.
        """
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self) -> Iterator[int]:
        """Get an iterator over the sampled indices.
        
        Returns:
            Iterator[int]: Iterator of sampled indices.
        """
        if len(self.sample_weights) == 0 or self.num_samples == 0:
            return iter([])
        # Sampling with replacement based on exact target distribution
        sampled_indices = torch.multinomial(
            self.sample_weights,
            self.num_samples,
            replacement=True,
            generator=self.generator,
        ).tolist()
        return iter(sampled_indices)

    def __len__(self) -> int:
        """Get the total number of samples yielded per epoch.
        
        Returns:
            int: Number of samples.
        """
        return self.num_samples


class DriveLMCategoryBalancedSampler(CategoryBalancedSampler):
    """Category-Balanced Sampler specifically configured for DriveLM datasets:
    - perception: 35%
    - prediction: 30%
    - planning: 30%
    - behavior: 5%
    """

    def __init__(
        self,
        dataset_data: Union[List[Dict[str, Any]], Any],
        target_ratios: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the DriveLMCategoryBalancedSampler.

        Args:
            dataset_data (Union[List[Dict[str, Any]], Any]): The dataset data.
            target_ratios (Optional[Dict[str, float]], optional): The target category ratios. Defaults to None.
            num_samples (Optional[int], optional): Total number of samples to yield. Defaults to None.
            seed (int, optional): Random seed. Defaults to 42.
        """
        ratios = target_ratios if target_ratios is not None else DRIVELM_CATEGORY_RATIOS
        super().__init__(
            dataset_data=dataset_data,
            target_ratios=ratios,
            num_samples=num_samples,
            seed=seed,
        )
