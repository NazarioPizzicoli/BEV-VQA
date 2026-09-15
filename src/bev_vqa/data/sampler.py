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
import math
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
    """
    Category-Balanced Sampler that samples items from the dataset according
    to predefined target category proportions.

    Attributes:
        dataset_data: List of dicts containing 'template_type' for each sample,
                      or a Dataset instance containing `.data`.
        target_ratios: Dict mapping category name -> target probability.
        num_samples: Total number of samples to yield per epoch.
    """

    def __init__(
        self,
        dataset_data: Union[List[Dict[str, Any]], Any],
        target_ratios: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        items = dataset_data.data if hasattr(dataset_data, "data") else dataset_data
        self.target_ratios = target_ratios or DEFAULT_CATEGORY_RATIOS
        self.seed = seed
        self.num_samples = num_samples if num_samples is not None else len(items)

        # Raggruppa gli indici per categoria
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

        # Normalizza le categorie note rispetto a target_ratios (case-insensitive)
        total_weight = sum(self.target_ratios.values())
        self.norm_ratios = {k.lower().strip(): v / total_weight for k, v in self.target_ratios.items()}

        # Calcola pesi per campione (inverse frequency weighting per categoria)
        self.sample_weights = torch.zeros(len(items), dtype=torch.double)
        assigned_count = 0

        for cat, target_prob in self.norm_ratios.items():
            indices = self.cat_indices.get(cat, [])
            if len(indices) > 0:
                # Ogni campione in questa categoria riceve peso = target_prob / count
                w = target_prob / len(indices)
                self.sample_weights[indices] = w
                assigned_count += len(indices)
            else:
                logger.warning(f"Categoria '{cat}' richiesta nei target ratios ma assente nel dataset!")

        # Eventuali categorie rimanenti / sconosciute
        remaining_indices = [
            idx for idx, w in enumerate(self.sample_weights) if w == 0.0
        ]
        if remaining_indices:
            # Assegna un peso residuale minimo distribuito
            residual_prob = 0.02
            w = residual_prob / len(remaining_indices)
            self.sample_weights[remaining_indices] = w
            logger.info(f"Assegnato peso residuale a {len(remaining_indices)} campioni fuori dalle categorie principali.")

        # Normalizza pesi complessivi
        sum_w = self.sample_weights.sum()
        if sum_w > 0:
            self.sample_weights /= sum_w

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Statistiche per logging
        stats = {cat: len(idxs) for cat, idxs in self.cat_indices.items()}
        logger.info(f"CategoryBalancedSampler inizializzato: {stats} | Target ratios: {self.norm_ratios}")

    def set_epoch(self, epoch: int):
        """Permette la variabilità del generatore tra epoche differenti."""
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self) -> Iterator[int]:
        if len(self.sample_weights) == 0 or self.num_samples == 0:
            return iter([])
        # Campionamento con rimpiazzo basato sulla distribuzione esatta target
        sampled_indices = torch.multinomial(
            self.sample_weights,
            self.num_samples,
            replacement=True,
            generator=self.generator,
        ).tolist()
        return iter(sampled_indices)

    def __len__(self) -> int:
        return self.num_samples


class DriveLMCategoryBalancedSampler(CategoryBalancedSampler):
    """
    Category-Balanced Sampler specifically configured for DriveLM datasets:
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
    ):
        ratios = target_ratios if target_ratios is not None else DRIVELM_CATEGORY_RATIOS
        super().__init__(
            dataset_data=dataset_data,
            target_ratios=ratios,
            num_samples=num_samples,
            seed=seed,
        )

