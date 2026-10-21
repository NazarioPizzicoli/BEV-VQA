"""I/O utilities for loading BEV features and model checkpoints."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def get_default_paths() -> Dict[str, str]:
    """Returns default dataset, annotation, and feature paths from env or standard locations."""
    bev_dir = os.environ.get(
        "BEV_VQA_BEV_DIR",
        "/media/nazario.pizzicoli/Datos/dataset/bev_features",
    )
    datasets_dir = os.environ.get(
        "BEV_VQA_DATASETS_DIR",
        "/media/nazario.pizzicoli/Datos/vqa_datasets/unified",
    )
    return {
        "bev_dir": bev_dir,
        "nuscenes_qa_train": f"{datasets_dir}/nuscenes_qa_train.json",
        "nuscenes_qa_val": f"{datasets_dir}/nuscenes_qa_val.json",
        "drivelm_train": f"{datasets_dir}/drivelm_train.json",
        "drivelm_val": f"{datasets_dir}/drivelm_val.json",
        "omnidrive_train": f"{datasets_dir}/omnidrive_descriptions_train.json",
        "omnidrive_val": f"{datasets_dir}/omnidrive_descriptions_val.json",
    }


def load_bev_features(
    sample_token: str,
    bev_dir: Union[Path, str],
    device: Union[str, torch.device] = "cpu",
) -> Optional[torch.Tensor]:
    """Loads pre-extracted BEV features for a sample token.

    Searches across split subdirectories ('val', 'train') and root bev_dir.
    Returns:
        Tensor of shape [1, C, H, W] if found, None otherwise.
    """
    bev_dir = Path(bev_dir)
    filename = f"{sample_token}.pt"

    for d in [bev_dir / "val", bev_dir / "train", bev_dir]:
        path = d / filename
        if path.exists():
            data = torch.load(path, map_location=device, weights_only=True)
            bev = data["features_fused"]
            return bev.unsqueeze(0) if bev.dim() == 3 else bev
    return None

