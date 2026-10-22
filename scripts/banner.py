#!/usr/bin/env python3
"""
Aesthetic & Informative ASCII Banner for BEV-VLM.
Displays hardware status, architecture configuration, mounted storage, and CLI hints.
"""

import os
from pathlib import Path
import sys

# ANSI Colors
CYAN = "\033[96m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_status(condition: bool, true_text: str = "MOUNTED", false_text: str = "NOT FOUND") -> str:
    if condition:
        return f"{GREEN}{BOLD}● {true_text}{RESET}"
    return f"{YELLOW}{DIM}○ {false_text}{RESET}"


def get_hardware_info():
    try:
        import torch
        pytorch_ver = torch.__version__
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_ver = torch.version.cuda
            hw_str = f"{gpu_name} ({vram_gb:.2f} GB VRAM)"
            compute_str = f"CUDA {cuda_ver} | PyTorch {pytorch_ver}"
        else:
            hw_str = "CPU Mode (No CUDA acceleration)"
            compute_str = f"PyTorch {pytorch_ver}"
    except ImportError:
        hw_str = "PyTorch not available"
        compute_str = "N/A"
    return hw_str, compute_str


def main():
    hw_str, compute_str = get_hardware_info()

    # Robust directory resolution (container & host)
    default_bev = "/data/dataset/bev_features"
    if not Path(default_bev).exists() and Path("/media/nazario.pizzicoli/Datos/dataset/bev_features").exists():
        default_bev = "/media/nazario.pizzicoli/Datos/dataset/bev_features"

    default_datasets = "/data/vqa_datasets/unified"
    if not Path(default_datasets).exists() and Path("/media/nazario.pizzicoli/Datos/vqa_datasets/unified").exists():
        default_datasets = "/media/nazario.pizzicoli/Datos/vqa_datasets/unified"

    bev_dir = Path(os.environ.get("BEV_VQA_BEV_DIR", default_bev))
    datasets_dir = Path(os.environ.get("BEV_VQA_DATASETS_DIR", default_datasets))
    ckpt_dir = Path("checkpoints/ablation_pos_embed/2d_polar_metric/drivelm/best_model")

    bev_ok = bev_dir.exists()
    datasets_ok = datasets_dir.exists()
    ckpt_ok = ckpt_dir.exists()

    banner = f"""{CYAN}{BOLD}
  ██████╗ ███████╗██╗   ██╗      ██╗   ██╗██╗     ███╗   ███╗
  ██╔══██╗██╔════╝██║   ██║      ██║   ██║██║     ████╗ ████║
  ██████╔╝█████╗  ██║   ██║      ██║   ██║██║     ██╔████╔██║
  ██╔══██╗██╔══╝  ╚██╗ ██╔╝      ╚██╗ ██╔╝██║     ██║╚██╔╝██║
  ██████╔╝███████╗ ╚████╔╝   ██╗  ╚████╔╝ ███████╗██║ ╚═╝ ██║
  ╚═════╝ ╚══════╝  ╚═══╝    ╚═╝   ╚═══╝  ╚══════╝╚═╝     ╚═╝{RESET}
{BLUE}{BOLD}  Bird's-Eye-View Vision-Language Model for Autonomous Driving{RESET}
{DIM}  ─────────────────────────────────────────────────────────────────────────────{RESET}
  {BOLD}SYSTEM ENVIRONMENT:{RESET}
    {DIM}├─{RESET} {BOLD}Hardware:{RESET}      {hw_str}
    {DIM}├─{RESET} {BOLD}Compute:{RESET}       {compute_str}
    {DIM}└─{RESET} {BOLD}Winning Model:{RESET} {CYAN}multiscale_peak{RESET} + {PURPLE}2d_polar_metric{RESET} (Qwen2.5-3B)

  {BOLD}STORAGE & VOLUMES:{RESET}
    {DIM}├─{RESET} {BOLD}BEV Features:{RESET}  {bev_dir} [{format_status(bev_ok)}]
    {DIM}├─{RESET} {BOLD}VQA Datasets:{RESET}  {datasets_dir} [{format_status(datasets_ok)}]
    {DIM}└─{RESET} {BOLD}Checkpoints:{RESET}   {ckpt_dir} [{format_status(ckpt_ok, "READY")}]

  {BOLD}QUICK WORKFLOWS:{RESET}
    {CYAN}interactive{RESET}      python scripts/inference_interactive.py --random --terminal-viz
    {CYAN}eval-drivelm{RESET}     python scripts/evaluate_drivelm_nlg.py --llm-judge --judge-samples 100
    {CYAN}eval-nuscenes{RESET}    python scripts/evaluate_nuscenes_qa.py --balanced-samples-per-cat 20
    {CYAN}run tests{RESET}        pytest tests/
{DIM}  ─────────────────────────────────────────────────────────────────────────────{RESET}
"""
    print(banner)


if __name__ == "__main__":
    main()
