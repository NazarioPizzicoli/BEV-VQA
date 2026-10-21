#!/usr/bin/env python3
"""
Master Orchestration Script for Projector Architecture Ablation Study.

Runs the 4 unified projector architectures:
1. 'dual_pooling' (Arch 1: CBAM-style Dual Pooling)
2. 'multiscale_peak' (Arch 2: FPN-style Multi-Scale Peak Injection)
3. 'hybrid_sparse' (Arch 3: SparseDrive-style 32 Grid + 8 Salient Peak Tokens = 40 Tokens)
4. 'question_aware' (Arch 4: InstructBLIP-style Language-Guided Cross-Attention)

Across both benchmarks:
- NuScenes-QA (3 epochs / 5000 steps, Concise Spatial Reasoning)
- DriveLM (5 epochs / 5000 steps, Generative Driving Decisions)

Warm-start: Reuses checkpoints/sota/stage1/stage1_projector_best.pt for shared layers.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from bev_vqa.constants import Colors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation_orchestrator")

ARCHITECTURES = [
    "dual_pooling",
    "multiscale_peak",
    "hybrid_sparse",
    "question_aware",
]

BENCHMARKS = [
    "nuscenes_qa",
    "drivelm",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator for Projector Architecture Ablation")
    parser.add_argument(
        "--arch",
        type=str,
        default="all",
        choices=["all"] + ARCHITECTURES,
        help="Architecture to train/evaluate (default: 'all')",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all"] + BENCHMARKS,
        help="Benchmark to run (default: 'all')",
    )
    parser.add_argument(
        "--stage1-projector",
        type=str,
        default="checkpoints/sota/stage1/stage1_projector_best.pt",
        help="Path to pre-trained stage 1 projector weights for warm-start",
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        type=str,
        default="checkpoints/ablation_projector",
        help="Base directory to save checkpoints",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="outputs/ablation_projector",
        help="Base directory to save evaluation reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast 2-step dry run for each task to verify end-to-end execution",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation",
    )
    return parser.parse_args()


def run_command(cmd: List[str], desc: str) -> bool:
    logger.info(f"{Colors.BOLD}{Colors.CYAN}>>> Executing: {desc}{Colors.RESET}")
    logger.info(f"Command: {' '.join(cmd)}")
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start
    if res.returncode == 0:
        logger.info(f"{Colors.GREEN}✓ {desc} completed in {elapsed:.1f}s{Colors.RESET}\n")
        return True
    else:
        logger.error(f"{Colors.RED}✗ {desc} failed with returncode {res.returncode}{Colors.RESET}\n")
        return False


def main():
    args = parse_args()
    py_exec = sys.executable

    archs = ARCHITECTURES if args.arch == "all" else [args.arch]
    benchmarks = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    base_ckpt = Path(args.base_checkpoint_dir)
    base_out = Path(args.base_output_dir)
    base_ckpt.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)

    summary_file = base_out / "ablation_summary.json"
    summary_data = {}
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary_data = json.load(f)
        except Exception:
            summary_data = {}

    total_runs = len(archs) * len(benchmarks)
    run_idx = 0

    print(f"\n{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  PROJECTOR ARCHITECTURE ABLATION STUDY ({total_runs} RUNS){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Architectures: {', '.join(archs)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Benchmarks:    {', '.join(benchmarks)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Warm-Start:    {args.stage1_projector}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Dry Run:       {args.dry_run}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}\n")

    for arch in archs:
        for bench in benchmarks:
            run_idx += 1
            run_name = f"{arch}_{bench}"
            logger.info(f"{Colors.BOLD}{Colors.YELLOW}[{run_idx}/{total_runs}] Starting Run: {arch} on {bench}...{Colors.RESET}")

            ckpt_dir = base_ckpt / arch / bench
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            rep_path = base_out / f"{arch}_{bench}_train_report.json"
            eval_rep_path = base_out / f"{arch}_{bench}_eval_report.json"

            # 1. Training Phase
            if bench == "nuscenes_qa":
                cmd = [
                    py_exec, "scripts/train_sota_nuscenes_boost.py",
                    "--arch-type", arch,
                    "--spatial-shape", "8", "4",
                    "--stage1-projector", args.stage1_projector,
                    "--output-dir", str(ckpt_dir),
                    "--report-path", str(rep_path),
                ]
            else:  # drivelm
                cmd = [
                    py_exec, "scripts/train_sota_drivelm_v2.py",
                    "--arch-type", arch,
                    "--spatial-shape", "8", "4",
                    "--stage1-projector", args.stage1_projector,
                    "--output-dir", str(ckpt_dir),
                    "--report-path", str(rep_path),
                ]

            if args.dry_run:
                cmd.append("--dry-run")

            train_success = run_command(cmd, f"Training {arch} on {bench}")
            if not train_success and not args.dry_run:
                logger.error(f"Run {run_name} failed. Continuing to next experiment.")
                continue

            # 2. Evaluation Phase
            best_model_path = ckpt_dir / "best_model"
            if not args.skip_eval and best_model_path.exists() and not args.dry_run:
                if bench == "nuscenes_qa":
                    eval_cmd = [
                        py_exec, "scripts/evaluate_nuscenes_qa.py",
                        "--checkpoint", str(best_model_path),
                        "--arch-type", arch,
                        "--spatial-shape", "8", "4",
                        "--report-path", str(eval_rep_path),
                    ]
                else:
                    eval_cmd = [
                        py_exec, "scripts/evaluate_drivelm_nlg.py",
                        "--checkpoint", str(best_model_path),
                        "--arch-type", arch,
                        "--spatial-shape", "8", "4",
                        "--report-path", str(eval_rep_path),
                    ]
                eval_success = run_command(eval_cmd, f"Evaluating {arch} on {bench}")
                if eval_success and eval_rep_path.exists():
                    try:
                        with open(eval_rep_path) as ef:
                            eval_metrics = json.load(ef)
                        summary_data[run_name] = eval_metrics
                        with open(summary_file, "w") as sf:
                            json.dump(summary_data, sf, indent=2)
                    except Exception as e:
                        logger.warning(f"Could not read eval metrics from {eval_rep_path}: {e}")

    logger.info(f"{Colors.BOLD}{Colors.GREEN}Ablation orchestrator execution complete.{Colors.RESET}")


if __name__ == "__main__":
    main()
