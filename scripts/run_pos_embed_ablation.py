#!/usr/bin/env python3
"""
Master Orchestrator for Compartment 2: Spatial & Metric Positional Encoding Ablation Study.

Compares 4 positional encoding inductive biases on the optimal Multiscale Peak Projector (8x4 grid, 32 tokens):
1. '1d_learned': 1D unconstrained learned embedding sequence (Baseline from Compartment 1)
2. '2d_cartesian_metric': Explicit 2D Fourier sinusoidal metric coordinates [X, Y] in [-50m, +50m]
3. '2d_polar_metric': Explicit 2D Fourier sinusoidal range-azimuth coordinates [r, theta]
4. '2d_separable_learned': 2D Separable grid learned embedding (row_embed [8x1] x col_embed [1x4])

Across benchmarks:
- NuScenes-QA (Spatial reasoning, exist, count, object, status, comparison)
- DriveLM (Perception, prediction, planning, behavior)

Evaluation includes:
- Classical metrics (Accuracy, BLEU-4, CIDEr, ROUGE-L, METEOR)
- SOTA LLM-as-a-Judge (Factual Accuracy, Driving Safety, Semantic Correctness, Hazard Rate)
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List

from bev_vqa.constants import Colors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pos_embed_ablation")

POS_EMBED_TYPES = [
    "1d_learned",
    "2d_cartesian_metric",
    "2d_polar_metric",
    "2d_separable_learned",
]

BENCHMARKS = [
    "nuscenes_qa",
    "drivelm",
]

DEFAULT_STAGE1 = "checkpoints/sota/stage1/stage1_projector_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator for Compartment 2: Spatial & Metric Positional Encoding Ablation")
    parser.add_argument(
        "--pos-embed",
        type=str,
        default="all",
        choices=["all"] + POS_EMBED_TYPES,
        help="Positional encoding strategy to run (default: 'all')",
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
        default=DEFAULT_STAGE1,
        help="Path to pre-trained stage 1 projector weights for warm-start",
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        type=str,
        default="checkpoints/ablation_pos_embed",
        help="Base directory to save checkpoints",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="outputs/ablation_pos_embed",
        help="Base directory to save evaluation reports",
    )
    parser.add_argument(
        "--reuse-compartment1-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse multiscale_peak checkpoint from Compartment 1 for 1d_learned baseline if present",
    )
    parser.add_argument(
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SOTA LLM-as-a-Judge evaluation during post-training benchmark",
    )
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate with LLM-as-a-Judge",
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

    pe_types = POS_EMBED_TYPES if args.pos_embed == "all" else [args.pos_embed]
    benchmarks = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    base_ckpt = Path(args.base_checkpoint_dir)
    base_out = Path(args.base_output_dir)
    base_ckpt.mkdir(parents=True, exist_ok=True)
    base_out.mkdir(parents=True, exist_ok=True)

    summary_file = base_out / "pos_embed_summary.json"
    summary_data = {}
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary_data = json.load(f)
        except Exception:
            summary_data = {}

    total_runs = len(pe_types) * len(benchmarks)
    run_idx = 0

    print(f"\n{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  COMPARTMENT 2: POSITIONAL ENCODING ABLATION STUDY ({total_runs} RUNS){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Projector:     multiscale_peak (8x4, 32 tokens){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  PE Strategies: {', '.join(pe_types)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Benchmarks:    {', '.join(benchmarks)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  LLM Judge:     {args.llm_judge} ({args.judge_samples} samples){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Warm-Start:    {args.stage1_projector}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Dry Run:       {args.dry_run}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}\n")

    for pe_type in pe_types:
        for bench in benchmarks:
            run_idx += 1
            run_name = f"{pe_type}_{bench}"
            logger.info(f"{Colors.BOLD}{Colors.YELLOW}[{run_idx}/{total_runs}] Starting Run: PE={pe_type} on {bench}...{Colors.RESET}")

            ckpt_dir = base_ckpt / pe_type / bench
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            rep_path = base_out / f"{pe_type}_{bench}_train_report.json"
            eval_rep_path = base_out / f"{pe_type}_{bench}_eval_report.json"

            # Check if we can reuse Compartment 1 multiscale_peak for 1d_learned baseline
            reused = False
            c1_ckpt = Path(f"checkpoints/ablation_projector/multiscale_peak/{bench}/best_model")
            if args.reuse_compartment1_baseline and pe_type == "1d_learned" and not args.dry_run:
                if c1_ckpt.exists() and (ckpt_dir / "best_model").exists():
                    logger.info(f"Reusing existing baseline {pe_type} on {bench} from Compartment 1.")
                    reused = True
                elif c1_ckpt.exists() and not (ckpt_dir / "best_model").exists():
                    logger.info(f"Copying Compartment 1 baseline checkpoint {c1_ckpt} to {ckpt_dir / 'best_model'}...")
                    shutil.copytree(c1_ckpt, ckpt_dir / "best_model", dirs_exist_ok=True)
                    reused = True

            # 1. Training Phase
            if not reused:
                if bench == "nuscenes_qa":
                    cmd = [
                        py_exec, "scripts/train_sota_nuscenes_boost.py",
                        "--arch-type", "multiscale_peak",
                        "--pos-embed-type", pe_type,
                        "--spatial-shape", "8", "4",
                        "--stage1-projector", args.stage1_projector,
                        "--output-dir", str(ckpt_dir),
                        "--report-path", str(rep_path),
                        "--save-steps", "5000",
                    ]
                else:  # drivelm
                    cmd = [
                        py_exec, "scripts/train_sota_drivelm_v2.py",
                        "--arch-type", "multiscale_peak",
                        "--pos-embed-type", pe_type,
                        "--spatial-shape", "8", "4",
                        "--stage1-projector", args.stage1_projector,
                        "--output-dir", str(ckpt_dir),
                        "--report-path", str(rep_path),
                        "--save-steps", "5000",
                    ]

                if args.dry_run:
                    cmd.append("--dry-run")

                train_success = run_command(cmd, f"Training multiscale_peak ({pe_type}) on {bench}")
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
                        "--arch-type", "multiscale_peak",
                        "--pos-embed-type", pe_type,
                        "--spatial-shape", "8", "4",
                        "--report-path", str(eval_rep_path),
                    ]
                else:
                    eval_cmd = [
                        py_exec, "scripts/evaluate_drivelm_nlg.py",
                        "--checkpoint", str(best_model_path),
                        "--arch-type", "multiscale_peak",
                        "--pos-embed-type", pe_type,
                        "--spatial-shape", "8", "4",
                        "--report-path", str(eval_rep_path),
                    ]

                if args.llm_judge:
                    eval_cmd.extend([
                        "--llm-judge",
                        "--judge-backend", "local",
                        "--judge-samples", str(args.judge_samples),
                    ])

                eval_success = run_command(eval_cmd, f"Evaluating multiscale_peak ({pe_type}) on {bench}")
                if eval_success and eval_rep_path.exists():
                    try:
                        with open(eval_rep_path) as ef:
                            eval_metrics = json.load(ef)
                        summary_data[run_name] = eval_metrics
                        with open(summary_file, "w") as sf:
                            json.dump(summary_data, sf, indent=2)
                    except Exception as e:
                        logger.warning(f"Could not read eval metrics from {eval_rep_path}: {e}")

    logger.info(f"{Colors.BOLD}{Colors.GREEN}Compartment 2 positional encoding ablation complete.{Colors.RESET}")


if __name__ == "__main__":
    main()
