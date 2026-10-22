#!/usr/bin/env python3
"""
DriveLM Generative Evaluation (NLG Metrics) CLI Wrapper.
"""

import argparse
from bev_vqa.evaluation.drivelm_nlg import evaluate_drivelm_nlg
from bev_vqa.utils.io import get_default_paths

_PATHS = get_default_paths()

def main():
    parser = argparse.ArgumentParser(description="DriveLM Generative NLG Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--drivelm-val-json", type=str, default=_PATHS.get('drivelm_val', ''))
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="multiscale_peak")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--spatial-shape", type=int, nargs=2, default=[8, 4], metavar=("H", "W"))
    parser.add_argument("--use-dual-pooling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-l2-pooling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2-pooling-gamma", type=float, default=1.0)
    parser.add_argument("--l2-pooling-learnable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-samples", type=int, default=1600)
    parser.add_argument("--bev-dir", type=str, default=_PATHS.get('bev_dir', ''))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--report-path", type=str, default="outputs/drivelm_eval_report.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-judge", action=argparse.BooleanOptionalAction, default=False, help="Enable SOTA LLM-as-a-Judge evaluation")
    parser.add_argument("--judge-backend", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-samples", type=int, default=100)
    parser.add_argument("--pos-embed-type", type=str, default="2d_polar_metric", choices=["1d_learned", "2d_cartesian_metric", "2d_polar_metric", "2d_separable_learned"])
    args = parser.parse_args()

    evaluate_drivelm_nlg(args)

if __name__ == "__main__":
    main()
