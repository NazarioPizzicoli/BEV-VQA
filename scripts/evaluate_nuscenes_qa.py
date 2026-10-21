#!/usr/bin/env python3
"""
Comprehensive NuScenes-QA Benchmark Evaluation Script.
"""

import argparse
from bev_vqa.evaluation.nuscenes_qa import evaluate_nuscenes_qa
from bev_vqa.utils.io import get_default_paths

_PATHS = get_default_paths()

def main():
    parser = argparse.ArgumentParser(description="NuScenes-QA Benchmark (Balanced & Natural Distribution)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--nuscenes-val-json", type=str, default=_PATHS.get('nuscenes_qa_val', ''))
    parser.add_argument("--bev-dir", type=str, default=_PATHS.get('bev_dir', ''))
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--spatial-shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--use-dual-pooling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-l2-pooling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2-pooling-gamma", type=float, default=1.0)
    parser.add_argument("--l2-pooling-learnable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--balanced-samples-per-cat", type=int, default=100)
    parser.add_argument("--natural-samples", type=int, default=500)
    parser.add_argument("--run-shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/nuscenes_qa_sota_report.json")
    parser.add_argument("--llm-judge", action=argparse.BooleanOptionalAction, default=False, help="Enable SOTA LLM-as-a-Judge evaluation")
    parser.add_argument("--judge-backend", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-samples", type=int, default=100)
    parser.add_argument("--pos-embed-type", type=str, default="1d_learned", choices=["1d_learned", "2d_cartesian_metric", "2d_polar_metric", "2d_separable_learned"])
    args = parser.parse_args()

    evaluate_nuscenes_qa(args)

if __name__ == "__main__":
    main()
