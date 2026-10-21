#!/usr/bin/env python3
"""
CLI script to evaluate autonomous driving VQA predictions using SOTA LLM-as-a-Judge.
Supports offline evaluation of existing reports and prediction dumps without re-running BEV-VLM.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

from bev_vqa.constants import Colors
from bev_vqa.evaluation.llm_judge import (
    APIJudgeBackend,
    DiskJudgeCache,
    LLMJudge,
    LocalHFBackend,
    print_judge_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_with_llm_judge")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions using SOTA LLM-as-a-Judge")
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path to an existing evaluation report JSON (containing sample_previews or predictions)",
    )
    parser.add_argument(
        "--predictions-json",
        type=str,
        default=None,
        help="Path to a dedicated predictions JSON list containing items with question, ground_truth, predicted",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["drivelm", "nuscenes_qa"],
        help="Benchmark type ('drivelm' or 'nuscenes_qa'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Execution backend: 'local' (HuggingFace) or 'api' (OpenAI/Gemini-compatible)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Judge model name or HF path (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for API backend",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key for API backend",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all available in input)",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="outputs/llm_judge_cache.json",
        help="Persistent disk cache file path",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSON path to save the judge evaluation report",
    )
    return parser.parse_args()


def load_samples(args) -> Tuple[str, List[Dict[str, Any]]]:
    """Loads prediction samples from either a report or dedicated JSON."""
    benchmark = args.benchmark
    samples: List[Dict[str, Any]] = []

    if args.predictions_json:
        p = Path(args.predictions_json)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            samples = data.get("predictions", data.get("sample_predictions", data.get("sample_previews", [])))
        if not benchmark:
            benchmark = "drivelm" if "drivelm" in str(p).lower() else "nuscenes_qa"

    elif args.report_path:
        p = Path(args.report_path)
        with open(p, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        if not benchmark:
            benchmark = "drivelm" if "drivelm" in str(p).lower() else "nuscenes_qa"

        if "predictions" in report_data:
            samples = report_data["predictions"]
        elif "sample_previews" in report_data:
            samples = report_data["sample_previews"]
        elif "sample_predictions" in report_data:
            samples = report_data["sample_predictions"]
        elif "balanced_evaluation" in report_data and "sample_predictions" in report_data["balanced_evaluation"]:
            samples = report_data["balanced_evaluation"]["sample_predictions"]
        else:
            raise ValueError(f"Could not find predictions list in report: {p}")

    else:
        raise ValueError("Must provide either --report-path or --predictions-json.")

    if not benchmark:
        benchmark = "drivelm"

    logger.info(f"Loaded {len(samples)} prediction samples for benchmark '{benchmark}'.")
    return benchmark, samples


def main():
    args = parse_args()
    benchmark, samples = load_samples(args)

    if not samples:
        logger.error("No valid samples found to evaluate.")
        sys.exit(1)

    # Initialize Backend
    if args.backend == "local":
        backend = LocalHFBackend(model_name_or_path=args.judge_model)
    else:
        backend = APIJudgeBackend(
            model_name=args.judge_model,
            api_key=args.api_key,
            base_url=args.api_base_url,
        )

    # Initialize Cache and Judge
    cache = DiskJudgeCache(cache_file=args.cache_file)
    judge = LLMJudge(backend=backend, benchmark=benchmark, cache=cache)

    # Run Evaluation
    summary, results = judge.evaluate_dataset(samples=samples, max_samples=args.max_samples)

    # Print Formatted Results
    print_judge_summary(summary)

    # Save Report
    if args.output_path:
        out_p = Path(args.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        final_payload = {
            "summary": summary,
            "detailed_verdicts": [r.to_dict() for r in results],
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"{Colors.GREEN}Judge report saved to: {out_p}{Colors.RESET}")


if __name__ == "__main__":
    main()
