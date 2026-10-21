"""
Core LLM-as-a-Judge orchestrator with robust JSON extraction, caching, and aggregate statistics.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from bev_vqa.constants import Colors
from bev_vqa.evaluation.llm_judge.backends import BaseJudgeBackend
from bev_vqa.evaluation.llm_judge.cache import DiskJudgeCache
from bev_vqa.evaluation.llm_judge.rubrics import (
    DRIVELM_JUDGE_SYSTEM_PROMPT,
    NUSCENES_QA_JUDGE_SYSTEM_PROMPT,
    build_drivelm_prompt,
    build_nuscenes_qa_prompt,
)

logger = logging.getLogger(__name__)


def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    """Robustly extracts and parses JSON payload from LLM responses.
    
    Handles markdown code fences, embedded JSON objects, and regex fallbacks.
    """
    clean_text = raw_text.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(clean_text)
    except Exception:
        pass

    # 2. Extract from markdown code fences ```json ... ```
    fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(fence_pattern, clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Extract outermost curly braces
    brace_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        # Clean trailing commas before close braces
        candidate = re.sub(r",\s*([\}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4. Regex attribute fallback extraction
    fallback: Dict[str, Any] = {"critique": clean_text[:200]}
    
    # Check for semantic correctness / factual accuracy
    for field in ["semantic_correctness", "factual_accuracy", "driving_safety", "causal_reasoning"]:
        num_match = re.search(rf'["\']?{field}["\']?\s*[:=]?\s*(\d+)', clean_text, re.IGNORECASE)
        if num_match:
            fallback[field] = int(num_match.group(1))

    # Check for boolean is_pass / is_correct
    for bool_field in ["is_pass", "is_correct"]:
        b_match = re.search(rf'["\']?{bool_field}["\']?\s*[:=]?\s*(true|false)', clean_text, re.IGNORECASE)
        if b_match:
            fallback[bool_field] = (b_match.group(1).lower() == "true")

    # Check for verdict
    v_match = re.search(r'["\']?verdict["\']?\s*[:=]?\s*["\']?([A-Z_]+)["\']?', clean_text, re.IGNORECASE)
    if v_match:
        fallback["verdict"] = v_match.group(1).upper()

    if any(k in fallback for k in ["semantic_correctness", "factual_accuracy", "is_correct", "is_pass"]):
        return fallback

    logger.warning(f"Could not parse valid JSON from judge output: {clean_text[:100]}...")
    return {
        "critique": f"Parsing failure. Raw response: {clean_text[:120]}",
        "semantic_correctness": 0,
        "driving_safety": 0,
        "causal_reasoning": 0,
        "factual_accuracy": 0,
        "is_correct": False,
        "is_pass": False,
        "verdict": "ERROR",
    }


@dataclass
class JudgeResult:
    sample_token: str
    category: str
    question: str
    ground_truth: str
    prediction: str
    critique: str
    scores: Dict[str, float]
    is_pass: bool
    verdict: str
    raw_output: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMJudge:
    """High-level LLM-as-a-Judge orchestrator for Autonomous Driving VQA."""

    def __init__(
        self,
        backend: BaseJudgeBackend,
        benchmark: str = "drivelm",
        cache: Optional[DiskJudgeCache] = None,
    ) -> None:
        self.backend = backend
        self.benchmark = benchmark.lower()
        self.cache = cache if cache is not None else DiskJudgeCache()

        if self.benchmark not in ["drivelm", "nuscenes_qa"]:
            raise ValueError(f"Unsupported benchmark: {benchmark}. Must be 'drivelm' or 'nuscenes_qa'.")

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
        category: str = "general",
        sample_token: str = "",
    ) -> JudgeResult:
        """Evaluates a single sample with caching."""
        cache_key = self.cache.compute_key(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            judge_model=self.backend.model_name,
            benchmark=self.benchmark,
        )

        cached_entry = self.cache.get(cache_key)
        if cached_entry is not None:
            return JudgeResult(
                sample_token=sample_token or cached_entry.get("sample_token", ""),
                category=category or cached_entry.get("category", "general"),
                question=question,
                ground_truth=ground_truth,
                prediction=prediction,
                critique=cached_entry.get("critique", ""),
                scores=cached_entry.get("scores", {}),
                is_pass=cached_entry.get("is_pass", False),
                verdict=cached_entry.get("verdict", "UNKNOWN"),
                raw_output=cached_entry.get("raw_output", ""),
            )

        # Build prompts based on benchmark
        if self.benchmark == "drivelm":
            sys_prompt = DRIVELM_JUDGE_SYSTEM_PROMPT
            user_prompt = build_drivelm_prompt(question, ground_truth, prediction, category)
        else:
            sys_prompt = NUSCENES_QA_JUDGE_SYSTEM_PROMPT
            user_prompt = build_nuscenes_qa_prompt(question, ground_truth, prediction, category)

        raw_output = self.backend.generate(system_prompt=sys_prompt, user_prompt=user_prompt)
        parsed = parse_judge_json(raw_output)

        if self.benchmark == "drivelm":
            scores = {
                "semantic_correctness": float(parsed.get("semantic_correctness", 0)),
                "driving_safety": float(parsed.get("driving_safety", 0)),
                "causal_reasoning": float(parsed.get("causal_reasoning", 0)),
            }
            is_pass = bool(parsed.get("is_pass", scores["semantic_correctness"] >= 6 and scores["driving_safety"] >= 7))
            verdict = parsed.get("verdict", "SAFE" if is_pass else "UNSAFE")
        else:
            scores = {
                "factual_accuracy": float(parsed.get("factual_accuracy", 0)),
            }
            is_pass = bool(parsed.get("is_correct", scores["factual_accuracy"] >= 7))
            verdict = "CORRECT" if is_pass else "INCORRECT"

        res = JudgeResult(
            sample_token=sample_token,
            category=category,
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            critique=parsed.get("critique", ""),
            scores=scores,
            is_pass=is_pass,
            verdict=verdict,
            raw_output=raw_output,
        )

        self.cache.set(cache_key, res.to_dict())
        return res

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
        save_cache_interval: int = 25,
    ) -> Tuple[Dict[str, Any], List[JudgeResult]]:
        """Evaluates a batch of samples and calculates multi-dimensional aggregate metrics."""
        eval_pool = samples if max_samples is None else samples[:max_samples]
        results: List[JudgeResult] = []

        logger.info(f"Running LLM Judge ({self.backend.model_name}) on {len(eval_pool)} {self.benchmark} samples...")

        for idx, it in enumerate(tqdm(eval_pool, desc=f"  Judging ({self.benchmark})")):
            res = self.evaluate_sample(
                question=it.get("question", ""),
                ground_truth=it.get("ground_truth", it.get("answer", "")),
                prediction=it.get("predicted", it.get("prediction", "")),
                category=it.get("category", it.get("template_type", "general")),
                sample_token=it.get("sample_token", ""),
            )
            results.append(res)

            if (idx + 1) % save_cache_interval == 0:
                self.cache.save()

        self.cache.save()
        summary = self._compute_summary(results)
        return summary, results

    def _compute_summary(self, results: List[JudgeResult]) -> Dict[str, Any]:
        """Computes statistical macro/micro summaries across categories."""
        if not results:
            return {"total_evaluated": 0}

        total = len(results)
        cat_results = defaultdict(list)
        for r in results:
            cat_results[r.category].append(r)

        if self.benchmark == "drivelm":
            avg_sem = sum(r.scores.get("semantic_correctness", 0) for r in results) / total
            avg_safe = sum(r.scores.get("driving_safety", 0) for r in results) / total
            avg_reason = sum(r.scores.get("causal_reasoning", 0) for r in results) / total
            pass_count = sum(1 for r in results if r.is_pass)
            hazard_count = sum(1 for r in results if r.verdict == "HAZARD" or r.scores.get("driving_safety", 0) <= 3)

            cat_metrics = {}
            for c, c_list in cat_results.items():
                c_tot = len(c_list)
                cat_metrics[c] = {
                    "count": c_tot,
                    "mean_semantic_correctness": round(sum(x.scores.get("semantic_correctness", 0) for x in c_list) / c_tot, 2),
                    "mean_driving_safety": round(sum(x.scores.get("driving_safety", 0) for x in c_list) / c_tot, 2),
                    "mean_causal_reasoning": round(sum(x.scores.get("causal_reasoning", 0) for x in c_list) / c_tot, 2),
                    "safety_pass_rate": round(sum(1 for x in c_list if x.is_pass) / c_tot * 100.0, 2),
                }

            return {
                "benchmark": "drivelm",
                "judge_model": self.backend.model_name,
                "total_evaluated": total,
                "overall_metrics": {
                    "mean_semantic_correctness": round(avg_sem, 2),
                    "mean_driving_safety": round(avg_safe, 2),
                    "mean_causal_reasoning": round(avg_reason, 2),
                    "safety_pass_rate": round(pass_count / total * 100.0, 2),
                    "hazard_rate": round(hazard_count / total * 100.0, 2),
                },
                "category_metrics": cat_metrics,
            }

        else:  # nuscenes_qa
            avg_fact = sum(r.scores.get("factual_accuracy", 0) for r in results) / total
            corr_count = sum(1 for r in results if r.is_pass)

            cat_metrics = {}
            for c, c_list in cat_results.items():
                c_tot = len(c_list)
                cat_metrics[c] = {
                    "count": c_tot,
                    "mean_factual_accuracy": round(sum(x.scores.get("factual_accuracy", 0) for x in c_list) / c_tot, 2),
                    "llm_judge_accuracy": round(sum(1 for x in c_list if x.is_pass) / c_tot * 100.0, 2),
                }

            return {
                "benchmark": "nuscenes_qa",
                "judge_model": self.backend.model_name,
                "total_evaluated": total,
                "overall_metrics": {
                    "mean_factual_accuracy": round(avg_fact, 2),
                    "llm_judge_accuracy": round(corr_count / total * 100.0, 2),
                },
                "category_metrics": cat_metrics,
            }


def print_judge_summary(summary: Dict[str, Any]) -> None:
    """Prints a beautiful formatted summary of LLM Judge evaluation."""
    bench = summary.get("benchmark", "")
    model = summary.get("judge_model", "")
    tot = summary.get("total_evaluated", 0)
    ov = summary.get("overall_metrics", {})
    cats = summary.get("category_metrics", {})

    print(f"\n{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  SOTA LLM-AS-A-JUDGE EVALUATION RESULTS ({bench.upper()}){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  Judge Model: {model} | Evaluated Samples: {tot}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")

    if bench == "drivelm":
        print(f"  Overall Semantic Correctness: {ov.get('mean_semantic_correctness', 0):.2f}/10")
        print(f"  Overall Driving Safety Score:  {Colors.BOLD}{ov.get('mean_driving_safety', 0):.2f}/10{Colors.RESET}")
        print(f"  Overall Causal Reasoning:      {ov.get('mean_causal_reasoning', 0):.2f}/10")
        print(f"  Driving Safety Pass Rate:      {Colors.BOLD}{Colors.GREEN}{ov.get('safety_pass_rate', 0):.1f}%{Colors.RESET}")
        print(f"  Critical Hazard Rate:          {Colors.RED}{ov.get('hazard_rate', 0):.1f}%{Colors.RESET}")
        print(f"\n  {'Category':<15} | {'Semantic (0-10)':<16} | {'Safety (0-10)':<14} | {'Pass Rate':<10}")
        print(f"  {'-'*65}")
        for c, m in cats.items():
            print(f"  {c:<15} | {m.get('mean_semantic_correctness', 0):>14.2f}  | {m.get('mean_driving_safety', 0):>12.2f}  | {m.get('safety_pass_rate', 0):>9.1f}%")
    else:
        print(f"  Overall Factual Accuracy:      {ov.get('mean_factual_accuracy', 0):.2f}/10")
        print(f"  LLM Judge Answer Accuracy:     {Colors.BOLD}{Colors.GREEN}{ov.get('llm_judge_accuracy', 0):.1f}%{Colors.RESET}")
        print(f"\n  {'Category':<15} | {'Factual Acc (0-10)':<20} | {'Judge Accuracy':<15}")
        print(f"  {'-'*55}")
        for c, m in cats.items():
            print(f"  {c:<15} | {m.get('mean_factual_accuracy', 0):>18.2f}  | {m.get('llm_judge_accuracy', 0):>14.1f}%")

    print(f"{Colors.GREEN}======================================================================{Colors.RESET}\n")
