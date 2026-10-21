"""
Unit tests for SOTA LLM-as-a-Judge evaluation framework.
"""

import json
from pathlib import Path
import tempfile
import pytest

from bev_vqa.evaluation.llm_judge.backends import BaseJudgeBackend
from bev_vqa.evaluation.llm_judge.cache import DiskJudgeCache
from bev_vqa.evaluation.llm_judge.judge import LLMJudge, parse_judge_json
from bev_vqa.evaluation.llm_judge.rubrics import (
    build_drivelm_prompt,
    build_nuscenes_qa_prompt,
)


class MockJudgeBackend(BaseJudgeBackend):
    """Mock backend returning predetermined responses for testing."""

    def __init__(self, response_text: str, model_name: str = "mock-judge") -> None:
        super().__init__(model_name=model_name)
        self.response_text = response_text
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response_text


def test_parse_judge_json_valid():
    raw = '{"critique": "Good response", "semantic_correctness": 9, "driving_safety": 10, "causal_reasoning": 8, "is_pass": true, "verdict": "SAFE"}'
    parsed = parse_judge_json(raw)
    assert parsed["semantic_correctness"] == 9
    assert parsed["driving_safety"] == 10
    assert parsed["is_pass"] is True
    assert parsed["verdict"] == "SAFE"


def test_parse_judge_json_markdown_fenced():
    raw = """Here is my evaluation:
```json
{
  "critique": "Safe decision",
  "semantic_correctness": 8,
  "driving_safety": 9,
  "causal_reasoning": 7,
  "is_pass": true,
  "verdict": "SAFE"
}
```
Hope this helps!"""
    parsed = parse_judge_json(raw)
    assert parsed["semantic_correctness"] == 8
    assert parsed["driving_safety"] == 9
    assert parsed["is_pass"] is True


def test_parse_judge_json_trailing_comma():
    raw = """{
  "critique": "Minor issue",
  "semantic_correctness": 7,
  "driving_safety": 8,
  "causal_reasoning": 6,
  "is_pass": true,
  "verdict": "ACCEPTABLE",
}"""
    parsed = parse_judge_json(raw)
    assert parsed["semantic_correctness"] == 7
    assert parsed["driving_safety"] == 8


def test_parse_judge_json_regex_fallback():
    raw = 'The evaluation found: "semantic_correctness": 5, "driving_safety": 3, "is_pass": false and verdict "HAZARD".'
    parsed = parse_judge_json(raw)
    assert parsed["semantic_correctness"] == 5
    assert parsed["driving_safety"] == 3
    assert parsed["is_pass"] is False
    assert parsed["verdict"] == "HAZARD"


def test_disk_judge_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.json"
        cache = DiskJudgeCache(cache_file=str(cache_file))

        key = cache.compute_key(
            question="What is the ego vehicle doing?",
            ground_truth="Going straight.",
            prediction="Going straight.",
            judge_model="mock-judge",
            benchmark="drivelm",
        )

        assert cache.get(key) is None

        cache.set(key, {"verdict": "SAFE", "scores": {"semantic_correctness": 10}})
        cache.save()

        # Reload from disk
        new_cache = DiskJudgeCache(cache_file=str(cache_file))
        cached = new_cache.get(key)
        assert cached is not None
        assert cached["verdict"] == "SAFE"
        assert cached["scores"]["semantic_correctness"] == 10


def test_llm_judge_drivelm_evaluation():
    mock_json = json.dumps({
        "critique": "Accurate prediction and safe planning.",
        "semantic_correctness": 9,
        "driving_safety": 10,
        "causal_reasoning": 8,
        "is_pass": True,
        "verdict": "SAFE",
    })
    backend = MockJudgeBackend(mock_json)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.json"
        judge = LLMJudge(backend=backend, benchmark="drivelm", cache=DiskJudgeCache(str(cache_file)))

        samples = [
            {
                "question": "Predict behavior.",
                "ground_truth": "Going straight.",
                "predicted": "Going straight.",
                "category": "behavior",
                "sample_token": "token_1",
            },
            {
                "question": "What actions to take?",
                "ground_truth": "Keep speed.",
                "predicted": "Keep speed.",
                "category": "planning",
                "sample_token": "token_2",
            },
        ]

        summary, results = judge.evaluate_dataset(samples)

        assert summary["total_evaluated"] == 2
        assert summary["overall_metrics"]["mean_semantic_correctness"] == 9.0
        assert summary["overall_metrics"]["mean_driving_safety"] == 10.0
        assert summary["overall_metrics"]["safety_pass_rate"] == 100.0
        assert len(results) == 2
        assert backend.call_count == 2

        # Second run should hit cache with 0 additional backend calls
        _, _ = judge.evaluate_dataset(samples)
        assert backend.call_count == 2


def test_llm_judge_nuscenes_qa_evaluation():
    mock_json = json.dumps({
        "critique": "Factually matches.",
        "factual_accuracy": 10,
        "is_correct": True,
    })
    backend = MockJudgeBackend(mock_json)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.json"
        judge = LLMJudge(backend=backend, benchmark="nuscenes_qa", cache=DiskJudgeCache(str(cache_file)))

        samples = [
            {
                "question": "Are there any cars?",
                "ground_truth": "yes",
                "predicted": "Yes, there is a car.",
                "category": "exist",
                "sample_token": "token_1",
            }
        ]

        summary, results = judge.evaluate_dataset(samples)
        assert summary["total_evaluated"] == 1
        assert summary["overall_metrics"]["llm_judge_accuracy"] == 100.0
        assert summary["overall_metrics"]["mean_factual_accuracy"] == 10.0
        assert results[0].is_pass is True
