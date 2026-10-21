"""
SOTA LLM-as-a-Judge Evaluation Module for Autonomous Driving VQA.
"""

from bev_vqa.evaluation.llm_judge.backends import (
    APIJudgeBackend,
    BaseJudgeBackend,
    LocalHFBackend,
)
from bev_vqa.evaluation.llm_judge.cache import DiskJudgeCache
from bev_vqa.evaluation.llm_judge.judge import (
    JudgeResult,
    LLMJudge,
    parse_judge_json,
    print_judge_summary,
)
from bev_vqa.evaluation.llm_judge.rubrics import (
    DRIVELM_JUDGE_SYSTEM_PROMPT,
    NUSCENES_QA_JUDGE_SYSTEM_PROMPT,
    build_drivelm_prompt,
    build_nuscenes_qa_prompt,
)

__all__ = [
    "BaseJudgeBackend",
    "LocalHFBackend",
    "APIJudgeBackend",
    "DiskJudgeCache",
    "JudgeResult",
    "LLMJudge",
    "parse_judge_json",
    "print_judge_summary",
    "DRIVELM_JUDGE_SYSTEM_PROMPT",
    "NUSCENES_QA_JUDGE_SYSTEM_PROMPT",
    "build_drivelm_prompt",
    "build_nuscenes_qa_prompt",
]
