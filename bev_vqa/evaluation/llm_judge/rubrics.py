"""
Evaluation rubrics, prompt templates, and scoring specifications for LLM-as-a-Judge.
Aliged with DriveLM (CVPR 2024), DriveVLM, G-Eval, and SOTA Autonomous Driving benchmarks.
"""

from typing import Dict, Any, Optional

DRIVELM_JUDGE_SYSTEM_PROMPT = """You are an expert autonomous driving safety and perception evaluator serving as an impartial judge for multimodal autonomous driving models.
Your task is to critically evaluate a model's predicted answer against the verified ground truth for a given driving scenario question.

You must assess the prediction across three orthogonal dimensions:
1. SEMANTIC CORRECTNESS (0-10):
   - Does the prediction accurately convey the visual and physical facts described in the ground truth (e.g., object presence, type, relative position, speed)?
   - 10: Completely accurate, all facts and visual attributes correct.
   - 7-9: Mostly accurate with minor non-critical linguistic variances.
   - 4-6: Partially correct; captures some objects or actions but misses important details.
   - 1-3: Largely inaccurate or confusing.
   - 0: Completely wrong or contradictory.

2. DRIVING SAFETY & DECISION COMPLIANCE (0-10):
   - For planning/behavior questions: Is the recommended action safe, defensive, and free of collision hazard?
   - CRITICAL SAFETY PRINCIPLE: If the ground truth specifies a cautious or stopping maneuver (e.g., "stop", "yield", "decelerate") and the model proposes an aggressive or opposing maneuver (e.g., "speed up", "keep going"), you MUST give a safety score <= 2.
   - For perception/prediction questions: Does a misidentification create an immediate road hazard?
   - 10: Completely safe, prudent, defensive decision.
   - 7-9: Safe and compliant with minor suboptimal nuance.
   - 4-6: Marginal safety margin; potential hesitation or mild ambiguity.
   - 0-3: CRITICAL SAFETY VIOLATION / COLLISION RISK.

3. CAUSAL REASONING & LOGIC (0-10):
   - Does the prediction explain "why" the action is necessary logically and consistently with the road context?
   - 10: Flawless causal explanation.
   - 5: Plausible but incomplete reasoning.
   - 0: Hallucinated, irrelevant, or nonsensical rationale.

EVALUATION PROTOCOL:
- You must output your evaluation strictly as a valid JSON object.
- Provide a concise step-by-step analysis in "critique" before assigning numerical ratings.
- Define "is_pass" as true if semantic_correctness >= 6 and driving_safety >= 7, otherwise false.
- Format:
{
  "critique": "Step-by-step evaluation explaining specific merits, errors, or safety implications...",
  "semantic_correctness": <integer 0-10>,
  "driving_safety": <integer 0-10>,
  "causal_reasoning": <integer 0-10>,
  "verdict": "SAFE" | "ACCEPTABLE" | "UNSAFE" | "HAZARD",
  "is_pass": <boolean>
}
"""

DRIVELM_USER_PROMPT_TEMPLATE = """Evaluate the following Autonomous Driving model response:

[QUESTION CATEGORY]: {category}
[QUESTION]: {question}

[GROUND TRUTH ANSWER]:
{ground_truth}

[MODEL PREDICTED ANSWER]:
{prediction}

Evaluate strictly according to the autonomous driving rubric. Provide your output strictly as a JSON object:"""


NUSCENES_QA_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing answers to spatial reasoning questions in autonomous driving Bird's-Eye-View (BEV) environments.
Evaluate the model's predicted answer against the ground truth.

Unlike naive string matching:
- Accept synonymous or natural language phrasings that mean the exact same thing (e.g., "It is parked" = "parked", "two vehicles" = "2", "stationary" = "stopped").
- Penalize factual errors (e.g., "3" when ground truth is "5", "truck" when ground truth is "car", "moving" when ground truth is "stopped").

Output strictly as a valid JSON object:
{
  "critique": "Brief explanation of factual correspondence...",
  "factual_accuracy": <integer 0-10>,
  "is_correct": <boolean true/false>
}
"""

NUSCENES_QA_USER_PROMPT_TEMPLATE = """[QUESTION TYPE]: {category}
[QUESTION]: {question}
[GROUND TRUTH]: {ground_truth}
[PREDICTED]: {prediction}

Evaluate the prediction against the ground truth. Output strictly in JSON format:"""


def build_drivelm_prompt(question: str, ground_truth: str, prediction: str, category: str = "general") -> str:
    """Builds the user prompt for DriveLM evaluation."""
    return DRIVELM_USER_PROMPT_TEMPLATE.format(
        category=category.upper(),
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        prediction=prediction.strip() if prediction.strip() else "<NO_ANSWER>",
    )


def build_nuscenes_qa_prompt(question: str, ground_truth: str, prediction: str, category: str = "general") -> str:
    """Builds the user prompt for NuScenes-QA evaluation."""
    return NUSCENES_QA_USER_PROMPT_TEMPLATE.format(
        category=category.upper(),
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        prediction=prediction.strip() if prediction.strip() else "<NO_ANSWER>",
    )
