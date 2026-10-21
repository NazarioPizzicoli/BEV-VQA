"""
Metrics and evaluation utilities for BEV-VQA.
"""

import re
import string
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def normalize_answer(s: str) -> str:
    """
    Normalizes the string for fair comparison.
    - Lowercase
    - Remove punctuation
    - Remove articles
    - Strip whitespace

    Args:
        s (str): The input string to normalize.

    Returns:
        str: The normalized string.
    """
    if not isinstance(s, str):
        return ""
        
    s = s.lower().strip()
    
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles (the, a, an) - basic English normalization
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    # Remove extra whitespace
    s = ' '.join(s.split())
    
    return s


def compute_nlg_metrics(
    predictions: Optional[Union[List[str], List[List[str]]]] = None,
    references: Optional[Union[List[str], List[List[str]]]] = None,
    hypotheses: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Computes NLG metrics (BLEU-1..4, ROUGE-L, CIDEr, METEOR) using pycocoevalcap and nltk.
    Supports both List[str] and List[List[str]] for references, with robust handling
    of empty strings and exceptions.

    Args:
        predictions (Optional[Union[List[str], List[List[str]]]], optional): The predicted sequences.
        references (Optional[Union[List[str], List[List[str]]]], optional): The ground truth sequences.
        hypotheses (Optional[List[str]], optional): Alternative name for predictions.
        **kwargs: Additional arguments, such as 'targets' for references.

    Returns:
        Dict[str, float]: Dictionary containing computed scores for each metric.
    """
    if hypotheses is not None:
        preds = hypotheses
        refs = references if references is not None else predictions
    elif kwargs.get("targets") is not None:
        refs = kwargs.get("targets")
        preds = predictions if predictions is not None else hypotheses
    elif references is not None and predictions is not None:
        # Check if caller passed (references, predictions) positionally
        if (
            len(predictions) > 0
            and isinstance(predictions[0], (list, tuple))
            and (len(references) == 0 or isinstance(references[0], str))
        ):
            refs = predictions
            preds = references
        else:
            preds = predictions
            refs = references
    elif predictions is not None and references is None:
        preds = predictions
        refs = []
    else:
        preds = []
        refs = []

    empty_res = {
        "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
        "rouge_l": 0.0, "cider": 0.0, "meteor": 0.0,
        "BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0,
        "ROUGE-L": 0.0, "CIDEr": 0.0, "METEOR": 0.0,
    }

    if not preds or not refs:
        return empty_res

    n = min(len(preds), len(refs))
    if n == 0:
        return empty_res

    # Check if all predictions or all references are empty strings
    def _is_empty_val(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, (list, tuple)):
            return not any(bool(str(x).strip()) for x in v if x is not None)
        return not bool(str(v).strip())

    if all(_is_empty_val(p) for p in preds[:n]) or all(_is_empty_val(r) for r in refs[:n]):
        return empty_res

    gts = {}
    res = {}
    for i in range(n):
        p_str = str(preds[i]).strip() if preds[i] is not None else ""
        res[i] = [p_str]

        r = refs[i]
        if isinstance(r, (list, tuple)):
            clean_r = [str(x).strip() for x in r if x is not None and str(x).strip()]
            gts[i] = clean_r if clean_r else [""]
        else:
            r_str = str(r).strip() if r is not None else ""
            gts[i] = [r_str]

    bleu_scores = [0.0, 0.0, 0.0, 0.0]
    rouge_score = 0.0
    cider_score = 0.0
    meteor_score_val = 0.0

    # 1. BLEU-1..4
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        bleu_scorer = Bleu(4)
        b_scores, _ = bleu_scorer.compute_score(gts, res)
        if isinstance(b_scores, (list, tuple)) and len(b_scores) == 4:
            bleu_scores = [float(s) for s in b_scores]
    except Exception as e:
        logger.warning(f"pycocoevalcap BLEU computation failed: {e}. Trying NLTK fallback...")
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            sf = SmoothingFunction().method1
            b_accum = [0.0, 0.0, 0.0, 0.0]
            valid_samples = 0
            for i in range(n):
                hyp_toks = res[i][0].lower().split()
                ref_toks_list = [r.lower().split() for r in gts[i] if r.strip()]
                if not ref_toks_list or not hyp_toks:
                    continue
                valid_samples += 1
                for order in range(1, 5):
                    weights = tuple([1.0 / order] * order + [0.0] * (4 - order))
                    b_accum[order - 1] += sentence_bleu(ref_toks_list, hyp_toks, weights=weights, smoothing_function=sf)
            if valid_samples > 0:
                bleu_scores = [s / n for s in b_accum]
        except Exception as e2:
            logger.warning(f"NLTK BLEU fallback also failed: {e2}")

    # 2. ROUGE-L
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        rouge_scorer = Rouge()
        # Filter out empty references that can cause division issues
        filtered_gts = {k: v for k, v in gts.items() if any(v)}
        filtered_res = {k: res[k] for k in filtered_gts}
        if filtered_gts:
            r_score, _ = rouge_scorer.compute_score(filtered_gts, filtered_res)
            rouge_score = float(r_score) * (len(filtered_gts) / n)
    except Exception as e:
        logger.warning(f"ROUGE computation fallback: {e}")

    # 3. CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        cider_scorer = Cider()
        filtered_gts = {k: v for k, v in gts.items() if any(v)}
        filtered_res = {k: res[k] for k in filtered_gts}
        if filtered_gts:
            c_score, _ = cider_scorer.compute_score(filtered_gts, filtered_res)
            cider_score = float(c_score) * (len(filtered_gts) / n)
    except Exception as e:
        logger.warning(f"CIDEr computation fallback: {e}")

    # 4. METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        meteor_scores = []
        for i in range(n):
            ref_tokens = [ref.lower().split() for ref in gts[i] if ref.strip()]
            hyp_tokens = res[i][0].lower().split()
            if ref_tokens and hyp_tokens:
                m = meteor_score(ref_tokens, hyp_tokens)
                meteor_scores.append(m)
            else:
                meteor_scores.append(0.0)
        if meteor_scores:
            meteor_score_val = float(sum(meteor_scores) / len(meteor_scores))
    except Exception as e:
        logger.warning(f"METEOR computation fallback: {e}")

    return {
        "bleu_1": round(bleu_scores[0] * 100.0, 2),
        "bleu_2": round(bleu_scores[1] * 100.0, 2),
        "bleu_3": round(bleu_scores[2] * 100.0, 2),
        "bleu_4": round(bleu_scores[3] * 100.0, 2),
        "rouge_l": round(rouge_score * 100.0, 2),
        "cider": round(cider_score, 3),
        "meteor": round(meteor_score_val * 100.0, 2),
        # Capitalized aliases for compatibility
        "BLEU-1": round(bleu_scores[0] * 100.0, 2),
        "BLEU-2": round(bleu_scores[1] * 100.0, 2),
        "BLEU-3": round(bleu_scores[2] * 100.0, 2),
        "BLEU-4": round(bleu_scores[3] * 100.0, 2),
        "ROUGE-L": round(rouge_score * 100.0, 2),
        "CIDEr": round(cider_score, 3),
        "METEOR": round(meteor_score_val * 100.0, 2),
    }

