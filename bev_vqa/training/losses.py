r"""
Loss functions for NuScenes-QA Boost Training (Intervention 3).

Includes:
1. Ordinal Gaussian Label Smoothing for numerical count answers (0-10):
   q(y|k) \propto \exp( - (y - k)^2 / (2 * \sigma^2) )
   penalizes off-by-1 errors smoothly instead of with an infinite CE cliff.

2. Class-Weighted Loss for status answers:
   counteracts bias towards 'moving' with class multipliers:
   - 'with rider': 4.0x
   - 'parked': 1.5x
   - 'stopped': 1.5x
   - 'moving' / others: 1.0x
"""

import math
import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

STATUS_WEIGHTS: Dict[str, float] = {
    "with rider": 4.0,
    "parked": 1.5,
    "stopped": 1.5,
    "moving": 1.0,
    "without rider": 1.0,
}


def get_status_weight(answer: str, weights_map: Optional[Dict[str, float]] = None) -> float:
    """
    Returns the loss multiplier for a given status answer.

    Args:
        answer (str): The status answer string.
        weights_map (Optional[Dict[str, float]], optional): Custom weights mapping. Defaults to None.

    Returns:
        float: The corresponding loss multiplier weight.
    """
    weights = weights_map or STATUS_WEIGHTS
    ans_clean = answer.lower().strip()
    
    # Direct matches
    if ans_clean in weights:
        return weights[ans_clean]
        
    # Substring matches sorted by descending length to match more specific strings first
    for k in sorted(weights.keys(), key=len, reverse=True):
        if k in ans_clean:
            return weights[k]
            
    return 1.0


def compute_gaussian_count_distribution(
    k: int,
    max_val: int = 10,
    sigma: float = 0.9,
) -> Dict[int, float]:
    """Computes Gaussian label distribution q(y|k) for y in [0, max_val]."""
    k_clamped = max(0, min(k, max_val))
    two_sigma_sq = 2.0 * (sigma ** 2)
    weights = [math.exp(-((y - k_clamped) ** 2) / two_sigma_sq) for y in range(max_val + 1)]
    total = sum(weights)
    return {y: w / total for y, w in enumerate(weights)}


class NuScenesBoostLoss(nn.Module):
    """
    Combined loss function for NuScenes-QA Boost SOTA optimization:
    - Ordinal Gaussian Label Smoothing on count tokens.
      Dynamically detects non-colliding vocabulary tokens (0-9 for BPE like Qwen)
      and avoids collision between the digit '1' and multi-token sequence '10' ('1', '0').
    - Class-Weighting on status questions ('with rider', 'parked', 'stopped').
    - Standard Cross-Entropy for multi-token answers (e.g. >=10) and all other categories.
    """

    def __init__(
        self,
        tokenizer: Any,
        count_sigma: float = 0.9,
        max_count: int = 10,
        status_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initializes the NuScenesBoostLoss.

        Args:
            tokenizer (Any): The tokenizer to map numbers to token IDs.
            count_sigma (float, optional): The standard deviation for Gaussian smoothing. Defaults to 0.9.
            max_count (int, optional): The maximum numerical count to apply smoothing. Defaults to 10.
            status_weights (Optional[Dict[str, float]], optional): Custom weights for status classes. Defaults to None.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.count_sigma = count_sigma
        self.max_count = max_count
        self.status_weights = status_weights or STATUS_WEIGHTS

        # Detect single-token non-colliding token IDs (0..max_count)
        self.count_token_ids: List[int] = []
        for y in range(self.max_count + 1):
            tokens = self.tokenizer.encode(str(y), add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] not in self.count_token_ids:
                self.count_token_ids.append(tokens[0])
            else:
                # Break if it's multi-token or collides (e.g. '10' -> ['1', '0'])
                break

        self.supported_max_count = len(self.count_token_ids) - 1

        # Precompute the Gaussian distribution matrix for each k in 0..supported_max_count
        if self.supported_max_count >= 1:
            dist_rows = [
                list(compute_gaussian_count_distribution(k, self.supported_max_count, self.count_sigma).values())
                for k in range(self.supported_max_count + 1)
            ]
            gaussian_matrix = torch.tensor(dist_rows, dtype=torch.float32)
        else:
            gaussian_matrix = torch.empty((0, 0), dtype=torch.float32)


        self.register_buffer("cand_tokens_tensor", torch.tensor(self.count_token_ids, dtype=torch.long))
        self.register_buffer("gaussian_matrix", gaussian_matrix)

    def _get_target_count(self, answer_str: str) -> Optional[int]:
        """Attempts to interpret the answer as an integer numerical value in [0, max_count]."""
        words = answer_str.lower().strip().split()
        if not words:
            return None
        w0 = words[0].strip(".,;:?!")
        try:
            val = int(w0)
            if 0 <= val <= self.max_count:
                return val
        except ValueError:
            pass
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        if w0 in word_to_num and word_to_num[w0] <= self.max_count:
            return word_to_num[w0]
        return None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        template_types: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Computes the combined loss on a batch of predictions.

        Args:
            logits (torch.Tensor): Model logits [B, SeqLen, VocabSize].
            labels (torch.Tensor): Target labels padded with -100 [B, SeqLen].
            template_types (Optional[List[str]], optional): List of question categories. Defaults to None.
            answers (Optional[List[str]], optional): List of ground-truth answer strings. Defaults to None.

        Returns:
            torch.Tensor: A scalar tensor containing the finite and differentiable loss.
        """
        B = logits.size(0)
        # Shift logits and labels for autoregressive prediction (Causal LM)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        sample_losses = []

        for b in range(B):
            sample_mask = (shift_labels[b] != -100)
            if not sample_mask.any():
                # No active answer tokens
                sample_losses.append(shift_logits[b].sum() * 0.0)
                continue

            sub_logits = shift_logits[b][sample_mask]
            sub_labels = shift_labels[b][sample_mask]

            cat = template_types[b] if template_types and b < len(template_types) else "unknown"
            ans = answers[b] if answers and b < len(answers) else ""

            # Case 1: Count category with numerical answer within non-colliding single tokens
            k_count = self._get_target_count(ans) if ans else None
            if cat == "count" and k_count is not None and k_count <= self.supported_max_count:
                # Apply Ordinal Gaussian Label Smoothing to the first token (the digit token)
                z_0 = sub_logits[0].float()  # [VocabSize] in float32 for stability
                log_probs_0 = F.log_softmax(z_0, dim=-1)

                q_dist = self.gaussian_matrix[k_count]
                cand_log_probs = log_probs_0[self.cand_tokens_tensor]
                count_loss = -(q_dist * cand_log_probs).sum()

                # If there are other tokens in the answer (e.g. EOS or subsequent digits), compute standard CE
                if sub_logits.size(0) > 1:
                    rem_logits = sub_logits[1:].float()
                    rem_labels = sub_labels[1:]
                    rem_loss = F.cross_entropy(rem_logits, rem_labels, reduction="mean")
                    loss_b = (count_loss + rem_loss * rem_labels.numel()) / sub_labels.numel()
                else:
                    loss_b = count_loss

            # Case 2: Status category with weighting for rare classes
            elif cat == "status":
                w = get_status_weight(ans, self.status_weights)
                ce = F.cross_entropy(sub_logits.float(), sub_labels, reduction="mean")
                loss_b = w * ce

            # Case 3: All other categories (object, exist, comparison, and multi-token count like 10)
            else:
                loss_b = F.cross_entropy(sub_logits.float(), sub_labels, reduction="mean")

            sample_losses.append(loss_b)

        if not sample_losses:
            return shift_logits.sum() * 0.0

        total_loss = torch.stack(sample_losses).mean()
        return total_loss
