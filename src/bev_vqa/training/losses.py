"""
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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

STATUS_WEIGHTS: Dict[str, float] = {
    "with rider": 4.0,
    "parked": 1.5,
    "stopped": 1.5,
    "moving": 1.0,
    "without rider": 1.0,
}


def get_status_weight(answer: str, weights_map: Optional[Dict[str, float]] = None) -> float:
    """Restituisce il moltiplicatore di perdita per la risposta di status."""
    weights = weights_map or STATUS_WEIGHTS
    ans_clean = answer.lower().strip()
    # Verifica corrispondenze dirette
    if ans_clean in weights:
        return weights[ans_clean]
    # Sotto-stringhe ordinate per lunghezza decrescente per matchare prima le più specifiche
    for k in sorted(weights.keys(), key=len, reverse=True):
        if k in ans_clean:
            return weights[k]
    return 1.0


def compute_gaussian_count_distribution(
    k: int,
    max_val: int = 10,
    sigma: float = 0.9,
) -> Dict[int, float]:
    """
    Calcola la distribuzione di etichette continue Gaussiane q(y|k) per y in [0, max_val].
    q(y|k) \propto \exp( - (y - k)^2 / (2 * \sigma^2) )
    """
    k_clamped = max(0, min(k, max_val))
    two_sigma_sq = 2.0 * (sigma ** 2)
    raw_weights = {}
    total = 0.0

    for y in range(max_val + 1):
        diff = float(y - k_clamped)
        w = math.exp(-(diff * diff) / two_sigma_sq)
        raw_weights[y] = w
        total += w

    return {y: w / total for y, w in raw_weights.items()}


class NuScenesBoostLoss(nn.Module):
    """
    Funzione di loss combinata per l'ottimizzazione SOTA NuScenes-QA Boost:
    - Ordinal Gaussian Label Smoothing sui token di conteggio (count).
      Rileva dinamicamente i token non collidenti del vocabolario (0-9 per BPE come Qwen)
      ed evita la collisione tra la cifra '1' e la sequenza multi-token '10' ('1', '0').
    - Class-Weighting su domande di tipo status ('with rider', 'parked', 'stopped').
    - Standard Cross-Entropy per risposte multi-token (es. >=10) e tutte le altre categorie.
    """

    def __init__(
        self,
        tokenizer,
        count_sigma: float = 0.9,
        max_count: int = 10,
        status_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.count_sigma = count_sigma
        self.max_count = max_count
        self.status_weights = status_weights or STATUS_WEIGHTS

        # Rileva i token ID non collidenti a token singolo (0..max_count)
        self.count_token_ids: List[int] = []
        for y in range(self.max_count + 1):
            tokens = self.tokenizer.encode(str(y), add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] not in self.count_token_ids:
                self.count_token_ids.append(tokens[0])
            else:
                # Se è multi-token o collide con un token già presente (es. '10' -> ['1', '0']), interrompi
                break

        self.supported_max_count = len(self.count_token_ids) - 1

        # Precalcola la matrice di distribuzione Gaussiana per ogni k in 0..supported_max_count
        if self.supported_max_count >= 1:
            two_sigma_sq = 2.0 * (self.count_sigma ** 2)
            dist_rows = []
            for k in range(self.supported_max_count + 1):
                raw = [math.exp(-((y - k) ** 2) / two_sigma_sq) for y in range(self.supported_max_count + 1)]
                tot = sum(raw)
                dist_rows.append([w / tot for w in raw])
            gaussian_matrix = torch.tensor(dist_rows, dtype=torch.float32)
        else:
            gaussian_matrix = torch.empty((0, 0), dtype=torch.float32)

        self.register_buffer("cand_tokens_tensor", torch.tensor(self.count_token_ids, dtype=torch.long))
        self.register_buffer("gaussian_matrix", gaussian_matrix)

    @property
    def gaussian_distributions(self) -> Dict[int, torch.Tensor]:
        """Proprietà di retro-compatibilità per distribuzioni Gaussiane precalcolate."""
        return {k: self.gaussian_matrix[k] for k in range(self.supported_max_count + 1)}

    def _get_target_count(self, answer_str: str) -> Optional[int]:
        """Tenta di interpretare la risposta come valore numerico intero in [0, max_count]."""
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
        Calcola la loss combinata su un batch di predizioni.

        Args:
            logits: Logit del modello [B, SeqLen, VocabSize]
            labels: Etichette target con padding a -100 [B, SeqLen]
            template_types: Lista delle categorie delle domande (opzionale)
            answers: Lista delle stringhe di risposta ground-truth (opzionale)

        Returns:
            Scalar tensor di loss finita e differenziabile.
        """
        B = logits.size(0)
        # Shift logits e labels per predizione autoregressiva (Causal LM)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        sample_losses = []

        for b in range(B):
            sample_mask = (shift_labels[b] != -100)
            if not sample_mask.any():
                # Nessun token di risposta attivo
                sample_losses.append(shift_logits[b].sum() * 0.0)
                continue

            sub_logits = shift_logits[b][sample_mask]
            sub_labels = shift_labels[b][sample_mask]

            cat = template_types[b] if template_types and b < len(template_types) else "unknown"
            ans = answers[b] if answers and b < len(answers) else ""

            # Caso 1: Categoria Count con risposta numerica entro i token singoli non collidenti
            k_count = self._get_target_count(ans) if ans else None
            if cat == "count" and k_count is not None and k_count <= self.supported_max_count:
                # Applica Ordinal Gaussian Label Smoothing al primo token (token della cifra)
                z_0 = sub_logits[0].float()  # [VocabSize] in float32 per stabilità
                log_probs_0 = F.log_softmax(z_0, dim=-1)

                q_dist = self.gaussian_matrix[k_count]
                cand_log_probs = log_probs_0[self.cand_tokens_tensor]
                count_loss = -(q_dist * cand_log_probs).sum()

                # Se ci sono altri token nella risposta (es. EOS o cifre successive), calcola CE standard
                if sub_logits.size(0) > 1:
                    rem_logits = sub_logits[1:].float()
                    rem_labels = sub_labels[1:]
                    rem_loss = F.cross_entropy(rem_logits, rem_labels, reduction="mean")
                    loss_b = (count_loss + rem_loss * rem_labels.numel()) / sub_labels.numel()
                else:
                    loss_b = count_loss

            # Caso 2: Categoria Status con ponderazione delle classi rare
            elif cat == "status":
                w = get_status_weight(ans, self.status_weights)
                ce = F.cross_entropy(sub_logits.float(), sub_labels, reduction="mean")
                loss_b = w * ce

            # Caso 3: Tutte le altre categorie (object, exist, comparison, e count multi-token es. 10)
            else:
                loss_b = F.cross_entropy(sub_logits.float(), sub_labels, reduction="mean")

            sample_losses.append(loss_b)

        if not sample_losses:
            return shift_logits.sum() * 0.0

        total_loss = torch.stack(sample_losses).mean()
        return total_loss
