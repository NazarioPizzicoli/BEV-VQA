#!/usr/bin/env python3
"""
DriveLM Generative Evaluation (NLG Metrics): BLEU-4, ROUGE-L, CIDEr, METEOR.

Valuta le capacità generative e discorsive del BEV-VLM sulle domande complesse di DriveLM:
- Percezione ('perception'): descrizione di oggetti e posizioni
- Predizione ('prediction'): anticipazione dei movimenti dei veicoli circostanti
- Pianificazione ('planning'): decisioni di guida dell'ego-veicolo (frenare, accelerare, svoltare)
- Comportamento ('behavior'): spiegazione delle manovre

Metriche calcolate:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 (pycocoevalcap)
- ROUGE-L (pycocoevalcap)
- CIDEr (pycocoevalcap)
- METEOR (nltk)
"""

import argparse
from collections import defaultdict
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import torch
from tqdm import tqdm

from bev_vqa.data.dataset import build_tokenizer, normalize_answer, VQA_SYSTEM_PROMPT
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(level=logging.WARNING)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ensure_nltk_resources():
    """Scarica le risorse NLTK minime se non già presenti."""
    for res in ["wordnet", "punkt", "omw-1.4"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


def load_drivelm_data(json_path: str, bev_dirs: List[Path], max_samples: int = 200, seed: int = 42) -> List[Dict]:
    """Carica e campiona domande da DriveLM val che hanno una BEV esistente."""
    with open(json_path) as f:
        data = json.load(f).get("questions", [])

    # Mappa token disponibili
    available_tokens = set()
    for b_dir in bev_dirs:
        if b_dir.exists():
            for p in b_dir.glob("*.pt"):
                available_tokens.add(p.stem)

    valid_questions = [q for q in data if q.get("sample_token") in available_tokens]
    print(f"DriveLM Val: {len(valid_questions):,} domande valide con BEV su {len(data):,} totali.")

    # Raggruppa per tipo per campionamento bilanciato (perception, prediction, planning)
    by_type = defaultdict(list)
    for q in valid_questions:
        by_type[q.get("template_type", "other")].append(q)

    random.seed(seed)
    target_types = ["perception", "prediction", "planning"]
    n_per_type = max_samples // len(target_types)

    sampled = []
    for t in target_types:
        pool = by_type.get(t, [])
        sampled.extend(random.sample(pool, min(n_per_type, len(pool))))

    # Completa se necessario
    if len(sampled) < max_samples:
        remaining = [q for q in valid_questions if q not in sampled]
        needed = max_samples - len(sampled)
        sampled.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(sampled)
    return sampled


def compute_nlg_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calcola le metriche NLG ufficiali COCO/VQA:
    BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, CIDEr, METEOR.
    """
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}

    # 1. BLEU
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)

    # 2. ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)

    # 3. CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    # 4. METEOR
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        m = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(m)
    avg_meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    return {
        "bleu_1": round(bleu_scores[0] * 100.0, 2),
        "bleu_2": round(bleu_scores[1] * 100.0, 2),
        "bleu_3": round(bleu_scores[2] * 100.0, 2),
        "bleu_4": round(bleu_scores[3] * 100.0, 2),
        "rouge_l": round(rouge_score * 100.0, 2),
        "cider": round(cider_score, 3),
        "meteor": round(avg_meteor * 100.0, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="DriveLM Generative NLG Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage2/best_model")
    parser.add_argument("--drivelm-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_val.json")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset/bev_features")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--report-path", type=str, default="outputs/drivelm_nlg_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_nltk_resources()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bev_base = Path(args.bev_dir)
    bev_dirs = [
        bev_base / "val",
        bev_base / "train",
    ]

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}  VALUTAZIONE GENERATIVA DRIVELM (METRICHE NLG: BLEU, ROUGE, CIDER){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Checkpoint:           {args.checkpoint}")
    print(f"  Campioni di Test:     {args.num_samples}")
    print(f"  Max Token Generati:   {args.max_new_tokens}")
    print(f"  Dispositivo:          {device}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    tokenizer = build_tokenizer(args.llm_path)

    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(arch_type=args.arch_type, num_tokens=args.num_tokens),
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()
    model.load_checkpoint(args.checkpoint)
    model.eval()

    # Mappa token disponibili a percorso file
    token_to_path = {}
    for b_dir in bev_dirs:
        if b_dir.exists():
            for p in b_dir.glob("*.pt"):
                if p.stem not in token_to_path:
                    token_to_path[p.stem] = p

    eval_items = load_drivelm_data(args.drivelm_val_json, bev_dirs, max_samples=args.num_samples, seed=args.seed)

    references = []
    hypotheses = []
    cat_refs = defaultdict(list)
    cat_hyps = defaultdict(list)
    sample_previews = []

    print(f"Inizio generazione risposte su {len(eval_items)} campioni DriveLM...")
    t0 = time.time()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(eval_items, desc="  DriveLM NLG Eval")):
            token = item["sample_token"]
            q = item["question"]
            gt = item["answer"]
            cat = item.get("template_type", "unknown")

            bev_path = token_to_path.get(token)
            if bev_path is None:
                continue

            bev_data = torch.load(bev_path, map_location="cpu", weights_only=True)
            bev = bev_data["features_fused"]
            if bev.dim() == 3:
                bev = bev.unsqueeze(0)
            bev = bev.to(device)

            prompt_text = (
                f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<|bev|>\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

            gen_ids = model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            references.append(gt)
            hypotheses.append(pred_text)

            cat_refs[cat].append(gt)
            cat_hyps[cat].append(pred_text)

            if len(sample_previews) < 6:
                sample_previews.append({
                    "sample_token": token,
                    "category": cat,
                    "question": q,
                    "ground_truth": gt,
                    "predicted": pred_text,
                })

            del bev, gen_ids

    elapsed = time.time() - t0
    print(f"\nGenerazione completata in {elapsed:.1f} s ({elapsed/len(eval_items)*1000:.1f} ms/campione).")

    # Calcolo metriche globali
    print("\nCalcolo metriche ufficiali NLG...")
    overall_metrics = compute_nlg_metrics(references, hypotheses)

    # Calcolo metriche per categoria
    category_metrics = {}
    for cat in ["perception", "prediction", "planning"]:
        if cat in cat_refs and len(cat_refs[cat]) > 0:
            category_metrics[cat] = compute_nlg_metrics(cat_refs[cat], cat_hyps[cat])

    # Stampa Risultati
    print(f"\n{BOLD}{GREEN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  RISULTATI BENCHMARK GENERATIVO DRIVELM (METRICHE NLG){RESET}")
    print(f"{BOLD}{GREEN}======================================================================{RESET}")
    print(f"  Campioni Valutati:    {len(references)}")
    print(f"  BLEU-1:               {BOLD}{overall_metrics['bleu_1']:.2f}%{RESET}")
    print(f"  BLEU-2:               {BOLD}{overall_metrics['bleu_2']:.2f}%{RESET}")
    print(f"  BLEU-3:               {BOLD}{overall_metrics['bleu_3']:.2f}%{RESET}")
    print(f"  BLEU-4:               {BOLD}{GREEN}{overall_metrics['bleu_4']:.2f}%{RESET}")
    print(f"  ROUGE-L:              {BOLD}{GREEN}{overall_metrics['rouge_l']:.2f}%{RESET}")
    print(f"  CIDEr:                {BOLD}{GREEN}{overall_metrics['cider']:.3f}{RESET}")
    print(f"  METEOR:               {BOLD}{GREEN}{overall_metrics['meteor']:.2f}%{RESET}")

    print(f"\n  Dettaglio per Categoria DriveLM:")
    print(f"  {'Categoria':<15} | {'Campioni':<10} | {'BLEU-4':<10} | {'ROUGE-L':<10} | {'CIDEr':<10} | {'METEOR':<10}")
    print(f"  {'-'*75}")
    for cat, m in category_metrics.items():
        n_c = len(cat_refs[cat])
        print(f"  {cat:<15} | {n_c:<10} | {m['bleu_4']:>8.2f}% | {m['rouge_l']:>8.2f}% | {m['cider']:>8.3f}  | {m['meteor']:>8.2f}%")

    print(f"\n{BOLD}{CYAN}----------------------------------------------------------------------{RESET}")
    print(f"{BOLD}{CYAN}  ESEMPI DI RISPOSTE GENERATIVE DRIVELM{RESET}")
    print(f"{BOLD}{CYAN}----------------------------------------------------------------------{RESET}")
    for idx, prev in enumerate(sample_previews, 1):
        print(f"\n{BOLD}Esempio #{idx} [{prev['category'].upper()}]:{RESET}")
        print(f"  Domanda:      {prev['question']}")
        print(f"  Ground Truth: {BOLD}{prev['ground_truth']}{RESET}")
        print(f"  BEV-VLM:      {GREEN}{prev['predicted']}{RESET}")

    # Salva Report JSON
    report = {
        "dataset": "DriveLM",
        "checkpoint": args.checkpoint,
        "num_evaluated": len(references),
        "overall_metrics": overall_metrics,
        "category_metrics": category_metrics,
        "sample_previews": sample_previews,
        "latency_seconds": round(elapsed, 1),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{GREEN}Report generativo salvato in: {report_path}{RESET}\n")


if __name__ == "__main__":
    main()
