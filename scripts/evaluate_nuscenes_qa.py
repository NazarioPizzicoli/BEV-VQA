#!/usr/bin/env python3
"""
Comprehensive NuScenes-QA Benchmark Evaluation Script.

Evaluates the BEV-VLM model on NuScenes-QA validation set across:
1. Balanced Distribution (macro-accuracy across exist, count, object, status, comparison)
2. Natural Distribution (natural frequency as in the original NuScenes-QA dataset)
3. Shuffle Test (measuring visual grounding delta: Real BEV vs Shuffled BEV)
4. Comparison against Text Baseline and Published Literature (BeLLA, MCAN)
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

import torch
from tqdm import tqdm

from bev_vqa.data.dataset import build_tokenizer, normalize_answer, VQA_SYSTEM_PROMPT
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_nuscenes_qa")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

PUBLISHED_BASELINES = {
    "Text-Only LLM (Zero-Shot)": {
        "overall": 19.6, "exist": 38.0, "count": 0.0, "object": 12.0, "status": 6.0, "comparison": 42.0
    },
    "CenterPoint + MCAN": {
        "overall": 61.2, "exist": 78.4, "count": 48.9, "object": 48.2, "status": 62.1, "comparison": 68.3
    },
    "BeLLA (State-of-the-Art)": {
        "overall": 66.8, "exist": 82.5, "count": 55.4, "object": 53.7, "status": 66.2, "comparison": 76.1
    }
}


def load_qa_data(json_path: str, bev_dir: Path) -> List[Dict]:
    with open(json_path) as f:
        data = json.load(f).get("questions", [])

    val_bev_dir = bev_dir / "val"
    train_bev_dir = bev_dir / "train"

    valid_questions = []
    for q in data:
        token = q.get("sample_token")
        if (val_bev_dir / f"{token}.pt").exists() or (train_bev_dir / f"{token}.pt").exists():
            valid_questions.append(q)

    logger.info(f"NuScenes-QA Val: {len(valid_questions):,} campioni validi trovati con BEV su {len(data):,} totali.")
    return valid_questions


def load_bev_tensor(sample_token: str, bev_dir: Path, device: torch.device) -> Optional[torch.Tensor]:
    for split in ["val", "train"]:
        p = bev_dir / split / f"{sample_token}.pt"
        if p.exists():
            data = torch.load(p, map_location="cpu", weights_only=True)
            feat = data["features_fused"]
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)
            return feat.to(device)
    return None


def run_evaluation(
    model: BEVVLM,
    tokenizer,
    items: List[Dict],
    bev_dir: Path,
    device: torch.device,
    desc: str = "Evaluating",
    max_new_tokens: int = 10,
    shuffled_tokens: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], List[Dict]]:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    results = []

    with torch.no_grad():
        for idx, it in enumerate(tqdm(items, desc=desc, dynamic_ncols=True)):
            token = it["sample_token"]
            fetch_token = shuffled_tokens[idx] if shuffled_tokens is not None else token
            bev = load_bev_tensor(fetch_token, bev_dir, device)
            if bev is None:
                continue

            q = it["question"]
            gt = normalize_answer(it["answer"])
            cat = it.get("template_type", "unknown")

            prompt = (
                f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<|bev|>\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            out = model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred = normalize_answer(tokenizer.decode(out[0], skip_special_tokens=True).strip())
            is_match = (pred == gt) or (gt in pred.split())

            if is_match:
                correct += 1
                cat_correct[cat] += 1
            total += 1
            cat_total[cat] += 1

            if len(results) < 15:
                results.append({
                    "sample_token": token,
                    "category": cat,
                    "question": q,
                    "ground_truth": gt,
                    "prediction": pred,
                    "correct": is_match,
                })
            del bev, out

    overall_acc = round((correct / max(total, 1)) * 100.0, 2)
    cat_accs = {}
    for c, tot in cat_total.items():
        cat_accs[c] = round((cat_correct[c] / tot) * 100.0, 2) if tot > 0 else 0.0

    return overall_acc, cat_accs, results


def main():
    parser = argparse.ArgumentParser(description="NuScenes-QA Benchmark (Balanced & Natural Distribution)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Percorso checkpoint modello da valutare")
    parser.add_argument("--nuscenes-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset/bev_features")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--balanced-samples-per-cat", type=int, default=100, help="Campioni per categoria per il test bilanciato (100 * 5 = 500)")
    parser.add_argument("--natural-samples", type=int, default=500, help="Campioni per la distribuzione naturale")
    parser.add_argument("--run-shuffle", action="store_true", default=True, help="Esegui lo shuffle test sul dataset bilanciato")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/nuscenes_qa_sota_report.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bev_dir = Path(args.bev_dir)

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}  BENCHMARK COMPLETO NUSCENES-QA: BILANCIATO vs NATURALE{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Checkpoint:           {args.checkpoint}")
    print(f"  BEV Features Dir:     {args.bev_dir}")
    print(f"  Dispositivo:          {device}")
    print(f"  Campioni Bilanciati:  {args.balanced_samples_per_cat * 5} (5 categorie x {args.balanced_samples_per_cat})")
    print(f"  Campioni Naturali:    {args.natural_samples}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    # Inizializza modello
    tokenizer = build_tokenizer(args.llm_path)
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(arch_type=args.arch_type, num_tokens=args.num_tokens),
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.load_checkpoint(args.checkpoint)
    logger.info("Modello caricato con successo!")

    # Carica dataset
    all_questions = load_qa_data(args.nuscenes_val_json, bev_dir)

    # 1. Dataset Bilanciato
    by_cat = defaultdict(list)
    for q in all_questions:
        cat = q.get("template_type", "unknown")
        by_cat[cat].append(q)

    target_cats = ["exist", "count", "object", "status", "comparison"]
    balanced_items = []
    for cat in target_cats:
        pool = by_cat[cat]
        n_take = min(args.balanced_samples_per_cat, len(pool))
        balanced_items.extend(random.sample(pool, n_take))
    random.shuffle(balanced_items)

    logger.info(f"\n>>> 1. Valutazione su Distribuzione Bilanciata ({len(balanced_items)} campioni)...")
    bal_acc, bal_cat_accs, bal_samples = run_evaluation(
        model, tokenizer, balanced_items, bev_dir, device, desc="Balanced Eval"
    )

    # Shuffle Test
    shuffle_data = {}
    if args.run_shuffle:
        logger.info(f"\n>>> 2. Shuffle Test su Dataset Bilanciato ({len(balanced_items)} campioni)...")
        tokens = [it["sample_token"] for it in balanced_items]
        shuffled_tokens = tokens.copy()
        random.seed(999)
        random.shuffle(shuffled_tokens)
        for i in range(len(tokens)):
            if tokens[i] == shuffled_tokens[i] and len(tokens) > 1:
                j = (i + 1) % len(tokens)
                shuffled_tokens[i], shuffled_tokens[j] = shuffled_tokens[j], shuffled_tokens[i]

        shuf_acc, shuf_cat_accs, _ = run_evaluation(
            model, tokenizer, balanced_items, bev_dir, device, desc="Shuffle Test", shuffled_tokens=shuffled_tokens
        )
        delta_grounding = round(bal_acc - shuf_acc, 2)
        shuffle_data = {
            "real_bev_accuracy": bal_acc,
            "shuffled_bev_accuracy": shuf_acc,
            "visual_grounding_delta": delta_grounding,
            "shuffled_category_accuracies": shuf_cat_accs,
        }

    # 3. Dataset Naturale
    logger.info(f"\n>>> 3. Valutazione su Distribuzione Naturale ({args.natural_samples} campioni)...")
    natural_items = random.sample(all_questions, min(args.natural_samples, len(all_questions)))
    nat_acc, nat_cat_accs, nat_samples = run_evaluation(
        model, tokenizer, natural_items, bev_dir, device, desc="Natural Eval"
    )

    # Stampa tabella riassuntiva
    print(f"\n{BOLD}{GREEN}========================================================================================{RESET}")
    print(f"{BOLD}{GREEN}  RISULTATI BENCHMARK NUSCENES-QA{RESET}")
    print(f"{BOLD}{GREEN}========================================================================================{RESET}")
    print(f"  {BOLD}Accuratezza Globale Bilanciata (Macro-Avg):{RESET} {BOLD}{bal_acc:.2f}%{RESET}")
    print(f"  {BOLD}Accuratezza Globale Naturale (True-Dist):{RESET}   {BOLD}{nat_acc:.2f}%{RESET}")
    if args.run_shuffle:
        print(f"  {BOLD}Shuffle Test Acc (BEV Random Permutata):{RESET}  {shuf_acc:.2f}%")
        print(f"  {BOLD}Delta Dipendenza Visiva (Grounding):{RESET}      {GREEN if delta_grounding > 0 else YELLOW}{delta_grounding:+.2f}%{RESET}")

    print(f"\n  {BOLD}{'Categoria':<15} | {'Bilanciata':<12} | {'Naturale':<12} | {'Text-Only LLM':<15} | {'BeLLA SOTA':<12}{RESET}")
    print(f"  {'-'*75}")
    for cat in target_cats:
        b_acc = bal_cat_accs.get(cat, 0.0)
        n_acc = nat_cat_accs.get(cat, 0.0)
        t_acc = PUBLISHED_BASELINES["Text-Only LLM (Zero-Shot)"].get(cat, 0.0)
        bella = PUBLISHED_BASELINES["BeLLA (State-of-the-Art)"].get(cat, 0.0)
        print(f"  {cat:<15} | {b_acc:>10.1f}% | {n_acc:>10.1f}% | {t_acc:>13.1f}% | {bella:>10.1f}%")
    print(f"  {'-'*75}")
    print(f"  {BOLD}{'OVERALL':<15} | {bal_acc:>10.1f}% | {nat_acc:>10.1f}% | {19.6:>13.1f}% | {66.8:>10.1f}%{RESET}")
    print(f"{GREEN}========================================================================================{RESET}\n")

    # Salva report JSON
    report = {
        "checkpoint": args.checkpoint,
        "balanced_evaluation": {
            "total_samples": len(balanced_items),
            "overall_accuracy": bal_acc,
            "category_accuracies": bal_cat_accs,
            "sample_predictions": bal_samples,
        },
        "natural_evaluation": {
            "total_samples": len(natural_items),
            "overall_accuracy": nat_acc,
            "category_accuracies": nat_cat_accs,
            "sample_predictions": nat_samples,
        },
        "shuffle_test": shuffle_data,
        "comparisons": PUBLISHED_BASELINES,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report completo salvato in: {report_path}")


if __name__ == "__main__":
    main()
