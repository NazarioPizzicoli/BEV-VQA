#!/usr/bin/env python3
"""
Stage 2 Grounded VQA Training: Enforcing Visual Dependency & Eliminating Language Shortcuts.

Strategia in due fasi per superare il bias statistico e certificare il grounding visivo:
- Fase 1 (Projector Grounding, 400 step):
  - LLM completamente congelato (Stage 1).
  - Addestramento esclusivo del Proiettore su NuScenes-QA + DriveLM (lr = 3e-4).
  - Forza i 32 visual tokens a trasmettere le informazioni necessarie all'LLM senza che
    quest'ultimo possa memorizzare shortcut linguistici nei propri pesi.
- Fase 2 (Joint Alignment con LoRA Delicato, 300 step):
  - Attivazione LoRA (Stage 2) con learning rate asimmetrico:
    lr_projector = 1e-4, lr_lora = 1.5e-5 (oltre 10 volte inferiore al proiettore).
  - Adatta la sintassi e la concisione dell'LLM senza distruggere la rappresentazione visiva.
- Metriche di Valutazione:
  - Accuratezza bilanciata su NuScenes-QA (5 categorie).
  - Test su Coppie Contrastanti (stessa domanda, scene diverse, risposte opposte):
    misura diretta di quante volte il modello cambia risposta al variare della BEV.
  - Shuffle Test globale (BEV reale vs permutata).
"""

import argparse
from collections import defaultdict
import json
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from bev_vqa.data.collate import collate_train
from bev_vqa.data.dataset import (
    BEVQADataset,
    BEVMixedDataset,
    build_tokenizer,
    normalize_answer,
    VQA_SYSTEM_PROMPT,
)
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grounded_vqa")

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

TEXT_BASELINE_ACC = {
    "overall": 19.6,
    "count": 0.0,
    "status": 6.0,
    "object": 12.0,
    "exist": 38.0,
    "comparison": 42.0,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_contrastive_pairs(val_dataset: BEVQADataset, max_pairs: int = 50) -> List[Tuple[Dict, Dict]]:
    """
    Estrae coppie di campioni di validazione con la STESSA identica domanda
    ma con risposte diverse in scene diverse (es. yes vs no, oppure car vs truck).
    """
    by_question = defaultdict(list)
    for it in val_dataset.data:
        by_question[it["question"]].append(it)

    pairs = []
    for q, items in by_question.items():
        ans_map = {}
        for it in items:
            a = normalize_answer(it["answer"])
            if a not in ans_map:
                ans_map[a] = it
            if len(ans_map) >= 2:
                break
        if len(ans_map) >= 2:
            keys = list(ans_map.keys())
            pairs.append((ans_map[keys[0]], ans_map[keys[1]]))
        if len(pairs) >= max_pairs:
            break

    return pairs


def evaluate_contrastive_pairs(
    model: BEVVLM,
    tokenizer,
    contrast_pairs: List[Tuple[Dict, Dict]],
    val_dataset: BEVQADataset,
    device: torch.device,
    max_new_tokens: int = 10,
) -> Dict:
    """
    Valuta se il modello cambia attivamente la predizione quando riceve la BEV corretta vs scambiata.
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    changed_count = 0
    real_correct = 0
    swapped_correct = 0
    total = len(contrast_pairs)

    with torch.no_grad():
        for it1, it2 in contrast_pairs:
            q = it1["question"]
            gt1 = normalize_answer(it1["answer"])
            gt2 = normalize_answer(it2["answer"])

            bev1 = val_dataset._load_bev_features(it1["sample_token"]).to(device)
            bev2 = val_dataset._load_bev_features(it2["sample_token"]).to(device)

            prompt = (
                f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<|bev|>\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            out1 = model.generate(
                bev=bev1,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            out2 = model.generate(
                bev=bev2,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            p1 = normalize_answer(tokenizer.decode(out1[0], skip_special_tokens=True).strip())
            p2 = normalize_answer(tokenizer.decode(out2[0], skip_special_tokens=True).strip())

            if p1 != p2:
                changed_count += 1

            if p1 == gt1 or gt1 in p1.split():
                real_correct += 1
            if p2 == gt2 or gt2 in p2.split():
                real_correct += 1

            # Scambiati: se diamo bev2 alla domanda it1 e bev1 alla domanda it2
            if p2 == gt1 or gt1 in p2.split():
                swapped_correct += 1
            if p1 == gt2 or gt2 in p1.split():
                swapped_correct += 1

            del bev1, bev2, out1, out2

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    tot_questions = total * 2
    changed_pct = round(changed_count / max(total, 1) * 100.0, 1)
    acc_real = round(real_correct / max(tot_questions, 1) * 100.0, 1)
    acc_swapped = round(swapped_correct / max(tot_questions, 1) * 100.0, 1)
    delta = round(acc_real - acc_swapped, 1)

    return {
        "num_pairs": total,
        "changed_predictions_pct": changed_pct,
        "real_bev_accuracy": acc_real,
        "swapped_bev_accuracy": acc_swapped,
        "contrastive_delta": delta,
    }


def evaluate_vqa_accuracy(
    model: BEVVLM,
    tokenizer,
    val_dataset: BEVQADataset,
    device: torch.device,
    max_samples: int = 150,
    max_new_tokens: int = 10,
) -> Dict:
    """Valutazione bilanciata sulle 5 categorie."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    by_cat = defaultdict(list)
    for it in val_dataset.data:
        by_cat[it.get("template_type", "unknown")].append(it)

    eval_items = []
    target_cats = ["exist", "count", "object", "status", "comparison"]
    n_per_cat = max(1, max_samples // len(target_cats))
    for cat in target_cats:
        eval_items.extend(by_cat[cat][:n_per_cat])

    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    sample_preds = []

    with torch.no_grad():
        for item in eval_items:
            token = item["sample_token"]
            question = item["question"]
            gt_answer = normalize_answer(item["answer"])
            cat = item.get("template_type", "unknown")

            bev_feat = val_dataset._load_bev_features(token)
            if bev_feat is None:
                continue

            bev = bev_feat.unsqueeze(0).to(device) if bev_feat.dim() == 3 else bev_feat.to(device)

            prompt_text = (
                f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<|bev|>\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

            gen_ids = model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            pred_norm = normalize_answer(pred_text)
            is_match = (pred_norm == gt_answer) or (gt_answer in pred_norm.split())

            if is_match:
                correct += 1
                cat_correct[cat] += 1
            total += 1
            cat_total[cat] += 1

            if len(sample_preds) < 5:
                sample_preds.append({
                    "sample_token": token,
                    "category": cat,
                    "question": question,
                    "ground_truth": gt_answer,
                    "predicted": pred_norm,
                    "match": is_match,
                })

            del bev, gen_ids

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    acc_overall = round(correct / max(total, 1) * 100.0, 2)
    cat_accuracies = {}
    for c, c_tot in cat_total.items():
        cat_accuracies[c] = round(cat_correct[c] / c_tot * 100.0, 1) if c_tot > 0 else 0.0

    return {
        "overall_accuracy": acc_overall,
        "total_evaluated": total,
        "category_accuracies": cat_accuracies,
        "sample_predictions": sample_preds,
    }


def run_shuffle_test(
    model: BEVVLM,
    tokenizer,
    val_dataset: BEVQADataset,
    device: torch.device,
    num_samples: int = 150,
) -> Dict:
    """Shuffle Test bilanciato."""
    logger.info(f"Esecuzione Shuffle Test su {num_samples} campioni...")
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    by_cat = defaultdict(list)
    for it in val_dataset.data:
        by_cat[it.get("template_type", "unknown")].append(it)

    items = []
    target_cats = ["exist", "count", "object", "status", "comparison"]
    n_per_cat = max(1, num_samples // len(target_cats))
    for cat in target_cats:
        items.extend(by_cat[cat][:n_per_cat])

    all_tokens = [it["sample_token"] for it in items]
    shuffled_tokens = all_tokens.copy()
    random.seed(999)
    random.shuffle(shuffled_tokens)
    for i in range(len(all_tokens)):
        if all_tokens[i] == shuffled_tokens[i] and len(all_tokens) > 1:
            j = (i + 1) % len(all_tokens)
            shuffled_tokens[i], shuffled_tokens[j] = shuffled_tokens[j], shuffled_tokens[i]

    real_correct = 0
    shuffled_correct = 0
    total = 0
    examples = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(items, desc="  Shuffle Test")):
            q = item["question"]
            gt = normalize_answer(item["answer"])

            bev_real = val_dataset._load_bev_features(item["sample_token"])
            bev_shuf = val_dataset._load_bev_features(shuffled_tokens[i])

            if bev_real is None or bev_shuf is None:
                continue

            bev_real_t = bev_real.unsqueeze(0).to(device) if bev_real.dim() == 3 else bev_real.to(device)
            bev_shuf_t = bev_shuf.unsqueeze(0).to(device) if bev_shuf.dim() == 3 else bev_shuf.to(device)

            prompt_text = (
                f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<|bev|>\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

            gen_real = model.generate(
                bev=bev_real_t,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            pred_real = normalize_answer(tokenizer.decode(gen_real[0], skip_special_tokens=True).strip())

            gen_shuf = model.generate(
                bev=bev_shuf_t,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            pred_shuf = normalize_answer(tokenizer.decode(gen_shuf[0], skip_special_tokens=True).strip())

            match_real = (pred_real == gt) or (gt in pred_real.split())
            match_shuf = (pred_shuf == gt) or (gt in pred_shuf.split())

            if match_real:
                real_correct += 1
            if match_shuf:
                shuffled_correct += 1
            total += 1

            if len(examples) < 3:
                examples.append({
                    "question": q,
                    "ground_truth": gt,
                    "pred_real_bev": pred_real,
                    "match_real": match_real,
                    "pred_shuffled_bev": pred_shuf,
                    "match_shuffled": match_shuf,
                })

            del bev_real_t, bev_shuf_t, gen_real, gen_shuf

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    real_acc = round((real_correct / max(total, 1)) * 100.0, 2)
    shuf_acc = round((shuffled_correct / max(total, 1)) * 100.0, 2)
    delta = round(real_acc - shuf_acc, 2)

    return {
        "num_tested": total,
        "real_bev_accuracy": real_acc,
        "shuffled_bev_accuracy": shuf_acc,
        "visual_dependency_delta": delta,
        "visual_grounded": delta > 0.0,
        "examples": examples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Grounded VQA Training")
    parser.add_argument("--nuscenes-train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_train.json")
    parser.add_argument("--nuscenes-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    parser.add_argument("--drivelm-train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_train.json")
    parser.add_argument("--drivelm-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_val.json")
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh")
    parser.add_argument("--stage1-projector", type=str, default="checkpoints/stage1/stage1_projector_best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage2_grounded")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--phase1-steps", type=int, default=400, help="Passi di solo Projector con LLM congelato")
    parser.add_argument("--phase2-steps", type=int, default=300, help="Passi congiunti Projector + LoRA con LR basso")
    parser.add_argument("--lr-phase1", type=float, default=3e-4)
    parser.add_argument("--lr-phase2-proj", type=float, default=1e-4)
    parser.add_argument("--lr-phase2-lora", type=float, default=1.5e-5)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/stage2_grounded_vqa_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}  STAGE 2 GROUNDED VQA TRAINING (ELIMINATING LANGUAGE SHORTCUTS){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Dispositivo:                {device}")
    print(f"  Fase 1 (Projector-Only):    {args.phase1_steps} step (lr={args.lr_phase1:.1e}, LLM congelato)")
    print(f"  Fase 2 (Joint LoRA Delicato): {args.phase2_steps} step (lr_proj={args.lr_phase2_proj:.1e}, lr_lora={args.lr_phase2_lora:.1e})")
    print(f"  Batch size effettivo:       {args.batch_size * args.grad_accum} (batch={args.batch_size}, accum={args.grad_accum})")
    print(f"  Output Checkpoint:          {output_dir}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    tokenizer = build_tokenizer(args.llm_path)

    logger.info("Caricamento dataset di training...")
    nuscenes_train = BEVQADataset(
        json_path=args.nuscenes_train_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="train",
        fraction=1.0,
    )
    drivelm_train = BEVQADataset(
        json_path=args.drivelm_train_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="train",
        fraction=1.0,
    )
    train_dataset = BEVMixedDataset([nuscenes_train, drivelm_train])
    logger.info(f"Dataset Misto Totale Train: {len(train_dataset):,} campioni")

    logger.info("Caricamento dataset di validazione NuScenes-QA val...")
    nuscenes_val = BEVQADataset(
        json_path=args.nuscenes_val_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="val",
        fraction=1.0,
    )
    contrast_pairs = build_contrastive_pairs(nuscenes_val, max_pairs=40)
    logger.info(f"Costruite {len(contrast_pairs)} coppie contrastanti per verifica della sensibilità visiva.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
        pin_memory=True,
        drop_last=True,
    )

    # Inizializzazione Modello
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(arch_type=args.arch_type, num_tokens=args.num_tokens),
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()

    if os.path.exists(args.stage1_projector):
        logger.info(f"Caricamento pesi Stage 1 Projector da: {args.stage1_projector}")
        model.load_projector(args.stage1_projector)

    model.llm.gradient_checkpointing_enable()

    # Valutazione iniziale di riferimento
    model.set_stage(1)
    init_eval = evaluate_vqa_accuracy(model, tokenizer, nuscenes_val, device, max_samples=args.eval_samples)
    init_contrast = evaluate_contrastive_pairs(model, tokenizer, contrast_pairs, nuscenes_val, device)
    logger.info(f"Accuratezza Iniziale: {init_eval['overall_accuracy']:.1f}% | Sensibilità Visiva su Coppie: {init_contrast['changed_predictions_pct']:.1f}% (Delta: {init_contrast['contrastive_delta']:+.1f}%)")

    # =========================================================================
    # FASE 1: PROJECTOR-ONLY VQA TRAINING (LLM FROZEN)
    # =========================================================================
    logger.info(f"\n{BOLD}{GREEN}>>> INIZIO FASE 1: Projector-Centric Grounding ({args.phase1_steps} step, LLM CONGELATO)...{RESET}")
    model.set_stage(1)
    opt_phase1 = torch.optim.AdamW(model.projector.parameters(), lr=args.lr_phase1, weight_decay=0.01)
    sched_phase1 = get_cosine_schedule_with_warmup(opt_phase1, num_warmup_steps=20, num_training_steps=args.phase1_steps)

    step_p1 = 0
    accum_loss = 0.0
    best_score = -999.0
    history = []

    pbar = tqdm(total=args.phase1_steps, desc="Fase 1 (Proj Only)", dynamic_ncols=True)
    stop_p1 = False
    train_iter = iter(train_loader)

    while not stop_p1:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bev = batch["bev"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(bev=bev, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum_loss += loss.item()
        del out, bev, input_ids, attention_mask, labels

        if (step_p1 + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            opt_phase1.step()
            sched_phase1.step()
            opt_phase1.zero_grad()

            cur_opt = (step_p1 + 1) // args.grad_accum
            pbar.update(1)
            pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{sched_phase1.get_last_lr()[0]:.2e}"})

            history.append({
                "phase": 1,
                "step": cur_opt,
                "loss": accum_loss,
            })
            accum_loss = 0.0

            if cur_opt % args.val_steps == 0 or cur_opt == args.phase1_steps:
                val_res = evaluate_vqa_accuracy(model, tokenizer, nuscenes_val, device, max_samples=args.eval_samples)
                cont_res = evaluate_contrastive_pairs(model, tokenizer, contrast_pairs, nuscenes_val, device)
                logger.info(
                    f"\n[Fase 1 Step {cur_opt}/{args.phase1_steps}] Acc: {val_res['overall_accuracy']:.1f}% | "
                    f"Sensibilità Visiva: {cont_res['changed_predictions_pct']:.1f}% | "
                    f"Delta Reale vs Scambiata: {cont_res['contrastive_delta']:+.1f}%"
                )
                logger.info(f"  Categorie: {val_res['category_accuracies']}")

                composite_score = val_res["overall_accuracy"] + 2.0 * cont_res["contrastive_delta"]
                if composite_score > best_score:
                    best_score = composite_score
                    model.save_checkpoint(str(output_dir / "best_model_phase1"))
                    logger.info(f"  {GREEN}★ Nuovo miglior checkpoint Fase 1 salvato!{RESET}")

            if cur_opt >= args.phase1_steps:
                stop_p1 = True
                break

        step_p1 += 1

    pbar.close()

    # =========================================================================
    # FASE 2: JOINT TRAINING CON LORA DELICATO
    # =========================================================================
    logger.info(f"\n{BOLD}{GREEN}>>> INIZIO FASE 2: Joint Alignment ({args.phase2_steps} step, LoRA LR={args.lr_phase2_lora:.1e})...{RESET}")
    model.set_stage(2)
    model.llm.gradient_checkpointing_enable()

    opt_phase2 = torch.optim.AdamW([
        {"params": list(model.projector.parameters()), "lr": args.lr_phase2_proj, "weight_decay": 0.01},
        {"params": [p for n, p in model.llm.named_parameters() if p.requires_grad], "lr": args.lr_phase2_lora, "weight_decay": 0.01},
    ])
    sched_phase2 = get_cosine_schedule_with_warmup(opt_phase2, num_warmup_steps=15, num_training_steps=args.phase2_steps)

    step_p2 = 0
    accum_loss = 0.0
    pbar = tqdm(total=args.phase2_steps, desc="Fase 2 (Joint Delicato)", dynamic_ncols=True)
    stop_p2 = False

    while not stop_p2:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bev = batch["bev"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(bev=bev, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum_loss += loss.item()
        del out, bev, input_ids, attention_mask, labels

        if (step_p2 + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            opt_phase2.step()
            sched_phase2.step()
            opt_phase2.zero_grad()

            cur_opt = (step_p2 + 1) // args.grad_accum
            pbar.update(1)
            pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr_lora": f"{sched_phase2.get_last_lr()[1]:.2e}"})

            history.append({
                "phase": 2,
                "step": cur_opt,
                "loss": accum_loss,
            })
            accum_loss = 0.0

            if cur_opt % args.val_steps == 0 or cur_opt == args.phase2_steps:
                val_res = evaluate_vqa_accuracy(model, tokenizer, nuscenes_val, device, max_samples=args.eval_samples)
                cont_res = evaluate_contrastive_pairs(model, tokenizer, contrast_pairs, nuscenes_val, device)
                logger.info(
                    f"\n[Fase 2 Step {cur_opt}/{args.phase2_steps}] Acc: {val_res['overall_accuracy']:.1f}% | "
                    f"Sensibilità Visiva: {cont_res['changed_predictions_pct']:.1f}% | "
                    f"Delta Reale vs Scambiata: {cont_res['contrastive_delta']:+.1f}%"
                )
                logger.info(f"  Categorie: {val_res['category_accuracies']}")

                composite_score = val_res["overall_accuracy"] + 2.0 * cont_res["contrastive_delta"]
                if composite_score > best_score:
                    best_score = composite_score
                    model.save_checkpoint(str(output_dir / "best_model"))
                    logger.info(f"  {GREEN}★ Nuovo miglior checkpoint globale salvato!{RESET}")

            if cur_opt >= args.phase2_steps:
                stop_p2 = True
                break

        step_p2 += 1

    pbar.close()

    # =========================================================================
    # VALUTAZIONE FINALE COMPLETA & SHUFFLE TEST
    # =========================================================================
    logger.info("\nCaricamento del miglior modello per benchmark finale...")
    if (output_dir / "best_model").exists():
        model.load_checkpoint(str(output_dir / "best_model"))
    elif (output_dir / "best_model_phase1").exists():
        model.load_checkpoint(str(output_dir / "best_model_phase1"))

    final_eval = evaluate_vqa_accuracy(model, tokenizer, nuscenes_val, device, max_samples=250)
    final_contrast = evaluate_contrastive_pairs(model, tokenizer, contrast_pairs, val_dataset=nuscenes_val, device=device)
    shuffle_results = run_shuffle_test(model, tokenizer, nuscenes_val, device, num_samples=150)

    print(f"\n{BOLD}{GREEN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  RISULTATI FINALI STAGE 2 GROUNDED VQA (GROUNDING VERIFICATO){RESET}")
    print(f"{BOLD}{GREEN}======================================================================{RESET}")
    print(f"  Accuratezza Finale VLM:         {BOLD}{final_eval['overall_accuracy']:.1f}%{RESET} (Baseline: {TEXT_BASELINE_ACC['overall']}%)")
    print(f"  Sensibilità Visiva su Coppie:   {BOLD}{final_contrast['changed_predictions_pct']:.1f}%{RESET} delle risposte cambiano con la BEV")
    print(f"  Accuratezza BEV Reale (Coppie): {final_contrast['real_bev_accuracy']:.1f}%")
    print(f"  Accuratezza BEV Scambiata:      {final_contrast['swapped_bev_accuracy']:.1f}%")
    print(f"  Delta Dipendenza Visiva Coppie: {GREEN if final_contrast['contrastive_delta'] > 0 else YELLOW}{final_contrast['contrastive_delta']:+.1f}%{RESET}")
    print(f"\n  Shuffle Test Globale (150 campioni bilanciati):")
    print(f"  - BEV Reale:                    {shuffle_results['real_bev_accuracy']:.1f}%")
    print(f"  - BEV Shuffled:                 {shuffle_results['shuffled_bev_accuracy']:.1f}%")
    print(f"  - Delta Visivo Globale:         {GREEN if shuffle_results['visual_grounded'] else YELLOW}{shuffle_results['visual_dependency_delta']:+.1f}%{RESET}")

    print(f"\n  Dettaglio Categorie NuScenes-QA:")
    print(f"  {'Categoria':<15} | {'VLM Acc':<10} | {'Text Baseline':<15} | {'Delta':<10}")
    print(f"  {'-'*55}")
    for cat, base_v in TEXT_BASELINE_ACC.items():
        if cat == "overall":
            continue
        vlm_v = final_eval["category_accuracies"].get(cat, 0.0)
        d = round(vlm_v - base_v, 1)
        c_str = GREEN if d > 0 else RESET
        print(f"  {cat:<15} | {vlm_v:>8.1f}% | {base_v:>13.1f}% | {c_str}{d:>+8.1f}%{RESET}")

    # Salva ultimo modello e tokenizer
    model.save_checkpoint(str(output_dir / "latest_model"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    report = {
        "stage": "2_grounded",
        "description": "Two-Stage Grounded VQA (Projector-First + Low-LR LoRA)",
        "phase1_steps": args.phase1_steps,
        "phase2_steps": args.phase2_steps,
        "final_accuracy": final_eval["overall_accuracy"],
        "text_baseline_accuracy": TEXT_BASELINE_ACC["overall"],
        "category_breakdown": final_eval["category_accuracies"],
        "contrastive_pairs_eval": final_contrast,
        "shuffle_test": shuffle_results,
        "sample_predictions": final_eval["sample_predictions"],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report Grounded VQA salvato in: {report_path}")


if __name__ == "__main__":
    main()
