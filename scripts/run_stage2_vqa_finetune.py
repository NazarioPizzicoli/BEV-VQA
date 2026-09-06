#!/usr/bin/env python3
"""
Block 5: Stage 2 End-to-End Multimodal VQA Fine-Tuning.

Finetuning congiunto del Proiettore Visivo e degli Adattatori LoRA:
- Inizializzazione: Proiettore preallenato dallo Stage 1 (stage1_projector_best.pt)
- Modello: BEVVLM (Qwen2.5-3B-Instruct + LoRA + DeeperConvProjector)
- Modalita: Stage 2 (Proiettore + LoRA trainabili: ~36.7M parametri, LLM base congelato)
- Dati: NuScenes-QA (ragionamento spaziale categorico) + DriveLM (percezione e guida)
- Valutazione:
  - Accuratezza su NuScenes-QA val suddivisa per categoria (count, exist, status, object, comparison)
  - Confronto diretto con la baseline linguistica (Text-Only: 19.6%)
  - Shuffle Test (BEV permutata vs reale) per certificare la reale percezione visiva
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

from bev_vqa.data.collate import collate_train, collate_eval
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
logger = logging.getLogger("stage2_vqa")

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Baseline linguistica (Block 3 - Text Only Zero-Shot)
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


def evaluate_vqa_accuracy(
    model: BEVVLM,
    tokenizer,
    val_dataset: BEVQADataset,
    device: torch.device,
    max_samples: int = 200,
    max_new_tokens: int = 15,
) -> Dict:
    """Valuta l'accuratezza di generazione risposte su NuScenes-QA val."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Campionamento bilanciato tra le 5 categorie per confronto 1:1 con la baseline
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

        with torch.no_grad():
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

        # Match check identico a Block 3: pred_norm == gt_answer oppure gt_answer in pred_norm.split()
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

    acc_overall = (correct / total * 100.0) if total > 0 else 0.0
    cat_accuracies = {}
    for c, c_tot in cat_total.items():
        cat_accuracies[c] = round(cat_correct[c] / c_tot * 100.0, 1) if c_tot > 0 else 0.0

    return {
        "overall_accuracy": round(acc_overall, 2),
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
    """
    Shuffle Test: Confronta l'accuratezza con la BEV corretta vs una BEV permutata.
    """
    logger.info(f"Esecuzione Shuffle Test su {num_samples} campioni...")
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Campionamento bilanciato tra le 5 categorie
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

            # BEV reale
            gen_real = model.generate(
                bev=bev_real_t,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            pred_real = normalize_answer(tokenizer.decode(gen_real[0], skip_special_tokens=True).strip())

            # BEV permutata
            gen_shuf = model.generate(
                bev=bev_shuf_t,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=15,
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

    real_acc = (real_correct / max(total, 1)) * 100.0
    shuf_acc = (shuffled_correct / max(total, 1)) * 100.0
    delta = real_acc - shuf_acc

    return {
        "num_tested": total,
        "real_bev_accuracy": round(real_acc, 2),
        "shuffled_bev_accuracy": round(shuf_acc, 2),
        "visual_dependency_delta": round(delta, 2),
        "visual_grounded": delta > 0.0,
        "examples": examples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 End-to-End VQA Finetuning")
    parser.add_argument("--nuscenes-train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_train.json")
    parser.add_argument("--nuscenes-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    parser.add_argument("--drivelm-train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_train.json")
    parser.add_argument("--drivelm-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_val.json")
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh")
    parser.add_argument("--stage1-projector", type=str, default="checkpoints/stage1/stage1_projector_best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage2")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16, help="Passi accumulo (effective batch size = 16)")
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=150)
    parser.add_argument("--final-eval-samples", type=int, default=300)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/stage2_vqa_report.json")
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
    print(f"{BOLD}{CYAN}  STAGE 2 VQA FINE-TUNING (PROJECTOR + LORA) - BLOCK 5{RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Dispositivo:             {device}")
    print(f"  LLM Backbone:            {args.llm_path}")
    print(f"  Architettura Projector:  {args.arch_type} ({args.num_tokens} visual tokens)")
    print(f"  Stage 1 Projector Pesi:  {args.stage1_projector}")
    print(f"  LR Projector / LoRA:     {args.lr_projector:.1e} / {args.lr_lora:.1e}")
    print(f"  Batch size effettivo:    {args.batch_size * args.grad_accum} (batch={args.batch_size}, accum={args.grad_accum})")
    print(f"  Passi di ottimizzazione: {args.max_steps}")
    print(f"  Output Checkpoint:       {output_dir}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    # 1. Tokenizer
    logger.info("Inizializzazione Tokenizer...")
    tokenizer = build_tokenizer(args.llm_path)

    # 2. Datasets
    logger.info("Caricamento Dataset NuScenes-QA train...")
    nuscenes_train = BEVQADataset(
        json_path=args.nuscenes_train_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=1,
        fraction=args.fraction,
    )
    logger.info(f"NuScenes-QA train: {len(nuscenes_train):,} campioni")

    logger.info("Caricamento Dataset DriveLM train...")
    drivelm_train = BEVQADataset(
        json_path=args.drivelm_train_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=1,
        fraction=args.fraction,
    )
    logger.info(f"DriveLM train: {len(drivelm_train):,} campioni")

    train_dataset = BEVMixedDataset([nuscenes_train, drivelm_train])
    logger.info(f"Dataset Misto Totale Train: {len(train_dataset):,} campioni")

    logger.info("Caricamento Dataset NuScenes-QA val per valutazione...")
    nuscenes_val = BEVQADataset(
        json_path=args.nuscenes_val_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="val",
        num_bev_tokens=1,
        fraction=1.0,
    )
    logger.info(f"NuScenes-QA val: {len(nuscenes_val):,} campioni")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
        pin_memory=True,
        drop_last=True,
    )

    # 3. Modello BEVVLM
    logger.info("Inizializzazione modello BEVVLM...")
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(
            arch_type=args.arch_type,
            num_tokens=args.num_tokens,
        ),
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()

    if os.path.exists(args.stage1_projector):
        logger.info(f"Caricamento pesi Stage 1 Projector da: {args.stage1_projector}")
        model.load_projector(args.stage1_projector)
    else:
        logger.warning(f"File {args.stage1_projector} non trovato! Inizializzazione da zero.")

    # ATTIVAZIONE STAGE 2 (Projector + LoRA)
    model.set_stage(2)
    model.llm.gradient_checkpointing_enable()
    logger.info("Gradient Checkpointing attivato su LLM per efficienza VRAM.")
    param_counts = model.count_trainable_parameters()
    logger.info(
        f"Parametri trainabili Stage 2: Proiettore={param_counts['projector_trainable']:,} | "
        f"LoRA={param_counts['llm_trainable']:,} | TOTALE={param_counts['total_trainable']:,} "
        f"({param_counts['total_trainable']/param_counts['total_params']*100:.2f}% del modello)"
    )

    # 4. Ottimizzatore
    proj_params = list(model.projector.parameters())
    lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": args.lr_projector, "weight_decay": args.weight_decay},
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": args.weight_decay},
    ], betas=(0.9, 0.95))

    total_opt_steps = args.max_steps
    num_warmup = max(1, int(total_opt_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_opt_steps,
    )

    # 5. Valutazione Iniziale
    logger.info("Valutazione Accuratezza iniziale pre-Stage 2...")
    init_eval = evaluate_vqa_accuracy(model, tokenizer, nuscenes_val, device, max_samples=args.eval_samples)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Accuratezza iniziale NuScenes-QA val: {init_eval['overall_accuracy']:.1f}%")
    logger.info(f"  Dettaglio categorie: {init_eval['category_accuracies']}")

    # 6. Training Loop Stage 2
    logger.info(f"{BOLD}Inizio Training Stage 2 ({total_opt_steps} step)...{RESET}")
    global_step = 0
    accumulated_loss = 0.0
    history = []
    best_acc = init_eval["overall_accuracy"]
    start_time = time.time()

    pbar = tqdm(total=total_opt_steps, desc="Stage 2 Finetune", dynamic_ncols=True)
    optimizer.zero_grad()
    model.train()

    stop_training = False
    for epoch in range(10):
        if stop_training:
            break
        for batch_idx, batch in enumerate(train_loader):
            bev = batch["bev"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                bev=bev,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / args.grad_accum
            loss.backward()
            accumulated_loss += loss.item()
            del outputs, loss, bev, input_ids, attention_mask, labels

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                cur_loss = accumulated_loss
                accumulated_loss = 0.0
                cur_lr = scheduler.get_last_lr()[1]

                pbar.update(1)
                pbar.set_postfix({"loss": f"{cur_loss:.4f}", "gnorm": f"{grad_norm:.2f}", "lr": f"{cur_lr:.2e}"})

                history.append({
                    "step": global_step,
                    "train_loss": cur_loss,
                    "grad_norm": grad_norm,
                    "lr": cur_lr,
                })

                if global_step % args.val_steps == 0 or global_step == total_opt_steps:
                    val_eval = evaluate_vqa_accuracy(model, tokenizer, nuscenes_val, device, max_samples=args.eval_samples)
                    cur_acc = val_eval["overall_accuracy"]
                    logger.info(
                        f"\nStep {global_step}/{total_opt_steps} | "
                        f"Train Loss: {cur_loss:.4f} | "
                        f"NuScenes-QA Acc: {cur_acc:.1f}% (Baseline: {TEXT_BASELINE_ACC['overall']}%) | "
                        f"Grad Norm: {grad_norm:.2f}"
                    )
                    logger.info(f"  Categorie: {val_eval['category_accuracies']}")

                    history[-1]["val_accuracy"] = cur_acc
                    history[-1]["category_accuracies"] = val_eval["category_accuracies"]

                    if cur_acc > best_acc:
                        best_acc = cur_acc
                        best_dir = output_dir / "best_model"
                        model.save_checkpoint(str(best_dir))
                        logger.info(f"  {GREEN}★ Nuovo record accuratezza ({cur_acc:.1f}%)! Salvato: {best_dir}{RESET}")

                if global_step >= total_opt_steps:
                    stop_training = True
                    break

    pbar.close()
    elapsed = time.time() - start_time

    # 7. Valutazione Finale Completa
    logger.info(f"\nValutazione finale su {args.final_eval_samples} campioni di NuScenes-QA val...")
    final_eval = evaluate_vqa_accuracy(
        model, tokenizer, nuscenes_val, device, max_samples=args.final_eval_samples
    )
    final_acc = final_eval["overall_accuracy"]
    visual_gain = round(final_acc - TEXT_BASELINE_ACC["overall"], 2)

    # 8. Shuffle Test
    shuffle_results = run_shuffle_test(model, tokenizer, nuscenes_val, device, num_samples=150)

    # 9. Stampa Risultati
    print(f"\n{BOLD}{GREEN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  STAGE 2 VQA RISULTATI UFFICIALI & CONFRONTO CON LA BASELINE{RESET}")
    print(f"{BOLD}{GREEN}======================================================================{RESET}")
    print(f"  Accuratezza Finale VLM:   {BOLD}{final_acc:.1f}%{RESET} ({final_eval['total_evaluated']} campioni)")
    print(f"  Baseline Text-Only:       {TEXT_BASELINE_ACC['overall']:.1f}%")
    gain_color = GREEN if visual_gain > 0 else YELLOW
    print(f"  {gain_color}{BOLD}Guadagno Visivo (Delta):  {visual_gain:+.1f}%{RESET}")
    print(f"\n  Dettaglio per Categoria:")
    print(f"  {'Categoria':<15} | {'VLM Acc':<10} | {'Text Baseline':<15} | {'Delta':<10}")
    print(f"  {'-'*55}")
    for cat, base_v in TEXT_BASELINE_ACC.items():
        if cat == "overall":
            continue
        vlm_v = final_eval["category_accuracies"].get(cat, 0.0)
        d = round(vlm_v - base_v, 1)
        c_str = GREEN if d > 0 else RESET
        print(f"  {cat:<15} | {vlm_v:>8.1f}% | {base_v:>13.1f}% | {c_str}{d:>+8.1f}%{RESET}")

    print(f"\n{BOLD}{CYAN}----------------------------------------------------------------------{RESET}")
    print(f"{BOLD}{CYAN}  SHUFFLE TEST (VERIFICA ANCORAGGIO VISIVO){RESET}")
    print(f"{BOLD}{CYAN}----------------------------------------------------------------------{RESET}")
    print(f"  Accuratezza BEV Reale:      {shuffle_results['real_bev_accuracy']:.1f}%")
    print(f"  Accuratezza BEV Shuffled:   {shuffle_results['shuffled_bev_accuracy']:.1f}%")
    print(f"  Delta Dipendenza Visiva:    {GREEN if shuffle_results['visual_grounded'] else YELLOW}{shuffle_results['visual_dependency_delta']:+.1f}%{RESET}")
    if shuffle_results["visual_grounded"]:
        print(f"  {GREEN}✓ PASS: La BEV reale produce un boost significativo rispetto alla BEV permutata!{RESET}")
    else:
        print(f"  {YELLOW}⚠ ATTENZIONE: La dipendenza dalla BEV permutata non e sufficientemente marcata.{RESET}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    # Salvataggio Pesi Finali
    latest_dir = output_dir / "latest_model"
    model.save_checkpoint(str(latest_dir))
    if not (output_dir / "best_model").exists():
        model.save_checkpoint(str(output_dir / "best_model"))

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    # Salva Report JSON completo
    report = {
        "stage": 2,
        "description": "End-to-End Multimodal VQA Fine-Tuning",
        "llm_backbone": args.llm_path,
        "projector_arch": args.arch_type,
        "trainable_parameters": param_counts["total_trainable"],
        "projector_parameters": param_counts["projector_trainable"],
        "lora_parameters": param_counts["llm_trainable"],
        "total_steps": global_step,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "training_time_seconds": round(elapsed, 2),
        "initial_accuracy": init_eval["overall_accuracy"],
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "text_baseline_accuracy": TEXT_BASELINE_ACC["overall"],
        "visual_gain_delta": visual_gain,
        "category_breakdown": {
            cat: {
                "vlm": final_eval["category_accuracies"].get(cat, 0.0),
                "text_baseline": TEXT_BASELINE_ACC.get(cat, 0.0),
                "delta": round(final_eval["category_accuracies"].get(cat, 0.0) - TEXT_BASELINE_ACC.get(cat, 0.0), 1),
            }
            for cat in ["count", "exist", "status", "object", "comparison"]
        },
        "shuffle_test": shuffle_results,
        "sample_predictions": final_eval["sample_predictions"],
        "checkpoints": {
            "best_model": str(output_dir / "best_model"),
            "latest_model": str(latest_dir),
            "tokenizer": str(tokenizer_dir),
        },
        "history_sample": history[::max(1, len(history) // 50)],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report Stage 2 salvato in: {report_path}")


if __name__ == "__main__":
    main()
