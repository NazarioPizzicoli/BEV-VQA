#!/usr/bin/env python3
"""
SOTA NuScenes-QA Record Training Script with GradScaler & Early Stopping.

Design:
- Dataset: Natural NuScenes-QA (Full 233k dataset, true real-world distribution).
- LoRA: r=32, alpha=64, lr=1.5e-4 on all linear modules (59.9M params).
- Projector: DeeperConv (lr=1.0e-4) initialized from Stage 1 pretraining.
- Precision: Mixed Precision with dynamic torch.cuda.amp.GradScaler (zero overflow / zero NaNs).
- Duration: Up to 10,000 steps with Early Stopping (patience=8 evaluations = 2,000 steps).
- Checkpointing: Evaluates validation accuracy every 250 steps, saving best_model on new peaks.
"""

import argparse
from collections import defaultdict
import json
import logging
import math
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
    build_tokenizer,
    normalize_answer,
    VQA_SYSTEM_PROMPT,
    BEV_TOKEN,
)
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nuscenes_record")

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_val_accuracy(
    model: BEVVLM,
    tokenizer,
    val_items: List[Dict],
    bev_dir: Path,
    device: torch.device,
    num_tokens: int = 32,
    max_new_tokens: int = 10,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    val_bev_dir = bev_dir / "val"
    train_bev_dir = bev_dir / "train"

    with torch.no_grad():
        for it in val_items:
            token = it["sample_token"]
            p_val = val_bev_dir / f"{token}.pt"
            p_train = train_bev_dir / f"{token}.pt"
            p = p_val if p_val.exists() else p_train
            if not p.exists():
                continue

            bev_data = torch.load(p, map_location="cpu", weights_only=True)
            bev = bev_data["features_fused"]
            if bev.dim() == 3:
                bev = bev.unsqueeze(0)
            bev = bev.to(device)

            q = it["question"]
            gt = normalize_answer(it["answer"])
            cat = it.get("template_type", "unknown")

            bev_placeholder = BEV_TOKEN * num_tokens
            messages = [
                {"role": "system", "content": VQA_SYSTEM_PROMPT},
                {"role": "user", "content": f"{bev_placeholder}\n{q}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
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
            del bev, out

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    overall_acc = round(correct / max(total, 1) * 100.0, 2)
    cat_accs = {}
    for c, c_tot in cat_total.items():
        cat_accs[c] = round(cat_correct[c] / c_tot * 100.0, 1) if c_tot > 0 else 0.0

    return overall_acc, cat_accs


def parse_args():
    parser = argparse.ArgumentParser(description="SOTA NuScenes-QA Record Training with GradScaler & Early Stopping")
    parser.add_argument("--train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_train.json")
    parser.add_argument("--val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset/bev_features")
    parser.add_argument("--stage1-projector", type=str, default="checkpoints/sota/stage1/stage1_projector_best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sota_nuscenes_record")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16, help="Effective batch size = 16")
    parser.add_argument("--lr-projector", type=float, default=1.0e-4)
    parser.add_argument("--lr-lora", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=10000, help="Passi massimi di ottimizzazione")
    parser.add_argument("--val-steps", type=int, default=250, help="Frequenza di validazione per checkpointing")
    parser.add_argument("--val-samples-per-cat", type=int, default=50, help="50 x 5 = 250 campioni di validazione")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping: 8 check x 250 step = 2000 step")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None, help="Riprendi il training da un checkpoint esistente (es. checkpoints/.../checkpoint_step_4000)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/sota_nuscenes_record_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bev_dir = Path(args.bev_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}  SOTA NUSCENES-QA RECORD TRAINING (GRADSCALER + EARLY STOPPING){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Dispositivo:                {device}")
    print(f"  Dataset:                    NuScenes-QA Naturale ({args.train_json})")
    print(f"  LoRA Configurazione:        r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Learning Rates:             LoRA={args.lr_lora:.1e}, Projector={args.lr_projector:.1e}")
    print(f"  Max Steps / Patience:       {args.max_steps} step (Patience = {args.patience} check = {args.patience * args.val_steps} step)")
    print(f"  Effective Batch Size:       {args.batch_size * args.grad_accum}")
    print(f"  Stabilità Numerica:         GradScaler con Dynamic Loss Scaling abilitato")
    print(f"  Pesi Proiettore Iniziali:   {args.stage1_projector}")
    print(f"  Directory Checkpoints:      {output_dir}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    tokenizer = build_tokenizer(args.llm_path)

    # 1. Dataset Naturale di Training
    logger.info("Caricamento dataset NuScenes-QA train naturale...")
    train_dataset = BEVQADataset(
        json_path=args.train_json,
        bev_dir=bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=args.num_tokens,
        fraction=1.0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset NuScenes-QA Naturale: {len(train_dataset):,} campioni caricati.")

    # 2. Pool di Validazione Bilanciato
    logger.info("Preparazione pool di validazione bilanciato...")
    with open(args.val_json) as f:
        raw_val = json.load(f).get("questions", [])

    val_by_cat = defaultdict(list)
    val_bev_dir = bev_dir / "val"
    train_bev_dir = bev_dir / "train"
    for q in raw_val:
        token = q.get("sample_token")
        if (val_bev_dir / f"{token}.pt").exists() or (train_bev_dir / f"{token}.pt").exists():
            val_by_cat[q.get("template_type", "unknown")].append(q)

    val_eval_items = []
    target_cats = ["exist", "count", "object", "status", "comparison"]
    for cat in target_cats:
        pool = val_by_cat[cat]
        val_eval_items.extend(random.sample(pool, min(args.val_samples_per_cat, len(pool))))
    random.shuffle(val_eval_items)
    logger.info(f"Pool di validazione: {len(val_eval_items)} campioni pronti.")

    # 3. Costruzione Modello
    logger.info("Inizializzazione BEV-VLM con LoRA r=32, alpha=64...")
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(arch_type=args.arch_type, num_tokens=args.num_tokens),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()

    if args.resume_from:
        logger.info(f"Caricamento pesi completi da checkpoint di ripresa: {args.resume_from}")
        model.load_checkpoint(args.resume_from)
    elif os.path.exists(args.stage1_projector):
        logger.info(f"Caricamento pesi Stage 1 Projector da: {args.stage1_projector}")
        model.load_projector(args.stage1_projector)

    model.set_stage(2)  # Projector + LoRA attivi
    model.llm.gradient_checkpointing_enable()

    param_counts = model.count_trainable_parameters()
    logger.info(
        f"Parametri Trainabili: Proiettore={param_counts['projector_trainable']:,} | "
        f"LoRA={param_counts['llm_trainable']:,} | Totale={param_counts['total_trainable']:,}"
    )

    # 4. Ottimizzatore, GradScaler e Scheduler
    optimizer = torch.optim.AdamW([
        {"params": list(model.projector.parameters()), "lr": args.lr_projector, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.llm.named_parameters() if p.requires_grad], "lr": args.lr_lora, "weight_decay": args.weight_decay},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Baseline Zero-Shot iniziale
    logger.info("Valutazione accuratezza iniziale pre-training...")
    init_acc, init_cat_accs = evaluate_val_accuracy(
        model, tokenizer, val_eval_items, bev_dir, device, num_tokens=args.num_tokens
    )
    logger.info(f"Accuratezza Iniziale: {init_acc:.2f}% (Categorie: {init_cat_accs})")

    # 5. Training Loop con Early Stopping
    logger.info(f"\n{BOLD}{GREEN}>>> INIZIO SOTA RECORD TRAINING NUSCENES-QA ({args.max_steps} step max)...{RESET}")
    global_step = 0
    accum_loss = 0.0
    best_acc = init_acc
    patience_counter = 0
    history = []
    start_time = time.time()

    pbar = tqdm(total=args.max_steps, desc="SOTA NuScenes-QA Record", dynamic_ncols=True)
    train_iter = iter(train_loader)
    sub_step = 0
    early_stopped = False

    while global_step < args.max_steps and not early_stopped:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bev = batch["bev"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(bev=bev, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.grad_accum

        if not torch.isfinite(loss):
            logger.warning(f"[Step {global_step}] Loss non finita ({loss.item()})! Micro-batch scartato.")
            optimizer.zero_grad()
            del out, bev, input_ids, attention_mask, labels
            continue

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        del out, bev, input_ids, attention_mask, labels

        sub_step += 1
        if sub_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)

            # Verifica che tutti i gradienti siano finiti prima del grad clipping
            grads_finite = all(
                torch.isfinite(p.grad).all().item()
                for p in model.trainable_parameters()
                if p.grad is not None
            )

            if not grads_finite:
                logger.warning(f"[Step {global_step}] Gradienti non finiti (NaN/Inf) rilevati! Step annullato.")
                optimizer.zero_grad()
                scaler.update()  # dimezza lo scaler factor
                accum_loss = 0.0
                continue

            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad()
            if scale_before <= scale_after:
                scheduler.step()

            global_step += 1
            cur_loss = accum_loss
            accum_loss = 0.0
            cur_lr_lora = scheduler.get_last_lr()[1]

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "lr": f"{cur_lr_lora:.2e}",
                "best_acc": f"{best_acc:.1f}%",
                "patience": f"{patience_counter}/{args.patience}"
            })

            history.append({
                "step": global_step,
                "loss": cur_loss,
                "lr_lora": cur_lr_lora,
            })

            # Validazione Periodica
            if global_step % args.val_steps == 0 or global_step == args.max_steps:
                if math.isnan(cur_loss):
                    logger.error(f"[Step {global_step}] Emergenza: Loss media NaN! Interruzione per preservare i pesi.")
                    early_stopped = True
                    break

                val_acc, val_cats = evaluate_val_accuracy(
                    model, tokenizer, val_eval_items, bev_dir, device, num_tokens=args.num_tokens
                )
                if val_acc == 0.0:
                    logger.error(f"[Step {global_step}] Emergenza: Accuratezza di validazione crollata a 0%! Interruzione.")
                    early_stopped = True
                    break

                logger.info(
                    f"\n[Step {global_step}/{args.max_steps}] Val Acc: {BOLD}{val_acc:.2f}%{RESET} "
                    f"(Record: {best_acc:.2f}%) | Loss: {cur_loss:.4f}"
                )
                logger.info(f"  Categorie: {val_cats}")
                history[-1]["val_accuracy"] = val_acc
                history[-1]["category_accuracies"] = val_cats

                if val_acc > best_acc:
                    delta_gain = val_acc - best_acc
                    best_acc = val_acc
                    patience_counter = 0
                    best_dir = output_dir / "best_model"
                    model.save_checkpoint(str(best_dir))
                    tokenizer.save_pretrained(str(best_dir / "tokenizer"))
                    logger.info(f"  {GREEN}★ NUOVO RECORD ACCURACY ({val_acc:.2f}%, +{delta_gain:.2f}%)! Salvato in: {best_dir}{RESET}")
                else:
                    patience_counter += 1
                    logger.info(f"  Nessun miglioramento. Patience counter: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        logger.info(f"\n{YELLOW}★ Early Stopping attivato dopo {args.patience} check consecutivi senza miglioramenti.{RESET}")
                        early_stopped = True
                        break

            # Salvataggio periodico di sicurezza
            if global_step % args.save_steps == 0:
                step_dir = output_dir / f"checkpoint_step_{global_step}"
                model.save_checkpoint(str(step_dir))
                logger.info(f"  Checkpoint periodico salvato in: {step_dir}")

    pbar.close()
    elapsed = time.time() - start_time

    # Salvataggio modello finale
    latest_dir = output_dir / "latest_model"
    model.save_checkpoint(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir / "tokenizer"))

    # Salva report JSON
    report = {
        "task": "NuScenes-QA SOTA Record Training (GradScaler + Early Stopping)",
        "llm_backbone": args.llm_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_steps": global_step,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr_projector": args.lr_projector,
        "lr_lora": args.lr_lora,
        "early_stopped": early_stopped,
        "training_time_hours": round(elapsed / 3600.0, 2),
        "initial_val_accuracy": init_acc,
        "best_val_accuracy": best_acc,
        "checkpoints": {
            "best_model": str(output_dir / "best_model"),
            "latest_model": str(latest_dir),
        },
        "history_sample": history[::max(1, len(history) // 100)],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report completo salvato in: {report_path}")

    print(f"\n{BOLD}{GREEN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  TRAINING SOTA NUSCENES-QA CONCLUSO CON SUCCESSO!{RESET}")
    print(f"{BOLD}{GREEN}======================================================================{RESET}")
    print(f"  Passi completati:    {global_step}/{args.max_steps}")
    print(f"  Tempo totale:        {elapsed / 3600.0:.2f} ore")
    print(f"  Accuratezza Iniziale: {init_acc:.2f}%")
    print(f"  Migliore Accuratezza: {BOLD}{best_acc:.2f}%{RESET}")
    print(f"  Best Checkpoint:     {output_dir / 'best_model'}")
    print(f"  Report:              {report_path}")
    print(f"{GREEN}======================================================================{RESET}\n")


if __name__ == "__main__":
    main()
