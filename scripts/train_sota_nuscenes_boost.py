#!/usr/bin/env python3
"""
SOTA NuScenes-QA Boost Training Script.

Implements all 3 Critical Interventions:
1. Intervention 1 - Isotropic 64-Token Resolution with Dual Pooling (Avg + Max) in DeeperConvProjector.
2. Intervention 2 - Category-Balanced Batching / Sampling (Count 30%, Status 30%, Object 20%, Exist 10%, Comparison 10%).
3. Intervention 3 - Class-Weighted Loss (Status: with rider 4.0x, parked 1.5x, stopped 1.5x) &
                     Ordinal Gaussian Label Smoothing (Count 0-10, sigma=0.9).
"""

import argparse
from collections import defaultdict
import json
import logging
import math
import os
from bev_vqa.utils.io import get_default_paths

_PATHS = get_default_paths()

from bev_vqa.utils.seed import set_seed
from bev_vqa.constants import Colors
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
from bev_vqa.data.sampler import CategoryBalancedSampler, DEFAULT_CATEGORY_RATIOS
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig
from bev_vqa.training.losses import NuScenesBoostLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nuscenes_boost")











def evaluate_val_accuracy(
    model: BEVVLM,
    tokenizer,
    val_items: List[Dict],
    bev_dir: Path,
    device: torch.device,
    num_tokens: int = 64,
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

            with torch.amp.autocast("cuda", dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
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
    parser = argparse.ArgumentParser(description="SOTA NuScenes-QA Boost Training with Interventions 1, 2, 3")
    parser.add_argument("--train-json", type=str, default=_PATHS.get('nuscenes_qa_train', ''))
    parser.add_argument("--val-json", type=str, default=_PATHS.get('nuscenes_qa_val', ''))
    parser.add_argument("--bev-dir", type=str, default=_PATHS.get('bev_dir', ''))
    parser.add_argument("--stage1-projector", type=str, default="checkpoints/sota/stage1/stage1_projector_best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sota_nuscenes_boost")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--pos-embed-type", type=str, default="1d_learned", choices=["1d_learned", "2d_cartesian_metric", "2d_polar_metric", "2d_separable_learned"])
    parser.add_argument("--num-tokens", type=int, default=64, help="Intervention 1: 64 tokens (8x8 isotropic grid)")
    parser.add_argument("--spatial-shape", type=int, nargs=2, default=None, metavar=("H", "W"), help="Explicit spatial pooling grid (H, W), e.g. 4 8 or 8 4 or 8 8")
    parser.add_argument("--use-dual-pooling", action=argparse.BooleanOptionalAction, default=True, help="Intervention 1: Dual Pooling (Avg + Max)")
    parser.add_argument("--use-l2-pooling", action=argparse.BooleanOptionalAction, default=False, help="Enable L2-norm salience weighted pooling")
    parser.add_argument("--l2-pooling-gamma", type=float, default=1.0, help="L2 pooling salience gamma/temperature")
    parser.add_argument("--l2-pooling-learnable", action=argparse.BooleanOptionalAction, default=False, help="Make L2 gamma learnable")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16, help="Effective batch size = 16")
    parser.add_argument("--lr-projector", type=float, default=1.0e-4)
    parser.add_argument("--lr-lora", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--val-steps", type=int, default=250)
    parser.add_argument("--val-samples-per-cat", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--initial-step", type=int, default=0, help="Initial global step when resuming")
    parser.add_argument("--count-sigma", type=float, default=0.9, help="Intervention 3: Gaussian smoothing sigma (0.8-1.0)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/sota_nuscenes_boost_report.json")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Esegue solo 2 step di test e termina")
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

    spatial_shape = tuple(args.spatial_shape) if args.spatial_shape is not None else None
    num_tokens = (spatial_shape[0] * spatial_shape[1]) if spatial_shape is not None else args.num_tokens
    grid_desc = f"{spatial_shape[0]}x{spatial_shape[1]} ({num_tokens} tokens)" if spatial_shape else f"{num_tokens} tokens"

    print(f"\n{Colors.BOLD}{Colors.CYAN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  SOTA NUSCENES-QA BOOST TRAINING (INTERVENTIONS 1, 2, 3){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}======================================================================{Colors.RESET}")
    print(f"  Device:                {device}")
    print(f"  Intervention 1 (Projector): {grid_desc} | Dual Pooling={args.use_dual_pooling}")
    print(f"  Intervention 2 (Sampling):  Category-Balanced (Count 30%, Status 30%, Object 20%, Exist 10%, Comp 10%)")
    print(f"  Intervention 3 (Loss):      Ordinal Gaussian Smoothing (sigma={args.count_sigma}) + Status Class Weights")
    print(f"  LoRA Configurazione:        r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Effective Batch Size:       {args.batch_size * args.grad_accum}")
    print(f"  Dry-Run Mode:               {args.dry_run}")
    print(f"  Directory Checkpoints:      {output_dir}")
    print(f"{Colors.CYAN}----------------------------------------------------------------------{Colors.RESET}\n")

    tokenizer = build_tokenizer(args.llm_path)

    # 1. Dataset e Sampler Bilanciato (Intervention 2)
    logger.info("Caricamento dataset NuScenes-QA train naturale...")
    train_fraction = 0.02 if args.dry_run else 1.0
    train_dataset = BEVQADataset(
        json_path=args.train_json,
        bev_dir=bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=num_tokens,
        fraction=train_fraction,
    )
    logger.info(f"Dataset NuScenes-QA Naturale: {len(train_dataset):,} samples caricati.")

    # Sampler bilanciato per categorie
    balanced_sampler = CategoryBalancedSampler(
        dataset_data=train_dataset.data,
        target_ratios=DEFAULT_CATEGORY_RATIOS,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler,
        num_workers=args.num_workers if not args.dry_run else 0,
        collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

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
    samples_per_cat = 1 if args.dry_run else args.val_samples_per_cat
    for cat in target_cats:
        pool = val_by_cat[cat]
        val_eval_items.extend(random.sample(pool, min(samples_per_cat, len(pool))))
    random.shuffle(val_eval_items)
    logger.info(f"Pool di validazione: {len(val_eval_items)} ready samples.")

    # 3. Costruzione Modello con Projector (Intervention 1)
    logger.info(f"Inizializzazione BEV-VLM (Projector={grid_desc}, Dual Pooling={args.use_dual_pooling})...")
    projector_cfg = ProjectorConfig(
        arch_type=args.arch_type,
        pos_embed_type=args.pos_embed_type,
        num_tokens=num_tokens,
        use_dual_pooling=args.use_dual_pooling,
        spatial_shape=spatial_shape,
        use_l2_pooling=args.use_l2_pooling,
        l2_pooling_gamma=args.l2_pooling_gamma,
        l2_pooling_learnable=args.l2_pooling_learnable,
    )
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=projector_cfg,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()

    if args.resume_from:
        logger.info(f"Caricamento pesi completi da checkpoint: {args.resume_from}")
        model.load_checkpoint(args.resume_from)
    elif os.path.exists(args.stage1_projector):
        logger.info(f"Caricamento pesi Stage 1 Projector da: {args.stage1_projector}")
        model.load_projector(args.stage1_projector, strict=False)

    model.set_stage(2)  # Projector + LoRA attivi
    model.llm.gradient_checkpointing_enable()

    param_counts = model.count_trainable_parameters()
    logger.info(
        f"Parametri Trainabili: Proiettore={param_counts['projector_trainable']:,} | "
        f"LoRA={param_counts['llm_trainable']:,} | Totale={param_counts['total_trainable']:,}"
    )

    # 4. Ottimizzatore, Boost Loss (Intervention 3) e GradScaler
    boost_loss_fn = NuScenesBoostLoss(
        tokenizer=tokenizer,
        count_sigma=args.count_sigma,
        max_count=10,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": list(model.projector.parameters()), "lr": args.lr_projector, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.llm.named_parameters() if p.requires_grad], "lr": args.lr_lora, "weight_decay": args.weight_decay},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"), init_scale=2048.0)

    # Baseline Zero-Shot iniziale e recupero best_acc precedente
    prev_best_acc = 0.0
    prev_history = []
    if args.resume_from:
        prev_report_candidates = [
            output_dir / "train_report.json",
            Path(args.resume_from).parent / "train_report.json",
            Path(args.resume_from) / "train_report.json",
        ]
        for p_rep in prev_report_candidates:
            if p_rep.exists():
                try:
                    with open(p_rep) as f:
                        prev_data = json.load(f)
                        prev_best_acc = float(prev_data.get("best_val_accuracy", 0.0))
                        prev_history = prev_data.get("history", [])
                    logger.info(f"Migliore accuratezza precedente caricata da {p_rep}: {prev_best_acc:.2f}% (history: {len(prev_history)} record)")
                    break
                except Exception as e:
                    logger.warning(f"Impossibile leggere report precedente: {e}")

    if not args.dry_run:
        logger.info("Initial pre-training accuracy evaluation...")
        init_acc, init_cat_accs = evaluate_val_accuracy(
            model, tokenizer, val_eval_items, bev_dir, device, num_tokens=num_tokens
        )
        logger.info(f"Accuratezza Iniziale: {init_acc:.2f}% (Categorie: {init_cat_accs})")
    else:
        init_acc = 0.0
        init_cat_accs = {}

    best_acc = max(init_acc, prev_best_acc)
    if init_acc > prev_best_acc and not args.dry_run:
        best_dir = output_dir / "best_model"
        model.save_checkpoint(str(best_dir))
        tokenizer.save_pretrained(str(best_dir / "tokenizer"))
        logger.info(f"Accuratezza iniziale ({init_acc:.2f}%) supera il record precedente ({prev_best_acc:.2f}%): saved best_model in {best_dir}")
    global_step = args.initial_step
    steps_to_train = 2 if args.dry_run else max(1, args.max_steps - global_step)
    max_steps = (global_step + 2) if args.dry_run else args.max_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(1, steps_to_train // 10)),
        num_training_steps=steps_to_train,
    )

    # 5. Training Loop con Early Stopping
    logger.info(f"\n{Colors.BOLD}{Colors.GREEN}>>> START SOTA BOOST TRAINING NUSCENES-QA ({global_step} -> {max_steps} step, Best Record: {best_acc:.2f}%)...{Colors.RESET}")
    accum_loss = 0.0
    patience_counter = 0
    history = list(prev_history) if prev_history else []
    start_time = time.time()

    pbar = tqdm(total=max_steps, initial=global_step, desc="SOTA NuScenes-QA Boost", ncols=130)
    train_iter = iter(train_loader)
    sub_step = 0
    early_stopped = False

    while global_step < max_steps and not early_stopped:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bev = batch["bev"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        template_types = batch.get("template_types")
        answers = batch.get("answers")

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            # Ottieni output e aligned_labels
            out, aligned_labels = model(
                bev=bev,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_aligned_labels=True,
            )
            # Calcola la Boost Loss personalizzata (Intervention 3)
            loss_raw = boost_loss_fn(
                logits=out.logits,
                labels=aligned_labels,
                template_types=template_types,
                answers=answers,
            )
            loss = loss_raw / args.grad_accum

        if not torch.isfinite(loss):
            logger.warning(f"[Step {global_step}] Loss non finita ({loss.item()})! Micro-batch scartato.")
            optimizer.zero_grad()
            del out, aligned_labels, bev, input_ids, attention_mask, labels
            continue

        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        del out, aligned_labels, bev, input_ids, attention_mask, labels

        sub_step += 1
        effective_step = (sub_step % args.grad_accum == 0) or (args.dry_run and sub_step >= 2)
        if effective_step:
            if device.type == "cuda":
                scaler.unscale_(optimizer)

            # Verifica che tutti i gradienti siano finiti prima del grad clipping
            grads_finite = all(
                torch.isfinite(p.grad).all().item()
                for p in model.trainable_parameters()
                if p.grad is not None
            )

            if not grads_finite:
                logger.warning(f"[Step {global_step}] Gradienti non finiti (NaN/Inf) rilevati! Step annullato.")
                if device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad()
                accum_loss = 0.0
                sub_step = 0
                continue

            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            if device.type == "cuda":
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer.zero_grad()
                if scale_before <= scale_after:
                    scheduler.step()
            else:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            sub_step = 0
            global_step += 1
            cur_loss = accum_loss
            accum_loss = 0.0
            cur_lr_lora = scheduler.get_last_lr()[1]

            pbar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "lr": f"{cur_lr_lora:.2e}",
                "best_acc": f"{best_acc:.1f}%",
                "patience": f"{patience_counter}/{args.patience}"
            })
            pbar.update(1)

            if global_step % 50 == 0:
                logger.info(f"Step {global_step}/{max_steps} | Loss: {cur_loss:.4f} | LR: {cur_lr_lora:.2e} | Best Acc: {best_acc:.2f}% | Patience: {patience_counter}/{args.patience}")

            history.append({
                "step": global_step,
                "loss": cur_loss,
                "lr_lora": cur_lr_lora,
            })

            # Validazione Periodica (salvo dry-run)
            if not args.dry_run and (global_step % args.val_steps == 0 or global_step == max_steps):
                if math.isnan(cur_loss):
                    logger.error(f"[Step {global_step}] Emergenza: Loss media NaN! Interruzione per preservare i pesi.")
                    early_stopped = True
                    break

                val_acc, val_cats = evaluate_val_accuracy(
                    model, tokenizer, val_eval_items, bev_dir, device, num_tokens=num_tokens
                )

                logger.info(
                    f"\n[Step {global_step}/{max_steps}] Val Acc: {Colors.BOLD}{val_acc:.2f}%{Colors.RESET} "
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
                    logger.info(f"  {Colors.GREEN}★ NEW ACCURACY RECORD ({val_acc:.2f}%, +{delta_gain:.2f}%)! Saved to: {best_dir}{Colors.RESET}")
                else:
                    patience_counter += 1
                    logger.info(f"  Nessun miglioramento. Patience counter: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        logger.info(f"\n{Colors.YELLOW}★ Early Stopping attivato dopo {args.patience} check consecutivi.{Colors.RESET}")
                        early_stopped = True
                        break

            # Salvataggio periodico di sicurezza
            if not args.dry_run and global_step % args.save_steps == 0:
                step_dir = output_dir / f"checkpoint_step_{global_step}"
                model.save_checkpoint(str(step_dir))
                logger.info(f"  Checkpoint periodico Saved to: {step_dir}")

    pbar.close()
    elapsed = time.time() - start_time

    # Saving final model
    latest_dir = output_dir / "latest_model"
    model.save_checkpoint(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir / "tokenizer"))

    # Assicura la presenza di best_model (indispensabile per successive eval anche in dry-run)
    best_dir = output_dir / "best_model"
    if not best_dir.exists():
        model.save_checkpoint(str(best_dir))
        tokenizer.save_pretrained(str(best_dir / "tokenizer"))

    # Salva report JSON
    report = {
        "task": "NuScenes-QA SOTA Boost Training (Interventions 1, 2, 3)",
        "interventions": {
            "intervention_1": f"Grid {grid_desc} DeeperConvProjector with Dual Pooling (Avg+Max={args.use_dual_pooling})",
            "intervention_2": "Category-Balanced Batching / Sampling (Count 30%, Status 30%, Object 20%, Exist 10%, Comp 10%)",
            "intervention_3": f"Ordinal Gaussian Label Smoothing (sigma={args.count_sigma}) + Status Class Weighting",
        },
        "llm_backbone": args.llm_path,
        "num_tokens": num_tokens,
        "spatial_shape": spatial_shape,
        "use_dual_pooling": args.use_dual_pooling,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_steps": global_step,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr_projector": args.lr_projector,
        "lr_lora": args.lr_lora,
        "early_stopped": early_stopped,
        "dry_run": args.dry_run,
        "training_time_seconds": round(elapsed, 2),
        "initial_val_accuracy": init_acc,
        "best_val_accuracy": best_acc,
        "checkpoints": {
            "best_model": str(best_dir),
            "latest_model": str(latest_dir),
        },
        "history": history,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report boost Saved to: {report_path}")

    print(f"\n{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  TRAINING SOTA NUSCENES-QA BOOST COMPLETATO!{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"  Passi completati:    {global_step}/{max_steps}")
    print(f"  Tempo:               {elapsed:.1f} secondi")
    print(f"  Accuratezza Iniziale: {init_acc:.2f}%")
    print(f"  Migliore Accuratezza: {Colors.BOLD}{best_acc:.2f}%{Colors.RESET}")
    print(f"  Best Checkpoint:     {output_dir / 'best_model'}")
    print(f"  Report:              {report_path}")
    print(f"{Colors.GREEN}======================================================================{Colors.RESET}\n")


if __name__ == "__main__":
    main()
