#!/usr/bin/env python3
"""
SOTA DriveLM Dedicated Training Script v2.

Features:
- Dataset: DriveLM Train (perception, prediction, planning, behavior)
- Full supervision on answer tokens with configurable max_length=384 (no truncation bug)
- Category-balanced sampling via DriveLMCategoryBalancedSampler
- LoRA: r=32, alpha=64, lr=1.5e-4 on all linear modules
- Projector: DeeperConv (lr=1.0e-4) initialized from Stage 1 pretraining
- Precision: Mixed Precision with dynamic GradScaler + Anti-NaN defense
- Periodic NLG validation (BLEU-1..4, ROUGE-L, CIDEr, METEOR) via src.training.metrics
- Checkpointing: Saves best_model when CIDEr/BLEU-4 reaches a new record
- Supports --dry-run to test 2 steps before launching long runs
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
from bev_vqa.constants import Colors, DRIVELM_SYSTEM_PROMPT
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from bev_vqa.data.collate import collate_train
from bev_vqa.data.dataset import (
    BEVQADataset,
    build_tokenizer,
    BEV_TOKEN,
)
from bev_vqa.data.sampler import DriveLMCategoryBalancedSampler
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig
from bev_vqa.training.metrics import compute_nlg_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_sota_drivelm_v2")





def load_drivelm_validation_pool(
    val_json_path: str,
    bev_dir: Path,
    samples_per_type: int = 40,
    seed: int = 42,
) -> List[Dict]:
    with open(val_json_path) as f:
        data = json.load(f).get("questions", [])

    candidate_subdirs = [bev_dir / "val", bev_dir / "train", bev_dir]
    if bev_dir.parent.exists():
        candidate_subdirs.extend([bev_dir.parent / "val", bev_dir.parent / "train"])

    def has_bev(token: str) -> bool:
        for d in candidate_subdirs:
            if (d / f"{token}.pt").exists():
                return True
        return False

    by_type = defaultdict(list)
    for q in data:
        token = q.get("sample_token")
        if has_bev(token):
            by_type[q.get("template_type", "other")].append(q)

    random.seed(seed)
    sampled = []
    target_types = ["perception", "prediction", "planning", "behavior"]
    for t in target_types:
        pool = by_type.get(t, [])
        if pool:
            sampled.extend(random.sample(pool, min(samples_per_type, len(pool))))

    # If any target types were missing, sample from general pool
    if not sampled and data:
        valid_any = [q for q in data if has_bev(q.get("sample_token"))]
        sampled = random.sample(valid_any, min(samples_per_type * 4, len(valid_any)))

    random.shuffle(sampled)
    logger.info(f"Pool di validazione DriveLM: {len(sampled)} ready samples ({dict((t, len(by_type[t])) for t in target_types)}).")
    return sampled


def resolve_sample_bev_tensor(sample_token: str, bev_dir: Path) -> Optional[torch.Tensor]:
    candidates = [
        bev_dir / "val" / f"{sample_token}.pt",
        bev_dir / "train" / f"{sample_token}.pt",
        bev_dir / f"{sample_token}.pt",
        bev_dir.parent / "val" / f"{sample_token}.pt",
        bev_dir.parent / "train" / f"{sample_token}.pt",
    ]
    for c in candidates:
        if c.exists():
            data = torch.load(c, map_location="cpu", weights_only=True)
            feat = data["features_fused"]
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)
            return feat
    return None


def evaluate_val_drivelm(
    model: BEVVLM,
    tokenizer,
    val_items: List[Dict],
    bev_dir: Path,
    device: torch.device,
    num_tokens: int = 32,
    max_new_tokens: int = 64,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()

    references = []
    hypotheses = []
    by_type_refs = defaultdict(list)
    by_type_hyps = defaultdict(list)

    with torch.no_grad():
        for it in val_items:
            token = it["sample_token"]
            bev = resolve_sample_bev_tensor(token, bev_dir)
            if bev is None:
                continue

            bev = bev.to(device)
            q = it["question"]
            gt = it["answer"].strip()
            cat = it.get("template_type", "other")

            bev_placeholder = BEV_TOKEN * num_tokens
            messages = [
                {"role": "system", "content": DRIVELM_SYSTEM_PROMPT},
                {"role": "user", "content": f"{bev_placeholder}\n{q}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda"))

            with autocast_ctx:
                out = model.generate(
                    bev=bev,
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()

            references.append(gt)
            hypotheses.append(pred)
            by_type_refs[cat].append(gt)
            by_type_hyps[cat].append(pred)
            del bev, out

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    overall = compute_nlg_metrics(predictions=hypotheses, references=references)
    cat_metrics = {}
    for cat in ["perception", "prediction", "planning", "behavior"]:
        if by_type_refs[cat]:
            cat_metrics[cat] = compute_nlg_metrics(predictions=by_type_hyps[cat], references=by_type_refs[cat])

    return overall, cat_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SOTA DriveLM Dedicated Training v2")
    default_train = os.environ.get("DRIVELM_TRAIN_JSON", "/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_train.json")
    default_val = os.environ.get("DRIVELM_VAL_JSON", "/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_val.json")
    default_bev = os.environ.get("BEV_FEATURES_DIR", "/media/nazario.pizzicoli/Datos/dataset/bev_features")

    parser.add_argument("--train-json", type=str, default=default_train)
    parser.add_argument("--val-json", type=str, default=default_val)
    parser.add_argument("--bev-dir", type=str, default=default_bev)
    parser.add_argument("--stage1-projector", type=str, default="checkpoints/sota/stage1/stage1_projector_best.pt")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/sota_drivelm_v2")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--pos-embed-type", type=str, default="1d_learned", choices=["1d_learned", "2d_cartesian_metric", "2d_polar_metric", "2d_separable_learned"])
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--spatial-shape", type=int, nargs=2, default=None, metavar=("H", "W"), help="Explicit spatial pooling grid (H, W), e.g. 4 8 or 8 4 or 8 8")
    parser.add_argument("--use-dual-pooling", action=argparse.BooleanOptionalAction, default=False, help="Enable Dual Pooling (Avg + Max)")
    parser.add_argument("--use-l2-pooling", action=argparse.BooleanOptionalAction, default=False, help="Enable L2-norm salience weighted pooling")
    parser.add_argument("--l2-pooling-gamma", type=float, default=1.0, help="L2 pooling salience gamma/temperature")
    parser.add_argument("--l2-pooling-learnable", action=argparse.BooleanOptionalAction, default=False, help="Make L2 gamma learnable")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=384, help="Sequence max length for prompt + answer")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16, help="Effective batch size = 16")
    parser.add_argument("--lr-projector", type=float, default=1.0e-4)
    parser.add_argument("--lr-lora", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--val-steps", type=int, default=250)
    parser.add_argument("--val-samples-per-type", type=int, default=35, help="35 x 4 samples")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/sota_drivelm_v2_report.json")
    parser.add_argument("--no-balanced-sampler", action="store_true", help="Disable CategoryBalancedSampler")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 steps to verify end-to-end pipeline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.dry_run:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}>>> DRY RUN MODE ACTIVATED: 2 training steps, 1 validation check <<<{Colors.RESET}\n")
        args.max_steps = 2
        args.grad_accum = 1
        args.val_steps = 1
        args.val_samples_per_type = 1
        args.save_steps = 2
        args.warmup_steps = 1
        args.num_workers = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bev_dir = Path(args.bev_dir)

    device = torch.device(args.device)

    spatial_shape = tuple(args.spatial_shape) if args.spatial_shape is not None else None
    num_tokens = (spatial_shape[0] * spatial_shape[1]) if spatial_shape is not None else args.num_tokens
    grid_desc = f"{spatial_shape[0]}x{spatial_shape[1]} ({num_tokens} tokens)" if spatial_shape else f"{num_tokens} tokens"

    print(f"\n{Colors.BOLD}{Colors.CYAN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  SOTA DRIVELM V2 GENERATIVE TRAINING (BLEU/ROUGE/CIDER/METEOR){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}======================================================================{Colors.RESET}")
    print(f"  Device:                {device}")
    print(f"  Dataset:                    DriveLM Train ({args.train_json})")
    print(f"  Griglia Spaziale:           {grid_desc} (Dual Pooling={args.use_dual_pooling})")
    print(f"  Max Length:                 {args.max_length} tokens (full answer supervision)")
    print(f"  LoRA Configurazione:        r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Balanced Sampler:           {not args.no_balanced_sampler} (DriveLMCategoryBalancedSampler)")
    print(f"  Learning Rates:             LoRA={args.lr_lora:.1e}, Projector={args.lr_projector:.1e}")
    print(f"  Max Steps / Patience:       {args.max_steps} step (Patience = {args.patience})")
    print(f"  Effective Batch Size:       {args.batch_size * args.grad_accum}")
    print(f"  Directory Checkpoints:      {output_dir}")
    print(f"{Colors.CYAN}----------------------------------------------------------------------{Colors.RESET}\n")

    tokenizer = build_tokenizer(args.llm_path)

    # 1. Dataset DriveLM Train
    logger.info("Caricamento dataset DriveLM train...")
    train_fraction = 0.01 if args.dry_run else 1.0
    train_dataset = BEVQADataset(
        json_path=args.train_json,
        bev_dir=bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=num_tokens,
        fraction=train_fraction,
        system_prompt=DRIVELM_SYSTEM_PROMPT,
        max_length=args.max_length,
    )

    if not args.no_balanced_sampler:
        sampler = DriveLMCategoryBalancedSampler(
            dataset_data=train_dataset.data,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    logger.info(f"Dataset DriveLM Train: {len(train_dataset):,} ready samples.")

    # 2. Pool di Validazione
    val_pool = load_drivelm_validation_pool(
        args.val_json,
        bev_dir=bev_dir,
        samples_per_type=args.val_samples_per_type,
        seed=args.seed,
    )
    if args.dry_run:
        val_pool = val_pool[:1]

    # 3. Costruzione Modello
    logger.info(f"Costruzione BEV-VLM (Projector: {args.arch_type}, Griglia: {grid_desc}, Dual Pooling={args.use_dual_pooling}, LoRA r={args.lora_r}, alpha={args.lora_alpha})...")
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
        logger.info(f"Caricamento pesi completi da checkpoint di ripresa: {args.resume_from}")
        model.load_checkpoint(args.resume_from)
    elif os.path.exists(args.stage1_projector):
        logger.info(f"Caricamento pesi Stage 1 Projector da: {args.stage1_projector}")
        model.load_projector(args.stage1_projector)

    model.set_stage(2)
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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Baseline Zero-Shot iniziale (salvo in dry-run per velocizzare il test)
    if not args.dry_run:
        logger.info("Initial metric evaluation pre-training su DriveLM...")
        init_metrics, init_cat = evaluate_val_drivelm(
            model, tokenizer, val_pool, bev_dir, device, num_tokens=num_tokens
        )
        logger.info(
            f"Metriche Iniziali: BLEU-4={init_metrics['bleu_4']:.2f}% | "
            f"ROUGE-L={init_metrics['rouge_l']:.2f}% | CIDEr={init_metrics['cider']:.3f} | METEOR={init_metrics['meteor']:.2f}%"
        )
    else:
        init_metrics = {"bleu_1": 0.0, "bleu_4": 0.0, "rouge_l": 0.0, "meteor": 0.0, "cider": 0.0}
        init_cat = {}

    # 5. Training Loop con Early Stopping
    logger.info(f"\n{Colors.BOLD}{Colors.GREEN}>>> START SOTA TRAINING DRIVELM V2 ({args.max_steps} step max)...{Colors.RESET}")
    global_step = 0
    accum_loss = 0.0
    best_cider = init_metrics["cider"]
    best_bleu4 = init_metrics["bleu_4"]
    patience_counter = 0
    history = []
    start_time = time.time()

    pbar = tqdm(total=args.max_steps, desc="SOTA DriveLM v2 Training", dynamic_ncols=True)
    train_iter = iter(train_loader)
    sub_step = 0
    epoch = 0
    early_stopped = False

    while global_step < args.max_steps and not early_stopped:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if not args.no_balanced_sampler and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        bev = batch["bev"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda"))

        with autocast_ctx:
            out = model(bev=bev, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.grad_accum

        if not torch.isfinite(loss):
            logger.warning(f"[Step {global_step}] Loss non finita ({loss.item()})! Micro-batch scartato e accumulo resettato.")
            optimizer.zero_grad()
            accum_loss = 0.0
            sub_step -= (sub_step % args.grad_accum)
            del out, bev, input_ids, attention_mask, labels
            continue

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        del out, bev, input_ids, attention_mask, labels

        sub_step += 1
        if sub_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)

            # Controllo gradienti
            grads_finite = all(
                torch.isfinite(p.grad).all().item()
                for p in model.trainable_parameters()
                if p.grad is not None
            )

            if not grads_finite:
                logger.warning(f"[Step {global_step}] Gradienti non finiti rilevati! Step annullato.")
                optimizer.zero_grad()
                scaler.step(optimizer)  # Safely notifies scaler of invalid gradients to reduce scale
                scaler.update()
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
                "cider": f"{best_cider:.2f}",
                "bleu4": f"{best_bleu4:.1f}%",
                "patience": f"{patience_counter}/{args.patience}"
            })

            history.append({
                "step": global_step,
                "loss": cur_loss,
                "lr_lora": cur_lr_lora,
            })

            # Validazione Periodica Generativa
            if global_step % args.val_steps == 0 or global_step == args.max_steps:
                if math.isnan(cur_loss):
                    logger.error(f"[Step {global_step}] Emergenza: Loss NaN! Arresto.")
                    early_stopped = True
                    break

                val_max_new_tokens = 16 if args.dry_run else 64
                val_metrics, val_by_cat = evaluate_val_drivelm(
                    model, tokenizer, val_pool, bev_dir, device, num_tokens=num_tokens, max_new_tokens=val_max_new_tokens
                )

                logger.info(
                    f"\n[Step {global_step}/{args.max_steps}] "
                    f"BLEU-4: {Colors.BOLD}{val_metrics['bleu_4']:.2f}%{Colors.RESET} | "
                    f"ROUGE-L: {val_metrics['rouge_l']:.2f}% | "
                    f"CIDEr: {Colors.BOLD}{val_metrics['cider']:.3f}{Colors.RESET} (Record: {best_cider:.3f}) | "
                    f"METEOR: {val_metrics['meteor']:.2f}% | Loss: {cur_loss:.4f}"
                )
                history[-1]["val_metrics"] = val_metrics
                history[-1]["category_metrics"] = val_by_cat

                # Aggiornamento Record (CIDEr o combinato)
                is_new_record = val_metrics["cider"] > best_cider or (val_metrics["cider"] >= best_cider and val_metrics["bleu_4"] > best_bleu4)
                if is_new_record or args.dry_run:
                    best_cider = max(best_cider, val_metrics["cider"])
                    best_bleu4 = max(best_bleu4, val_metrics["bleu_4"])
                    patience_counter = 0
                    best_dir = output_dir / "best_model"
                    model.save_checkpoint(str(best_dir))
                    tokenizer.save_pretrained(str(best_dir / "tokenizer"))
                    logger.info(f"  {Colors.GREEN}★ DRIVELM V2 RECORD (CIDEr: {best_cider:.3f}, BLEU-4: {best_bleu4:.2f}%)! Saved to: {best_dir}{Colors.RESET}")
                else:
                    patience_counter += 1
                    logger.info(f"  Nessun miglioramento. Patience: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        logger.info(f"\n{Colors.YELLOW}★ Early Stopping attivato dopo {args.patience} check consecutivi.{Colors.RESET}")
                        early_stopped = True
                        break

            # Salvataggio periodico di sicurezza
            if global_step % args.save_steps == 0:
                step_dir = output_dir / f"checkpoint_step_{global_step}"
                model.save_checkpoint(str(step_dir))
                logger.info(f"  Checkpoint Saved to: {step_dir}")

    pbar.close()
    elapsed = time.time() - start_time

    latest_dir = output_dir / "latest_model"
    model.save_checkpoint(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir / "tokenizer"))

    report = {
        "task": "DriveLM SOTA Dedicated Generative Training v2",
        "llm_backbone": args.llm_path,
        "spatial_shape": spatial_shape,
        "num_tokens": num_tokens,
        "use_dual_pooling": args.use_dual_pooling,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_steps": global_step,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr_projector": args.lr_projector,
        "lr_lora": args.lr_lora,
        "early_stopped": early_stopped,
        "training_time_hours": round(elapsed / 3600.0, 4),
        "initial_metrics": init_metrics,
        "best_cider": best_cider,
        "best_bleu_4": best_bleu4,
        "checkpoints": {
            "best_model": str(output_dir / "best_model"),
            "latest_model": str(latest_dir),
        },
        "history_sample": history,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  TRAINING SOTA DRIVELM V2 COMPLETATO CON SUCCESSO!{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"  Passi completati:     {global_step}/{args.max_steps}")
    print(f"  Tempo totale:         {elapsed:.1f}s ({elapsed / 3600.0:.3f} ore)")
    print(f"  Miglior CIDEr:        {best_cider:.3f}")
    print(f"  Miglior BLEU-4:       {best_bleu4:.2f}%")
    print(f"  Best Checkpoint:      {output_dir / 'best_model'}")
    print(f"  Report:               {report_path}")
    print(f"{Colors.GREEN}======================================================================{Colors.RESET}\n")


if __name__ == "__main__":
    main()
