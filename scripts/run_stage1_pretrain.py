#!/usr/bin/env python3
"""
Block 4: Stage 1 Pretraining (BEV-Language Alignment).

Allinea lo spazio di feature della BEV allo spazio semantico del linguaggio naturale:
- Modello: BEVVLM (Qwen2.5-3B-Instruct + DeeperConvProjector)
- Modalità: Stage 1 (solo Projector trainabile, LLM e LoRA congelati)
- Dati: Omnidrive driving scene descriptions (28,130 train, 6,019 val)
- Prompt:
  <|im_start|>system
  You are given a bird's-eye-view feature map of the driving scene. Describe what you see.<|im_end|>
  <|im_start|>user
  <|bev|>
  <|im_end|>
  <|im_start|>assistant
  {description}<|im_end|>
- Loss: Cross-Entropy calcolata esclusivamente sui token della descrizione generata.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from bev_vqa.data.collate import collate_train
from bev_vqa.data.dataset import BEVPretrainDataset, build_tokenizer, PRETRAIN_SYSTEM_PROMPT
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1_pretrain")

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


def evaluate_val_loss(
    model: BEVVLM,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 25,
) -> float:
    """Calcola la validation loss su un sottoinsieme di batch."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            bev = batch["bev"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                bev=bev,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += out.loss.item()
            count += 1
            del bev, input_ids, attention_mask, labels, out

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
    return total_loss / max(count, 1)


def generate_qualitative_samples(
    model: BEVVLM,
    tokenizer,
    val_dataset: BEVPretrainDataset,
    device: torch.device,
    num_samples: int = 3,
) -> List[Dict[str, str]]:
    """Genera descrizioni qualitative su scene BEV fisse di validazione."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = []

    system_prompt = PRETRAIN_SYSTEM_PROMPT
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|bev|>\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        for idx in range(min(num_samples, len(val_dataset))):
            item = val_dataset.data[idx]
            token = item["sample_token"]
            ground_truth = item["description"]
            bev_feat = val_dataset._load_bev_features(token)

            if bev_feat is None:
                continue

            bev = bev_feat.unsqueeze(0).to(device) if bev_feat.dim() == 3 else bev_feat.to(device)

            gen_ids = model.generate(
                bev=bev,
                input_ids=prompt_tokens["input_ids"],
                attention_mask=prompt_tokens["attention_mask"],
                max_new_tokens=80,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            results.append({
                "sample_token": token,
                "ground_truth": ground_truth,
                "prediction": pred_text,
            })
            del bev, gen_ids

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Pretraining: BEV-Language Alignment")
    parser.add_argument("--train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/omnidrive_descriptions_train.json")
    parser.add_argument("--val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/omnidrive_descriptions_val.json")
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage1")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv", choices=["deeper_conv", "qformer"])
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8, help="Passi di accumulo (effective batch size = batch_size * grad_accum = 16)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Numero massimo di passi di ottimizzazione")
    parser.add_argument("--val-steps", type=int, default=100, help="Frequenza valutazione validation loss")
    parser.add_argument("--val-batches", type=int, default=25, help="Batch di validazione per step")
    parser.add_argument("--save-steps", type=int, default=500, help="Frequenza salvataggio checkpoint periodico")
    parser.add_argument("--fraction", type=float, default=1.0, help="Frazione del dataset (1.0 = tutto)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/stage1_pretrain_report.json")
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
    print(f"{BOLD}{CYAN}  STAGE 1 PRETRAINING: BEV-LANGUAGE ALIGNMENT (BLOCK 4){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Dispositivo:             {device}")
    print(f"  LLM Backbone:            {args.llm_path}")
    print(f"  Architettura Projector:  {args.arch_type} ({args.num_tokens} visual tokens)")
    print(f"  Batch size per step:     {args.batch_size}")
    print(f"  Gradient Accumulation:   {args.grad_accum} (Effective batch size = {args.batch_size * args.grad_accum})")
    print(f"  Learning Rate:           {args.lr}")
    print(f"  Frazione dataset:        {args.fraction * 100:.1f}%")
    print(f"  Output Checkpoint:       {output_dir}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    # 1. Caricamento Tokenizer
    logger.info("Inizializzazione Tokenizer...")
    tokenizer = build_tokenizer(args.llm_path)

    # 2. Caricamento Dataset
    logger.info(f"Caricamento Training Dataset da {args.train_json}...")
    train_dataset = BEVPretrainDataset(
        json_path=args.train_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=1,
        fraction=args.fraction,
    )
    logger.info(f"Training samples caricati: {len(train_dataset):,}")

    logger.info(f"Caricamento Validation Dataset da {args.val_json}...")
    val_dataset = BEVPretrainDataset(
        json_path=args.val_json,
        bev_dir=args.bev_dir,
        tokenizer=tokenizer,
        split="val",
        num_bev_tokens=1,
        fraction=min(args.fraction * 2.0, 1.0),
    )
    logger.info(f"Validation samples caricati: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_train(b, pad_id=tokenizer.pad_token_id),
        pin_memory=True,
    )

    # 3. Istanziazione Modello BEVVLM
    logger.info("Inizializzazione modello BEVVLM...")
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(
            arch_type=args.arch_type,
            num_tokens=args.num_tokens,
        ),
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()  # Stabilità assoluta float32 per BatchNorm e AdamW

    # Stage 1: solo proiettore trainabile
    model.set_stage(1)
    param_counts = model.count_trainable_parameters()
    logger.info(
        f"Parametri: Projector={param_counts['projector_trainable']:,} trainabili "
        f"(LLM frozen={param_counts['llm_total']:,})"
    )

    # 4. Ottimizzatore & Scheduler
    trainable_params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_opt_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None and args.max_steps < total_opt_steps:
        total_opt_steps = args.max_steps

    num_warmup = max(1, int(total_opt_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_opt_steps,
    )

    logger.info(f"Passi di ottimizzazione totali: {total_opt_steps} (warmup={num_warmup})")

    # Baseline Zero-Shot iniziale
    logger.info("Calcolo Validation Loss iniziale pre-training (Zero-Shot Projector)...")
    init_val_loss = evaluate_val_loss(model, val_loader, device, max_batches=args.val_batches)
    logger.info(f"Initial Validation Loss: {init_val_loss:.4f}")

    logger.info("Generazione qualitativa pre-training (Projector non allineato)...")
    init_qualitative = generate_qualitative_samples(model, tokenizer, val_dataset, device, num_samples=2)
    for q in init_qualitative:
        logger.info(f"  [Sample {q['sample_token'][:8]}]")
        logger.info(f"    GT:   {q['ground_truth'][:100]}...")
        logger.info(f"    PRED: {q['prediction'][:100]}...")

    # 5. Training Loop
    logger.info(f"{BOLD}Inizio Training Stage 1...{RESET}")
    global_step = 0
    accumulated_loss = 0.0
    history = []
    best_val_loss = init_val_loss
    start_time = time.time()

    pbar = tqdm(total=total_opt_steps, desc="Stage 1 Pretrain", dynamic_ncols=True)

    optimizer.zero_grad()
    model.train()

    stop_training = False
    for epoch in range(args.epochs):
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
                # Clip grad norm sul proiettore
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                cur_loss = accumulated_loss
                accumulated_loss = 0.0
                cur_lr = scheduler.get_last_lr()[0]

                pbar.update(1)
                pbar.set_postfix({"loss": f"{cur_loss:.4f}", "gnorm": f"{grad_norm:.2f}", "lr": f"{cur_lr:.2e}"})

                history.append({
                    "step": global_step,
                    "train_loss": cur_loss,
                    "grad_norm": grad_norm,
                    "lr": cur_lr,
                })

                # Validazione periodica
                if global_step % args.val_steps == 0 or global_step == total_opt_steps:
                    val_loss = evaluate_val_loss(model, val_loader, device, max_batches=args.val_batches)
                    logger.info(
                        f"\nStep {global_step}/{total_opt_steps} | "
                        f"Train Loss: {cur_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Grad Norm: {grad_norm:.2f} | LR: {cur_lr:.2e}"
                    )

                    history[-1]["val_loss"] = val_loss

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = output_dir / 'stage1_projector_best.pt'
                        model.save_projector(str(best_path))
                        logger.info(f"  {GREEN}★ Nuovo record Val Loss ({val_loss:.4f})! Salvato: {best_path}{RESET}")

                # Salvataggio periodico
                if global_step % args.save_steps == 0:
                    latest_path = output_dir / f"stage1_projector_step{global_step}.pt"
                    model.save_projector(str(latest_path))

                if global_step >= total_opt_steps:
                    stop_training = True
                    break

    pbar.close()
    elapsed = time.time() - start_time

    # 6. Valutazione Finale & Generazione Qualitativa
    logger.info("\nCalcolo Validation Loss finale...")
    final_val_loss = evaluate_val_loss(model, val_loader, device, max_batches=args.val_batches * 2)
    logger.info(f"Final Validation Loss: {final_val_loss:.4f} (Miglioramento: {init_val_loss - final_val_loss:+.4f})")

    logger.info("Generazione qualitativa post-training...")
    final_qualitative = generate_qualitative_samples(model, tokenizer, val_dataset, device, num_samples=3)

    print(f"\n{BOLD}{GREEN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  QUALITATIVE SAMPLES AFTER STAGE 1 PRETRAINING{RESET}")
    print(f"{BOLD}{GREEN}======================================================================{RESET}")
    for i, q in enumerate(final_qualitative):
        print(f"\n{BOLD}Sample #{i+1} [Token: {q['sample_token'][:12]}]{RESET}")
        print(f"  {BOLD}Ground Truth:{RESET} {q['ground_truth']}")
        print(f"  {BOLD}Model Pred:  {RESET} {q['prediction']}")
    print(f"{GREEN}----------------------------------------------------------------------{RESET}\n")

    # Salvataggio pesi finali
    final_path = output_dir / "stage1_projector_latest.pt"
    model.save_projector(str(final_path))
    if not (output_dir / 'stage1_projector_best.pt').exists():
        model.save_projector(str(output_dir / 'stage1_projector_best.pt'))

    # Salva tokenizer con token <|bev|> configurato
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"Tokenizer salvato in: {tokenizer_dir}")

    # Salva Report JSON
    report = {
        "stage": 1,
        "description": "BEV-Language Alignment Pretraining",
        "llm_backbone": args.llm_path,
        "projector_arch": args.arch_type,
        "num_visual_tokens": args.num_tokens,
        "trainable_parameters": param_counts['projector_trainable'],
        "frozen_llm_parameters": param_counts['llm_total'],
        "total_optimization_steps": global_step,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "training_time_seconds": round(elapsed, 2),
        "initial_val_loss": round(init_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "val_loss_delta": round(final_val_loss - init_val_loss, 4),
        "qualitative_initial": init_qualitative,
        "qualitative_final": final_qualitative,
        "checkpoints": {
            "best_projector": str(output_dir / 'stage1_projector_best.pt'),
            "latest_projector": str(final_path),
            "tokenizer": str(tokenizer_dir),
        },
        "history_sample": history[::max(1, len(history) // 50)],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report completo salvato in: {report_path}")

    print(f"\n{BOLD}{GREEN}✓ STAGE 1 PRETRAINING COMPLETATO CON SUCCESSO!{RESET}")
    print(f"  Tempo totale:    {elapsed / 60:.1f} minuti")
    print(f"  Val Loss:        {init_val_loss:.4f} -> {final_val_loss:.4f}")
    print(f"  Best Projector:  {output_dir / 'stage1_projector_best.pt'}")
    print(f"  Report:          {report_path}\n")


if __name__ == "__main__":
    main()
