#!/usr/bin/env python3
"""
SOTA Dedicated Training for NuScenes-QA (Targeting >60% Accuracy).

Key Architectural & Training Enhancements:
1. High-Capacity LoRA (r=64, alpha=128) matching BeLLA SOTA configuration.
2. Full joint training (Projector lr=1e-4, LoRA lr=2e-4) with cosine decay.
3. Stratified balanced sampling across all 5 categories (exist, count, object, status, comparison)
   to directly eliminate class imbalance and fix the count/object bottlenecks.
4. Strict concise answering prompt for exact-token classification.
5. Rigorous validation checkpointing tracking overall and per-category accuracy every 250 steps.
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from bev_vqa.data.collate import collate_train
from bev_vqa.data.dataset import (
    build_tokenizer,
    normalize_answer,
    BEV_TOKEN,
)
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sota_nuscenes_qa")

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

SOTA_NUSCENES_SYSTEM_PROMPT = (
    "You are an autonomous driving perception system. You are given a bird's-eye-view "
    "feature map of the scene. Answer the question about the scene strictly with a single "
    "word, number, or exact category name."
)

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


class StratifiedBalancedNuScenesDataset(Dataset):
    """
    Dataset con campionamento stratificato ed equo per categoria:
    alterna in modo circolare tra exist, count, object, status, comparison
    garantendo gradienti perfettamente bilanciati ad ogni batch.
    """
    def __init__(
        self,
        json_path: str,
        bev_dir: Path,
        tokenizer,
        split: str = "train",
        num_bev_tokens: int = 32,
    ):
        self.bev_dir = bev_dir / split
        self.tokenizer = tokenizer
        self.split = split
        self.num_bev_tokens = num_bev_tokens

        with open(json_path) as f:
            data = json.load(f).get("questions", [])

        # Filtro BEV esistenti
        existing_tokens = {p.stem for p in self.bev_dir.glob("*.pt")} if self.bev_dir.exists() else set()
        valid = [q for q in data if q.get("sample_token") in existing_tokens]
        logger.info(f"NuScenes-QA {split}: {len(valid):,} domande con BEV valida su {len(data):,} totali.")

        # Raggruppamento per le 5 categorie
        self.by_category = defaultdict(list)
        for q in valid:
            cat = q.get("template_type", "unknown")
            self.by_category[cat].append(q)

        self.target_cats = ["exist", "count", "object", "status", "comparison"]
        for cat in self.target_cats:
            logger.info(f"  - Categoria '{cat}': {len(self.by_category[cat]):,} campioni")

        # Shuffling iniziale di ogni pool
        for cat in self.target_cats:
            random.shuffle(self.by_category[cat])

        # Determina lunghezza virtuale del dataset
        self.max_cat_len = max(len(self.by_category[c]) for c in self.target_cats)
        self.total_len = self.max_cat_len * len(self.target_cats)

    def __len__(self) -> int:
        return self.total_len

    def _load_bev_tensor(self, sample_token: str) -> torch.Tensor:
        p = self.bev_dir / f"{sample_token}.pt"
        data = torch.load(p, map_location="cpu", weights_only=True)
        feat = data["features_fused"]
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        return feat

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Alterna ciclicamente tra le 5 categorie
        cat_idx = idx % len(self.target_cats)
        cat = self.target_cats[cat_idx]
        pool = self.by_category[cat]
        item_idx = (idx // len(self.target_cats)) % len(pool)
        item = pool[item_idx]

        token = item["sample_token"]
        q = item["question"]
        ans = normalize_answer(item["answer"])

        bev_features = self._load_bev_tensor(token).squeeze(0)  # [128, 200, 200]

        bev_placeholder = BEV_TOKEN * self.num_bev_tokens
        messages = [
            {"role": "system", "content": SOTA_NUSCENES_SYSTEM_PROMPT},
            {"role": "user", "content": f"{bev_placeholder}\n{q}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        full_text = prompt + ans + self.tokenizer.eos_token
        tokenized_full = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=220,
        )

        input_ids = tokenized_full.input_ids[0]
        attention_mask = tokenized_full.attention_mask[0]
        labels = input_ids.clone()
        prompt_len = min(tokenized_prompt.input_ids.shape[1], input_ids.shape[0])
        labels[:prompt_len] = -100

        return {
            "bev": bev_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def evaluate_val_accuracy(
    model: BEVVLM,
    tokenizer,
    val_items: List[Dict],
    bev_dir: Path,
    device: torch.device,
    num_tokens: int = 32,
    max_new_tokens: int = 10,
) -> Tuple[float, Dict[str, float], List[Dict]]:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    samples_preview = []

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
                {"role": "system", "content": SOTA_NUSCENES_SYSTEM_PROMPT},
                {"role": "user", "content": f"{bev_placeholder}\n{q}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

            if len(samples_preview) < 5:
                samples_preview.append({
                    "category": cat,
                    "question": q,
                    "gt": gt,
                    "pred": pred,
                    "match": is_match,
                })
            del bev, out

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()

    overall_acc = round(correct / max(total, 1) * 100.0, 2)
    cat_accs = {}
    for c, c_tot in cat_total.items():
        cat_accs[c] = round(cat_correct[c] / c_tot * 100.0, 1) if c_tot > 0 else 0.0

    return overall_acc, cat_accs, samples_preview


def parse_args():
    parser = argparse.ArgumentParser(description="SOTA NuScenes-QA Dedicated High-Accuracy Training")
    parser.add_argument("--train-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_train.json")
    parser.add_argument("--val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    parser.add_argument("--bev-dir", type=str, default="/media/nazario.pizzicoli/Datos/dataset/bev_features")
    parser.add_argument("--stage1-projector", type=str, default="checkpoints/sota/stage1/stage1_projector_best.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sota_nuscenes_qa")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16, help="Effective batch size = 16")
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8000, help="Passi di ottimizzazione totali")
    parser.add_argument("--val-steps", type=int, default=250, help="Intervallo di validazione")
    parser.add_argument("--val-samples-per-cat", type=int, default=50, help="50 x 5 = 250 campioni di validazione")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default="outputs/sota_nuscenes_qa_train_report.json")
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
    print(f"{BOLD}{CYAN}  SOTA NUSCENES-QA DEDICATED TRAINING (HIGH-CAPACITY LoRA r=64){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}")
    print(f"  Dispositivo:                {device}")
    print(f"  LoRA Configurazione:        r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Learning Rates:             LoRA={args.lr_lora:.1e}, Projector={args.lr_projector:.1e}")
    print(f"  Passi di Ottimizzazione:    {args.max_steps} (Effective Batch Size = {args.batch_size * args.grad_accum})")
    print(f"  Pesi Proiettore Iniziali:   {args.stage1_projector}")
    print(f"  Directory Checkpoints:      {output_dir}")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")

    tokenizer = build_tokenizer(args.llm_path)

    # 1. Dataset di Training Stratificato
    logger.info("Caricamento dataset di training stratificato...")
    train_dataset = StratifiedBalancedNuScenesDataset(
        json_path=args.train_json,
        bev_dir=bev_dir,
        tokenizer=tokenizer,
        split="train",
        num_bev_tokens=args.num_tokens,
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

    # 2. Pool di Validazione Bilanciato
    logger.info("Caricamento campioni di validazione bilanciati per checkpointing...")
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
    logger.info(f"Pool di validazione pronto: {len(val_eval_items)} campioni bilanciati.")

    # 3. Costruzione Modello con LoRA r=64
    logger.info("Inizializzazione modello BEV-VLM con LoRA r=64...")
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(arch_type=args.arch_type, num_tokens=args.num_tokens),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()

    if os.path.exists(args.stage1_projector):
        logger.info(f"Caricamento pesi Stage 1 Projector da: {args.stage1_projector}")
        model.load_projector(args.stage1_projector)

    model.set_stage(2)  # Proiettore + LoRA sbloccati
    model.llm.gradient_checkpointing_enable()

    param_counts = model.count_trainable_parameters()
    logger.info(
        f"Parametri Trainabili: Proiettore={param_counts['projector_trainable']:,} | "
        f"LoRA={param_counts['llm_trainable']:,} | Totale={param_counts['total_trainable']:,}"
    )

    # 4. Ottimizzatore congiunto SOTA
    optimizer = torch.optim.AdamW([
        {"params": list(model.projector.parameters()), "lr": args.lr_projector, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.llm.named_parameters() if p.requires_grad], "lr": args.lr_lora, "weight_decay": args.weight_decay},
    ])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Valutazione Zero-Shot iniziale
    logger.info("Valutazione accuratezza iniziale pre-training...")
    init_acc, init_cat_accs, _ = evaluate_val_accuracy(
        model, tokenizer, val_eval_items, bev_dir, device, num_tokens=args.num_tokens
    )
    logger.info(f"Accuratezza Iniziale: {init_acc:.2f}% (Categorie: {init_cat_accs})")

    # 5. Training Loop
    logger.info(f"\n{BOLD}{GREEN}>>> INIZIO SOTA TRAINING NUSCENES-QA ({args.max_steps} step)...{RESET}")
    global_step = 0
    accum_loss = 0.0
    best_acc = init_acc
    history = []
    start_time = time.time()

    pbar = tqdm(total=args.max_steps, desc="Training SOTA NuScenes-QA", dynamic_ncols=True)
    train_iter = iter(train_loader)
    sub_step = 0

    while global_step < args.max_steps:
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

        sub_step += 1
        if sub_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            cur_loss = accum_loss
            accum_loss = 0.0
            cur_lr_lora = scheduler.get_last_lr()[1]

            pbar.update(1)
            pbar.set_postfix({"loss": f"{cur_loss:.4f}", "lr": f"{cur_lr_lora:.2e}", "best_acc": f"{best_acc:.1f}%"})

            history.append({
                "step": global_step,
                "loss": cur_loss,
                "lr_lora": cur_lr_lora,
            })

            # Validazione periodica e salvataggio miglior modello
            if global_step % args.val_steps == 0 or global_step == args.max_steps:
                val_acc, val_cats, val_previews = evaluate_val_accuracy(
                    model, tokenizer, val_eval_items, bev_dir, device, num_tokens=args.num_tokens
                )
                logger.info(
                    f"\n[Step {global_step}/{args.max_steps}] Val Acc: {BOLD}{val_acc:.2f}%{RESET} "
                    f"(Record: {best_acc:.2f}%) | Loss: {cur_loss:.4f}"
                )
                logger.info(f"  Categorie: {val_cats}")
                history[-1]["val_accuracy"] = val_acc
                history[-1]["category_accuracies"] = val_cats

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_dir = output_dir / "best_model"
                    model.save_checkpoint(str(best_dir))
                    tokenizer.save_pretrained(str(best_dir / "tokenizer"))
                    logger.info(f"  {GREEN}★ NUOVO RECORD ACCURACY ({val_acc:.2f}%)! Salvato in: {best_dir}{RESET}")

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
    logger.info(f"Ultimo modello salvato in: {latest_dir}")

    # Salva report JSON
    report = {
        "task": "NuScenes-QA SOTA Dedicated Training",
        "llm_backbone": args.llm_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_optimization_steps": global_step,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr_projector": args.lr_projector,
        "lr_lora": args.lr_lora,
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
    logger.info(f"Report di addestramento salvato in: {report_path}")

    print(f"\n{BOLD}{GREEN}======================================================================{RESET}")
    print(f"{BOLD}{GREEN}  TRAINING SOTA NUSCENES-QA COMPLETATO CON SUCCESSO!{RESET}")
    print(f"{BOLD}{GREEN}======================================================================{RESET}")
    print(f"  Tempo totale:        {elapsed / 3600.0:.2f} ore")
    print(f"  Accuratezza Iniziale: {init_acc:.2f}%")
    print(f"  Migliore Accuratezza: {BOLD}{best_acc:.2f}%{RESET}")
    print(f"  Best Checkpoint:     {output_dir / 'best_model'}")
    print(f"  Report:              {report_path}")
    print(f"{GREEN}======================================================================{RESET}\n")


if __name__ == "__main__":
    main()
