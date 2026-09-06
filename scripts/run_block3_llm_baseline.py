#!/usr/bin/env python3
"""
Block 3: LLM Text-Only Baseline & LoRA Verification.

Obiettivo:
1. Verificare il caricamento e l'impronta VRAM di Qwen2.5-3B-Instruct + LoRA (r=16, alpha=32)
   sulla GPU RTX 2080 Ti (11 GB).
2. Valutare l'accuratezza linguistica Zero-Shot su NuScenes-QA val (suddivisa per le 5 categorie:
   exist, count, object, status, comparison).
3. Eseguire un mini-finetuning LoRA (Text-Only) per verificare la retropropagazione dei gradienti,
   la convergenza della loss e il picco massimo di VRAM.
4. Stabilire il benchmark di riferimento testuale (baseline) che il VLM dovrà battere.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("block3_llm")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def normalize_answer(text: str) -> str:
    """Normalizza la risposta rimuovendo punteggiatura, spazi extra e maiuscole."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TextQADataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int = 128, is_train: bool = True):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

        self.system_prompt = (
            "You are a driving assistant. Answer the question about the driving scene concisely with a single word or short phrase."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        q_text = item["question"]
        a_text = item["answer"].strip()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": q_text},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if self.is_train:
            full_text = prompt + a_text + self.tokenizer.eos_token
            enc_full = self.tokenizer(full_text, max_length=self.max_len, truncation=True, return_tensors="pt")
            enc_prompt = self.tokenizer(prompt, max_length=self.max_len, truncation=True, return_tensors="pt")

            input_ids = enc_full["input_ids"].squeeze(0)
            attention_mask = enc_full["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Maschera il prompt calcolando la loss solo sui token della risposta
            prompt_len = enc_prompt["input_ids"].shape[1]
            labels[:prompt_len] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            enc = self.tokenizer(prompt, max_length=self.max_len, truncation=True, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "question": q_text,
                "ground_truth": a_text,
                "template_type": item.get("template_type", "unknown")
            }


def collate_train(batch, pad_token_id):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, attention_mask, labels = [], [], []

    for item in batch:
        l = item["input_ids"].shape[0]
        pad_len = max_len - l

        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


# ---------------------------------------------------------------------------
# Valutazione Zero-Shot
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_zero_shot(
    model,
    tokenizer,
    eval_items: List[Dict],
    device: torch.device,
    max_new_tokens: int = 8
) -> Dict:
    model.eval()
    dataset = TextQADataset(eval_items, tokenizer, is_train=False)

    correct = 0
    total = len(dataset)
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    samples_preview = []

    for item in tqdm(dataset, desc="  Valutazione Zero-Shot"):
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        gt = normalize_answer(item["ground_truth"])
        cat = item["template_type"]

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_tokens = gen[0, input_ids.shape[1]:]
        pred_raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        pred = normalize_answer(pred_raw)

        is_match = (pred == gt or gt in pred.split())
        if is_match:
            correct += 1
            cat_correct[cat] += 1
        cat_total[cat] += 1

        if len(samples_preview) < 5:
            samples_preview.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "prediction": pred_raw,
                "is_match": is_match,
                "type": cat
            })

    acc_overall = 100.0 * correct / total
    cat_acc = {cat: 100.0 * cat_correct[cat] / cat_total[cat] for cat in cat_total}

    return {
        "overall_acc": acc_overall,
        "per_category": cat_acc,
        "total_evaluated": total,
        "samples": samples_preview
    }


# ---------------------------------------------------------------------------
# Mini-Training LoRA
# ---------------------------------------------------------------------------

def run_mini_finetuning(
    model,
    train_loader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 2e-4,
    grad_accum: int = 8
) -> Tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="  Mini-Finetuning LoRA")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum
        steps += 1

        if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    return total_loss / max(steps, 1), peak_vram


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 3: LLM Text-Only Baseline")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct", help="Nome LLM")
    parser.add_argument("--n-eval", type=int, default=250, help="Campioni per valutazione Zero-Shot")
    parser.add_argument("--n-train", type=int, default=300, help="Campioni per mini-finetuning LoRA")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  BEV-VQA Block 3: LLM Text-Only Baseline & LoRA Health Check{RESET}")
    print(f"  Model:  {args.model_name}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"{BOLD}{'='*72}{RESET}\n")

    # 1. Carica Tokenizer e Modello
    print(f"{BOLD}[1/4] Caricamento Tokenizer e Qwen2.5-3B-Instruct in float16...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  {GREEN}✓ Modello caricato con successo. VRAM iniziale: {base_vram:.2f} GB / 11 GB{RESET}")

    # 2. Configura LoRA
    print(f"\n{BOLD}[2/4] Configurazione LoRA (r=16, alpha=32, target: all-linear)...{RESET}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    lora_pct = 100.0 * trainable_params / all_params
    print(f"  Parametri totali:     {all_params / 1e9:.2f} B")
    print(f"  Parametri trainabili: {trainable_params / 1e6:.2f} M ({lora_pct:.2f}% del totale)")
    lora_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  {GREEN}✓ LoRA applicato. VRAM dopo LoRA: {lora_vram:.2f} GB / 11 GB{RESET}")

    # 3. Valutazione Zero-Shot su NuScenes-QA
    print(f"\n{BOLD}[3/4] Valutazione Zero-Shot su NuScenes-QA (campione bilanciato per tipo)...{RESET}")
    nsqa_val_path = Path("/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    with open(nsqa_val_path) as f:
        val_all = json.load(f)["questions"]

    # Campiona 50 domande per ciascuna delle 5 categorie principali
    by_type = defaultdict(list)
    for q in val_all:
        by_type[q.get("template_type", "other")].append(q)

    eval_items = []
    n_per_cat = args.n_eval // 5
    for t in ["exist", "count", "object", "status", "comparison"]:
        eval_items.extend(by_type[t][:n_per_cat])

    zero_shot_res = evaluate_zero_shot(model, tokenizer, eval_items, device)

    print(f"\n  {BOLD}Risultati Zero-Shot (Text-Only):{RESET}")
    print(f"  {'Accuratezza Totale:':<26} {BOLD}{zero_shot_res['overall_acc']:.1f}%{RESET}")
    for cat, acc in zero_shot_res["per_category"].items():
        print(f"    - {cat:<18}: {acc:.1f}%")

    # 4. Mini-Finetuning LoRA
    print(f"\n{BOLD}[4/4] Test Mini-Finetuning LoRA ({args.n_train} campioni, gradient accum = 8)...{RESET}")
    nsqa_train_path = Path("/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_train.json")
    with open(nsqa_train_path) as f:
        train_all = json.load(f)["questions"][:args.n_train]

    train_ds = TextQADataset(train_all, tokenizer, is_train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_train(b, tokenizer.pad_token_id)
    )

    final_loss, peak_vram = run_mini_finetuning(model, train_loader, device, epochs=1, lr=2e-4, grad_accum=8)
    print(f"  {GREEN}✓ Training completato con successo. Loss finale: {final_loss:.4f}{RESET}")
    print(f"  {GREEN}✓ Picco massimo VRAM durante il training: {peak_vram:.2f} GB / 11 GB{RESET}")

    # 5. Salva Report
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model_name,
        "base_vram_gb": base_vram,
        "peak_vram_gb": peak_vram,
        "total_parameters": all_params,
        "trainable_parameters": trainable_params,
        "zero_shot_acc_overall": zero_shot_res["overall_acc"],
        "zero_shot_per_category": zero_shot_res["per_category"],
        "sample_predictions": zero_shot_res["samples"],
        "finetuning_final_loss": final_loss,
        "passed": peak_vram < 10.5
    }
    with open(out_dir / "block3_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  REPORT FINALE BLOCK 3: LLM TEXT-ONLY BASELINE{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"  {'Parametro':<36} {'Valore':>20}")
    print(f"  {'-'*60}")
    print(f"  {'Modello Base':<36} {'Qwen2.5-3B-Instruct':>20}")
    print(f"  {'VRAM di Base (fp16)':<36} {f'{base_vram:.2f} GB':>20}")
    print(f"  {'Picco VRAM con LoRA in Training':<36} {f'{peak_vram:.2f} GB':>20}")
    print(f"  {'Margine VRAM libero su 11GB':<36} {f'{11.0 - peak_vram:.2f} GB':>20}")
    overall_str = f"{zero_shot_res['overall_acc']:.1f}%"
    print(f"  {'Accuratezza Zero-Shot (Text-Only)':<36} {overall_str:>20}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"\n  Report salvato in {out_dir / 'block3_report.json'}\n")


if __name__ == "__main__":
    main()
