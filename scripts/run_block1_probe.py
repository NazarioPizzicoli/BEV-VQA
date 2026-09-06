#!/usr/bin/env python3
"""
Block 1: BEV Feature Probe (Fast NVMe Memmap Version - Zero OOM Risk).

Obiettivo:
Dimostrare se le feature BEV [128, 200, 200] contengono segnale per rispondere
alle domande 'exist' (yes/no), a piena risoluzione spaziale.

Innovazione tecnica per l'efficienza:
- Memory-Mapped files (np.memmap) memorizzati sull'SSD NVMe locale veloce.
- Zero rischio OOM: i file risiedono su SSD (57 GB disponibili), caricati
  dall'OS in streaming ad altissima velocità (2 GB/s) per batch.
- Baseline Text-Only pura: addestrata in ~3 secondi senza toccare i file BEV.
- Full BEV+Text CNN probe: 15 epoche completate in ~1-2 minuti.
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("block1_probe")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data Preparation & NVMe Memmap Caching
# ---------------------------------------------------------------------------

def select_balanced_samples(
    questions: List[Dict],
    bev_dir: Path,
    n_total: int,
    split_name: str
) -> List[Dict]:
    n_per_class = n_total // 2
    yes_items, no_items = [], []

    for q in questions:
        ans = q.get("answer", "").strip().lower()
        if ans not in ("yes", "no"):
            continue
        token = q["sample_token"]
        pt_path = bev_dir / f"{token}.pt"
        if not pt_path.exists():
            continue

        if ans == "yes" and len(yes_items) < n_per_class:
            yes_items.append(q)
        elif ans == "no" and len(no_items) < n_per_class:
            no_items.append(q)

        if len(yes_items) >= n_per_class and len(no_items) >= n_per_class:
            break

    selected = yes_items + no_items
    random.shuffle(selected)
    logger.info(
        f"[{split_name}] Selezionati {len(selected)} campioni bilanciati "
        f"(Yes: {len(yes_items)}, No: {len(no_items)})."
    )
    return selected


def create_or_load_memmap(
    items: List[Dict],
    bev_dir: Path,
    cache_path: Path,
    split_name: str
) -> np.memmap:
    """
    Crea o carica un file memory-mapped sull'SSD locale.
    Questo evita OOM e consente letture istantanee a batch durante il training.
    """
    n_samples = len(items)
    shape = (n_samples, 128, 200, 200)

    if cache_path.exists() and cache_path.stat().st_size == (n_samples * 128 * 200 * 200 * 2):
        logger.info(f"[{split_name}] Trovata cache NVMe valida: {cache_path} ({cache_path.stat().st_size / 1e9:.2f} GB)")
        return np.memmap(cache_path, dtype=np.float16, mode="r", shape=shape)

    logger.info(f"[{split_name}] Creazione cache NVMe memmap in {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mmap_arr = np.memmap(cache_path, dtype=np.float16, mode="w+", shape=shape)

    for idx, item in enumerate(tqdm(items, desc=f"  Scrittura cache NVMe {split_name}")):
        pt_path = bev_dir / f"{item['sample_token']}.pt"
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features_fused"].squeeze(0).cpu().numpy().astype(np.float16)
        mmap_arr[idx] = feat

    mmap_arr.flush()
    logger.info(f"[{split_name}] Cache completata ({n_samples * 128 * 200 * 200 * 2 / 1e9:.2f} GB scritti su SSD NVMe).")
    return np.memmap(cache_path, dtype=np.float16, mode="r", shape=shape)


class MemmapProbeDataset(Dataset):
    def __init__(self, mmap_arr: np.memmap, q_ids: torch.Tensor, labels: torch.Tensor):
        self.mmap_arr = mmap_arr
        self.q_ids = q_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Lettura streaming a zero-RAM da NVMe:
        bev_np = np.array(self.mmap_arr[idx])
        bev_tensor = torch.from_numpy(bev_np).float()
        return {
            "bev": bev_tensor,
            "q_ids": self.q_ids[idx],
            "label": self.labels[idx]
        }


class TextOnlyProbeDataset(Dataset):
    def __init__(self, q_ids: torch.Tensor, labels: torch.Tensor):
        self.q_ids = q_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "q_ids": self.q_ids[idx],
            "label": self.labels[idx]
        }


def build_vocab(items: List[Dict], max_words: int = 3000) -> Dict[str, int]:
    counts = Counter()
    for item in items:
        counts.update(item["question"].lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in counts.most_common(max_words):
        vocab[w] = len(vocab)
    return vocab


def tokenize_questions(items: List[Dict], vocab: Dict[str, int], max_len: int = 24) -> torch.Tensor:
    all_tokens = []
    for item in items:
        words = item["question"].lower().split()[:max_len]
        ids = [vocab.get(w, 1) for w in words]
        ids += [0] * (max_len - len(ids))
        all_tokens.append(ids)
    return torch.tensor(all_tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Modelli di Probing
# ---------------------------------------------------------------------------

class BEVConvBackbone(nn.Module):
    """
    CNN a 4 stadi convoluzionali progressivi su BEV reale [128, 200, 200]:
    200x200 -> 100x100 -> 50x50 -> 25x25 -> 13x13 -> AdaptiveAvgPool -> [out_dim]
    """
    def __init__(self, in_channels: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)
        return self.proj(feats)


class TextBackbone(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, q_ids: torch.Tensor) -> torch.Tensor:
        mask = (q_ids != 0).float().unsqueeze(-1)
        emb = self.embed(q_ids) * mask
        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.mlp(pooled)


class TextOnlyProbeModel(nn.Module):
    def __init__(self, vocab_size: int, q_dim: int = 128):
        super().__init__()
        self.text_enc = TextBackbone(vocab_size, embed_dim=128, out_dim=q_dim)
        self.head = nn.Sequential(
            nn.Linear(q_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, q_ids: torch.Tensor) -> torch.Tensor:
        q_feat = self.text_enc(q_ids)
        return self.head(q_feat)


class FullBEVProbeModel(nn.Module):
    def __init__(self, vocab_size: int, bev_dim: int = 256, q_dim: int = 128):
        super().__init__()
        self.bev_enc = BEVConvBackbone(in_channels=128, out_dim=bev_dim)
        self.text_enc = TextBackbone(vocab_size, embed_dim=128, out_dim=q_dim)
        self.fusion = nn.Sequential(
            nn.Linear(bev_dim + q_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, bev: torch.Tensor, q_ids: torch.Tensor) -> torch.Tensor:
        bev_feat = self.bev_enc(bev)
        text_feat = self.text_enc(q_ids)
        fused = torch.cat([bev_feat, text_feat], dim=-1)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# Training & Eval
# ---------------------------------------------------------------------------

def train_and_eval_text_only(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3
) -> Tuple[float, Dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_acc = 0.0
    best_stats = {}

    for epoch in range(epochs):
        model.train()
        total_loss, total_count = 0.0, 0
        for batch in train_loader:
            q_ids = batch["q_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(q_ids)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_count += len(labels)

        # Eval
        model.eval()
        val_correct, val_total = 0, 0
        class_correct = {0: 0, 1: 0}
        class_total = {0: 0, 1: 0}

        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["q_ids"].to(device)
                labels = batch["label"].to(device)

                preds = model(q_ids).argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                for c in [0, 1]:
                    mask = labels == c
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()

        acc = 100.0 * val_correct / val_total
        if acc > best_acc:
            best_acc = acc
            best_stats = {
                "acc": acc,
                "acc_no": 100.0 * class_correct[0] / max(class_total[0], 1),
                "acc_yes": 100.0 * class_correct[1] / max(class_total[1], 1),
            }

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  [Text-Only] Epoca {epoch+1:2d}/{epochs} | Loss: {total_loss/total_count:.4f} | Acc: {acc:.1f}% (Best: {best_acc:.1f}%)")

    return best_acc, best_stats


def train_and_eval_full(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 5e-4
) -> Tuple[float, Dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_acc = 0.0
    best_stats = {}

    for epoch in range(epochs):
        model.train()
        total_loss, total_count = 0.0, 0
        for batch in train_loader:
            bev = batch["bev"].to(device, non_blocking=True)
            q_ids = batch["q_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(bev, q_ids)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            total_count += len(labels)

        # Eval
        model.eval()
        val_correct, val_total = 0, 0
        class_correct = {0: 0, 1: 0}
        class_total = {0: 0, 1: 0}

        with torch.no_grad():
            for batch in val_loader:
                bev = batch["bev"].to(device, non_blocking=True)
                q_ids = batch["q_ids"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    preds = model(bev, q_ids).argmax(dim=-1)

                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                for c in [0, 1]:
                    mask = labels == c
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()

        acc = 100.0 * val_correct / val_total
        if acc > best_acc:
            best_acc = acc
            best_stats = {
                "acc": acc,
                "acc_no": 100.0 * class_correct[0] / max(class_total[0], 1),
                "acc_yes": 100.0 * class_correct[1] / max(class_total[1], 1),
            }

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  [BEV+Text]  Epoca {epoch+1:2d}/{epochs} | Loss: {total_loss/total_count:.4f} | Acc: {acc:.1f}% (Best: {best_acc:.1f}%)")

    return best_acc, best_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 1: Fast NVMe Memmap BEV Feature Probe")
    parser.add_argument("--train-samples", type=int, default=1200, help="Campioni bilanciati train (default: 1200)")
    parser.add_argument("--val-samples", type=int, default=400, help="Campioni bilanciati val (default: 400)")
    parser.add_argument("--epochs", type=int, default=15, help="Epoche di training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  BEV-VQA Block 1: BEV Feature Probe (Piena Risoluzione [128, 200, 200]){RESET}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Campioni: {args.train_samples} Train / {args.val_samples} Val (Bilanciati 50/50)")
    print(f"  Storage Strategy: NVMe Memmap (Zero rischio OOM, velocità streaming massima)")
    print(f"{BOLD}{'='*72}{RESET}\n")

    bev_base = Path("/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh")
    unified = Path("/media/nazario.pizzicoli/Datos/vqa_datasets/unified")
    cache_dir = Path("/home/nazario.pizzicoli/BEV-VQA/cache")

    # 1. Carica e seleziona campioni bilanciati
    print(f"{BOLD}[1/5] Selezione campioni bilanciati (Yes/No)...{RESET}")
    with open(unified / "nuscenes_qa_train.json") as f:
        train_all = [q for q in json.load(f)["questions"] if q.get("template_type") == "exist"]
    with open(unified / "nuscenes_qa_val.json") as f:
        val_all = [q for q in json.load(f)["questions"] if q.get("template_type") == "exist"]

    train_items = select_balanced_samples(train_all, bev_base / "train", args.train_samples, "Train")
    val_items = select_balanced_samples(val_all, bev_base / "val", args.val_samples, "Val")

    # 2. Tokenizzazione testo
    print(f"\n{BOLD}[2/5] Tokenizzazione domande...{RESET}")
    vocab = build_vocab(train_items)
    train_q = tokenize_questions(train_items, vocab)
    val_q = tokenize_questions(val_items, vocab)
    train_labels = torch.tensor([1 if q["answer"].strip().lower() == "yes" else 0 for q in train_items], dtype=torch.long)
    val_labels = torch.tensor([1 if q["answer"].strip().lower() == "yes" else 0 for q in val_items], dtype=torch.long)
    print(f"  Vocabolario: {len(vocab)} parole.")

    # 3. Cache Memmap su NVMe per BEV
    print(f"\n{BOLD}[3/5] Setup cache NVMe per le feature BEV a piena risoluzione...{RESET}")
    train_mmap = create_or_load_memmap(train_items, bev_base / "train", cache_dir / "probe_train_bev.bin", "Train")
    val_mmap = create_or_load_memmap(val_items, bev_base / "val", cache_dir / "probe_val_bev.bin", "Val")

    # 4. Baseline Text-Only
    print(f"\n{BOLD}[4/5] Training Baseline TEXT-ONLY (solo domanda, zero BEV)...{RESET}")
    text_train_loader = DataLoader(TextOnlyProbeDataset(train_q, train_labels), batch_size=args.batch_size, shuffle=True)
    text_val_loader = DataLoader(TextOnlyProbeDataset(val_q, val_labels), batch_size=args.batch_size, shuffle=False)

    text_model = TextOnlyProbeModel(vocab_size=len(vocab)).to(device)
    best_text_acc, text_stats = train_and_eval_text_only(
        text_model, text_train_loader, text_val_loader, device, epochs=args.epochs, lr=1e-3
    )
    print(f"  {GREEN}✓ Text-Only Best Accuracy: {best_text_acc:.1f}%{RESET}")

    # 5. Full Probe (BEV + Text)
    print(f"\n{BOLD}[5/5] Training FULL PROBE (BEV [128, 200, 200] + Domanda)...{RESET}")
    full_train_loader = DataLoader(
        MemmapProbeDataset(train_mmap, train_q, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    full_val_loader = DataLoader(
        MemmapProbeDataset(val_mmap, val_q, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    full_model = FullBEVProbeModel(vocab_size=len(vocab)).to(device)
    best_full_acc, full_stats = train_and_eval_full(
        full_model, full_train_loader, full_val_loader, device, epochs=args.epochs, lr=5e-4
    )
    print(f"  {GREEN}✓ Full (BEV + Text) Best Accuracy: {best_full_acc:.1f}%{RESET}")

    # 6. Risultati & Report
    gap = best_full_acc - best_text_acc
    passed = gap >= 5.0

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  REPORT FINALE BLOCK 1: BEV FEATURE PROBE{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"  {'Metodo':<36} {'Acc Totale':>12} {'Acc No':>10} {'Acc Yes':>10}")
    print(f"  {'-'*70}")
    print(f"  {'Casuale (Random)':<36} {'50.0%':>12} {'50.0%':>10} {'50.0%':>10}")
    print(f"  {'Maggioranza (Majority)':<36} {'50.0%':>12} {'100.0%':>10} {'0.0%':>10}")
    print(f"  {'Text-Only (solo domanda)':<36} {best_text_acc:>11.1f}% {text_stats.get('acc_no', 0.0):>9.1f}% {text_stats.get('acc_yes', 0.0):>9.1f}%")
    print(f"  {'Full (BEV + Domanda)':<36} {best_full_acc:>11.1f}% {full_stats.get('acc_no', 0.0):>9.1f}% {full_stats.get('acc_yes', 0.0):>9.1f}%")
    print(f"  {'-'*70}")
    delta_color = GREEN if gap >= 5.0 else (YELLOW if gap > 0 else RED)
    print(f"  {BOLD}Contributo informativo BEV (Δ):{RESET} {delta_color}{BOLD}{gap:>+10.1f}%{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")

    if passed:
        print(f"\n  {GREEN}{BOLD}✓ PASSATO: Le feature BEV contengono segnale solido (+{gap:.1f}% vs testo)!{RESET}")
        print(f"  Procediamo con la pipeline VLM sapendo che la percezione visiva è funzionante.\n")
    else:
        print(f"\n  {RED}{BOLD}✗ NON PASSATO: Il contributo della BEV è di {gap:.1f}% (richiesto: >= +5.0%){RESET}\n")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "NuScenes-QA (exist questions)",
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "resolution": "128x200x200",
        "random_baseline": 50.0,
        "majority_baseline": 50.0,
        "text_only_acc": best_text_acc,
        "text_only_stats": text_stats,
        "full_probe_acc": best_full_acc,
        "full_probe_stats": full_stats,
        "bev_gap": gap,
        "passed": passed
    }
    with open(out_dir / "block1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report salvato in {out_dir / 'block1_report.json'}")


if __name__ == "__main__":
    main()
