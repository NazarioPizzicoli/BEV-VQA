#!/usr/bin/env python3
"""
Block 1: De-Biased Spatial BEV Probe (Option A - FiLM Visual Reasoning).

Test rigoroso e de-biassato per verificare l'estrazione di segnale visivo dalla BEV:
- Campiona esclusivamente domande 'exist' in cui per la STESSA identica domanda
  esistono sia risposte YES che NO in proporzione 1:1.
- In questo modo la baseline linguistica (Text-Only) è matematicamente vincolata al 50.0%.
- Qualsiasi accuratezza superiore al 50% è imputabile al 100% alla percezione visiva dalla BEV.

Architettura visiva: FiLM (Feature-wise Linear Modulation)
- Ad ogni livello convoluzionale (200x200 -> 100x100 -> 50x50 -> 25x25),
  il vettore della domanda genera fattori di scala (gamma) e traslazione (beta)
  per guidare i filtri convoluzionali sull'oggetto richiesto.
- 100% nativo PyTorch cuDNN/cuBLAS (nessun bisogno di compilatori C esterni).
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
logger = logging.getLogger("debiased_probe")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data Selection: Strictly De-Biased
# ---------------------------------------------------------------------------

def get_debiased_dataset(questions: List[Dict], bev_dir: Path, max_samples: int = 1200) -> List[Dict]:
    by_q = {}
    for q in questions:
        if q.get("template_type") == "exist":
            t = q["question"].strip()
            a = q["answer"].strip().lower()
            if a in ("yes", "no") and (bev_dir / f"{q['sample_token']}.pt").exists():
                by_q.setdefault(t, {"yes": [], "no": []})[a].append(q)

    pairs_yes, pairs_no = [], []
    for t, p in by_q.items():
        n = min(len(p["yes"]), len(p["no"]))
        if n >= 2:
            pairs_yes.extend(p["yes"][:n])
            pairs_no.extend(p["no"][:n])

    half = max_samples // 2
    if len(pairs_yes) > half:
        idx = list(range(len(pairs_yes)))
        random.shuffle(idx)
        pairs_yes = [pairs_yes[i] for i in idx[:half]]
        pairs_no = [pairs_no[i] for i in idx[:half]]

    result = pairs_yes + pairs_no
    random.shuffle(result)
    return result


def create_or_load_memmap(items: List[Dict], bev_dir: Path, cache_path: Path, split_name: str) -> np.memmap:
    n_samples = len(items)
    shape = (n_samples, 128, 200, 200)

    if cache_path.exists() and cache_path.stat().st_size == (n_samples * 128 * 200 * 200 * 2):
        logger.info(f"[{split_name}] Trovata cache NVMe valida: {cache_path}")
        return np.memmap(cache_path, dtype=np.float16, mode="r", shape=shape)

    logger.info(f"[{split_name}] Scrittura cache NVMe memmap in {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mmap_arr = np.memmap(cache_path, dtype=np.float16, mode="w+", shape=shape)

    for idx, item in enumerate(tqdm(items, desc=f"  Scrittura cache {split_name}")):
        pt_path = bev_dir / f"{item['sample_token']}.pt"
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features_fused"].squeeze(0).cpu().numpy().astype(np.float16)
        mmap_arr[idx] = feat

    mmap_arr.flush()
    return np.memmap(cache_path, dtype=np.float16, mode="r", shape=shape)


class ProbeDataset(Dataset):
    def __init__(self, mmap_arr: np.memmap, q_ids: torch.Tensor, labels: torch.Tensor):
        self.mmap_arr = mmap_arr
        self.q_ids = q_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bev_np = np.array(self.mmap_arr[idx])
        bev_tensor = torch.from_numpy(bev_np).float()
        return {
            "bev": bev_tensor,
            "q_ids": self.q_ids[idx],
            "label": self.labels[idx]
        }


class TextOnlyDataset(Dataset):
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
# Modelli
# ---------------------------------------------------------------------------

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
        emb = (self.embed(q_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.mlp(emb)


class TextOnlyBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.enc = TextBackbone(vocab_size, embed_dim=embed_dim, out_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, q_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(q_ids))


class FiLMConvBlock(nn.Module):
    """Blocco convoluzionale con modulazione feature-wise guidata dal testo."""
    def __init__(self, in_c: int, out_c: int, text_dim: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.film = nn.Linear(text_dim, 2 * out_c)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        film_params = self.film(text_emb)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class FiLMBEVProbe(nn.Module):
    """
    Rete Convoluzionale per Visual Reasoning su Mappa BEV:
    - 4 stadi convoluzionali a piena risoluzione [128, 200, 200]
    - Ogni strato è modulato dal testo della domanda (FiLM).
    - 100% nativo PyTorch cuDNN/cuBLAS (veloce, stabile, zero dipendenze esterne).
    """
    def __init__(self, vocab_size: int, text_dim: int = 128):
        super().__init__()
        self.text_enc = TextBackbone(vocab_size, embed_dim=128, out_dim=text_dim)

        self.block1 = FiLMConvBlock(128, 128, text_dim)   # 200 -> 100
        self.block2 = FiLMConvBlock(128, 256, text_dim)   # 100 -> 50
        self.block3 = FiLMConvBlock(256, 256, text_dim)   # 50 -> 25
        self.block4 = FiLMConvBlock(256, 256, text_dim)   # 25 -> 13

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(256 + text_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, bev: torch.Tensor, q_ids: torch.Tensor) -> torch.Tensor:
        t_feat = self.text_enc(q_ids)

        x = self.block1(bev, t_feat)
        x = self.block2(x, t_feat)
        x = self.block3(x, t_feat)
        x = self.block4(x, t_feat)

        x_pooled = self.pool(x).flatten(1)
        fused = torch.cat([x_pooled, t_feat], dim=-1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Training & Eval
# ---------------------------------------------------------------------------

def train_and_eval(model, train_loader, val_loader, device, is_text_only=False, epochs=15, lr=5e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            q_ids = batch["q_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            if is_text_only:
                logits = model(q_ids)
            else:
                bev = batch["bev"].to(device, non_blocking=True)
                logits = model(bev, q_ids)

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["q_ids"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                if is_text_only:
                    preds = model(q_ids).argmax(dim=-1)
                else:
                    bev = batch["bev"].to(device, non_blocking=True)
                    preds = model(bev, q_ids).argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += len(labels)

        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            name = "Text-Only" if is_text_only else "FiLM BEV Probe"
            print(f"  [{name:14s}] Epoca {epoch+1:2d}/{epochs} | Acc: {acc:.1f}% (Best: {best_acc:.1f}%)")

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 1: De-biased BEV Probe with FiLM")
    parser.add_argument("--epochs", type=int, default=20, help="Epoche")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  BEV-VQA Block 1: DE-BIASED FEATURE PROBE (FiLM Visual Reasoning){RESET}")
    print(f"  Dataset: Coppie di domande a invarianza linguistica perfetta (50% YES / 50% NO)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"{BOLD}{'='*72}{RESET}\n")

    bev_base = Path("/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh")
    unified = Path("/media/nazario.pizzicoli/Datos/vqa_datasets/unified")
    cache_dir = Path("/home/nazario.pizzicoli/BEV-VQA/cache")

    # 1. Carica domande de-biassate
    print(f"{BOLD}[1/4] Estrazione subset de-biassato (invarianza linguistica)...{RESET}")
    with open(unified / "nuscenes_qa_train.json") as f:
        train_all = json.load(f)["questions"]
    with open(unified / "nuscenes_qa_val.json") as f:
        val_all = json.load(f)["questions"]

    train_items = get_debiased_dataset(train_all, bev_base / "train", max_samples=1200)
    val_items = get_debiased_dataset(val_all, bev_base / "val", max_samples=400)

    print(f"  Train: {len(train_items)} campioni (50% Yes / 50% No esatti)")
    print(f"  Val:   {len(val_items)} campioni (50% Yes / 50% No esatti)")

    # 2. Tokenizzazione
    vocab = build_vocab(train_items)
    train_q = tokenize_questions(train_items, vocab)
    val_q = tokenize_questions(val_items, vocab)
    train_labels = torch.tensor([1 if q["answer"].strip().lower() == "yes" else 0 for q in train_items], dtype=torch.long)
    val_labels = torch.tensor([1 if q["answer"].strip().lower() == "yes" else 0 for q in val_items], dtype=torch.long)

    # 3. Cache NVMe
    print(f"\n{BOLD}[2/4] Setup cache NVMe memmap...{RESET}")
    train_mmap = create_or_load_memmap(train_items, bev_base / "train", cache_dir / "debiased_train_bev.bin", "Train")
    val_mmap = create_or_load_memmap(val_items, bev_base / "val", cache_dir / "debiased_val_bev.bin", "Val")

    text_train_loader = DataLoader(TextOnlyDataset(train_q, train_labels), batch_size=args.batch_size, shuffle=True)
    text_val_loader = DataLoader(TextOnlyDataset(val_q, val_labels), batch_size=args.batch_size, shuffle=False)

    full_train_loader = DataLoader(ProbeDataset(train_mmap, train_q, train_labels), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    full_val_loader = DataLoader(ProbeDataset(val_mmap, val_q, val_labels), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Text-Only Baseline
    print(f"\n{BOLD}[3/4] Test Baseline TEXT-ONLY (atteso esattamente ~50.0%)...{RESET}")
    text_model = TextOnlyBaseline(len(vocab)).to(device)
    best_text_acc = train_and_eval(text_model, text_train_loader, text_val_loader, device, is_text_only=True, epochs=args.epochs, lr=1e-3)
    print(f"  {GREEN}✓ Text-Only Baseline Accuracy: {best_text_acc:.1f}%{RESET}")

    # 5. FiLM BEV Probe
    print(f"\n{BOLD}[4/4] Test FiLM BEV PROBE (Visual Reasoning 2D)...{RESET}")
    film_model = FiLMBEVProbe(len(vocab), text_dim=128).to(device)
    best_bev_acc = train_and_eval(film_model, full_train_loader, full_val_loader, device, is_text_only=False, epochs=args.epochs, lr=5e-4)
    print(f"  {GREEN}✓ FiLM BEV Probe Accuracy: {best_bev_acc:.1f}%{RESET}")

    # 6. Confronto
    gap = best_bev_acc - best_text_acc

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  VERDETTO FINALE DE-BIASED BEV PROBE{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"  {'Metodo':<40} {'Accuratezza':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Casuale (Random)':<40} {'50.0%':>12}")
    print(f"  {'Text-Only Baseline (senza BEV)':<40} {best_text_acc:>11.1f}%")
    print(f"  {'FiLM BEV Probe (BEV + Domanda)':<40} {best_bev_acc:>11.1f}%")
    print(f"  {'-'*70}")
    delta_color = GREEN if gap >= 5.0 else (YELLOW if gap > 0 else RED)
    print(f"  {BOLD}Segnale Visivo Puro Estratto dalla BEV (Δ):{RESET} {delta_color}{BOLD}{gap:>+10.1f}%{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")

    out_file = Path(__file__).resolve().parent.parent / "outputs" / "debiased_probe_report.json"
    with open(out_file, "w") as f:
        json.dump({
            "text_only_acc": best_text_acc,
            "bev_film_acc": best_bev_acc,
            "visual_signal_gap": gap,
            "passed": gap >= 5.0
        }, f, indent=2)
    print(f"\nReport salvato in {out_file}\n")


if __name__ == "__main__":
    main()
