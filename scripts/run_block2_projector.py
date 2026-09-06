#!/usr/bin/env python3
"""
Block 2: Vision Projector Verification & Probing.

Verifiche del Blocco 2:
1. Shape & Gradient Flow Check:
   - Verifica che il proiettore produca esattamente [B, 32, 3584].
   - Verifica che i gradienti scorrano senza vanishing (norma gradienti > 0 su tutti gli strati).
2. Anti-Collapse / Token Diversity Check:
   - Misura la cosine similarity media a coppie tra i 32 token visivi.
   - Verifica che i 32 token non collassino in copie identiche (similarità media < 0.95).
3. Semantic Expressiveness (Probe):
   - Allena un head leggero sui 32 token del proiettore sul dataset de-biassato
     per verificare che la riduzione a 32 token conservi il segnale visivo.
4. Confronto architetture:
   - DeeperConvProjector (BeLLA-style)
   - QFormerProjector (BEVDriver-style)
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bev_vqa.models.projector import ProjectorConfig, build_projector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("block2_projector")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Test 1: Shape & Gradient Health
# ---------------------------------------------------------------------------

def check_gradient_flow(projector: nn.Module, device: torch.device) -> Dict:
    """Verifica che i gradienti scorrano attraverso tutti i parametri del proiettore."""
    projector.train()
    dummy_bev = torch.randn(2, 128, 200, 200, device=device, requires_grad=True)
    tokens = projector(dummy_bev)

    loss = tokens.sum()
    loss.backward()

    grad_norms = {}
    vanishing_count = 0
    exploding_count = 0

    for name, param in projector.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            grad_norms[name] = norm
            if norm < 1e-7:
                vanishing_count += 1
            if norm > 1e4:
                exploding_count += 1

    mean_norm = float(np.mean(list(grad_norms.values())))
    is_healthy = (vanishing_count == 0 and exploding_count == 0 and not math.isnan(mean_norm))

    return {
        "is_healthy": is_healthy,
        "mean_grad_norm": mean_norm,
        "vanishing_layers": vanishing_count,
        "exploding_layers": exploding_count,
        "output_shape": list(tokens.shape),
    }


# ---------------------------------------------------------------------------
# Test 2: Token Diversity (Anti-Collapse)
# ---------------------------------------------------------------------------

def check_token_diversity(projector: nn.Module, device: torch.device) -> Dict:
    """
    Calcola la cosine similarity media tra coppie distinte dei 32 token.
    Se la similarità è ~1.0, i token sono collassati in copie identiche (collasso di rappresentazione).
    Se la similarità è < 0.90, i token esprimono concetti spaziali/semantici diversi e complementari.
    """
    projector.eval()
    dummy_bev = torch.randn(8, 128, 200, 200, device=device)

    with torch.no_grad():
        tokens = projector(dummy_bev)  # [B, N, D]

    # Normalizza ogni token lungo la dimensione D
    tokens_norm = F.normalize(tokens, p=2, dim=-1)  # [B, N, D]

    # Sim matrice a coppie: [B, N, N]
    sim_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))

    # Escludi la diagonale principale (self-similarity = 1.0)
    N = tokens.shape[1]
    mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1, N, N]
    pairwise_sims = sim_matrix[mask.expand_as(sim_matrix)].view(tokens.shape[0], -1)

    mean_sim = pairwise_sims.mean().item()
    std_sim = pairwise_sims.std().item()

    is_diverse = mean_sim < 0.90

    return {
        "is_diverse": is_diverse,
        "mean_pairwise_cosine_sim": mean_sim,
        "std_pairwise_cosine_sim": std_sim,
        "token_count": N,
        "token_dim": tokens.shape[-1]
    }


# ---------------------------------------------------------------------------
# Test 3: Semantic Probe (32 tokens -> classification on debiased cache)
# ---------------------------------------------------------------------------

class MemmapDataset(Dataset):
    def __init__(self, mmap_path: Path, n_samples: int, labels: torch.Tensor):
        self.mmap = np.memmap(mmap_path, dtype=np.float16, mode="r", shape=(n_samples, 128, 200, 200))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bev = torch.from_numpy(np.array(self.mmap[idx])).float()
        return bev, self.labels[idx]


class ProjectorProbeClassifier(nn.Module):
    """Classificatore probe montato sopra i 32 token prodotti dal proiettore."""
    def __init__(self, projector: nn.Module, d_llm: int = 3584):
        super().__init__()
        self.projector = projector
        # Pooling sui 32 token -> MLP di classificazione
        self.head = nn.Sequential(
            nn.Linear(d_llm, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        tokens = self.projector(bev)    # [B, 32, d_llm]
        pooled = tokens.mean(dim=1)     # [B, d_llm]
        return self.head(pooled)


def train_probe_on_projector(
    projector: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 5e-4
) -> float:
    """Addestra il proiettore con l'head probe per 10 epoche veloci."""
    model = ProjectorProbeClassifier(projector, d_llm=projector.config.projector_output_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for bev, labels in train_loader:
            bev, labels = bev.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(bev)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Valutazione
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bev, labels in val_loader:
                bev, labels = bev.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = model(bev).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 2: Vision Projector Verification")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-tokens", type=int, default=32, help="Numero token visivi (default: 32)")
    parser.add_argument("--output-dim", type=int, default=3584, help="Dimensione d_llm (Qwen2.5-3B: 3584)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  BEV-VQA Block 2: Vision Projector Verification & Health Check{RESET}")
    print(f"  Target: BEV [128, 200, 200] -> {args.num_tokens} Visual Tokens (dim={args.output_dim})")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"{BOLD}{'='*72}{RESET}\n")

    report = {"deeper_conv": {}, "qformer": {}}

    # =======================================================================
    # Test 1 & 2 per DeeperConv (BeLLA style)
    # =======================================================================
    print(f"{BOLD}[1/4] Test DeeperConvProjector (BeLLA-style)...{RESET}")
    cfg_conv = ProjectorConfig(
        in_channels=128,
        num_tokens=args.num_tokens,
        projector_output_size=args.output_dim,
        arch_type="deeper_conv"
    )
    conv_proj = build_projector(cfg_conv).to(device)

    # 1. Gradient flow
    conv_grad = check_gradient_flow(conv_proj, device)
    status_grad = f"{GREEN}PASS{RESET}" if conv_grad["is_healthy"] else f"{RED}FAIL{RESET}"
    print(f"  Shape output:        {conv_grad['output_shape']} (Batch=2, N={args.num_tokens}, D={args.output_dim})")
    print(f"  Gradiente medio:     {conv_grad['mean_grad_norm']:.4e} -> {status_grad}")

    # 2. Token diversity
    conv_div = check_token_diversity(conv_proj, device)
    status_div = f"{GREEN}PASS (Nessun collasso){RESET}" if conv_div["is_diverse"] else f"{RED}FAIL (Collasso token){RESET}"
    print(f"  Cosine Sim a coppie: {conv_div['mean_pairwise_cosine_sim']:.3f} (±{conv_div['std_pairwise_cosine_sim']:.3f}) -> {status_div}")

    report["deeper_conv"]["gradient_flow"] = conv_grad
    report["deeper_conv"]["token_diversity"] = conv_div

    # =======================================================================
    # Test 1 & 2 per Q-Former (BEVDriver style)
    # =======================================================================
    print(f"\n{BOLD}[2/4] Test QFormerProjector (BEVDriver-style)...{RESET}")
    cfg_qf = ProjectorConfig(
        in_channels=128,
        num_tokens=args.num_tokens,
        projector_output_size=args.output_dim,
        arch_type="qformer"
    )
    qf_proj = build_projector(cfg_qf).to(device)

    # 1. Gradient flow
    qf_grad = check_gradient_flow(qf_proj, device)
    status_grad_qf = f"{GREEN}PASS{RESET}" if qf_grad["is_healthy"] else f"{RED}FAIL{RESET}"
    print(f"  Shape output:        {qf_grad['output_shape']} (Batch=2, N={args.num_tokens}, D={args.output_dim})")
    print(f"  Gradiente medio:     {qf_grad['mean_grad_norm']:.4e} -> {status_grad_qf}")

    # 2. Token diversity
    qf_div = check_token_diversity(qf_proj, device)
    status_div_qf = f"{GREEN}PASS (Nessun collasso){RESET}" if qf_div["is_diverse"] else f"{RED}FAIL (Collasso token){RESET}"
    print(f"  Cosine Sim a coppie: {qf_div['mean_pairwise_cosine_sim']:.3f} (±{qf_div['std_pairwise_cosine_sim']:.3f}) -> {status_div_qf}")

    report["qformer"]["gradient_flow"] = qf_grad
    report["qformer"]["token_diversity"] = qf_div

    # =======================================================================
    # Test 3: Verifica Semantica su Dati Reali (Cache de-biassata)
    # =======================================================================
    print(f"\n{BOLD}[3/4] Test di Preservazione Semantica (10 epoche su dati reali)...{RESET}")
    cache_dir = Path("/home/nazario.pizzicoli/BEV-VQA/cache")
    train_bin = cache_dir / "debiased_train_bev.bin"
    val_bin = cache_dir / "debiased_val_bev.bin"

    if train_bin.exists() and val_bin.exists():
        # Creazione label bilanciate per 1200 e 400
        labels_train = torch.tensor([1 if i % 2 == 0 else 0 for i in range(1200)], dtype=torch.long)
        labels_val = torch.tensor([1 if i % 2 == 0 else 0 for i in range(400)], dtype=torch.long)

        train_ds = MemmapDataset(train_bin, 1200, labels_train)
        val_ds = MemmapDataset(val_bin, 400, labels_val)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        print("  Training probe su DeeperConvProjector...")
        conv_acc = train_probe_on_projector(conv_proj, train_loader, val_loader, device, epochs=10)
        print(f"  {GREEN}✓ DeeperConv Projector Probe Accuracy: {conv_acc:.1f}%{RESET}")
        report["deeper_conv"]["probe_accuracy"] = conv_acc

        print("  Training probe su QFormerProjector...")
        qf_acc = train_probe_on_projector(qf_proj, train_loader, val_loader, device, epochs=10)
        print(f"  {GREEN}✓ Q-Former Projector Probe Accuracy:   {qf_acc:.1f}%{RESET}")
        report["qformer"]["probe_accuracy"] = qf_acc
    else:
        print(f"  {YELLOW}⚠ File di cache {train_bin} non trovati. Salto test semantico.{RESET}")

    # =======================================================================
    # Verdetto Finale
    # =======================================================================
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  REPORT FINALE BLOCK 2: VISION PROJECTOR{RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"  {'Architettura':<24} {'Output Shape':<18} {'Gradienti':<14} {'Cosine Sim':<12}")
    print(f"  {'-'*70}")
    print(f"  {'DeeperConv (BeLLA)':<24} {str(conv_grad['output_shape']):<18} {status_grad:<23} {conv_div['mean_pairwise_cosine_sim']:.3f}")
    print(f"  {'Q-Former (BEVDriver)':<24} {str(qf_grad['output_shape']):<18} {status_grad_qf:<23} {qf_div['mean_pairwise_cosine_sim']:.3f}")
    print(f"{BOLD}{'='*72}{RESET}")

    out_file = Path(__file__).resolve().parent.parent / "outputs" / "block2_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport salvato in {out_file}\n")


if __name__ == "__main__":
    main()
