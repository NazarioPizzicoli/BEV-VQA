#!/usr/bin/env python3
"""
BEV-VLM Interactive Inference & Demo Tool.

Permette di testare il modello multimodale BEV-VLM addestrato su qualsiasi scena NuScenes:
1. Modalità Interattiva (REPL):
   python scripts/run_inference.py --interactive
   - Naviga tra le scene NuScenes, visualizza le domande reali e fai qualsiasi domanda libera.
2. Modalità Singola Query (CLI):
   python scripts/run_inference.py --token <sample_token> --question "Are there any cars ahead?"
3. Modalità Benchmark Campione:
   python scripts/run_inference.py --sample-val --num-samples 5
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from bev_vqa.data.dataset import build_tokenizer, normalize_answer, VQA_SYSTEM_PROMPT
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig

logging.basicConfig(level=logging.WARNING)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


class BEVInferenceEngine:
    """Motore di inferenza per BEV-VLM."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/stage2/best_model",
        llm_path: str = "Qwen/Qwen2.5-3B-Instruct",
        arch_type: str = "deeper_conv",
        num_tokens: int = 32,
        bev_dirs: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if bev_dirs is None:
            self.bev_dirs = [
                Path("/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh/val"),
                Path("/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh/train"),
            ]
        else:
            self.bev_dirs = [Path(p) for p in bev_dirs]

        print(f"{CYAN}Caricamento tokenizer da: {llm_path}...{RESET}")
        self.tokenizer = build_tokenizer(llm_path)

        print(f"{CYAN}Inizializzazione architettura BEVVLM (d_llm=2048, tokens={num_tokens})...{RESET}")
        cfg = BEVVLMConfig(
            llm_name_or_path=llm_path,
            projector_config=ProjectorConfig(arch_type=arch_type, num_tokens=num_tokens),
        )
        self.model = BEVVLM(cfg, self.tokenizer).to(self.device)
        self.model.projector.float()

        ckpt_path = Path(checkpoint_dir)
        if ckpt_path.exists():
            print(f"{GREEN}Caricamento pesi checkpoint da: {checkpoint_dir}...{RESET}")
            self.model.load_checkpoint(checkpoint_dir)
        else:
            print(f"{YELLOW}ATTENZIONE: Checkpoint {checkpoint_dir} non trovato. Pesi non inizializzati!{RESET}")

        self.model.eval()
        print(f"{BOLD}{GREEN}✓ Motore di inferenza pronto su dispositivo: {self.device}{RESET}\n")

        # Cache dei token disponibili
        self.available_tokens = {}
        for b_dir in self.bev_dirs:
            if b_dir.exists():
                for p in b_dir.glob("*.pt"):
                    if p.stem not in self.available_tokens:
                        self.available_tokens[p.stem] = p

    def find_bev_path(self, sample_token: str) -> Optional[Path]:
        """Cerca il file .pt della BEV per il sample_token dato."""
        if sample_token in self.available_tokens:
            return self.available_tokens[sample_token]
        for b_dir in self.bev_dirs:
            p = b_dir / f"{sample_token}.pt"
            if p.exists():
                self.available_tokens[sample_token] = p
                return p
        return None

    def load_bev_tensor(self, sample_token: str) -> Optional[torch.Tensor]:
        """Carica il tensore BEV [1, 128, 200, 200]."""
        path = self.find_bev_path(sample_token)
        if path is None:
            return None
        data = torch.load(path, map_location="cpu", weights_only=True)
        feat = data["features_fused"]
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        return feat.to(self.device)

    def answer_question(
        self,
        sample_token: str,
        question: str,
        max_new_tokens: int = 15,
        temperature: float = 0.0,
    ) -> Dict:
        """
        Esegue l'inferenza multimodale per una data domanda sulla scena NuScenes specificata.
        """
        bev = self.load_bev_tensor(sample_token)
        if bev is None:
            return {
                "success": False,
                "error": f"Feature BEV per il token {sample_token} non trovate nei percorsi specificati.",
            }

        prompt_text = (
            f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<|bev|>\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            gen_ids = self.model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.time() - t0) * 1000.0

        raw_answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        norm_answer = normalize_answer(raw_answer)

        return {
            "success": True,
            "sample_token": sample_token,
            "question": question,
            "raw_answer": raw_answer,
            "normalized_answer": norm_answer,
            "latency_ms": round(elapsed_ms, 1),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 2) if torch.cuda.is_available() else 0.0,
        }


def run_interactive_session(engine: BEVInferenceEngine, questions_db: Optional[List[Dict]] = None):
    """Sessione REPL interattiva nel terminale."""
    print(f"\n{BOLD}{MAGENTA}======================================================================{RESET}")
    print(f"{BOLD}{MAGENTA}  MODALITÀ DEMO INTERATTIVA BEV-VLM (DIGITA '/help' PER I COMANDI){RESET}")
    print(f"{BOLD}{MAGENTA}======================================================================{RESET}")
    print(f"  Scene BEV disponibili indicizzate: {len(engine.available_tokens):,}")
    print(f"  Comandi speciali:")
    print(f"    - '/random'         : Sceglie una scena casuale")
    print(f"    - '/token <token>'  : Imposta una specifica scena tramite sample_token")
    print(f"    - '/qa'             : Mostra le domande ufficiali associate alla scena")
    print(f"    - '/quit' o 'exit'  : Esce dalla demo")
    print(f"{MAGENTA}----------------------------------------------------------------------{RESET}\n")

    # Mappa domande per token
    token_to_qa = defaultdict(list)
    if questions_db:
        for it in questions_db:
            token_to_qa[it.get("sample_token")].append(it)

    # Scelta token iniziale
    token_list = list(engine.available_tokens.keys())
    if not token_list:
        print(f"{YELLOW}Nessun file BEV trovato nelle directory specificate.{RESET}")
        return

    current_token = random.choice(token_list)
    print(f"{GREEN}Scena corrente attiva:{RESET} {BOLD}{current_token}{RESET}")
    q_count = len(token_to_qa.get(current_token, []))
    print(f"({q_count} domande ufficiali trovate per questa scena)\n")

    while True:
        try:
            user_input = input(f"{BOLD}{BLUE}[Scena: {current_token[:8]}...] Domanda > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita dalla sessione.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "exit", "quit", "q"]:
            print("Sessione terminata.")
            break

        if user_input.lower() == "/random":
            current_token = random.choice(token_list)
            print(f"\n{GREEN}Nuova scena selezionata:{RESET} {BOLD}{current_token}{RESET}")
            q_count = len(token_to_qa.get(current_token, []))
            print(f"({q_count} domande ufficiali disponibili. Digita '/qa' per vederle)\n")
            continue

        if user_input.lower().startswith("/token "):
            parts = user_input.split()
            if len(parts) > 1:
                req_token = parts[1].strip()
                if engine.find_bev_path(req_token):
                    current_token = req_token
                    print(f"\n{GREEN}Scena impostata con successo:{RESET} {BOLD}{current_token}{RESET}\n")
                else:
                    print(f"{YELLOW}Token '{req_token}' non trovato nei file BEV indicizzati.{RESET}\n")
            continue

        if user_input.lower() == "/qa":
            q_list = token_to_qa.get(current_token, [])
            if not q_list:
                print(f"{YELLOW}Nessuna domanda ufficiale nel file JSON per questa scena.{RESET}\n")
            else:
                print(f"\n{BOLD}{CYAN}Domande ufficiali NuScenes-QA per questa scena:{RESET}")
                for idx, q_item in enumerate(q_list[:8], 1):
                    print(f"  {idx}. [{q_item.get('template_type', 'qa')}] {q_item['question']} -> {GREEN}{q_item['answer']}{RESET}")
                print()
            continue

        if user_input.lower() == "/help":
            print("Comandi: /random, /token <id>, /qa, /quit")
            continue

        # Esecuzione Inferenza
        res = engine.answer_question(current_token, user_input)
        if not res["success"]:
            print(f"{YELLOW}Errore: {res['error']}{RESET}\n")
            continue

        print(f"\n{BOLD}{GREEN}BEV-VLM Risposta:{RESET} {BOLD}{res['raw_answer']}{RESET}  {CYAN}({res['latency_ms']} ms){RESET}\n")


def run_sample_eval(engine: BEVInferenceEngine, num_samples: int = 5, val_json_path: Optional[str] = None):
    """Valuta un piccolo campione casuale di validazione e stampa i risultati dettagliati."""
    if val_json_path is None:
        val_json_path = "/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json"

    val_path = Path(val_json_path)
    if not val_path.exists():
        print(f"{YELLOW}File {val_json_path} non trovato!{RESET}")
        return

    with open(val_path) as f:
        data = json.load(f).get("questions", [])

    valid_samples = [d for d in data if engine.find_bev_path(d.get("sample_token")) is not None]
    if not valid_samples:
        print(f"{YELLOW}Nessun campione valido trovato con file BEV associato.{RESET}")
        return

    samples = random.sample(valid_samples, min(num_samples, len(valid_samples)))

    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}  VALUTAZIONE CAMPIONE INFERENZA ({len(samples)} campioni casuali){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    correct = 0
    for idx, item in enumerate(samples, 1):
        token = item["sample_token"]
        q = item["question"]
        gt = normalize_answer(item["answer"])
        cat = item.get("template_type", "unknown")

        res = engine.answer_question(token, q)
        pred = res["normalized_answer"]
        is_match = (pred == gt or gt in pred.split())
        if is_match:
            correct += 1

        status_str = f"{GREEN}✓ CORRETTO{RESET}" if is_match else f"{YELLOW}✗ ERRATO{RESET}"

        print(f"{BOLD}Campione #{idx}:{RESET}")
        print(f"  Token:         {token}")
        print(f"  Categoria:     {cat}")
        print(f"  Domanda:       {q}")
        print(f"  Ground Truth:  {BOLD}{gt}{RESET}")
        print(f"  BEV-VLM:       {BOLD}{res['raw_answer']}{RESET}  [{status_str}]  ({res['latency_ms']} ms)")
        print(f"{CYAN}{'-'*70}{RESET}")

    acc = correct / len(samples) * 100.0
    print(f"\n{BOLD}Accuratezza su questo campione ({len(samples)} campioni): {GREEN if acc >= 50 else YELLOW}{acc:.1f}%{RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="BEV-VLM Inference & Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage2/best_model", help="Cartella del checkpoint salvato")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--arch-type", type=str, default="deeper_conv")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--token", type=str, default=None, help="Sample token per inferenza a singola query")
    parser.add_argument("--question", type=str, default=None, help="Domanda in linguaggio naturale")
    parser.add_argument("--interactive", action="store_true", help="Avvia la sessione interattiva REPL")
    parser.add_argument("--sample-val", action="store_true", help="Esegue l'inferenza su campioni casuali di validazione")
    parser.add_argument("--num-samples", type=int, default=5, help="Numero di campioni per --sample-val")
    parser.add_argument("--nuscenes-val-json", type=str, default="/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json")
    return parser.parse_args()


def main():
    args = parse_args()

    engine = BEVInferenceEngine(
        checkpoint_dir=args.checkpoint,
        llm_path=args.llm_path,
        arch_type=args.arch_type,
        num_tokens=args.num_tokens,
    )

    # 1. Inferenza a singola query da linea di comando
    if args.token and args.question:
        res = engine.answer_question(args.token, args.question)
        print(f"\n{BOLD}{CYAN}--- Risultato Inferenza ---{RESET}")
        print(f"  Scena Token:  {res['sample_token']}")
        print(f"  Domanda:      {res['question']}")
        print(f"  Risposta:     {BOLD}{GREEN}{res['raw_answer']}{RESET}")
        print(f"  Latenza:      {res['latency_ms']} ms")
        print(f"  VRAM In Uso:  {res['vram_allocated_gb']} GB\n")
        return

    # 2. Valutazione su campione
    if args.sample_val:
        run_sample_eval(engine, num_samples=args.num_samples, val_json_path=args.nuscenes_val_json)
        return

    # 3. Sessione Interattiva (Default se nessun argomento specificato o con --interactive)
    questions_db = None
    if os.path.exists(args.nuscenes_val_json):
        try:
            with open(args.nuscenes_val_json) as f:
                questions_db = json.load(f).get("questions", [])
        except Exception:
            pass

    run_interactive_session(engine, questions_db=questions_db)


if __name__ == "__main__":
    main()
