import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

def check_bev_features(bev_dir: str, n_samples: int = 100) -> Dict[str, Any]:
    """Verifica le feature BEV (shape, dtype, range)."""
    bev_path = Path(bev_dir)
    files = list(bev_path.glob("*.pt"))
    if not files:
        return {"status": "FAIL", "details": f"Nessun file .pt trovato in {bev_dir}", "data": {}}
        
    files = files[:n_samples]
    stats = {"shapes": set(), "dtypes": set(), "all_zeros": 0}
    
    for f in tqdm(files, desc="Checking BEV features"):
        try:
            data = torch.load(f, map_location="cpu")
            if "features_fused" not in data:
                return {"status": "FAIL", "details": f"Chiave features_fused mancante in {f.name}", "data": {}}
                
            tensor = data["features_fused"]
            stats["shapes"].add(tuple(tensor.shape))
            stats["dtypes"].add(str(tensor.dtype))
            
            if torch.all(tensor == 0):
                stats["all_zeros"] += 1
                
        except Exception as e:
            return {"status": "FAIL", "details": f"Errore caricamento {f.name}: {str(e)}", "data": {}}
            
    if stats["all_zeros"] > 0:
        return {"status": "FAIL", "details": f"Trovati {stats['all_zeros']} tensori tutti zeri", "data": stats}
        
    if len(stats["shapes"]) > 1:
        return {"status": "FAIL", "details": f"Shape non uniformi: {stats['shapes']}", "data": stats}
        
    return {"status": "PASS", "details": "BEV features valide", "data": stats}

def check_dataset_json(json_path: str) -> Dict[str, Any]:
    """Verifica la struttura del JSON del dataset."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            
        if "info" not in data:
            return {"status": "FAIL", "details": "Chiave 'info' mancante", "data": {}}
            
        is_pretrain = "descriptions" in data
        is_vqa = "questions" in data
        
        if not is_pretrain and not is_vqa:
            return {"status": "FAIL", "details": "Mancano chiavi 'descriptions' o 'questions'", "data": {}}
            
        items = data.get("descriptions", data.get("questions", []))
        if not items:
            return {"status": "FAIL", "details": "Dataset vuoto", "data": {}}
            
        # Controlla campi richiesti nel primo elemento
        first = items[0]
        if "sample_token" not in first:
            return {"status": "FAIL", "details": "Chiave 'sample_token' mancante negli item", "data": {}}
            
        return {"status": "PASS", "details": "JSON strutturalmente valido", "data": {"num_items": len(items)}}
    except Exception as e:
        return {"status": "FAIL", "details": f"Errore lettura JSON: {str(e)}", "data": {}}

def check_bev_coverage(json_path: str, bev_dir: str) -> Dict[str, Any]:
    """Verifica che ogni sample nel JSON abbia le feature BEV."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            
        items = data.get("descriptions", data.get("questions", []))
        bev_path = Path(bev_dir)
        
        missing = 0
        total = len(items)
        
        for item in tqdm(items, desc="Checking BEV coverage"):
            token = item["sample_token"]
            if not (bev_path / f"{token}.pt").exists():
                missing += 1
                
        if missing > 0:
            return {"status": "FAIL", "details": f"{missing}/{total} sample mancanti in {bev_dir}", "data": {"missing": missing}}
            
        return {"status": "PASS", "details": "Tutti i sample hanno le feature BEV", "data": {"total": total}}
    except Exception as e:
        return {"status": "FAIL", "details": f"Errore coverage: {str(e)}", "data": {}}

def check_class_distribution(json_path: str) -> Dict[str, Any]:
    """Analizza la distribuzione di template e risposte."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            
        if "questions" not in data:
            return {"status": "PASS", "details": "Dataset pretrain (non VQA), skip distrib.", "data": {}}
            
        templates = Counter()
        answers = Counter()
        
        for q in data["questions"]:
            templates[q.get("template_type", "none")] += 1
            answers[str(q.get("answer", ""))] += 1
            
        return {
            "status": "PASS", 
            "details": "Distribuzione calcolata", 
            "data": {
                "top_templates": dict(templates.most_common(5)),
                "top_answers": dict(answers.most_common(10))
            }
        }
    except Exception as e:
        return {"status": "FAIL", "details": f"Errore distribuzione: {str(e)}", "data": {}}

def check_duplicates(json_path: str) -> Dict[str, Any]:
    """Cerca duplicati (sample_token + question)."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            
        if "questions" not in data:
            return {"status": "PASS", "details": "Dataset pretrain, skip duplicati QA", "data": {}}
            
        seen = set()
        duplicates = 0
        
        for q in data["questions"]:
            key = (q["sample_token"], q["question"])
            if key in seen:
                duplicates += 1
            seen.add(key)
            
        if duplicates > 0:
            return {"status": "FAIL", "details": f"Trovati {duplicates} duplicati esatti", "data": {"duplicates": duplicates}}
            
        return {"status": "PASS", "details": "Nessun duplicato trovato", "data": {}}
    except Exception as e:
        return {"status": "FAIL", "details": f"Errore duplicati: {str(e)}", "data": {}}

def run_all_checks(bev_dir: str, dataset_paths: List[str]) -> Dict[str, Any]:
    """Esegue tutti i controlli di sanità."""
    results = {}
    
    logger.info("Running BEV features check...")
    results["bev_features"] = check_bev_features(bev_dir)
    
    for path in dataset_paths:
        name = Path(path).name
        logger.info(f"Checking dataset {name}...")
        
        results[f"{name}_json"] = check_dataset_json(path)
        if results[f"{name}_json"]["status"] == "PASS":
            results[f"{name}_coverage"] = check_bev_coverage(path, bev_dir)
            results[f"{name}_distrib"] = check_class_distribution(path)
            results[f"{name}_duplicates"] = check_duplicates(path)
            
    return results
