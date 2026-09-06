"""
BEV-VLM model definition.
Combines Projector, LLM (Qwen2.5-3B-Instruct) and LoRA.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, TaskType

from bev_vqa.models.projector import ProjectorConfig, build_projector

logger = logging.getLogger(__name__)

@dataclass
class BEVVLMConfig:
    """Configurazione principale per il modello BEV-VLM."""
    llm_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    projector_config: ProjectorConfig = field(default_factory=ProjectorConfig)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bev_token: str = "<|bev|>"

class BEVVLM(nn.Module):
    """
    BEV-VLM model: BEV Projector + Qwen2.5 (LoRA).
    """
    def __init__(self, config: BEVVLMConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        logger.info(f"Caricamento LLM da {config.llm_name_or_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name_or_path,
            torch_dtype=torch.float16,
        )

        # Assicura che la dimensione di output del proiettore corrisponda a d_llm
        llm_hidden_size = self.llm.config.hidden_size
        self.config.projector_config.projector_output_size = llm_hidden_size

        # Aggiunta del token speciale <|bev|> se non presente nel vocabolario
        if (config.bev_token not in self.tokenizer.get_vocab() and 
            config.bev_token not in self.tokenizer.get_added_vocab()):
            logger.info(f"Aggiunta token speciale {config.bev_token}")
            self.tokenizer.add_special_tokens({"additional_special_tokens": [config.bev_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.bev_token_id = self.tokenizer.convert_tokens_to_ids(config.bev_token)

        # Congela tutti i parametri base dell'LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Configura e applica LoRA
        logger.info("Configurazione LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # Costruisci il proiettore (mantenuto in float32 per massima stabilita numerica dell'ottimizzatore)
        logger.info(f"Creazione Projector ({config.projector_config.arch_type}) -> d_llm={llm_hidden_size}")
        self.projector = build_projector(config.projector_config)

        # Imposta lo stage iniziale su Stage 1 (solo proiettore)
        self.set_stage(1)

    def set_stage(self, stage: int):
        """
        Imposta i parametri trainabili in base allo stage di training.
        Stage 1: solo proiettore (allineamento visivo-testuale).
        Stage 2: proiettore + LoRA adapters (finetuning end-to-end).
        """
        logger.info(f"Impostazione training stage: {stage}")
        if stage == 1:
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.llm.parameters():
                param.requires_grad = False
        elif stage == 2:
            for param in self.projector.parameters():
                param.requires_grad = True
            for name, param in self.llm.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")

    def trainable_parameters(self):
        """Ritorna la lista dei parametri con requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable_parameters(self) -> Dict[str, int]:
        """Ritorna statistiche dettagliate sui parametri."""
        proj_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        proj_total = sum(p.numel() for p in self.projector.parameters())
        llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        llm_total = sum(p.numel() for p in self.llm.parameters())
        return {
            "projector_trainable": proj_trainable,
            "projector_total": proj_total,
            "llm_trainable": llm_trainable,
            "llm_total": llm_total,
            "total_trainable": proj_trainable + llm_trainable,
            "total_params": proj_total + llm_total,
        }

    def prepare_multimodal_inputs(
        self,
        bev: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sostituisce il/i token <|bev|> con la sequenza di visual tokens generata dal proiettore.
        Aggiorna contemporaneamente attention_mask e labels alla nuova lunghezza sequenza.
        """
        # Embeddings testuali di base
        base_embed = self.llm.get_input_embeddings()
        text_embeds = base_embed(input_ids)  # [B, S, d_llm]

        if bev is None:
            return text_embeds, attention_mask, labels

        # Assicura compatibilità di dtype tra BEV e proiettore
        first_proj_param = next(self.projector.parameters())
        if bev.dtype != first_proj_param.dtype:
            bev = bev.to(dtype=first_proj_param.dtype)

        # Genera i visual tokens dal proiettore
        visual_tokens = self.projector(bev)  # [B, N, d_llm]
        if visual_tokens.dtype != text_embeds.dtype:
            visual_tokens = visual_tokens.to(dtype=text_embeds.dtype)

        B, S, D = text_embeds.shape
        N = visual_tokens.shape[1]

        new_embeds = []
        new_masks = [] if attention_mask is not None else None
        new_labels = [] if labels is not None else None

        for b in range(B):
            bev_positions = (input_ids[b] == self.bev_token_id).nonzero(as_tuple=True)[0]
            if len(bev_positions) == 0:
                # Nessun token BEV in questa sequenza
                new_embeds.append(text_embeds[b])
                if attention_mask is not None:
                    new_masks.append(attention_mask[b])
                if labels is not None:
                    new_labels.append(labels[b])
            else:
                start_pos = bev_positions[0].item()
                end_pos = bev_positions[-1].item() + 1

                # Sostituisci l'intero intervallo [start_pos:end_pos] con i visual tokens [N, D]
                prefix_emb = text_embeds[b, :start_pos]
                suffix_emb = text_embeds[b, end_pos:]
                combined_emb = torch.cat([prefix_emb, visual_tokens[b], suffix_emb], dim=0)
                new_embeds.append(combined_emb)

                if attention_mask is not None:
                    prefix_m = attention_mask[b, :start_pos]
                    suffix_m = attention_mask[b, end_pos:]
                    vis_m = torch.ones((N,), dtype=attention_mask.dtype, device=attention_mask.device)
                    combined_m = torch.cat([prefix_m, vis_m, suffix_m], dim=0)
                    new_masks.append(combined_m)

                if labels is not None:
                    prefix_l = labels[b, :start_pos]
                    suffix_l = labels[b, end_pos:]
                    # Loss non calcolata sui token visivi (etichetta -100)
                    vis_l = torch.full((N,), -100, dtype=labels.dtype, device=labels.device)
                    combined_l = torch.cat([prefix_l, vis_l, suffix_l], dim=0)
                    new_labels.append(combined_l)

        # Allineamento lunghezze con padding se necessario
        max_len = max(e.size(0) for e in new_embeds)
        padded_embeds = []
        padded_masks = [] if attention_mask is not None else None
        padded_labels = [] if labels is not None else None

        for b in range(B):
            cur_len = new_embeds[b].size(0)
            pad_len = max_len - cur_len
            if pad_len > 0:
                pad_emb = torch.zeros((pad_len, D), dtype=new_embeds[b].dtype, device=new_embeds[b].device)
                padded_embeds.append(torch.cat([new_embeds[b], pad_emb], dim=0))
                if attention_mask is not None:
                    pad_m = torch.zeros((pad_len,), dtype=new_masks[b].dtype, device=new_masks[b].device)
                    padded_masks.append(torch.cat([new_masks[b], pad_m], dim=0))
                if labels is not None:
                    pad_l = torch.full((pad_len,), -100, dtype=new_labels[b].dtype, device=new_labels[b].device)
                    padded_labels.append(torch.cat([new_labels[b], pad_l], dim=0))
            else:
                padded_embeds.append(new_embeds[b])
                if attention_mask is not None:
                    padded_masks.append(new_masks[b])
                if labels is not None:
                    padded_labels.append(new_labels[b])

        final_embeds = torch.stack(padded_embeds, dim=0)
        final_masks = torch.stack(padded_masks, dim=0) if attention_mask is not None else None
        final_labels = torch.stack(padded_labels, dim=0) if labels is not None else None

        return final_embeds, final_masks, final_labels

    def forward(
        self,
        bev: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass end-to-end.
        """
        inputs_embeds, attention_mask, labels = self.prepare_multimodal_inputs(
            bev=bev,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        bev: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generazione di testo condizionata da BEV e prompt.
        Restituisce i token ID generati dall'LLM.
        """
        inputs_embeds, attention_mask, _ = self.prepare_multimodal_inputs(
            bev=bev,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

    def save_projector(self, save_path: str):
        """Salva i pesi del solo proiettore."""
        torch.save(self.projector.state_dict(), save_path)
        logger.info(f"Pesi del proiettore salvati in: {save_path}")

    def load_projector(self, load_path: str):
        """Carica i pesi del solo proiettore."""
        state_dict = torch.load(load_path, map_location="cpu")
        self.projector.load_state_dict(state_dict)
        logger.info(f"Pesi del proiettore caricati da: {load_path}")

