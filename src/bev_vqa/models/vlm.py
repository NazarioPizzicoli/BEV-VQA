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
        
        logger.info(f"Caricamento LLM da {config.llm_name_or_path} (Loading LLM)")
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name_or_path,
            torch_dtype=torch.float16,
            device_map="cpu", # will be moved later
        )
        
        # Add <|bev|> token if not present
        if self.tokenizer.convert_tokens_to_ids(config.bev_token) == self.tokenizer.unk_token_id:
            logger.info(f"Aggiunta token {config.bev_token} (Adding BEV token)")
            self.tokenizer.add_special_tokens({"additional_special_tokens": [config.bev_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer))
            
        self.bev_token_id = self.tokenizer.convert_tokens_to_ids(config.bev_token)
        
        # Freeze all LLM params
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Apply LoRA
        logger.info("Configurazione LoRA (Configuring LoRA)")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        # Build projector
        logger.info("Creazione Projector (Building Projector)")
        self.projector = build_projector(config.projector_config)
        
        # Set initial stage
        self.set_stage(1)

    def set_stage(self, stage: int):
        """
        Imposta i parametri trainabili in base allo stage di training.
        Stage 1: solo proiettore (solo allineamento visivo-testuale).
        Stage 2: proiettore + LoRA (finetuning end-to-end).
        """
        logger.info(f"Impostazione training stage: {stage} (Setting stage {stage})")
        if stage == 1:
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.llm.parameters():
                param.requires_grad = False
        elif stage == 2:
            for param in self.projector.parameters():
                param.requires_grad = True
            for param in self.llm.parameters():
                # Peft handles requiring grad for LoRA layers if inference_mode=False
                pass
            # Re-enable LoRA grad just in case
            self.llm.train()
        else:
            raise ValueError(f"Unknown stage: {stage}")
            
    def trainable_parameters(self):
        """Ritorna la lista dei parametri con requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def _build_inputs_embeds(self, bev: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Sostituisce gli embedding del token <|bev|> con i visual tokens generati dal proiettore.
        """
        visual_tokens = self.projector(bev)  # [B, N, d_llm]
        
        # Normal LLM embeddings
        base_model = getattr(self.llm, "model", self.llm) # Handle PeftModel wrapper
        base_model = getattr(base_model, "model", base_model) # Handle Qwen2Model
        inputs_embeds = base_model.embed_tokens(input_ids) # [B, S, d_llm]
        
        B, S, D = inputs_embeds.shape
        _, N, _ = visual_tokens.shape
        
        # Find position of bev token
        new_inputs_embeds = []
        for i in range(B):
            bev_positions = (input_ids[i] == self.bev_token_id).nonzero(as_tuple=True)[0]
            if len(bev_positions) == 0:
                # No bev token, just use original embeds
                new_inputs_embeds.append(inputs_embeds[i])
            else:
                pos = bev_positions[0]
                # Splice in visual tokens
                prefix = inputs_embeds[i, :pos]
                suffix = inputs_embeds[i, pos+1:]
                embed = torch.cat([prefix, visual_tokens[i], suffix], dim=0)
                new_inputs_embeds.append(embed)
                
        return torch.stack(new_inputs_embeds, dim=0)

    def forward(
        self,
        bev: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass.
        """
        inputs_embeds = self._build_inputs_embeds(bev, input_ids)
        
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        bev: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generazione di testo (Inference).
        """
        inputs_embeds = self._build_inputs_embeds(bev, input_ids)
        
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
