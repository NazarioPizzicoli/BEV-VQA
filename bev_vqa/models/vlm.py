"""
BEV-VLM model definition.
Combines Projector, LLM (Qwen2.5-3B-Instruct) and LoRA.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from bev_vqa.constants import BEV_TOKEN
from .projector import ProjectorConfig, build_projector, get_spatial_pool_shape

logger = logging.getLogger(__name__)


@dataclass
class BEVVLMConfig:
    """Main configuration for the BEV-VLM model.
    
    Attributes:
        llm_name_or_path (str): The name or path of the LLM model.
        projector_config (ProjectorConfig): Configuration for the projector.
        lora_r (int): LoRA attention dimension.
        lora_alpha (int): LoRA alpha parameter.
        lora_dropout (float): LoRA dropout probability.
        lora_target_modules (List[str]): List of modules to apply LoRA to.
        bev_token (str): Special token representing the BEV input.
    """
    llm_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    projector_config: ProjectorConfig = field(default_factory=ProjectorConfig)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bev_token: str = BEV_TOKEN


class BEVVLM(nn.Module):
    """BEV-VLM model combining a BEV Projector with an LLM (Qwen2.5) using LoRA."""
    
    def __init__(self, config: BEVVLMConfig, tokenizer: AutoTokenizer) -> None:
        """Initializes the BEVVLM model.
        
        Args:
            config (BEVVLMConfig): Configuration for the model.
            tokenizer (AutoTokenizer): Tokenizer for the LLM.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        logger.info(f"Loading LLM from {config.llm_name_or_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name_or_path,
            torch_dtype=torch.float16,
        )

        # Ensure the projector output dimension matches d_llm
        llm_hidden_size = self.llm.config.hidden_size
        self.config.projector_config.projector_output_size = llm_hidden_size

        # Add the special <|bev|> token if not present in the vocabulary
        if (config.bev_token not in self.tokenizer.get_vocab() and 
            config.bev_token not in self.tokenizer.get_added_vocab()):
            logger.info(f"Adding special token {config.bev_token}")
            self.tokenizer.add_special_tokens({"additional_special_tokens": [config.bev_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.bev_token_id = self.tokenizer.convert_tokens_to_ids(config.bev_token)

        # Freeze all base LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Configure and apply LoRA
        logger.info("Configuring LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # Build the projector (kept in float32 for maximum optimizer numerical stability)
        logger.info(f"Creating Projector ({config.projector_config.arch_type}) -> d_llm={llm_hidden_size}")
        self.projector = build_projector(config.projector_config)
        if hasattr(self.projector, "num_tokens"):
            self.config.projector_config.num_tokens = self.projector.num_tokens

        # Set the initial stage to Stage 1 (projector only)
        self.set_stage(1)

    def set_stage(self, stage: int) -> None:
        """Sets the trainable parameters based on the training stage.
        
        Stage 1: Projector only (visual-textual alignment).
        Stage 2: Projector + LoRA adapters (end-to-end finetuning).
        
        Args:
            stage (int): The training stage (1 or 2).
            
        Raises:
            ValueError: If the stage is not 1 or 2.
        """
        logger.info(f"Setting training stage: {stage}")
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

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns the list of parameters with requires_grad=True.
        
        Returns:
            List[nn.Parameter]: Trainable parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable_parameters(self) -> Dict[str, int]:
        """Returns detailed statistics on the parameters.
        
        Returns:
            Dict[str, int]: A dictionary containing parameter counts.
        """
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
        """Replaces the <|bev|> token(s) with the sequence of visual tokens generated by the projector.
        Simultaneously updates attention_mask and labels to the new sequence length.
        
        Args:
            bev (Optional[torch.Tensor]): BEV input tensor.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (Optional[torch.Tensor], optional): Attention mask.
            labels (Optional[torch.Tensor], optional): Labels.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: Embedded inputs, updated attention mask, and updated labels.
        """
        # Base textual embeddings
        base_embed = self.llm.get_input_embeddings()
        text_embeds = base_embed(input_ids)

        if bev is None:
            return text_embeds, attention_mask, labels

        # Ensure dtype compatibility between BEV and projector
        first_proj_param = next(self.projector.parameters())
        if bev.dtype != first_proj_param.dtype:
            bev = bev.to(dtype=first_proj_param.dtype)

        # Generate visual tokens from the projector
        if getattr(self.projector, "requires_text", False):
            visual_tokens = self.projector(bev, text_embeds=text_embeds)
        else:
            visual_tokens = self.projector(bev)
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
                # No BEV token in this sequence
                new_embeds.append(text_embeds[b])
                if attention_mask is not None:
                    new_masks.append(attention_mask[b])
                if labels is not None:
                    new_labels.append(labels[b])
            else:
                start_pos = bev_positions[0].item()
                end_pos = bev_positions[-1].item() + 1

                # Replace the entire interval [start_pos:end_pos] with the visual tokens [N, D]
                prefix_emb = text_embeds[b, :start_pos]
                suffix_emb = text_embeds[b, end_pos:]
                combined_emb = torch.cat([prefix_emb, visual_tokens[b].to(prefix_emb.dtype), suffix_emb], dim=0)
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
                    # Loss is not calculated on visual tokens (label -100)
                    vis_l = torch.full((N,), -100, dtype=labels.dtype, device=labels.device)
                    combined_l = torch.cat([prefix_l, vis_l, suffix_l], dim=0)
                    new_labels.append(combined_l)

        # Length alignment with padding if necessary
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
        return_aligned_labels: bool = False,
    ) -> Union[CausalLMOutputWithPast, Tuple[CausalLMOutputWithPast, Optional[torch.Tensor]]]:
        """End-to-end forward pass.
        
        Args:
            bev (Optional[torch.Tensor]): BEV input tensor.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (Optional[torch.Tensor], optional): Attention mask.
            labels (Optional[torch.Tensor], optional): Labels.
            return_aligned_labels (bool, optional): Whether to return aligned labels.
            
        Returns:
            Union[CausalLMOutputWithPast, Tuple[CausalLMOutputWithPast, Optional[torch.Tensor]]]: 
                Model output, optionally with aligned labels.
        """
        inputs_embeds, attention_mask, aligned_labels = self.prepare_multimodal_inputs(
            bev=bev,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=aligned_labels,
        )

        if return_aligned_labels:
            return out, aligned_labels
        return out

    @torch.no_grad()
    def generate(
        self,
        bev: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Text generation conditioned on BEV and prompt.
        
        Args:
            bev (Optional[torch.Tensor]): BEV input tensor.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (Optional[torch.Tensor], optional): Attention mask.
            **kwargs: Additional generation arguments.
            
        Returns:
            torch.Tensor: Generated token IDs from the LLM.
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

    def save_projector(self, save_path: str) -> None:
        """Saves only the projector weights.
        
        Args:
            save_path (str): The path to save the projector weights.
        """
        torch.save(self.projector.state_dict(), save_path)
        logger.info(f"Projector weights saved in: {save_path}")

    def load_projector(self, load_path: str, strict: bool = False) -> None:
        """Loads only the projector weights with support for pos_embed interpolation and flexible weights.
        
        Args:
            load_path (str): The path to load the projector weights from.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match. Defaults to False.
        """
        state_dict = torch.load(load_path, map_location="cpu")
        if hasattr(self.projector, "pos_embed") and "pos_embed" in state_dict:
            old_pe = state_dict["pos_embed"]
            new_pe = self.projector.pos_embed
            C = old_pe.shape[-1]
            N_old = old_pe.shape[1]
            N_new = new_pe.shape[1]

            # Retrieve the original grid shape from the checkpoint if present, otherwise fallback
            if "pool_shape_tensor" in state_dict:
                old_h, old_w = tuple(state_dict.pop("pool_shape_tensor").tolist())
            else:
                old_h, old_w = get_spatial_pool_shape(N_old)

            new_h, new_w = getattr(self.projector, "pool_shape", get_spatial_pool_shape(N_new))

            if (old_h, old_w) != (new_h, new_w) or old_pe.shape != new_pe.shape:
                try:
                    pe_2d = old_pe.view(1, old_h, old_w, C).permute(0, 3, 1, 2)
                    pe_interp = F.interpolate(pe_2d, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    state_dict["pos_embed"] = pe_interp.permute(0, 2, 3, 1).reshape(1, N_new, C)
                    logger.info(f"Interpolated pos_embed from {old_h}x{old_w} ({N_old}) to {new_h}x{new_w} ({N_new})")
                except Exception as e:
                    logger.warning(f"Could not interpolate pos_embed ({e}), keeping newly initialized pos_embed.")
                    state_dict.pop("pos_embed")
        elif "pool_shape_tensor" in state_dict:
            state_dict.pop("pool_shape_tensor")
        res = self.projector.load_state_dict(state_dict, strict=strict)
        logger.info(f"Projector weights loaded from: {load_path} (load result={res})")

    def save_checkpoint(self, save_dir: str) -> None:
        """Saves the complete checkpoint: projector + LoRA adapters.
        
        Args:
            save_dir (str): Directory to save the checkpoint.
        """
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), p / "projector.pt")
        self.llm.save_pretrained(p / "lora_adapters")
        logger.info(f"Stage 2 checkpoint saved in: {save_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Loads the complete checkpoint: projector + LoRA adapters.
        
        Args:
            checkpoint_dir (str): Directory to load the checkpoint from.
        """
        p = Path(checkpoint_dir)
        proj_path = p / "projector.pt"
        if proj_path.exists():
            self.load_projector(str(proj_path))
        lora_dir = p / "lora_adapters"
        if lora_dir.exists():
            weights = load_peft_weights(str(lora_dir))
            set_peft_model_state_dict(self.llm, weights)
            logger.info(f"LoRA adapters successfully loaded from: {lora_dir}")
