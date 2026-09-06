"""
Projector modules for BEV-VQA.
Contains BeLLA-style DeeperConv and BEVDriver-style QFormer projectors.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ProjectorConfig:
    """Configurazione del proiettore (Projector configuration)"""
    in_channels: int = 128
    num_tokens: int = 32
    projector_output_size: int = 3584 # d_llm for Qwen2.5-3B
    arch_type: str = "deeper_conv" # "deeper_conv" or "qformer"
    # per QFormer:
    hidden_size: int = 768
    num_layers: int = 4
    num_heads: int = 12
    # per DeeperConv:
    hidden_channels: int = 256


class DeeperConvProjector(nn.Module):
    """
    BeLLA-style Deeper Conv projector.
    Reduces BEV spatial dimensions via 4 strided convs, then adaptive pool,
    flattens, and maps to d_llm via MLP.
    """
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        
        # 4 layers of Conv2d 3x3 stride 2, BatchNorm, GELU
        channels = [config.in_channels] + [config.hidden_channels] * 4
        layers = []
        for i in range(4):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels[i+1]),
                nn.GELU()
            ])
        self.conv_stack = nn.Sequential(*layers)
        
        # Adaptive pooling to reach exactly sqrt(num_tokens) spatial dims
        pool_size = int(math.sqrt(config.num_tokens))
        if pool_size * pool_size != config.num_tokens:
            raise ValueError(f"num_tokens ({config.num_tokens}) must be a perfect square for DeeperConvProjector")
            
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # MLP mapping to d_llm
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_channels, config.projector_output_size),
            nn.GELU(),
            nn.Linear(config.projector_output_size, config.projector_output_size)
        )
        self.out_norm = nn.LayerNorm(config.projector_output_size)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, 128, 200, 200] (in_channels, H, W)
        Returns:
            visual_tokens: [B, N, d_llm]
        """
        # bev: [B, C, H, W]
        x = self.conv_stack(bev)
        x = self.pool(x)
        
        # [B, C, h, w] -> [B, C, h*w] -> [B, h*w, C]
        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        
        x = self.mlp(x)
        x = self.out_norm(x)
        
        return x


class QFormerProjector(nn.Module):
    """
    BEVDriver-style QFormer projector.
    Uses learnable queries and cross-attention over BEV features.
    """
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        
        self.bev_proj = nn.Linear(config.in_channels, config.hidden_size)
        
        # Learned queries and positional embeddings
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_tokens, config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        
        self.bev_pos_embed = nn.Parameter(torch.zeros(1, 200 * 200, config.hidden_size)) # Assuming 200x200 max
        self.bev_pos_embed.data.normal_(mean=0.0, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.out_proj = nn.Linear(config.hidden_size, config.projector_output_size)
        self.out_norm = nn.LayerNorm(config.projector_output_size)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, 128, 200, 200]
        Returns:
            visual_tokens: [B, N, d_llm]
        """
        B, C, H, W = bev.shape
        # Flatten spatial dims: [B, H*W, C]
        bev_flat = bev.view(B, C, H * W).transpose(1, 2)
        
        # Project BEV to hidden size
        bev_embed = self.bev_proj(bev_flat)
        
        # Add positional embedding to BEV
        # Only take up to H*W in case it's smaller
        bev_embed = bev_embed + self.bev_pos_embed[:, :H*W, :]
        
        # Queries: [B, N, hidden_size]
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Concatenate queries and BEV embeddings for self/cross attention in standard transformer
        # [B, N + H*W, hidden_size]
        concat_embed = torch.cat([queries, bev_embed], dim=1)
        
        # Apply transformer
        out_embed = self.transformer(concat_embed)
        
        # Extract query outputs
        query_out = out_embed[:, :self.config.num_tokens, :]
        
        # Final projection
        x = self.out_proj(query_out)
        x = self.out_norm(x)
        
        return x


def build_projector(config: ProjectorConfig) -> nn.Module:
    """Factory function per creare il proiettore (Projector factory)"""
    if config.arch_type == "deeper_conv":
        return DeeperConvProjector(config)
    elif config.arch_type == "qformer":
        return QFormerProjector(config)
    else:
        raise ValueError(f"Unknown projector architecture: {config.arch_type}")
