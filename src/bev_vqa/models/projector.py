"""
Projector modules for BEV-VQA.

Architecture Options:
1. DeeperConvProjector (BeLLA-style):
   - Progressive strided convolutions on BEV [128, 200, 200]
   - Spatial adaptive pooling to target tokens (e.g., 32 tokens via 4x8 or 36 via 6x6)
   - 2-layer MLP projection to LLM hidden dimension (d_llm)
   - Final LayerNorm (no artificial scale/norm degradation)

2. QFormerProjector (BEVDriver-style):
   - Lightweight conv stem reducing 200x200 -> 25x25 (625 spatial keys/values)
   - N learnable query tokens (e.g. 32)
   - Multi-layer Cross-Attention (queries attend to 625 BEV spatial features)
   - Projection to d_llm + LayerNorm
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectorConfig:
    """Configurazione del proiettore visivo."""
    in_channels: int = 128
    num_tokens: int = 32
    projector_output_size: int = 2048  # d_llm per Qwen2.5-3B
    arch_type: str = "deeper_conv"     # "deeper_conv" oppure "qformer"
    # Parametri per DeeperConv:
    hidden_channels: int = 256
    num_conv_layers: int = 4
    # Parametri per Q-Former:
    qformer_hidden_size: int = 768
    qformer_num_layers: int = 3
    qformer_num_heads: int = 8


def get_spatial_pool_shape(num_tokens: int) -> Tuple[int, int]:
    """Determina la forma 2D (h, w) ottimale per ottenere esattamente num_tokens."""
    root = int(round(math.sqrt(num_tokens)))
    if root * root == num_tokens:
        return (root, root)
    # Cerca i migliori divisori (es. per 32: 4x8)
    for h in range(root, 0, -1):
        if num_tokens % h == 0:
            return (h, num_tokens // h)
    return (1, num_tokens)


class DeeperConvProjector(nn.Module):
    """
    BeLLA-style Deeper Conv projector.
    Convoluzioni progressive strided per preservare gerarchia spaziale,
    pooling adattivo a N token e proiezione lineare nello spazio del LLM.
    """
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config

        channels = [config.in_channels]
        for i in range(config.num_conv_layers):
            channels.append(config.hidden_channels)

        layers = []
        for i in range(config.num_conv_layers):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.GELU()
            ])
        self.conv_stack = nn.Sequential(*layers)

        # Risoluzione spaziale target per ottenere esattamente num_tokens
        self.pool_shape = get_spatial_pool_shape(config.num_tokens)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_shape)

        # Positional embedding spaziale per dare identità unica ai 32 token
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens, config.hidden_channels))
        nn.init.normal_(self.pos_embed, std=0.02)

        # MLP di allineamento dimensionale verso d_llm
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_channels, config.projector_output_size),
            nn.GELU(),
            nn.Linear(config.projector_output_size, config.projector_output_size)
        )
        self.out_norm = nn.LayerNorm(config.projector_output_size)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, 128, 200, 200]
        Returns:
            visual_tokens: [B, num_tokens, d_llm]
        """
        x = self.conv_stack(bev)           # [B, hidden_channels, H_conv, W_conv]
        x = self.pool(x)                   # [B, hidden_channels, h_pool, w_pool]

        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)  # [B, num_tokens, hidden_channels]
        x = x + self.pos_embed[:, :x.shape[1], :] # Aggiunta coordinate spaziali

        x = self.mlp(x)                    # [B, num_tokens, d_llm]
        x = self.out_norm(x)
        return x


class QFormerCrossAttentionBlock(nn.Module):
    """Singolo blocco Q-Former: Self-Attention sulle query + Cross-Attention sui token BEV."""
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, queries: torch.Tensor, bev_tokens: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention tra le query
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + sa_out)

        # 2. Cross-Attention: le query interrogano i token spaziali BEV
        ca_out, _ = self.cross_attn(query=queries, key=bev_tokens, value=bev_tokens)
        queries = self.norm2(queries + ca_out)

        # 3. FFN
        mlp_out = self.mlp(queries)
        queries = self.norm3(queries + mlp_out)
        return queries


class QFormerProjector(nn.Module):
    """
    BEVDriver-style Q-Former projector:
    - Riduce prima la BEV a una griglia 25x25 (625 token) con uno stem convoluzionale leggero.
    - Utilizza N query apprendibili (default: 32) che interrogano i token spaziali della BEV.
    - Proietta le query interrogate nello spazio del LLM.
    """
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        hidden = config.qformer_hidden_size

        # Conv stem: riduce 200x200 -> 100x100 -> 50x50 -> 25x25 (625 token)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(config.in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )

        # N Query apprendibili (BEVDriver style)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_tokens, hidden))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Positional embedding per la griglia 25x25 (625 posizioni)
        self.bev_pos_embed = nn.Parameter(torch.zeros(1, 25 * 25, hidden))
        nn.init.normal_(self.bev_pos_embed, std=0.02)

        # Blocchi di Cross-Attention
        self.blocks = nn.ModuleList([
            QFormerCrossAttentionBlock(hidden, config.qformer_num_heads)
            for _ in range(config.qformer_num_layers)
        ])

        # Proiezione verso d_llm
        self.out_proj = nn.Linear(hidden, config.projector_output_size)
        self.out_norm = nn.LayerNorm(config.projector_output_size)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, 128, 200, 200]
        Returns:
            visual_tokens: [B, num_tokens, d_llm]
        """
        B = bev.shape[0]

        # Stem convoluzionale: [B, 128, 200, 200] -> [B, hidden, 25, 25]
        bev_spatial = self.conv_stem(bev)
        _, C, H, W = bev_spatial.shape
        bev_tokens = bev_spatial.view(B, C, H * W).transpose(1, 2)  # [B, 625, hidden]
        bev_tokens = bev_tokens + self.bev_pos_embed[:, :H * W, :]

        # Query espanse sul batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_tokens, hidden]

        # Cross-Attention
        for block in self.blocks:
            queries = block(queries, bev_tokens)

        # Output projection
        visual_tokens = self.out_proj(queries)
        visual_tokens = self.out_norm(visual_tokens)
        return visual_tokens


def build_projector(config: ProjectorConfig) -> nn.Module:
    """Factory per istanziare il proiettore selezionato."""
    if config.arch_type == "deeper_conv":
        return DeeperConvProjector(config)
    elif config.arch_type == "qformer":
        return QFormerProjector(config)
    else:
        raise ValueError(f"Architettura proiettore sconosciuta: {config.arch_type}")
