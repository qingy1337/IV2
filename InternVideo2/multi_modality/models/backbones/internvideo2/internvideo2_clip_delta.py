# --- Reused/Adapted Components from InternVideo2 ---
from .internvideo2_clip_vision import CrossAttention, AttentiveBlock, AttentionPoolingBlock, RMSNorm, LayerScale, Attention, Mlp, Block, PatchEmbed

import logging
import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange

# Assuming your provided InternVideo2 components (CrossAttention, AttentiveBlock, etc.)
# are in the same file or accessible via relative import like `.pos_embed`.
# For this example, I'll assume they are in the current scope or a utils file.
# You might need to adjust imports based on your project structure.
try:
    from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
except ImportError:
    # Fallback if running as a script, assuming pos_embed.py is in the same directory
    # You'll need to create a dummy pos_embed.py with these functions if it doesn't exist
    # or provide the actual implementation.
    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        # Dummy implementation
        grid_h = grid_w = grid_size
        grid = np.arange(grid_h * grid_w).reshape(grid_h, grid_w)
        pos_embed = np.zeros((grid_h * grid_w, embed_dim))
        if cls_token:
            pos_embed = np.zeros((1 + grid_h * grid_w, embed_dim))
        print(f"Warning: Using dummy get_2d_sincos_pos_embed. Output shape: {pos_embed.shape}")
        return pos_embed

    def get_1d_sincos_pos_embed(embed_dim, t_size):
        # Dummy implementation
        pos_embed = np.zeros((t_size, embed_dim))
        print(f"Warning: Using dummy get_1d_sincos_pos_embed. Output shape: {pos_embed.shape}")
        return pos_embed

    import numpy as np


# --- Re-pasting necessary components from your InternVideo2 code ---
# (CrossAttention, AttentiveBlock, AttentionPoolingBlock, RMSNorm, LayerScale, Attention, Mlp, Block, PatchEmbed)
