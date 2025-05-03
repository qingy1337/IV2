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

from .internvideo2_clip_vision import *

def log(text):
    print('---- WindowInternVideo2.forward() ----\n' + text)

class WindowInternVideo2(InternVideo2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_transformer = UpdateTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=self.norm_layer_for_blocks,
            num_layers=8
        )

        # Initialize state
        self.reset_state()

    def reset_state(self, reset_current_embedding = True):
        """
        Resets the internal state of the WindowInternVideo2 model.

        This includes:
        - Resetting the frame count to 0.
        - Optionally resetting the current embedding to None (if `reset_current_embedding` is True).
        - Clearing the frame buffer.

        Args:
            reset_current_embedding (bool): Whether to reset the current embedding.
                                            Defaults to True. If False, the current embedding is preserved.
        """
        self.frame_count = 0

        if reset_current_embedding:
            self.current_embedding = None

        # Buffer to collect frames until we have 8
        self.frame_buffer = []

    def pop_last_frame(self):
        """
        Removes and returns the last frame from the frame buffer, if it exists.

        Returns:
            torch.Tensor: The last frame in the buffer (shape [B, C, H, W]) or None if the buffer is empty.
        """
        if len(self.frame_buffer) > 0:
            last_frame = self.frame_buffer.pop()
            return last_frame
        else:
            return None

    def forward(self, x, use_image = False, force_full_forward=False):
        """
        Args:
            x: Input frame [B, C, H, W] or sequence [B, C, T, H, W].
            force_full_forward: Force full forward pass.
            use_image: Passed on to the InternVideo2.forward() function (if it's a full forward).

        ,------------------------------------------------------------------------------------.
        | force_full_forward == True:                                                        |
        |   > 8 frames will be added to the frame_buffer.                                    |
        |   > The embedding for those 8 frames will be calculated using the original forward |
        |     function.                                                                      |
        |------------------------------------------------------------------------------------|
        | force_full_forward == False:                                                       |
        |   > Function expects [B, C, H, W] shape.                                           |
        |   > The new (singular) frame will be added to the frame buffer.                    |
        |   > The embedding is calculated via the UpdateTransformer.                         |
        `------------------------------------------------------------------------------------'
        """
        # Check input shape
        if len(x.shape) == 4:  # Single frame
            B, C, H, W = x.shape
            x = x.unsqueeze(2)  # Add T dimension [B, C, 1, H, W]

        B, C, T, H, W = x.shape

        # Add new frame(s) to buffer
        self.frame_buffer.extend([x[:,:,i] for i in range(T)])

        if force_full_forward: # Full forward pass with the original model
            frames = torch.stack(self.frame_buffer[-8:], dim=2)  # [B, C, 8, H, W]

            self.current_embedding = super().forward(frames, use_image=use_image)

            # ───────────────────────────────────────
            #   We don't need to reset the state here, at least for training.
            #
            #   Resetting the state looks like:
            #       self.reset_state(reset_current_embedding = False)
            # ───────────────────────────────────────
        else:
            # Do update with new frame(s)
            for i in range(T):
                frame = x[:,:,i:i+1]  # [B, C, 1, H, W]
                new_frame_tokens = self.patch_embed(frame)  # [B, 1, L, C]
                B, T, L, C = new_frame_tokens.shape
                new_frame_tokens = new_frame_tokens.view([B, T * L, C])

                self.current_embedding = self.update_transformer(
                    self.current_embedding,
                    new_frame_tokens
                )
                self.frame_count += 1


        return self.current_embedding

    def forward_inference(self, x, use_image = False, force_full_forward=False):
        """
        Args:
            x: Input frame [B, C, H, W] or sequence [B, C, T, H, W].
            force_full_forward: Force full forward pass.
            use_image: Passed on to the InternVideo2.forward() function (if it's a full forward).

        ,------------------------------------------------------------------------------------.
        | force_full_forward == True:                                                        |
        |   > 8 frames will be added to the frame_buffer.                                    |
        |   > The embedding for those 8 frames will be calculated using the original forward |
        |     function.                                                                      |
        |------------------------------------------------------------------------------------|
        | force_full_forward == False:                                                       |
        |   > Function expects [B, C, H, W] shape.                                           |
        |   > The new (singular) frame will be added to the frame buffer.                    |
        |   > The embedding is calculated via the UpdateTransformer.                         |
        `------------------------------------------------------------------------------------'
        """
        # Check input shape
        if len(x.shape) == 4:  # Single frame
            B, C, H, W = x.shape
            x = x.unsqueeze(2)  # Add T dimension [B, C, 1, H, W]

        B, C, T, H, W = x.shape

        # Add new frame(s) to buffer
        self.frame_buffer.extend([x[:,:,i] for i in range(T)])

        # If we have 8 frames or force_full_forward
        if force_full_forward or len(self.frame_buffer) >= 8:
            # Take last 8 frames and do full forward
            frames = torch.stack(self.frame_buffer[-8:], dim=2)  # [B, C, 8, H, W]

            self.current_embedding = super().forward(frames, use_image=use_image)

            self.reset_state(reset_current_embedding = False)
        else:
            # Do update with new frame(s)
            for i in range(T):
                frame = x[:,:,i:i+1]  # [B, C, 1, H, W]
                new_frame_tokens = self.patch_embed(frame)  # [B, 1, L, C]
                B, T, L, C = new_frame_tokens.shape
                new_frame_tokens = new_frame_tokens.view([B, T * L, C])

                self.current_embedding = self.update_transformer(
                    self.current_embedding,
                    new_frame_tokens
                )
                self.frame_count += 1

        return self.current_embedding

class UpdateTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer,
        num_layers: int = 8  # Smaller than main transformer
    ):
        super().__init__()

        # Query token to extract updated embedding
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Layers to process update
        self.layers = nn.ModuleList([
            UpdateLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer
            ) for _ in range(num_layers)
        ])

        # Initialize query token
        trunc_normal_(self.query_token, std=.02)

    def forward(self, prev_embedding, new_frame_tokens):
        # prev_embedding: [B, embed_dim]
        # new_frame_tokens: [B, 256, embed_dim] (256 tokens from new frame)

        # Expand prev_embedding for attention
        prev_embedding = prev_embedding.unsqueeze(1)  # [B, 1, embed_dim]

        # Start with query token
        x = self.query_token.expand(prev_embedding.shape[0], -1, -1)

        # Process through update layers
        for layer in self.layers:
            x = layer(x, prev_embedding, new_frame_tokens)

        return x.squeeze(1)  # Back to [B, embed_dim]

class UpdateLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer
    ):
        super().__init__()

        # Cross attention to previous embedding
        self.norm1 = norm_layer(embed_dim)
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=qkv_bias,
            batch_first=True  # Add this parameter
        )

        # Cross attention to new tokens
        self.norm2 = norm_layer(embed_dim)
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=qkv_bias,
            batch_first=True  # Add this parameter
        )

        # MLP block
        self.norm3 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU
        )

    def forward(self, x, prev_embedding, new_tokens):
        # Apply normalization first
        query1 = self.norm1(x)
        key1 = value1 = self.norm1(prev_embedding)

        # First cross attention
        attn_output1, _ = self.cross_attn1(query1, key1, value1)
        x = x + attn_output1

        # Second cross attention
        query2 = self.norm2(x)
        key2 = value2 = self.norm2(new_tokens)

        attn_output2, _ = self.cross_attn2(query2, key2, value2)
        x = x + attn_output2

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x
