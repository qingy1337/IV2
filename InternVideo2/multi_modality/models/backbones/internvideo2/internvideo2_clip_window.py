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

class WindowInternVideo_old(InternVideo2):
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

class WindowInternVideo(InternVideo2):
    def __init__(self, *args, num_update_layers=8, **kwargs):
        """
        Initializes the WindowInternVideo model.

        Args:
            *args: Arguments passed to the base InternVideo2 class.
            num_update_layers (int): Number of layers in the UpdateTransformer. Defaults to 8.
            **kwargs: Keyword arguments passed to the base InternVideo2 class.
        """
        super().__init__(*args, **kwargs)

        # Assuming base class sets self.embed_dim, self.num_heads, etc.
        # Pass required parameters from base class attributes
        self.update_transformer = UpdateTransformer(
            embed_dim=self.embed_dim,
            num_heads=getattr(self, 'num_heads', 12), # Get from base or use default
            mlp_ratio=getattr(self, 'mlp_ratio', 4.0),
            qkv_bias=getattr(self, 'qkv_bias', True),
            norm_layer=getattr(self, 'norm_layer_for_blocks', nn.LayerNorm), # Get from base or use default
            num_layers=num_update_layers
        )

        # The internal state (current_embedding, frame_buffer etc.) is removed
        # and must be managed externally.


    def forward_full(self, x, use_image = False):
        """
        Computes the embedding for a window of frames using the base model's
        full forward pass.

        Args:
            x: Input frames [B, C, T, H, W].
            use_image (bool): Passed to the base InternVideo2.forward() function.

        Returns:
            torch.Tensor: The computed embedding [B, L, C]. This tensor will
                          have requires_grad=True if the base model parameters do.
        """
        if not (len(x.shape) == 5):
             raise ValueError(f"forward_full expects input shape [B, C, T, H, W], but got {x.shape}")

        embedding = super().forward(x, use_image=use_image)
        return embedding

    def forward_update(self, frame, prev_embedding):
        """
        Updates the embedding with a single new frame using the UpdateTransformer.
        Args:
            frame: Input frame [B, C, H, W].
            prev_embedding: The POOLED embedding from the previous step [B, C_embed_dim].
        Returns:
            torch.Tensor: The updated POOLED embedding [B, C_embed_dim].
        """
        if not (len(frame.shape) == 4):
                raise ValueError(f"forward_update expects frame shape [B, C, H, W], but got {frame.shape}")

        # --- MODIFIED CHECK ---
        # Original check that caused the error:
        # if prev_embedding is not None and not (len(prev_embedding.shape) == 3):
        #      raise ValueError(f"forward_update expects prev_embedding shape [B, L, C] or None, but got {prev_embedding.shape}")
        # Modified check assuming prev_embedding is the [B, C_embed_dim] pooled embedding:
        if prev_embedding is not None:
            if not (len(prev_embedding.shape) == 2 and prev_embedding.shape[1] == self.embed_dim):
                raise ValueError(f"forward_update expects prev_embedding shape [B, embed_dim ({self.embed_dim})] or None, but got {prev_embedding.shape}")
            prev_embedding = prev_embedding.to(frame.device)
        else:
            # Handle case where prev_embedding is None (e.g., very first update if not initialized by forward_full)
            # This might indicate a logic error in the training loop if prev_embedding is expected.
            # For now, let's assume training loop always initializes it.
            # If UpdateTransformer requires a non-None prev_embedding, this path needs care.
            # Given your training loop initializes it from forward_full, this 'else' might not be hit.
            raise ValueError("prev_embedding cannot be None after the initial step in the current training logic.")


        frame_5d = frame.unsqueeze(2) # Shape [B, C, 1, H, W]
        new_frame_tokens = self.patch_embed(frame_5d) # Expected output: e.g., [B, 1, num_patches_one_frame, embed_dim]

        # Reshape new_frame_tokens to [B, L_new, C_embed_dim]
        # L_new here is the number of patch tokens for a single frame.
        B_nft, T_nft, L_patch_nft, C_nft = new_frame_tokens.shape # T_nft should be 1
        new_frame_tokens = new_frame_tokens.view(B_nft, T_nft * L_patch_nft, self.embed_dim)

        # This call now assumes that UpdateTransformer can take:
        # prev_embedding: [B, C_embed_dim]
        # new_frame_tokens: [B, L_new, C_embed_dim]
        # And returns an updated embedding of shape: [B, C_embed_dim]
        updated_embedding = self.update_transformer(
            prev_embedding,
            new_frame_tokens
        )

        # Ensure updated_embedding is also [B, C_embed_dim]
        if not (len(updated_embedding.shape) == 2 and updated_embedding.shape[0] == frame.shape[0] and updated_embedding.shape[1] == self.embed_dim):
            raise ValueError(f"UpdateTransformer produced an output of shape {updated_embedding.shape}, but expected [B, {self.embed_dim}]")

        return updated_embedding

    # Remove reset_state, pop_last_frame, original forward, forward_inference
    # These methods are replaced by forward_full and forward_update

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
