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
        super().__init__(*args, **kwargs)
        # embed_dim, num_heads, etc. are available from base class
        self.update_transformer = UpdateTransformer(
            embed_dim=self.embed_dim, # This is InternVideo2's main embed_dim (e.g., 1408)
            num_heads=getattr(self, 'num_heads', 16),
            mlp_ratio=getattr(self, 'mlp_ratio', 48/11),
            qkv_bias=getattr(self, 'qkv_bias', False),
            norm_layer=getattr(self, 'norm_layer_for_blocks', partial(RMSNorm, eps=1e-6)),
            num_layers=num_update_layers
        )
        # num_frames and tubelet_size are important for patch_embed and pos_embed consistency
        self.num_model_frames = self.patch_embed.grid_size[0] * self.patch_embed.tubelet_size
        self.model_tubelet_size = self.patch_embed.tubelet_size

        self.grid_size = self.patch_embed.grid_size


    def _extract_pre_projector_features(self, x, use_image=False):
        """
        Replicates InternVideo2.forward() up to before the final clip_projector.
        Returns sequence of features: [B, NumTokens, self.embed_dim].
        NumTokens = (num_patches_spatiotemporal) + 1 (for CLS token).
        """
        # Input x: [B, C, T_in, H, W]
        # Ensure T_in matches what patch_embed expects, or handle pos_embed carefully.
        # If T_in for x is self.num_model_frames, pos_embed logic is simpler.
        # For single frame in forward_update, patch_embed T_in=1.

        # --- 1. Patch Embedding ---
        # Type casting for patch_embed input
        x_dtype = self.patch_embed.proj.weight.dtype
        feat = self.patch_embed(x.type(x_dtype)) # Output: B x T_patch x L_patch x C_embed_dim

        B, T_patch, L_patch, C_emb = feat.shape
        feat = feat.view([B, T_patch * L_patch, C_emb])

        # --- 2. CLS Token ---
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls_tokens, feat), dim=1) # [B, 1 + T_patch*L_patch, C_embed_dim]

        # --- 3. Positional Embedding ---
        # This part is complex and MUST match InternVideo2's logic precisely,
        # especially how `use_image` and varying T_patch affect it.

        current_num_patches = T_patch * L_patch
        expected_spatial_patches_per_frame = self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2]

        if self.sep_pos_embed:
            # For sep_pos_embed, spatial and temporal are separate.
            # self.pos_embed_spatial: [1, H_grid*W_grid, C_emb]
            # self.pos_embed_temporal: [1, T_grid, C_emb]
            # self.grid_size[0] is T_grid for FULL configured num_frames

            if use_image: # Typically means T_patch = 1
                if T_patch != 1:
                    # This case needs specific definition or indicates an issue.
                    # If use_image=True, input x to patch_embed should have T=1 frame.
                    print(f"Warning: use_image=True but T_patch={T_patch}. Assuming T_patch=1 for pos_embed.")
                # Use only spatial pos_embed, repeated/expanded if needed, but typically it's just one set of spatial embeddings.
                _pos_embed_spatial = self.pos_embed_spatial
                if _pos_embed_spatial.shape[1] != L_patch:
                     raise ValueError(f"L_patch ({L_patch}) mismatch with self.pos_embed_spatial.shape[1] ({_pos_embed_spatial.shape[1]}) for use_image=True")
                pos_embed = _pos_embed_spatial
            else: # Video clip
                if T_patch != self.grid_size[0]:
                    # This can happen if forward_full is called with a different number of frames
                    # than the model was configured with. Positional embedding interpolation/slicing might be needed.
                    # For simplicity, assume T_patch == self.grid_size[0] for non-use_image cases.
                    raise ValueError(f"T_patch ({T_patch}) mismatch with model's temporal grid size ({self.grid_size[0]}) for sep_pos_embed video.")

                pos_embed = self.pos_embed_spatial.repeat(
                    1, T_patch, 1 # Repeat spatial part T_patch times
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal, # Temporal part
                    expected_spatial_patches_per_frame, # L_patch is expected_spatial_patches_per_frame here
                    dim=1,
                )

            pos_embed = torch.cat(
                [self.pos_embed_cls.expand(B, -1, -1), pos_embed], dim=1,
            )
        else: # Combined 3D pos_embed
            # self.pos_embed is [1, 1 + T_grid * H_grid * W_grid, C_emb]
            if use_image: # T_patch = 1
                if T_patch != 1: print(f"Warning: use_image=True but T_patch={T_patch}. Assuming T_patch=1 for pos_embed.")
                cls_pos_embed = self.pos_embed[:, :1, :]
                # Temporally average the patch positional embeddings from the full 3D pos_embed
                # L_patch here is num_spatial_patches for one frame
                img_pos_embed = self.pos_embed[:, 1:, :].view(
                    1, self.grid_size[0], expected_spatial_patches_per_frame, C_emb
                ).mean(dim=1) # Average over T_grid dimension -> [1, L_spatial, C_emb]
                if img_pos_embed.shape[1] != L_patch:
                    raise ValueError(f"L_patch ({L_patch}) mismatch with calculated img_pos_embed shape ({img_pos_embed.shape[1]}) for use_image=True non-sep.")
                pos_embed = torch.cat([cls_pos_embed, img_pos_embed], dim=1)
            else: # Video clip
                if (1 + current_num_patches) != self.pos_embed.shape[1] :
                    # This implies input clip to forward_full does not match configured T*H*W patches
                    # Needs interpolation or error.
                    raise ValueError(f"Num patches in input ({current_num_patches}) + CLS does not match self.pos_embed size ({self.pos_embed.shape[1]}) for video non-sep.")
                pos_embed = self.pos_embed

        feat = feat + pos_embed.to(feat.device, non_blocking=True)

        # --- 4. Transformer Blocks ---
        residual = None
        for blk in self.blocks:
            if isinstance(feat, tuple) and len(feat) == 2: # Unpack if previous block returned (tensor, residual)
                feat, residual = feat
            feat = blk(feat, residual=residual)

        # After loop, handle final (tensor, residual) if present (from fused RMSNorm)
        if isinstance(feat, tuple) and len(feat) == 2:
            _f_tensor, _r_tensor = feat
            if _r_tensor is not None:
                feat = _f_tensor + _r_tensor
            else: # Should not happen if _r_tensor is None and it's a tuple (unless block logic changes)
                feat = _f_tensor

        # feat is now the sequence of features [B, NumTokens, self.embed_dim]
        return feat

    def forward_full(self, x, use_image=False):
        """
        Computes the CLS token embedding for a window of frames using the base model's
        logic before the final projection.

        Args:
            x: Input frames [B, C, T, H, W].
            use_image (bool): Passed to the feature extraction logic.

        Returns:
            torch.Tensor: The CLS token embedding [B, self.embed_dim].
        """
        if not (len(x.shape) == 5):
            raise ValueError(f"forward_full expects input shape [B, C, T, H, W], but got {x.shape}")

        # Get sequence of features [B, NumTokens, self.embed_dim]
        pre_projection_features = self._extract_pre_projector_features(x, use_image=use_image)

        # Extract the CLS token (assumed to be at index 0)
        cls_embedding = pre_projection_features[:, 0]  # Shape: [B, self.embed_dim]
        return cls_embedding

    def forward_update(self, frame, prev_embedding):
        """
        Updates the embedding with a single new frame using the UpdateTransformer.
        Args:
            frame: Input frame [B, C, H, W].
            prev_embedding: The POOLED CLS token embedding from the previous step [B, self.embed_dim].
        Returns:
            torch.Tensor: The updated POOLED CLS-like embedding [B, self.embed_dim].
        """
        if not (len(frame.shape) == 4):
            raise ValueError(f"forward_update expects frame shape [B, C, H, W], but got {frame.shape}")

        if prev_embedding is not None:
            if not (len(prev_embedding.shape) == 2 and prev_embedding.shape[1] == self.embed_dim):
                raise ValueError(f"forward_update expects prev_embedding shape [B, embed_dim ({self.embed_dim})] or None, but got {prev_embedding.shape}")
            prev_embedding = prev_embedding.to(frame.device) # Ensure device match
        else:
            raise ValueError("prev_embedding cannot be None after the initial step.")

        frame_5d = frame.unsqueeze(2)  # Shape [B, C, 1, H, W]

        # Patch embed the single frame.
        # self.patch_embed's tubelet_size must be 1 for this to work cleanly on a single frame.
        # InternVideo2 default tubelet_size is 1.
        if self.model_tubelet_size != 1:
            raise NotImplementedError("forward_update with single frame requires model tubelet_size=1 for patch_embed")

        x_dtype = self.patch_embed.proj.weight.dtype
        new_frame_patch_tokens = self.patch_embed(frame_5d.type(x_dtype))
        # Output: [B, 1, num_spatial_patches_one_frame, self.embed_dim]

        B_nft, T_nft, L_patch_nft, C_nft = new_frame_patch_tokens.shape # T_nft will be 1
        # Reshape to [B, L_new, self.embed_dim] where L_new = num_spatial_patches_one_frame
        new_frame_patch_tokens = new_frame_patch_tokens.view(B_nft, T_nft * L_patch_nft, self.embed_dim)

        # Add positional embedding for the new frame tokens.
        # This requires careful handling: these are spatial patches for a single time step.
        # We need a pos_embed of shape [1, L_patch_nft, self.embed_dim]

        # If sep_pos_embed: use self.pos_embed_spatial directly if L_patch_nft matches.
        # If not sep_pos_embed: use the spatial part of self.pos_embed for a single time step (e.g., t=0 or averaged).
        # This logic should mirror how use_image=True is handled for pos_embed in _extract_pre_projector_features
        # if we consider a single frame as an "image".

        num_spatial_patches_expected = self.patch_embed.grid_size[1] * self.patch_embed.grid_size[2]
        if L_patch_nft != num_spatial_patches_expected:
            raise ValueError(f"Num spatial patches from new frame ({L_patch_nft}) mismatch expected ({num_spatial_patches_expected})")

        if self.sep_pos_embed:
            frame_pos_embed = self.pos_embed_spatial # [1, num_spatial_patches, C_emb]
        else: # Combined 3D pos_embed
            # Take the pos_embed corresponding to the first time slice of spatial patches
            # Or, use the same logic as use_image=True for non-sep: temporally averaged spatial pos_embed
            frame_pos_embed = self.pos_embed[:, 1 : 1 + num_spatial_patches_expected, :].view(
                1, self.grid_size[0], num_spatial_patches_expected, self.embed_dim
            ).mean(dim=1) # [1, num_spatial_patches, C_emb]

        new_frame_patch_tokens = new_frame_patch_tokens + frame_pos_embed.to(new_frame_patch_tokens.device, non_blocking=True)

        updated_embedding = self.update_transformer(
            prev_embedding,       # [B, self.embed_dim] (pooled CLS-like)
            new_frame_patch_tokens  # [B, L_new, self.embed_dim] (sequence of new frame's spatial patches)
        )

        if not (len(updated_embedding.shape) == 2 and updated_embedding.shape[0] == frame.shape[0] and updated_embedding.shape[1] == self.embed_dim):
            raise ValueError(f"UpdateTransformer produced an output of shape {updated_embedding.shape}, but expected [B, {self.embed_dim}]")

        return updated_embedding


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
