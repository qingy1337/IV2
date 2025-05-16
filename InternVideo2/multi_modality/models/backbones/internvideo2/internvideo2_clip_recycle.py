# --- Reused/Adapted Components from InternVideo2 ---
from .internvideo2_clip_vision import CrossAttention, AttentiveBlock, AttentionPoolingBlock, RMSNorm, LayerScale, Attention, Mlp, Block, PatchEmbed

from .mobileclip import TextTransformer, ClipTokenizer, VisionTransformer, vit_b16

import logging
import math
import torch
import timm
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

# --- Start of ViT-Lite and Streaming Student Model ---

class ViTLiteStudent(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=14,
            in_chans=3,
            student_embed_dim=768,  # Smaller embed_dim for student
            student_depth=4,        # Fewer layers for student
            student_num_heads=12,   # Can be different
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.1,
            init_values=1e-5,
            qk_normalization=False, # Simplified from original
            norm_layer_for_blocks=partial(nn.LayerNorm, eps=1e-6), # Using nn.LayerNorm for simplicity
            sep_pos_embed=False,    # Matches InternVideo2's option
            # Student-specific parameters for PatchEmbed
            student_num_frames=1,   # ViT-Lite processes one frame (or small chunk)
            student_tubelet_size=1,
            layerscale_no_force_fp32=False, # From InternVideo2
    ):
        super().__init__()
        self.student_embed_dim = student_embed_dim
        self.sep_pos_embed = sep_pos_embed
        self.student_num_frames = student_num_frames # T for ViT-Lite

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=student_embed_dim,
            num_frames=student_num_frames, # Critical: student processes few frames
            tubelet_size=student_tubelet_size,
            norm_layer=None # Norm is applied after proj in PatchEmbed
        )
        num_patches = self.patch_embed.num_patches # Spatial patches for T=1
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, student_embed_dim))

        if self.sep_pos_embed:
            # Spatial embedding for the patches of a single frame
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, num_patches, student_embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, student_embed_dim))
            # No temporal pos embed here as T_student=1 for ViT-Lite's direct input
        else:
            # Combined class token and spatial patches for a single frame
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, student_embed_dim))

        self.init_pos_embed_student() # Custom init for student

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, student_depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=student_embed_dim,
                num_heads=student_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer_for_blocks,
                drop_path=dpr[i],
                init_values=init_values,
                # Simpler Attention/MLP for student (no flash/fused by default)
                use_flash_attn=False,
                use_fused_mlp=False,
                qk_normalization=qk_normalization,
                layerscale_no_force_fp32=layerscale_no_force_fp32,
            )
            for i in range(student_depth)])

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def init_pos_embed_student(self):
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=.02)
            trunc_normal_(self.pos_embed_cls, std=.02)
            # Example using 2D sincos if desired (requires patch_embed.grid_size to be set correctly for T=1)
            # H_grid = self.patch_embed.grid_size[1] # Assuming grid_size is (T_grid, H_grid, W_grid)
            # W_grid = self.patch_embed.grid_size[2]
            # pos_embed_spatial_data = get_2d_sincos_pos_embed(self.student_embed_dim, H_grid) # Assuming square grid for simplicity
            # self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial_data).float().unsqueeze(0))
        else:
            trunc_normal_(self.pos_embed, std=.02)
            # Example: Initialize with 2D sincos for patches + CLS
            # H_grid = self.patch_embed.grid_size[1]
            # W_grid = self.patch_embed.grid_size[2]
            # pos_embed_data = get_2d_sincos_pos_embed(self.student_embed_dim, H_grid, cls_token=True)
            # self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): # Handle standard LayerNorm with bias
            if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, RMSNorm): # Handle RMSNorm (no bias)
            # RMSNorm does not have a bias attribute
            nn.init.constant_(m.weight, 1.0)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def forward(self, x_frame):
        # x_frame is expected to be (B, C, H, W) or (B, C, T_student, H, W) where T_student=1
        if x_frame.ndim == 4: # B, C, H, W
            x_frame = x_frame.unsqueeze(2) # Add T dimension: B, C, 1, H, W

        x = self.patch_embed(x_frame.type(self.dtype)) # Output: B, T_student_grid, L_spatial, C
        # For T_student_grid = 1 (since student_num_frames=1, student_tubelet_size=1 for PatchEmbed)
        # x shape: B, 1, L_spatial, C.  Need to squeeze T_student_grid.
        x = x.squeeze(1) # B, L_spatial, C

        B, L_spatial, C_dim = x.shape
        assert L_spatial == self.num_patches
        assert C_dim == self.student_embed_dim

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # B, 1+L_spatial, C

        if self.sep_pos_embed:
            pos_embed_patches = self.pos_embed_spatial.expand(B, -1, -1)
            pos_embed_final = torch.cat([self.pos_embed_cls.expand(B, -1, -1), pos_embed_patches], dim=1)
        else:
            pos_embed_final = self.pos_embed.expand(B, -1, -1)

        x = x + pos_embed_final

        for blk in self.blocks:
            x = blk(x) # Block from InternVideo2 takes single tensor input

        # We'll take the CLS token as the frame feature
        frame_feature = x[:, 0] # (B, student_embed_dim)
        return frame_feature


class StreamingInternVideo2Student(nn.Module):
    def __init__(
            self,
            # --- Parameters for the MobileCLIP ViT ---
            vit_lite_model_name="vit_b16",
            vit_lite_proj_dim=512, # Projection dimension
            vit_lite_embed_dim=768, # Output dimension
            # --- RNN parameters ---
            rnn_type='lstm', # 'lstm' or 'gru'
            rnn_hidden_size=1024,
            rnn_num_layers=1,
            rnn_dropout=0.0, # Dropout for RNN layers (if rnn_num_layers > 1)
            # Output FC layers parameters
            fc_hidden_layers=[512], # List of hidden layer sizes for FC part, empty for direct projection
            teacher_clip_embed_dim=768, # Dimension of the teacher's output
        ):
        super().__init__()

        # Create a MobileCLIP VisionTransformer class.
        self.vit_lite = timm.create_model(
            vit_lite_model_name,
            projection_dim = vit_lite_proj_dim
        )

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type.lower()

        # Note: The RNN input_size should match the output dimension of the MobileCLIP ViT
        # when it is eventually plugged in. Using student_embed_dim as assumed here.
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=vit_lite_embed_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True, # Expects (batch, seq, feature)
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=vit_lite_embed_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0
            )
        else:
            raise NotImplementedError(f"Unsupported RNN type: {rnn_type}. Choose 'lstm' / 'gru'.")

        # Fully Connected layers to project RNN output to teacher's embedding dimension
        fc_layers = []
        current_dim = rnn_hidden_size
        if fc_hidden_layers:
            for h_dim in fc_hidden_layers:
                fc_layers.append(nn.Linear(current_dim, h_dim))
                fc_layers.append(nn.ReLU()) # Or other activation
                # fc_layers.append(nn.LayerNorm(h_dim)) # Optional LayerNorm
                # fc_layers.append(nn.Dropout(0.1)) # Optional Dropout
                current_dim = h_dim
        fc_layers.append(nn.Linear(current_dim, teacher_clip_embed_dim))
        self.output_fc = nn.Sequential(*fc_layers)

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            return (h0, c0)
        return h0

    def forward(self, single_frame_input, prev_hidden_state):
        """
        Processes a single frame (or a small chunk of frames) and updates the hidden state.

        Args:
            single_frame_input (torch.Tensor): Input frame(s) for the ViT-Lite.
                Shape: (B, C, H, W) if student_num_frames_processed_by_vit=1
                Shape: (B, C, T_chunk, H, W) if student_num_frames_processed_by_vit > 1
            prev_hidden_state (tuple or torch.Tensor): Previous hidden state from the RNN.
                For LSTM: (h_prev, c_prev)
                For GRU: h_prev

        Returns:
            student_embedding (torch.Tensor): The output embedding for the current step.
                                            Shape: (B, teacher_clip_embed_dim)
            current_hidden_state (tuple or torch.Tensor): The updated RNN hidden state.
        """
        # single_frame_input shape: (B, C, T_chunk, H, W) or (B, C, H, W)
        # ViT-Lite expects (B, C, T_chunk_for_vit, H, W)
        # Ensure T_chunk_for_vit matches what ViT-Lite's PatchEmbed is configured for
        frame_feature = self.vit_lite(single_frame_input) # (B, student_embed_dim)

        # RNN expects input of shape (batch, seq_len, input_size)
        # Here, seq_len is 1 because we process one ViT-Lite output at a time
        rnn_input = frame_feature.unsqueeze(1) # (B, 1, student_embed_dim)

        rnn_output, current_hidden_state = self.rnn(rnn_input, prev_hidden_state)
        # rnn_output shape: (B, 1, rnn_hidden_size)

        # We only care about the output of the last (and only) time step
        rnn_output_last_step = rnn_output.squeeze(1) # (B, rnn_hidden_size)

        student_embedding = self.output_fc(rnn_output_last_step) # (B, teacher_clip_embed_dim)

        return student_embedding, current_hidden_state

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # --- Make sure the InternVideo2 components are defined or imported above ---
    # This is just to make the script runnable standalone for basic checks.
    # You would integrate this into your actual training/inference pipeline.

    # Configuration for the student model
    batch_size = 2
    img_size = 224 # Should match teacher's patch processing
    patch_size = 14
    teacher_output_dim = 768 # Example: output dimension of full InternVideo2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_config = {
        "in_chans": 3,
        "patch_size": patch_size,
        "img_size": img_size,
        "vit_qkv_bias": True,
        "vit_drop_path_rate": 0.05,
        "student_embed_dim": 384,   # Smaller ViT
        "student_depth": 4,         # Fewer layers
        "student_num_heads": 6,
        "vit_mlp_ratio": 3.0,
        "vit_init_values": None, # LayerScale disabled by default
        "vit_qk_normalization": False,
        "vit_sep_pos_embed": True, # Try True or False
        "vit_norm_layer_type": "rmsnorm",
        "rnn_type": 'lstm',
        "rnn_hidden_size": 512,
        "rnn_num_layers": 1,
        "fc_hidden_layers": [256],
        "teacher_clip_embed_dim": teacher_output_dim,
        "student_num_frames_processed_by_vit": 1, # Process 1 frame at a time in ViT-Lite
        "student_tubelet_size_for_vit": 1,
    }

    student_model = StreamingInternVideo2Student(**student_config).to(device)
    student_model.eval() # Or train()

    print(f"Student model created with {sum(p.numel() for p in student_model.parameters())/1e6:.2f}M parameters.")

    # Simulate streaming a few frames
    num_stream_steps = 5
    current_hidden = student_model.init_hidden(batch_size, device)

    for i in range(num_stream_steps):
        # Dummy single frame input for each step
        # For ViT-Lite processing 1 frame: (B, C, H, W)
        dummy_frame = torch.randn(batch_size, 3, img_size, img_size).to(device)

        with torch.no_grad():
            output_embedding, current_hidden = student_model(dummy_frame, current_hidden)

        print(f"Step {i+1}: Output embedding shape: {output_embedding.shape}")
        if student_config["rnn_type"] == 'lstm':
            print(f"  LSTM hidden state h shape: {current_hidden[0].shape}, c shape: {current_hidden[1].shape}")
        else:
            print(f"  GRU hidden state shape: {current_hidden.shape}")

    # To train this model, you would:
    # 1. Generate target embeddings from the full InternVideo2 for sliding windows.
    # 2. For each window, unroll the student model frame by frame.
    # 3. At each step (or at the end of the window), compare the student's output
    #    embedding with the teacher's embedding for that window.
    # 4. Calculate a loss (e.g., MSE or CosineEmbeddingLoss) and backpropagate.
