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

# --- Start of Streaming Student Model ---

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
        # ViT-Lite expects (B, C, H, W)
        # Ensure T_chunk_for_vit matches what ViT-Lite's PatchEmbed is configured for

        if len(single_frame_input.shape) == 5:
            single_frame_input = single_frame_input.squeeze(2) # Remove the T_chunk dimension

        frame_feature, _ = self.vit_lite.extract_features(single_frame_input) # (B, student_embed_dim)

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
