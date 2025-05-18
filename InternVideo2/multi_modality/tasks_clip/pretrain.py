import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import os
from torch.nn import CosineEmbeddingLoss
import torch.backends.cudnn as cudnn
import pickle
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate
from huggingface_hub import hf_hub_download
from torchvision import transforms
from tqdm import tqdm
import wandb
from torch.utils.data import ConcatDataset

import copy
from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader_rs, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from tasks_clip.retrieval_utils import evaluation_wrapper
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)

def save_debug_step_data(output_dir, global_step, frame_idx,
                         new_frame_input, # Input to streaming_vision_encoder
                         current_hidden_state_input, # Hidden state input to streaming_vision_encoder
                         actual_window_input, # Input to vision_encoder (full model)
                         stream_embedding_output, # Output of streaming pipeline
                         target_embedding_output, # Output of target pipeline
                         model_state_dict,
                         config=None): # Optional: save config for completeness
    """
    Saves all relevant tensors and model state for a single debug step.
    """
    step_dir = os.path.join(output_dir, f"debug_step_{global_step}_frame_{frame_idx}")
    os.makedirs(step_dir, exist_ok=True)

    # Save tensors
    torch.save(new_frame_input.cpu(), os.path.join(step_dir, "new_frame_input.pt"))
    torch.save(actual_window_input.cpu(), os.path.join(step_dir, "actual_window_input.pt"))
    torch.save(stream_embedding_output.cpu(), os.path.join(step_dir, "stream_embedding_output.pt"))
    torch.save(target_embedding_output.cpu(), os.path.join(step_dir, "target_embedding_output.pt"))

    # Save hidden state (can be a tuple of tensors)
    # Ensure hidden state tensors are also moved to CPU before saving if they are on GPU
    if isinstance(current_hidden_state_input, tuple):
        cpu_hidden_state = tuple(h.cpu() for h in current_hidden_state_input)
    elif isinstance(current_hidden_state_input, torch.Tensor):
        cpu_hidden_state = current_hidden_state_input.cpu()
    else:
        cpu_hidden_state = current_hidden_state_input # Or raise error if unexpected type

    with open(os.path.join(step_dir, "current_hidden_state_input.pkl"), "wb") as f:
        pickle.dump(cpu_hidden_state, f)

    # Save model state dict
    torch.save(model_state_dict, os.path.join(step_dir, "model_state_dict.pth"))

    # Save config (optional)
    if config:
        with open(os.path.join(step_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

    print(f"Saved debug data for global_step {global_step}, frame_idx {frame_idx} to {step_dir}")


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.transforms.functional import InterpolationMode

# --- Helper to read frames from video ---
def _frame_from_video(video_cap):
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        # frame is numpy array (H, W, C), BGR format
        yield frame # Yields BGR numpy array

# --- Preprocessing function (Needs to match the model's transform) ---
# You should get this transform directly from your model instance if possible
# e.g., intern_model.transform or intern_model.inference_transform
def get_inference_transform(img_size):
     return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),  # Converts PIL Image [0,255] to [C,H,W] tensor [0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

# --- Preprocess single frame for model input ---
def preprocess_frame(frame_bgr_np, transform, device):
    """
    Preprocesses a single frame (BGR numpy array) for model inference.
    Output: [1, C, H, W] tensor on specified device, normalized [0, 1]
    """
    frame_rgb_np = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb_np)

    # Apply the transform pipeline
    # Output shape: [C, H, W]
    transformed_tensor_chw = transform(pil_image)

    # Add batch dimension and move to device
    frame_tensor_batch = transformed_tensor_chw.unsqueeze(0).to(device) # [1, C, H, W]

    return frame_tensor_batch

# --- Evaluation Function ---
def evaluate_streaming_similarity(
    model,
    device,
    transform, # The preprocessing transform
    video_path,
    model_max_frames,
    output_dir,
    global_step, # Current training step for filename
    config
):
    """
    Evaluates the cosine similarity between streaming and full window features
    for a specific video and saves a plot.

    Returns the average cosine similarity over the comparable frames.
    """
    # Ensure model is in evaluation mode and on the correct device
    model.eval()
    model.to(device) # Ensure model is on device, though it should be already

    cosine_similarities = []
    frame_indices_for_plot = []
    avg_similarity = -1.0 # Default value if no frames processed

    logger.info(f"Starting evaluation on video: {video_path}")

    # 1) Read all frames for the current video
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        logger.error(f"Error: Could not open video {video_path} for evaluation.")
        return avg_similarity # Return default if video can't be opened

    all_frames_raw = list(_frame_from_video(video_cap)) # List of numpy arrays (H, W, C, BGR)
    video_cap.release()

    if len(all_frames_raw) < model_max_frames:
        logger.warning(f"Evaluation video {video_path} has {len(all_frames_raw)} frames, less than MODEL_MAX_FRAMES ({model_max_frames}). Skipping evaluation.")
        return avg_similarity # Return default if video is too short

    # Use torch.no_grad() for inference
    with torch.no_grad():
        # 2) Initialize streaming model's hidden state for this evaluation run
        # Batch size is 1 for single video inference
        curr_hidden_state_streaming = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

        # 3) Warm-up streaming model with the first (MODEL_MAX_FRAMES - 1) frames
        # Process frames from index 0 up to MODEL_MAX_FRAMES - 2
        logger.info(f"Warming up streaming model for evaluation with first {model_max_frames - 1} frames...")
        for i in range(model_max_frames - 1):
            frame_data = all_frames_raw[i] # Get BGR numpy array

            # Preprocess single frame -> [1, C, H, W] tensor on device
            frame_tensor_batch = preprocess_frame(frame_data, transform, device) # [1, C, H, W]

            # Add temporal dimension (T=1) for streaming encoder input [B, C, T=1, H, W]
            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2) # [1, C, 1, H, W]

            # Pass the frame and the previous hidden state to the streaming encoder
            raw_stream_embedding_dummy, curr_hidden_state_streaming = model.streaming_vision_encoder(
                frame_tensor_streaming_input, # Input is [1, C, 1, H, W]
                curr_hidden_state_streaming
            )
        logger.info(f"Warm-up complete for evaluation.")


        # 4) Slide through the rest of the video, frame by frame
        #    Loop from the index of the frame that completes the first window (MODEL_MAX_FRAMES - 1)
        #    Up to the last frame of the video
        logger.info(f"Processing and comparing from frame {model_max_frames - 1} onwards...")
        # No tqdm here to avoid interfering with training progress bar
        for frame_idx in range(model_max_frames - 1, len(all_frames_raw)):
            # --- Streaming Model Feature (using the *current* frame and state) ---
            current_frame_data_streaming = all_frames_raw[frame_idx] # BGR numpy array

            # Preprocess the *current* frame for the streaming encoder
            frame_tensor_batch = preprocess_frame(current_frame_data_streaming, transform, device) # [1, C, H, W]

            # Add temporal dimension (T=1) for streaming encoder input [B, C, T=1, H, W]
            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2) # [1, C, 1, H, W]

            # Pass the current frame and the previous hidden state to the streaming encoder
            raw_stream_embedding, new_hidden_state = model.streaming_vision_encoder(
                frame_tensor_streaming_input, # Input is [1, C, 1, H, W]
                curr_hidden_state_streaming
            )

            # Align and Normalize the raw streaming embedding
            if config.model.use_streaming_vision_align:
                aligned_stream_embedding = model.streaming_vision_align(raw_stream_embedding)
            else:
                aligned_stream_embedding = model.vision_align(raw_stream_embedding)
            stream_embedding = aligned_stream_embedding / (aligned_stream_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            # Update the hidden state for the next frame
            curr_hidden_state_streaming = new_hidden_state

            # --- Full Model Feature for the corresponding window ---
            # The window of MODEL_MAX_FRAMES frames ends at the current frame_idx
            window_start_idx = frame_idx - model_max_frames + 1
            window_end_idx = frame_idx + 1 # Slicing is exclusive at the end
            current_window_frames_data = all_frames_raw[window_start_idx : window_end_idx] # List of BGR numpy arrays

            # Preprocess all frames in the window and stack them
            # List of [1, C, H, W] tensors -> Stack -> [MODEL_MAX_FRAMES, 1, C, H, W]
            list_of_frame_tensors = [preprocess_frame(f, transform, device) for f in current_window_frames_data]
            stacked_window_tensor_T_B_C_H_W = torch.stack(list_of_frame_tensors, dim=0) # Shape: [T, B=1, C, H, W]

            # Reshape for the full vision encoder [B, C, T, H, W]
            window_tensor_full = stacked_window_tensor_T_B_C_H_W.unsqueeze(0).squeeze(2).permute(0, 2, 1, 3, 4) # Shape: [1, C, MODEL_MAX_FRAMES, H, W]

            # Pass the full window tensor to the full vision encoder
            raw_target_embedding = model.vision_encoder(window_tensor_full)

            # Align and Normalize the raw target embedding
            aligned_target_embedding = model.vision_align(raw_target_embedding)
            target_embedding = aligned_target_embedding / (aligned_target_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            # --- Cosine Similarity ---
            similarity = torch.nn.functional.cosine_similarity(stream_embedding, target_embedding, dim=1)

            sim_value = similarity.item()
            cosine_similarities.append(sim_value)
            frame_indices_for_plot.append(frame_idx) # Store the actual frame index

        # --- Evaluation Complete ---
        if cosine_similarities:
            avg_similarity = sum(cosine_similarities) / len(cosine_similarities)
            logger.info(f"Evaluation complete. Average Cosine Similarity: {avg_similarity:.4f}")

            # --- Plotting and Saving ---
            plt.figure(figsize=(12, 6))
            plt.plot(frame_indices_for_plot, cosine_similarities, 'g-', label='Cosine Similarity (Streaming vs Full Window)')
            plt.xlabel(f'Frame Number (Window of {model_max_frames} frames ending at this frame)')
            plt.ylabel('Cosine Similarity')
            plt.title(f'Feature Similarity Over Time - Video: {os.path.basename(video_path)}\nTraining Step: {global_step}')
            plt.legend()
            plt.grid(True)
            plt.ylim(-0.1, 1.1) # Cosine similarity range
            plt.axhline(y=avg_similarity, color='b', linestyle='--', label=f'Average: {avg_similarity:.4f}')
            plt.legend()

            # Define save path
            graph_save_dir = join(output_dir, 'cosine_sim_graphs')
            os.makedirs(graph_save_dir, exist_ok=True)
            graph_filename = f'graph_step_{global_step:07d}.png' # Use padded step number
            graph_save_path = join(graph_save_dir, graph_filename)

            plt.savefig(graph_save_path)
            logger.info(f"Saved evaluation plot to {graph_save_path}")

            # Close the plot figure to free memory
            plt.close('all')
        else:
            logger.warning("No cosine similarities were calculated during evaluation.")


    # Set model back to training mode
    model.train()
    logger.info("Evaluation complete. Model set back to train() mode.")

    return avg_similarity


# Assuming the imports and helper functions (_frame_from_video, get_inference_transform,
# preprocess_frame, create_dummy_video, evaluate_streaming_similarity,
# MetricLogger, SmoothedValue, MetaLoader_rs, get_media_types, CosineEmbeddingLoss,
# is_main_process, log_dict_to_wandb, save_debug_step_data, logger) are defined above.
# Also assuming InternVideo2_CLIP_Small class and its methods are available.

# --- The main training function ---
def train(
    model,
    train_loaders, # List of DataLoaders
    mobileclip_train_loaders, # List of DataLoaders for MobileCLIP data
    optimizer,
    tokenizer, # Placeholder, not used in this specific loss
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type, # e.g., torch.float16 if use_half_precision else torch.float32
    skip_num=0,
    log_debug=False,
):
    """
    Performs one epoch of training with periodic evaluation.
    """
    model_without_ddp = model.module if config.distributed else model
    model.train()

    logger.info('-'*20)

    # Sanity check for which params are unfrozen
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            logger.info(f"Unfrozen Parameter: {name}")

    logger.info('-'*20)

    try:
        inference_transform = model_without_ddp.transform
        IMG_SIZE = model_without_ddp.config.model.vision_encoder.img_size
    except AttributeError:
        logger.warning("Model does not have 'transform' or 'config.model.vision_encoder.img_size'. Using default.")
        IMG_SIZE = config.get('size_t', 224)
        inference_transform = get_inference_transform(IMG_SIZE)

    mobileclip_transform = transforms.Compose(
        [
            transforms.Resize(
                config.inputs.image_res, # Use the provided image_size
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(config.inputs.image_res), # Use the provided image_size
            transforms.ToTensor(),
        ]
    )

    EVAL_FREQ_STEPS = config.eval_freq_steps
    logger.info(f"Getting evaluation video from {config.eval_video_repo_id} ({config.eval_video_filename})")
    EVAL_VIDEO_PATH = hf_hub_download(repo_id=config.eval_video_repo_id, filename=config.eval_video_filename, repo_type="dataset")
    EVAL_PLOT_OUTPUT_DIR = config.eval_plot_output_dir
    os.makedirs(EVAL_PLOT_OUTPUT_DIR, exist_ok=True)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("eval_avg_sim", SmoothedValue(window=1, fmt="{value:.4f}")) # For periodic eval

    active_loss_names = ["loss_cosine"]
    additional_logs = ["cosine_similarity"] # For in-batch similarity

    # Metrics for original data
    media_types = get_media_types(train_loaders)
    for loss_name in active_loss_names:
        for media_type_key in media_types:
            metric_logger.add_meter(f"{media_type_key}-{loss_name}", SmoothedValue(window=1, fmt="{value:.4f}"))
    for loss_name in additional_logs:
        for media_type_key in media_types:
            metric_logger.add_meter(f"{media_type_key}-{loss_name}", SmoothedValue(window=1, fmt="{value:.4f}"))

    # Metrics for MobileCLIP data
    # Assuming mobileclip_train_loaders also yields a "media_type" or we can define one
    # For simplicity, let's assume a single "mobileclip" media type from its loader
    mc_media_types = ["mobileclip"] # Or derive using get_media_types(mobileclip_train_loaders)
    for loss_name in active_loss_names:
        for media_type_key in mc_media_types: # Use a distinct key
            metric_logger.add_meter(f"{media_type_key}-{loss_name}", SmoothedValue(window=1, fmt="{value:.4f}"))
    for loss_name in additional_logs:
        for media_type_key in mc_media_types:
            metric_logger.add_meter(f"{media_type_key}-{loss_name}", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for loader in train_loaders: loader.sampler.set_epoch(epoch)
        for loader in mobileclip_train_loaders: loader.sampler.set_epoch(epoch)

    # Aggregate loaders
    train_loader_agg = MetaLoader_rs(name2loader=dict(list(zip(media_types, train_loaders))), skip_num=skip_num)
    mobileclip_loader_agg = MetaLoader_rs(name2loader=dict(list(zip(mc_media_types, mobileclip_train_loaders))), skip_num=skip_num)

    # Determine the number of batches for the tqdm progress bar
    # Your logs indicate they have the same length (6202 batches each).
    num_batches_train = len(train_loader_agg)
    num_batches_mc = len(mobileclip_loader_agg)

    num_batches_to_iterate = min(num_batches_train, num_batches_mc)
    if num_batches_train != num_batches_mc:
        logger.warning(
            f"Train loaders have {num_batches_train} batches, MobileCLIP loaders have {num_batches_mc} batches. "
            f"Iterating for {num_batches_to_iterate} batches (the minimum)."
        )
    # If they are guaranteed to be the same, you can just use num_batches_train

    # Create an iterator by zipping the two MetaLoaders. DO NOT convert to list().
    combined_iterable = zip(train_loader_agg, mobileclip_loader_agg)

    # tqdm can take an iterable directly and an optional 'total' for the progress bar display.
    progress_bar = tqdm(
        combined_iterable,
        total=num_batches_to_iterate,
        desc=header,
        disable=not is_main_process()
    )

    MODEL_MAX_FRAMES = config.num_frames
    cosine_loss_base_fn = CosineEmbeddingLoss()

    def cosine_sim_loss(student_embedding, teacher_embedding):
        B = student_embedding.shape[0]
        target = torch.ones(B, dtype=student_embedding.dtype, device=student_embedding.device)
        return cosine_loss_base_fn(student_embedding, teacher_embedding, target)

    # Iterate over the progress_bar, which yields items from combined_iterable
    # The 'i' from enumerate is still useful for log_freq checks if needed.
    for i, data_pair in enumerate(progress_bar):
        # Unpack the data pair from the zipped loaders
        (media_type, (image, text, idx)), (mc_media_type, (mc_image, mc_text, mc_idx)) = data_pair

        # Move input data to the designated compute device
        image = image.to(device, non_blocking=True)
        # idx = idx.to(device, non_blocking=True) # Not used here
        mc_image = mc_image.to(device, non_blocking=True)
        # mc_idx = mc_idx.to(device, non_blocking=True) # Not used here

        if log_debug and i == 0 : # Log first batch
            logger.info(f"Original data: image shape: {image.shape}, text: {text}")
            logger.info(f"MobileCLIP data: mc_image shape: {mc_image.shape}, mc_text: {mc_text}")

        # Tokenizing text is skipped as it's not used in this video-only loss example

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            # Process Original Data
            # > image is shape [B, T, C, H, W], permute to [B, C, T, H, W]
            image = image.permute(0, 2, 1, 3, 4)
            B_orig, C_orig, T_orig, H_orig, W_orig = image.shape
            assert T_orig >= MODEL_MAX_FRAMES, f"Video (orig) has {T_orig} frames, needs {MODEL_MAX_FRAMES}."

            # Process MobileCLIP Data
            # > mc_image is shape [B, T, C, H, W], permute to [B, C, T, H, W]
            mc_image = mc_image.permute(0, 2, 1, 3, 4)
            B_mc, C_mc, T_mc, H_mc, W_mc = mc_image.shape
            assert T_mc >= MODEL_MAX_FRAMES, f"Video (MC) has {T_mc} frames, needs {MODEL_MAX_FRAMES}."

            # Warm up and accumulate the hidden state on the first `MODEL_MAX_FRAMES - 1` frames.
            curr_hidden_state = model.streaming_vision_encoder.init_hidden(batch_size=B_mc, device=device)
            with torch.no_grad():
                for frame_idx in range(MODEL_MAX_FRAMES - 1):
                    initial_frame_mc = mc_image[:, :, frame_idx, :, :].unsqueeze(2)
                    _, curr_hidden_state = model.streaming_vision_encoder(initial_frame_mc, curr_hidden_state)

            # Determine the number of sliding window steps
            # This should be the same for both if T_orig and T_mc are the same.
            # If T can differ, loop min(T_orig, T_mc) - MODEL_MAX_FRAMES + 1 times.
            # For simplicity, assuming T_orig and T_mc are equal due to dataset prep.
            num_sliding_windows = T_orig - (MODEL_MAX_FRAMES - 1)
            assert num_sliding_windows == T_mc - (MODEL_MAX_FRAMES - 1), "Video lengths mismatch between original and mc data streams!"

            total_loss_accumulator = 0.0 # Accumulate loss over frames in a batch

            # Iterate over frames for loss calculation
            # The loop goes from the first frame that completes a window up to the last frame.
            combined_cosine_similarity_avg = None

            for frame_window_step_idx in range(num_sliding_windows):
                # current_frame_in_video_idx is the index of the frame being fed to the streaming encoder
                # This frame is the *last* frame of the current window.
                current_frame_in_video_idx = (MODEL_MAX_FRAMES - 1) + frame_window_step_idx

                # --- Stream Embedding Calculation (using mc_image and streaming encoder) ---
                current_streaming_frame_mc = mc_image[:, :, current_frame_in_video_idx, :, :].unsqueeze(2)
                # Pass mc_image frame and its hidden state to the streaming encoder
                raw_stream_emb_mc, new_hidden_state_mc_updated = model.streaming_vision_encoder(
                    current_streaming_frame_mc, curr_hidden_state # Use mc data and hidden state
                )
                # Align the streaming output (shared alignment layer)
                if config.model.use_streaming_vision_align:
                    aligned_stream_emb_mc = model_without_ddp.streaming_vision_align(raw_stream_emb_mc)
                else:
                    aligned_stream_emb_mc = model_without_ddp.vision_align(raw_stream_emb_mc)
                stream_embedding = aligned_stream_emb_mc / (aligned_stream_emb_mc.norm(dim=-1, keepdim=True) + 1e-9)

                # --- Target Embedding Calculation (using original image and full encoder) ---
                window_start_idx = current_frame_in_video_idx - MODEL_MAX_FRAMES + 1
                window_end_idx = current_frame_in_video_idx + 1
                # Get the full window from the ORIGINAL image data
                current_window_frames_orig = image[:, :, window_start_idx:window_end_idx, :, :]

                with torch.no_grad(): # Target computation should not backprop
                    # Pass original image window to the standard vision encoder
                    raw_target_emb_orig = model_without_ddp.vision_encoder(current_window_frames_orig)
                    # Align the target output (shared alignment layer)
                    aligned_target_emb_orig = model_without_ddp.vision_align(raw_target_emb_orig)
                    target_embedding = aligned_target_emb_orig / (aligned_target_emb_orig.norm(dim=-1, keepdim=True) + 1e-9)

                # --- Loss Calculation (Streaming MC embedding vs. Target Original embedding) ---
                # Calculate the loss between the streaming embedding from mc_image
                # and the target embedding from image
                loss = cosine_sim_loss(stream_embedding, target_embedding)

                # Calculate cosine similarity for logging (between the embeddings used for loss)
                if not combined_cosine_similarity_avg:
                    combined_cosine_similarity_avg = torch.nn.functional.cosine_similarity(
                        stream_embedding.detach(), target_embedding.detach(), dim=1
                    ).mean().item()
                else:
                    combined_cosine_similarity_avg += torch.nn.functional.cosine_similarity(
                        stream_embedding.detach(), target_embedding.detach(), dim=1
                    ).mean().item()

                # --- Accumulate Loss ---
                # Accumulate the single loss value for this frame step
                total_loss_accumulator += loss

                # --- Update hidden states for the next frame in the sliding window ---
                # ONLY update the hidden state for the streaming encoder (which processed mc_image)
                curr_hidden_state = tuple(h.detach().clone() for h in new_hidden_state_mc_updated)


                # --- Logging for this frame step (optional, can be noisy) ---
                # For simplicity, we'll log aggregated metrics after all frames in a batch are processed
                # Or, if you log per frame, remember that global_step also increments per frame.

                # --- Optional Debug Saving for the first frame of the first batch ---
                # Adjusted debug saving to reflect the inputs used for the single loss
                if log_debug and i == 0 and frame_window_step_idx == 0 :
                     logger.info(f"Saving debug data at global step {global_step}, frame index {current_frame_in_video_idx}")
                     logger.info(f"Saving to {config.output_dir}")
                     save_debug_step_data(
                        output_dir=config.output_dir, global_step=global_step, frame_idx=current_frame_in_video_idx,
                        new_frame_input=current_streaming_frame_mc[0].cpu(), # Input to streaming encoder
                        current_hidden_state_input=tuple(h[0].detach().cpu() for h in curr_hidden_state), # Hidden state input
                        actual_window_input=current_window_frames_orig[0].cpu(), # Input to target encoder
                        stream_embedding_output=stream_embedding[0].cpu(), # Output of streaming pipeline
                        target_embedding_output=target_embedding[0].cpu(), # Output of target pipeline
                        model_state_dict=model_without_ddp.state_dict()
                    )
            # Average loss over the frames in the batch item
            # This is important: backward pass should be on the average or sum of losses
            # from all sliding windows in the batch.
            final_batch_loss = total_loss_accumulator / num_sliding_windows
            average_cosine_sim = combined_cosine_similarity_avg / num_sliding_windows

        # --- Backpropagation and Optimization (after processing all frames in a batch item) ---
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            model.backward(final_batch_loss)
            model.step()
        else:
            optimizer.zero_grad()
            if config.use_half_precision:
                scaler.scale(final_batch_loss).backward()
                if config.optimizer.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                final_batch_loss.backward()
                if config.optimizer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()

            scheduler.step() # Assuming step-based scheduler

        # --- Logging Metrics (after optimizer step for the batch) ---
        # Log the per-batch averaged losses and similarities
        # Using a clear metric key name, e.g., 'video-stream-target-loss'
        # Note: The logged loss/sim below is from the *last* frame step iteration.
        # If you want the average over frames, calculate it before the optimizer step or log total_loss_accumulator / num_sliding_windows
        # Let's log the average loss for the batch item after the loop.
        metric_logger.update(**{'video-stream-target-loss': final_batch_loss.item()})
        # Log the similarity from the last frame step (similarly, could average over frames if desired)
        metric_logger.update(**{'video-stream-target-sim': average_cosine_sim})


        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if hasattr(model_without_ddp, 'temp'):
            metric_logger.update(temperature=model_without_ddp.temp.item())

        # Increment global_step once per batch item (outer loop iteration)
        global_step += 1

        # --- Periodic Evaluation ---
        if global_step % EVAL_FREQ_STEPS == 0 and is_main_process():
            logger.info(f"Performing periodic evaluation at global step {global_step}...")
            # Assuming evaluate_streaming_similarity uses the standard streaming path (input -> stream_encoder -> stream_align)
            avg_sim = evaluate_streaming_similarity(
                model=model_without_ddp, device=device, transform=mobileclip_transform,
                video_path=EVAL_VIDEO_PATH, model_max_frames=MODEL_MAX_FRAMES,
                output_dir=config.output_dir, global_step=global_step,
                config = config
            )
            metric_logger.update(eval_avg_sim=avg_sim)
            logger.info(f"Evaluation at step {global_step} complete. Avg Sim: {avg_sim:.4f}")
            model.train() # Ensure model is back in training mode

        # --- Log to console and W&B ---
        if i % log_freq == 0: # Log on log_freq and last iteration
            log_payload = {
                "lr": optimizer.param_groups[0]["lr"],
                "video_stream_target_loss": final_batch_loss.item(), # Log the batch averaged loss
                "video_stream_target_sim": average_cosine_sim # Log last frame's sim
            }
            if hasattr(model_without_ddp, 'temp'): log_payload["temp"] = model_without_ddp.temp.item()
            if global_step > 0 and global_step % EVAL_FREQ_STEPS == 0 and is_main_process():
                log_payload["eval_sim"] = metric_logger.meters['eval_avg_sim'].value

            progress_bar.set_postfix(log_payload)

            # Actual logging via logger
            if is_main_process(): # Avoid DDP spam
                logger.info(f"{header} [{i}] {metric_logger}")

            if is_main_process() and config.wandb.enable: # Log to W&B
                averaged_logs = metric_logger.get_global_avg_dict() # These are windowed averages
                log_dict_to_wandb(averaged_logs, step=global_step, prefix="train/")

        # --- Iteration-based Checkpointing ---
        if config.get('save_iter', 0) > 0 and global_step % config.save_iter == 0:
            # ... (checkpointing logic remains largely the same, ensure it's called correctly)
            logger.info(f"Saving checkpoint at global step {global_step}")
            # ... (your existing checkpoint saving code) ...
            if is_main_process() and not (hasattr(config, "deepspeed") and config.deepspeed.enable):
                save_obj = {
                    "model": model_without_ddp.state_dict(), # Save unwrapped model
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if config.use_half_precision else None,
                    "config": config, "epoch": epoch, "global_step": global_step,
                }
                checkpoint_filename = join(config.output_dir, f"ckpt_iter{global_step:07d}.pth")
                torch.save(save_obj, checkpoint_filename)
                logger.info(f"Saved iteration checkpoint to {checkpoint_filename}")


        # --- Debugging Hooks ---
        if config.debug and global_step >= 20: # Adjusted for batch-level global_step
            logger.info("Debug mode: breaking training loop early.")
            break # Break outer batch loop


    # --- Training Loop End ---
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
    if is_main_process() and config.wandb.enable:
        # Need to update metric logger setup at the start of the function
        # to include the new keys ('video-stream-target-loss', 'video-stream-target-sim')
        # and remove the old ones ('orig-loss_cosine', 'orig-cosine_similarity', etc.)
        log_dict_to_wandb(metric_logger.get_global_avg_dict(), step=global_step, prefix=f"epoch_{epoch}/")
        log_dict_to_wandb(metric_logger.get_global_avg_dict(), step=global_step, prefix="train/")

    return global_step


def clone_collate_fn(batch):
    # Recursively clone every Tensor in the sample so its storage is fresh
    def clone_item(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        elif isinstance(x, (list, tuple)):
            return type(x)(clone_item(y) for y in x)
        elif isinstance(x, dict):
            return {k: clone_item(v) for k, v in x.items()}
        else:
            return x

    batch = [clone_item(sample) for sample in batch]
    return default_collate(batch)

def setup_dataloaders(config, mode="pt"):
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    mobileclip_train_datasets = create_dataset(f"{mode}_train", config)
    media_types   = get_media_types(train_datasets)

    if not config.distributed:
        raise NotImplementedError

    # one GPU-batch size per media type
    batch_size = [config.inputs.batch_size[k] for k in media_types]
    samplers   = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size   = batch_size,
        num_workers  = [config.num_workers] * len(media_types),
        is_trains    = [True] * len(media_types),
        collate_fns  = [clone_collate_fn] * len(media_types),   # ← here!
    )

    mobileclip_train_loaders = create_loader(
        mobileclip_train_datasets,
        samplers,
        batch_size   = batch_size,
        num_workers  = [config.num_workers] * len(media_types),
        is_trains    = [True] * len(media_types),
        collate_fns  = [clone_collate_fn] * len(media_types),   # ← here!
    )

    # eval side stays the same (you probably don’t hit this bug there)
    test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size   = [config.inputs.batch_size_test[d.media_type] for d in test_datasets],
        num_workers  = [config.num_workers] * len(test_datasets),
        is_trains    = [False] * len(test_datasets),
        collate_fns  = [None]   * len(test_datasets),
    )

    test_name2loaders = dict(zip(test_dataset_names, test_loaders))
    return train_loaders, test_name2loaders, media_types, mobileclip_train_loaders


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types, mobileclip_train_loaders = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders) * 247 # Using each individual frame for training

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1

    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=is_pretrain,
        find_unused_parameters=True,
        num_steps_per_epoch=num_steps_per_epoch,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    best = 0
    best_epoch = 0

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start training")
    logger.info(f"Epoch: {start_epoch}")
    start_time = time.time()
    start_step = start_epoch * num_steps_per_epoch
    for epoch in range(start_epoch, config.scheduler.epochs):
        if not config.evaluate:
            global_step = train(
                model,
                train_loaders,
                mobileclip_train_loaders,
                optimizer,
                tokenizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                data_type,
                skip_num = global_step - start_step,
                log_debug = True
            )

        # save checkpoint befor evaluation
        # only save those with gradient
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            if config.get("save_latest", False):
                tag = "ckpt_latest.pth"
            else:
                tag = f"ckpt_{epoch:02d}.pth"
            model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)

        elif is_main_process():
            state_dict = model_without_ddp.state_dict()
            param_grad_dict = {
                k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
            }
            for k in list(state_dict.keys()):
                if k in param_grad_dict.keys() and not param_grad_dict[k]:
                    # delete parameters that do not require gradient
                    logger.info(f"Not saving {k}")
                    del state_dict[k]

            save_obj = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
            }
            if config.get("save_latest", False):
                torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            else:
                torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        # evaluation
        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            eval_res = {}
            for test_name, test_loader in test_name2loaders.items():
                if test_name not in config.test_types:
                    logger.info(
                        f"Skip eval {test_name} split. All test_types {config.test_types}"
                    )
                    continue
                res = evaluation_wrapper(
                    model_without_ddp, test_loader, tokenizer, device, config, data_type=data_type, prefix=test_name
                )
                eval_res.update(res)

        # save the best checkpoint
        if is_main_process():
            # log to wandb
            if config.wandb.enable:
                for p, v in eval_res.items():
                    log_dict_to_wandb(v, step=global_step, prefix=p)

            if config.stop_key is not None and config.stop_key in eval_res:
                cur_r_mean = eval_res[config.stop_key]["r_mean"]
            else:  # None
                cur_r_mean = best + 1  # save the last as the best

            eval_res = pd.DataFrame(eval_res)
            logger.info(f"Epoch {epoch}")
            logger.info(f"\n{eval_res.transpose().to_string(max_cols=30)}")

            eval_res.to_json(join(config.output_dir, "eval_res_latest.json"))

            if not config.evaluate and cur_r_mean > best:
                if not hasattr(config, "deepspeed") or not config.deepspeed.enable:
                    torch.save(save_obj, join(config.output_dir, "ckpt_best.pth"))
                eval_file = "eval_res_best.json"
                eval_res.to_json(join(config.output_dir, eval_file))
                best = cur_r_mean
                best_epoch = epoch

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            r_mean_best = torch.tensor([0.0, 0.0]).to(device)
            if is_main_process():
                r_mean_best[0] = cur_r_mean
                r_mean_best[1] = best
            dist.broadcast(r_mean_best, 0)
            cur_r_mean, best = r_mean_best[0].item(), r_mean_best[1].item()

            if not config.evaluate and cur_r_mean > best:
                model.save_checkpoint(config.output_dir, tag="ckpt_best.pth", save_latest=False, exclude_frozen_parameters=True)

        if config.evaluate:
            break

        start_step = global_step

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
