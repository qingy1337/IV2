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
    global_step # Current training step for filename
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
            aligned_stream_embedding = model.streaming_vision_align(raw_stream_embedding)
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
    train_loaders,
    optimizer,
    tokenizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type,
    skip_num=0,
    log_debug=False,
):
    """
    Performs one epoch of training with periodic evaluation.
    """
    # Set the model to training mode
    model.train()

    # Get the inference transform from the model instance
    # Assuming your model has a 'transform' attribute
    # If not, you might need to create it using get_inference_transform(config.model.vision_encoder.img_size)
    # Make sure config.model.vision_encoder.img_size is available or use a default like config.size_t
    try:
        inference_transform = model.transform # Access transform from model
        # Assuming model config is accessible and has img_size
        IMG_SIZE = model.config.model.vision_encoder.img_size
    except AttributeError:
         logger.warning("Model does not have a 'transform' or 'config.model.vision_encoder.img_size' attribute. Using default transform.")
         IMG_SIZE = config.get('size_t', 224) # Fallback to config if model attributes not found
         inference_transform = get_inference_transform(IMG_SIZE)

    # --- Configuration for Periodic Evaluation ---
    EVAL_FREQ_STEPS = config.eval_freq_steps
    # Define the specific video path for evaluation

    logger.info(f"Getting evaluation video from {config.eval_video_repo_id} ({config.eval_video_filename})")

    EVAL_VIDEO_PATH = hf_hub_download(repo_id=config.eval_video_repo_id, filename=config.eval_video_filename, repo_type="dataset")
    # Define the output directory for evaluation plots
    EVAL_PLOT_OUTPUT_DIR = config.eval_plot_output_dir

    os.makedirs(EVAL_PLOT_OUTPUT_DIR, exist_ok=True) # Ensure directory exists

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    active_loss_names = ["loss_cosine"]

    media_types = get_media_types(train_loaders)

    for loss_name in active_loss_names:
        for media_type_key in media_types:
            metric_logger.add_meter(
                f"{media_type_key}-{loss_name}", SmoothedValue(window=1, fmt="{value:.4f}")
            )

    # Add meter for evaluation similarity
    # Using window=1 here as we only log one value per evaluation step
    metric_logger.add_meter("eval_avg_sim", SmoothedValue(window=1, fmt="{value:.4f}"))

    additional_logs = ["cosine_similarity"] # This is similarity within the training batch

    for loss_name in additional_logs:
        for media_type_key in media_types:
            metric_logger.add_meter(
                f"{media_type_key}-{loss_name}", SmoothedValue(window=1, fmt="{value:.4f}")
            )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch)

    train_loader_agg = MetaLoader_rs(name2loader=dict(list(zip(media_types, train_loaders))), skip_num=skip_num)

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader_agg, log_freq, header)

    MODEL_MAX_FRAMES = config.num_frames

    # --- Loss function (Cosine embedding loss) ---
    cosine_loss_base_fn = CosineEmbeddingLoss()

    def cosine_sim_loss(student_embedding, teacher_embedding):
        B, clip_embed_dim = student_embedding.shape
        target = torch.ones(B, dtype=student_embedding.dtype, device=student_embedding.device)
        output = cosine_loss_base_fn(student_embedding, teacher_embedding, target)
        return output

    # --- Start of training loop ---
    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        # Move input data to the designated compute device
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if log_debug:
            logger.info(f"Logging data for debugging: image shape: {image.shape}, text: {text}, idx: {idx}")

        # Tokenize text data (though not used in the current video-only loss)
        # text_input = tokenizer(text).to(device) # Keep this line if text is ever used

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            # > image is shape [B, T, C, H, W]
            image = image.permute(0, 2, 1, 3, 4) # -> [B, C, T, H, W]

            B, C, T, H, W = image.shape
            assert T >= MODEL_MAX_FRAMES, f"Video has {T} frames, needs at least {MODEL_MAX_FRAMES}."

            # Initialize hidden state for this batch
            # This hidden state will be updated throughout the training loop for this batch.
            curr_hidden_state = model.streaming_vision_encoder.init_hidden(batch_size=B, device=device)

            # Warm-up streaming encoder for the first MODEL_MAX_FRAMES - 1 frames
            # These frames update the state but don't contribute to the loss calculation in this loop
            # Use torch.no_grad() for warm-up frames
            with torch.no_grad():
                 for initial_frame_idx in range(MODEL_MAX_FRAMES - 1):
                    # Select the frame across the batch dimension
                    initial_frame = image[:, :, initial_frame_idx, :, :] # [B, C, H, W]
                    # Add T=1 dimension for streaming encoder
                    initial_frame_input = initial_frame.unsqueeze(2) # [B, C, 1, H, W]

                    # Pass the frame and update the hidden state
                    _, new_hidden_state = model.streaming_vision_encoder(
                        initial_frame_input,
                        curr_hidden_state
                    )
                    # Update hidden state for the next step in the warm-up
                    curr_hidden_state = new_hidden_state


            # Now, iterate over each of the remaining frames (from MODEL_MAX_FRAMES - 1 onwards)
            # Each iteration corresponds to a window ending at new_frame_idx,
            # and the streaming encoder processing the frame at new_frame_idx
            # The loop starts from the frame index that completes the first full window (MODEL_MAX_FRAMES - 1)
            # up to the last frame index (T - 1).
            for new_frame_idx in range(MODEL_MAX_FRAMES - 1, T):
                # The current frame being processed by the streaming encoder is at index new_frame_idx
                current_streaming_frame = image[:, :, new_frame_idx, :, :] # [B, C, H, W]
                # Add T=1 dimension for streaming encoder input
                current_streaming_frame_input = current_streaming_frame.unsqueeze(2) # [B, C, 1, H, W]

                # --- Calculate Stream Embedding for the current_streaming_frame ---
                # Pass the current frame and the previous hidden state
                raw_stream_embedding, new_hidden_state = model.streaming_vision_encoder(
                    current_streaming_frame_input, # Input is [B, C, 1, H, W]
                    curr_hidden_state # Pass the state from the *previous* frame (or warm-up)
                )

                # Align and Normalize
                aligned_stream_embedding = model_without_ddp.streaming_vision_align(raw_stream_embedding)
                stream_embedding = aligned_stream_embedding / (aligned_stream_embedding.norm(dim=-1, keepdim=True) + 1e-9)

                # --- Calculate Target Embedding for the current window ---
                # The current window ends at new_frame_idx
                current_window_start_idx = new_frame_idx - MODEL_MAX_FRAMES + 1
                current_window_end_idx = new_frame_idx + 1 # Slice end index is exclusive

                # Select the window of frames across the batch dimension
                actual_current_window_frames = image[:, :, current_window_start_idx:current_window_end_idx, :, :] # [B, C, MODEL_MAX_FRAMES, H, W]

                # Use no_grad for the target embedding calculation as it's treated as the fixed target
                with torch.no_grad():
                    # Pass the full window tensor to the full vision encoder
                    raw_target_embedding = model_without_ddp.vision_encoder(actual_current_window_frames)

                    # Align and Normalize
                    aligned_target_embedding = model_without_ddp.vision_align(raw_target_embedding)
                    target_embedding = aligned_target_embedding / (aligned_target_embedding.norm(dim=-1, keepdim=True) + 1e-9)

                    # Calculate cosine similarity for logging (detached)
                    cosine_similarity = torch.nn.functional.cosine_similarity(stream_embedding.detach(), target_embedding.detach(), dim=1)
                    cosine_similarity_avg = cosine_similarity.mean().item() # Get scalar value


                # --- Calculate Loss ---
                # Both predicted (stream_embedding) and target (target_embedding) are now [B, C_embed_dim]
                # The loss is calculated between the stream embedding (which requires gradients)
                # and the target embedding (which is detached).
                loss = cosine_sim_loss(stream_embedding, target_embedding)

                # --- Optional Debug Saving ---
                # This saves data for the *first* item of the *first* batch in the current iteration
                # It might be large, consider making this conditional or saving less often
                # Save only once per epoch for the first comparable window of the first batch
                if log_debug and i == 0 and new_frame_idx == MODEL_MAX_FRAMES - 1:
                     logger.info(f"Saving debug data at global step {global_step}, frame index {new_frame_idx}")
                     try:
                         save_debug_step_data(
                            output_dir = config.output_dir, # Use config output dir
                            global_step = global_step,
                            frame_idx = new_frame_idx,
                            new_frame_input = current_streaming_frame[0].cpu(), # Save first item from batch
                            current_hidden_state_input = tuple(h[0].detach().clone().cpu() for h in curr_hidden_state), # Save first item from batch
                            actual_window_input = actual_current_window_frames[0].cpu(), # Save first item from batch
                            stream_embedding_output = stream_embedding[0].cpu(), # Save first item from batch
                            target_embedding_output = target_embedding[0].cpu(), # Save first item from batch
                            model_state_dict = model_without_ddp.state_dict()
                        )
                         logger.info(f"==== Saved Debug Step Data to {config.output_dir}/ ====")
                     except Exception as e:
                         logger.error(f"Error saving debug data: {e}", exc_info=True)


                loss_dict = dict(loss_cosine=loss)
                total_loss = sum(loss_dict.values())


                # --- Backpropagation and Optimization ---
                # Check if using DeepSpeed
                if hasattr(config, "deepspeed") and config.deepspeed.enable:
                    model.backward(total_loss)
                    model.step()
                else:
                    # Standard PyTorch / AMP training step
                    # Using scaler for mixed precision if enabled
                    optimizer.zero_grad() # Reset gradients from previous step
                    # Use scaler.scale() for loss if using AMP
                    if config.use_half_precision:
                         scaler.scale(total_loss).backward()
                         if config.optimizer.max_grad_norm > 0:
                             # Unscale before clipping
                             scaler.unscale_(optimizer)
                             torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                         scaler.step(optimizer)
                         scaler.update() # Update the scale for the next iteration
                    else: # Standard training without AMP scaler
                        total_loss.backward()
                        if config.optimizer.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                        optimizer.step()

                    # Scheduler step is typically done *after* optimizer.step()
                    # and can be done per batch or per epoch.
                    # If using a step-based scheduler (like OneCycleLR), call it here.
                    # If using epoch-based, call it outside this loop after iterating through frames.
                    # Assuming step-based scheduler:
                    scheduler.step()


                # --- Update Hidden State for the next frame ---
                # Detach the hidden state to prevent gradients from flowing through the RNN across time steps
                # Clone to ensure it's a new tensor, not just a view
                curr_hidden_state = tuple(h.detach().clone() for h in new_hidden_state)


                # --- Logging Metrics ---
                # Log loss for the current frame within the window
                for loss_name in active_loss_names:
                    loss_value = loss_dict[loss_name].item() # Get scalar value
                    # Update metric logger for this media type and loss component
                    metric_logger.update(**{f"{media_type}-{loss_name}": loss_value})

                # Log the batch-wise cosine similarity
                metric_logger.update(**{f"{media_type}-cosine_similarity": cosine_similarity_avg})

                # Update metric logger with the current learning rate and model temperature
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                # Assuming model_without_ddp has a 'temp' attribute
                if hasattr(model_without_ddp, 'temp'):
                     metric_logger.update(temperature=model_without_ddp.temp.item())


                # --- Periodic Evaluation ---
                # Perform evaluation every EVAL_FREQ_STEPS global steps
                # Check if it's the main process to avoid redundant evaluations in DDP
                if global_step % EVAL_FREQ_STEPS == 0:
                     logger.info(f"Performing periodic evaluation at global step {global_step}...")
                     # Call the evaluation function
                     avg_sim = evaluate_streaming_similarity(
                         model=model_without_ddp, # Pass the unwrapped model
                         device=device,
                         transform=inference_transform,
                         video_path=EVAL_VIDEO_PATH,
                         model_max_frames=MODEL_MAX_FRAMES,
                         output_dir=config.output_dir, # Use config output dir for saving plots
                         global_step=global_step
                     )
                     # Log the evaluation result
                     metric_logger.update(eval_avg_sim=avg_sim)
                     logger.info(f"Evaluation at step {global_step} complete. Average Similarity: {avg_sim:.4f}")
                     # Ensure model is back in training mode after evaluation
                     model.train()


                # Log aggregated metrics to Weights & Biases (wandb) periodically if enabled
                # This logging happens every `log_freq` steps *within* the batch/frame loop
                if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
                    # Get the globally averaged metrics from the logger (synchronized across processes if distributed)
                    # Note: `get_global_avg_dict` should handle synchronization internally
                    averaged_logs = metric_logger.get_global_avg_dict()
                    # Send the logs to wandb, associated with the current global step
                    log_dict_to_wandb(averaged_logs, step=global_step, prefix="train/")


                # Increment the global step counter
                global_step += 1

                # Log a Divider every 100 steps (this is separate from metric_logger.log_every)
                if global_step % 100 == 0:
                    logger.info('─'*80)
                    logger.info(f"Step: {global_step}")
                    logger.info(f"Current Frame Index within Batch Video: {new_frame_idx}/{T-1}")
                    logger.info(f"Batch-wise Cosine Similarity | {cosine_similarity_avg*100:.2f}%")
                    logger.info(f"Cosine Embedding Loss      | {total_loss.item():.4f}") # Use item() for scalar
                    logger.info(f"Learning Rate              | {optimizer.param_groups[0]['lr']:.6f}")
                    if hasattr(model_without_ddp, 'temp'):
                         logger.info(f"Temperature                | {model_without_ddp.temp.item():.4f}")
                    # If evaluation just happened, log that as well
                    if global_step > 0 and global_step % EVAL_FREQ_STEPS == 0 and is_main_process():
                        logger.info(f"Evaluation Average Sim     | {metric_logger.meters['eval_avg_sim'].value:.4f}")
                    logger.info('─'*80)


                # --- Debugging Hooks ---
                # Optional early termination conditions for debugging
                if config.debug and global_step % 20 == 0:
                    logger.info("Debug mode: breaking training loop early (step condition).")
                    # Break the inner frame loop
                    break
                if config.debug and global_step % (2 * log_freq + 3) == 0: # Another arbitrary break condition
                     logger.info("Debug mode: breaking training loop early (log freq condition).")
                     # Break the inner frame loop
                     break

                # --- Iteration-based Checkpointing ---
                # Save a checkpoint at specified step intervals if `save_iter` > 0
                # Checkpoint after incrementing global_step
                if config.get('save_iter', 0) > 0 and global_step % config.save_iter == 0:
                    logger.info(f"Saving checkpoint at global step {global_step}")
                    if hasattr(config, "deepspeed") and config.deepspeed.enable:
                        # DeepSpeed handles checkpoint saving logic
                        checkpoint_tag = f"ckpt_iter{global_step:07d}.pth" # Padded step number
                        # Exclude frozen parameters to save space if needed
                        model.save_checkpoint(config.output_dir, tag=checkpoint_tag, save_latest=False, exclude_frozen_parameters=True)
                    elif is_main_process(): # Only the main process saves checkpoints in standard DDP
                        # Get the model's state dictionary
                        state_dict = model_without_ddp.state_dict()
                        # Identify parameters that are frozen (do not require gradients)
                        param_requires_grad_dict = {
                            name: param.requires_grad for (name, param) in model_without_ddp.named_parameters()
                        }
                        # Create a list of keys corresponding to frozen parameters
                        keys_to_remove = []
                        for param_name in state_dict.keys():
                            # Check if the parameter exists in named_parameters (might not if it's a buffer)
                            if param_name in param_requires_grad_dict and not param_requires_grad_dict[param_name]:
                                keys_to_remove.append(param_name)
                        # Remove frozen parameters from the state dictionary before saving
                        if keys_to_remove:
                            logger.info(f"Removing {len(keys_to_remove)} frozen parameters from checkpoint: {keys_to_remove}")
                            for param_name in keys_to_remove:
                                # Ensure key exists before deleting
                                if param_name in state_dict:
                                     del state_dict[param_name]

                        # Assemble the checkpoint object including model, optimizer, scheduler states, etc.
                        save_obj = {
                            "model": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict() if config.use_half_precision else None, # Save scaler state if using AMP
                            "config": config,             # Save config for reproducibility
                            "epoch": epoch,               # Current epoch
                            "global_step": global_step,   # Current step
                        }
                        # Define the checkpoint filename
                        checkpoint_filename = join(config.output_dir, f"ckpt_iter{global_step:07d}.pth") # Padded step number
                        # Save the checkpoint object to disk
                        torch.save(save_obj, checkpoint_filename)
                        logger.info(f"Saved iteration checkpoint to {checkpoint_filename}")

            # If debug break happened in the inner loop, break the outer batch loop too
            if config.debug and (global_step % 20 == 0 or global_step % (2 * log_freq + 3) == 0):
                 logger.info("Debug mode: breaking batch loop early.")
                 break # Break the outer batch loop

    # --- Training Loop End ---

    # Synchronize metrics across all distributed processes before logging final epoch stats
    # This is important even if logging happens per step within the loop,
    # as the final epoch average needs to be accurate.
    metric_logger.synchronize_between_processes()
    # Log the averaged metrics for the completed epoch
    logger.info(f"Averaged stats for Epoch [{epoch}] across all processes: {metric_logger.global_avg()}")

    # Log final epoch average metrics to wandb if enabled and on main process
    if is_main_process() and config.wandb.enable:
         # Get the globally averaged metrics (already synchronized)
         averaged_logs = metric_logger.get_global_avg_dict()
         # Log epoch-level averages. You might use a separate step counter for epochs
         # or log with the final global_step of the epoch.
         log_dict_to_wandb(averaged_logs, step=global_step, prefix=f"epoch_{epoch}/")
         # Also log to train/ prefix at the final step
         log_dict_to_wandb(averaged_logs, step=global_step, prefix="train/")


    # Return the updated global step count
    return global_step

# Note: You need to ensure the following are defined and imported correctly:
# - MetricLogger, SmoothedValue (from vision_language_pretraining.common.utils.metric_logger or similar)
# - MetaLoader_rs, get_media_types (from vision_language_pretraining.data.loader or similar)
# - CosineEmbeddingLoss (from torch.nn)
# - is_main_process (from vision_language_pretraining.common.utils.dist_utils or similar)
# - log_dict_to_wandb (from vision_language_pretraining.common.utils.logger or similar, assuming wandb integration)
# - save_debug_step_data (your custom function)
# - logger (standard Python logging object)
# - Your InternVideo2_CLIP_Small model class and its methods (streaming_vision_encoder, vision_encoder, streaming_vision_align, vision_align, init_hidden, transform, config)
# - config object with necessary attributes (log_freq, distributed, use_half_precision, use_bf16, optimizer.max_grad_norm, wandb.enable, output_dir, num_frames, size_t, save_iter, debug, deepspeed.enable)
# - tokenizer object
# - optimizer, scheduler, scaler objects instantiated before calling train.

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
    return train_loaders, test_name2loaders, media_types


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
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
                optimizer,
                tokenizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                data_type,
                skip_num = global_step - start_step
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
