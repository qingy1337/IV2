import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
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

def train(
    model,                # The neural network model being trained
    train_loaders,        # A list or dictionary of data loaders for the training datasets
    optimizer,            # The optimization algorithm (e.g., Adam, SGD)
    tokenizer,            # Tokenizer for processing text data
    epoch,                # The current epoch number
    global_step,          # The total number of training steps performed so far
    device,               # The computing device (CPU or GPU)
    scheduler,            # Learning rate scheduler
    scaler,               # Gradient scaler for automatic mixed precision (AMP)
    config,               # Configuration object containing hyperparameters and settings
    data_type,            # The data type for AMP (e.g., torch.float16, torch.bfloat16)
    skip_num=0,            # Number of batches to skip at the beginning of the epoch (for resuming)
    log_debug=False,   # Debugging logs for backprop
):
    """
    Performs one epoch of training.

    Args:
        model: The model to be trained.
        train_loaders: Data loaders for the training data.
        optimizer: The optimizer instance.
        tokenizer: The text tokenizer.
        epoch: Current epoch index.
        global_step: Current global training step.
        device: The device to train on.
        scheduler: The learning rate scheduler.
        scaler: The gradient scaler for mixed precision.
        config: The experiment configuration.
        data_type: The data type for AMP.
        skip_num: Number of batches to skip.

    Returns:
        int: The updated global step count after the epoch.
    """
    # Set the model to training mode (enables dropout, batch norm updates, etc.)
    model.train()

    # Initialize MetricLogger to track and average metrics during training
    metric_logger = MetricLogger(delimiter="  ")
    # Add meters to track learning rate and the model's temperature parameter (often used in contrastive losses)
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    # Determine the names of the active loss components based on non-zero weights in the config
    active_loss_names = ["loss_mse"]

    # Identify the different types of media (e.g., 'image', 'video') present in the training loaders
    media_types = get_media_types(train_loaders)

    # Add specific meters for each active loss component and each media type
    for loss_name in active_loss_names:
        for media_type_key in media_types:
            metric_logger.add_meter(
                f"{media_type_key}-{loss_name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )

    # Define the header for logging messages for this epoch
    header = f"Train Epoch: [{epoch}]"
    # Get the frequency (in steps) for logging training progress
    log_freq = config.log_freq

    # If using distributed training, ensure the sampler shuffles data correctly for the current epoch
    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch) # Necessary for DistributedSampler reproducibility/shuffling

    # Create a MetaLoader that aggregates data from different media-specific loaders
    # It also handles skipping initial batches if resuming training part-way through an epoch
    train_loader_agg = MetaLoader_rs(name2loader=dict(list(zip(media_types, train_loaders))), skip_num=skip_num)

    # Get the underlying model instance, unwrapping it from DDP if necessary
    model_without_ddp = model.module if config.distributed else model
    # Wrap the aggregated data loader with the metric logger to enable periodic logging
    iterator = metric_logger.log_every(train_loader_agg, log_freq, header)

    # --- Training Loop Start ---
    # Iterate over batches provided by the MetaLoader

    MODEL_MAX_FRAMES = config.num_frames

    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        # Move input data to the designated compute device
        image = image.to(device, non_blocking=True) # Use non_blocking for potential speedup with pinned memory
        idx = idx.to(device, non_blocking=True)     # Index, potentially used for contrastive learning negative sampling

        if log_debug:
            logger.info(f"Logging data for debugging: image shape: {image.shape}, text: {text}, idx: {idx}")

        # Tokenize text data and move it to the device
        text_input = tokenizer(text).to(device)

        # Enable automatic mixed precision context if configured
        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            # Forward pass: Feed data through the model to get predictions and calculate losses

            # ,-----------------------------------------------------------------.
            # | Assumption: media_type is video, image is shape [B, T, C, H, W] |
            # `-----------------------------------------------------------------'

            image = image.permute(0, 2, 1, 3, 4)

            # ,------------------------------------.
            # | Now image is shape [B, C, T, H, W] |
            # `------------------------------------'

            # Iterate over each frame in the video (time dimension)
            B, C, T, H, W = image.shape
            assert T >= MODEL_MAX_FRAMES, f"Video has {T} frames, needs at least {MODEL_MAX_FRAMES}."

            # Define a pooling function (make sure this aligns with how InternVideo2 gets its final embedding)
            def pool_embedding_sequence(emb_sequence):
                # emb_sequence is [B, L, C_embed_dim]
                if emb_sequence.ndim == 3:
                    # Example: Mean pooling. If InternVideo2 uses a CLS token, adapt this.
                    # Or if InternVideo2 has a specific pooling layer (e.g. self.temporal_pool)
                    # you might want to use that concept if applicable.
                    return emb_sequence.mean(dim=1) # -> [B, C_embed_dim]
                elif emb_sequence.ndim == 2: # Already pooled
                    return emb_sequence
                else:
                    raise ValueError(f"Unexpected embedding shape for pooling: {emb_sequence.shape}")

            # --- Initial Step ---
            # Get the embedding for the very first window (frames 0 to MODEL_MAX_FRAMES-1)
            # This will serve as the "previous_embedding" for the first call to forward_update
            initial_window_frames = image[:, :, :MODEL_MAX_FRAMES, :, :] # e.g., frames 0..7 if MODEL_MAX_FRAMES=8
            with torch.no_grad():
                # Get the unpooled sequence from the base model
                prev_emb_sequence = model_without_ddp.vision_encoder.forward_full(initial_window_frames)
                # Pool it to be [B, C_embed_dim]
                # This pooled embedding is for the window (0 to MODEL_MAX_FRAMES-1)
                pooled_prev_window_embedding = pool_embedding_sequence(prev_emb_sequence.clone().detach())

            # Skip the first MODEL_MAX_FRAMES frames
            for t_new_frame_idx in range(MODEL_MAX_FRAMES, T):
                new_frame = image[:, :, t_new_frame_idx, :, :]

                if log_debug:
                    logger.info(f"On frame [{t_new_frame_idx-MODEL_MAX_FRAMES}:{t_new_frame_idx}], the previous embedding's shape is {pooled_prev_window_embedding.shape}")

                predicted_pooled_current_window_embedding = model.vision_encoder.forward_update( # New embedding using the UpdateTransformer
                    new_frame,
                    prev_embedding = pooled_prev_window_embedding
                )

                # --- Calculate Target for the current window ---
                # The current window starts at (t_new_frame_idx - MODEL_MAX_FRAMES + 1)
                # and ends at t_new_frame_idx (inclusive).
                current_window_start_idx = t_new_frame_idx - MODEL_MAX_FRAMES + 1
                current_window_end_idx = t_new_frame_idx + 1 # Slice end index is exclusive

                actual_current_window_frames = image[:, :, current_window_start_idx:current_window_end_idx, :, :]

                with torch.no_grad():
                    # Get the unpooled sequence for the current window's target
                    target_emb_sequence_current_window = model_without_ddp.vision_encoder.forward_full(actual_current_window_frames)
                    # Pool it to get the target [B, C_embed_dim]
                    target_pooled_current_window_embedding = pool_embedding_sequence(target_emb_sequence_current_window)

                    if log_debug:
                        target_norm = torch.linalg.norm(target_pooled_current_window_embedding, dim=-1).mean()
                        logger.info(f"Target embedding mean L2 norm: {target_norm.item():.4f}")
                        # Optionally, print min/max values
                        target_min = target_pooled_current_window_embedding.min()
                        target_max = target_pooled_current_window_embedding.max()
                        logger.info(f"Target embedding min: {target_min.item():.4f}, max: {target_max.item():.4f}")

                if log_debug:
                    logger.info(f"Norm of predicted_pooled_current_window_embedding: {predicted_pooled_current_window_embedding.norm()}")

                # --- Calculate Loss ---
                # Both predicted and target are now [B, C_embed_dim]
                loss = torch.nn.functional.mse_loss(predicted_pooled_current_window_embedding, target_pooled_current_window_embedding)
                loss_dict = dict(loss_mse=loss)
                total_loss = sum(loss_dict.values())

                if log_debug:
                    logger.info(f"1 Total Loss requires grad: {total_loss.requires_grad}")
                # --- Backpropagation and Optimization ---
                # Check if using DeepSpeed for optimized distributed training
                if hasattr(config, "deepspeed") and config.deepspeed.enable:
                    # DeepSpeed engine handles backward pass, gradient synchronization, and optimizer step
                    model.backward(total_loss) # Use DeepSpeed's backward method
                    model.step()              # Use DeepSpeed's step method (includes optimizer step, LR schedule)

                    if log_debug:
                        logger.info("BACKWARD SUCCESSFUL!")
                        logger.info(f"2 Total Loss requires grad: {total_loss.requires_grad}")
                else:
                    # Standard PyTorch / AMP training step
                    # Check if not using float16 AMP or explicitly using bfloat16 (which often doesn't require scaling)
                    if not config.use_half_precision or config.get('use_bf16', True):
                        # --- Standard Precision or BFloat16 ---
                        optimizer.zero_grad() # Reset gradients from previous step
                        total_loss.backward() # Compute gradients of the loss w.r.t. model parameters
                        # Apply gradient clipping if specified in the config to prevent exploding gradients
                        if config.optimizer.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                        optimizer.step()      # Update model parameters based on computed gradients
                        scheduler.step()      # Update learning rate according to the schedule
                    else:
                        # --- Float16 Mixed Precision with GradScaler ---
                        optimizer.zero_grad() # Reset gradients
                        # Scale the loss before backpropagation to prevent gradient underflow with float16
                        scaler.scale(total_loss).backward()
                        # Apply gradient clipping if specified (must unscale gradients first)
                        if config.optimizer.max_grad_norm > 0:
                            scaler.unscale_(optimizer) # Unscale gradients before clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                        # scaler.step() first unscales gradients, checks for inf/NaNs, then calls optimizer.step()
                        scaler.step(optimizer)
                        # scaler.update() adjusts the scaling factor for the next iteration
                        scaler.update()
                        scheduler.step()      # Update learning rate

                if log_debug:
                    logger.info(f"3 Total Loss requires grad: {total_loss.requires_grad}")
                # --- Logging Metrics ---
                # Update metric logger with the values of individual loss components for the current batch
                # loss_dict = {"loss_mse": loss_for_logging}
                for loss_name in active_loss_names:
                    loss_value = loss_dict[loss_name]
                    # Ensure value is a standard Python number for logging
                    loss_value = loss_value if isinstance(loss_value, float) else loss_value.item()
                    metric_logger.update(**{f"{media_type}-{loss_name}": loss_value}) # Log loss per media type
                # Update metric logger with the current learning rate and model temperature
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(temperature=model_without_ddp.temp.item())

                # Log aggregated metrics to Weights & Biases (wandb) periodically if enabled
                if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
                    # Get the globally averaged metrics from the logger (synchronized across processes if distributed)
                    averaged_logs = metric_logger.get_global_avg_dict()
                    # Send the logs to wandb, associated with the current global step
                    log_dict_to_wandb(averaged_logs, step=global_step, prefix="train/")

                # Increment the global step counter
                global_step += 1

                # Log Step Info
                if global_step % 100 == 0:
                    logger.info(f"+{'-'*50}\n| Training @ step [{global_step:,}]")

                if log_debug:
                    logger.info(f"4 Total Loss requires grad: {total_loss.requires_grad}")

                # --- Debugging Hooks ---
                # Optional early termination conditions for debugging
                if config.debug and global_step % 20 == 0:
                    logger.info("Debug mode: breaking training loop early (step condition).")
                    break
                if config.debug and global_step % (2 * log_freq + 3) == 0: # Another arbitrary break condition
                    logger.info("Debug mode: breaking training loop early (log freq condition).")
                    break

                # --- Iteration-based Checkpointing ---
                # Save a checkpoint at specified step intervals if `save_iter` > 0
                if config.get('save_iter', 0) and global_step % config.save_iter == 0:
                    if log_debug:
                        logger.info(f"5 Total Loss requires grad: {total_loss.requires_grad}")
                    if hasattr(config, "deepspeed") and config.deepspeed.enable:
                        # DeepSpeed handles checkpoint saving logic
                        checkpoint_tag = f"ckpt_iter{global_step:02d}.pth"
                        # Exclude frozen parameters to save space if needed
                        model.save_checkpoint(config.output_dir, tag=checkpoint_tag, save_latest=False, exclude_frozen_parameters=True)
                    elif is_main_process(): # Only the main process saves checkpoints in standard DDP
                        # Get the model's state dictionary
                        if log_debug:
                            logger.info(f"6 Total Loss requires grad: {total_loss.requires_grad}")
                        state_dict = model_without_ddp.state_dict()
                        # Identify parameters that are frozen (do not require gradients)
                        param_requires_grad_dict = {
                            name: param.requires_grad for (name, param) in model_without_ddp.named_parameters()
                        }
                        # Create a list of keys corresponding to frozen parameters
                        keys_to_remove = []
                        for param_name in state_dict.keys():
                            if param_name in param_requires_grad_dict and not param_requires_grad_dict[param_name]:
                                keys_to_remove.append(param_name)
                        # Remove frozen parameters from the state dictionary before saving
                        if keys_to_remove:
                            logger.info(f"Removing {len(keys_to_remove)} frozen parameters from checkpoint: {keys_to_remove}")
                            for param_name in keys_to_remove:
                                del state_dict[param_name]

                        # Assemble the checkpoint object including model, optimizer, scheduler states, etc.
                        save_obj = {
                            "model": state_dict,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), # Important for resuming AMP training
                            "config": config,             # Save config for reproducibility
                            "epoch": epoch,               # Current epoch
                            "global_step": global_step,   # Current step
                        }
                        # Define the checkpoint filename
                        checkpoint_filename = join(config.output_dir, f"ckpt_iter{global_step:02d}.pth")
                        # Save the checkpoint object to disk
                        torch.save(save_obj, checkpoint_filename)
                        logger.info(f"Saved iteration checkpoint to {checkpoint_filename}")


                # --- Prepare for next iteration ---
                # The target embedding of the current window becomes the "previous" embedding for the next update.
                # Detach to prevent gradients from flowing back multiple steps if not intended,
                # and it's good practice since it's a target from a no_grad block.
                pooled_prev_window_embedding = target_pooled_current_window_embedding.clone().detach()

    # --- Training Loop End ---

    # Synchronize metrics across all distributed processes before logging final epoch stats
    metric_logger.synchronize_between_processes()
    # Log the averaged metrics for the completed epoch
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
    # Return the updated global step count
    return global_step

from torch.utils.data._utils.collate import default_collate
import torch

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
