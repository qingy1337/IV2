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

import torch
# Assume MetricLogger, SmoothedValue, get_media_types, MetaLoader_rs, logger,
# is_main_process, log_dict_to_wandb, join are defined elsewhere
from utils import MetricLogger, SmoothedValue, get_media_types, MetaLoader_rs, logger, is_main_process, log_dict_to_wandb
from os.path import join

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
    skip_num=0            # Number of batches to skip at the beginning of the epoch (for resuming)
):
    """
    Performs one epoch of training with step taken for each MSE loss calculation.

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
    # We'll update these meters *after* each inner step. They will average
    # the per-step values over the logging window.
    metric_logger = MetricLogger(delimiter="  ")
    # Add meters to track learning rate and the model's temperature parameter
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    # Determine the names of the active loss components (still just MSE for this case)
    active_loss_names = ["loss_mse"]

    # Identify the different types of media
    media_types = get_media_types(train_loaders)

    # Add specific meters for each active loss component and each media type
    for loss_name in active_loss_names:
        for media_type_key in media_types:
            # This meter will track the average of the per-step losses within the logging window
            metric_logger.add_meter(
                f"{media_type_key}-{loss_name}", SmoothedValue(window=100, fmt="{value:.4f}")
            )

    # Define the header for logging messages
    header = f"Train Epoch: [{epoch}]"
    # Get the frequency for logging progress (based on outer batch iterations)
    # The metric logger will still log based on the outer iterator 'i'.
    # The WandB logging happens based on global_step, which increments per inner step.
    log_freq = config.log_freq

    # If using distributed training, ensure the sampler shuffles data correctly
    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch)

    # Create MetaLoader
    train_loader_agg = MetaLoader_rs(name2loader=dict(list(zip(media_types, train_loaders))), skip_num=skip_num)

    # Get the underlying model instance
    model_without_ddp = model.module if config.distributed else model
    # Wrap the aggregated data loader with the metric logger
    # The iterator will yield batches, and the logger will trigger based on batch index 'i'
    iterator = metric_logger.log_every(train_loader_agg, log_freq, header)

    # --- Training Loop Start ---
    # Iterate over batches
    MODEL_MAX_FRAMES = config.num_frames

    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        # Move input data to the designated compute device
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        logger.info(f"Logging data for debugging: image shape: {image.shape}, text: {text}, idx: {idx}")

        # Tokenize text data and move it to the device
        text_input = tokenizer(text).to(device)

        # Permute image shape for processing [B, C, T, H, W]
        image = image.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = image.shape

        assert T >= MODEL_MAX_FRAMES, f"Video has shape {image.shape}, T should be >= {MODEL_MAX_FRAMES}."

        # Reset vision encoder state for this batch
        model.vision_encoder.reset_state()

        # Extract the first MODEL_MAX_FRAMES frames (for initial state/full forward target calculation)
        num_frames = min(T, MODEL_MAX_FRAMES)
        # Note: The original code uses `frames` (the initial segment) for the target calculation
        # inside the inner loop. This might be a specific architectural choice.
        # We are keeping this logic as is, only changing the optimization step frequency.
        frames = image[:, :, :num_frames, :, :]

        # Calculate initial full forward pass embedding (used in the loss calculation below)
        initial_full_embedding = model.vision_encoder(frames, force_full_forward=True)


        # Iterate over each frame starting from MODEL_MAX_FRAMES
        for t in range(MODEL_MAX_FRAMES, T):
            frame = image[:, :, t, :, :] # Get the current frame [B, C, H, W]

            # Enable automatic mixed precision context if configured
            with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
                # Calculate window embedding using the UpdateTransformer (uses state)
                window_embedding = model.vision_encoder(frame)

                # Calculate the target embedding for frame 't'.
                # Following the original code's logic, this target is based on the *initial* segment
                # 'frames' using a full forward pass *without* state.
                # This specific target logic might be unusual for sequence prediction
                # but we preserve it as it was in the original code snippet.
                with torch.no_grad():
                    model.vision_encoder.reset_state() # Need to reset to get clean full forward
                    target_embedding = model.vision_encoder(frames[:, :, -MODEL_MAX_FRAMES:, :, :], force_full_forward = True)

                # Calculate MSE loss for the current frame/window prediction
                loss = torch.nn.functional.mse_loss(target_embedding, window_embedding)

            # --- Per-loss optimization step ---
            # Zero gradients *before* the backward pass for this specific loss 't'
            if hasattr(config, "deepspeed") and config.deepspeed.enable:
                 # DeepSpeed engine handles zeroing and step
                 model.backward(loss)
                 model.step()
                 # Scheduler step within DeepSpeed step or managed separately?
                 # DeepSpeed schedulers are typically part of engine.step() or engine.lr_scheduler
                 # Assuming it's handled or needs explicit call depending on DeepSpeed config.
                 # For standard config, we call scheduler.step() below.
            else:
                # Standard PyTorch / AMP training step
                optimizer.zero_grad() # Zero gradients specifically for this loss
                if not config.use_half_precision or config.get('use_bf16', True):
                    # Standard Precision or BFloat16
                    loss.backward() # Compute gradients for this loss
                    if config.optimizer.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    optimizer.step() # Update based on this loss's gradients
                    scheduler.step() # Update learning rate

                else:
                    # Float16 Mixed Precision with GradScaler
                    scaler.scale(loss).backward() # Scale and backpropagate this loss
                    if config.optimizer.max_grad_norm > 0:
                        scaler.unscale_(optimizer) # Unscale before clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    scaler.step(optimizer) # Step based on this loss's gradients
                    scaler.update()        # Update scaler
                    scheduler.step()       # Update learning rate


            # --- Logging Metrics for this Step ---
            # Update metric logger with the value of the current individual loss
            # This meter will average the per-step losses over its window
            loss_value = loss.item()
            metric_logger.update(**{f"{media_type}-loss_mse": loss_value})

            # Update metric logger with the current learning rate and temperature
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # Check if temp is a parameter that requires grad or a buffer/constant
            temperature_value = model_without_ddp.temp.item() if hasattr(model_without_ddp.temp, 'item') else model_without_ddp.temp
            metric_logger.update(temperature=temperature_value)

            # Increment the global step counter *after* each optimization step
            global_step += 1

            # Log Step Info (this will be more frequent now)
            logger.info(f"Training -- Step [{global_step:,}]")

            # --- Debugging Hooks (checked per inner step) ---
            if config.debug and global_step % 20 == 0:
                logger.info("Debug mode: breaking training loop early (step condition).")
                # This break exits the inner loop. The outer loop will continue.
                # If you want to stop the entire batch/epoch, you might need a flag.
                # For simple debug breaks, this is usually sufficient.
                break
            if config.debug and global_step % (2 * log_freq + 3) == 0:
                 logger.info("Debug mode: breaking training loop early (log freq condition).")
                 break # Exits inner loop

            # --- Iteration-based Checkpointing (checked per inner step) ---
            if config.get('save_iter', 0) and global_step % config.save_iter == 0:
                # Checkpointing logic remains similar, but happens more often
                if hasattr(config, "deepspeed") and config.deepspeed.enable:
                    checkpoint_tag = f"ckpt_iter{global_step:02d}.pth"
                    model.save_checkpoint(config.output_dir, tag=checkpoint_tag, save_latest=False, exclude_frozen_parameters=True)
                elif is_main_process():
                    state_dict = model_without_ddp.state_dict()
                    param_requires_grad_dict = {
                        name: param.requires_grad for (name, param) in model_without_ddp.named_parameters()
                    }
                    keys_to_remove = [
                        param_name for param_name in state_dict.keys()
                        if param_name in param_requires_grad_dict and not param_requires_grad_dict[param_name]
                    ]
                    if keys_to_remove:
                        logger.info(f"Removing {len(keys_to_remove)} frozen parameters from checkpoint: {keys_to_remove}")
                        for param_name in keys_to_remove:
                            del state_dict[param_name]

                    save_obj = {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    checkpoint_filename = join(config.output_dir, f"ckpt_iter{global_step:02d}.pth")
                    torch.save(save_obj, checkpoint_filename)
                    logger.info(f"Saved iteration checkpoint to {checkpoint_filename}")

            # Log aggregated metrics to Weights & Biases periodically based on global_step
            if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
                 # This will log the average of all metric updates since the last wandb log call
                 averaged_logs = metric_logger.get_global_avg_dict() # Gets average over the metric logger's window
                 log_dict_to_wandb(averaged_logs, step=global_step, prefix="train/")


        # --- End of Inner Loop (processing frames within a batch) ---
        # No total loss calculation or single step needed here anymore

    # --- Training Loop End (processing batches) ---

    # Synchronize metrics across processes for final epoch stats
    # The logger will average all the per-step updates over the whole epoch
    metric_logger.synchronize_between_processes()
    # Log the averaged stats for the completed epoch
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
    num_steps_per_epoch = sum(len(d) for d in train_loaders) * 247 # 247 steps per video

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
