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
    Performs one epoch of training with optimization steps for each MSE loss calculation.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=100, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=100, fmt="{value:.4f}"))
    active_loss_names = ["loss_mse"]
    media_types = get_media_types(train_loaders)

    for loss_name in active_loss_names:
        for media_type_key in media_types:
            metric_logger.add_meter(
                f"{media_type_key}-{loss_name}", SmoothedValue(window=100, fmt="{value:.4f}")
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

    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        # text_input = tokenizer(text).to(device)

        logger.info(f"Logging data for debugging: image shape: {image.shape}, text: {text}, idx: {idx}")

        # Prepare the video frames
        image = image.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        B, C, T, H, W = image.shape
        assert T >= MODEL_MAX_FRAMES, f"Video has shape {image.shape}, T should be >= {MODEL_MAX_FRAMES}."

        # Calculate initial full forward pass embedding
        model.vision_encoder.reset_state()
        initial_embedding = model.vision_encoder(image[:, :, :MODEL_MAX_FRAMES, :, :], force_full_forward=True)

        # Process each subsequent frame with individual optimization steps
        for t in range(MODEL_MAX_FRAMES, T):
            frame = image[:, :, t, :, :]  # [B, C, H, W]

            with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
                # Get window embedding using UpdateTransformer
                window_embedding = model.vision_encoder(frame)

                # Calculate full forward embedding for comparison
                with torch.no_grad():
                    model.vision_encoder.reset_state()
                    full_forward_embedding = model.vision_encoder(
                        image[:, :, t-MODEL_MAX_FRAMES+1:t+1, :, :],
                        force_full_forward=True
                    )

                # Calculate MSE loss for this frame
                loss_mse = torch.nn.functional.mse_loss(full_forward_embedding, window_embedding)
                loss_dict = {"loss_mse": loss_mse}
                total_loss = loss_mse

            # --- Backpropagation and Optimization ---
            if hasattr(config, "deepspeed") and config.deepspeed.enable:
                model.backward(total_loss)
                model.step()
            else:
                if not config.use_half_precision or config.get('use_bf16', True):
                    optimizer.zero_grad()
                    total_loss.backward()
                    if config.optimizer.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                else:
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    if config.optimizer.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

            # Update metrics for this step
            for loss_name in active_loss_names:
                loss_value = loss_dict[loss_name]
                loss_value = loss_value if isinstance(loss_value, float) else loss_value.item()
                metric_logger.update(**{f"{media_type}-{loss_name}": loss_value})

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(temperature=model_without_ddp.temp.item())

            # Log to wandb if enabled
            if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
                averaged_logs = metric_logger.get_global_avg_dict()
                log_dict_to_wandb(averaged_logs, step=global_step, prefix="train/")

            global_step += 1
            logger.info(f"Training -- Step [{global_step:,}]")

            # Debugging hooks
            if config.debug and global_step % 20 == 0:
                logger.info("Debug mode: breaking training loop early (step condition).")
                break
            if config.debug and global_step % (2 * log_freq + 3) == 0:
                logger.info("Debug mode: breaking training loop early (log freq condition).")
                break

            # Checkpointing
            if config.get('save_iter', 0) and global_step % config.save_iter == 0:
                if hasattr(config, "deepspeed") and config.deepspeed.enable:
                    checkpoint_tag = f"ckpt_iter{global_step:02d}.pth"
                    model.save_checkpoint(config.output_dir, tag=checkpoint_tag, save_latest=False, exclude_frozen_parameters=True)
                elif is_main_process():
                    state_dict = model_without_ddp.state_dict()
                    param_requires_grad_dict = {
                        name: param.requires_grad for (name, param) in model_without_ddp.named_parameters()
                    }
                    keys_to_remove = []
                    for param_name in state_dict.keys():
                        if param_name in param_requires_grad_dict and not param_requires_grad_dict[param_name]:
                            keys_to_remove.append(param_name)
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

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
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
