# Copyright (c) Sophont, Inc
# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/main.py
# capi: https://github.com/facebookresearch/capi/blob/main/train_capi.py

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from matplotlib import pyplot as plt
from streaming import StreamingDataset
from torch import Tensor
from torch.utils.data import DataLoader

import data.mri_data as mri_data
import smri_mae.masking as masking
import smri_mae.model_mae as models_mae
import smri_mae.utils as ut
import smri_mae.visualization as vis

DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"

MODELS_DICT = models_mae.__dict__


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    global_rank = ut.get_rank()
    is_master = global_rank == 0
    world_size = ut.get_world_size()
    device = torch.device(args.device)
    ut.random_seed(args.seed, rank=global_rank)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_cfg_path = output_dir / "config.yaml"
        if out_cfg_path.exists():
            prev_cfg = OmegaConf.load(out_cfg_path)
            assert args == prev_cfg, "current config doesn't match previous config"
        else:
            OmegaConf.save(args, out_cfg_path)

        if args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.name,
                notes=args.notes,
                config=OmegaConf.to_container(args),
            )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("pretraining 3D ViTMAE")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"cwd: {Path.cwd()}")
    print(ut.get_sha())
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # data loaders
    train_loader, eval_loaders, mask_fn = create_data_loaders(args)

    # model
    model = MODELS_DICT[args.model](
        img_size=args.img_size,
        in_chans=args.get("in_chans", 1),
        patch_size=args.patch_size,
        **(args.get("model_kwargs") or {}),
    )
    model.to(device)
    print("model:", model, sep="\n")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params: {num_params / 1e6:.1f}M")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.compile:
        model = torch.compile(model)

    # optimizer
    total_batch_size = args.batch_size * args.accum_iter * world_size
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum x {world_size} gpus"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        print(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        print(f"lr: {args.lr:.2e}")

    param_groups = ut.get_param_groups(model)
    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    optimizer = torch.optim.AdamW(param_groups, betas=betas)

    epoch_num_batches = len(train_loader)
    steps_per_epoch = epoch_num_batches // args.accum_iter
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_schedule = ut.WarmupThenCosine(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
    )
    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    ut.load_model(args, model_without_ddp, optimizer, loss_scaler)

    print(f"start training for {args.epochs} epochs")
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            loss_scaler,
            lr_schedule,
            epoch,
            device,
            mask_fn=mask_fn,
        )

        eval_stats = {}
        eval_plots = {}
        for name, loader in eval_loaders.items():
            stats, plots = evaluate(
                args,
                model,
                loader,
                epoch,
                device,
                eval_name=name,
                mask_fn=mask_fn,
            )
            eval_stats.update(stats)
            eval_plots.update(plots)

        merged_stats = {"epoch": epoch, **train_stats, **eval_stats}
        if is_master:
            with (output_dir / "log.json").open("a") as f:
                print(json.dumps(merged_stats), file=f)

            for plot_name, img in eval_plots.items():
                plot_name = plot_name.replace("/", "__")
                img.save(output_dir / f"{plot_name}__{epoch:05d}.png")

        ut.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler)
        sync_checkpoints_to_r2(args, output_dir)

    if args.distributed:
        torch.distributed.destroy_process_group()

    total_time = time.monotonic() - start_time
    print(f"done! training time: {datetime.timedelta(seconds=int(total_time))}")


def mri_collate(
    samples: list[dict],
    *,
    augmentation: dict | None = None,
    include_meta: bool = True,
) -> dict[str, Tensor]:
    if augmentation:
        samples = [mri_data.augment_sample(sample, augmentation) for sample in samples]
        masks = [
            torch.as_tensor(sample["img_mask"].copy(), dtype=torch.bool) for sample in samples
        ]
    else:
        masks = [torch.as_tensor(sample["img_mask"].copy()) for sample in samples]
    batch = {
        "image": torch.stack([torch.as_tensor(sample["image"].copy()) for sample in samples]),
        "img_mask": torch.stack(masks),
    }
    if include_meta:
        batch["meta"] = [mri_data.make_collatable(sample["meta"]) for sample in samples]
    return batch


def create_data_loaders(args: DictConfig):
    mask_fn = masking.create_masking(
        args.masking,
        mask_ratio=args.mask_ratio,
        img_size=args.img_size,
        patch_size=args.patch_size,
        **(args.get("masking_kwargs") or {}),
    )
    print("mask generator:", mask_fn, sep="\n")

    data_loaders = {}
    dataset_names = [args.train_dataset] + args.eval_datasets

    for dataset_name in dataset_names:
        dataset_config = args.datasets[dataset_name].copy()
        print(f"loading dataset: {dataset_name}\n\n{OmegaConf.to_yaml(dataset_config)}")
        drop_last = dataset_config.pop("drop_last")
        dataset_kwargs = OmegaConf.to_container(dataset_config, resolve=True)

        loader = DataLoader(
            StreamingDataset(**dataset_kwargs),
            batch_size=args.batch_size,
            collate_fn=partial(
                mri_collate,
                augmentation=args.augmentation
                if dataset_name == args.train_dataset and args.augmentation.enabled
                else None,
                include_meta=dataset_name != args.train_dataset,
            ),
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
            drop_last=drop_last,
        )

        data_loaders[dataset_name] = loader

    train_loader = data_loaders.pop(args.train_dataset)
    return train_loader, data_loaders, mask_fn


def sync_checkpoints_to_r2(args: DictConfig, output_dir: Path) -> None:
    r2_sync_url = args.get("r2_sync")
    if not r2_sync_url or not ut.is_main_process():
        return

    cmd = ["aws", "s3", "sync", str(output_dir), str(r2_sync_url), "--profile", "r2"]
    print(f"syncing checkpoints to R2: {output_dir} -> {r2_sync_url}")
    subprocess.run(cmd, check=True)


def train_one_epoch(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
    mask_fn: masking.RandomMasking,
):
    model.train()

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", ut.SmoothedValue())
    header = f"Train: [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)
    steps_per_epoch = epoch_num_batches // args.accum_iter

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10
    profile_steps = int(args.get("profile_steps", 0) or 0)

    amp_dtype = getattr(torch, args.amp_dtype)
    use_cuda = device.type == "cuda"
    if use_cuda and args.presend_cuda:
        data_loader = ut.pre_send_to_cuda_wrapper(data_loader, device, dtype_map={torch.float16: amp_dtype})

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        profile_step = batch_idx < profile_steps

        if use_cuda and not args.presend_cuda:
            if profile_step:
                torch.cuda.synchronize()
                h2d_start = time.perf_counter()
            batch = ut.send_data(batch, device, dtype_map={torch.float16: amp_dtype})
            if profile_step:
                torch.cuda.synchronize()
                metric_logger.update(h2d_time=time.perf_counter() - h2d_start)

        batch_step = batch_idx + 1
        global_step = epoch * steps_per_epoch + batch_step // args.accum_iter
        lr = lr_schedule[global_step]
        need_update = batch_step % args.accum_iter == 0

        if need_update:
            ut.update_lr(optimizer.param_groups, lr)

        images = batch["image"]
        masks = mri_data.unpack_img_mask_batch(batch["img_mask"], images.shape[1:])
        batch["img_mask"] = masks

        if profile_step and use_cuda:
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss, mask_stats = model(
                images,
                img_mask=masks,
                mask_ratio=args.mask_ratio,
                pred_mask_ratio=args.pred_mask_ratio,
                mask_fn=mask_fn if args.masking != "random" else None,
                include_mask_stats=profile_step,
                with_state=False,
            )
        if profile_step and use_cuda:
            torch.cuda.synchronize()
            metric_logger.update(forward_time=time.perf_counter() - forward_start)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        if profile_step and use_cuda:
            torch.cuda.synchronize()
            backward_start = time.perf_counter()

        grad_norm = ut.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        if profile_step and use_cuda:
            torch.cuda.synchronize()
            metric_logger.update(backward_time=time.perf_counter() - backward_start)

        if need_update:
            grad_norm_value = grad_norm.item()
            metric_logger.update(
                loss=loss_value,
                lr=lr,
                grad=grad_norm_value,
                **mask_stats,
            )
            if log_wandb:
                wandb_stats = {
                    "train/loss": loss_value,
                    "train/lr": lr,
                    "train/grad": grad_norm_value,
                    **{f"train/{k}": v for k, v in mask_stats.items()},
                }
                wandb.log(wandb_stats, step=int(1000 * (epoch + batch_step / epoch_num_batches)))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
    mask_fn: masking.RandomMasking,
):
    model.eval()

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    log_wandb = args.wandb and ut.is_main_process()

    epoch_num_batches = len(data_loader)
    if epoch_num_batches <= 0:
        raise ValueError(f"eval loader {eval_name!r} has zero batches")

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10
    num_batches = min(num_batches, epoch_num_batches)
    example_step = random.randint(1, num_batches)
    amp_dtype = getattr(torch, args.amp_dtype)
    use_cuda = device.type == "cuda"
    if use_cuda and args.presend_cuda:
        data_loader = ut.pre_send_to_cuda_wrapper(data_loader, device, dtype_map={torch.float16: amp_dtype})

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=epoch_num_batches)
    ):
        if use_cuda and not args.presend_cuda:
            batch = ut.send_data(batch, device, dtype_map={torch.float16: amp_dtype})

        batch_step = batch_idx + 1

        images = batch["image"]
        img_mask = mri_data.unpack_img_mask_batch(batch["img_mask"], images.shape[1:])
        batch["img_mask"] = img_mask

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss, state = model(
                images,
                img_mask=img_mask,
                mask_ratio=args.mask_ratio,
                pred_mask_ratio=args.pred_mask_ratio,
                mask_fn=mask_fn if args.masking != "random" else None,
            )

        metric_logger.update(loss=loss)

        if batch_step == example_step:
            example_data = {
                "batch": ut.send_data(batch, "cpu"),
                "state": ut.send_data(state, "cpu"),
            }

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats ({eval_name}):", metric_logger)
    stats = {f"eval/{eval_name}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

    print(f"Making plots ({eval_name}): example={example_step}")
    plots = make_plots(args, **example_data)
    plots = {f"eval/{eval_name}/{k}": img for k, img in plots.items()}

    if log_wandb:
        wandb.log(stats, step=1000 * (epoch + 1))
        wandb.log(
            {k: wandb.Image(img, caption=f"example={example_step}") for k, img in plots.items()},
            step=1000 * (epoch + 1),
        )
    return stats, plots


def make_plots(
    args: DictConfig,
    batch: dict[str, Tensor],
    state: dict[str, Tensor],
) -> dict[str, Image.Image]:
    fig_kwargs = args.get("fig_kwargs", {})

    images = batch["image"]
    img_mask = batch.get("img_mask")
    if img_mask is not None:
        img_mask = img_mask.expand_as(images)

    raw_mean, raw_std = vis.raw_stats_from_batch(batch)

    plots = {}
    mask_pred_fig = vis.plot_mask_pred(
        target=images,
        pred=state["pred_images"],
        visible_mask=state["visible_mask"],
        pred_mask=state["pred_mask"],
        img_mask=img_mask,
        patch_size=args.patch_size,
        raw_mean=raw_mean,
        raw_std=raw_std,
        **ut.filter_kwargs(vis.plot_mask_pred, fig_kwargs),
    )
    plots["mask_pred"] = vis.fig2pil(mask_pred_fig)
    plt.close(mask_pred_fig)

    return plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
