from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import lightning as pl
import torch
from hydra.utils import instantiate

from src.modules.lr_schedulers import WarmupThenReduceLROnPlateau


def resolve_warmup_steps(*, module: pl.LightningModule, sched_cfg: Mapping[str, Any]) -> int:
    """
    从 scheduler 配置解析 warmup step 数。

    输入参数:
        - module: pl.LightningModule, 当前 wrapper; 提供 trainer.estimated_stepping_batches
        - sched_cfg: Mapping[str, Any], scheduler 配置; 包含 total_steps/warmup_steps/warmup_ratio

    输出:
        - warmup_steps: int, 线性 warmup 覆盖的 optimizer step 数
    """
    total_steps = sched_cfg["total_steps"]
    if total_steps is None:
        total_steps = getattr(module.trainer, "estimated_stepping_batches", None)
    if total_steps is None or int(total_steps) <= 0:
        raise RuntimeError("warmup scheduler requires a positive total_steps value.")
    total_steps = int(total_steps)

    warmup_steps = sched_cfg["warmup_steps"]
    if warmup_steps is None:
        warmup_ratio = float(sched_cfg["warmup_ratio"])
        if not (0.0 <= warmup_ratio < 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}.")
        warmup_steps = int(round(total_steps * warmup_ratio))
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps], got warmup_steps={warmup_steps}, total_steps={total_steps}."
        )
    return warmup_steps


def build_warmup_only_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    sched_cfg: Mapping[str, Any],
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    构建仅包含 step 级线性 warmup 的 scheduler。

    输入参数:
        - optimizer: torch.optim.Optimizer, 被调度的优化器
        - sched_cfg: Mapping[str, Any], scheduler 配置; 包含 warmup_start_factor
        - warmup_steps: int, 已解析出的 warmup step 数

    输出:
        - scheduler: torch.optim.lr_scheduler.LRScheduler, Lightning step 级调度器
    """
    start_factor = float(sched_cfg["warmup_start_factor"])
    if not (0.0 < start_factor <= 1.0):
        raise ValueError(f"warmup_start_factor must be in (0, 1], got {start_factor}.")
    if warmup_steps == 0 or start_factor == 1.0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )


def build_warmup_plateau_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    sched_cfg: Mapping[str, Any],
    warmup_steps: int,
    pending_state: Mapping[str, Any] | None,
) -> WarmupThenReduceLROnPlateau:
    """
    构建 step 级 warmup + validation 级 plateau 的组合 scheduler。

    输入参数:
        - optimizer: torch.optim.Optimizer, 被调度的优化器
        - sched_cfg: Mapping[str, Any], scheduler 配置; 包含 warmup 与 plateau 参数
        - warmup_steps: int, 已解析出的 warmup step 数
        - pending_state: Mapping[str, Any] | None, checkpoint 暂存的 plateau 状态

    输出:
        - scheduler: WarmupThenReduceLROnPlateau, 组合调度器对象
    """
    scheduler = WarmupThenReduceLROnPlateau(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_start_factor=float(sched_cfg["warmup_start_factor"]),
        mode=str(sched_cfg["mode"]),
        factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]),
        threshold=float(sched_cfg["threshold"]),
        threshold_mode=str(sched_cfg["threshold_mode"]),
        cooldown=int(sched_cfg["cooldown"]),
        min_lr=sched_cfg["min_lr"],
        eps=float(sched_cfg["eps"]),
    )
    if pending_state is not None:
        scheduler.load_state_dict(dict(pending_state))
    return scheduler


def configure_stage1_optimizers(
    *,
    module: pl.LightningModule,
    optimizer_config: Any,
    scheduler_config: Any,
    interval: str,
    frequency: int,
    monitor_metric: str,
    pending_warmup_plateau_state: Mapping[str, Any] | None,
) -> tuple[Any, int, WarmupThenReduceLROnPlateau | None, Mapping[str, Any] | None]:
    """
    构造 Stage1 wrapper 的 optimizer 与 scheduler 配置。

    输入参数:
        - module: pl.LightningModule, 当前 wrapper; 提供 parameters 与 trainer 上下文
        - optimizer_config: Any, Hydra optimizer 配置、callable 或 None
        - scheduler_config: Any, Hydra scheduler 配置、callable 或 None
        - interval: str, Lightning scheduler interval
        - frequency: int, Lightning scheduler frequency
        - monitor_metric: str, plateau/checkpoint 监控指标 key
        - pending_warmup_plateau_state: Mapping[str, Any] | None, checkpoint 暂存的 plateau 状态

    输出:
        - config: Any, Lightning configure_optimizers 返回值
        - candidate_warmup_steps: int, candidate builder warmup step 数
        - warmup_plateau_scheduler: WarmupThenReduceLROnPlateau | None, 手动 plateau scheduler
        - pending_warmup_plateau_state: Mapping[str, Any] | None, 消费后剩余的暂存状态
    """
    # filter, 仅保留需要梯度的参数
    trainable_params = filter(lambda param: param.requires_grad, module.parameters())
    if optimizer_config is None:
        optimizer = torch.optim.AdamW(params=trainable_params, lr=1e-4, weight_decay=1e-2)
    elif isinstance(optimizer_config, functools.partial):
        optimizer = optimizer_config(params=trainable_params)
    elif callable(optimizer_config) and not hasattr(optimizer_config, "keys"):
        optimizer = optimizer_config(params=trainable_params)
    else:
        optimizer = instantiate(optimizer_config, params=trainable_params)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Failed to instantiate optimizer.")

    if scheduler_config is None:
        return {"optimizer": optimizer}, 0, None, pending_warmup_plateau_state

    if isinstance(scheduler_config, functools.partial):
        scheduler = scheduler_config(optimizer=optimizer)
        candidate_warmup_steps = 0
    elif callable(scheduler_config) and not hasattr(scheduler_config, "keys"):
        scheduler = scheduler_config(optimizer=optimizer)
        candidate_warmup_steps = 0
    elif hasattr(scheduler_config, "get") and scheduler_config.get("name", None) == "warmup_plateau":
        candidate_warmup_steps = resolve_warmup_steps(module=module, sched_cfg=scheduler_config)
        warmup_plateau_scheduler = build_warmup_plateau_scheduler(
            optimizer=optimizer,
            sched_cfg=scheduler_config,
            warmup_steps=candidate_warmup_steps,
            pending_state=pending_warmup_plateau_state,
        )
        return (
            {"optimizer": optimizer, "lr_scheduler": warmup_plateau_scheduler.lightning_warmup_config()},
            candidate_warmup_steps,
            warmup_plateau_scheduler,
            None,
        )
    elif hasattr(scheduler_config, "get") and scheduler_config.get("name", None) == "warmup_only":
        candidate_warmup_steps = resolve_warmup_steps(module=module, sched_cfg=scheduler_config)
        scheduler = build_warmup_only_scheduler(
            optimizer=optimizer,
            sched_cfg=scheduler_config,
            warmup_steps=candidate_warmup_steps,
        )
    else:
        scheduler = instantiate(scheduler_config, optimizer=optimizer)
        candidate_warmup_steps = 0
        if hasattr(scheduler, "__call__") and not hasattr(scheduler, "step"):
            scheduler = scheduler()

    return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,
                "interval": interval,
                "frequency": frequency,
            },
        },
        candidate_warmup_steps,
        None,
        pending_warmup_plateau_state,
    )
