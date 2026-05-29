from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lightning as pl
import torch
import csv
import json


_ALLOWED_SCOPES = {"global", "by_source_folder"}


def build_metric_key(
    *,
    panel: str,
    metric: str,
    num_classes: int,
    scope: str,
    subpanel: str | None,
    source_folder: str | None,
    task_class_name: str | None,
) -> str:
    """
    构造验证日志 key: panel /(subpanel)/ scope / source_folder / metric_leaf

    输入参数:
        - panel: str, 顶层面板名, 如 val_score / val_uncapped / val_refined
        - metric: str, 指标 leaf 名, 如 F1 / PRAUC / p_sampling_p50
        - num_classes: int, task 类别总数; 二分类时不追加 task class suffix
        - scope: str, 指标作用域; 只允许 global / by_source_folder
        - subpanel: str | None, 子面板名; val_uncapped 使用 best/sampling
        - source_folder: str | None, 原始 source folder 名; scope=by_source_folder 时必须传入
        - task_class_name: str | None, task class 名; num_classes>2 时追加到 metric suffix

    输出:
        - key: str, W&B/Lightning 使用的指标 key
    """
    if scope not in _ALLOWED_SCOPES:
        raise ValueError(f"scope 只允许 {_ALLOWED_SCOPES}, 实际 {scope!r}。")
    if scope == "by_source_folder" and source_folder is None:
        raise ValueError("scope=by_source_folder 时必须传入 source_folder。")
    if scope == "global" and source_folder is not None:
        raise ValueError("scope=global 时不应传入 source_folder。")

    # str, 多分类 task-class 后缀后的指标名
    metric_leaf = metric
    if num_classes > 2 and task_class_name is not None:
        metric_leaf = f"{metric}_{task_class_name}"

    # list[str], 从左到右的 key 路径分段
    parts = [panel]
    if subpanel is not None:
        parts.append(subpanel)
    parts.append(scope)
    if scope == "by_source_folder":
        parts.append(str(source_folder))
    parts.append(metric_leaf)
    return "/".join(parts)


def log_scalar_payload(
    *,
    module: pl.LightningModule,
    payload: Mapping[str, torch.Tensor],
    monitor_metric: str,
    sync_dist: bool,
) -> None:
    """
    将标量 payload 写入 Lightning 日志: 把一个 epoch 里所有 step 的值做聚合，最后在 epoch 结束时记录(epoch级别，所以不会有 step-level 曲线)。

    输入参数:
        - module: pl.LightningModule, 当前 wrapper 模块
        - payload: Mapping[str, torch.Tensor], 要记录的 key: tensor 对
        - monitor_metric: str, 需要显示到 progress bar 的主监控指标 key
        - sync_dist: bool, 是否由 Lightning 同步 DDP 标量

    输出:
        - None, 原地调用 module.log
    """
    for key, value in payload.items():
        module.log(
            key,
            value,
            prog_bar=key == monitor_metric,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
        )


def log_wandb_curves(
    *,
    module: pl.LightningModule,
    curves: Mapping[str, Any],
    validation_index: int,
    every_n: int,
) -> None:
    """
    在 global zero 上记录 W&B 曲线。

    输入参数:
        - module: pl.LightningModule, 当前 wrapper 模块
        - curves: Mapping[str, Any], 曲线名到 class CurvePayload 的映射
        - validation_index: int, 当前 validation 序号, 从 0 或 1 开始由调用方约定
        - every_n: int, 每隔多少次 validation 上传一次

    输出:
        - None, logger 非 W&B 或非 global zero 时 no-op
    """
    def _is_wandb_logger(logger: Any) -> bool:
        """
        判断 logger 是否暴露 W&B experiment 接口: experiment = getattr(logger, "experiment", None)

        输入参数:
            - logger: Any, Lightning logger 或 logger collection

        输出:
            - is_wandb: bool, True 表示可按 W&B table 路径记录曲线
        """
        experiment = getattr(logger, "experiment", None)
        return experiment is not None and hasattr(experiment, "log")

    trainer = getattr(module, "trainer", None)
    if trainer is not None and not bool(getattr(trainer, "is_global_zero", True)):
        return
    if validation_index % int(every_n) != 0:
        return
    logger = getattr(module, "logger", None)
    if not _is_wandb_logger(logger):
        return
    experiment = logger.experiment
    for key, curve in curves.items():
        experiment.log({key: asdict(curve) if hasattr(curve, "__dataclass_fields__") else curve})


def write_validation_artifacts(
    *,
    run_dir: Path,
    output_subdir: str,
    epoch: int,
    global_step: int,
    payload: Any,
) -> None:
    """
    写出 validation diagnostics 本地 artifact。

    输入参数:
        - run_dir: Path, 当前训练 run 根目录
        - output_subdir: str, validation artifact 子目录名
        - epoch: int, 当前 epoch index
        - global_step: int, 当前 global step
        - payload: Any, class CpcDiagnosticsPayload 或等价对象

    输出:
        - None, 在 run_dir 下写 summary/warnings/curves 文件
    """
    # Path, 当前 epoch 的 diagnostics 输出目录
    epoch_dir = run_dir / output_subdir / f"epoch_{int(epoch):06d}"
    # Path, 曲线 CSV 输出目录
    curves_dir = epoch_dir / "curves"
    # Path, histogram 预留目录
    histograms_dir = epoch_dir / "histograms"
    curves_dir.mkdir(parents=True, exist_ok=True)
    histograms_dir.mkdir(parents=True, exist_ok=True)

    scalars = getattr(payload, "scalars", {})
    warnings = getattr(payload, "warnings", ())
    curves = getattr(payload, "curves", {})
    # dict[str, Any], JSON summary 可序列化内容
    summary = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "scalars": {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value for key, value in scalars.items()},
    }
    (epoch_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (epoch_dir / "warnings.json").write_text(
        json.dumps([asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in warnings], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for key, curve in curves.items():
        # tuple[str, ...], 曲线列名
        columns = tuple(getattr(curve, "columns"))
        # tuple[tuple[float, ...], ...], 曲线行
        rows = tuple(getattr(curve, "rows"))
        safe_name = key.replace("/", "__")
        with (curves_dir / f"{safe_name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
