"""Selector 专属的单机/单卡训练入口。"""

from __future__ import annotations

import argparse
import os
import random
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import SelectorDataset, freeze_input_clg_list
from .wrapper import SelectorWrapper, build_selector_wrapper_from_config


def seed_everything(selector_seed: int) -> None:
    """
    用一个 selector_seed 控制 Python、NumPy、CPU 与 CUDA 初始化随机性。

    输入参数:
        - selector_seed: int, 当前 Selector run 的唯一随机 seed

    输出:
        - None，原地设置各随机数生成器
    """
    seed = int(selector_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """
    从 DataLoader 为当前 worker 分配的 torch seed 派生 NumPy/Python seed。

    输入参数:
        - worker_id: int, DataLoader worker 局部编号；只用于函数签名

    输出:
        - None，原地设置当前 worker 随机状态
    """
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_sample_to_device(value: Any, device: torch.device) -> Any:
    """
    递归把一个 ragged CLG sample 的 tensor 移到训练设备。

    输入参数:
        - value: Any, tensor、dict、list、tuple 或不需移动的 identity 标量
        - device: torch.device, 当前训练设备

    输出:
        - moved: Any，与输入结构相同，所有 tensor 位于 device
    """
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_sample_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_sample_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_sample_to_device(item, device) for item in value)
    return value


def _build_dataset(config: Mapping[str, Any], frozen_path: Path, split: str, require_oracle: bool) -> SelectorDataset:
    """
    从共享 config 构造一个固定 split 的 SelectorDataset。

    输入参数:
        - config: Mapping[str,Any], 完整 Selector 配置
        - frozen_path: Path, 当前 run 的 input_CLG_list.json
        - split: str, train 或 validation
        - require_oracle: bool, 是否读取 overlap 并现场生成监督

    输出:
        - dataset: SelectorDataset, 启动后长度固定的 CLG 数据集
    """
    return SelectorDataset(
        input_clg_list_path=frozen_path,
        stage1_outputs_root=config["paths"]["stage1_outputs_root"],
        upstream_root=config["paths"]["upstream_root"],
        split=split,
        lambda_count=float(config["data"]["lambda_count"]),
        require_oracle=require_oracle,
        density_clip_percentile=tuple(float(value) for value in config["data"]["density_clip_percentile"]),
        pdb_cache_size=int(config["data"]["pdb_cache_size"]),
    )


def _validate_source_dimensions(
    train_dimensions: Mapping[str, Mapping[str, int]],
    validation_dimensions: Mapping[str, Mapping[str, int]],
) -> None:
    """
    确认 train/validation 的实际 A/P 字段和通道完全一致。

    输入参数:
        - train_dimensions: Mapping, train 首样本解析的来源维度
        - validation_dimensions: Mapping, validation 首样本解析的来源维度

    输出:
        - None；不一致时 fail-fast，避免模型在后续 batch 才暴露 schema 漂移
    """
    normalized_train = {modality: dict(values) for modality, values in train_dimensions.items()}
    normalized_validation = {modality: dict(values) for modality, values in validation_dimensions.items()}
    if normalized_train != normalized_validation:
        raise ValueError(
            "train/validation Selector source dimensions 不一致: "
            f"train={normalized_train}, validation={normalized_validation}"
        )


def _run_validation(
    wrapper: SelectorWrapper,
    loader: DataLoader[list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, float]:
    """
    在固定 validation CLG 清单上计算逐 batch 加权平均 loss。

    输入参数:
        - wrapper: SelectorWrapper, 当前模型
        - loader: DataLoader, validation loader，不 shuffle
        - device: torch.device, 当前执行设备

    输出:
        - metrics: dict[str,float]，按 CLG 数加权的各项 validation loss
    """
    wrapper.eval()
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            samples = move_sample_to_device(batch, device)
            losses = wrapper.compute_loss(samples, None)
            batch_size = len(samples)
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size
            count += batch_size
    if count == 0:
        raise ValueError("固定 validation SelectorDataset 不能为空。")
    return {name: value / count for name, value in sums.items()}


def _atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """
    将 Selector checkpoint 写入同目录临时文件后原子发布。

    输入参数:
        - path: Path, 正式 checkpoint 路径
        - payload: dict[str,Any], state_dict、resolved config、epoch 与 validation loss

    输出:
        - None，成功后 path 完整存在
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_training(config: DictConfig) -> dict[str, Any]:
    """
    冻结输入清单并以 validation total loss 最小选择 Selector BEST。

    输入参数:
        - config: DictConfig, 从 `configs/selector/*.yaml` 读取的完整显式配置

    输出:
        - result: dict[str,Any]，包含实际 checkpoint 路径/种类、epoch、validation total loss
          与 source dimensions；试跑不会在结果字段中冒充正式 BEST
    """
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    run_dir = Path(str(resolved["paths"]["selector_run_dir"]))
    seed = int(resolved["data"]["selector_seed"])
    seed_everything(seed)
    frozen_path = freeze_input_clg_list(
        stage1_outputs_root=resolved["paths"]["stage1_outputs_root"],
        selector_run_dir=run_dir,
        stage1_model_name=str(resolved["stage1_model_name"]),
        split_order=resolved["data"]["split_order"],
        input_clg_list_path=resolved["paths"]["input_CLG_list_path"],
        formal_run=bool(resolved["data"]["formal_run"]),
        expected_validation_pdb_ids_path=resolved["paths"]["expected_validation_pdb_ids_path"],
    )
    train_dataset = _build_dataset(resolved, frozen_path, "train", True)
    validation_dataset = _build_dataset(resolved, frozen_path, "validation", True)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Selector train/validation 冻结清单均必须非空。")
    train_dimensions = train_dataset.source_dimensions()
    validation_dimensions = validation_dataset.source_dimensions()
    _validate_source_dimensions(train_dimensions, validation_dimensions)

    resolved["model"]["resolved_source_dimensions"] = train_dimensions
    resolved["paths"]["frozen_input_CLG_list_path"] = str(frozen_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(resolved), run_dir / "resolved_config.yaml")

    wrapper = build_selector_wrapper_from_config(resolved, train_dimensions)
    device_name = str(resolved["train"]["device"])
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("train.device 只允许显式 cpu 或 cuda。")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前进程没有可用 GPU。")
    device = torch.device(device_name)
    wrapper.to(device)

    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "batch_size": int(resolved["train"]["batch_size"]),
        "num_workers": int(resolved["train"]["num_workers"]),
        "collate_fn": SelectorDataset.collate_fn,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "pin_memory": bool(resolved["train"]["pin_memory"]),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        persistent_workers=int(resolved["train"]["num_workers"]) > 0,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        persistent_workers=int(resolved["train"]["num_workers"]) > 0,
        **loader_arguments,
    )
    optimizer = torch.optim.AdamW(
        wrapper.parameters(),
        lr=float(resolved["optimizer"]["learning_rate"]),
        weight_decay=float(resolved["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(resolved["optimizer"]["plateau_factor"]),
        patience=int(resolved["optimizer"]["plateau_patience"]),
        threshold=float(resolved["optimizer"]["plateau_threshold"]),
        threshold_mode="abs",
    )

    best_loss = float("inf")
    best_epoch = -1
    checkpoint_name = "BEST.ckpt" if bool(resolved["data"]["formal_run"]) else "TRIAL_BEST.ckpt"
    best_path = run_dir / "checkpoints" / checkpoint_name
    for epoch in range(int(resolved["train"]["max_epochs"])):
        wrapper.train()
        for batch in train_loader:
            samples = move_sample_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = wrapper.compute_loss(samples, None)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), float(resolved["optimizer"]["gradient_clip_norm"]))
            optimizer.step()
        validation_metrics = _run_validation(wrapper, validation_loader, device)
        validation_total = float(validation_metrics["total_loss"])
        scheduler.step(validation_total)
        if validation_total < best_loss:
            best_loss = validation_total
            best_epoch = epoch
            _atomic_save_checkpoint(
                best_path,
                {
                    "state_dict": wrapper.state_dict(),
                    "epoch": epoch,
                    "validation_total_loss": validation_total,
                    "validation_metrics": validation_metrics,
                    "resolved_config": resolved,
                    "source_dimensions": train_dimensions,
                },
            )

    return {
        "checkpoint": str(best_path),
        "checkpoint_kind": "formal_best" if bool(resolved["data"]["formal_run"]) else "trial_best",
        "checkpoint_epoch": best_epoch,
        "validation_total_loss": best_loss,
        "source_dimensions": train_dimensions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    解析单个 Selector YAML 并启动训练。

    输入参数:
        - argv: Sequence[str] | None, CLI 参数；None 表示读取 sys.argv

    输出:
        - exit_code: int, 成功为 0
    """
    parser = argparse.ArgumentParser(description="训练 AdaLigand Stage1 CCLN Selector")
    parser.add_argument("--config", required=True, help="完整 Selector YAML 配置路径")
    arguments = parser.parse_args(argv)
    config = OmegaConf.load(arguments.config)
    result = run_training(config)
    print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
