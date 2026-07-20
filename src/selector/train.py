"""Selector 专属的单机单设备训练入口。

主要入口:
    - `run_training`: 冻结输入清单、构造 train/validation 数据集、训练 CCLN，并
      以固定 validation CLG 上的总损失选择最佳 checkpoint。
    - `PdbGroupedBatchSampler`: 先按 PDB 分组再切分小批次，配合 Dataset 的 PDB
      聚合归档缓存减少重复解压。

训练只支持显式 CPU 或单张 CUDA 设备。正式运行写 `BEST.ckpt`，试运行写
`TRIAL_BEST.ckpt`；两者都保存完整包装器参数、已解析配置和实际输入通道契约。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

from .dataset import SelectorDataset, freeze_input_clg_list
from .wrapper import SelectorWrapper, build_selector_wrapper_from_config


def seed_everything(selector_seed: int) -> None:
    """
    用一个 ``selector_seed`` 控制 Python、NumPy、CPU 与 CUDA 随机性。

    输入参数:
        - selector_seed: int, 当前 Selector 运行的唯一随机种子

    副作用:
        - 原地设置 Python、NumPy、PyTorch CPU 以及全部可见 CUDA 设备的随机状态
    """
    seed = int(selector_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """
    从 DataLoader 为当前子进程分配的 torch 种子派生 NumPy/Python 种子。

    输入参数:
        - worker_id: int, DataLoader 子进程局部编号；函数不直接使用该值，因为
          PyTorch 已把编号混入 `torch.initial_seed()`

    副作用:
        - 原地设置当前 DataLoader 子进程的 NumPy 与 Python 随机状态
    """
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_sample_to_device(value: Any, device: torch.device) -> Any:
    """
    递归把一个含变长实体表的 CLG 样本中的 tensor 移到指定设备。

    输入参数:
        - value: Any, tensor、dict、list、tuple 或不需要移动的身份标量
        - device: torch.device, 当前训练设备

    输出:
        - moved: Any，与输入容器结构相同；所有 tensor 位于 `device`，字符串、
          数值和闭包索引元组保持原值
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
    从共享配置构造一个固定数据划分的 SelectorDataset。

    输入参数:
        - config: Mapping[str, Any], 完整 Selector 配置
        - frozen_path: Path, 当前运行的 `input_CLG_list.json`
        - split: str, `train` 或 `validation`
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


class PdbGroupedBatchSampler(Sampler[list[int]]):
    """
    按 PDB 成组打乱 CLG，避免小缓存反复解压同一个聚合归档。

    输入参数:
        - records: Sequence[Any], `SelectorInputRecord` 序列
        - batch_size: int, 每个小批次的 CLG 数量上限
        - seed: int, 控制 PDB 分组顺序和组内 CLG 顺序的基础随机种子
    """

    def __init__(
        self,
        records: Sequence[Any],
        batch_size: int,
        seed: int,
    ) -> None:
        """
        按 `(split, pdb_id)` 建立不跨 PDB 的样本下标组。

        输入参数:
            - records: Sequence[Any], 与 Dataset 下标同序的冻结 CLG 身份记录
            - batch_size: int, 每个小批次最多包含的 CLG 数，必须为正
            - seed: int, 每轮可复现打乱使用的基础随机种子

        分组语义:
            - 同一 PDB 的下标保持在同一组内，一个小批次不会跨 PDB。
            - PDB 首次出现顺序只用于建立组；每轮实际组顺序和组内顺序由
              `(seed, epoch)` 共同决定。
        """
        if int(batch_size) <= 0:
            raise ValueError("Selector batch_size 必须为正。")
        groups: dict[tuple[str, str], list[int]] = {}
        for index, record in enumerate(records):
            groups.setdefault((str(record.split), str(record.pdb_id)), []).append(index)
        self.groups = tuple(tuple(indices) for indices in groups.values())
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch，使 PDB 顺序与 PDB 内 CLG 顺序可复现地变化。

        输入参数:
            - epoch: int, 当前训练 epoch
        """
        self.epoch = int(epoch)

    def __iter__(self):
        """
        逐 PDB 产出小批次；一个小批次不跨 PDB。

        输出:
            - batches: Iterator[list[int]], 每项为同一 PDB 的 Dataset 局部下标列表
        """
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        group_order = rng.permutation(len(self.groups)).tolist()
        for group_index in group_order:
            indices = np.asarray(self.groups[group_index], dtype=np.int64)
            indices = indices[rng.permutation(indices.size)]
            for begin in range(0, indices.size, self.batch_size):
                yield indices[begin : begin + self.batch_size].tolist()

    def __len__(self) -> int:
        """
        返回每个 PDB 独立切分小批次后的总批次数。

        输出:
            - count: int, 当前 sampler 将产生的 batch 数
        """
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups)


def _validate_source_dimensions(
    train_dimensions: Mapping[str, Mapping[str, int]],
    validation_dimensions: Mapping[str, Mapping[str, int]],
) -> None:
    """
    确认 train/validation 的实际 A/P 字段和通道完全一致。

    输入参数:
        - train_dimensions: Mapping, train 首个样本解析的 A/P 字段及通道数
        - validation_dimensions: Mapping, validation 首个样本解析的 A/P 字段及通道数

    异常:
        - 字段集合或任一末维通道数不一致时立即抛出异常，避免模型在后续小批次
          才暴露输入契约漂移
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
    在固定 validation CLG 清单上计算按 CLG 数加权的平均损失。

    输入参数:
        - wrapper: SelectorWrapper, 当前模型
        - loader: DataLoader, 不打乱顺序的 validation 数据加载器
        - device: torch.device, 当前执行设备

    输出:
        - metrics: dict[str, float]，各小批次损失先乘当前 CLG 数累加，再除以
          validation CLG 总数；因此最后一个不足整批的小批次不会被过度加权
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
        - payload: dict[str, Any], `state_dict`、已解析配置、epoch 和 validation 损失

    落盘:
        - 先在目标目录写唯一临时文件，再用同文件系统原子替换正式路径；
          成功返回时 `path` 是完整 checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_training(config: DictConfig) -> dict[str, Any]:
    """
    冻结输入清单，并以 validation 总损失最小选择 Selector 最佳 checkpoint。

    输入参数:
        - config: DictConfig, 从 `configs/selector/*.yaml` 读取的完整显式配置

    输出:
        - result: dict[str, Any]，包含:
            - `checkpoint`: str, 实际 checkpoint 路径
            - `checkpoint_kind`: str, `formal_best` 或 `trial_best`
            - `checkpoint_epoch`: int, 最佳 validation loss 对应 epoch
            - `validation_total_loss`: float, 最佳 validation total loss
            - `source_dimensions`: dict[str, dict[str, int]], A/P 来源末维通道映射

    落盘:
        - `input_CLG_list.json`: 当前运行不可变的 PDB/CLG 输入清单。
        - `resolved_config.yaml`: 含实际 A/P 来源通道和冻结清单路径的完整配置。
        - `checkpoints/BEST.ckpt` 或 `TRIAL_BEST.ckpt`: validation 总损失严格下降时
          原子替换；不会按训练损失选择。
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

    # DataLoader 主生成器和子进程派生种子均从同一个 Selector 随机种子开始。
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "num_workers": int(resolved["train"]["num_workers"]),
        "collate_fn": SelectorDataset.collate_fn,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "pin_memory": bool(resolved["train"]["pin_memory"]),
    }
    train_batch_sampler = PdbGroupedBatchSampler(
        train_dataset.records,
        batch_size=int(resolved["train"]["batch_size"]),
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        persistent_workers=int(resolved["train"]["num_workers"]) > 0,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(resolved["train"]["batch_size"]),
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

    # 只依据固定 validation 清单上的 `total_loss` 严格下降更新最佳 checkpoint。
    best_loss = float("inf")
    best_epoch = -1
    checkpoint_name = "BEST.ckpt" if bool(resolved["data"]["formal_run"]) else "TRIAL_BEST.ckpt"
    best_path = run_dir / "checkpoints" / checkpoint_name
    for epoch in range(int(resolved["train"]["max_epochs"])):
        train_batch_sampler.set_epoch(epoch)
        wrapper.train()
        for batch in train_loader:
            samples = move_sample_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = wrapper.compute_loss(samples, None)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), float(resolved["optimizer"]["gradient_clip_norm"]))
            optimizer.step()
        # 每个 epoch 完成后在不打乱的固定 validation CLG 清单上完整评估一次。
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
    解析一个完整 Selector YAML 配置并启动训练。

    输入参数:
        - argv: Sequence[str] | None, CLI 参数；None 表示读取 sys.argv

    输出:
        - exit_code: int, 训练完成并输出结果摘要时为 0；配置、数据或训练异常向上
          抛出，由进程返回非零状态
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
