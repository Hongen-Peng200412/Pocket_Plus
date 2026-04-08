"""
BalancedForegroundSampler
nnU-Net 风格的每批次前景平衡采样器。

主要保证:
- 每个 batch 中前景样本数量 n_fg = ceil(batch_size * foreground_ratio)，不足时从前景池有放回抽取。剩余 n_bg = batch_size - n_fg 从完整样本池（含前景+背景）随机抽取。
- 使用 torch.Generator 内部管理随机状态，不修改全局 RNG，保证与 train.py 中 SeededEpochRandomSampler 的全局种子机制完全兼容。且通过 set_epoch(epoch) 切换 epoch，同一 epoch + seed 保证相同的样本序列。
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class BalancedForegroundSampler(Sampler[int]):
    def __init__(
        self,
        foreground_indices: list[int],
        background_indices: list[int],
        total_size: int,
        foreground_ratio: float,
        batch_size: int,
        seed: int,
    ) -> None:
        """
        nnU-Net 风格的每批次前景平衡采样器。

        输入参数:
            - foreground_indices: list[int], 保证含正类体素的样本索引(来自 BoxPointDataset.foreground_indices, class_name != "random_BOX")
            - background_indices: list[int], 多数为纯背景的样本索引(来自 BoxPointDataset.background_indices, class_name == "random_BOX")
            - total_size: int, 标量, 数据集总样本数 (len(dataset))
            - foreground_ratio: float, 每 batch 中前景最低占比, 建议值 0.33
            - batch_size: int, 标量, 每个 batch 的大小
            - seed: int, 标量, 基准随机种子; 实际种子 = seed + epoch
        """
        # list[int], 保证含正类体素的样本索引
        self.foreground_indices = list(foreground_indices)
        # list[int], 多数为纯背景的样本索引（背景池）
        self.background_indices = list(background_indices)
        # list[int], 全量样本索引池（前景 + 背景）
        self.all_indices = list(range(total_size))
        # int, 标量, 数据集总样本数
        self.total_size = int(total_size)
        # float, 标量, 每 batch 中前景最低占比
        self.foreground_ratio = float(foreground_ratio)
        # int, 标量, 每个 batch 的大小
        self.batch_size = int(batch_size)
        # int, 标量, 基准随机种子
        self.seed = int(seed)
        # int, 标量, 当前 epoch（由 set_epoch 更新）
        self.epoch = 0
        # int, 标量, 每 batch 中前景样本数（向上取整）
        self.n_fg_per_batch = math.ceil(self.batch_size * self.foreground_ratio)

    def set_epoch(self, epoch: int) -> None:
        """
        切换 epoch，使每个 epoch 的随机顺序不同但可复现。

        输入参数:
            - epoch: int, 标量, 当前 epoch 编号
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """
        输出:
            - int, 标量, 索引序列总长度 = total_size
        """
        return self.total_size

    def __iter__(self) -> Iterator[int]:
        """
        生成各 batch 的索引序列。

        每 batch 的构成:
            1. n_fg 个索引从前景池（有放回）随机抽取
            2. n_bg 个索引从全量样本池（有放回）随机抽取
            3. batch 内随机打乱

        输出:
            - Iterator[int], 长度为 total_size 的索引序列
        """
        # torch.Generator, 内部独立随机状态，不污染全局 RNG
        g = torch.Generator()
        # int, 标量, 实际种子 = 基准种子 + 当前 epoch，保证跨 epoch 不同但可复现
        g.manual_seed(self.seed + self.epoch)

        # int, 标量, 本 epoch 需生成的 batch 数（向上取整）
        num_batches = math.ceil(self.total_size / self.batch_size)

        # torch.Tensor, (len(foreground_indices),), 前景池张量，用于快速随机采样
        fg_pool = torch.tensor(self.foreground_indices, dtype=torch.long)
        # torch.Tensor, (total_size,), 全量样本池张量
        all_pool = torch.tensor(self.all_indices, dtype=torch.long)

        # list[int], 拼接后的完整索引序列
        indices: list[int] = []
        for _ in range(num_batches):
            # int, 标量, 本 batch 中前景数量
            n_fg = self.n_fg_per_batch
            # int, 标量, 本 batch 中非强制前景数量
            n_bg = self.batch_size - n_fg

            # torch.Tensor, (n_fg,), 从前景池有放回随机采样的前景索引
            fg_idx = torch.randint(0, len(fg_pool), (n_fg,), generator=g)
            fg_samples = fg_pool[fg_idx]
            # torch.Tensor, (n_bg,), 从全量样本池有放回随机采样的索引
            bg_idx = torch.randint(0, len(all_pool), (n_bg,), generator=g)
            bg_samples = all_pool[bg_idx]

            # torch.Tensor, (batch_size,), 当前 batch 的全部索引
            batch_indices = torch.cat([fg_samples, bg_samples])
            # torch.Tensor, (batch_size,), batch 内随机打乱
            shuffle_perm = torch.randperm(self.batch_size, generator=g)
            batch_indices = batch_indices[shuffle_perm]
            indices.extend(batch_indices.tolist())

        # 截断到 total_size，丢弃最后一个不满 batch 可能多出的部分
        return iter(indices[: self.total_size])
