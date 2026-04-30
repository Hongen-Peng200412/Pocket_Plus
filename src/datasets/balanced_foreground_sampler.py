"""
BalancedForegroundSampler
每批次前景平衡采样器(不放回全遍历 + 前景比例递补), 支持多卡 DDP。

主要保证:
- 每个 epoch 内所有样本**不放回**遍历一次(与 PyTorch 默认 RandomSampler 一致)。
- 组装 batch 时, 若当前 batch 中前景比例 < foreground_ratio, 则从前景候补池中取样替换相应数量的背景样本, 以保证最低前景占比。前景候补池在单 epoch 内也不放回。
- 使用 torch.Generator 内部管理随机状态, 不修改全局 RNG, 保证与 train.py 中 SeededEpochRandomSampler 的全局种子机制完全兼容。通过 set_epoch(epoch) 切换 epoch, 同一 epoch + seed 保证相同的样本序列。
- **随机性保证**: 只要 global_batch_size 和 seed 一致, 不论使用多少张卡并行(只要卡数整除 global_batch_size), 全局样本序列完全一致。卡数只影响分发, 不影响序列生成和递补。
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
        global_batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        """
        每批次前景平衡采样器(不放回全遍历 + 前景比例递补), 内置 DistributedSampler 分发逻辑。

        输入参数:
            - foreground_indices: list[int], 保证含正类体素的样本索引(来自 BoxPointDataset.foreground_indices, class_name != "random_BOX")
            - background_indices: list[int], 多数为纯背景的样本索引(来自 BoxPointDataset.background_indices, class_name == "random_BOX")
            - total_size: int, 标量, 数据集总样本数 (len(dataset))
            - foreground_ratio: float, 每个全局 batch 中前景最低占比, 建议值 0.2
            - global_batch_size: int, 标量, 全局 batch 大小 (= per_device_bs × world_size); 前景递补以此为粒度
            - seed: int, 标量, 基准随机种子; 实际种子 = seed + epoch
            - rank: int, 标量, 当前进程在分布式环境中的全局排名 (0 ~ world_size-1)
            - world_size: int, 标量, 分布式环境中的总进程数; 单卡时传 1

        随机性保证:
            只要 global_batch_size 和 seed 一致, 不论 world_size 取多少(只要整除 global_batch_size),
            全局样本序列(递补后)完全一致。world_size 仅影响最终的 round-robin 分发。
        """
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) 必须能被 world_size ({world_size}) 整除"
            )
        # list[int], 保证含正类体素的样本索引
        self.foreground_indices = list(foreground_indices)
        # list[int], 多数为纯背景的样本索引（背景池）
        self.background_indices = list(background_indices)
        # set[int], 前景索引集合, 用于 O(1) 判定
        self._fg_set: set[int] = set(self.foreground_indices)
        # int, 标量, 数据集总样本数
        self.total_size = int(total_size)
        # float, 标量, 每个全局 batch 中前景最低占比
        self.foreground_ratio = float(foreground_ratio)
        # int, 标量, 全局 batch 大小
        self.global_batch_size = int(global_batch_size)
        # int, 标量, 基准随机种子
        self.seed = int(seed)
        # int, 标量, 当前进程排名
        self.rank = int(rank)
        # int, 标量, 总进程数
        self.world_size = int(world_size)
        # int, 标量, 当前 epoch（由 set_epoch 更新）
        self.epoch = 0
        # int, 标量, 每个全局 batch 中前景最少样本数（向上取整）
        self.n_fg_min = math.ceil(self.global_batch_size * self.foreground_ratio)
        # int, 标量, padding 后总样本数, 保证能被 world_size 整除
        self._padded_size = math.ceil(self.total_size / self.world_size) * self.world_size
        # int, 标量, 每个 rank 分到的样本数
        self._per_rank_size = self._padded_size // self.world_size

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
            - int, 标量, 当前 rank 在本 epoch 中将产出的索引数量
        """
        return self._per_rank_size

    def __iter__(self) -> Iterator[int]:
        """
        生成当前 epoch、当前 rank 的索引序列。

        流程:
            1. 对全量样本索引做不放回随机排列(randperm), 保证每个样本恰好出现一次。
               若 total_size 不整除 world_size, 用序列开头的样本 padding 尾部。
            2. 同时对前景索引做不放回随机排列, 作为前景候补池。
            3. 以 global_batch_size 为粒度逐 batch 切片, 检查前景数量是否 >= n_fg_min:
               - 若已满足, 直接保留。
               - 若不足, 从前景候补池取出 deficit 个前景索引, 替换 batch 中相应数量的背景索引(从 batch 末尾的背景位置开始替换)。
            4. batch 内最终顺序再做一次随机打乱。
            5. 拼接所有 batch 得到全局序列。
            6. round-robin 分发: 当前 rank 取 global_indices[rank::world_size]。

        注意:
            - 被前景候补替换掉的背景样本在该 epoch 内不再出现, 换入的前景样本也会多看一次。
            - 前景候补池耗尽后, 后续 batch 不再做递补(退化为纯随机)。
            - 全局序列的生成仅依赖 seed + epoch + global_batch_size, 与 world_size 无关。

        输出:
            - Iterator[int], 长度为 _per_rank_size 的索引序列
        """
        # torch.Generator, 内部独立随机状态，不污染全局 RNG
        g = torch.Generator()
        # int, 标量, 实际种子 = 基准种子 + 当前 epoch，保证跨 epoch 不同但可复现
        g.manual_seed(self.seed + self.epoch)

        # ---- 1. 全量不放回排列 + padding ----
        # torch.Tensor, (total_size,), int64, 全量样本的不放回随机排列
        main_perm = torch.randperm(self.total_size, generator=g)
        if self._padded_size > self.total_size:
            # int, 标量, 需要补齐的样本数
            n_pad = self._padded_size - self.total_size
            # torch.Tensor, (padded_size,), int64, 补齐后的排列(用开头样本填充尾部)
            main_perm = torch.cat([main_perm, main_perm[:n_pad]])

        # ---- 2. 前景候补池 ----
        # torch.Tensor, (len(foreground_indices),), int64, 前景候补池的不放回随机排列
        fg_pool_perm = torch.tensor(self.foreground_indices, dtype=torch.long)
        fg_pool_perm = fg_pool_perm[torch.randperm(len(fg_pool_perm), generator=g)]
        # int, 标量, 前景候补池当前消费位置
        fg_pool_pos = 0

        # ---- 3. 逐全局 batch 做前景递补 ----
        # list[int], 递补后的全局索引序列
        global_indices: list[int] = []
        # int, 标量, main_perm 当前消费位置
        pos = 0
        while pos < self._padded_size:
            # int, 标量, 当前全局 batch 实际大小(最后一个 batch 可能不足 global_batch_size)
            end = min(pos + self.global_batch_size, self._padded_size)
            # list[int], 当前全局 batch 的索引
            batch = main_perm[pos:end].tolist()
            actual_bs = len(batch)

            # 统计当前 batch 中的前景数量
            # int, 标量, 当前 batch 中前景样本数
            n_fg_in_batch = sum(1 for idx in batch if idx in self._fg_set)
            # int, 标量, 需要递补的前景数量
            deficit = max(0, min(self.n_fg_min, actual_bs) - n_fg_in_batch)

            if deficit > 0 and fg_pool_pos < len(fg_pool_perm):
                # 从前景候补池中取出 deficit 个前景索引
                # int, 标量, 本次实际可取出的前景候补数量
                n_take = min(deficit, len(fg_pool_perm) - fg_pool_pos)
                # list[int], 取出的前景候补索引
                fg_take = fg_pool_perm[fg_pool_pos: fg_pool_pos + n_take].tolist()
                fg_pool_pos += n_take

                # 找到 batch 中的背景位置, 从末尾开始替换
                # list[int], batch 中属于背景的位置索引(从末尾到开头)
                bg_positions = [i for i in range(actual_bs - 1, -1, -1) if batch[i] not in self._fg_set]
                for replace_i, fg_idx in zip(bg_positions[:n_take], fg_take):
                    batch[replace_i] = fg_idx

            # batch 内随机打乱
            # list[int], 打乱后的 batch 内排列
            perm = torch.randperm(actual_bs, generator=g).tolist()
            batch = [batch[p] for p in perm]

            global_indices.extend(batch)
            pos = end

        # ---- 4. round-robin 分发到当前 rank ----
        # list[int], 当前 rank 分到的索引子集
        my_indices = global_indices[self.rank::self.world_size]
        return iter(my_indices[:self._per_rank_size])
