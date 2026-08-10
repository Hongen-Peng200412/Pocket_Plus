# -*- coding: utf-8 -*-
"""把 Stage1 逐 PDB 请求切成物理 batch，并按完整 batch 分配给 DDP rank."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from src.datasets.stage1_requests import Stage1TrainingRequestSet


class Stage1PdbBatchSampler(Sampler[list[int]]):
    """生成 PDB 连续、前景比例稳定且不补齐尾部的物理 batch.

    batch_size 直接使用训练入口已经确定的单 rank 物理 batch，不在此处
    对齐或改写。请求不足一个完整 DDP 物理步时直接丢弃，不复制请求，也不
    产生缩小的 batch。
    """

    def __init__(
        self,
        request_source: Stage1TrainingRequestSet,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        batch_size = int(batch_size)
        num_replicas = int(num_replicas)
        rank = int(rank)
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if num_replicas <= 0:
            raise ValueError("num_replicas 必须为正整数。")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank 必须位于 [0, num_replicas)。")

        self.request_source = request_source
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = int(request_source.epoch)

    def set_epoch(self, epoch: int) -> None:
        """同步重建请求身份和本训练周期的 batch 顺序."""

        self.epoch = int(epoch)
        self.request_source.set_epoch(self.epoch)

    def _ordered_indices(self) -> list[int]:
        """按 PDB 块和物理 batch 片段排列当前请求下标."""

        # tuple[ResolvedStage1Crop, ...]，请求源已经按随机化后的 PDB 顺序保存连续 PDB 块。
        requests = self.request_source.requests
        # 末尾常量 1 把 batch 内部排列随机流与请求生成随机流分开。
        rng = np.random.default_rng(
            np.random.SeedSequence([self.request_source.seed, self.epoch, 1])
        )
        ordered_indices: list[int] = []
        block_start = 0
        while block_start < len(requests):
            pdb_id = requests[block_start].pdb_id
            block_end = block_start + 1
            while block_end < len(requests) and requests[block_end].pdb_id == pdb_id:
                block_end += 1

            # 当前 PDB 共 N 个请求，其中 F 个 center 或 bias 请求是前景 BOX。
            # int64, (F,)，数值索引 requests。
            foreground = np.asarray(
                [
                    index
                    for index in range(block_start, block_end)
                    if requests[index].role in {"center", "bias"}
                ],
                dtype=np.int64,
            )
            # int64, (N-F,)，数值索引 requests；其余 context 请求是背景 BOX。
            background = np.asarray(
                [
                    index
                    for index in range(block_start, block_end)
                    if requests[index].role == "context"
                ],
                dtype=np.int64,
            )
            rng.shuffle(foreground)
            rng.shuffle(background)

            foreground_cursor = 0
            background_cursor = 0
            foreground_left = int(foreground.size)
            request_left = block_end - block_start
            while request_left > 0:
                # 当前 PDB 先填满已有物理 batch 的剩余槽位，再继续进入后续物理 batch。
                batch_slots = self.batch_size - (len(ordered_indices) % self.batch_size)
                segment_size = min(request_left, batch_slots)
                # 当前 PDB 跨物理 batch 延续同一组剩余计数。
                # 本片段按约定的非负数四舍五入规则分配前景 BOX。
                foreground_slots = math.floor(
                    segment_size * foreground_left / request_left + 0.5
                )
                # set[int]，本片段中读取前景请求的位置；其余位置读取背景请求。
                foreground_positions = set(
                    rng.choice(
                        segment_size,
                        size=foreground_slots,
                        replace=False,
                    ).tolist()
                )
                for position in range(segment_size):
                    if position in foreground_positions:
                        ordered_indices.append(int(foreground[foreground_cursor]))
                        foreground_cursor += 1
                    else:
                        ordered_indices.append(int(background[background_cursor]))
                        background_cursor += 1
                foreground_left -= foreground_slots
                request_left -= segment_size

            block_start = block_end
        return ordered_indices

    def __iter__(self) -> Iterator[list[int]]:
        ordered_indices = self._ordered_indices()
        # 一个完整 DDP 物理步包含每个 rank 各一个 batch，共 batch_size * num_replicas 个请求。
        distributed_step_size = self.batch_size * self.num_replicas
        # 只保留每个 DDP rank 都能取得完整物理 batch 的前缀；尾部请求不用于补齐或重复。
        consumed_count = (
            len(ordered_indices) // distributed_step_size
        ) * distributed_step_size
        complete_batch_count = consumed_count // self.batch_size
        # 物理 batch 按 rank 交错分配；同一训练步的 rank 0 batch 在 rank 1 batch 之前。
        for physical_batch_index in range(
            self.rank,
            complete_batch_count,
            self.num_replicas,
        ):
            start = physical_batch_index * self.batch_size
            end = start + self.batch_size
            yield ordered_indices[start:end]

    def __len__(self) -> int:
        distributed_step_size = self.batch_size * self.num_replicas
        return len(self.request_source) // distributed_step_size
