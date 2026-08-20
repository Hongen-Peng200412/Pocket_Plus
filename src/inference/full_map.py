# -*- coding: utf-8 -*-
"""用无 padding 80³ 滑窗生成完整图配体概率.

主要入口 :func:`infer_full_map` 在同一个调用中重叠 CPU 请求物化, 页锁定
H2D, GPU 前向, 异步 D2H 与有序 CPU 融合. 并行不改变窗口顺序, batch
边界或 float32 Gaussian 累加顺序.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FullMapResult:
    """保存一个 PDB 的完整概率图与几何.

    字段:
        - probability_map: float32 `(D, H, W)`, 完整 ZYX 网格融合概率.
        - origin_xyz: float32 `(3,)`, 完整图世界 XYZ 角点坐标, 单位 Å.
        - voxel_size_xyz: float32 `(3,)`, 世界 XYZ 体素尺寸, 单位 Å/voxel.
        - window_count: int, 实际执行的无 padding 80³ 窗口数.
        - wall_seconds: float, 当前 PDB 完整图调用的墙钟秒数.
        - materialize_wait_seconds: float, 主线程等待 CPU 请求物化 future 的累计秒数.
        - fusion_wait_seconds: float, 主线程因融合队列背压或最终收口等待的累计秒数.
    """

    probability_map: np.ndarray
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    window_count: int
    wall_seconds: float
    materialize_wait_seconds: float
    fusion_wait_seconds: float


def window_starts_zyx(
    full_shape_zyx: Sequence[int],
    window_shape_zyx: Sequence[int],
    stride_zyx: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """按 Z, Y, X 字典序生成覆盖完整图边界的无 padding 窗口起点.

    输入参数:
        - full_shape_zyx: Sequence[int], 完整图三轴长度 `(D, H, W)`.
        - window_shape_zyx: Sequence[int], 滑窗三轴长度 `(D_window, H_window, W_window)`.
        - stride_zyx: Sequence[int], 相邻窗口在 Z, Y, X 三轴的显式步长.

    返回值:
        - starts: tuple[tuple[int, int, int], ...], 每项是一个窗口的完整图 ZYX 起点.

    首项为 `(0, 0, 0)`. 每轴末项强制等于 `full_length - window_length`.
    """

    full_shape = tuple(int(value) for value in full_shape_zyx)
    window_shape = tuple(int(value) for value in window_shape_zyx)
    stride = tuple(int(value) for value in stride_zyx)
    axes: list[tuple[int, ...]] = []
    for length, window, step in zip(full_shape, window_shape, stride):
        if length < window or step <= 0:
            raise ValueError(
                f"完整图轴长度必须不小于窗口且 stride 为正: {(length, window, step)}."
            )
        values = list(range(0, length - window + 1, step))
        if values[-1] != length - window:
            values.append(length - window)
        axes.append(tuple(values))
    return tuple((z, y, x) for z in axes[0] for y in axes[1] for x in axes[2])


def gaussian_window_weight(
    window_shape_zyx: Sequence[int],
    sigma: float,
) -> np.ndarray:
    """计算规范化 ZYX 坐标上的 float32 三维 Gaussian 融合权重.

    输入参数:
        - window_shape_zyx: Sequence[int], 滑窗三轴长度 `(D_window, H_window, W_window)`.
        - sigma: float, 规范化坐标中的标准差; 每轴从 -1 到 1 均匀取样, 因此正式值 0.5 不表示 40 个体素.

    返回值:
        - weight: float32 `(D_window, H_window, W_window)`, 每个滑窗体素参与完整图融合的正权重.
    """

    shape = tuple(int(value) for value in window_shape_zyx)
    axes = [np.linspace(-1.0, 1.0, length, dtype=np.float32) for length in shape]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    squared_radius = z * z + y * y + x * x
    return np.exp(-squared_radius / np.float32(2.0 * float(sigma) ** 2)).astype(
        np.float32, copy=False
    )


# ================================================================================================


def infer_full_map(
    dataset: Any,
    collator: Any,
    wrapper: Any,
    pdb_id: str,
    device: str,
    stride_zyx: Sequence[int],
    sigma: float,
    window_batch_size: int,
    window_workers: int,
    prefetch_batches: int,
    precision: str,
    pending_fusion_batches: int,
) -> FullMapResult:
    """用有界 CPU/GPU 流水生成一个 PDB 的完整图配体概率.

    输入参数:
        - dataset: Stage1Dataset, 提供 `full_map_context(pdb_id)` 和 `materialize_request(ResolvedStage1Crop)`.
        - collator: callable, 把同一 PDB 的 80³ 请求列表拼成模型 batch.
        - wrapper: torch.nn.Module, `forward_voxel_probability(batch)` 返回 `(B_window, 1, 80, 80, 80)` 配体 logits.
        - pdb_id: str, 当前完整图所属 PDB 标识.
        - device: str, 模型前向设备, 例如 `cuda:0` 或 `cpu`.
        - stride_zyx: Sequence[int], 无 padding 滑窗在完整图 ZYX 三轴的显式步长.
        - sigma: float, `gaussian_window_weight()` 使用的规范化 Gaussian 标准差.
        - window_batch_size: int, 一次模型前向的滑窗数量.
        - window_workers: int, CPU 请求物化线程数.
        - prefetch_batches: int, 尚未送入模型的最大预取 batch 数.
        - precision: str, GPU autocast 精度, 正式取 bf16 或 float16; 其他值禁用混合精度.
        - pending_fusion_batches: int, 已完成模型前向但尚未完成 CPU 融合的最大 batch 数.

    返回值:
        - result: FullMapResult, 保存 float32 完整图概率, 世界几何, 窗口数量和三类墙钟计时.

    CPU 线程提前物化 batch, 当前线程独占 GPU, 单独融合线程按提交顺序等待 CUDA event. 并行不改变窗口顺序, batch 边界或 float32 累加顺序.
    """

    from src.datasets.stage1_requests import ResolvedStage1Crop

    started_at = time.perf_counter()
    materialize_wait_seconds = 0.0
    fusion_wait_seconds = 0.0
    # full_shape 按完整图 ZYX 排列; voxel_size 和 origin 按世界 XYZ 排列, 单位分别为 Å/voxel 和 Å.
    full_shape, voxel_size, origin, _ = dataset.full_map_context(pdb_id)
    # starts 按 ZYX 字典序保存完整图窗口起点; weight 是全部 80³ 窗口共用的融合权重.
    starts = window_starts_zyx(full_shape, (80, 80, 80), stride_zyx)
    weight = gaussian_window_weight((80, 80, 80), sigma)
    # float32, (D, H, W), 分别累计加权配体概率和 Gaussian 权重.
    weighted_sum = np.zeros(full_shape, dtype=np.float32)
    weight_sum = np.zeros(full_shape, dtype=np.float32)
    start_batches = tuple(
        starts[offset : offset + int(window_batch_size)]
        for offset in range(0, len(starts), int(window_batch_size))
    )

    def materialize_batch(
        batch_starts: Sequence[tuple[int, int, int]],
    ) -> tuple[dict[str, Any], tuple[tuple[int, int, int], ...]]:
        """在线程池中物化一批完整图 ZYX 起点对应的 80³ 请求.

        输入参数:
            - batch_starts: 长度 B_window 的完整图 ZYX 起点序列; 顺序也是返回 batch 的窗口顺序.

        返回值:
            - batch: `dataset.collate_fn` 拼装的 Stage1 模型批次; 字段契约由 `src/datasets/README.md` 的 Stage1 batch 表定义.
            - batch_starts: 长度 B_window 的 ZYX 起点元组, 供融合线程把每张概率窗放回完整图.

        本模块不改 batch 字段并直接交给 wrapper; CUDA 推理时张量已锁页.
        """

        requests = [
            ResolvedStage1Crop(
                pdb_id=pdb_id,
                box_start_zyx=start,
                require_targets=False,
                role="sliding",
            )
            for start in batch_starts
        ]
        batch = collator([dataset.materialize_request(request) for request in requests])
        if torch.device(device).type == "cuda":
            batch = {
                name: value.pin_memory() if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
        return batch, tuple(batch_starts)

    def fuse_batch(
        host_probability: torch.Tensor,
        ready_event: torch.cuda.Event | None,
        batch_starts: tuple[tuple[int, int, int], ...],
    ) -> None:
        """在线程池中等待一批 D2H 完成, 再按窗口顺序融合概率.

        输入参数:
            - host_probability: float32 `(B_window, 80, 80, 80)` CPU 张量, 保存每个滑窗的配体概率.
            - ready_event: CUDA event 或 None; 非 None 时先等待当前批次的异步 D2H 完成.
            - batch_starts: 长度 B_window 的完整图 ZYX 起点, 与 `host_probability` 第一维逐窗对齐.

        本函数按输入顺序原位累加外层 `weighted_sum` 和 `weight_sum`, 不返回新对象.
        """

        if ready_event is not None:
            ready_event.synchronize()
        probabilities = host_probability.numpy()
        for probability, (z, y, x) in zip(probabilities, batch_starts):
            slices = (
                slice(z, z + 80),
                slice(y, y + 80),
                slice(x, x + 80),
            )
            weighted_sum[slices] += probability * weight
            weight_sum[slices] += weight

    # 请求物化, GPU 前向和 CPU Gaussian 融合形成三个相邻阶段; 两个队列都保持窗口顺序.
    materializer = ThreadPoolExecutor(
        max_workers=int(window_workers),
        thread_name_prefix="stage1-window",
    )
    fusion_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="stage1-fusion",
    )
    prepared: deque[Future[tuple[dict[str, Any], tuple[tuple[int, int, int], ...]]]] = (
        deque()
    )
    pending_fusions: deque[Future[None]] = deque()
    next_batch = 0
    try:
        while next_batch < min(int(prefetch_batches), len(start_batches)):
            prepared.append(
                materializer.submit(materialize_batch, start_batches[next_batch])
            )
            next_batch += 1

        batch_device = torch.device(device)
        use_cuda = batch_device.type == "cuda"
        with torch.inference_mode():
            while prepared:
                wait_started_at = time.perf_counter()
                cpu_batch, batch_starts = prepared.popleft().result()
                materialize_wait_seconds += time.perf_counter() - wait_started_at
                if next_batch < len(start_batches):
                    prepared.append(
                        materializer.submit(
                            materialize_batch,
                            start_batches[next_batch],
                        )
                    )
                    next_batch += 1
                if use_cuda:
                    model_batch = {
                        key: (
                            value.to(batch_device, non_blocking=True)
                            if torch.is_tensor(value)
                            else value
                        )
                        for key, value in cpu_batch.items()
                    }
                    autocast_dtype = (
                        torch.bfloat16 if precision == "bf16" else torch.float16
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=precision in {"bf16", "float16"},
                    ):
                        logits = wrapper.forward_voxel_probability(model_batch)
                    probability = torch.sigmoid(logits[:, 0]).to(torch.float32)
                    while len(pending_fusions) >= int(pending_fusion_batches):
                        wait_started_at = time.perf_counter()
                        pending_fusions.popleft().result()
                        fusion_wait_seconds += time.perf_counter() - wait_started_at
                    # 概率异步复制到锁页 CPU 张量, 融合线程等待同一 CUDA stream 的 event.
                    host_probability = torch.empty(
                        probability.shape,
                        dtype=torch.float32,
                        device="cpu",
                        pin_memory=True,
                    )
                    host_probability.copy_(probability, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(batch_device))
                    pending_fusions.append(
                        fusion_executor.submit(
                            fuse_batch,
                            host_probability,
                            event,
                            batch_starts,
                        )
                    )
                else:
                    model_batch = {
                        key: (
                            value.to(batch_device) if torch.is_tensor(value) else value
                        )
                        for key, value in cpu_batch.items()
                    }
                    logits = wrapper.forward_voxel_probability(model_batch)
                    host_probability = torch.sigmoid(logits[:, 0]).to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    pending_fusions.append(
                        fusion_executor.submit(
                            fuse_batch,
                            host_probability,
                            None,
                            batch_starts,
                        )
                    )
        while pending_fusions:
            wait_started_at = time.perf_counter()
            pending_fusions.popleft().result()
            fusion_wait_seconds += time.perf_counter() - wait_started_at
    finally:
        materializer.shutdown(wait=True, cancel_futures=True)
        fusion_executor.shutdown(wait=True, cancel_futures=True)

    # 零权重体素会使概率图无定义, 必须在产物发布前停止.
    if np.any(weight_sum <= 0.0):
        raise RuntimeError(f"{pdb_id}: 滑窗没有覆盖完整图的全部体素.")
    probability_map = np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros_like(weighted_sum),
        where=weight_sum > 0.0,
    )
    return FullMapResult(
        probability_map=probability_map,
        origin_xyz=np.asarray(origin, dtype=np.float32),
        voxel_size_xyz=np.asarray(voxel_size, dtype=np.float32),
        window_count=len(starts),
        wall_seconds=time.perf_counter() - started_at,
        materialize_wait_seconds=materialize_wait_seconds,
        fusion_wait_seconds=fusion_wait_seconds,
    )
