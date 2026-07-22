"""80³/stride40 窗口的 float32 Gaussian 完整图概率融合。

窗口生成器强制加入每个轴的末端起点，因此所有窗口都是真实、无 padding 的 80³
crop 且覆盖全图。重叠区域累积 ``weight * probability`` 和 ``weight_sum``，最后
逐 voxel 相除；producer-specific hardmask 只在完整融合后应用一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.artifacts.io import atomic_savez_compressed, atomic_write_json
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import mark_role_complete

from .probability import logits_to_probability, postprocess_ligand_probability


@dataclass(frozen=True)
class FullMapResult:
    """
    保存一张完整图的融合概率与可审计融合统计。

    输入参数:
        - probability_map: np.ndarray, (D,H,W), float32，完整图 ZYX voxel grid 上完成 producer 后处理的概率
        - weight_sum: np.ndarray, (D,H,W), float32，完整图 ZYX voxel grid 上全部窗口 Gaussian 权重和
        - window_starts_zyx: tuple[tuple[int,int,int],...], 实际执行的完整图离散 ZYX voxel-index 窗口 corner 起点
    """

    probability_map: np.ndarray
    weight_sum: np.ndarray
    window_starts_zyx: tuple[tuple[int, int, int], ...]


def window_starts_1d(length: int, window_size: int, stride: int) -> tuple[int, ...]:
    """
    生成覆盖一条轴且强制包含末端真实窗口的升序起点。

    输入参数:
        - length: int, 完整轴长度，必须不小于 window_size
        - window_size: int, 窗口长度，正式值为 80
        - stride: int, 常规步幅，正式值为 40

    输出:
        - starts: tuple[int,...], 单轴离散 voxel-index 窗口 corner 起点，包含 0 与 `length-window_size` 并去重升序
    """
    if length < window_size or window_size <= 0 or stride <= 0:
        raise ValueError("length/window_size/stride 不能形成真实无 padding 窗口")
    starts = list(range(0, length - window_size + 1, stride))
    starts.append(length - window_size)
    return tuple(sorted(set(starts)))


def window_starts_zyx(
    full_shape_zyx: Sequence[int],
    window_shape_zyx: Sequence[int],
    stride_zyx: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """
    生成完整三维网格的确定性 ZYX 窗口起点笛卡尔积。

    输入参数:
        - full_shape_zyx: Sequence[int], (3,), 完整图 shape
        - window_shape_zyx: Sequence[int], (3,), 正式值为 (80,80,80)
        - stride_zyx: Sequence[int], (3,), 正式值为 (40,40,40)

    输出:
        - starts: tuple[tuple[int,int,int],...], 完整图离散 ZYX voxel-index 窗口 corner 起点，按 Z 外层、Y 中层、X 内层排列
    """
    full_shape = tuple(int(value) for value in full_shape_zyx)
    window_shape = tuple(int(value) for value in window_shape_zyx)
    stride = tuple(int(value) for value in stride_zyx)
    if len(full_shape) != 3 or len(window_shape) != 3 or len(stride) != 3:
        raise ValueError("full/window/stride shape 都必须是长度 3 的 ZYX")
    per_axis = [
        window_starts_1d(length, window, step)
        for length, window, step in zip(full_shape, window_shape, stride)
    ]
    return tuple(tuple(int(value) for value in start) for start in product(*per_axis))


def gaussian_window_weight(
    window_shape_zyx: Sequence[int],
    sigma: float,
) -> np.ndarray:
    """
    按各轴线性映射到 [-1,1] 的公式构造固定 float32 Gaussian 权重。

    输入参数:
        - window_shape_zyx: Sequence[int], (3,), 正式窗口为 (80,80,80)
        - sigma: float, 规范化坐标中的 Gaussian sigma，正式值为 0.5

    输出:
        - weight: np.ndarray, (Dz,Dy,Dx), float32，严格正且中心高、边缘低
    """
    shape = tuple(int(value) for value in window_shape_zyx)
    if len(shape) != 3 or any(value <= 1 for value in shape) or sigma <= 0:
        raise ValueError("window_shape_zyx 与 sigma 不合法")
    axes = [
        np.linspace(-1.0, 1.0, num=length, dtype=np.float32) for length in shape
    ]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    squared_radius = z * z + y * y + x * x
    return np.exp(-squared_radius / np.float32(2.0 * sigma * sigma)).astype(
        np.float32, copy=False
    )


def infer_full_map(
    model: Any,
    full_shape_zyx: Sequence[int],
    window_batch_builder: Callable[[Sequence[tuple[int, int, int]]], Mapping[str, Any]],
    stage1_model_name: str,
    receptor_hardmask_full: np.ndarray | None,
    window_batch_size: int,
    window_shape_zyx: tuple[int, int, int] = (80, 80, 80),
    stride_zyx: tuple[int, int, int] = (40, 40, 40),
    gaussian_sigma: float = 0.5,
) -> FullMapResult:
    """
    流式执行 voxel-only 窗口推理并以 float32 Gaussian 融合完整图。

    输入参数:
        - model: Any, 已 eval 的完整 Stage1 wrapper，必须实现
          `forward_voxel_probability(batch)->voxel_logits_ligand`
        - full_shape_zyx: Sequence[int], (3,), 当前完整图 ZYX voxel-grid shape
        - window_batch_builder: Callable, 接收一批完整图离散 ZYX voxel-index 窗口 corner 起点并返回统一 Dataset/Collator batch
        - stage1_model_name: str, 当前 producer 名
        - receptor_hardmask_full: np.ndarray | None, (D,H,W), 两个 Find 必填的完整图 ZYX voxel-grid receptor home-voxel mask；unet_c1 传 None
        - window_batch_size: int, 单次 voxel-only forward 的窗口数
        - window_shape_zyx: tuple[int,int,int], 正式值为 (80,80,80)
        - stride_zyx: tuple[int,int,int], 正式值为 (40,40,40)
        - gaussian_sigma: float, 正式值为 0.5

    输出:
        - result: FullMapResult, 融合无 5-voxel 裁边，且所有 voxel `weight_sum>0`
    """
    if window_batch_size <= 0:
        raise ValueError("window_batch_size 必须为正整数")
    # tuple[int,int,int], (3,), 当前完整概率图的 ZYX shape
    full_shape = tuple(int(value) for value in full_shape_zyx)
    # tuple[tuple[int,int,int],...], (N_window,3), 确定性滑窗起点表
    starts = window_starts_zyx(full_shape, window_shape_zyx, stride_zyx)
    # np.ndarray[float32], (80,80,80), 每个窗口复用的严格正 Gaussian 权重
    weight = gaussian_window_weight(window_shape_zyx, gaussian_sigma)
    # np.ndarray[float32], (D,H,W), 累计每个窗口的 weight * probability
    probability_sum = np.zeros(full_shape, dtype=np.float32)
    # np.ndarray[float32], (D,H,W), 累计每个 voxel 实际收到的窗口权重
    weight_sum = np.zeros(full_shape, dtype=np.float32)

    try:
        import torch

        no_grad_context = torch.no_grad
    except ImportError:
        from contextlib import nullcontext

        no_grad_context = nullcontext

    with no_grad_context():
        for offset in range(0, len(starts), int(window_batch_size)):
            # tuple[tuple[int,int,int],...], (B_window,3), 当前一次 voxel-only forward 的起点
            batch_starts = starts[offset : offset + int(window_batch_size)]
            batch = window_batch_builder(batch_starts)
            # torch.Tensor | np.ndarray, (B_window,1,80,80,80), sigmoid 前 ligand logits
            logits = model.forward_voxel_probability(batch)
            # np.ndarray[float32], (B_window,80,80,80), 当前窗口 batch 的 ligand 概率
            probabilities = logits_to_probability(logits)
            if probabilities.shape != (len(batch_starts), *tuple(window_shape_zyx)):
                raise ValueError(
                    "voxel-only 输出 shape 与窗口 batch 不一致: "
                    f"actual={probabilities.shape}, expected={(len(batch_starts), *window_shape_zyx)}"
                )
            for row, (z_start, y_start, x_start) in enumerate(batch_starts):
                # tuple[slice,slice,slice], 当前 80³ 窗口在完整 ZYX 网格中的目标区域
                slices = (
                    slice(z_start, z_start + window_shape_zyx[0]),
                    slice(y_start, y_start + window_shape_zyx[1]),
                    slice(x_start, x_start + window_shape_zyx[2]),
                )
                probability_sum[slices] += weight * probabilities[row]
                weight_sum[slices] += weight
    if not bool(np.all(weight_sum > 0.0)):
        raise RuntimeError("滑窗未覆盖完整网格，存在 weight_sum<=0 的 voxel")
    # np.ndarray[float32], (D,H,W), Gaussian 归一化后的连续完整图概率
    fused = np.divide(
        probability_sum,
        weight_sum,
        out=np.empty_like(probability_sum),
    ).astype(np.float32, copy=False)
    # np.ndarray[float32], (D,H,W), 应用 Find hardmask 或 unet 直通后的正式概率
    processed = postprocess_ligand_probability(
        fused, stage1_model_name, receptor_hardmask_full
    )
    return FullMapResult(
        probability_map=processed,
        weight_sum=weight_sum,
        window_starts_zyx=starts,
    )


def publish_full_map(
    paths: Stage1ArtifactPaths,
    result: FullMapResult,
    origin_xyz: Sequence[float],
    voxel_size_xyz: Sequence[float],
    window_shape_zyx: tuple[int, int, int] = (80, 80, 80),
    stride_zyx: tuple[int, int, int] = (40, 40, 40),
    gaussian_sigma: float = 0.5,
) -> None:
    """
    原子发布完整图概率、几何与最后的 probability `_COMPLETE`。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 路径
        - result: FullMapResult, 已完成后处理的融合结果
        - origin_xyz: Sequence[float], (3,), 完整 voxel grid corner 的世界 XYZ 坐标
        - voxel_size_xyz: Sequence[float], (3,), XYZ Å/voxel
        - window_shape_zyx: tuple[int,int,int], 正式窗口 shape
        - stride_zyx: tuple[int,int,int], 正式窗口 stride
        - gaussian_sigma: float, 正式 Gaussian sigma

    输出:
        - None, payload 与 geometry 都发布后才写 role 完成标记
    """
    probability = np.asarray(result.probability_map, dtype=np.float32)
    origin = np.asarray(origin_xyz, dtype=np.float32)
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if origin.shape != (3,) or not bool(np.all(np.isfinite(origin))):
        raise ValueError("origin_xyz 必须是长度 3 的有限 XYZ 坐标")
    if voxel_size.shape != (3,) or not bool(np.all(np.isfinite(voxel_size))) or bool(np.any(voxel_size <= 0)):
        raise ValueError("voxel_size_xyz 必须是长度 3 的有限正数 XYZ 体素尺寸")

    def validate(arrays: Mapping[str, np.ndarray]) -> None:
        """
        校验临时重读的完整图 probability 数组。

        输入参数:
            - arrays: Mapping[str,np.ndarray], 包含 `probability_map` 的临时 NPZ 字段

        输出:
            - None: dtype、完整图 ZYX shape 与有限值契约全部满足时返回
        """
        value = np.asarray(arrays["probability_map"])
        if value.dtype != np.dtype(np.float32) or value.shape != probability.shape:
            raise ValueError("probability_map 的 dtype/shape 不符合发布前契约")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError("probability_map 含非有限值")

    atomic_savez_compressed(
        paths.probability_npz,
        {
            "probability_map": probability,
            "origin_xyz": origin,
            "voxel_size_xyz": voxel_size,
        },
        validator=validate,
    )
    atomic_write_json(
        paths.probability_geometry_json,
        {
            "full_shape_zyx": list(probability.shape),
            "origin_xyz": [float(value) for value in origin],
            "voxel_size_xyz": [float(value) for value in voxel_size],
            "window_shape_zyx": list(window_shape_zyx),
            "stride_zyx": list(stride_zyx),
            "gaussian_sigma": float(gaussian_sigma),
        },
    )
    mark_role_complete(paths, "probability")
