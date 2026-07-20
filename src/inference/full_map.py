"""用无 padding 的 80³ 滑窗和 float32 Gaussian 权重融合完整图概率。

主要入口:
    - `window_starts_zyx`: 生成覆盖完整 ZYX 网格且强制包含每轴末端真实窗口的确定性起点表。
    - `infer_full_map`: 批量执行 `forward_voxel_probability`，累积 `weight*probability` 与 `weight_sum`，逐 voxel 归一化后应用 producer 专属 hardmask。
    - `publish_full_map`: 发布 `probability_map.npz`、`geometry.json` 和最后的 probability `_COMPLETE`。

所有窗口都是完整、无 padding 的 80³ crop。重叠融合始终使用 float32；Find receptor hardmask 只在完整图融合结束后应用一次，`unet_c1` 概率直接通过。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from src.artifacts.io import atomic_savez_compressed, atomic_write_json
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import mark_role_complete

from .probability import logits_to_probability, postprocess_ligand_probability


@dataclass(frozen=True)
class FullMapResult:
    """
    保存一张完整图的融合概率与可审计融合统计。

    输入参数:
        - probability_map: float32, (D, H, W), 完整 ZYX voxel 网格上已经完成 producer 专属后处理的融合概率。
        - weight_sum: float32, (D, H, W), 每个完整图 voxel 实际收到的全部窗口 Gaussian 权重和，数值必须严格大于 0。
        - window_starts_zyx: tuple[tuple[int, int, int], ...], (N_window, 3), 实际执行的完整图离散 ZYX 窗口起点，按 Z、Y、X 笛卡尔积顺序排列。
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
        - full_shape_zyx: Sequence[int], (3,), 完整图的 ZYX voxel 形状 `(D, H, W)`。
        - window_shape_zyx: Sequence[int], (3,), 滑窗的 ZYX voxel 形状，正式值为 `(80, 80, 80)`。
        - stride_zyx: Sequence[int], (3,), 常规滑窗步幅，正式值为 `(40, 40, 40)`。

    输出:
        - starts: tuple[tuple[int, int, int], ...], (N_window, 3), 完整图离散 ZYX voxel-index 窗口起点，按 Z 外层、Y 中层、X 内层排列。
    """
    full_shape = tuple(int(value) for value in full_shape_zyx)
    window_shape = tuple(int(value) for value in window_shape_zyx)
    stride = tuple(int(value) for value in stride_zyx)
    if len(full_shape) != 3 or len(window_shape) != 3 or len(stride) != 3:
        raise ValueError("full/window/stride shape 都必须是长度 3 的 ZYX")
    # list[tuple[int, ...]], 长度 3；分别保存 Z、Y、X 三轴覆盖起点，且每轴都包含 0 和合法末端起点。
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
        - window_shape_zyx: Sequence[int], (3,), Gaussian 权重网格的 ZYX 形状，正式值为 `(80, 80, 80)`。
        - sigma: float, 每轴映射到 `[-1, 1]` 后的 Gaussian 标准差，正式值为 0.5。

    输出:
        - weight: float32, (D_window, H_window, W_window), 严格为正、中心高且边缘低的三维各向同性 Gaussian 权重。
    """
    shape = tuple(int(value) for value in window_shape_zyx)
    if len(shape) != 3 or any(value <= 1 for value in shape) or sigma <= 0:
        raise ValueError("window_shape_zyx 与 sigma 不合法")
    axes = [np.linspace(-1.0, 1.0, num=length, dtype=np.float32) for length in shape]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    # float32, (D_window, H_window, W_window), 每个局部 ZYX voxel 在规范化坐标中的平方半径。
    squared_radius = z * z + y * y + x * x
    return np.exp(-squared_radius / np.float32(2.0 * sigma * sigma)).astype(np.float32, copy=False)



##### 核心函数: 预测 #####
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
        - model: Any, 已切换到推理模式的完整 Stage1 模型包装器；必须实现 `forward_voxel_probability(batch)` 并返回单通道 ligand logits。
        - full_shape_zyx: Sequence[int], `(3,)`，当前完整图的 ZYX voxel 网格形状 `(D, H, W)`。
        - window_batch_builder: Callable, 接收一批完整图离散 ZYX voxel-index 窗口起点，返回统一 Dataset/Collator 生成的模型批次。
        - stage1_model_name: str, 当前 Stage1 模型来源名，决定融合后的 hardmask 规则。
        - receptor_hardmask_full: np.ndarray | None, `(D, H, W)`，Find 必填的完整图 receptor home-voxel 布尔 mask；True voxel 的最终概率清零，unet_c1 传 None。
        - window_batch_size: int, 模型推理时用的 batch_size: 单次仅计算 voxel 概率的模型调用所含窗口数。
        - window_shape_zyx: tuple[int, int, int], 滑窗 ZYX 形状，正式值为 `(80, 80, 80)`。
        - stride_zyx: tuple[int, int, int], 常规滑窗 ZYX 步幅，正式值为 `(40, 40, 40)`。
        - gaussian_sigma: float, 规范化局部坐标中的 Gaussian 标准差，正式值为 0.5。

    输出:
        - result: FullMapResult, 融合无 5-voxel 裁边，且所有 voxel `weight_sum>0`
    """
    if window_batch_size <= 0:
        raise ValueError("window_batch_size 必须为正整数")
    # tuple[int, int, int], 当前完整概率图的 ZYX 形状 `(D, H, W)`。
    full_shape = tuple(int(value) for value in full_shape_zyx)
    # tuple[tuple[int, int, int], ...], (N_window, 3), 覆盖完整图的确定性离散 ZYX 滑窗起点。
    starts = window_starts_zyx(full_shape, window_shape_zyx, stride_zyx)
    # float32, (80, 80, 80), 每个窗口复用的严格正 Gaussian 权重。
    weight = gaussian_window_weight(window_shape_zyx, gaussian_sigma)
    # float32, (D, H, W), 每个完整图 voxel 累积的 `Gaussian weight * window probability`。
    probability_sum = np.zeros(full_shape, dtype=np.float32)
    # float32, (D, H, W), 每个完整图 voxel 实际收到的全部窗口 Gaussian 权重和。
    weight_sum = np.zeros(full_shape, dtype=np.float32)

    try:
        no_grad_context = torch.no_grad
    except ImportError:
        from contextlib import nullcontext

        no_grad_context = nullcontext

    with no_grad_context():
        for offset in range(0, len(starts), int(window_batch_size)):
            # tuple[tuple[int, int, int], ...], (B_window, 3), 当前一次 voxel-only forward 的离散 ZYX 窗口起点。
            batch_starts = starts[offset : offset + int(window_batch_size)]
            batch = window_batch_builder(batch_starts)
            # torch.Tensor | np.ndarray, (B_window, 1, 80, 80, 80), sigmoid 前的单通道 ligand logits。
            logits = model.forward_voxel_probability(batch)
            # float32, (B_window, 80, 80, 80), 当前窗口 batch 的逐 voxel ligand 概率。
            probabilities = logits_to_probability(logits)
            if probabilities.shape != (len(batch_starts), *tuple(window_shape_zyx)):
                raise ValueError(f"voxel-only 输出 shape 与窗口 batch 不一致: actual={probabilities.shape}, expected={(len(batch_starts), *window_shape_zyx)}")
            for row, (z_start, y_start, x_start) in enumerate(batch_starts):
                # tuple[slice, slice, slice], 当前 80³ 窗口在完整 ZYX 网格中的无 padding 目标区域。
                slices = (
                    slice(z_start, z_start + window_shape_zyx[0]),
                    slice(y_start, y_start + window_shape_zyx[1]),
                    slice(x_start, x_start + window_shape_zyx[2]),
                )
                probability_sum[slices] += weight * probabilities[row]
                weight_sum[slices] += weight
    if not bool(np.all(weight_sum > 0.0)):
        raise RuntimeError("滑窗未覆盖完整网格，存在 weight_sum<=0 的 voxel")
    # float32, (D, H, W), `probability_sum/weight_sum` 得到的 Gaussian 归一化完整图连续概率。
    fused = np.divide(
        probability_sum,
        weight_sum,
        out=np.empty_like(probability_sum),
    ).astype(np.float32, copy=False)
    # float32, (D, H, W), 应用 Find receptor hardmask 或 unet_c1 直通规则后的正式完整图概率。
    processed = postprocess_ligand_probability(fused, stage1_model_name, receptor_hardmask_full)
    return FullMapResult(
        probability_map=processed,
        weight_sum=weight_sum,
        window_starts_zyx=starts,
    )


##### 简单包装(保存) #####
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
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 的 probability NPZ、geometry JSON 与状态路径。
        - result: FullMapResult, 已完成 Gaussian 融合和 producer 后处理的完整图结果。
        - origin_xyz: Sequence[float], (3,), 完整 voxel 网格起点的世界 XYZ 坐标，单位 Å。
        - voxel_size_xyz: Sequence[float], (3,), 完整图沿世界 XYZ 三轴的体素尺寸，单位 Å/voxel。
        - window_shape_zyx: tuple[int, int, int], 生成该完整图时使用的窗口 ZYX 形状。
        - stride_zyx: tuple[int, int, int], 生成该完整图时使用的窗口 ZYX 常规步幅。
        - gaussian_sigma: float, 生成该完整图时使用的规范化 Gaussian 标准差。

    落盘产物:
        - `probability_map.npz`: NPZ，包含 `probability_map: float32, (D, H, W)`、`origin_xyz: float32, (3,)` 和 `voxel_size_xyz: float32, (3,)`；概率图轴顺序为 ZYX，两个几何字段按世界 XYZ 排列。
        - `geometry.json`: JSON，包含完整图形状、世界 XYZ 原点、世界 XYZ 体素尺寸、窗口形状、步幅和 Gaussian sigma。
        - `status/probability/_COMPLETE`: JSON，在前两个 payload 都原子发布后最后写入。
    """
    probability = np.asarray(result.probability_map, dtype=np.float32)
    # float32, (3,), 完整 voxel 网格角点的世界 XYZ 坐标，单位 Å；与 `geometry.json["origin_xyz"]` 表示同一几何值。
    origin = np.asarray(origin_xyz, dtype=np.float32)
    # float32, (3,), 完整图沿世界 XYZ 三轴的体素尺寸，单位 Å/voxel；与 `geometry.json["voxel_size_xyz"]` 表示同一几何值。
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if origin.shape != (3,) or not bool(np.all(np.isfinite(origin))):
        raise ValueError("origin_xyz 必须是长度 3 的有限 XYZ 坐标")
    if voxel_size.shape != (3,) or not bool(np.all(np.isfinite(voxel_size))) or bool(np.any(voxel_size <= 0)):
        raise ValueError("voxel_size_xyz 必须是长度 3 的有限正数 XYZ 体素尺寸")

    def validate(arrays: Mapping[str, np.ndarray]) -> None:
        """
        校验临时重读的完整图 probability 数组。

        输入参数:
            - arrays: Mapping[str, np.ndarray], 临时重读的 NPZ 字段；精确包含 `probability_map`、`origin_xyz` 和 `voxel_size_xyz`。

        输出:
            - None: 三个字段的集合、dtype、shape、有限值与几何逐值一致性全部满足时返回
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
