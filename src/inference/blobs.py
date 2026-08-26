# -*- coding: utf-8 -*-
"""从完整概率图计算可选 Li 阈值并提取单阈值 26 连通区域.

主要入口 :func:`li_probability_threshold` 返回逐 PDB Li 阈值,
:func:`extract_probability_blobs` 返回任意显式阈值的全部区域稳定稀疏表. 文件
发布由调用方负责. 本模块不按 `min_voxels` 删除区域.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def li_probability_threshold(
    probability_map: np.ndarray,
    denominator: int,
) -> tuple[float, int, float]:
    """计算逐 PDB Li 阈值并向上量化到显式概率网格.

    输入参数:
        - probability_map: 数值数组 ``(D, H, W)``, 完整图 ZYX 配体概率.
        - denominator: 正整数, 概率网格分母; 本次实验传入 32768.

    返回值:
        - raw_threshold: float, 以完整图均值初始化并迭代得到的 Li 最小交叉熵阈值.
        - grid_index: int, ``ceil(raw_threshold * denominator)`` 对应的网格编号.
        - applied_threshold: float, ``grid_index / denominator``; 连通区域提取包含该端点.

    每轮把 ``probability <= threshold`` 视为背景, 其余体素视为前景. 迭代容差
    固定为 ``1e-5``, 最多更新 100 次. 向上量化保证实际截断不会纳入低于 Li
    原始阈值的体素.
    """
    # float64, (D, H, W), 当前 PDB 的完整图概率; Li 均值和对数迭代均使用 float64.
    probability = np.asarray(probability_map, dtype=np.float64)
    # float64, (D*H*W,), 完整图概率的一维视图; 背景和前景按当前阈值反复切分.
    values = probability.reshape(-1)
    # raw_threshold 从全图均值开始; 常数图会在首次切分时直接结束.
    raw_threshold = float(values.mean())
    # minimum 和 maximum 限制迭代阈值始终落在当前完整图的实际概率范围内.
    minimum = float(values.min())
    maximum = float(values.max())
    # tiny 防止均值为零时执行 log(0); 它只参与 Li 公式的数值计算.
    tiny = np.finfo(np.float64).tiny
    for _ in range(100):
        # float64, (N_background,), 当前阈值端点及以下的背景概率.
        background = values[values <= raw_threshold]
        # float64, (N_foreground,), 严格高于当前阈值的前景概率.
        foreground = values[values > raw_threshold]
        if background.size == 0 or foreground.size == 0:
            break
        # 两侧均值是 Li 最小交叉熵更新式的唯一统计量.
        mean_background = max(float(background.mean()), tiny)
        mean_foreground = max(float(foreground.mean()), tiny)
        # logarithm_difference 是两侧均值对数之差, 对应 Li 更新式的分母.
        logarithm_difference = np.log(mean_background) - np.log(mean_foreground)
        if logarithm_difference == 0.0:
            break
        # updated_threshold 是本轮 Li 更新值, 再限制到完整图实际概率范围内.
        updated_threshold = float(
            np.clip(
                (mean_background - mean_foreground) / logarithm_difference,
                minimum,
                maximum,
            )
        )
        if abs(updated_threshold - raw_threshold) <= 1e-5:
            raw_threshold = updated_threshold
            break
        raw_threshold = updated_threshold
    # grid_index 向上取整, 使应用阈值不低于 raw_threshold.
    grid_index = int(
        np.clip(
            np.ceil(raw_threshold * int(denominator)),
            0,
            int(denominator),
        )
    )
    # applied_threshold 是后续 ``probability >= threshold`` 实际使用的概率端点.
    applied_threshold = float(grid_index) / float(denominator)
    return raw_threshold, grid_index, applied_threshold


# ================================================================================================


def extract_probability_blobs(
    probability_map: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """提取并稳定排序一个阈值下的全部 26 邻域连通区域.

    输入参数:
        - probability_map: float32 ``(D, H, W)``, 完整图 ZYX 配体概率.
        - threshold: float, 包含端点的概率阈值.

    返回值:
        - blob_index: int32 ``(N_blob,)``, 排序后的连续 blob 编号.
        - voxel_offsets: int64 ``(N_blob+1,)``, 以半开区间同步切分 `voxel_index_global_zyx` 与 `source_probability`; 首值为 0, 末值为 L_voxel.
        - voxel_index_global_zyx: int32 ``(L_voxel, 3)``, 完整图 ZYX 索引.
        - source_probability: float32 ``(L_voxel,)``, 来源体素的完整图概率.
        - source_probability_mean: float32 ``(N_blob,)``, 各区域平均概率.
        - voxel_count: int32 ``(N_blob,)``, 各区域体素数.
        - fits_centered_box: bool ``(N_blob,)``, 包围盒是否可被合法 80³ BOX 容纳.
        - centered_box_start_zyx: int32 ``(N_blob, 3)``, (合法的)让blob中心尽可能居中的 BOX 起点; 不可容纳时为 ``-1``.
        - source_threshold_value: float32 ``(1,)``, 本次阈值.

    区域按平均概率降序, 再按最小完整图 C-order 线性索引升序. 本函数不按
    ``min_voxels`` 删除区域.
    """
    # float32, (D, H, W), 完整图 ZYX 配体概率; D/H/W 分别对应 Z/Y/X 体素轴.
    probability = np.asarray(probability_map, dtype=np.float32)
    # 整数数组, (D, H, W), 0 表示低于阈值的背景体素, 1:count 表示 26 邻域连通区域编号.
    labels, count = ndimage.label(
        probability >= np.float32(threshold),
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    # 每项依次保存正式平均概率, 最小 C-order 线性编号, ZYX 坐标, 逐体素概率, BOX 可容纳标志和 BOX 起点.
    records: list[tuple[float, int, np.ndarray, np.ndarray, bool, np.ndarray]] = []
    # int64, (3,), 完整概率图的 ZYX 形状; 用于限制 80³ BOX 起点不越过完整图边界.
    full_shape = np.asarray(probability.shape, dtype=np.int64)
    # int64, (L_positive,), 所有阈值内体素在 probability.reshape(-1) 中的 C-order 线性编号.
    linear_positive = np.flatnonzero(labels.reshape(-1)).astype(np.int64)
    # 整数数组, (L_positive,), 与 linear_positive 逐体素对齐的连通区域编号.
    positive_label = labels.reshape(-1)[linear_positive]
    # int64, (L_positive,), 按连通区域编号稳定分组时对 linear_positive 第一维采用的重排下标.
    label_order = np.argsort(positive_label, kind="stable")
    # int64, (L_positive,), 按连通区域编号连续排列的完整图 C-order 线性编号.
    linear_by_label = linear_positive[label_order]
    # int64, (N_blob,), 每个连通区域包含的体素数; 数组第 i 项对应标签 i + 1.
    label_counts = np.bincount(
        positive_label,
        minlength=int(count) + 1,
    )[1:]
    # int64, (N_blob + 1,), 把 linear_by_label 切成各连通区域; 首值为 0, 末值为 L_positive.
    label_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(label_counts, dtype=np.int64))
    )
    # float32, (D*H*W,), 完整概率图的 C-order 一维视图; 数值由 linear_by_label 直接寻址.
    probability_flat = probability.reshape(-1)
    for label_id in range(1, int(count) + 1):
        begin = int(label_offsets[label_id - 1])
        end = int(label_offsets[label_id])
        # int64, (K_blob,), 当前连通区域体素在完整概率图中的 C-order 线性编号.
        linear = linear_by_label[begin:end]
        # int32, (K_blob, 3), 当前连通区域在完整图中的 ZYX 体素索引.
        coordinates = np.column_stack(
            np.unravel_index(linear, probability.shape)
        ).astype(np.int32)
        # float32, (K_blob,), 与 coordinates 第一维逐体素对齐的完整图配体概率.
        values = probability_flat[linear].astype(np.float32)
        # int64, (3,), 当前连通区域包围盒两端的完整图 ZYX 体素索引, 端点均包含.
        minimum = coordinates.min(axis=0).astype(np.int64)
        maximum = coordinates.max(axis=0).astype(np.int64)
        # int64, (3,), 让连通区域体素质心落在 80³ BOX 中心附近的候选 ZYX 起点.
        requested = np.rint(
            coordinates.astype(np.float64).mean(axis=0) + 0.5 - 40.0
        ).astype(np.int64)
        # int64, (3,), 同时容纳区域包围盒且位于完整图内时允许的 BOX 起点闭区间.
        lowest_start = np.maximum(0, maximum - 79)
        highest_start = np.minimum(minimum, full_shape - 80)
        # bool 标量, True 表示三个 ZYX 轴都存在合法 80³ BOX 起点.
        fits = bool(np.all(lowest_start <= highest_start))
        # int64, (3,), 可容纳时是离质心候选最近的合法起点; 不可容纳时仅保留未裁切候选且不会落盘.
        start = np.clip(requested, lowest_start, highest_start) if fits else requested
        # float32 标量, 当前区域所有来源体素概率以 float64 求均值后的正式排序值.
        source_mean = np.float32(values.mean(dtype=np.float64))
        records.append(
            (
                float(source_mean),
                int(linear[0]),
                coordinates,
                values,
                fits,
                start.astype(np.int32) if fits else np.full(3, -1, dtype=np.int32),
            )
        )
    # records 的候选轴先按正式 float32 均值降序, 再按最小完整图 C-order 线性编号升序.
    records.sort(key=lambda record: (-record[0], record[1]))
    # int32, (N_blob,), 排序后每个连通区域的来源体素数.
    counts = np.asarray([record[2].shape[0] for record in records], dtype=np.int32)
    # int64, (N_blob + 1,), 同步切分返回的 voxel_index_global_zyx 与 source_probability; 首值为 0, 末值为 L_voxel.
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    return {
        "blob_index": np.arange(len(records), dtype=np.int32),
        "voxel_offsets": offsets,
        "voxel_index_global_zyx": (
            np.concatenate([record[2] for record in records], axis=0)
            if records
            else np.empty((0, 3), dtype=np.int32)
        ),
        "source_probability": (
            np.concatenate([record[3] for record in records], axis=0)
            if records
            else np.empty((0,), dtype=np.float32)
        ),
        "source_probability_mean": np.asarray(
            [record[0] for record in records],
            dtype=np.float32,
        ),
        "voxel_count": counts,
        "fits_centered_box": np.asarray(
            [record[4] for record in records],
            dtype=np.bool_,
        ),
        "centered_box_start_zyx": (
            np.stack([record[5] for record in records], axis=0)
            if records
            else np.empty((0, 3), dtype=np.int32)
        ),
        "source_threshold_value": np.asarray([threshold], dtype=np.float32),
    }
