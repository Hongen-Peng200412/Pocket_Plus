# -*- coding: utf-8 -*-
"""计算 F1 basic 与 F3 centered 候选分数.

主要入口 :func:`score_centered_candidates` 直接返回每个 centered 条目的
float32 分数. 最终 `score` 和 `selected` 由单 PDB 发布事务直接写入.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree


def build_gaussian_distance_table(
    centered: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """计算每个 A 原子到所属来源 blob 最近体素中心的世界距离.

    输入字段:
        - voxel_offsets: int64 `(N_candidate + 1,)`, 以半开区间切分 `voxel_index_local_zyx`; 首值为 0, 末值为 L_voxel.
        - voxel_index_local_zyx: 数值数组 `(L_voxel, 3)`, 每个来源体素在所属 80³ BOX 内的 ZYX 索引.
        - A_offsets: int64 `(N_candidate + 1,)`, 以半开区间同步切分 `A_coord_local_xyz` 和 `A_probability`; 首值为 0, 末值为 N_atom.
        - A_coord_local_xyz: 数值数组 `(N_atom, 3)`, A 原子在所属 80³ BOX 内的 XYZ 体素坐标.
        - A_probability: 数值数组 `(N_atom,)`, 与 A 原子逐项对齐的配体概率.
        - voxel_size_world: 数值数组 `(N_candidate, 3)`, 每个候选的世界 XYZ 体素尺寸, 单位 Å/voxel.

    返回字段:
        - A_offsets: int64 `(N_candidate + 1,)`, 以半开区间切分两个返回的 A 原子值表; 首值为 0, 末值为 N_atom.
        - A_distance_to_source: float32 `(N_atom,)`, A 原子到同一候选来源体素中心的最近世界距离, 单位 Å; 超过 5 Å 为 Inf.
        - A_probability: float32 `(N_atom,)`, 与距离逐原子对齐的模型概率.
    """

    voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
    atom_offsets = np.asarray(centered["A_offsets"], dtype=np.int64)
    local_zyx = np.asarray(centered["voxel_index_local_zyx"], dtype=np.float64)
    atom_xyz = np.asarray(centered["A_coord_local_xyz"], dtype=np.float64)
    voxel_size_xyz = np.asarray(centered["voxel_size_world"], dtype=np.float64)
    distances = np.full(atom_xyz.shape[0], np.inf, dtype=np.float32)
    for entry_index in range(voxel_offsets.size - 1):
        voxel_slice = slice(
            int(voxel_offsets[entry_index]), int(voxel_offsets[entry_index + 1])
        )
        atom_slice = slice(
            int(atom_offsets[entry_index]), int(atom_offsets[entry_index + 1])
        )
        source_xyz = (local_zyx[voxel_slice][:, [2, 1, 0]] + 0.5) * voxel_size_xyz[
            entry_index
        ][None, :]
        receptor_xyz = atom_xyz[atom_slice] * voxel_size_xyz[entry_index][None, :]
        if receptor_xyz.shape[0]:
            # float64, (N_A_entry,), 超过 5 Å 的查询结果为 Inf, Gaussian 求和时排除.
            distance, _ = cKDTree(source_xyz).query(
                receptor_xyz,
                k=1,
                distance_upper_bound=5.0,
            )
            distances[atom_slice] = distance.astype(np.float32)
    return {
        "A_offsets": atom_offsets,
        "A_distance_to_source": distances,
        "A_probability": np.asarray(centered["A_probability"], dtype=np.float32),
    }


def sum_gaussian_atom_terms(
    atom_offsets: np.ndarray,
    atom_distance: np.ndarray,
    atom_probability: np.ndarray,
    tau_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按唯一正式数值顺序计算每个候选的 Gaussian 正项和负项.

    输入参数:
        - atom_offsets: int64 `(N_candidate + 1,)`, 同步切分 atom_distance 和 atom_probability; 首值为 0, 末值为 N_atom.
        - atom_distance: float32 `(N_atom,)`, A 原子到来源体素的最近距离, 单位 Å; Inf 表示 5 Å 截断外原子.
        - atom_probability: float32 `(N_atom,)`, 与 atom_distance 逐原子对齐的模型概率.
        - tau_angstrom: float, Gaussian 距离标准差, 单位 Å.

    返回值:
        - positive: float32 `(N_candidate,)`, 每个候选的 `sum(exp(-d²/(2*tau²)) * p)`.
        - negative: float32 `(N_candidate,)`, 每个候选的 `sum(exp(-d²/(2*tau²)) * (1-p))`.

    权重和归约使用 float64, 每个候选的结果最后规范为 float32. Inf 距离不参与求和.
    """

    offsets = np.asarray(atom_offsets, dtype=np.int64)
    distance = np.asarray(atom_distance, dtype=np.float32)
    probability = np.asarray(atom_probability, dtype=np.float32)
    positive = np.zeros(offsets.size - 1, dtype=np.float32)
    negative = np.zeros_like(positive)
    # 逐候选使用 float64 归约, 保持校准搜索与正式推理完全相同的数值顺序.
    for index in range(positive.size):
        begin = int(offsets[index])
        end = int(offsets[index + 1])
        candidate_distance = distance[begin:end]
        included = np.isfinite(candidate_distance)
        weight = np.exp(
            -(candidate_distance[included].astype(np.float64) ** 2)
            / (2.0 * float(tau_angstrom) ** 2)
        )
        candidate_probability = probability[begin:end][included]
        positive[index] = np.sum(weight * candidate_probability, dtype=np.float64)
        negative[index] = np.sum(
            weight * (1.0 - candidate_probability), dtype=np.float64
        )
    return positive, negative


# ================================================================================================


def score_centered_candidates(
    centered: Mapping[str, np.ndarray],
    score_mode: str,
    score_parameters: Mapping[str, float],
) -> np.ndarray:
    """按显式模式计算每个 centered 候选的最终分数.

    输入参数:
        - centered: centered 产物字段映射; `source_probability_mean` 是 float32 `(N_candidate,)` 来源 blob 平均概率.
        - score_mode: 字符串; `source_mean` 使用来源平均概率, `find_gaussian` 叠加 A 原子 Gaussian 正负项.
        - score_parameters: 浮点数映射; Find 模式读取 tau 和两个 lambda, 来源均值模式不读取字段.

    返回值:
        - score: float32 `(N_candidate,)`, 与 `centered["source_blob_index"]` 逐候选对齐的分数.

    Find 模式还读取 :func:`build_gaussian_distance_table` 列出的体素和 A 原子字段.
    分数为 `source_mean + lambda_positive * positive - lambda_negative * negative`.
    正负项只累加 5 Å 内的 A 原子, 不按原子数归一化.
    """

    source_mean = np.asarray(centered["source_probability_mean"], dtype=np.float32)
    if score_mode == "source_mean":
        return source_mean.copy()
    tau_angstrom = float(score_parameters["tau_angstrom"])
    lambda_positive = float(score_parameters["lambda_positive"])
    lambda_negative = float(score_parameters["lambda_negative"])
    atom_table = build_gaussian_distance_table(centered)
    positive, negative = sum_gaussian_atom_terms(
        atom_table["A_offsets"],
        atom_table["A_distance_to_source"],
        atom_table["A_probability"],
        tau_angstrom,
    )
    return (
        source_mean
        + np.float32(lambda_positive) * positive
        - np.float32(lambda_negative) * negative
    ).astype(np.float32, copy=False)
