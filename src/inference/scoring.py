# -*- coding: utf-8 -*-
"""计算 F1 basic 与 F3 centered 候选分数和最终选择标志.

主要入口 :func:`score_centered_candidates` 直接返回每个 centered 条目的
float32 分数. :func:`select_centered_candidates` 将显式体素数和分数阈值写回
归档数组, 不改变任何特征或坐标值表.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree


# ================================================================================================


def build_gaussian_distance_table(
    centered: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """计算每个 A 原子到所属来源 blob 最近体素中心的世界距离.

    返回 ``A_offsets: int64 (N_candidate+1,)``,
    ``A_distance_to_source: float32 (N_atom,)`` 和
    ``A_probability: float32 (N_atom,)``. 距离超过 5 Å 的原子保存为 ``Inf``,
    使多个 Gaussian 参数组合复用一次 KD-tree 查询.
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
        source_xyz = (
            local_zyx[voxel_slice][:, [2, 1, 0]] + 0.5
        ) * voxel_size_xyz[entry_index][None, :]
        receptor_xyz = atom_xyz[atom_slice] * voxel_size_xyz[entry_index][None, :]
        if receptor_xyz.shape[0]:
            # float64 (N_A_entry,), 超过 5 Å 的查询结果为 Inf, Gaussian 求和时排除.
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

    输入 offsets 为 int64 ``(N_candidate+1,)``; 距离与概率为对齐的 float32
    ``(N_atom,)``. 5 Å 外原子以 ``Inf`` 表示并排除. 权重和求和使用 float64,
    每个候选的两项结果最终规范为 float32 ``(N_candidate,)``.
    """

    offsets = np.asarray(atom_offsets, dtype=np.int64)
    distance = np.asarray(atom_distance, dtype=np.float32)
    probability = np.asarray(atom_probability, dtype=np.float32)
    positive = np.zeros(offsets.size - 1, dtype=np.float32)
    negative = np.zeros_like(positive)
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


def score_centered_candidates(
    centered: Mapping[str, np.ndarray],
    score_mode: str,
    score_parameters: Mapping[str, float],
) -> np.ndarray:
    """计算 source mean 或 Find Gaussian 分数.

    ``score_mode='source_mean'`` 只返回 ``source_probability_mean``. 模式
    ``find_gaussian`` 对每个 A 原子计算到来源 blob 最近体素中心的世界距离,
    只累加 5 Å 内的 Gaussian 权重. 正项和负项均直接求和, 不归一化.
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


def select_centered_candidates(
    centered: Mapping[str, np.ndarray],
    score: np.ndarray,
    score_threshold: float,
    min_voxels: int,
) -> dict[str, np.ndarray]:
    """复制 centered 数组并按显式分数阈值和最小体素数更新选择字段."""

    result = {name: np.asarray(value) for name, value in centered.items()}
    values = np.asarray(score, dtype=np.float32)
    voxel_count = np.diff(np.asarray(centered["voxel_offsets"], dtype=np.int64))
    result["score"] = values
    result["selected"] = (
        (values >= np.float32(score_threshold))
        & (voxel_count >= int(min_voxels))
    ).astype(np.bool_)
    return result
