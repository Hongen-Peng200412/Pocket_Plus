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

    source_mean = np.asarray(centered["source_probability_mean"], dtype=np.float64)
    if score_mode == "source_mean":
        return source_mean.astype(np.float32)
    tau_angstrom = float(score_parameters["tau_angstrom"])
    lambda_positive = float(score_parameters["lambda_positive"])
    lambda_negative = float(score_parameters["lambda_negative"])
    voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
    atom_offsets = np.asarray(centered["A_offsets"], dtype=np.int64)
    local_zyx = np.asarray(centered["voxel_index_local_zyx"], dtype=np.float64)
    atom_xyz = np.asarray(centered["A_coord_local_xyz"], dtype=np.float64)
    atom_probability = np.asarray(centered["A_probability"], dtype=np.float64)
    voxel_size_xyz = np.asarray(centered["voxel_size_world"], dtype=np.float64)
    score = source_mean.copy()
    for entry_index in range(source_mean.size):
        voxel_slice = slice(
            int(voxel_offsets[entry_index]),
            int(voxel_offsets[entry_index + 1]),
        )
        atom_slice = slice(
            int(atom_offsets[entry_index]),
            int(atom_offsets[entry_index + 1]),
        )
        source_xyz = (
            local_zyx[voxel_slice][:, [2, 1, 0]] + 0.5
        ) * voxel_size_xyz[entry_index][None, :]
        receptor_xyz = atom_xyz[atom_slice] * voxel_size_xyz[entry_index][None, :]
        if receptor_xyz.shape[0] == 0:
            continue
        distance, _ = cKDTree(source_xyz).query(
            receptor_xyz,
            k=1,
            distance_upper_bound=5.0,
        )
        included = np.isfinite(distance)
        weight = np.exp(
            -(distance[included] ** 2) / (2.0 * float(tau_angstrom) ** 2)
        )
        probability = atom_probability[atom_slice][included]
        score[entry_index] += float(lambda_positive) * float(np.sum(weight * probability))
        score[entry_index] -= float(lambda_negative) * float(np.sum(weight * (1.0 - probability)))
    return score.astype(np.float32)


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
