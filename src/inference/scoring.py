# -*- coding: utf-8 -*-
"""计算候选的 basic/Gaussian 分数并执行实验性逐 PDB 比例选择.

主要入口 :func:`score_centered_candidates` 返回每个候选的 float32 分数,
:func:`select_score_ratio_candidates` 在固定预过滤总体中返回 top-ratio 选择.
最终 `score` 和 `selected` 由调用方发布.
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
        - A_offsets: int64 `(N_candidate + 1,)`, 以半开区间同步切分 `A_distance_to_source` 与 `A_probability`; 首值为 0, 末值为 N_atom.
        - A_distance_to_source: float32 `(N_atom,)`, A 原子到同一候选来源体素中心的最近世界距离, 单位 Å; 超过 5 Å 为 Inf.
        - A_probability: float32 `(N_atom,)`, 与距离逐原子对齐的模型概率.
    """
    # int64, (N_candidate + 1,), 以半开区间切分来源 blob 的局部 ZYX 体素表.
    voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
    # int64, (N_candidate + 1,), 以半开区间同步切分 A 原子坐标, 距离和概率.
    atom_offsets = np.asarray(centered["A_offsets"], dtype=np.int64)
    # float64, (L_voxel, 3), 每个来源体素在所属 80³ BOX 内的 ZYX 索引.
    local_zyx = np.asarray(centered["voxel_index_local_zyx"], dtype=np.float64)
    # float64, (N_atom, 3), 每个 A 原子在所属 80³ BOX 内的 XYZ 体素坐标.
    atom_xyz = np.asarray(centered["A_coord_local_xyz"], dtype=np.float64)
    # float64, (N_candidate, 3), 每个候选的世界 XYZ 体素尺寸, 单位 Å/voxel.
    voxel_size_xyz = np.asarray(centered["voxel_size_world"], dtype=np.float64)
    # float32, (N_atom,), 每个 A 原子到同一候选来源体素中心的最近距离; Inf 表示 5 Å 外或没有来源体素.
    distances = np.full(atom_xyz.shape[0], np.inf, dtype=np.float32)
    for entry_index in range(voxel_offsets.size - 1):
        # voxel_slice: 当前候选在来源体素表中的半开区间.
        voxel_slice = slice(
            int(voxel_offsets[entry_index]), int(voxel_offsets[entry_index + 1])
        )
        # atom_slice: 当前候选在 A 原子表中的半开区间.
        atom_slice = slice(
            int(atom_offsets[entry_index]), int(atom_offsets[entry_index + 1])
        )
        # float64, (K_source, 3), 来源体素中心相对 BOX 角点的世界 XYZ 坐标, 单位 Å.
        source_xyz = (local_zyx[voxel_slice][:, [2, 1, 0]] + 0.5) * voxel_size_xyz[entry_index][None, :]
        # float64, (N_A_entry, 3), A 原子相对同一 BOX 角点的世界 XYZ 坐标, 单位 Å.
        receptor_xyz = atom_xyz[atom_slice] * voxel_size_xyz[entry_index][None, :]
        if receptor_xyz.shape[0]:
            # float64, (N_A_entry,), 每个 A 原子到当前来源 blob 最近体素中心的世界距离, 单位 Å.
            distance, _ = cKDTree(source_xyz).query(
                receptor_xyz,
                k=1,
            )
            distances[atom_slice] = np.where(distance <= 5.0, distance, np.inf).astype(np.float32)
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
    # int64, (N_candidate + 1,), 以半开区间同步切分 A 原子距离与概率.
    offsets = np.asarray(atom_offsets, dtype=np.int64)
    # float32, (N_atom,), A 原子到所属来源 blob 的最近距离, 单位 Å; Inf 不参与求和.
    distance = np.asarray(atom_distance, dtype=np.float32)
    # float32, (N_atom,), 与 distance 逐原子对齐的配体概率.
    probability = np.asarray(atom_probability, dtype=np.float32)
    # float32, (N_candidate,), 每个候选的 Gaussian 加权正项.
    positive = np.zeros(offsets.size - 1, dtype=np.float32)
    # float32, (N_candidate,), 每个候选的 Gaussian 加权负项.
    negative = np.zeros_like(positive)
    # 逐候选使用 float64 归约, 保持校准搜索与正式推理完全相同的数值顺序.
    for index in range(positive.size):
        begin = int(offsets[index])
        end = int(offsets[index + 1])
        # float32, (N_A_entry,), 当前候选 A 原子的最近距离.
        candidate_distance = distance[begin:end]
        # bool, (N_A_entry,), True 表示该原子位于来源 blob 的 5 Å 距离截断内.
        included = np.isfinite(candidate_distance)
        # float64, (N_A_included,), 距离越近权重越高的 Gaussian 权重.
        weight = np.exp(
            -(candidate_distance[included].astype(np.float64) ** 2)
            / (2.0 * float(tau_angstrom) ** 2)
        )
        # float32, (N_A_included,), 与 weight 逐原子对齐的模型概率.
        candidate_probability = probability[begin:end][included]
        positive[index] = np.sum(weight * candidate_probability, dtype=np.float64)
        negative[index] = np.sum(weight * (1.0 - candidate_probability), dtype=np.float64)
    return positive, negative


def select_score_ratio_candidates(
    scores: np.ndarray,
    prefilter_eligible: np.ndarray,
    score_ratio_threshold: float,
) -> np.ndarray:
    """从固定预过滤总体中保留分数最高的一定比例候选.

    输入参数:
        - scores: float32 ``(N_candidate,)``, 当前 PDB 的候选分数.
        - prefilter_eligible: bool ``(N_candidate,)``, 固定来源体素数预过滤结果.
        - score_ratio_threshold: float, 当前 PDB 需要保留的候选比例.

    返回值:
        - selected: bool ``(N_candidate,)``, 恰有 ``floor(N*r+0.5)`` 个 True;
          ``N`` 是固定预过滤通过数, ``r`` 是 `score_ratio_threshold`.

    分数降序并列时保持候选原顺序. 本函数只实现比例选择; 最终
    ``min_voxels`` 在调用位置独立应用, 不改变比例总体和保留数量.
    """
    # float32, (N_candidate,), 当前 PDB 的候选排序分数.
    candidate_scores = np.asarray(scores, dtype=np.float32)
    # int64, (N_eligible,), 固定预过滤通过候选在原候选轴上的下标.
    eligible_indices = np.flatnonzero(
        np.asarray(prefilter_eligible, dtype=np.bool_)
    )
    # selected 与原候选轴对齐; 比例为零或没有合格候选时保持全 False.
    selected = np.zeros(candidate_scores.shape, dtype=np.bool_)
    # keep_count 按 half-up 最近整数规则由固定总体大小和比例共同确定.
    keep_count = int(
        np.floor(eligible_indices.size * float(score_ratio_threshold) + 0.5)
    )
    keep_count = min(max(keep_count, 0), int(eligible_indices.size))
    if keep_count == 0:
        return selected
    # stable_order 在分数并列时保留 eligible_indices 的原候选顺序.
    stable_order = np.argsort(
        -candidate_scores[eligible_indices],
        kind="stable",
    )
    # int64, (keep_count,), 当前 PDB 固定总体中正式保留的原候选下标.
    kept_indices = eligible_indices[stable_order[:keep_count]]
    selected[kept_indices] = True
    return selected


# ================================================================================================


def score_centered_candidates(
    centered: Mapping[str, np.ndarray],
    score_mode: str,
    score_parameters: Mapping[str, float],
) -> np.ndarray:
    """按显式模式计算每个 centered 候选的最终分数.

    输入参数:
        - centered: centered 产物字段映射; `source_probability_mean` 是 float32 `(N_candidate,)` 来源 blob 平均概率.
        - score_mode: 字符串; `basic` 与实验性 `basic_ratio` 使用来源平均概率, `gaussian` 叠加 A 原子 Gaussian 正负项.
        - score_parameters: 浮点数映射; Find 模式读取 tau 和两个 lambda, 来源均值模式不读取字段.

    返回值:
        - score: float32 `(N_candidate,)`, 与 `centered["source_blob_index"]` 逐候选对齐的分数.

    Find 模式还读取 :func:`build_gaussian_distance_table` 列出的体素和 A 原子字段.
    分数为 `source_mean + lambda_positive * positive - lambda_negative * negative`.
    正负项只累加 5 Å 内的 A 原子, 不按原子数归一化.
    """
    # float32, (N_candidate,), 每个来源 blob 在完整图概率图中的平均概率.
    source_mean = np.asarray(centered["source_probability_mean"], dtype=np.float32)
    if score_mode in {"basic", "basic_ratio"}:
        return source_mean.copy()
    tau_angstrom = float(score_parameters["tau_angstrom"])
    lambda_positive = float(score_parameters["lambda_positive"])
    lambda_negative = float(score_parameters["lambda_negative"])
    # atom_table: 以 A_offsets 对齐的 A 原子距离与概率表, 候选轴与 source_mean 一致.
    atom_table = build_gaussian_distance_table(centered)
    # positive, negative: float32, (N_candidate,), 每个候选的 Gaussian 加权正项与负项.
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
