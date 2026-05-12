from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VoxelPredCacheData:
    """
    voxel-only 推理概率缓存数据。

    输入参数:
        - ligand_pred: np.ndarray, (D, H, W), 已按 hardmask 约束的 ligand 概率图
        - receptor_pred: np.ndarray | None, (D, H, W), 已按 hardmask 约束的 receptor 概率图
        - hardmask: np.ndarray, (D, H, W), 原子落点占据掩码(取值0或1)
        - resampled_emdb: np.ndarray, (D, H, W), 重采样真实密度图
        - origin: np.ndarray, (3,), 世界坐标原点(x,y,z)
        - voxel_size: np.ndarray, (3,), 体素大小(x,y,z)

        - gt_ligand_mask: np.ndarray | None, (D, H, W), GT ligand 二值掩码
        - gt_instance_label: np.ndarray | None, (D, H, W), GT instance 标签(0为背景)
        - meta: dict[str, Any], 缓存上下文信息
    """
    ligand_pred: np.ndarray
    receptor_pred: np.ndarray | None
    hardmask: np.ndarray
    resampled_emdb: np.ndarray
    origin: np.ndarray
    voxel_size: np.ndarray
    gt_ligand_mask: np.ndarray | None
    gt_instance_label: np.ndarray | None
    meta: dict[str, Any]



@dataclass
class VoxelCandidate:
    """
    单个 voxel 后处理候选区域。

    输入参数:
        - instance_id: int, 预测 instance ID, 从 1 开始连续
        - voxel_count: int, 当前 instance 内体素数量
        - score_mean: float, 当前 instance 内 score_map 平均值
        - score_max: float, 当前 instance 内 score_map 最大值
        - center_voxel_zyx: tuple[float, float, float], instance 中心体素坐标(z,y,x)
        - center_world_xyz: tuple[float, float, float], instance 中心世界坐标(x,y,z)
        - bbox_min_zyx: tuple[int, int, int], instance 包围盒最小体素坐标(z,y,x)
        - bbox_max_zyx: tuple[int, int, int], instance 包围盒最大体素坐标(z,y,x)
    """
    instance_id: int
    voxel_count: int
    score_mean: float
    score_max: float
    center_voxel_zyx: tuple[float, float, float]
    center_world_xyz: tuple[float, float, float]
    bbox_min_zyx: tuple[int, int, int]
    bbox_max_zyx: tuple[int, int, int]



@dataclass
class VoxelPostprocessResult:
    """
    单图 ligand 概率图后处理结果。

    输入参数:
        - prob_map: np.ndarray, (D, H, W), 输入 ligand 概率图
        - score_map: np.ndarray, (D, H, W), basic 或 advanced 评分图

        - binary_mask_raw: np.ndarray, (D, H, W), 初始阈值二值掩码
        - instance_label_raw: np.ndarray, (D, H, W), 初始连通域标签

        - binary_mask_filtered: np.ndarray, (D, H, W), 筛选后的二值掩码
        - instance_label_filtered: np.ndarray, (D, H, W), 筛选后的 instance 标签
        - candidates: list[VoxelCandidate], 可变长度, 最终候选区域列表
    """
    prob_map: np.ndarray
    score_map: np.ndarray
    binary_mask_raw: np.ndarray
    instance_label_raw: np.ndarray
    binary_mask_filtered: np.ndarray
    instance_label_filtered: np.ndarray
    candidates: list[VoxelCandidate]
