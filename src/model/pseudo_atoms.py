# -*- coding: utf-8 -*-
"""
=============================================================================
PseudoAtomGenerator: 本文件掌管伪原子的生成、注入、移除等全过程
=============================================================================
为每个 BOX 在 forward 时现场生成一组伪原子，让它们在指定阶段和真实原子一起参与迭代，增强点分支对体素密度信息的摄取。
伪原子 `valid_mask=False`，不参与监督损失。
=============================================================================
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


# ============================================================
# 工具函数
# ============================================================
def _centered_world_to_local_voxel(
    coord_centered_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:
    """
    将以 BOX 中心为原点的世界坐标转为连续 voxel corner 坐标。

    输入参数:
        - coord_centered_world: np.ndarray, (N, 3), 以 BOX 中心为原点的世界坐标 (x,y,z)
        - voxel_size_world: np.ndarray, (3,), 每个 voxel 的世界尺寸 (x,y,z)
        - box_shape_zyx: np.ndarray, (3,), BOX 体素网格大小 (Z,Y,X)

    输出:
        - coord_local_voxel: np.ndarray, (N, 3), 连续 voxel 坐标 (x,y,z), corner 语义
    """
    # np.ndarray, (3,), BOX 尺寸 (x,y,z) 版本
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    return (coord_centered_world / voxel_size_world) + (0.5 * box_shape_xyz)


def _centered_world_to_world(
    coord_centered_world: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:
    """
    将以 BOX 中心为原点的世界坐标恢复为绝对世界坐标。

    输入参数:
        - coord_centered_world: np.ndarray, (N, 3), 以 BOX 中心为原点的世界坐标 (x,y,z)
        - box_origin_world: np.ndarray, (3,), BOX 左下近角点世界坐标 (x,y,z)
        - voxel_size_world: np.ndarray, (3,), voxel 世界尺寸 (x,y,z)
        - box_shape_zyx: np.ndarray, (3,), BOX 体素网格大小 (Z,Y,X)

    输出:
        - coord_world: np.ndarray, (N, 3), 绝对世界坐标 (x,y,z)
    """
    # np.ndarray, (3,), BOX 中心世界坐标
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    box_center_world = box_origin_world + 0.5 * box_shape_xyz * voxel_size_world
    return coord_centered_world + box_center_world


def _compute_is_in_core_box(
    coord_world: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:
    """
    判断每个点是否在 core BOX 内部(与真实原子相同逻辑)。

    输入参数:
        - coord_world: np.ndarray, (N, 3), 世界坐标 (x,y,z)
        - box_origin_world: np.ndarray, (3,), BOX 左下近角点世界坐标
        - voxel_size_world: np.ndarray, (3,), voxel 世界尺寸
        - box_shape_zyx: np.ndarray, (3,), BOX 体素网格大小 (Z,Y,X)

    输出:
        - is_in_core: np.ndarray, (N,), bool
    """
    # np.ndarray, (3,), BOX 最大边界世界坐标
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    box_max_world = box_origin_world + box_shape_xyz * voxel_size_world
    # np.ndarray, (N,), bool, 同时满足 >= origin 且 < max
    in_box = np.all(coord_world >= box_origin_world[None, :], axis=1)
    in_box &= np.all(coord_world < box_max_world[None, :], axis=1)
    return in_box


def _thin_by_clustering(
    pseudo_coords: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    对伪原子按 radius-关系使用贪心策略, 按顺序扫描，已被更早点"覆盖"(距离 < radius)的点跳过。

    输入参数:
        - pseudo_coords: np.ndarray, (M, 3), 待稀疏化的伪原子坐标
        - radius: float, 聚类半径

    输出:
        - keep_indices: np.ndarray, (K,), int64, 保留的伪原子在输入数组中的下标
    """
    if pseudo_coords.shape[0] == 0:
        return np.array([], dtype=np.int64)
    # cKDTree, 伪原子坐标的 KD 树
    tree = cKDTree(pseudo_coords)
    # set[int], 已被覆盖(待跳过)的下标集合
    covered = set()
    # list[int], 保留的代表下标
    keep = []
    for idx in range(pseudo_coords.shape[0]):
        if idx in covered:
            continue
        keep.append(idx)
        # list[int], 距离当前代表 < radius 的所有邻居下标
        neighbors = tree.query_ball_point(pseudo_coords[idx], radius)
        covered.update(neighbors)
    return np.array(keep, dtype=np.int64)






# ============================================================
# PseudoAtomGenerator 主类
# ============================================================
class PseudoAtomGenerator:
    def __init__(
        self,
        base_count: int,
        scale_factor: float,
        max_sample_rounds: int,
        init_feat_mode: str,
        init_feat_noise_std: float,
        neighbor_radius: float,
        enable_density_weighting: bool,
        density_channel_index: int,
        density_prob_base: float,
        delete_too_close_radius: float,
        delete_too_far_radius: float,
        lifecycle: list[bool],
    ) -> None:
        """
            为每个 BOX 现场生成伪原子并处理注入/移除。不含可学习参数。

            输入参数:
                - base_count: int, 固定基数 b
                - scale_factor: float, 比例系数 k, 伪原子目标数 = b + k × N_real
                - max_sample_rounds: int, 采样-删除循环最大轮次, 建议值 3
                - init_feat_mode: str, "zero" / "neighbor_mean"
                - init_feat_noise_std: float, 特征高斯噪声 σ, 建议值 0.0
                - neighbor_radius: float, "neighbor_mean" 模式邻域半径(Å), 建议值 3.0
                - enable_density_weighting: bool, 是否启用密度加权采样
                - density_channel_index: int, voxel_grid 中差图通道索引
                - density_prob_base: float, 密度概率基底, 建议值 0.1
                - delete_too_close_radius: float, r0: 对距参考点 < r0 的伪原子做聚类稀疏化, 0.0 不启用
                - delete_too_far_radius: float, r1: 删除距所有参考点 > r1 的伪原子, 0.0 不启用
                - lifecycle: list[bool], 长度=3, [embed_head, point_backbone, atom_head]
        """
        self.base_count = int(base_count)
        self.scale_factor = float(scale_factor)
        self.max_sample_rounds = int(max_sample_rounds)
        if init_feat_mode not in ("zero", "neighbor_mean"):
            raise ValueError(f"init_feat_mode 必须为 'zero' 或 'neighbor_mean', 当前为 '{init_feat_mode}'")
        self.init_feat_mode = str(init_feat_mode)
        self.init_feat_noise_std = float(init_feat_noise_std)
        self.neighbor_radius = float(neighbor_radius)
        self.enable_density_weighting = bool(enable_density_weighting)
        self.density_channel_index = int(density_channel_index)
        self.density_prob_base = float(density_prob_base)
        self.delete_too_close_radius = float(delete_too_close_radius)
        self.delete_too_far_radius = float(delete_too_far_radius)
        if len(lifecycle) != 3:
            raise ValueError(f"lifecycle 长度必须为 3, 当前为 {len(lifecycle)}")
        self.lifecycle = [bool(v) for v in lifecycle]






    # =================================================================================================================
    # 生成与初始化逻辑
    # =================================================================================================================
    # 根据密度信息算采样密度
    def _compute_density_probs(
        self,
        batch: dict[str, Any],
        box_idx: int,
        half_extent: np.ndarray,
        voxel_size: np.ndarray,
        box_shape_zyx: np.ndarray,
    ) -> np.ndarray:
        """
        从 voxel_grid 的差图通道计算展平的归一化密度概率。

        输入参数:
            - batch: dict, collate 后 batch
            - box_idx: int, 当前 BOX 索引
            - half_extent: np.ndarray, (3,), 半跨度
            - voxel_size: np.ndarray, (3,), voxel 尺寸
            - box_shape_zyx: np.ndarray, (3,), BOX 网格大小 (Z,Y,X)

        输出:
            - probs: np.ndarray, (D*H*W,), 归一化概率
        """
        # torch.Tensor, (D, H, W), 差图通道
        density_grid = batch["voxel_grid"][box_idx, self.density_channel_index].detach().cpu()
        # np.ndarray, (D*H*W,), 展平密度值
        density_flat = density_grid.numpy().ravel().astype(np.float64)
        # np.ndarray, (D*H*W,), 未归一化概率 = base + 密度值
        unnorm = np.maximum(self.density_prob_base + density_flat, 0.0)
        total = unnorm.sum()
        if total <= 0:
            # 退化为均匀
            return np.ones_like(unnorm) / float(unnorm.shape[0])
        return (unnorm / total).astype(np.float64)

    # 有密度信息时的采样
    def _density_weighted_sample(
        self,
        n: int,
        density_probs: np.ndarray,
        voxel_size: np.ndarray,
        box_shape_zyx: np.ndarray,
    ) -> np.ndarray:
        """
        按密度概率选体素，然后在体素内部均匀随机偏移。

        输入参数:
            - n: int, 需要采样的点数
            - density_probs: np.ndarray, (D*H*W,), 展平的归一化密度概率
            - voxel_size: np.ndarray, (3,), voxel 世界尺寸 (x,y,z)
            - box_shape_zyx: np.ndarray, (3,), BOX 体素网格大小 (Z,Y,X)

        输出:
            - coords: np.ndarray, (n, 3), centered_world 坐标
        """
        # int, D/H/W
        D, H, W = int(box_shape_zyx[0]), int(box_shape_zyx[1]), int(box_shape_zyx[2])
        total_voxels = D * H * W
        if total_voxels == 0:
            return np.empty((0, 3), dtype=np.float32)

        # np.ndarray, (n,), 按概率选中的展平体素索引
        chosen_flat = np.random.choice(total_voxels, size=n, replace=True, p=density_probs)

        # np.ndarray, (n,), 各轴的体素离散索引 (Z/Y/X)
        iz = chosen_flat // (H * W)
        iy = (chosen_flat % (H * W)) // W
        ix = chosen_flat % W

        # np.ndarray, (n, 3), corner 语义下的体素坐标 = idx + 0.5
        voxel_center_lv = np.stack([ix + 0.5, iy + 0.5, iz + 0.5], axis=1).astype(np.float32)
        # np.ndarray, (n, 3), 体素内均匀随机偏移 [-0.5, 0.5)
        jitter = np.random.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
        # np.ndarray, (n, 3), 最终 local_voxel 坐标
        local_voxel = voxel_center_lv + jitter

        # np.ndarray, (3,), BOX 尺寸 (x,y,z)
        box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
        # np.ndarray, (n, 3), 转为 centered_world
        centered_world = (local_voxel - 0.5 * box_shape_xyz) * voxel_size
        return centered_world

    # (包括无密度信息时的)采样位置
    def _sample_positions(
        self,
        n: int,
        half_extent: np.ndarray,
        density_probs: np.ndarray | None,
        voxel_size: np.ndarray,
        box_shape_zyx: np.ndarray,
    ) -> np.ndarray:
        """
        在 BOX 范围内采样 n 个点的 centered_world 坐标。

        输入参数:
            - n: int, 需要采样的点数
            - half_extent: np.ndarray, (3,), BOX 在 centered_world 坐标系中的半跨度 (x,y,z)
            - density_probs: np.ndarray | None, (D*H*W,), 展平的密度概率(仅密度加权时非空)
            - voxel_size: np.ndarray, (3,), voxel 世界尺寸 (x,y,z)
            - box_shape_zyx: np.ndarray, (3,), BOX 体素网格大小 (Z,Y,X)

        输出:
            - coords: np.ndarray, (n, 3), centered_world 坐标
        """
        if density_probs is not None:
            return self._density_weighted_sample(n, density_probs, voxel_size, box_shape_zyx)
        # np.ndarray, (n, 3), 在 [-half_extent, +half_extent] 内均匀采样
        return np.random.uniform(-half_extent, half_extent, size=(n, 3)).astype(np.float32)

    # 删除碰撞/孤立的伪原子
    def _apply_deletion(
        self,
        new_coords: np.ndarray,
        ref_coords: np.ndarray,
        real_coords: np.ndarray,
    ) -> np.ndarray:
        """
        对新采样的伪原子应用两个独立的删除过滤器。

        r0 (delete_too_close_radius): 对距参考点集 < r0 的伪原子做聚类稀疏化。
            先找出这些"太近"的伪原子，对它们按 r0 邻接关系做贪心聚类，每个簇只保留一个代表。
        r1 (delete_too_far_radius): 删除距所有参考点 > r1 的伪原子。

        输入参数:
            - new_coords: np.ndarray, (M, 3), 新采样的伪原子坐标
            - ref_coords: np.ndarray, (R, 3), 参考点集(真实原子 + 已采集伪原子)
            - real_coords: np.ndarray, (N, 3), 真实原子坐标

        输出:
            - filtered_coords: np.ndarray, (K, 3), 过滤后保留的伪原子坐标
        """
        if new_coords.shape[0] == 0:
            return new_coords

        # np.ndarray, (M,), bool, 总保留掩码
        keep = np.ones(new_coords.shape[0], dtype=bool)

        if self.delete_too_close_radius > 0 and ref_coords.shape[0] > 0:
            r0 = self.delete_too_close_radius
            # cKDTree, 参考点集的 KD 树
            ref_tree = cKDTree(ref_coords)
            # np.ndarray, (M,), float, 每个新点到最近参考点的距离
            dists_to_ref, _ = ref_tree.query(new_coords, k=1)
            # np.ndarray, (M,), bool, 距参考点 < r0 的点
            too_close_mask = dists_to_ref < r0
            # int, 有多少点太近
            n_too_close = int(too_close_mask.sum())
            if n_too_close > 0:
                # np.ndarray, (n_too_close,), int, "太近"点在 new_coords 中的下标
                too_close_indices = np.where(too_close_mask)[0]
                # np.ndarray, (n_too_close, 3), "太近"点的坐标
                too_close_coords = new_coords[too_close_indices]
                # np.ndarray, (K_keep,), int, 聚类后保留的"太近"点的局部下标
                local_keep = _thin_by_clustering(too_close_coords, r0)
                # set[int], 聚类后需要删除的"太近"点在 new_coords 中的全局下标
                too_close_remove = set(too_close_indices.tolist()) - set(too_close_indices[local_keep].tolist())
                for idx in too_close_remove:
                    keep[idx] = False

        if self.delete_too_far_radius > 0 and ref_coords.shape[0] > 0:
            r1 = self.delete_too_far_radius
            # cKDTree, 参考点集的 KD 树(可能与上面相同，但避免强制依赖)
            ref_tree = cKDTree(ref_coords)
            # np.ndarray, (M,), float, 每个新点到最近参考点的距离
            dists_to_ref, _ = ref_tree.query(new_coords, k=1)
            # np.ndarray, (M,), bool, 距所有参考点 > r1 的点
            too_far_mask = dists_to_ref > r1
            keep[too_far_mask] = False

        return new_coords[keep]

    # 初始化特征
    def _init_features(
        self,
        pseudo_coords: np.ndarray,
        real_coords: np.ndarray,
        real_feat: np.ndarray,
        feat_dim: int,
    ) -> np.ndarray:
        """
        为伪原子初始化特征。

        输入参数:
            - pseudo_coords: np.ndarray, (M, 3), 伪原子 centered_world 坐标
            - real_coords: np.ndarray, (N, 3), 真实原子坐标
            - real_feat: np.ndarray, (N, F), 真实原子特征
            - feat_dim: int, 特征维度 F

        输出:
            - pseudo_feat: np.ndarray, (M, F), 伪原子特征
        """
        n_pseudo = pseudo_coords.shape[0]
        if n_pseudo == 0:
            return np.empty((0, feat_dim), dtype=np.float32)

        if self.init_feat_mode == "zero":
            # np.ndarray, (M, F), 全零特征
            pseudo_feat = np.zeros((n_pseudo, feat_dim), dtype=np.float32)
        else:
            # "neighbor_mean"
            pseudo_feat = np.zeros((n_pseudo, feat_dim), dtype=np.float32)
            if real_coords.shape[0] > 0:
                # cKDTree, 真实原子坐标的 KD 树
                real_tree = cKDTree(real_coords)
                # list[list[int]], 每个伪原子在 neighbor_radius 内的真实原子下标
                neighbor_lists = real_tree.query_ball_point(pseudo_coords, self.neighbor_radius)
                for i, nbr_indices in enumerate(neighbor_lists):
                    if len(nbr_indices) > 0:
                        pseudo_feat[i] = real_feat[nbr_indices].mean(axis=0)

        if self.init_feat_noise_std > 0:
            pseudo_feat += np.random.randn(*pseudo_feat.shape).astype(np.float32) * self.init_feat_noise_std

        return pseudo_feat


    # 总函数
    def generate(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
            为整个 batch 生成伪原子(numpy/CPU 阶段)。

            输入参数:
                - batch: dict[str, Any], collate 后的 batch 字典

            输出:
                - pseudo_dict: dict[str, Any], 与真实原子字段同构的伪原子数据
                    - pseudo_coord_centered_world: torch.Tensor, (sumM, 3)
                    - pseudo_coord_local_voxel: torch.Tensor, (sumM, 3)
                    - pseudo_coord_world: torch.Tensor, (sumM, 3)
                    - pseudo_feat: torch.Tensor, (sumM, F)
                    - pseudo_valid_mask: torch.Tensor, (sumM,), 全 False
                    - pseudo_label: torch.Tensor, (sumM,), 全 0
                    - pseudo_is_in_core_box: torch.Tensor, (sumM,), bool
                    - pseudo_batch_index: torch.Tensor, (sumM,), long
                    - pseudo_counts: torch.Tensor, (B,), long
        """
        # int, batch 大小
        batch_size = int(batch["box_shape_zyx"].shape[0])
        # torch.Tensor, (sumN,), long, 真实原子的 batch 归属
        real_batch_index = batch["atom_batch_index"]
        # torch.Tensor, (sumN, 3), 真实原子以 BOX 中心为原点的世界坐标
        real_coord_cw = batch["atom_coord_centered_world"]
        # torch.Tensor, (sumN, F), 真实原子特征
        real_feat = batch["atom_feat"]
        # int, 原子特征维度
        feat_dim = int(real_feat.shape[1]) if real_feat.ndim == 2 else 0

        all_pseudo_cw: list[np.ndarray] = []       # list[(Mi, 3)]
        all_pseudo_lv: list[np.ndarray] = []       # list[(Mi, 3)]
        all_pseudo_w: list[np.ndarray] = []         # list[(Mi, 3)]
        all_pseudo_feat: list[np.ndarray] = []      # list[(Mi, F)]
        all_pseudo_core: list[np.ndarray] = []      # list[(Mi,)]
        pseudo_counts: list[int] = []

        for box_idx in range(batch_size):
            # --- 取当前 BOX 的几何元信息 ---
            # np.ndarray, (3,), BOX 体素网格大小 (Z,Y,X)
            box_shape_zyx = batch["box_shape_zyx"][box_idx].cpu().numpy().astype(np.float32)
            # np.ndarray, (3,), 每个 voxel 的世界尺寸 (x,y,z)
            voxel_size = batch["voxel_size_world"][box_idx].cpu().numpy().astype(np.float32)
            # np.ndarray, (3,), BOX 左下近角点世界坐标 (x,y,z)
            box_origin = batch["box_origin_world"][box_idx].cpu().numpy().astype(np.float32)
            # np.ndarray, (3,), BOX 尺寸 (x,y,z)
            box_shape_xyz = box_shape_zyx[[2, 1, 0]]
            # np.ndarray, (3,), BOX 在 centered_world 坐标系中的半跨度
            half_extent = 0.5 * box_shape_xyz * voxel_size

            # --- 取当前 BOX 的真实原子 ---
            # torch.Tensor, (N_i,), bool, 当前 BOX 的真实原子掩码
            real_mask_i = (real_batch_index == box_idx)
            # np.ndarray, (N_i, 3), 当前 BOX 真实原子 centered_world 坐标
            real_cw_i = real_coord_cw[real_mask_i].cpu().numpy().astype(np.float32)
            # np.ndarray, (N_i, F), 当前 BOX 真实原子特征
            real_feat_i = real_feat[real_mask_i].cpu().numpy().astype(np.float32)
            # int, 当前 BOX 真实原子数
            n_real_i = int(real_cw_i.shape[0])
            # int, 目标伪原子数
            n_target = self.base_count + int(self.scale_factor * n_real_i)

            if n_target <= 0:
                all_pseudo_cw.append(np.empty((0, 3), dtype=np.float32))
                all_pseudo_lv.append(np.empty((0, 3), dtype=np.float32))
                all_pseudo_w.append(np.empty((0, 3), dtype=np.float32))
                all_pseudo_feat.append(np.empty((0, feat_dim), dtype=np.float32))
                all_pseudo_core.append(np.empty((0,), dtype=bool))
                pseudo_counts.append(0)
                continue

            # --- 密度权重预计算 ---
            # np.ndarray | None, 密度加权概率图(仅 enable_density_weighting 时有值)
            density_probs = None
            if self.enable_density_weighting:
                density_probs = self._compute_density_probs(batch, box_idx, half_extent, voxel_size, box_shape_zyx)

            # --- 多轮采样-删除循环 ---
            # list[np.ndarray], 已采集的伪原子 centered_world 坐标
            collected_coords: list[np.ndarray] = []
            n_collected = 0
            for _ in range(self.max_sample_rounds):
                n_need = n_target - n_collected
                if n_need <= 0:
                    break
                # np.ndarray, (n_need, 3), 新撒的伪原子 centered_world 坐标
                new_coords = self._sample_positions(n_need, half_extent, density_probs, voxel_size, box_shape_zyx)

                # --- 删除机制 ---
                # np.ndarray, (n_ref, 3), 参考点集 = 真实原子 + 已采集伪原子
                if n_collected > 0:
                    ref_coords = np.concatenate([real_cw_i] + collected_coords, axis=0)
                else:
                    ref_coords = real_cw_i

                new_coords = self._apply_deletion(new_coords, ref_coords, real_cw_i)
                if new_coords.shape[0] > 0:
                    collected_coords.append(new_coords)
                    n_collected += new_coords.shape[0]

            # --- 截断到 n_target ---
            if n_collected > 0:
                # np.ndarray, (n_collected, 3), 拼接的全部伪原子 centered_world 坐标
                pseudo_cw_i = np.concatenate(collected_coords, axis=0)[:n_target]
            else:
                pseudo_cw_i = np.empty((0, 3), dtype=np.float32)
            # int, 当前 BOX 最终伪原子数
            n_pseudo_i = int(pseudo_cw_i.shape[0])

            # --- 坐标转换 ---
            # np.ndarray, (n_pseudo_i, 3), 连续 voxel 坐标
            pseudo_lv_i = _centered_world_to_local_voxel(pseudo_cw_i, voxel_size, box_shape_zyx)
            # np.ndarray, (n_pseudo_i, 3), 绝对世界坐标
            pseudo_w_i = _centered_world_to_world(pseudo_cw_i, box_origin, voxel_size, box_shape_zyx)

            # --- 初始特征 ---
            # np.ndarray, (n_pseudo_i, F), 伪原子初始特征
            pseudo_feat_i = self._init_features(pseudo_cw_i, real_cw_i, real_feat_i, feat_dim)

            # --- is_in_core_box ---
            # np.ndarray, (n_pseudo_i,), bool
            pseudo_core_i = _compute_is_in_core_box(pseudo_w_i, box_origin, voxel_size, box_shape_zyx)

            all_pseudo_cw.append(pseudo_cw_i)
            all_pseudo_lv.append(pseudo_lv_i)
            all_pseudo_w.append(pseudo_w_i)
            all_pseudo_feat.append(pseudo_feat_i)
            all_pseudo_core.append(pseudo_core_i)
            pseudo_counts.append(n_pseudo_i)

        # --- 组装并转 torch ---
        device = real_coord_cw.device
        dtype_float = real_coord_cw.dtype
        # torch.Tensor, (B,), long, 每个 BOX 的伪原子数
        pseudo_counts_t = torch.tensor(pseudo_counts, dtype=torch.long)
        # int, 伪原子总数
        total_pseudo = int(pseudo_counts_t.sum().item())
        # torch.Tensor, (sumM,), long, 伪原子 batch 归属
        pseudo_batch_index = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), pseudo_counts_t
        )

        if total_pseudo > 0:
            pseudo_cw_all = np.concatenate(all_pseudo_cw, axis=0)
            pseudo_lv_all = np.concatenate(all_pseudo_lv, axis=0)
            pseudo_w_all = np.concatenate(all_pseudo_w, axis=0)
            pseudo_feat_all = np.concatenate(all_pseudo_feat, axis=0)
            pseudo_core_all = np.concatenate(all_pseudo_core, axis=0)
        else:
            pseudo_cw_all = np.empty((0, 3), dtype=np.float32)
            pseudo_lv_all = np.empty((0, 3), dtype=np.float32)
            pseudo_w_all = np.empty((0, 3), dtype=np.float32)
            pseudo_feat_all = np.empty((0, feat_dim), dtype=np.float32)
            pseudo_core_all = np.empty((0,), dtype=bool)

        return {
            "pseudo_coord_centered_world": torch.tensor(pseudo_cw_all, dtype=dtype_float, device=device),
            "pseudo_coord_local_voxel": torch.tensor(pseudo_lv_all, dtype=dtype_float, device=device),
            "pseudo_coord_world": torch.tensor(pseudo_w_all, dtype=dtype_float, device=device),
            "pseudo_feat": torch.tensor(pseudo_feat_all, dtype=dtype_float, device=device),
            "pseudo_valid_mask": torch.zeros(total_pseudo, dtype=torch.bool, device=device),
            "pseudo_label": torch.zeros(total_pseudo, dtype=torch.long, device=device),
            "pseudo_is_in_core_box": torch.tensor(pseudo_core_all, dtype=torch.bool, device=device),
            "pseudo_batch_index": pseudo_batch_index.to(device),
            "pseudo_counts": pseudo_counts_t.to(device),
        }

    # =================================================================================================================







    # =================================================================================================================
    # 对原有 batch inject / remove
    # =================================================================================================================
    def inject(
        self,
        batch: dict[str, Any],
        pseudo_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], list[tuple[int, int]]]:
        """
        将伪原子注入 batch: 对于每个样本, 把伪原子对应的信息与特征追加在真实原子之后。

        输入参数:
            - batch: dict[str, Any], 当前 batch (浅拷贝后修改)
            - pseudo_dict: dict[str, Any], generate() 返回的伪原子数据

        输出:
            - new_batch: dict[str, Any], 注入后的 batch
            - split_info: list[tuple[int, int]], 每个 BOX 的 (n_real, n_pseudo)
        """
        # int, batch 大小
        batch_size = int(batch["box_shape_zyx"].shape[0])
        # torch.Tensor, (B,), long, 每个 BOX 的真实原子数
        real_counts = batch["atom_counts"] if "atom_counts" in batch else self._compute_counts(batch)
        # torch.Tensor, (B,), long, 每个 BOX 的伪原子数
        pseudo_counts = pseudo_dict["pseudo_counts"]
        # list[tuple[int, int]], 长度为B, 每个样本的 (n_real, n_pseudo)
        split_info = [
            (int(real_counts[i].item()), int(pseudo_counts[i].item()))
            for i in range(batch_size)
        ]


        # ------------------ 按 BOX 交错拼接: [real_0, pseudo_0, real_1, pseudo_1, ...] ------------------
        # 构建交错索引映射
        device = batch["atom_coord_centered_world"].device
        # 字段映射: batch 字段名 -> pseudo_dict 字段名
        field_map = {
            "atom_coord_centered_world": "pseudo_coord_centered_world",
            "atom_coord_local_voxel": "pseudo_coord_local_voxel",
            "atom_coord_world": "pseudo_coord_world",
            "atom_feat": "pseudo_feat",
            "atom_valid_mask": "pseudo_valid_mask",
            "atom_label": "pseudo_label",
            "atom_is_in_core_box": "pseudo_is_in_core_box",
        }
        # dict[str, Any], 浅拷贝
        new_batch = {**batch}

        # 逐字段交错拼接 
        real_offsets_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        pseudo_offsets_start = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            real_offsets_start[1:] = torch.cumsum(real_counts[:-1], dim=0)
            pseudo_offsets_start[1:] = torch.cumsum(pseudo_counts[:-1], dim=0)

        for batch_field, pseudo_field in field_map.items():
            if batch_field not in batch or pseudo_field not in pseudo_dict:
                continue
            real_tensor = batch[batch_field]
            pseudo_tensor = pseudo_dict[pseudo_field]
            # 交错拼接
            chunks: list[torch.Tensor] = []
            for i in range(batch_size):
                nr = int(real_counts[i].item())
                np_ = int(pseudo_counts[i].item())
                rs = int(real_offsets_start[i].item())
                ps = int(pseudo_offsets_start[i].item())
                if nr > 0:
                    chunks.append(real_tensor[rs : rs + nr])
                if np_ > 0:
                    chunks.append(pseudo_tensor[ps : ps + np_])
            if len(chunks) > 0:
                new_batch[batch_field] = torch.cat(chunks, dim=0)
            else:
                new_batch[batch_field] = real_tensor.new_empty((0,) + real_tensor.shape[1:] if real_tensor.ndim > 1 else (0,))

        # 更新索引辅助字段
        # torch.Tensor, (B,), long, 新的每 BOX 原子数
        new_counts = real_counts + pseudo_counts
        new_batch["atom_counts"] = new_counts
        # torch.Tensor, (B,), long, 新偏移
        new_batch["atom_offsets"] = torch.cumsum(new_counts, dim=0)
        # torch.Tensor, (sumN+sumM,), long, 新 batch 归属
        new_batch["atom_batch_index"] = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=device), new_counts
        )

        return new_batch, split_info



    def remove(
        self,
        batch: dict[str, Any],
        split_info: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """
        从 batch 中移除伪原子，恢复为仅含真实原子。

        输入参数:
            - batch: dict[str, Any], 含伪原子的 batch(由 def inject 生成, 同样本真实原子在前, 伪原子在后)
            - split_info: list[tuple[int, int]], 每个 BOX 的 (n_real, n_pseudo)

        输出:
            - new_batch: dict[str, Any], 仅含真实原子的 batch
        """
        device = batch["atom_coord_centered_world"].device
        batch_size = len(split_info)

        # torch.Tensor, (sumN+sumM,), bool, 真实原子掩码
        total_mixed = sum(nr + np_ for nr, np_ in split_info)
        real_mask = torch.zeros(total_mixed, dtype=torch.bool, device=device)
        offset = 0
        real_counts_list: list[int] = []
        for nr, np_ in split_info:
            real_mask[offset : offset + nr] = True
            offset += nr + np_
            real_counts_list.append(nr)

        # dict[str, Any], 浅拷贝
        new_batch = {**batch}
        # 需要裁剪的 atom 字段
        atom_fields = (
            "atom_coord_centered_world",
            "atom_coord_local_voxel",
            "atom_coord_world",
            "atom_feat",
            "atom_valid_mask",
            "atom_label",
            "atom_is_in_core_box",
        )
        for field_name in atom_fields:
            if field_name in batch and batch[field_name] is not None:
                new_batch[field_name] = batch[field_name][real_mask]

        # 恢复索引辅助字段
        real_counts_t = torch.tensor(real_counts_list, dtype=torch.long, device=device)
        new_batch["atom_counts"] = real_counts_t
        new_batch["atom_offsets"] = torch.cumsum(real_counts_t, dim=0)
        new_batch["atom_batch_index"] = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=device), real_counts_t
        )

        return new_batch






    # ------------------------------------- 工具函数 -------------------------------------
    @staticmethod
    def _compute_counts(batch: dict[str, Any]) -> torch.Tensor:
        """
        从 atom_batch_index 反推每个 BOX 的原子数(当 batch 中没有 atom_counts 时)。

        输入参数:
            - batch: dict[str, Any]

        输出:
            - counts: torch.Tensor, (B,), long
        """
        batch_index = batch["atom_batch_index"]
        batch_size = int(batch["box_shape_zyx"].shape[0])
        if batch_index.numel() == 0:
            return torch.zeros(batch_size, dtype=torch.long, device=batch_index.device)
        return torch.bincount(batch_index, minlength=batch_size)

    @staticmethod
    def build_real_mask(split_info: list[tuple[int, int]]) -> torch.Tensor:
        """
        从 split_info 构建真实原子掩码(供外部使用): atom head 结束后，会调用 build_real_mask(split_info), 裁剪到只剩真实原子

        输入参数:
            - split_info: list[tuple[int, int]], 每个 BOX 的 (n_real, n_pseudo)

        输出:
            - real_mask: torch.Tensor, (sumN+sumM,), bool
        """
        total = sum(nr + np_ for nr, np_ in split_info)
        mask = torch.zeros(total, dtype=torch.bool)
        offset = 0
        for nr, np_ in split_info:
            mask[offset : offset + nr] = True
            offset += nr + np_
        return mask
