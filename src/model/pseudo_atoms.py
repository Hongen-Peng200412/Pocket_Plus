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


def _tensor_to_numpy(
    tensor: torch.Tensor,
    dtype: Any | None = None,
) -> np.ndarray:
    tensor_cpu = tensor.detach().cpu()
    if tensor_cpu.is_floating_point() and tensor_cpu.dtype != torch.float32:
        tensor_cpu = tensor_cpu.to(dtype=torch.float32)

    array = tensor_cpu.numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


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
        recycle_policy: str = "pos",
    ) -> None:
        """
        为每个 BOX 生成伪原子，并统一管理注入 / 移除与 recycle 语义。
        输入参数:
            - base_count: int, 每个 BOX 的固定伪原子基数 b
            - scale_factor: float, 按真实原子数放大的比例系数 k
            - max_sample_rounds: int, “采样 → 删除过滤 → 补采”的最大循环轮次
            - init_feat_mode: str, 伪原子初始特征模式, 可选 "zero" / "neighbor_mean"
            - init_feat_noise_std: float, 伪原子特征初始化时叠加的高斯噪声标准差
            - neighbor_radius: float, "neighbor_mean" 模式的邻域半径, 单位 Å
            - enable_density_weighting: bool, 是否按 `voxel_grid` 密度图做加权采样
            - density_channel_index: int, 密度采样使用的 `voxel_grid` 通道索引
            - density_prob_base: float, 密度采样的概率基底项
            - delete_too_close_radius: float, 聚类稀疏化半径 r0, 0.0 表示关闭
            - delete_too_far_radius: float, 远距离删除半径 r1, 0.0 表示关闭

            - lifecycle: list[bool], 长度 = 3, 依次表示 [embed_head, point_backbone, atom_head] 三个阶段是否存在伪原子
            - recycle_policy: str, recycle 期间伪原子的跨轮保留策略
                - "non": 不保留任何伪原子状态; 下一轮重新采样位置并重新初始化属性
                - "pos": 仅保留位置与静态元数据; 下一轮按 `init_feat_mode` 重新初始化属性
                - "all": 保留位置、属性与 point recycle 隐状态; 语义上最接近旧版 persist
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
        # str, 标量, recycle 阶段的伪原子生命周期策略
        self.recycle_policy = str(recycle_policy).strip().lower()
        if self.recycle_policy not in {"non", "pos", "all", "fixed"}:
            raise ValueError(
                "recycle_policy must be one of "
                "['non', 'pos', 'all', 'fixed']"
            )


    def keep_features_across_recycle(self) -> bool:
        """
        返回当前策略是否跨 recycle 保留伪原子属性。
        `non`/`pos`/`all` 每轮重新初始化 feat; `fixed` 直接沿用上一轮的 pseudo_feat。

        输出:
            - keep_features: bool, 标量, 仅 `fixed` 为 True
        """
        return self.recycle_policy == "fixed"

    def keep_point_recycle_state_across_recycle(self) -> bool:
        """
        返回当前策略是否跨 recycle 保留 point 分支的伪原子隐状态。

        输出:
            - keep_point_state: bool, 标量, True 表示下一轮复用伪原子的 `point_recycle_out`
        """
        return self.recycle_policy in {"all", "fixed"}

    def keep_position_across_recycle(self) -> bool:
        """
        返回当前策略是否跨 recycle 保留伪原子位置。

        输出:
            - keep_position: bool, 标量, True 表示下一轮沿用上一轮的伪原子几何布局
        """
        return self.recycle_policy in {"pos", "all", "fixed"}



    # 这个函数最后阅读(调用很多前面的函数)
    def prepare_pseudo_dict_for_recycle(
        self,
        batch: dict[str, Any],
        cached_pseudo_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        根据当前 recycle policy 的策略, 依据 batch(可能用来提供近邻特征) 调整当前伪原子字典 cached_pseudo_dict。

        输入参数:
            - batch: dict[str, Any], 当前轮 real-only batch 视图
            - cached_pseudo_dict: dict[str, Any] | None, 上一轮或 embed 阶段缓存的伪原子模板

        输出:
            - pseudo_dict: dict[str, Any], 与当前 batch 对齐的本轮伪原子字典, 有如下选项：
                - self.keep_position_across_recycle() == False: 直接调用 generate(batch), 从头构建伪原子
                - self.keep_position_across_recycle() == True:
                    - self.keep_features_across_recycle() == True: 直接沿用缓存
                    - self.keep_features_across_recycle() == False: 重新初始化特征
        """
        if cached_pseudo_dict is None or not self.keep_position_across_recycle():
            return self.generate(batch)
        if self.keep_features_across_recycle():
            # fixed: 保留位置 + feat, 直接沿用缓存
            return {**cached_pseudo_dict}
        # pos / all: 保留位置, 重新初始化 feat
        return self.reinitialize_pseudo_features(batch=batch, pseudo_dict=cached_pseudo_dict)






    # =================================================================================================================
    # 生成与初始化逻辑
    # =================================================================================================================
    # 根据密度信息算采样密度  
    def _compute_density_probs(  # TODO: 之后考虑支持多个密度信息
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
        density_grid = batch["voxel_grid"][box_idx, self.density_channel_index]
        # np.ndarray, (D*H*W,), 展平密度值
        density_flat = _tensor_to_numpy(density_grid).ravel().astype(np.float64)
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
            - half_extent: np.ndarray, (3,), BOX 在 centered_world 坐标系中的半跨度 (x,y,z):= 0.5 * box_shape_xyz * voxel_size
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
        real_coords: np.ndarray,
    ) -> np.ndarray:
        """
        对新采样的伪原子应用三个独立的过滤阶段，确保空间分布合理且无碰撞。

        阶段 1 (防冲撞): 剔除距真实原子 < r0 的新候选点。
        阶段 2 (防越界): 剔除距所有真实原子 > r1 的新候选点，限制在实际受体表面包络内。
        阶段 3 (防扎堆): 对同批次幸存的新候选点，以 r0 为半径做内部贪心聚类稀疏化，每个相交簇仅留其一。

        输入参数:
            - new_coords: np.ndarray, (M, 3), 待评估的新采样的伪原子坐标
            - real_coords: np.ndarray, (N, 3), 真实原子坐标(防冲撞与防越界的参考基准)

        输出:
            - filtered_coords: np.ndarray, (K, 3), 经过严格过滤后保留的安全伪原子坐标
        """
        if new_coords.shape[0] == 0:
            return new_coords
        # np.ndarray, (M,), bool, 当前批次的全局保留掩码
        keep = np.ones(new_coords.shape[0], dtype=bool)

        # 阶段 1 & 2：基于真实原子的距离过滤 (冲撞过滤与极远过滤)
        if real_coords.shape[0] > 0:
            # cKDTree, 基于真实原子建立 KD 树
            real_tree = cKDTree(real_coords)
            # np.ndarray, (M,), float, 所有候选点到最近真实原子的距离
            dists_to_real, _ = real_tree.query(new_coords, k=1)
            
            # 阶段 1：冲撞过滤 (仅与真实原子)
            if self.delete_too_close_radius > 0:
                r0 = self.delete_too_close_radius
                too_close_mask = dists_to_real < r0
                keep[too_close_mask] = False

            # 阶段 2：极远过滤 (仅跟真实原子)
            if self.delete_too_far_radius > 0:
                r1 = self.delete_too_far_radius
                too_far_mask = dists_to_real > r1
                keep[too_far_mask] = False

        # 阶段 3：伪原子互斥聚类
        if self.delete_too_close_radius > 0:
            r0 = self.delete_too_close_radius
            # np.ndarray, (S,), int, 目前所有幸存候选点在 new_coords 下标
            survivors = np.where(keep)[0]
            if survivors.shape[0] > 0:
                # np.ndarray, (S, 3), 幸存点坐标
                survivor_coords = new_coords[survivors]
                # np.ndarray, (K_keep,), int, 按 r0 调用贪心选点后返回相对于 survivor_coords 的保留下标
                local_keep = _thin_by_clustering(survivor_coords, r0)
                
                # set[int], 在幸存者集中落选的集合
                local_drop = set(range(survivors.shape[0])) - set(local_keep.tolist())
                for drop_idx in local_drop:
                    orig_idx = survivors[drop_idx]
                    keep[orig_idx] = False

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
        按照 self.init_feat_mode 为伪原子初始化特征。

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


    # 关于伪原子初始化的总函数
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
            box_shape_zyx = _tensor_to_numpy(batch["box_shape_zyx"][box_idx], dtype=np.float32)
            # np.ndarray, (3,), 每个 voxel 的世界尺寸 (x,y,z)
            voxel_size = _tensor_to_numpy(batch["voxel_size_world"][box_idx], dtype=np.float32)
            # np.ndarray, (3,), BOX 左下近角点世界坐标 (x,y,z)
            box_origin = _tensor_to_numpy(batch["box_origin_world"][box_idx], dtype=np.float32)
            # np.ndarray, (3,), BOX 尺寸 (x,y,z)
            box_shape_xyz = box_shape_zyx[[2, 1, 0]]
            # np.ndarray, (3,), BOX 在 centered_world 坐标系中的半跨度
            half_extent = 0.5 * box_shape_xyz * voxel_size

            # --- 取当前 BOX 的真实原子 ---
            # torch.Tensor, (N_i,), bool, 当前 BOX 的真实原子掩码
            real_mask_i = (real_batch_index == box_idx)
            # np.ndarray, (N_i, 3), 当前 BOX 真实原子 centered_world 坐标
            real_cw_i = _tensor_to_numpy(real_coord_cw[real_mask_i], dtype=np.float32)
            # np.ndarray, (N_i, F), 当前 BOX 真实原子特征
            real_feat_i = _tensor_to_numpy(real_feat[real_mask_i], dtype=np.float32)
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
                # 默认生成 5 倍候选
                n_sample = n_need * 5
                # np.ndarray, (n_sample, 3), 新撒的伪原子 centered_world 坐标
                new_coords = self._sample_positions(n_sample, half_extent, density_probs, voxel_size, box_shape_zyx)

                # --- 删除机制 ---
                new_coords = self._apply_deletion(new_coords, real_cw_i)
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
    # 与真实原子的交互
    # =================================================================================================================
    # 基于 real-only batch 重新初始化 `pseudo_feat`
    def reinitialize_pseudo_features(
        self,
        batch: dict[str, Any],
        pseudo_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        在固定伪原子位置的前提下，基于当前 real-only batch 重新初始化 `pseudo_feat`。

        输入参数:
            - batch: dict[str, Any], 当前轮 real-only batch 视图
            - pseudo_dict: dict[str, Any], 已缓存的伪原子模板; 其坐标与计数将被原样复用

        输出:
            - refreshed_pseudo_dict: dict[str, Any], 坐标不变但 `pseudo_feat` 已重建的伪原子字典
        """
        # torch.Tensor, (sumN, F), 当前 real atom 的属性特征
        real_feat = batch["atom_feat"]
        # torch.Tensor, (sumM, 3), 已缓存的伪原子 centered_world 坐标
        pseudo_coord_cw = pseudo_dict["pseudo_coord_centered_world"]
        # int, 当前 batch 的 real atom 特征维度
        feat_dim = int(real_feat.shape[1]) if real_feat.ndim == 2 else 0

        device = real_feat.device
        dtype = real_feat.dtype
        n_pseudo_total = pseudo_coord_cw.shape[0]

        if self.init_feat_mode == "zero":
            pseudo_feat_t = torch.zeros((n_pseudo_total, feat_dim), dtype=dtype, device=device)
            if self.init_feat_noise_std > 0:
                pseudo_feat_t += torch.randn_like(pseudo_feat_t) * self.init_feat_noise_std
        else:
            # torch.Tensor, (sumN,), long, 当前 real atom 的 batch 索引
            real_batch_index = batch["atom_batch_index"]
            # torch.Tensor, (sumN, 3), 当前 real atom 的 centered_world 坐标
            real_coord_cw = batch["atom_coord_centered_world"]
            # torch.Tensor, (B,), long, 每个 BOX 的伪原子数
            pseudo_counts = pseudo_dict["pseudo_counts"]
            
            # list[np.ndarray], 每个 BOX 的伪原子特征
            refreshed_feat_list: list[np.ndarray] = []
            pseudo_offset = 0
            for box_idx in range(int(pseudo_counts.shape[0])):
                # int, 当前 BOX 的伪原子数
                n_pseudo_i = int(pseudo_counts[box_idx].item())
                # torch.Tensor, (N_i,), bool, 当前 BOX 的 real atom 掩码
                real_mask_i = real_batch_index == box_idx
                # np.ndarray, (N_i, 3), 当前 BOX 的 real atom centered_world 坐标
                real_cw_i = _tensor_to_numpy(real_coord_cw[real_mask_i], dtype=np.float32)
                # np.ndarray, (N_i, F), 当前 BOX 的 real atom 特征
                real_feat_i = _tensor_to_numpy(real_feat[real_mask_i], dtype=np.float32)
                # np.ndarray, (M_i, 3), 当前 BOX 的伪原子 centered_world 坐标
                pseudo_cw_i = _tensor_to_numpy(
                    pseudo_coord_cw[pseudo_offset : pseudo_offset + n_pseudo_i],
                    dtype=np.float32,
                )
                # np.ndarray, (M_i, F), 在固定坐标上重新初始化得到的伪原子特征
                refreshed_feat_i = self._init_features(
                    pseudo_coords=pseudo_cw_i,
                    real_coords=real_cw_i,
                    real_feat=real_feat_i,
                    feat_dim=feat_dim,
                )
                refreshed_feat_list.append(refreshed_feat_i)
                pseudo_offset += n_pseudo_i

            if refreshed_feat_list:
                pseudo_feat = np.concatenate(refreshed_feat_list, axis=0)
            else:
                pseudo_feat = np.empty((0, feat_dim), dtype=np.float32)
                
            pseudo_feat_t = torch.tensor(
                pseudo_feat,
                dtype=dtype,
                device=device,
            )

        refreshed_pseudo_dict = {**pseudo_dict}
        refreshed_pseudo_dict["pseudo_feat"] = pseudo_feat_t
        return refreshed_pseudo_dict


    # 从 batch 中提取伪原子
    def extract_pseudo_dict_from_batch(
        self,
        batch: dict[str, Any],
        split_info: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """
        从当前 mixed batch 中提取伪原子子字典(不该 batch)。

        输入参数:
            - batch: dict[str, Any], 当前 mixed batch; 原子布局满足 `[real_i, pseudo_i]` 交错约定
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`

        输出:
            - pseudo_dict: dict[str, Any], 与 `generate()` 返回格式一致的伪原子字典
        """
        device = batch["atom_coord_centered_world"].device
        batch_size = len(split_info)
        # torch.Tensor, (B,), long, 每个 BOX 的伪原子数
        pseudo_counts = torch.tensor([np_ for _, np_ in split_info], dtype=torch.long, device=device)
        # int, 当前 mixed batch 的总原子数
        total_mixed = sum(nr + np_ for nr, np_ in split_info)
        # torch.Tensor, (sumN+sumM,), bool, 伪原子掩码
        pseudo_mask = torch.zeros(total_mixed, dtype=torch.bool, device=device)
        offset = 0
        for nr, np_ in split_info:
            if np_ > 0:
                pseudo_mask[offset + nr : offset + nr + np_] = True
            offset += nr + np_

        pseudo_batch_index = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=device),
            pseudo_counts,
        )
        return {
            "pseudo_coord_centered_world": batch["atom_coord_centered_world"][pseudo_mask],
            "pseudo_coord_local_voxel": batch["atom_coord_local_voxel"][pseudo_mask],
            "pseudo_coord_world": batch["atom_coord_world"][pseudo_mask],
            "pseudo_feat": batch["atom_feat"][pseudo_mask],
            "pseudo_valid_mask": batch["atom_valid_mask"][pseudo_mask],
            "pseudo_label": batch["atom_label"][pseudo_mask],
            "pseudo_is_in_core_box": batch["atom_is_in_core_box"][pseudo_mask],
            "pseudo_batch_index": pseudo_batch_index,
            "pseudo_counts": pseudo_counts,
        }


    # 对原有 batch inject 伪原子
    def inject(
        self,
        batch: dict[str, Any],
        pseudo_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], list[tuple[int, int]]]:
        """
        将伪原子注入 real-only batch: 对于每个样本, 把伪原子对应的信息与特征追加在真实原子之后。

        输入参数:
            - batch: dict[str, Any], 当前 batch (浅拷贝后修改)
            - pseudo_dict: dict[str, Any], generate() 返回的伪原子数据

        输出:
            - new_batch: dict[str, Any], 注入后的 batch
            - split_info: list[tuple[int, int]], 每个 BOX 的 (n_real, n_pseudo)
        """
        # int, batch 大小
        batch_size = int(batch["box_shape_zyx"].shape[0])

        # 更新 batch["atom_counts"]
        # torch.Tensor, (B,), long, 每个 BOX 的真实原子数
        real_counts = batch.get("atom_counts")
        total_real = int(batch["atom_batch_index"].shape[0])  # 全batch真实原子总数
        if real_counts is None or int(real_counts.sum().item()) != total_real:
            real_counts = self._compute_counts(batch)

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
            "atom_global_indices": None,
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
            if batch_field not in batch:
                if batch_field == "atom_global_indices":
                    continue
                raise ValueError(f"batch_field {batch_field} not in batch")
            real_tensor = batch[batch_field]
            if pseudo_field is None:
                pseudo_tensor = torch.full(
                    (int(pseudo_counts.sum().item()),),
                    fill_value=-1,
                    dtype=real_tensor.dtype,
                    device=device,
                )
            else:
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


    # 对原有 batch remove 伪原子
    def remove(
        self,
        batch: dict[str, Any],
        split_info: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """
        从 mixed batch 中移除伪原子，恢复为仅含真实原子的 real-only batch。

        输入参数:
            - batch: dict[str, Any], 含伪原子的 batch(由 def inject 生成, 同样本真实原子在前, 伪原子在后)
            - split_info: list[tuple[int, int]], 每个 BOX 的 (n_real, n_pseudo)

        输出:
            - new_batch: dict[str, Any], 仅含真实原子的 real-only batch
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
            "atom_global_indices",
        )
        # 恢复属性
        for field_name in atom_fields:
            if field_name in batch and batch[field_name] is not None:
                new_batch[field_name] = batch[field_name][real_mask]
        # 恢复索引
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
        由 atom_batch_index 反过来导出每个 BOX 的原子数。

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

    @staticmethod
    def interleave_real_and_pseudo_tensor(
        real_tensor: torch.Tensor | None,
        split_info: list[tuple[int, int]],
        pseudo_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        按 `[real_i, pseudo_i]` 的交错布局，将 real-only 张量扩展为 mixed 张量。

        输入参数:
            - real_tensor: torch.Tensor | None, `(sumN_real, ...)` 或 `(sumN_real+sumM, ...)`, 当前 real-only 张量
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`
            - pseudo_tensor: torch.Tensor | None, `(sumM, ...)`, 需要写入 pseudo 槽位的张量; 为 None 时补零

        输出:
            - mixed_tensor: torch.Tensor | None, `(sumN_real+sumM, ...)`, 交错布局后的 mixed 张量
        """
        if real_tensor is None:
            return None
        total_real = sum(nr for nr, _ in split_info)
        total_pseudo = sum(np_ for _, np_ in split_info)
        total_mixed = total_real + total_pseudo
        if real_tensor.shape[0] == total_mixed and pseudo_tensor is None:
            return real_tensor
        if real_tensor.shape[0] != total_real:
            raise RuntimeError(
                f"Expected a real-only tensor of length {total_real} or a mixed tensor of length {total_mixed}, "
                f"but got {real_tensor.shape[0]}."
            )
        if pseudo_tensor is not None and pseudo_tensor.shape[0] != total_pseudo:
            raise RuntimeError(
                f"Expected a pseudo tensor of length {total_pseudo}, but got {pseudo_tensor.shape[0]}."
            )
        suffix_shape = tuple(real_tensor.shape[1:])
        chunks: list[torch.Tensor] = []
        real_offset = 0
        pseudo_offset = 0
        for nr, np_ in split_info:
            if nr > 0:
                chunks.append(real_tensor[real_offset : real_offset + nr])
                real_offset += nr
            if np_ > 0:
                if pseudo_tensor is None:
                    chunks.append(real_tensor.new_zeros((np_,) + suffix_shape))
                else:
                    chunks.append(
                        pseudo_tensor[pseudo_offset : pseudo_offset + np_].to(
                            device=real_tensor.device,
                            dtype=real_tensor.dtype,
                        )
                    )
                pseudo_offset += np_
        if chunks:
            return torch.cat(chunks, dim=0)
        return real_tensor.new_zeros((0,) + suffix_shape)

    @staticmethod
    def extract_pseudo_tensor_from_mixed(
        mixed_tensor: torch.Tensor | None,
        split_info: list[tuple[int, int]],
    ) -> torch.Tensor | None:
        """
        从 mixed 张量中抽取伪原子子张量。

        输入参数:
            - mixed_tensor: torch.Tensor | None, `(sumN_real+sumM, ...)`, 按 `[real_i, pseudo_i]` 交错布局的 mixed 张量
            - split_info: list[tuple[int, int]], 长度 = B, 每个 BOX 的 `(n_real, n_pseudo)`

        输出:
            - pseudo_tensor: torch.Tensor | None, `(sumM, ...)`, 按 BOX 顺序拼接后的伪原子子张量
        """
        if mixed_tensor is None:
            return None
        total_mixed = sum(nr + np_ for nr, np_ in split_info)
        if mixed_tensor.shape[0] != total_mixed:
            raise RuntimeError(
                f"Expected a mixed tensor of length {total_mixed}, but got {mixed_tensor.shape[0]}."
            )
        chunks: list[torch.Tensor] = []
        offset = 0
        for nr, np_ in split_info:
            if np_ > 0:
                chunks.append(mixed_tensor[offset + nr : offset + nr + np_])
            offset += nr + np_
        if chunks:
            return torch.cat(chunks, dim=0)
        return mixed_tensor.new_empty((0,) + tuple(mixed_tensor.shape[1:]))
