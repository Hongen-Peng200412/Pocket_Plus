# -*- coding: utf-8 -*-
"""
=============================================================================
Stage1EmbedHead
=============================================================================
embed head 前置模块：
    - 接收原子级点云数据 (atom_feat + atom_coord)
    - 在点上做共享编码 (若干层 Block)
    - 分叉：
        1. 体素路径: 将 per-atom hidden 通过 scatter_mean 聚合到体素网格 → voxel_pdb_embed_grid
        2. 点路径 (可选): 输出 per-atom hidden → embed_point_feat
    - 支持渐进感受野裁剪: 每经过一个 block, 可按配置缩小 buffer 半径
    - 作为过滤器: 输出裁剪后的原子字段, 下游模块直接使用

坐标约定同 box_point_dataset.py:
    - atom_coord_local_voxel: corner 语义连续体素坐标, 顺序 (x, y, z)
    - atom_coord_centered_world: 以 BOX 中心为原点的世界坐标, 顺序 (x, y, z)
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

_PTV3_HEAD_IMPORT_ERROR: Exception | None = None
try:
    from src.model.PTV3bakcbone.model import (
        Point,
        Block,
        resolve_act_layer,
    )
except Exception as exc:
    Point = None
    Block = None
    resolve_act_layer = None
    _PTV3_HEAD_IMPORT_ERROR = exc


# ============================================================
# Buffer 裁剪工具函数
# ============================================================
def trim_buffer_atoms(
    point_feat: torch.Tensor,
    point_coord: torch.Tensor,
    point_batch: torch.Tensor,
    point_offset: torch.Tensor,
    atom_is_in_core_box: torch.Tensor,
    atom_coord_local_voxel: torch.Tensor,
    box_shape_zyx: torch.Tensor,
    voxel_size_world: torch.Tensor,
    allowed_buffer_radius_world: float,
) -> dict[str, torch.Tensor]:
    """
    裁剪 buffer 原子: 保留 core box 内原子 + buffer 中距 core box 边界 ≤ allowed_buffer_radius_world 的原子。

    当 allowed_buffer_radius_world 为 inf 或足够大时, 等价于不裁剪。

    输入参数:
        - point_feat: torch.Tensor, (N, C), 当前 per-atom 特征
        - point_coord: torch.Tensor, (N, 3), 中心化世界坐标
        - point_batch: torch.Tensor, (N,), batch 索引
        - point_offset: torch.Tensor, (B,), PTV3 风格结束偏移
        - atom_is_in_core_box: torch.Tensor, (N,), bool, 是否在 core box 内
        - atom_coord_local_voxel: torch.Tensor, (N, 3), corner 语义下连续体素坐标 (x, y, z)
        - box_shape_zyx: torch.Tensor, (B, 3), 体素网格尺寸 (Z, Y, X)
        - voxel_size_world: torch.Tensor, (B, 3), 每个 voxel 在世界坐标系下的尺寸 (x, y, z)
        - allowed_buffer_radius_world: float, 允许的 buffer 半径(世界坐标, Å)

    输出:
        - result: dict[str, torch.Tensor]
            - "point_feat": (N', C)
            - "point_coord": (N', 3)
            - "point_batch": (N',)
            - "point_offset": (B,)
            - "atom_is_in_core_box": (N',) bool
            - "atom_coord_local_voxel": (N', 3)
            - "keep_mask": (N,) bool, 用于恢复原序列的掩码(keep_mask[i]代表之前的第i个元素是否被保留)
    """
    total_n = point_feat.shape[0]
    if total_n == 0:
        return {
            "point_feat": point_feat,
            "point_coord": point_coord,
            "point_batch": point_batch,
            "point_offset": point_offset,
            "atom_is_in_core_box": atom_is_in_core_box,
            "atom_coord_local_voxel": atom_coord_local_voxel,
            "keep_mask": atom_is_in_core_box,
        }

    # 对于无限半径或全部 core 原子的情况, 直接返回
    if allowed_buffer_radius_world == float("inf") or atom_is_in_core_box.all():
        # torch.Tensor, (N,), bool, 全 True 掩码
        keep_mask = torch.ones(total_n, dtype=torch.bool, device=point_feat.device)
        return {
            "point_feat": point_feat,
            "point_coord": point_coord,
            "point_batch": point_batch,
            "point_offset": point_offset,
            "atom_is_in_core_box": atom_is_in_core_box,
            "atom_coord_local_voxel": atom_coord_local_voxel,
            "keep_mask": keep_mask,
        }

    # 计算 buffer 原子到 core box 边界的距离
    # torch.Tensor, (N, 3), 每个原子对应的 voxel_size (按 batch 索引取)
    per_atom_voxel_size = voxel_size_world[point_batch.long()]  # (N, 3)
    # torch.Tensor, (N, 3), 每个原子的局部世界坐标 (以 BOX origin 即 corner 为原点)
    atom_local_world = atom_coord_local_voxel * per_atom_voxel_size
    # torch.Tensor, (N, 3), 每个原子对应的 BOX 尺寸 (世界坐标)
    per_atom_box_shape_xyz = box_shape_zyx[point_batch.long()][:, [2, 1, 0]].to(dtype=per_atom_voxel_size.dtype)
    box_extent_world = per_atom_box_shape_xyz * per_atom_voxel_size  # (N, 3)

    # torch.Tensor, (N, 3), 原子到 core box 各轴下界和上界的距离(负值表示在 box 内)
    dist_low = -atom_local_world                   # 距离下界: 若 < 0 则在 box 内
    dist_high = atom_local_world - box_extent_world  # 距离上界: 若 < 0 则在 box 内
    dist_to_box = torch.maximum(dist_low, dist_high)
    # torch.Tensor, (N,), 到 core box 的 L∞ 距离(各轴最大距离)
    dist_to_core = dist_to_box.max(dim=-1).values
    # torch.Tensor, (N,), bool, core 内原子 或 buffer 距离在允许范围内的原子
    keep_mask = atom_is_in_core_box | (dist_to_core <= allowed_buffer_radius_world)

    if keep_mask.all():
        return {
            "point_feat": point_feat,
            "point_coord": point_coord,
            "point_batch": point_batch,
            "point_offset": point_offset,
            "atom_is_in_core_box": atom_is_in_core_box,
            "atom_coord_local_voxel": atom_coord_local_voxel,
            "keep_mask": keep_mask,
        }

    # 裁剪
    # torch.Tensor, (N',), 保留的原子索引
    kept_feat = point_feat[keep_mask]
    kept_coord = point_coord[keep_mask]
    kept_batch = point_batch[keep_mask]
    kept_core = atom_is_in_core_box[keep_mask]
    kept_local_voxel = atom_coord_local_voxel[keep_mask]

    # 重建 offset: 统计每个 batch 中保留了多少原子
    batch_size = int(point_offset.shape[0])
    # torch.Tensor, (B,), int64, 每个 batch 保留的原子数
    new_counts = torch.zeros(batch_size, dtype=torch.long, device=point_feat.device)
    new_counts.scatter_add_(
        dim=0,
        index=kept_batch.long(),
        src=torch.ones_like(kept_batch, dtype=torch.long),
    )
    # torch.Tensor, (B,), int64, PTV3 风格结束偏移
    new_offset = new_counts.cumsum(dim=0)

    return {
        "point_feat": kept_feat,
        "point_coord": kept_coord,
        "point_batch": kept_batch,
        "point_offset": new_offset,
        "atom_is_in_core_box": kept_core,
        "atom_coord_local_voxel": kept_local_voxel,
        "keep_mask": keep_mask,
    }

def scatter_to_voxel_grid(
    point_feat: torch.Tensor,
    atom_coord_local_voxel: torch.Tensor,
    point_batch: torch.Tensor,
    box_shape_zyx: torch.Tensor,
    batch_size: int,
    reduce: str,
) -> torch.Tensor:
    """
    将 per-atom 特征 scatter 到体素网格上。

    输入参数:
        - point_feat: torch.Tensor, (N, C), per-atom 特征
        - atom_coord_local_voxel: torch.Tensor, (N, 3), 连续体素坐标 (x, y, z), corner 语义
        - point_batch: torch.Tensor, (N,), batch 索引
        - box_shape_zyx: torch.Tensor, (B, 3), 体素网格尺寸 (Z, Y, X)
        - batch_size: int, batch 大小
        - reduce: str, 聚合方式, "mean" 或 "sum"

    输出:
        - voxel_grid: torch.Tensor, (B, C, D, H, W), 聚合后的体素网格
    """
    channels = int(point_feat.shape[1])
    # int, 体素网格各维度大小 (假设 batch 内所有 BOX 形状一致)
    d_val = int(box_shape_zyx[0, 0].item())
    h_val = int(box_shape_zyx[0, 1].item())
    w_val = int(box_shape_zyx[0, 2].item())
    total_voxels = batch_size * d_val * h_val * w_val

    if point_feat.shape[0] == 0:
        return point_feat.new_zeros((batch_size, channels, d_val, h_val, w_val))

    # torch.Tensor, (N, 3), int64, 将 corner 语义坐标 floor 到体素格点索引
    voxel_idx_xyz = atom_coord_local_voxel.floor().long()
    
    # torch.Tensor, (N,), bool, 如果超出索引，直接丢掉
    valid_mask = (
        (voxel_idx_xyz[:, 0] >= 0) & (voxel_idx_xyz[:, 0] < w_val) &
        (voxel_idx_xyz[:, 1] >= 0) & (voxel_idx_xyz[:, 1] < h_val) &
        (voxel_idx_xyz[:, 2] >= 0) & (voxel_idx_xyz[:, 2] < d_val)
    )
    if not valid_mask.all():
        voxel_idx_xyz = voxel_idx_xyz[valid_mask]
        point_feat = point_feat[valid_mask]
        point_batch = point_batch[valid_mask]

    # torch.Tensor, (N,), int64, 线性索引 = batch * (D*H*W) + z * (H*W) + y * W + x
    linear_idx = (
        point_batch.long() * (d_val * h_val * w_val)
        + voxel_idx_xyz[:, 2] * (h_val * w_val)
        + voxel_idx_xyz[:, 1] * w_val
        + voxel_idx_xyz[:, 0]
    )

    # torch.Tensor, (total_voxels, C), 累加结果
    voxel_sum = point_feat.new_zeros((total_voxels, channels))
    # torch.Tensor, (N, C), 将 linear_idx 扩展到 C 维做 scatter_add
    idx_expanded = linear_idx.unsqueeze(1).expand(-1, channels)
    voxel_sum.scatter_add_(dim=0, index=idx_expanded, src=point_feat)

    if reduce == "mean":
        # torch.Tensor, (total_voxels, 1), 每个体素接收的原子计数
        voxel_count = point_feat.new_zeros((total_voxels, 1))
        voxel_count.scatter_add_(
            dim=0,
            index=linear_idx.unsqueeze(1),
            src=torch.ones(point_feat.shape[0], 1, device=point_feat.device, dtype=point_feat.dtype),
        )
        # 避免除零
        voxel_sum = voxel_sum / voxel_count.clamp(min=1.0)

    # torch.Tensor, (B, C, D, H, W), reshape 到体素网格
    return voxel_sum.view(batch_size, d_val, h_val, w_val, channels).permute(0, 4, 1, 2, 3).contiguous()




# ============================================================
# Stage1 Embed Head 主模块
# ============================================================
class Stage1EmbedHead(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        embed_hidden_dim: int,
        embed_voxel_out_channels: int,
        embed_point_out_channels: int,
        num_trunk_blocks: int,
        num_voxel_blocks: int,
        num_point_blocks: int,
        trunk_buffer_radii: Sequence[float],
        voxel_buffer_radii: Sequence[float],
        point_buffer_radii: Sequence[float],
        num_heads: int,
        patch_size: int,
        serialization_orders: Sequence[str],
        shuffle_orders: bool,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        scatter_reduce: str,
        ffn_type: str,
        mlp_ratio: int,
        act_layer_name: str,
        point_grid_size: float,
        cpe_impl: str,
        cpe_kernel_size: int,
        cpe_receptive_field: float,
        pointconv_block_max_neighbors: int,
        drop_path: float,
        pre_norm: bool,

        embed_residual_enabled: bool,    # bool, 是否启用残差融合
        embed_point_gate_enabled: bool,   # bool, point 路径 gate 是否可学习 (仅 embed_residual_enabled=True 时有效)
        embed_voxel_gate_enabled: bool,   # bool, voxel 路径 gate 是否可学习 (仅 embed_residual_enabled=True 时有效)
    ) -> None:
        """
        Stage1 embed head 前置模块, 将原子级点云编码为体素网格嵌入特征和(可选的)点特征。
        作为过滤器: 输出裁剪后的原子字段 + 全局 keep_mask, 下游模块使用裁剪后的数据。

        初始化参数:
            - atom_feature_dim: int, 原子原始特征维度, 建议值 49
            - embed_hidden_dim: int, 共享 trunk 的隐藏通道数, 建议值 64
            - embed_voxel_out_channels: int, 聚合到体素网格的输出通道数, 建议值 16
            - embed_point_out_channels: int, 输出给点分支的 per-atom 特征通道数, 0 表示不输出

            - num_trunk_blocks: int, 共享 trunk 的 Block 数, 建议值 3
            - num_voxel_blocks: int, 体素分支专用 block 数, 建议值 2
            - num_point_blocks: int, 点分支专用 block 数, 建议值 2
            - trunk_buffer_radii: Sequence[float], 长度=num_trunk_blocks, 每个 trunk block 后允许的 buffer 半径(Å)
            - voxel_buffer_radii: Sequence[float], 长度=num_voxel_blocks, 每个 voxel block 后允许的 buffer 半径(Å)
            - point_buffer_radii: Sequence[float], 长度=num_point_blocks, 每个 point block 后允许的 buffer 半径(Å)

            - num_heads: int, 注意力头数
            - patch_size: int, SerializedAttention 的 patch size
            - serialization_orders: Sequence[str], 序列化顺序
            - shuffle_orders: bool, 是否在每次 forward 随机打乱序列化顺序
            - qkv_bias: bool, QKV 线性层是否带 bias
            - qk_scale: float | None, QK 缩放因子
            - attn_drop: float, 注意力 dropout
            - proj_drop: float, 输出投影 dropout
            - enable_rpe: bool, 是否启用相对位置编码
            - enable_flash: bool, 是否启用 flash attention
            - upcast_attention: bool, 是否在注意力计算前上转精度
            - upcast_softmax: bool, 是否在 softmax 前上转精度
            - scatter_reduce: str, 聚合方式, "mean" 或 "sum"
            - ffn_type: str, FFN 类型, "mlp"/"gated"/"none"
            - mlp_ratio: int, FFN 隐藏层膨胀倍率(仅 ffn_type != "none" 时生效)
            - act_layer_name: str, 激活函数名称, 支持 "gelu"/"silu"/"relu"/"leakyrelu"
            - point_grid_size: float, 内部 Point 对象的离散化粒度
            - cpe_impl: str, CPE 实现方式 "none"/"sparseconv"/"pointconv"
            - cpe_kernel_size: int, sparseconv CPE 卷积核大小(仅 cpe_impl="sparseconv" 时生效), 建议值 5
            - cpe_receptive_field: float, pointconv CPE 世界坐标感受野半径(Å)(仅 cpe_impl="pointconv" 时生效), 建议值 2.0
            - pointconv_block_max_neighbors: int, pointconv CPE 每个点最大邻居数(仅 cpe_impl="pointconv" 时生效), 建议值 16
            - drop_path: float, 随机深度(stochastic depth) drop 概率, 建议值 0.0
            - pre_norm: bool, 是否使用预归一化(True: pre-LN, False: post-LN), 建议值 True

            - embed_residual_enabled: bool, 是否启用残差融合
            - embed_point_gate_enabled: bool, point 路径 gate 是否可学习 (仅 embed_residual_enabled=True 时有效)
            - embed_voxel_gate_enabled: bool, voxel 路径 gate 是否可学习 (仅 embed_residual_enabled=True 时有效)

        前向输入:
            - atom_feat, atom_coord_centered_world, atom_batch_index, atom_offsets,
              atom_coord_local_voxel, box_shape_zyx, voxel_size_world, atom_is_in_core_box

        前向输出:
            - dict:
                - "voxel_pdb_embed_grid": (B, embed_voxel_out_channels, D, H, W)
                - "embed_point_feat": (sumN', embed_point_out_channels) 或 None
                - "atom_feat": (sumN', F_atom), 裁剪后的原子特征
                - "atom_coord_centered_world": (sumN', 3)
                - "atom_batch_index": (sumN',)
                - "atom_offsets": (B,)
                - "atom_coord_local_voxel": (sumN', 3)
                - "atom_is_in_core_box": (sumN',) bool
                - "global_keep_mask": (sumN,) bool, 从原始 sumN 到最终裁剪后的掩码(0代表被剪掉)
        """
        super().__init__()

        # 参数校验
        if num_trunk_blocks > 0 and len(trunk_buffer_radii) != num_trunk_blocks:
            raise ValueError(f"trunk_buffer_radii 长度({len(trunk_buffer_radii)}) != num_trunk_blocks({num_trunk_blocks})")
        if len(voxel_buffer_radii) != num_voxel_blocks:
            raise ValueError(f"voxel_buffer_radii 长度({len(voxel_buffer_radii)}) != num_voxel_blocks({num_voxel_blocks})")
        if len(point_buffer_radii) != num_point_blocks:
            raise ValueError(f"point_buffer_radii 长度({len(point_buffer_radii)}) != num_point_blocks({num_point_blocks})")
        if Block is None:
            raise ImportError("Stage1EmbedHead 需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR
        if resolve_act_layer is None:
            raise ImportError("解析激活函数需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR

        self.atom_feature_dim = int(atom_feature_dim)
        self.embed_hidden_dim = int(embed_hidden_dim)
        self.embed_voxel_out_channels = int(embed_voxel_out_channels)
        self.embed_point_out_channels = int(embed_point_out_channels)
        self.num_trunk_blocks = int(num_trunk_blocks)
        self.num_voxel_blocks = int(num_voxel_blocks)
        self.num_point_blocks = int(num_point_blocks)
        self.trunk_buffer_radii = tuple(float(r) for r in trunk_buffer_radii)
        self.voxel_buffer_radii = tuple(float(r) for r in voxel_buffer_radii)
        self.point_buffer_radii = tuple(float(r) for r in point_buffer_radii)
        self.serialization_orders = tuple(str(o) for o in serialization_orders)
        self.shuffle_orders = bool(shuffle_orders)
        self.scatter_reduce = str(scatter_reduce)
        self.has_point_output = self.embed_point_out_channels > 0
        self.point_grid_size = float(point_grid_size)
        self.cpe_impl = str(cpe_impl)
        self.embed_residual_enabled = bool(embed_residual_enabled)

        # type, 激活函数类
        act_cls = resolve_act_layer(str(act_layer_name))

        # --- 残差融合 ---
        if self.embed_residual_enabled:
            # point 路径: linear(atom_feature_dim → embed_point_out_channels) + gate
            if self.has_point_output:
                self.embed_point_add_proj = nn.Linear(self.atom_feature_dim, self.embed_point_out_channels)
                if embed_point_gate_enabled:
                    self.embed_point_gate = nn.Parameter(torch.tensor(0.1))
                else:
                    self.register_buffer("embed_point_gate", torch.tensor(1.0))
            else:
                self.embed_point_add_proj = None
                self.register_buffer("embed_point_gate", torch.tensor(1.0))

            # voxel 路径: linear(atom_feature_dim → embed_voxel_out_channels) + gate
            self.embed_voxel_add_proj = nn.Linear(self.atom_feature_dim, self.embed_voxel_out_channels)
            if embed_voxel_gate_enabled:
                self.embed_voxel_gate = nn.Parameter(torch.tensor(0.1))
            else:
                self.register_buffer("embed_voxel_gate", torch.tensor(1.0))
        else:
            # 不启用残差: 所有投影层为 None，gate 为 buffer(1.0)
            self.embed_point_add_proj = None
            self.register_buffer("embed_point_gate", torch.tensor(1.0))
            self.embed_voxel_add_proj = None
            self.register_buffer("embed_voxel_gate", torch.tensor(1.0))

        # --- 通用 Block 参数(所有 block 共享) ---
        _block_kwargs = dict(
            channels=self.embed_hidden_dim,
            num_heads=int(num_heads),
            patch_size=int(patch_size),
            cpe_impl=self.cpe_impl,
            cpe_kernel_size=int(cpe_kernel_size),
            cpe_receptive_field=float(cpe_receptive_field),
            pointconv_block_max_neighbors=int(pointconv_block_max_neighbors),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(proj_drop),
            enable_rpe=bool(enable_rpe),
            enable_flash=bool(enable_flash),
            upcast_attention=bool(upcast_attention),
            upcast_softmax=bool(upcast_softmax),
            ffn_type=str(ffn_type),
            mlp_ratio=int(mlp_ratio),
            act_layer=act_cls,
            pre_norm=bool(pre_norm),
            drop_path=float(drop_path),
        )

        # --- 输入投影 ---
        # nn.Sequential, (sumN, atom_feature_dim) -> (sumN, embed_hidden_dim), 原子特征到隐藏维度的投影
        input_proj_hidden = max(int(embed_hidden_dim), int(atom_feature_dim))
        self.input_proj = nn.Sequential(
            nn.Linear(int(atom_feature_dim), input_proj_hidden),
            nn.LayerNorm(input_proj_hidden),
            act_cls(),
            nn.Linear(input_proj_hidden, self.embed_hidden_dim),
        )

        # ------------------ 共享 Trunk blocks ------------------
        self.trunk_blocks = nn.ModuleList()
        for block_idx in range(self.num_trunk_blocks):
            self.trunk_blocks.append(
                Block(
                    order_index=int(block_idx % len(self.serialization_orders)),
                    **_block_kwargs,
                )
            )

        # ------------------ 体素专用 blocks ------------------
        self.voxel_blocks = nn.ModuleList()
        for block_idx in range(self.num_voxel_blocks):
            self.voxel_blocks.append(
                Block(
                    order_index=int((self.num_trunk_blocks + block_idx) % len(self.serialization_orders)),
                    **_block_kwargs,
                )
            )
        # nn.Sequential, (sumN, embed_hidden_dim) -> (sumN, embed_voxel_out_channels), 体素输出投影
        self.voxel_out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_hidden_dim),
            nn.Linear(self.embed_hidden_dim, self.embed_voxel_out_channels),
        )

        # ------------------ 点分支专用 blocks (可选) ------------------
        if self.has_point_output:
            self.point_blocks = nn.ModuleList()
            for block_idx in range(self.num_point_blocks):
                self.point_blocks.append(
                    Block(
                        order_index=int((self.num_trunk_blocks + block_idx) % len(self.serialization_orders)),
                        **_block_kwargs,
                    )
                )
            # nn.Sequential, (sumN, embed_hidden_dim) -> (sumN, embed_point_out_channels), 点输出投影
            self.point_out_proj = nn.Sequential(
                nn.LayerNorm(self.embed_hidden_dim),
                nn.Linear(self.embed_hidden_dim, self.embed_point_out_channels),
            )
        else:
            self.point_blocks = nn.ModuleList()
            self.point_out_proj = None


    def _make_point_and_serialize(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        batch: torch.Tensor,
        offset: torch.Tensor,
    ) -> Any:
        """
        构建 Point 对象并执行序列化(用于每次裁剪后重建)。

        输入参数:
            - feat: torch.Tensor, (N, C)
            - coord: torch.Tensor, (N, 3)
            - batch: torch.Tensor, (N,)
            - offset: torch.Tensor, (B,)

        输出:
            - point: Point
        """
        point = Point({
            "feat": feat,
            "coord": coord,
            "batch": batch,
            "offset": offset,
            "grid_size": self.point_grid_size,
        })
        point.serialization(order=self.serialization_orders, shuffle_orders=self.shuffle_orders)
        return point


    def _run_blocks_with_trim(
        self,
        point: Any,
        blocks: nn.ModuleList,
        buffer_radii: tuple[float, ...],
        cur_coord: torch.Tensor,
        cur_batch: torch.Tensor,
        cur_offset: torch.Tensor,
        cur_core: torch.Tensor,  # 当前 atom_is_in_core_box
        cur_local_voxel: torch.Tensor,
        box_shape_zyx: torch.Tensor,
        voxel_size_world: torch.Tensor,
        global_keep_mask: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行一系列 Block 并在每个 block 后按配置裁剪 buffer 原子。

        输入参数:
            - point: Point, 当前点对象
            - blocks: nn.ModuleList, 要执行的 Block 列表
            - buffer_radii: tuple[float, ...], 与 blocks 等长的 buffer 半径列表
            - cur_*: 当前状态的各字段
            - box_shape_zyx, voxel_size_world: batch-level 参数
            - global_keep_mask: torch.Tensor, (sumN_original,), bool, 当前阶段前的全局掩码

        输出:
            - (point, cur_coord, cur_batch, cur_offset, cur_core, cur_local_voxel, global_keep_mask)
        """
        for block_idx, block in enumerate(blocks):
            point = block(point)
            allowed_r = buffer_radii[block_idx]
            if allowed_r < float("inf"):
                trim_result = trim_buffer_atoms(
                    point_feat=point.feat,
                    point_coord=cur_coord,
                    point_batch=cur_batch,
                    point_offset=cur_offset,
                    atom_is_in_core_box=cur_core,
                    atom_coord_local_voxel=cur_local_voxel,
                    box_shape_zyx=box_shape_zyx,
                    voxel_size_world=voxel_size_world,
                    allowed_buffer_radius_world=allowed_r,
                )
                # torch.Tensor, (N_current,), bool, 本次裁剪的局部掩码
                local_mask = trim_result["keep_mask"]
                if not local_mask.all():
                    cur_batch = trim_result["point_batch"]
                    cur_offset = trim_result["point_offset"]
                    cur_core = trim_result["atom_is_in_core_box"]
                    cur_local_voxel = trim_result["atom_coord_local_voxel"]
                    cur_coord = trim_result["point_coord"]
                    # 更新全局掩码: global_keep_mask 中当前为 True 的位置, 只有那些 local_mask 也为 True 的才保留
                    active_positions = global_keep_mask.nonzero(as_tuple=True)[0]  # 返回 global_keep_mask 中为 True 的索引
                    global_keep_mask[active_positions[~local_mask]] = False
                    # 重建 Point 对象
                    point = self._make_point_and_serialize(
                        feat=trim_result["point_feat"],
                        coord=cur_coord,
                        batch=cur_batch,
                        offset=cur_offset,
                    )
        return point, cur_coord, cur_batch, cur_offset, cur_core, cur_local_voxel, global_keep_mask


    def forward(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        atom_coord_local_voxel: torch.Tensor,
        box_shape_zyx: torch.Tensor,
        voxel_size_world: torch.Tensor,
        atom_is_in_core_box: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """
        Embed head 前向: 对原子做共享编码, 然后分叉输出体素嵌入网格和(可选的)点特征。
        同时返回裁剪后的原子字段, 供 stage1_model 更新 batch 使用。

        输入参数:
            - atom_feat: torch.Tensor, (sumN, F_atom), batch 内全部原子的原始特征(49)
            - atom_coord_centered_world: torch.Tensor, (sumN, 3), 以 BOX 中心为原点的世界坐标
            - atom_batch_index: torch.Tensor, (sumN,), 每个原子所属 batch 索引
            - atom_offsets: torch.Tensor, (B,), PTV3 风格结束偏移
            - atom_coord_local_voxel: torch.Tensor, (sumN, 3), corner 连续体素坐标 (x, y, z)
            - box_shape_zyx: torch.Tensor, (B, 3), 体素网格尺寸 (Z, Y, X)
            - voxel_size_world: torch.Tensor, (B, 3), 体素世界尺寸 (x, y, z)
            - atom_is_in_core_box: torch.Tensor, (sumN,), bool, 是否在 core box 内

        输出:
            - dict[str, torch.Tensor | None]:
                - "voxel_pdb_embed_grid": (B, embed_voxel_out_channels, D, H, W)
                - "voxel_embed_per_atom": (sumN', embed_voxel_out_channels), scatter 前 per-atom 特征
                - "voxel_batch_index": (sumN',), voxel 路径的 batch 索引
                - "voxel_coord_local_voxel": (sumN', 3), voxel 路径的局部体素坐标


                - "embed_point_feat": (sumN', embed_point_out_channels) 或 None
                - "atom_feat": (sumN', F_atom), 裁剪后原子的原始特征(49)
                - "atom_coord_centered_world": (sumN', 3)
                - "atom_batch_index": (sumN',)
                - "atom_offsets": (B,)
                - "atom_coord_local_voxel": (sumN', 3)
                - "atom_is_in_core_box": (sumN',) bool
                
                - "global_keep_mask": (sumN,) bool
        """
        batch_size = int(atom_offsets.shape[0])
        total_n = int(atom_feat.shape[0])
        if total_n == 0:
            d_val = int(box_shape_zyx[0, 0].item())
            h_val = int(box_shape_zyx[0, 1].item())
            w_val = int(box_shape_zyx[0, 2].item())
            voxel_grid = atom_feat.new_zeros((batch_size, self.embed_voxel_out_channels, d_val, h_val, w_val))
            point_out = atom_feat.new_zeros((0, self.embed_point_out_channels)) if self.has_point_output else None
            return {
                "voxel_pdb_embed_grid": voxel_grid,
                "embed_point_feat": point_out,
                "atom_feat": atom_feat,
                "atom_coord_centered_world": atom_coord_centered_world,
                "atom_batch_index": atom_batch_index,
                "atom_offsets": atom_offsets,
                "atom_coord_local_voxel": atom_coord_local_voxel,
                "atom_is_in_core_box": atom_is_in_core_box,
                "global_keep_mask": torch.ones(0, dtype=torch.bool, device=atom_feat.device),
            }


        # --- 输入投影 ---
        # torch.Tensor, (sumN, embed_hidden_dim), 投影后的原子特征
        hidden = self.input_proj(atom_feat)
        # --- 构建 Point 对象 ---
        point = self._make_point_and_serialize(
            feat=hidden,
            coord=atom_coord_centered_world,
            batch=atom_batch_index.long(),
            offset=atom_offsets.long(),
        )
        # --- 初始化全局掩码: 追踪从原始 sumN 到最终裁剪后的映射 ---
        # torch.Tensor, (sumN,), bool, 初始全 True
        global_keep_mask = torch.ones(total_n, dtype=torch.bool, device=atom_feat.device)
        # 维护裁剪所需的辅助张量
        cur_batch = atom_batch_index.long()
        cur_offset = atom_offsets.long()
        cur_core = atom_is_in_core_box
        cur_local_voxel = atom_coord_local_voxel
        cur_coord = atom_coord_centered_world



        # ------------------------------------------------------ 共享 Trunk ------------------------------------------------------
        point, cur_coord, cur_batch, cur_offset, cur_core, cur_local_voxel, global_keep_mask = \
            self._run_blocks_with_trim(
                point=point,
                blocks=self.trunk_blocks,
                buffer_radii=self.trunk_buffer_radii,
                cur_coord=cur_coord,
                cur_batch=cur_batch,
                cur_offset=cur_offset,
                cur_core=cur_core,
                cur_local_voxel=cur_local_voxel,
                box_shape_zyx=box_shape_zyx,
                voxel_size_world=voxel_size_world,
                global_keep_mask=global_keep_mask,
            )
        # 分叉: 保存 trunk 后的状态用于点分支
        if self.has_point_output:
            trunk_feat_for_point = point.feat.clone()
            trunk_batch_for_point = cur_batch.clone()
            trunk_offset_for_point = cur_offset.clone()
            trunk_core_for_point = cur_core.clone()
            trunk_local_voxel_for_point = cur_local_voxel.clone()
            trunk_coord_for_point = cur_coord.clone()
            trunk_global_keep_for_point = global_keep_mask.clone()




        # ------------------------------------------------------ 体素专用 blocks ------------------------------------------------------
        voxel_point = point
        v_batch = cur_batch
        v_offset = cur_offset
        v_core = cur_core
        v_local_voxel = cur_local_voxel
        v_coord = cur_coord
        # 体素分支独立的全局掩码副本(不影响点分支)
        v_global_keep = global_keep_mask.clone()

        voxel_point, v_coord, v_batch, v_offset, v_core, v_local_voxel, v_global_keep = \
            self._run_blocks_with_trim(
                point=voxel_point,
                blocks=self.voxel_blocks,
                buffer_radii=self.voxel_buffer_radii,
                cur_coord=v_coord,
                cur_batch=v_batch,
                cur_offset=v_offset,
                cur_core=v_core,
                cur_local_voxel=v_local_voxel,
                box_shape_zyx=box_shape_zyx,
                voxel_size_world=voxel_size_world,
                global_keep_mask=v_global_keep,
            )

        # 体素输出投影 + scatter
        # torch.Tensor, (N_voxel, embed_voxel_out_channels), 投影后的体素特征
        voxel_feat_per_atom = self.voxel_out_proj(voxel_point.feat)
        # torch.Tensor, (B, embed_voxel_out_channels, D, H, W), 聚合到体素网格
        voxel_pdb_embed_grid = scatter_to_voxel_grid(
            point_feat=voxel_feat_per_atom,
            atom_coord_local_voxel=v_local_voxel,
            point_batch=v_batch,
            box_shape_zyx=box_shape_zyx,
            batch_size=batch_size,
            reduce=self.scatter_reduce,
        )



        # ------------------------------------------------------ 点分支专用 blocks (可选) ------------------------------------------------------
        # 点分支的裁剪结果决定最终返回给 stage1_model 的原子字段
        embed_point_feat: torch.Tensor | None = None
        if self.has_point_output:
            p_point = self._make_point_and_serialize(
                feat=trunk_feat_for_point,
                coord=trunk_coord_for_point,
                batch=trunk_batch_for_point,
                offset=trunk_offset_for_point,
            )
            p_batch = trunk_batch_for_point
            p_offset = trunk_offset_for_point
            p_core = trunk_core_for_point
            p_local_voxel = trunk_local_voxel_for_point
            p_coord = trunk_coord_for_point
            p_global_keep = trunk_global_keep_for_point

            p_point, p_coord, p_batch, p_offset, p_core, p_local_voxel, p_global_keep = \
                self._run_blocks_with_trim(
                    point=p_point,
                    blocks=self.point_blocks,
                    buffer_radii=self.point_buffer_radii,
                    cur_coord=p_coord,
                    cur_batch=p_batch,
                    cur_offset=p_offset,
                    cur_core=p_core,
                    cur_local_voxel=p_local_voxel,
                    box_shape_zyx=box_shape_zyx,
                    voxel_size_world=voxel_size_world,
                    global_keep_mask=p_global_keep,
                )

            # 点输出投影
            # torch.Tensor, (N_point, embed_point_out_channels), 投影后的点特征
            embed_point_feat = self.point_out_proj(p_point.feat)

            # 点分支的裁剪结果作为最终过滤结果
            final_global_keep = p_global_keep
            final_coord = p_coord
            final_batch = p_batch
            final_offset = p_offset
            final_core = p_core
            final_local_voxel = p_local_voxel
            # 保存裁剪前 atom_feat 对应的原始特征子集 (供残差使用)
            trimmed_atom_feat_for_point = atom_feat[p_global_keep]
            trimmed_atom_feat_for_voxel = atom_feat[v_global_keep]
        else:
            # 无点分支: 使用 trunk 的裁剪结果
            final_global_keep = global_keep_mask
            final_coord = cur_coord
            final_batch = cur_batch
            final_offset = cur_offset
            final_core = cur_core
            final_local_voxel = cur_local_voxel
            trimmed_atom_feat_for_point = atom_feat[global_keep_mask]
            trimmed_atom_feat_for_voxel = atom_feat[v_global_keep]



        # ---------- 残差融合: point 路径 ----------
        if self.embed_point_add_proj is not None and embed_point_feat is not None:
            # torch.Tensor, (N', embed_point_out_channels), 原始原子特征投影到 embed 空间
            projected_atom_for_point = self.embed_point_add_proj(trimmed_atom_feat_for_point)
            # torch.Tensor, (N', embed_point_out_channels), 投影后的原子特征 + gated embed 特征
            embed_point_feat = projected_atom_for_point + self.embed_point_gate * embed_point_feat



        # ---------- 残差融合: voxel 路径 (per-atom, scatter 前) ----------
        if self.embed_voxel_add_proj is not None:
            # torch.Tensor, (N', embed_voxel_out_channels), 原始原子特征投影到 embed voxel 空间
            projected_atom_for_voxel = self.embed_voxel_add_proj(trimmed_atom_feat_for_voxel)
            # torch.Tensor, (N', embed_voxel_out_channels), 投影后的原子特征 + gated embed 特征
            voxel_feat_per_atom = projected_atom_for_voxel + self.embed_voxel_gate * voxel_feat_per_atom
            # 重新 scatter 到体素网格
            voxel_pdb_embed_grid = scatter_to_voxel_grid(
                point_feat=voxel_feat_per_atom,
                atom_coord_local_voxel=v_local_voxel,
                point_batch=v_batch,
                box_shape_zyx=box_shape_zyx,
                batch_size=batch_size,
                reduce=self.scatter_reduce,
            )

        # -------- 确定最终 atom_feat: 残差启用且有 point 输出时为 64 维, 否则为 49 维 --------
        if self.embed_residual_enabled and self.has_point_output and embed_point_feat is not None:
            final_atom_feat = embed_point_feat
        else:
            # 无残差或无点分支: 返回原始 49 维特征
            final_atom_feat = atom_feat[final_global_keep]

        return {
            "voxel_pdb_embed_grid": voxel_pdb_embed_grid,
            "voxel_embed_per_atom": voxel_feat_per_atom,  # (N', embed_voxel_out_channels), 残差融合后 per-atom 特征
            "voxel_batch_index": v_batch,                  # voxel 路径的 batch 索引
            "voxel_coord_local_voxel": v_local_voxel,     # voxel 路径的局部体素坐标

            "embed_point_feat": embed_point_feat,         # 残差融合后的点特征
            "atom_feat": final_atom_feat,                  # 64 维(残差启用+点分支) 或 49 维
            "atom_coord_centered_world": final_coord,
            "atom_batch_index": final_batch,
            "atom_offsets": final_offset,
            "atom_coord_local_voxel": final_local_voxel,
            "atom_is_in_core_box": final_core,

            "global_keep_mask": final_global_keep,
            "embed_point_add_proj": self.embed_point_add_proj,  # Linear(49→64), 供伪原子对齐(可能为 None)
        }
