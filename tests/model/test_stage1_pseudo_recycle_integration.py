# -*- coding: utf-8 -*-
"""
Stage1 伪原子 recycle 策略的整网集成测试。

覆盖范围:
    1. `non` / `pos` / `all`
       三种 policy 在 embed head + point backbone + atom head 串联下的 forward 闭环。
    2. 多轮 recycle 时 `point_recycle_in.shape[0] == atom_feat.shape[0]` 的长度一致性,
       防止再次出现“上一轮 recycle state 是 real-only、下一轮 batch 已 reinject pseudo”的拼接错误。
    3. Lightning `trainer.fit()` 的训练闭环 smoke test, 验证伪原子逻辑开启后仍能正常训练,
       且在一个线性可分的 toy 任务上能够显著降低 loss 并得到较高的 PR-AUC。
"""
from __future__ import annotations

import functools
from typing import Any

import lightning as pl
import pytest
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.stage1_model import VolumePointStage1Model
from src.wrappers.voxel_point_stage1 import VoxelPointStage1Wrapper


def _clone_tree(value: Any) -> Any:
    """
    递归克隆 batch 树结构，避免一次 forward 对下一次测试样本产生原地污染。
    输入参数:
        - value: Any, 可能为 dict / list / tuple / torch.Tensor / 其他标量对象

    输出:
        - cloned_value: Any, 与原结构同形的深拷贝结果
    """
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _make_stage1_batch() -> dict[str, torch.Tensor]:
    """
    构造一个可用于 forward / fit 的 toy batch。
    设计要点:
        - 真实原子标签由 `atom_feat[:, 0]` 的正负号决定, 形成线性可分任务
        - 其中 1 个原子被标记为 buffer atom, 由 toy embed head 裁掉, 用于验证
          embed 后 `atom_counts` / `atom_offsets` 的同步更新逻辑
        - batch 内包含 2 个 BOX, 便于检查 `split_info` 与 per-box offset

    输出:
        - batch: dict[str, torch.Tensor], 满足 `VolumePointStage1Model.forward()` 与
          `VoxelPointStage1Wrapper.training_step()` 的最小字段要求
    """
    # torch.Tensor, (2, 1, 4, 4, 4), toy voxel 网格; 两个 BOX 使用不同常值, 便于 voxel->point 融合时保持可区分性
    voxel_grid = torch.zeros((2, 1, 4, 4, 4), dtype=torch.float32)
    voxel_grid[0] = 1.0
    voxel_grid[1] = -1.0

    # torch.Tensor, (8, 4), 真实原子输入特征; 第 0 维与标签线性相关, 便于 smoke 训练快速收敛
    atom_feat = torch.tensor(
        [
            [2.2, 0.2, 0.0, 0.0],
            [1.6, -0.1, 0.1, 0.0],
            [-1.4, 0.3, 0.0, 0.0],   # buffer atom, 后续会被 toy embed head 裁掉
            [-2.1, 0.1, 0.0, 0.0],
            [-1.7, -0.2, 0.0, 0.0],
            [1.8, 0.0, 0.0, 0.0],
            [1.3, 0.4, 0.0, 0.0],
            [-1.9, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # torch.Tensor, (8, 3), 中心化世界坐标; 仅用于 point_state 与 atom token 构造
    atom_coord_centered_world = torch.tensor(
        [
            [-1.0, -0.5, -0.5],
            [-0.2, 0.2, -0.4],
            [1.5, 1.5, 1.5],
            [0.7, -0.5, 0.1],
            [1.0, 0.4, -0.2],
            [-1.0, 0.7, 0.3],
            [0.2, -0.2, 0.9],
            [1.1, 0.1, -0.7],
        ],
        dtype=torch.float32,
    )

    # torch.Tensor, (8, 3), BOX 内连续 voxel corner 坐标; 全部落在 [0, 4) 内
    atom_coord_local_voxel = torch.tensor(
        [
            [0.6, 0.7, 0.8],
            [1.4, 1.2, 0.8],
            [3.2, 3.1, 3.0],
            [0.7, 1.1, 2.2],
            [1.8, 2.5, 1.7],
            [2.3, 0.9, 1.3],
            [2.8, 1.6, 2.6],
            [1.2, 2.2, 0.7],
        ],
        dtype=torch.float32,
    )

    # torch.Tensor, (8,), long, 原子所属 BOX 索引
    atom_batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    # torch.Tensor, (2,), long, 每个 BOX 的原子数与累计 offset
    atom_counts = torch.tensor([4, 4], dtype=torch.long)
    atom_offsets = torch.tensor([4, 8], dtype=torch.long)
    # torch.Tensor, (8,), long, 二分类目标标签
    atom_label = torch.tensor([1, 1, 0, 0, 0, 1, 1, 0], dtype=torch.long)
    # torch.Tensor, (8,), bool, toy embed head 会裁掉第 3 个原子, 其余都保留
    atom_is_in_core_box = torch.tensor([True, True, False, True, True, True, True, True], dtype=torch.bool)

    return {
        "voxel_grid": voxel_grid,
        "box_shape_zyx": torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.long),
        "voxel_size_world": torch.ones((2, 3), dtype=torch.float32),
        "box_origin_world": torch.zeros((2, 3), dtype=torch.float32),
        "atom_feat": atom_feat,
        "atom_coord_centered_world": atom_coord_centered_world,
        "atom_coord_local_voxel": atom_coord_local_voxel,
        "atom_coord_world": atom_coord_local_voxel.clone(),
        "atom_batch_index": atom_batch_index,
        "atom_counts": atom_counts,
        "atom_offsets": atom_offsets,
        "atom_valid_mask": torch.ones((8,), dtype=torch.bool),
        "atom_label": atom_label,
        "atom_is_in_core_box": atom_is_in_core_box,
    }


def _make_stage1_training_batch() -> dict[str, torch.Tensor]:
    """
    构造专用于 `trainer.fit()` smoke test 的 batch。
    与 `_make_stage1_batch()` 的差异:
        - 将 `atom_is_in_core_box` 全部置为 True, 避免训练 smoke 被 embed 裁剪后的监督对齐问题干扰
        - 这样可以把训练检查聚焦在“三种 recycle policy 是否都能稳定完成训练闭环并学到 toy 目标”

    输出:
        - batch: dict[str, torch.Tensor], 适合整网训练 smoke test 的 toy batch
    """
    batch = _make_stage1_batch()
    batch["atom_is_in_core_box"] = torch.ones_like(batch["atom_is_in_core_box"], dtype=torch.bool)
    return batch


class _ToyVoxelBackbone(nn.Module):
    """
    生成可微分的 toy voxel 特征, 供 `point_feature_hook` 做 voxel->point 融合。
    输入参数:
        - 无额外构造参数

    前向输入:
        - voxel_grid: torch.Tensor, (B, C_in, D, H, W), 当前 voxel 输入

    前向输出:
        - output_dict: dict[str, Any], 包含 `voxel_c0` 与 `voxel_recycle_out`
    """

    def __init__(self) -> None:
        super().__init__()
        self.return_feature_keys = ("voxel_c0",)
        self.feature_channels_by_name = {"voxel_c0": 2}
        self.voxel_proj = nn.LazyConv3d(2, kernel_size=1, bias=False)

    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None = None,
        return_feature_keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """
        计算 toy voxel 特征。
        输入参数:
            - voxel_grid: torch.Tensor, (B, C_in, D, H, W), 当前体素输入
            - recycle_in: torch.Tensor | None, 未使用的 voxel recycle 输入
            - return_feature_keys: tuple[str, ...], 需要导出的 voxel 特征名

        输出:
            - output_dict: dict[str, Any], 与 `VolumePointStage1Model` 约定兼容的 voxel 输出字典
        """
        voxel_feat = torch.tanh(self.voxel_proj(voxel_grid))
        return {
            "voxel_features": {"voxel_c0": voxel_feat},
            "voxel_logits_aux": None,
            "voxel_recycle_out": None,
        }


class _ToyPointBackbone(nn.Module):
    """
    使用 `atom_feat` 与 `recycle_in` 构造 toy point 特征, 并显式记录长度一致性。
    输入参数:
        - out_channels: int, 标量, point backbone 输出通道数
        - point_grid_size: float, 标量, 写入 `point_state["grid_size"]` 的网格大小

    前向输入:
        - atom_feat: torch.Tensor, (sumN, C_atom), 当前点特征
        - recycle_in: torch.Tensor | None, (sumN, C_recycle), 上一轮 recycle 状态

    前向输出:
        - output_dict: dict[str, Any], point backbone 输出字典
    """

    def __init__(self, out_channels: int, point_grid_size: float) -> None:
        super().__init__()
        self.backend = "toy"
        self.out_channels = int(out_channels)
        self.point_grid_size = float(point_grid_size)
        self.feature_channels_by_name = {
            "point_feat": self.out_channels,
            "unused_point": self.out_channels,
        }
        self.atom_feature_dim = 4
        self.input_proj = nn.LazyLinear(self.out_channels)
        self.recycle_proj = nn.Linear(self.out_channels, self.out_channels)
        # list[dict[str, Any]], 记录每轮 forward 的点数、坐标与 recycle 输入, 便于校验 pseudo 语义
        self.forward_records: list[dict[str, Any]] = []

    def forward(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        recycle_in: torch.Tensor | None = None,
        point_feature_hook=None,
        return_feature_names: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """
        计算 toy point 特征与 recycle 输出。
        输入参数:
            - atom_feat: torch.Tensor, (sumN, C_atom), 当前原子特征
            - atom_coord_centered_world: torch.Tensor, (sumN, 3), 当前原子坐标
            - atom_batch_index: torch.Tensor, (sumN,), long, 当前原子所属 BOX
            - atom_offsets: torch.Tensor, (B,), long, 当前原子 offset
            - recycle_in: torch.Tensor | None, (sumN, C_recycle), 上一轮 point recycle 输出
            - point_feature_hook: callable | None, voxel->point 融合钩子
            - return_feature_names: tuple[str, ...], 需要导出的点特征名

        输出:
            - output_dict: dict[str, Any], `VolumePointStage1Model` 需要的 point 输出
        """
        num_atoms = int(atom_feat.shape[0])
        recycle_length = None if recycle_in is None else int(recycle_in.shape[0])
        self.forward_records.append(
            {
                "num_atoms": num_atoms,
                "recycle_len": recycle_length,
                "coord": atom_coord_centered_world.detach().clone(),
                "recycle_in": None if recycle_in is None else recycle_in.detach().clone(),
            }
        )
        if recycle_in is not None and recycle_in.shape[0] != atom_feat.shape[0]:
            raise AssertionError(
                f"point recycle length mismatch: atom_feat={atom_feat.shape[0]}, recycle_in={recycle_in.shape[0]}"
            )

        # torch.Tensor, (sumN, C_recycle), 缺失 recycle 输入时使用零向量占位
        recycle_in_or_zero = (
            recycle_in
            if recycle_in is not None
            else atom_feat.new_zeros((atom_feat.shape[0], self.out_channels))
        )
        # torch.Tensor, (sumN, C_point), 由 atom_feat 与 recycle state 共同生成的 toy 点特征
        point_feat = torch.tanh(self.input_proj(torch.cat([atom_feat, recycle_in_or_zero], dim=-1)))
        # torch.Tensor, (sumN, C_point), 供下一轮 recycle 使用的 point 状态
        point_recycle_out = torch.tanh(self.recycle_proj(point_feat))
        point_state = {
            "coord": atom_coord_centered_world,
            "batch": atom_batch_index.long(),
            "offset": atom_offsets.long(),
            "grid_size": self.point_grid_size,
        }
        point_feature_dict = {}
        if "point_feat" in tuple(return_feature_names):
            point_feature_dict["point_feat"] = point_feat
        if "unused_point" in tuple(return_feature_names):
            point_feature_dict["unused_point"] = point_feat
        return {
            "point_feat": point_feat,
            "point_state": point_state,
            "point_recycle_out": point_recycle_out,
            "point_feature_dict": point_feature_dict,
        }


class _ToyAttentionStack(nn.Module):
    """
    代替真实 SerializedAttention 的简化 attention stack。
    输入参数:
        - channels: int, 标量, token 隐藏维度

    前向输入:
        - point_state: dict[str, Any], 未使用的 point 状态
        - token_feat: torch.Tensor, (sumN, C), atom token 特征

    前向输出:
        - output_token_feat: torch.Tensor, (sumN, C), 简化 MLP 处理后的 token 特征
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(channels), int(channels)),
            nn.ReLU(),
        )

    def forward(self, point_state: dict[str, Any], token_feat: torch.Tensor) -> torch.Tensor:
        return self.net(token_feat)


class _ToyEmbedHead(nn.Module):
    """
    用于测试 delayed pseudo inject + embed 裁剪逻辑的 toy embed head。
    输入参数:
        - atom_feature_dim: int, 标量, embed 前真实原子特征维度
        - embed_voxel_out_channels: int, 标量, 输出给 voxel 分支的 embed 通道数
        - embed_point_out_channels: int, 标量, 输出给 point 分支的 embed 通道数

    前向输出:
        - output_dict: dict[str, Any], 与 `Stage1EmbedHead` 的关键输出字段保持兼容
    """

    def __init__(
        self,
        atom_feature_dim: int,
        embed_voxel_out_channels: int,
        embed_point_out_channels: int,
    ) -> None:
        super().__init__()
        self.atom_feature_dim = int(atom_feature_dim)
        self.embed_voxel_out_channels = int(embed_voxel_out_channels)
        self.embed_point_out_channels = int(embed_point_out_channels)
        self.has_point_output = True
        self.point_proj = nn.Linear(self.atom_feature_dim, self.embed_point_out_channels)

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
    ) -> dict[str, torch.Tensor]:
        """
        模拟 embed head 的“裁剪 + point 输出 + voxel 输出”三件事。
        输入参数:
            - atom_feat: torch.Tensor, (sumN, C_atom), embed 前原子特征
            - atom_coord_centered_world: torch.Tensor, (sumN, 3), 原子中心化世界坐标
            - atom_batch_index: torch.Tensor, (sumN,), long, 原子所属 BOX
            - atom_offsets: torch.Tensor, (B,), long, 原子 offset
            - atom_coord_local_voxel: torch.Tensor, (sumN, 3), 原子局部 voxel 坐标
            - box_shape_zyx: torch.Tensor, (B, 3), BOX 体素尺寸
            - voxel_size_world: torch.Tensor, (B, 3), voxel 世界尺寸
            - atom_is_in_core_box: torch.Tensor, (sumN,), bool, toy 裁剪掩码

        输出:
            - output_dict: dict[str, torch.Tensor], `VolumePointStage1Model` 需要的 embed 输出
        """
        del atom_offsets, voxel_size_world
        # torch.Tensor, (sumN,), bool, 仅保留 core atom, 用于触发 embed 裁剪后的计数更新逻辑
        global_keep_mask = atom_is_in_core_box.bool()
        kept_atom_feat = atom_feat[global_keep_mask]
        kept_atom_coord = atom_coord_centered_world[global_keep_mask]
        kept_atom_batch = atom_batch_index[global_keep_mask].long()
        kept_atom_local_voxel = atom_coord_local_voxel[global_keep_mask]
        kept_atom_core = atom_is_in_core_box[global_keep_mask]

        # torch.Tensor, (B,), long, 裁剪后每个 BOX 的真实原子数与 offset
        batch_size = int(box_shape_zyx.shape[0])
        kept_counts = torch.bincount(kept_atom_batch, minlength=batch_size)
        kept_offsets = kept_counts.cumsum(dim=0)
        # torch.Tensor, (sumN_kept, C_embed_point), embed head 的点分支输出
        embed_point_feat = self.point_proj(kept_atom_feat)
        # torch.Tensor, (B, C_embed_voxel, D, H, W), embed head 的体素分支输出
        d_size, h_size, w_size = [int(v) for v in box_shape_zyx[0].tolist()]
        voxel_embed = atom_feat.new_zeros((batch_size, self.embed_voxel_out_channels, d_size, h_size, w_size))

        return {
            "global_keep_mask": global_keep_mask,
            "atom_feat": kept_atom_feat,
            "atom_coord_centered_world": kept_atom_coord,
            "atom_batch_index": kept_atom_batch,
            "atom_offsets": kept_offsets,
            "atom_coord_local_voxel": kept_atom_local_voxel,
            "atom_is_in_core_box": kept_atom_core,
            "embed_point_feat": embed_point_feat,
            "voxel_pdb_embed_grid": voxel_embed,
        }


class _MaskedBCEWithLogitsLoss(nn.Module):
    """
    兼容 wrapper 调用签名的 toy BCE loss。
    输入参数:
        - logits: torch.Tensor, (sumN, 1), atom logits
        - target: torch.Tensor, (sumN,), long, 二分类标签
        - hardmask: torch.Tensor | None, (sumN,), bool, 有效原子掩码

    输出:
        - loss: torch.Tensor, 标量, 经过掩码过滤后的 BCEWithLogitsLoss
    """

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        hardmask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # torch.Tensor, (sumN,), float, squeeze 后的 logits
        logits_flat = logits.squeeze(-1)
        target_float = target.float()
        if hardmask is not None:
            valid_mask = hardmask.bool()
            logits_flat = logits_flat[valid_mask]
            target_float = target_float[valid_mask]
        if logits_flat.numel() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits_flat, target_float, reduction=reduction)


class _RepeatedBatchDataset(Dataset):
    """
    重复返回同一模板 batch 的 toy 数据集。
    输入参数:
        - template_batch: dict[str, torch.Tensor], 原始模板 batch
        - length: int, 标量, 数据集长度
    """

    def __init__(self, template_batch: dict[str, torch.Tensor], length: int) -> None:
        self.template_batch = _clone_tree(template_batch)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return _clone_tree(self.template_batch)


class _ToyStage1DataModule(pl.LightningDataModule):
    """
    为整网 smoke 训练构造最小 DataModule。
    输入参数:
        - template_batch: dict[str, torch.Tensor], train / val 共享的模板 batch
        - train_length: int, 标量, 训练集长度
        - val_length: int, 标量, 验证集长度
    """

    def __init__(self, template_batch: dict[str, torch.Tensor], train_length: int, val_length: int) -> None:
        super().__init__()
        self.template_batch = _clone_tree(template_batch)
        self.train_length = int(train_length)
        self.val_length = int(val_length)

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.train_dataset = _RepeatedBatchDataset(self.template_batch, self.train_length)
        self.val_dataset = _RepeatedBatchDataset(self.template_batch, self.val_length)

    @staticmethod
    def _collate_fn(batch_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        由于每个 step 只取 1 个模板 batch, 这里直接透传该字典。
        输入参数:
            - batch_list: list[dict[str, torch.Tensor]], 长度 = 1 的 DataLoader 批次

        输出:
            - batch_dict: dict[str, torch.Tensor], 原始 batch 字典
        """
        if len(batch_list) != 1:
            raise AssertionError(f"Expected batch_size=1, got {len(batch_list)}")
        return batch_list[0]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=self._collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=self._collate_fn)


def _build_model(recycle_policy: str, lifecycle: list[bool]) -> tuple[VolumePointStage1Model, _ToyPointBackbone]:
    """
    构造带 toy backbone 的 `VolumePointStage1Model`。
    输入参数:
        - recycle_policy: str, 标量, 当前测试使用的伪原子 recycle policy
        - lifecycle: list[bool], 长度 = 3, 当前测试使用的伪原子生命周期

    输出:
        - model: VolumePointStage1Model, 待测整网模型
        - point_backbone: _ToyPointBackbone, 便于测试读取 forward 记录
    """
    voxel_backbone = _ToyVoxelBackbone()
    point_backbone = _ToyPointBackbone(out_channels=8, point_grid_size=0.25)
    embed_head = _ToyEmbedHead(
        atom_feature_dim=4,
        embed_voxel_out_channels=1,
        embed_point_out_channels=6,
    )

    model = VolumePointStage1Model(
        voxel_backbone=voxel_backbone,
        point_backbone=point_backbone,
        point_fusion_map={"unused_point": "voxel_c0"},
        point_fusion_modes=("concat_linear",),
        sampler_modes=("nearest",),
        fusion_mlp_ratio=1.0,
        fusion_proj_drop=0.0,
        atom_head_hidden_dim=12,
        atom_head_num_heads=1,
        atom_head_patch_size=4,
        atom_head_num_layers=1,
        atom_head_serialization_orders=("z",),
        atom_head_shuffle_orders=False,
        atom_head_qkv_bias=False,
        atom_head_qk_scale=None,
        atom_head_attn_drop=0.0,
        atom_head_proj_drop=0.0,
        atom_head_enable_rpe=False,
        atom_head_enable_flash=False,
        atom_head_upcast_attention=False,
        atom_head_upcast_softmax=False,
        atom_logit_dim=1,
        enable_recycling=True,
        max_recycles=2,
        randomize_recycles=False,
        detach_recycle_states=False,
        act_layer_name="gelu",
        ffn_type="mlp",
        atom_head_ffn_type="none",
        atom_head_mlp_ratio=4,
        atom_head_cpe_impl="none",
        atom_head_cpe_kernel_size=5,
        atom_head_cpe_receptive_field=2.0,
        atom_head_pointconv_max_neighbors=16,
        atom_head_drop_path=0.0,
        atom_head_pre_norm=True,
        embed_head=embed_head,
        pseudo_atom_cfg={
            "base_count": 2,
            "scale_factor": 0.0,
            "max_sample_rounds": 2,
            "init_feat_mode": "zero",
            "init_feat_noise_std": 0.0,
            "neighbor_radius": 3.0,
            "enable_density_weighting": False,
            "density_channel_index": 0,
            "density_prob_base": 0.1,
            "delete_too_close_radius": 0.0,
            "delete_too_far_radius": 0.0,
            "lifecycle": lifecycle,
            "recycle_policy": recycle_policy,
        },
    )
    model.atom_attention_stack = _ToyAttentionStack(channels=12)
    return model, point_backbone


@pytest.mark.parametrize(
    ("recycle_policy", "lifecycle"),
    [
        ("non", [False, True, False]),
        ("pos", [False, True, False]),
        ("all", [False, True, False]),
    ],
)
def test_stage1_forward_supports_three_pseudo_recycle_policies(
    recycle_policy: str,
    lifecycle: list[bool],
) -> None:
    """
    三种 recycle policy 都应能在多轮 recycle + embed 裁剪下顺利完成 forward。
    检查点:
        - forward 不抛 shape mismatch
        - point backbone 每轮收到的 `recycle_in` 长度都与 `atom_feat` 一致
        - 最终 `atom_logits` 与 `point_outputs["point_feat"]` 都只保留真实原子
    """
    torch.manual_seed(7)
    np.random.seed(7)

    model, point_backbone = _build_model(recycle_policy=recycle_policy, lifecycle=lifecycle)
    batch = _make_stage1_batch()
    expected_real_after_embed = int(batch["atom_is_in_core_box"].sum().item())

    model.eval()
    with torch.no_grad():
        outputs = model(_clone_tree(batch))

    assert outputs["recycle_passes_used"] == 2
    assert outputs["atom_logits"].shape == (expected_real_after_embed, 1)
    assert outputs["fused_point_feat"].shape[0] == expected_real_after_embed
    assert outputs["point_outputs"]["point_feat"].shape[0] == expected_real_after_embed
    assert len(point_backbone.forward_records) == 2
    for record in point_backbone.forward_records:
        if record["recycle_len"] is not None:
            assert record["recycle_len"] == record["num_atoms"]
        assert record["num_atoms"] > expected_real_after_embed

    # toy 配置下 embed 后 BOX0/BOX1 的真实原子数分别为 3 和 4, 每个 BOX 注入 2 个伪原子:
    # mixed 顺序因此固定为 [3 real, 2 pseudo, 4 real, 2 pseudo]。
    pseudo_indices = torch.tensor([3, 4, 9, 10], dtype=torch.long)
    first_pseudo_coord = point_backbone.forward_records[0]["coord"][pseudo_indices]
    second_pseudo_coord = point_backbone.forward_records[1]["coord"][pseudo_indices]
    second_recycle_in = point_backbone.forward_records[1]["recycle_in"]
    assert second_recycle_in is not None
    second_pseudo_recycle = second_recycle_in[pseudo_indices]
    second_pseudo_recycle_zero = torch.zeros_like(second_pseudo_recycle)

    if recycle_policy == "non":
        assert not torch.allclose(first_pseudo_coord, second_pseudo_coord)
        assert torch.allclose(second_pseudo_recycle, second_pseudo_recycle_zero)
    elif recycle_policy == "pos":
        assert torch.allclose(first_pseudo_coord, second_pseudo_coord)
        assert torch.allclose(second_pseudo_recycle, second_pseudo_recycle_zero)
    else:
        assert torch.allclose(first_pseudo_coord, second_pseudo_coord)
        assert not torch.allclose(second_pseudo_recycle, second_pseudo_recycle_zero)


def test_all_policy_supports_atom_head_lifecycle() -> None:
    """
    `all` 应支持 point backbone 与 atom head 同时存在伪原子。
    """
    torch.manual_seed(3)
    np.random.seed(3)
    model, point_backbone = _build_model(
        recycle_policy="all",
        lifecycle=[False, True, True],
    )
    batch = _make_stage1_batch()

    model.eval()
    with torch.no_grad():
        outputs = model(_clone_tree(batch))

    expected_real_after_embed = int(batch["atom_is_in_core_box"].sum().item())
    assert outputs["atom_logits"].shape == (expected_real_after_embed, 1)
    assert outputs["point_outputs"]["point_feat"].shape[0] == expected_real_after_embed
    assert len(point_backbone.forward_records) == 2


@pytest.mark.parametrize("recycle_policy", ["non", "pos", "all"])
def test_voxel_point_stage1_wrapper_fit_smoke_covers_three_policies(
    recycle_policy: str,
    tmp_path,
) -> None:
    """
    在三种 pseudo recycle policy 下跑一遍 `trainer.fit()` smoke test。
    检查点:
        - 训练过程无异常
        - 训练后 atom loss 低于训练前
        - PR-AUC 明显高于随机水平, 说明 toy 目标可以被学到
    """
    torch.manual_seed(11)
    np.random.seed(11)

    batch = _make_stage1_training_batch()
    model, point_backbone = _build_model(
        recycle_policy=recycle_policy,
        lifecycle=[False, True, False],
    )
    wrapper = VoxelPointStage1Wrapper(
        backbone=model,
        atom_loss=_MaskedBCEWithLogitsLoss(),
        optimizer=functools.partial(torch.optim.Adam, lr=0.05),
        scheduler=None,
        atom_loss_weight=1.0,
        voxel_aux_loss_weight=0.0,
        monitor_metric="val/atom_pr_auc",
        interval="epoch",
        frequency=1,
        compile=False,
    )
    datamodule = _ToyStage1DataModule(
        template_batch=batch,
        train_length=8,
        val_length=2,
    )

    wrapper.eval()
    with torch.no_grad():
        pre_outputs = wrapper(_clone_tree(batch))
        pre_loss = float(wrapper._compute_atom_loss(outputs=pre_outputs, batch=_clone_tree(batch)).item())

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        deterministic=True,
    )
    trainer.fit(wrapper, datamodule=datamodule)
    validate_metrics = trainer.validate(wrapper, datamodule=datamodule, verbose=False)[0]

    wrapper.eval()
    with torch.no_grad():
        post_outputs = wrapper(_clone_tree(batch))
        post_loss = float(wrapper._compute_atom_loss(outputs=post_outputs, batch=_clone_tree(batch)).item())

    assert post_loss < pre_loss
    assert float(validate_metrics["val/atom_pr_auc"]) >= 0.90
    for record in point_backbone.forward_records:
        if record["recycle_len"] is not None:
            assert record["recycle_len"] == record["num_atoms"]
