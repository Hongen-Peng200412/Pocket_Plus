"""
Stage1 体素-点云联合模型的清理后主流程。

对齐契约（修改时必须全量同步）:
    - 本段、CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md、src/model/pseudo_atoms.py、src/model/stage1_atom_head.py 和 tests/model/test_stage1_model.py 必须同步更新。
    - forward 输入 batch 在 _run_embed_head_once 之前必须是 real-only; 伪原子不得进入 embed head。
    - P anchors 只允许在最后一次 recycle 的 _prepare_pseudo_batch 后进入 point backbone; 01 阶段 _prepare_pseudo_batch 返回 real-only batch、None layout、空 pseudo_outputs。
    - mixed layout 若存在, 必须来自 pseudo_atoms.inject_pseudo_atoms, 每个 BOX 内顺序固定为 `[real_i..., pseudo_i...]`, 先真实原子, 然后再是伪原子。

forward 输入的 batch 关键字段契约:
    - voxel_grid: torch.Tensor, (B, C_in, D, H, W), floating, voxel backbone 输入密度/特征体。
    - box_shape_zyx: torch.Tensor, (B, 3), int64/long, 每个 BOX 的体素尺寸, 轴顺序 (z, y, x)。
    - voxel_size_world: torch.Tensor, (B, 3), floating, 每个 voxel 的世界坐标尺寸, 轴顺序 (x, y, z)。
    - atom_feat: torch.Tensor, (N_real, F_atom) 或 mixed 路径下 (N_all, F_atom), floating, 点分支输入原子/P anchor 特征。
    - atom_coord_centered_world: torch.Tensor, (N_real, 3) 或 (N_all, 3), floating, 以 BOX 中心为原点的世界坐标, 轴顺序 (x, y, z)。
    - atom_coord_local_voxel: torch.Tensor, (N_real, 3) 或 (N_all, 3), floating, corner 语义连续局部体素坐标, 轴顺序 (x, y, z)。
    - atom_coord_world: torch.Tensor, (N_real, 3) 或 (N_all, 3), floating, 绝对世界坐标, 轴顺序 (x, y, z)。
    - atom_batch_index: torch.Tensor, (N_real,) 或 (N_all,), int64/long, 每个点所属 BOX 索引。
    - atom_offsets: torch.Tensor, (B,), int64/long, 每个 BOX 在展平点序列中的结束偏移。
    - atom_counts: torch.Tensor, (B,), int64/long, 每个 BOX 的点数; wrapper-facing 输出必须恢复为 real-only counts。
    - atom_label: torch.Tensor, (N_real,) 或 (N_all,), int64/long, real atom 监督标签; P anchor 槽位只允许作为占位 0。
    - atom_valid_mask: torch.Tensor, (N_real,) 或 (N_all,), bool, real atom 监督掩码; P anchor 槽位必须为 False。
    - atom_is_in_core_box: torch.Tensor, (N_real,) 或 (N_all,), bool, 点是否在 core box 内。
    - atom_global_indices: torch.Tensor, (N_real,) 或 (N_all,), int64/long, 真实原子全局索引; P anchor 槽位为 -1。
    - real_mask: torch.Tensor, (N_all,), bool, mixed-only 字段, True 表示 real atom。
    - pseudo_mask: torch.Tensor, (N_all,), bool, mixed-only 字段, True 表示 P anchor。

forward 输出契约:
    - atom_logits: torch.Tensor | None, (N_real, atom_logit_dim), floating, wrapper-facing real-only atom logits。
    - atom_target: torch.Tensor | None, (N_real,), long, wrapper-facing real-only atom 标签。
    - atom_valid_mask: torch.Tensor | None, (N_real,), bool, wrapper-facing real-only 监督掩码。
    - atom_counts: torch.Tensor | None, (B,), long, wrapper-facing real-only counts。
    - atom_tokens: torch.Tensor | None, (N_all, C_token), floating, atom head token projection 前输入; mixed 路径保留全点。
    - atom_hidden: torch.Tensor | None, (N_all, C_hidden), floating, atom head shared attention 输出; mixed 路径保留全点。
    - pseudo_feature: torch.Tensor | None, (N_pseudo, C_pseudo), floating, P anchor refined feature; 01 阶段或 real-only 路径为 None。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn

from src.model.stage1_atom_head import Stage1AtomHead
from src.model.stage1_embed_head import scatter_to_voxel_grid
from src.model.pseudo_atoms import (
    PseudoAtomLayout,
    extract_real_point_output,
    extract_real_tensor_from_mixed,
    interleave_real_and_pseudo_tensor,
)

_PTV3_IMPORT_ERROR: Exception | None = None
try:
    from src.model.PTV3bakcbone.model import resolve_act_layer
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    resolve_act_layer = None
    _PTV3_IMPORT_ERROR = exc


class VolumePointStage1Model(nn.Module):
    def __init__(
        self,
        voxel_backbone: nn.Module | Any,
        point_backbone: nn.Module | Any,
        point_fusion_map: dict[str, str] | None,
        point_fusion_modes: Sequence[str],
        sampler_modes: Sequence[str],
        fusion_mlp_ratio: float,
        fusion_proj_drop: float,
        atom_head_hidden_dim: int,
        atom_head_num_heads: int,
        atom_head_patch_size: int,
        atom_head_num_layers: int,
        atom_head_serialization_orders: Sequence[str],
        atom_head_shuffle_orders: bool,
        atom_head_qkv_bias: bool,
        atom_head_qk_scale: float | None,
        atom_head_attn_drop: float,
        atom_head_proj_drop: float,
        atom_head_enable_rpe: bool,
        atom_head_enable_flash: bool,
        atom_head_upcast_attention: bool,
        atom_head_upcast_softmax: bool,
        atom_logit_dim: int,
        enable_recycling: bool,
        max_recycles: int,
        randomize_recycles: bool,
        detach_recycle_states: bool,
        act_layer_name: str,
        ffn_type: str,
        atom_head_ffn_type: str,
        atom_head_mlp_ratio: int,
        atom_head_cpe_impl: str,
        atom_head_cpe_kernel_size: int,
        atom_head_cpe_receptive_field: float,
        atom_head_pointconv_max_neighbors: int,
        atom_head_drop_path: float,
        atom_head_pre_norm: bool,
        atom_head_append_coord_mask: bool,
        enable_atom_head: bool = True,
        embed_head: nn.Module | Any | None = None,
        pseudo_atom_cfg: dict | None = None,
        prior_prob: float | None = None,
        prior_probs: Sequence[float] | None = None,
        online_pdb_feature: bool = False,
        online_pdb_feature_reduce: str = "sum",
        online_pdb_feature_dim: int = 49,
        atom_head_pseudo_feature_dim: int | None = None,
    ) -> None:
        """
        Stage1 体素-点云联合模型, voxel backbone 每轮 recycle, P anchors 只在最后一轮注入。

        输入参数:
            - voxel_backbone: nn.Module | Any, 体素分支模块或 Hydra 配置
            - point_backbone: nn.Module | Any, 点分支模块或 Hydra 配置
            - enable_atom_head: bool, 是否构造 Stage1AtomHead
            - embed_head: nn.Module | Any | None, embed head 模块或 Hydra 配置
            - pseudo_atom_cfg: dict | None, legacy 字段; 新流程只允许 None

            - prior_prob: float | None, 单通道 sigmoid 正类先验概率
            - prior_probs: Sequence[float] | None, 多通道 softmax 类别先验概率
            - online_pdb_feature: bool, embed head 无 voxel 输出时是否在线 scatter raw atom_feat
            - online_pdb_feature_reduce: str, 在线 scatter 聚合方式
            - online_pdb_feature_dim: int, 在线 scatter 的 raw atom 特征通道数

            - point_fusion_map: dict[str, str] | None, point 变量名到 voxel 特征名的融合映射
            - point_fusion_modes: Sequence[str], 每个 point 变量的融合模式
            - sampler_modes: Sequence[str], 每个 point 变量的 voxel 采样模式
            - fusion_mlp_ratio: float, concat_linear 融合 MLP 隐藏层倍率
            - fusion_proj_drop: float, concat_linear 融合 MLP dropout 概率

            - atom_head_hidden_dim: int, atom head 隐藏通道数
            - atom_head_num_heads: int, atom head attention 头数
            - atom_head_patch_size: int, atom head SerializedAttention patch 点数
            - atom_head_num_layers: int, atom head Block 层数
            - atom_head_serialization_orders: Sequence[str], atom head 序列化顺序列表
            - atom_head_shuffle_orders: bool, atom head forward 时是否打乱序列化顺序
            - atom_head_qkv_bias: bool, atom head QKV 是否使用 bias
            - atom_head_qk_scale: float | None, atom head QK 缩放因子
            - atom_head_attn_drop: float, atom head attention dropout 概率
            - atom_head_proj_drop: float, atom head 输出投影 dropout 概率
            - atom_head_enable_rpe: bool, atom head 是否启用相对位置编码
            - atom_head_enable_flash: bool, atom head 是否启用 flash attention
            - atom_head_upcast_attention: bool, atom head attention 矩阵乘法前是否上转精度
            - atom_head_upcast_softmax: bool, atom head softmax 前是否上转精度
            - atom_logit_dim: int, real atom logits 输出通道数
            - atom_head_pseudo_feature_dim: int | None, P anchor feature 输出通道数; None 表示等于 atom_head_hidden_dim
            - atom_head_ffn_type: str, atom head Block FFN 类型
            - atom_head_mlp_ratio: int, atom head FFN 隐藏层膨胀倍率
            - atom_head_cpe_impl: str, atom head CPE 实现方式, 取值 "none" / "pointconv"
            - atom_head_cpe_kernel_size: int, legacy 字段; pointconv CPE 不消费该值
            - atom_head_cpe_receptive_field: float, atom head pointconv CPE 感受野半径
            - atom_head_pointconv_max_neighbors: int, atom head pointconv CPE 每点最大邻居数
            - atom_head_drop_path: float, atom head Block 随机深度概率
            - atom_head_pre_norm: bool, atom head Block 是否使用 pre-norm
            - atom_head_append_coord_mask: bool, 是否把 centered-world 坐标与 atom_valid_mask 拼入 atom token

            - enable_recycling: bool, 是否启用 recycle
            - max_recycles: int, 最大 recycle 轮数
            - randomize_recycles: bool, 训练态是否随机采样 recycle 轮数
            - detach_recycle_states: bool, recycle 状态跨轮传递时是否 detach
            - act_layer_name: str, 激活函数名称
            - ffn_type: str, point backbone Block FFN 类型

        前向输出:
            - outputs: dict[str, Any], 包含 voxel/point 输出、real-only atom supervised 字段与可选 pseudo_feature
        """
        super().__init__()
        if pseudo_atom_cfg is not None:
            raise ValueError("旧 pseudo_atom_cfg 已删除；P anchors 将由 sparse refine anchor pipeline 提供。")
        if resolve_act_layer is None:
            raise ImportError("VolumePointStage1Model 需要 PTV3 resolve_act_layer。") from _PTV3_IMPORT_ERROR

        self.embed_head = embed_head if isinstance(embed_head, nn.Module) else instantiate(embed_head) if embed_head is not None else None
        self.online_pdb_feature = bool(online_pdb_feature)
        self.online_pdb_feature_reduce = str(online_pdb_feature_reduce)
        self.online_pdb_feature_dim = int(online_pdb_feature_dim)

        # nn.Module, 体素分支模块
        self.voxel_backbone = voxel_backbone if isinstance(voxel_backbone, nn.Module) else instantiate(voxel_backbone)
        # nn.Module, 点分支模块
        self.point_backbone = point_backbone if isinstance(point_backbone, nn.Module) else instantiate(point_backbone)
        self.point_fusion_items = tuple(
            (str(point_name), voxel_name_str)
            for point_name, voxel_name in (point_fusion_map or {}).items()
            if voxel_name is not None and (voxel_name_str := str(voxel_name).strip()) != ""
        )
        self.point_fusion_modes = tuple(str(mode_name).lower() for mode_name in point_fusion_modes)
        self.sampler_modes = tuple(str(mode_name).lower() for mode_name in sampler_modes)
        self.enable_recycling = bool(enable_recycling)
        self.max_recycles = int(max_recycles)
        self.randomize_recycles = bool(randomize_recycles)
        self.detach_recycle_states = bool(detach_recycle_states)
        self.ffn_type = str(ffn_type)
        if not (len(self.point_fusion_items) == len(self.point_fusion_modes) == len(self.sampler_modes)):
            raise ValueError("point_fusion_map、point_fusion_modes 与 sampler_modes 的长度必须一致。")
        unsupported_sampler_modes = [mode_name for mode_name in self.sampler_modes if mode_name not in {"trilinear", "nearest"}]
        if unsupported_sampler_modes:
            raise ValueError(f"Unsupported sampler_modes={unsupported_sampler_modes}, supported=['nearest', 'trilinear']")
        if self.max_recycles <= 0:
            raise ValueError("max_recycles must be > 0")

        self.point_fusion_map = {point_name: voxel_name for point_name, voxel_name in self.point_fusion_items}
        self.sampler_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.sampler_modes)
        }
        self.fusion_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.point_fusion_modes)
        }
        self.voxel_feature_names_to_return = tuple(
            dict.fromkeys(
                tuple(str(feature_name) for feature_name in self.voxel_backbone.return_feature_keys)
                + tuple(voxel_name for _, voxel_name in self.point_fusion_items)
            )
        )
        self.point_feature_names_to_return = tuple(
            dict.fromkeys([point_name for point_name, _ in self.point_fusion_items] + ["point_feat"])
        )

        available_voxel_feature_names = tuple(self.voxel_backbone.feature_channels_by_name.keys())
        available_point_feature_names = tuple(self.point_backbone.feature_channels_by_name.keys())
        for point_name, voxel_name in self.point_fusion_items:
            if point_name not in available_point_feature_names:
                raise KeyError(f"Unknown point fusion name={point_name}, available={available_point_feature_names}")
            if voxel_name not in available_voxel_feature_names:
                raise KeyError(f"Unknown voxel fusion name={voxel_name}, available={available_voxel_feature_names}")

        # type[nn.Module], 统一激活函数类
        act_cls = resolve_act_layer(str(act_layer_name))
        # nn.ModuleDict, point 变量名 -> voxel-to-point concat_linear 融合 MLP
        self.point_fusion_modules = nn.ModuleDict()
        for point_name, voxel_name in self.point_fusion_items:
            fusion_mode = self.fusion_mode_by_point_name[point_name]
            if fusion_mode != "concat_linear":
                raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
            point_channels = int(self.point_backbone.feature_channels_by_name[point_name])
            voxel_channels = int(self.voxel_backbone.feature_channels_by_name[voxel_name])
            fusion_input_dim = point_channels + voxel_channels
            fusion_hidden_dim = max(point_channels, int(round(float(fusion_input_dim) * float(fusion_mlp_ratio))))
            self.point_fusion_modules[point_name] = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                act_cls(),
                nn.Dropout(float(fusion_proj_drop)),
                nn.Linear(fusion_hidden_dim, point_channels),
            )

        self.enable_atom_head = bool(enable_atom_head)
        if self.enable_atom_head:
            self.atom_head_append_coord_mask = bool(atom_head_append_coord_mask)
            self.atom_head = Stage1AtomHead(
                point_channels=int(self.point_backbone.out_channels),
                hidden_dim=int(atom_head_hidden_dim),
                num_heads=int(atom_head_num_heads),
                patch_size=int(atom_head_patch_size),
                num_layers=int(atom_head_num_layers),
                serialization_orders=atom_head_serialization_orders,
                shuffle_orders=bool(atom_head_shuffle_orders),
                qkv_bias=bool(atom_head_qkv_bias),
                qk_scale=atom_head_qk_scale,
                attn_drop=float(atom_head_attn_drop),
                proj_drop=float(atom_head_proj_drop),
                enable_rpe=bool(atom_head_enable_rpe),
                enable_flash=bool(atom_head_enable_flash),
                upcast_attention=bool(atom_head_upcast_attention),
                upcast_softmax=bool(atom_head_upcast_softmax),
                atom_logit_dim=int(atom_logit_dim),
                pseudo_feature_dim=atom_head_pseudo_feature_dim,
                atom_head_ffn_type=str(atom_head_ffn_type),
                mlp_ratio=int(atom_head_mlp_ratio),
                act_layer=act_cls,
                cpe_impl=str(atom_head_cpe_impl),
                cpe_kernel_size=int(atom_head_cpe_kernel_size),
                cpe_receptive_field=float(atom_head_cpe_receptive_field),
                pointconv_block_max_neighbors=int(atom_head_pointconv_max_neighbors),
                drop_path=float(atom_head_drop_path),
                pre_norm=bool(atom_head_pre_norm),
                append_coord_mask=bool(atom_head_append_coord_mask),
                prior_prob=prior_prob,
                prior_probs=prior_probs,
            )
        else:
            self.atom_head_append_coord_mask = False
            self.atom_head = None


    # ---------------------------------------- 纯粹工具函数 ----------------------------------------
    @staticmethod
    def _voxel_xyz_to_grid_sample_xyz(
        point_coord_local_voxel: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 BOX 内连续 voxel corner 坐标转换为 grid_sample [-1, 1]归一化坐标。

        输入参数:
            - point_coord_local_voxel: torch.Tensor, (N, 3), 连续体素 corner 坐标, 顺序 x/y/z
            - box_shape_zyx: torch.Tensor, (3,), BOX 体素尺寸, 顺序 z/y/x

        输出:
            - grid_xyz: torch.Tensor, (N, 3), grid_sample 使用的归一化坐标, 顺序 x/y/z
        """
        # torch.Tensor, (3,), BOX 尺寸的 x/y/z 顺序
        box_shape_xyz = box_shape_zyx.to(device=point_coord_local_voxel.device, dtype=point_coord_local_voxel.dtype)[[2, 1, 0]]
        # torch.Tensor, (N, 3), corner 坐标转 center-index 坐标
        point_coord_center_index = point_coord_local_voxel - 0.5
        # torch.Tensor, (3,), align_corners=True 归一化分母
        denom_xyz = torch.clamp(box_shape_xyz - 1.0, min=1.0)
        # torch.Tensor, (N, 3), [-1, 1] 归一化采样坐标
        grid_xyz = (2.0 * point_coord_center_index / denom_xyz) - 1.0
        return grid_xyz

    @staticmethod
    def _centered_world_xyz_to_local_voxel_xyz(
        point_coord_centered_world: torch.Tensor,
        voxel_size_world: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 centered-world 坐标转换为 BOX 内连续 voxel corner 坐标。

        输入参数:
            - point_coord_centered_world: torch.Tensor, (N, 3), centered-world 坐标: 以BOX中心为原点, 顺序 x/y/z
            - voxel_size_world: torch.Tensor, (3,), voxel 世界尺寸, 顺序 x/y/z
            - box_shape_zyx: torch.Tensor, (3,), BOX 体素尺寸, 顺序 z/y/x

        输出:
            - point_coord_local_voxel: torch.Tensor, (N, 3), 连续体素 corner 坐标, 顺序 x/y/z
        """
        # torch.Tensor, (3,), BOX 尺寸的 x/y/z 顺序
        box_shape_xyz = box_shape_zyx.to(device=point_coord_centered_world.device, dtype=point_coord_centered_world.dtype)[[2, 1, 0]]
        return point_coord_centered_world / voxel_size_world.to(
            device=point_coord_centered_world.device,
            dtype=point_coord_centered_world.dtype,
        ) + (0.5 * box_shape_xyz)

    @staticmethod
    def _counts_from_offsets(atom_offsets: torch.Tensor) -> torch.Tensor:
        """
        从累计 offset 恢复每个 BOX 的点数。

        输入参数:
            - atom_offsets: torch.Tensor, (B,), long, PTV3 风格累计 offset

        输出:
            - atom_counts: torch.Tensor, (B,), long, 每个 BOX 的点数
        """
        if atom_offsets.numel() == 0:
            return atom_offsets.new_zeros((0,), dtype=torch.long)
        atom_counts = atom_offsets.clone()
        atom_counts[1:] = atom_counts[1:] - atom_counts[:-1]
        return atom_counts

    def set_input_channels(self, in_channels: int) -> None:
        """
        将体素输入通道数设置请求透传给 voxel backbone。

        输入参数:
            - in_channels: int, 数据集 voxel_grid 通道数, 不含 embed/online scatter 追加通道

        输出:
            - None, 原地调用 voxel_backbone.set_input_channels
        """
        actual_in_channels = int(in_channels)
        if self.embed_head is not None and self.embed_head.has_voxel_output:
            extra = int(self.embed_head.embed_voxel_out_channels)
            if self.embed_head.add_occupancy_channels:
                extra += 2
            actual_in_channels += extra
        elif self.online_pdb_feature:
            actual_in_channels += self.online_pdb_feature_dim
        if hasattr(self.voxel_backbone, "set_input_channels"):
            self.voxel_backbone.set_input_channels(actual_in_channels)




    # ---------------------------------------- 体素———>点 的特征融合逻辑 ----------------------------------------
    def _sample_voxel_feature_single_box(
        self,
        voxel_feat_one_box: torch.Tensor,
        point_coord_centered_world_one_box: torch.Tensor,
        voxel_size_world_one_box: torch.Tensor,
        box_shape_zyx_one_box: torch.Tensor,
        fusion_mode: str,
        sampler_mode: str,
    ) -> torch.Tensor:
        """
        对单个 BOX 的点采样一份体素特征。

        输入参数:
            - voxel_feat_one_box: torch.Tensor, (1, C, D, H, W) 或 (C, D, H, W), 单个 BOX 的体素特征图
            - point_coord_centered_world_one_box: torch.Tensor, (N_i, 3), 当前 BOX 点坐标
            - voxel_size_world_one_box: torch.Tensor, (3,), 当前 BOX voxel 世界尺寸
            - box_shape_zyx_one_box: torch.Tensor, (3,), 当前 BOX 体素尺寸, 顺序 z/y/x
            - fusion_mode: str, 当前融合模式, 取值 concat_linear
            - sampler_mode: str, voxel 采样模式, 取值 trilinear / nearest

        输出:
            - sampled_feat_one_box: torch.Tensor, (N_i, C), 采样后的点级体素特征
        """
        if voxel_feat_one_box.ndim == 4:
            voxel_feat_one_box = voxel_feat_one_box.unsqueeze(0)
        point_count = int(point_coord_centered_world_one_box.shape[0])
        if point_count == 0:
            return voxel_feat_one_box.new_empty((0, int(voxel_feat_one_box.shape[1])))
        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        grid_sample_mode = "bilinear" if sampler_mode == "trilinear" else "nearest"
        point_coord_local_voxel = self._centered_world_xyz_to_local_voxel_xyz(
            point_coord_centered_world=point_coord_centered_world_one_box,
            voxel_size_world=voxel_size_world_one_box,
            box_shape_zyx=box_shape_zyx_one_box,
        )
        # torch.Tensor, (N_i, 3), grid_sample 归一化坐标
        grid_xyz = self._voxel_xyz_to_grid_sample_xyz(point_coord_local_voxel, box_shape_zyx_one_box)
        # torch.Tensor, (1, N_i, 1, 1, 3), 5D grid_sample 采样网格
        grid = grid_xyz.view(1, point_count, 1, 1, 3)
        # torch.Tensor, (1, C, N_i, 1, 1), 单 BOX 采样结果
        sampled = F.grid_sample(
            input=voxel_feat_one_box,
            grid=grid,
            mode=grid_sample_mode,
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1).contiguous()

    def _sample_voxel_feature_batch(
        self,
        voxel_feat: torch.Tensor,
        point_coord_centered_world: torch.Tensor,
        point_batch_index: torch.Tensor,
        voxel_size_world: torch.Tensor,
        box_shape_zyx: torch.Tensor,
        fusion_mode: str,
        sampler_mode: str,
    ) -> torch.Tensor:
        """
        对 batch 内所有点采样指定体素特征。

        输入参数:
            - voxel_feat: torch.Tensor, (B, C, D, H, W), 体素特征图
            - point_coord_centered_world: torch.Tensor, (N, 3), 当前点坐标
            - point_batch_index: torch.Tensor, (N,), 当前点所属 BOX 索引
            - voxel_size_world: torch.Tensor, (B, 3), 每个 BOX voxel 世界尺寸
            - box_shape_zyx: torch.Tensor, (B, 3), 每个 BOX 体素尺寸
            - fusion_mode: str, 当前融合模式, 取值 concat_linear
            - sampler_mode: str, voxel 采样模式, 取值 trilinear / nearest

        输出:
            - sampled_feat: torch.Tensor, (N, C), 采样后的点级体素特征
        """
        point_count = int(point_coord_centered_world.shape[0])
        if point_count == 0:
            return voxel_feat.new_empty((0, int(voxel_feat.shape[1])))
        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        batch_size = int(box_shape_zyx.shape[0])
        # torch.Tensor, (N, C), 保持点顺序的采样结果
        sampled_feat = voxel_feat.new_empty((point_count, int(voxel_feat.shape[1])))
        for box_idx in range(batch_size):
            # torch.Tensor, (N,), bool, 当前 BOX 点掩码
            point_mask = point_batch_index == box_idx
            sampled_feat[point_mask] = self._sample_voxel_feature_single_box(
                voxel_feat_one_box=voxel_feat[box_idx : box_idx + 1],
                point_coord_centered_world_one_box=point_coord_centered_world[point_mask],
                voxel_size_world_one_box=voxel_size_world[box_idx],
                box_shape_zyx_one_box=box_shape_zyx[box_idx],
                fusion_mode=fusion_mode,
                sampler_mode=sampler_mode,
            ).to(dtype=sampled_feat.dtype)
        return sampled_feat

    def _fuse_point_variable(
        self,
        feature_name: str,
        point_like: Any,
        voxel_output_dict: dict[str, Any],
        batch: dict[str, Any],
    ) -> Any:
        """
        对点云分支中名为 feature_name 的变量执行 voxel-to-point 融合。

        输入参数:
            - feature_name: str, 当前 point 变量名
            - point_like: Any, 当前 Point 对象, 包含 feat/coord/batch
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 voxel backbone 输出, 只用到 voxel_output_dict["voxel_features"][voxel_name], 而 voxel_name = self.point_fusion_map[feature_name]
            - batch: dict[str, Any], 与 point_like 同布局的 real 或 mixed batch

        输出:
            - point_like: Any, 更新 feat 后的 Point 对象
        """
        if feature_name not in self.point_fusion_map:
            return point_like
        voxel_name = self.point_fusion_map[feature_name]
        fusion_mode = self.fusion_mode_by_point_name[feature_name]
        sampler_mode = self.sampler_mode_by_point_name[feature_name]
        # torch.Tensor, (N, C_voxel), 当前点集采样得到的体素特征
        sampled_voxel_feat = self._sample_voxel_feature_batch(
            voxel_feat=voxel_output_dict["voxel_features"][voxel_name],
            point_coord_centered_world=point_like.coord,
            point_batch_index=point_like.batch,
            voxel_size_world=batch["voxel_size_world"],
            box_shape_zyx=batch["box_shape_zyx"],
            fusion_mode=fusion_mode,
            sampler_mode=sampler_mode,
        )
        # torch.Tensor, (N, C_point + C_voxel), 融合 MLP 输入特征
        fusion_input = torch.cat([point_like.feat, sampled_voxel_feat], dim=-1)
        point_like.feat = self.point_fusion_modules[feature_name](fusion_input)
        return point_like




    # ---------------------------------------- 各组件正式 forward ----------------------------------------
    def _run_embed_head_once(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        在 recycle 循环前运行 real-only embed head 并同步裁剪后的 atom 字段。

        输入参数:
            - batch: dict[str, Any], collate 后的 real-only batch

        输出:
            - batch: dict[str, Any], real-only canonical batch, 若 embed head 裁剪则字段已同步
            - embed_output: dict[str, Any] | None, embed head 输出; 未启用时为 None
        """
        if self.embed_head is None:
            return batch, None
        # dict[str, Any], embed head 输出, 包含裁剪后的 real atom 字段与可选 voxel grid
        embed_output = self.embed_head(
            atom_feat=batch["atom_feat"],
            atom_coord_centered_world=batch["atom_coord_centered_world"],
            atom_batch_index=batch["atom_batch_index"],
            atom_offsets=batch["atom_offsets"],
            atom_coord_local_voxel=batch["atom_coord_local_voxel"],
            box_shape_zyx=batch["box_shape_zyx"],
            voxel_size_world=batch["voxel_size_world"],
            atom_is_in_core_box=batch["atom_is_in_core_box"],
        )
        # torch.Tensor, (N_before,), bool, embed head 从原始 real-only 点到裁剪后点的保留掩码
        global_keep_mask = embed_output["global_keep_mask"]
        batch = {**batch}
        batch["atom_feat"] = embed_output["atom_feat"]
        batch["atom_coord_centered_world"] = embed_output["atom_coord_centered_world"]
        batch["atom_batch_index"] = embed_output["atom_batch_index"]
        batch["atom_offsets"] = embed_output["atom_offsets"]
        batch["atom_counts"] = self._counts_from_offsets(batch["atom_offsets"])
        batch["atom_coord_local_voxel"] = embed_output["atom_coord_local_voxel"]
        batch["atom_is_in_core_box"] = embed_output["atom_is_in_core_box"]
        for key in ("atom_label", "atom_valid_mask", "atom_coord_world", "atom_global_indices"):
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][global_keep_mask]
        return batch, embed_output

    def _build_voxel_input(self, batch: dict[str, Any], embed_output: dict[str, Any] | None) -> torch.Tensor:
        """
        构造 voxel backbone 输入张量(不要在这里生成候选集合 C)。

        输入参数:
            - batch: dict[str, Any], real-only canonical batch
            - embed_output: dict[str, Any] | None, embed head 输出; 未启用时为 None

        输出:
            - voxel_input: torch.Tensor, (B, C_in, D, H, W), voxel backbone 输入体素张量
        """
        if embed_output is not None and embed_output.get("voxel_pdb_embed_grid") is not None:
            # torch.Tensor, (B, C_embed, D, H, W), embed head 体素输出
            fused_voxel_grid = embed_output["voxel_pdb_embed_grid"]
            return torch.cat([batch["voxel_grid"], fused_voxel_grid], dim=1)
        if self.online_pdb_feature:
            with torch.no_grad():
                # torch.Tensor, (B, online_pdb_feature_dim, D, H, W), raw atom_feat 在线 scatter 体素网格
                raw_pdb_grid = scatter_to_voxel_grid(
                    point_feat=batch["atom_feat"].detach(),
                    atom_coord_local_voxel=batch["atom_coord_local_voxel"],
                    point_batch=batch["atom_batch_index"],
                    box_shape_zyx=batch["box_shape_zyx"],
                    batch_size=int(batch["box_shape_zyx"].shape[0]),
                    reduce=self.online_pdb_feature_reduce,
                    add_occupancy_channels=False,
                )
            return torch.cat([batch["voxel_grid"], raw_pdb_grid], dim=1)
        return batch["voxel_grid"]

    def _run_voxel_backbone(
        self,
        voxel_input: torch.Tensor,
        voxel_recycle_in: torch.Tensor | None,
    ) -> dict[str, Any]:
        """
        执行一轮 voxel backbone。

        输入参数:
            - voxel_input: torch.Tensor, (B, C_in, D, H, W), voxel backbone 输入
            - voxel_recycle_in: torch.Tensor | None, 上一轮 voxel recycle 状态

        输出:
            - voxel_output_dict: dict[str, Any], 当前轮 voxel 输出字典
        """
        return self.voxel_backbone(
            voxel_grid=voxel_input,
            recycle_in=voxel_recycle_in,
            return_feature_keys=self.voxel_feature_names_to_return,
        )

    def _prepare_pseudo_batch(
        self,
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], PseudoAtomLayout | None, dict[str, Any]]:
        """
        在最后一轮 voxel backbone 后准备 P anchor mixed batch。

        输入参数:
            - batch: dict[str, Any], 当前 real-only canonical batch
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 _run_voxel_backbone() 输出; 后续 03/04 将读取 voxel_logits_ligand 生成 C/P

        输出:
            - point_batch: dict[str, Any], 01 阶段仍为 real-only batch; 后续阶段可返回 mixed batch
            - pseudo_layout: PseudoAtomLayout | None, 01 阶段为 None; 后续阶段描述 real/P mixed 布局
            - pseudo_outputs: dict[str, Any], 01 阶段为空; 后续阶段透传 C/P 元数据
        """
        del voxel_output_dict
        return batch, None, {}

    def _run_point_backbone(
        self,
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
        point_recycle_in: torch.Tensor | None,
        pseudo_layout: PseudoAtomLayout | None = None,
    ) -> dict[str, Any]:
        """
        执行一轮 point backbone, mixed 路径下同步扩展 real-only recycle 输入。

        输入参数:
            - batch: dict[str, Any], real-only 或 mixed batch, 与 pseudo_layout 对齐
            - voxel_output_dict: dict[str, Any], 当前 recycle 的 voxel 输出
            - point_recycle_in: torch.Tensor | None, (sumN_real, C_point), 上一轮 real-only point recycle 状态
            - pseudo_layout: PseudoAtomLayout | None, mixed layout; None 表示 real-only 路径

        输出:
            - point_output_dict: dict[str, Any], point backbone 原始输出, mixed 路径下保留 mixed 顺序
        """
        # torch.Tensor | None, (sumN_current, C_point), 当前轮传给 point backbone 的 recycle 状态
        current_point_recycle = (
            interleave_real_and_pseudo_tensor(point_recycle_in, pseudo_layout)
            if pseudo_layout is not None
            else point_recycle_in
        )
        if getattr(self.point_backbone, "backend", None) == "zeros":
            point_output_dict = self.point_backbone.build_zeros_output(
                atom_feat=batch["atom_feat"],
                atom_coord_centered_world=batch["atom_coord_centered_world"],
                atom_batch_index=batch["atom_batch_index"],
                atom_offsets=batch["atom_offsets"],
                return_feature_names=self.point_feature_names_to_return,
            )
        else:
            def point_feature_hook(feature_name: str, point_like: Any) -> Any:
                return self._fuse_point_variable(
                    feature_name=feature_name,
                    point_like=point_like,
                    voxel_output_dict=voxel_output_dict,
                    batch=batch,
                )

            point_output_dict = self.point_backbone(
                atom_feat=batch["atom_feat"],
                atom_coord_centered_world=batch["atom_coord_centered_world"],
                atom_batch_index=batch["atom_batch_index"],
                atom_offsets=batch["atom_offsets"],
                recycle_in=current_point_recycle,
                point_feature_hook=point_feature_hook,
                return_feature_names=self.point_feature_names_to_return,
            )
        return point_output_dict

    def _run_atom_head(
        self,
        outputs: dict[str, Any],
        atom_head_batch: dict[str, Any],
        pseudo_layout: PseudoAtomLayout | None,
    ) -> None:
        """
        在最后一轮 point backbone 后运行 Stage1AtomHead, 并把监督字段裁成 real-only。

        输入参数:
            - outputs: dict[str, Any], 最后一轮 backbone 输出汇总, 将会原地写入 atom head 输出
            - atom_head_batch: dict[str, Any], 与 outputs["fused_point_feat"] 同布局的 real 或 mixed batch, 仅用于提供 pseudo_mask 和真实原子的信息(atom label、atom_valid_mask 等), 不提供特征
            - pseudo_layout: PseudoAtomLayout | None, mixed layout; None 表示 real-only 路径

        输出:
            - None, 原地更新 outputs 中 atom_tokens/atom_hidden/atom_logits/pseudo_feature 与 supervised 字段
        """
        if self.atom_head is None:
            outputs["atom_tokens"] = None
            outputs["atom_hidden"] = None
            outputs["atom_logits"] = None
            outputs["pseudo_feature"] = None
            return

        # torch.Tensor | None, (N_all,), bool, mixed 路径下 True 表示 P anchor
        pseudo_mask = atom_head_batch.get("pseudo_mask") if pseudo_layout is not None else None
        atom_head_output = self.atom_head(
            point_feat=outputs["fused_point_feat"],
            point_state=outputs["point_state"],
            atom_coord_centered_world=atom_head_batch["atom_coord_centered_world"],
            atom_valid_mask=atom_head_batch["atom_valid_mask"],
            pseudo_mask=pseudo_mask,
        )
        outputs.update(atom_head_output)

        if pseudo_layout is None:
            outputs["atom_target"] = atom_head_batch.get("atom_label")
            outputs["atom_valid_mask"] = atom_head_batch.get("atom_valid_mask")
            outputs["atom_counts"] = atom_head_batch.get("atom_counts")
            outputs["atom_coord_local_voxel"] = atom_head_batch.get("atom_coord_local_voxel")
            outputs["atom_is_in_core_box"] = atom_head_batch.get("atom_is_in_core_box")
            outputs["atom_global_indices"] = atom_head_batch.get("atom_global_indices")
            return

        outputs["atom_target"] = extract_real_tensor_from_mixed(atom_head_batch.get("atom_label"), pseudo_layout)
        outputs["atom_valid_mask"] = extract_real_tensor_from_mixed(atom_head_batch.get("atom_valid_mask"), pseudo_layout)
        outputs["atom_counts"] = pseudo_layout.real_counts.to(device=outputs["fused_point_feat"].device)
        outputs["atom_coord_local_voxel"] = extract_real_tensor_from_mixed(
            atom_head_batch.get("atom_coord_local_voxel"), pseudo_layout
        )
        outputs["atom_is_in_core_box"] = extract_real_tensor_from_mixed(
            atom_head_batch.get("atom_is_in_core_box"), pseudo_layout
        )
        outputs["atom_global_indices"] = extract_real_tensor_from_mixed(
            atom_head_batch.get("atom_global_indices"), pseudo_layout
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        执行 Stage1 前向, P anchor 准备点固定在最后一轮 recycle。

        输入参数:
            - batch: dict[str, Any], box_point_collate 输出的 real-only batch

        输出:
            - outputs: dict[str, Any], 最后一轮 voxel/point/atom 输出
        """
        if not self.enable_recycling:
            recycle_steps = 1
        elif self.training and self.randomize_recycles:
            recycle_steps = int(torch.randint(1, self.max_recycles + 1, (1,)).item())
        else:
            recycle_steps = self.max_recycles

        batch, embed_output = self._run_embed_head_once(batch)
        # torch.Tensor, (B, C_in, D, H, W), recycle 循环内复用的 voxel 输入
        voxel_input = self._build_voxel_input(batch, embed_output)
        voxel_recycle_in: torch.Tensor | None = None
        point_recycle_in: torch.Tensor | None = None
        outputs: dict[str, Any] = {}
        last_atom_head_batch: dict[str, Any] = batch
        last_pseudo_layout: PseudoAtomLayout | None = None


        for recycle_idx in range(recycle_steps):
            voxel_output_dict = self._run_voxel_backbone(voxel_input, voxel_recycle_in)
            is_final_recycle = recycle_idx == recycle_steps - 1
            if is_final_recycle:
                point_batch, pseudo_layout, pseudo_outputs = self._prepare_pseudo_batch(batch, voxel_output_dict)
            else:
                point_batch, pseudo_layout, pseudo_outputs = batch, None, {}

            point_output_dict = self._run_point_backbone(
                batch=point_batch,
                voxel_output_dict=voxel_output_dict,
                point_recycle_in=point_recycle_in,
                pseudo_layout=pseudo_layout,
            )

            if not is_final_recycle:
                if pseudo_layout is not None:
                    raise RuntimeError("非最后一轮 recycle 不允许注入 P anchors。")
                voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
                if voxel_recycle_in is not None and self.detach_recycle_states:
                    voxel_recycle_in = voxel_recycle_in.detach()
                point_recycle_in = point_output_dict["point_recycle_out"]
                if point_recycle_in is not None and self.detach_recycle_states:
                    point_recycle_in = point_recycle_in.detach()
                continue

            if is_final_recycle:
                if pseudo_layout is not None:
                    # real_batch 用来提供真实原子的位置与监督信息
                    # real_point_output_dict 用来提供真实原子的特征等中间结果
                    real_batch, _real_point_feat, _real_point_state, real_point_output_dict = extract_real_point_output(
                        mixed_batch=point_batch,
                        fused_point_feat=point_output_dict["point_feat"],
                        point_output_dict=point_output_dict,
                        layout=pseudo_layout,
                    )
                else:
                    real_batch = point_batch
                    real_point_output_dict = point_output_dict
                last_atom_head_batch = point_batch
                last_pseudo_layout = pseudo_layout
                outputs = {
                    "fused_point_feat": point_output_dict["point_feat"],
                    "point_state": point_output_dict["point_state"],
                    "atom_target": real_batch.get("atom_label"),
                    "atom_valid_mask": real_batch.get("atom_valid_mask"),
                    "atom_counts": real_batch.get("atom_counts"),
                    "atom_coord_local_voxel": real_batch.get("atom_coord_local_voxel"),
                    "atom_is_in_core_box": real_batch.get("atom_is_in_core_box"),
                    "atom_global_indices": real_batch.get("atom_global_indices"),
                    "voxel_logits_aux": voxel_output_dict["voxel_logits_aux"],
                    "voxel_logits_ligand": voxel_output_dict.get("voxel_logits_ligand"),
                    "voxel_outputs": voxel_output_dict,
                    "point_outputs": real_point_output_dict,
                    "embed_output": embed_output,
                    "voxel_recycle_out": voxel_output_dict["voxel_recycle_out"],
                    "point_recycle_out": real_point_output_dict["point_recycle_out"],
                    **pseudo_outputs,
                }

        self._run_atom_head(outputs, atom_head_batch=last_atom_head_batch, pseudo_layout=last_pseudo_layout)
        outputs["recycle_passes_used"] = recycle_steps
        return outputs
