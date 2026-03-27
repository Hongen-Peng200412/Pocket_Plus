from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import nn

_PTV3_HEAD_IMPORT_ERROR: Exception | None = None
try:
    from PTV3bakcbone.model import Point, SerializedAttention
except Exception as exc:  # pragma: no cover - 依赖当前本地环境
    Point = None
    SerializedAttention = None
    _PTV3_HEAD_IMPORT_ERROR = exc


class Stage1SerializedAttentionLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        order_index: int,
        qkv_bias: bool,
        qk_scale: float | None,
        attn_drop: float,
        proj_drop: float,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
    ) -> None:
        """
            Stage1 atom head 的单层残差 SerializedAttention。

            输入参数:
                - channels: int, token 特征通道数
                - num_heads: int, 注意力头数
                - patch_size: int, SerializedAttention 的 patch size
                - order_index: int, 当前层使用的序列化顺序索引
                - qkv_bias: bool, QKV 线性层是否带 bias
                - qk_scale: float | None, QK 缩放因子
                - attn_drop: float, 注意力 dropout
                - proj_drop: float, 输出投影 dropout
                - enable_rpe: bool, 是否启用相对位置编码
                - enable_flash: bool, 是否启用 flash attention
                - upcast_attention: bool, 是否在注意力计算前上转精度
                - upcast_softmax: bool, 是否在 softmax 前上转精度

            输出:
                - point: Point, `point.feat` 已更新后的点对象
        """
        super().__init__()
        if SerializedAttention is None:
            raise ImportError("Stage1SerializedAttentionLayer 需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR

        self.norm = nn.LayerNorm(int(channels))
        self.attn = SerializedAttention(
            channels=int(channels),
            num_heads=int(num_heads),
            patch_size=int(patch_size),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(proj_drop),
            order_index=int(order_index),
            enable_rpe=bool(enable_rpe),
            enable_flash=bool(enable_flash),
            upcast_attention=bool(upcast_attention),
            upcast_softmax=bool(upcast_softmax),
        )

    def forward(self, point: Any) -> Any:
        """
        对当前 token 执行一层残差 SerializedAttention。

        输入参数:
            - point: Point, 当前点对象；`point.feat` 形状为 `(N, C)`

        输出:
            - point: Point, 执行一层注意力后的点对象
        """
        # torch.Tensor, `(N, C)`，残差分支的输入特征。
        residual_feat = point.feat
        point.feat = self.norm(point.feat)
        point = self.attn(point)
        point.feat = residual_feat + point.feat
        return point


class Stage1SerializedAttentionStack(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        num_layers: int,
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
    ) -> None:
        """
        由 k 层残差 SerializedAttention 组成的 atom head。

        输入参数:
            - channels: int, token 特征通道数。
            - num_heads: int, 多头注意力头数。
            - patch_size: int, SerializedAttention 的 patch 大小。
            - num_layers: int, 堆叠层数。
            - serialization_orders: Sequence[str], 允许使用的序列化顺序列表。
            - shuffle_orders: bool, 是否在每次 forward 开始时随机打乱序列化顺序列表。
            - qkv_bias: bool, QKV 线性层是否使用 bias。
            - qk_scale: float | None, QK 缩放因子。
            - attn_drop: float, 注意力 dropout 概率。
            - proj_drop: float, 输出投影 dropout 概率。
            - enable_rpe: bool, 是否启用相对位置编码。
            - enable_flash: bool, 是否启用 flash attention。
            - upcast_attention: bool, 是否在注意力计算前上转精度。
            - upcast_softmax: bool, 是否在 softmax 前上转精度。

        输出:
            - token_feat: torch.Tensor, `(sumN, C)`，经过 k 层 attention 后的 atom token 特征。
        """
        super().__init__()
        self.channels = int(channels)
        self.serialization_orders = tuple(str(order_name) for order_name in serialization_orders)
        self.shuffle_orders = bool(shuffle_orders)

        self.layers = nn.ModuleList(
            [
                Stage1SerializedAttentionLayer(
                    channels=self.channels,
                    num_heads=int(num_heads),
                    patch_size=int(patch_size),
                    order_index=int(layer_idx % len(self.serialization_orders)),
                    qkv_bias=bool(qkv_bias),
                    qk_scale=qk_scale,
                    attn_drop=float(attn_drop),
                    proj_drop=float(proj_drop),
                    enable_rpe=bool(enable_rpe),
                    enable_flash=bool(enable_flash),
                    upcast_attention=bool(upcast_attention),
                    upcast_softmax=bool(upcast_softmax),
                )
                for layer_idx in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.channels)

    def forward(self, point_state: dict[str, Any], token_feat: torch.Tensor) -> torch.Tensor:
        """
            执行完整的 k 层 SerializedAttention。

            输入参数:
                - point_state: dict[str, Any], 点分支输出的点状态
                - token_feat: torch.Tensor, `(N, C)`, 进入 atom head 的 token 特征

            输出:
                - output_token_feat: torch.Tensor, `(N, C)`, atom head 输出特征
        """
        if token_feat.shape[0] == 0:
            return token_feat
        if Point is None:
            raise ImportError("Stage1SerializedAttentionStack 需要 PTV3 相关依赖。") from _PTV3_HEAD_IMPORT_ERROR

        # dict[str, Any]，用点状态与当前 token 特征重建的 Point 字典。
        point_dict = {
            "feat": token_feat,
            "coord": point_state["coord"],
            "batch": point_state["batch"],
            "offset": point_state["offset"],
            "grid_size": point_state["grid_size"],
        }
        if "grid_coord" in point_state:
            point_dict["grid_coord"] = point_state["grid_coord"]

        # Point, `(sumN, C)` + 点坐标/批次/网格信息，atom head 注意力的直接输入对象。
        point = Point(point_dict)
        # Point, 序列化后的点对象；后续每层注意力将按这里确定的顺序处理 token。
        point.serialization(order=self.serialization_orders, shuffle_orders=self.shuffle_orders)

        for layer in self.layers:
            # Point, 当前层残差 SerializedAttention 更新后的点对象。
            point = layer(point)

        # torch.Tensor, `(sumN, C)`，stack 末尾 LayerNorm 后的 atom token 特征。
        return self.output_norm(point.feat)


class VolumePointStage1Model(nn.Module):
    def __init__(
        self,
        voxel_backbone: nn.Module | Any,
        point_backbone: nn.Module | Any,
        point_fusion_map: dict[str, str | None],
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
        detach_recycle_states: bool,   # True!
    ) -> None:
        """
            Stage1 总装配模型。

            输入参数:
                - voxel_backbone: nn.Module | Any, 体素分支模块或 Hydra 配置
                - point_backbone: nn.Module | Any, 点分支模块或 Hydra 配置


                - point_fusion_map: dict[str, str | None], 点分支变量名到体素特征名的映射；若子配置想关闭继承来的某个融合项，可将对应 value 设为 null
                - sampler_modes: Sequence[str], 与 `point_fusion_map` 和 `point_fusion_modes` 等长的 voxel->point 取样策略列表；每个元素对应一次命名点变量融合
                - point_fusion_modes: Sequence[str], 与 `point_fusion_map` 等长的融合策略列表
                
                - fusion_mlp_ratio: float, `concat_linear` 融合模块的隐藏层倍率
                - fusion_proj_drop: float, `concat_linear` 融合模块的 dropout


                - atom_head_hidden_dim: int, atom head 隐藏维度
                - atom_head_num_heads: int, atom head 注意力头数
                - atom_head_patch_size: int, atom head patch size
                - atom_head_num_layers: int, atom head 堆叠层数
                - atom_head_serialization_orders: Sequence[str], atom head 序列化顺序
                - atom_head_shuffle_orders: bool, atom head 是否打乱序列化顺序
                - atom_head_qkv_bias: bool, atom head 的 QKV 线性层是否带 bias
                - atom_head_qk_scale: float | None, atom head 的 QK 缩放因子
                - atom_head_attn_drop: float, atom head 注意力 dropout
                - atom_head_proj_drop: float, atom head 输出投影 dropout
                - atom_head_enable_rpe: bool, atom head 是否启用相对位置编码
                - atom_head_enable_flash: bool, atom head 是否启用 flash attention
                - atom_head_upcast_attention: bool, atom head 是否在注意力前上转精度
                - atom_head_upcast_softmax: bool, atom head 是否在 softmax 前上转精度
                - atom_logit_dim: int, atom 分类头输出维度


                - enable_recycling: bool, 是否启用 recycle
                - max_recycles: int, 最大 recycle 次数
                - randomize_recycles: bool, 训练态是否随机 recycle 次数
                - detach_recycle_states: bool, recycle 轮间是否截断梯度


            输出:
                - forward() 返回 dict[str, Any]
                    - `"atom_logits"`: torch.Tensor, `(sumN, C_logit)`, atom 分类 logits
                    - `"atom_hidden"`: torch.Tensor, `(sumN, C_head)`, 经过 k 层 SerializedAttention 后、但没经过分类MLP前的 atom 特征
                    - `"atom_tokens"`: torch.Tensor, `(sumN, C_token_in)`, atom head 输入的 token，由点特征、中心化世界坐标与 atom 有效标记拼接而成
                    - `"sampled_point_fusion_feat_dict"`: dict[str, torch.Tensor], 每个点变量名对应的采样体素特征, 仅用于记录
                    - `"voxel_outputs"`: dict[str, Any], 体素backbone(跑完最后一次recycle)输出的dict
                    - `"point_outputs"`: dict[str, Any], 点backbone(跑完最后一次recycle)输出的dict
                    - `"recycle_passes_used"`: int, 本次 forward 实际使用的 recycle 次数
        """
        super().__init__()

        # nn.Module, 体素分支模块；允许直接传模块实例或 Hydra 配置对象。
        self.voxel_backbone = voxel_backbone if isinstance(voxel_backbone, nn.Module) else instantiate(voxel_backbone)
        # nn.Module, 点分支模块；允许直接传模块实例或 Hydra 配置对象。
        self.point_backbone = point_backbone if isinstance(point_backbone, nn.Module) else instantiate(point_backbone)
        self.point_fusion_items = tuple(
            (str(point_name), voxel_name_str)
            for point_name, voxel_name in point_fusion_map.items()
            if voxel_name is not None and (voxel_name_str := str(voxel_name).strip()) != ""
        )
        # tuple[str, ...]，与 `point_fusion_items` 逐项对应的融合策略列表。
        self.point_fusion_modes = tuple(str(mode_name).lower() for mode_name in point_fusion_modes)
        # tuple[str, ...]，与 `point_fusion_items` 逐项对应的 voxel->point 采样策略列表。
        self.sampler_modes = tuple(str(mode_name).lower() for mode_name in sampler_modes)
        self.enable_recycling = bool(enable_recycling)
        self.max_recycles = int(max_recycles)
        self.randomize_recycles = bool(randomize_recycles)
        self.detach_recycle_states = bool(detach_recycle_states)
        if not (len(self.point_fusion_items) == len(self.point_fusion_modes) == len(self.sampler_modes)):
            raise ValueError("point_fusion_map、point_fusion_modes 与 sampler_modes 的长度必须一致。")
        unsupported_sampler_modes = [mode_name for mode_name in self.sampler_modes if mode_name not in {"trilinear", "nearest"}]
        if unsupported_sampler_modes:
            raise ValueError(f"Unsupported sampler_modes={unsupported_sampler_modes}, supported=['nearest', 'trilinear']")
        if self.max_recycles <= 0:
            raise ValueError("max_recycles must be > 0")





        # dict[str, str]，每个需要融合的点变量对应的体素特征名。
        self.point_fusion_map = {point_name: voxel_name for point_name, voxel_name in self.point_fusion_items}
        # dict[str, str]，每个需要融合的点变量对应的采样策略。
        self.sampler_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.sampler_modes)
        }
        # dict[str, str]，每个需要融合的点变量对应的融合策略。
        self.fusion_mode_by_point_name = {
            point_name: mode_name
            for (point_name, _), mode_name in zip(self.point_fusion_items, self.point_fusion_modes)
        }




        # tuple[str, ...]，按“体素分支显式请求返回”与“融合阶段实际需要”取并集后的体素变量名。
        self.voxel_feature_names_to_return = tuple(
            dict.fromkeys(
                tuple(str(feature_name) for feature_name in self.voxel_backbone.return_feature_keys)
                + tuple(voxel_name for _, voxel_name in self.point_fusion_items)
            )
        )
        # tuple[str, ...]，点分支本轮需要导出的命名变量名；至少包含最终的 `point_feat`。
        self.point_feature_names_to_return = tuple(
            dict.fromkeys([point_name for point_name, _ in self.point_fusion_items] + ["point_feat"])
        )




        # 收集体素和点分支的所有变量名, 查看要融合的字典 self.point_fusion_items 的合法性
        available_voxel_feature_names = tuple(self.voxel_backbone.feature_channels_by_name.keys())
        available_point_feature_names = tuple(self.point_backbone.feature_channels_by_name.keys())
        for point_name, voxel_name in self.point_fusion_items:
            if point_name not in available_point_feature_names:
                raise KeyError(f"Unknown point fusion name={point_name}, available={available_point_feature_names}")
            if voxel_name not in available_voxel_feature_names:
                raise KeyError(f"Unknown voxel fusion name={voxel_name}, available={available_voxel_feature_names}")





        # 为融合设定模块————————本代码块目前只支持 concat_linear
        self.point_fusion_modules = nn.ModuleDict()
        for point_name, voxel_name in self.point_fusion_items:
            fusion_mode = self.fusion_mode_by_point_name[point_name]
            point_channels = int(self.point_backbone.feature_channels_by_name[point_name])
            voxel_channels = int(self.voxel_backbone.feature_channels_by_name[voxel_name])
            if fusion_mode != "concat_linear":
                raise ValueError(f"Unsupported point fusion mode={fusion_mode}")

            fusion_input_dim = point_channels + voxel_channels
            fusion_hidden_dim = max(
                point_channels,
                int(round(float(fusion_input_dim) * float(fusion_mlp_ratio))),
            )
            self.point_fusion_modules[point_name] = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),   # NOTE: 未来可能需要灵活化激活函数
                nn.Dropout(float(fusion_proj_drop)),
                nn.Linear(fusion_hidden_dim, point_channels),
            )



        # 为 atom head 设定模块
        atom_token_input_dim = int(self.point_backbone.out_channels) + 3 + 1
        self.atom_token_proj = nn.Sequential(
            nn.Linear(atom_token_input_dim, int(atom_head_hidden_dim)),
            nn.LayerNorm(int(atom_head_hidden_dim)),
            nn.GELU(),
        )
        self.atom_attention_stack = Stage1SerializedAttentionStack(
            channels=int(atom_head_hidden_dim),
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
        )
        self.atom_logit_head = nn.Sequential(
            nn.Linear(int(atom_head_hidden_dim), int(atom_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(atom_head_hidden_dim), int(atom_logit_dim)),
        )




    # -------------------------------------------------------- 工具函数 --------------------------------------------------------
    @staticmethod
    def _voxel_xyz_to_grid_sample_xyz(
        point_coord_local_voxel: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 BOX 内连续 voxel corner 坐标转换为 `grid_sample` 需要的归一化坐标。

        输入参数:
            - point_coord_local_voxel: torch.Tensor, `(N, 3)`, 当前点在 BOX 内的连续 voxel corner 坐标，顺序为 `(x, y, z)`
            - box_shape_zyx: torch.Tensor, `(3,)`, 当前 BOX 的体素尺寸，顺序为 `(Z, Y, X)`

        输出:
            - grid_xyz: torch.Tensor, `(N, 3)`, 可直接送入 `grid_sample(..., align_corners=True)` 的接近于[-1,1]的归一化坐标(在外壳可能有少许超出)
        """
        # torch.Tensor, `(3,)`，当前 BOX 尺寸的 `(x, y, z)` 版本。
        box_shape_xyz = box_shape_zyx.to(
            device=point_coord_local_voxel.device,
            dtype=point_coord_local_voxel.dtype,
        )[[2, 1, 0]]
        # torch.Tensor, `(N, 3)`，从 corner 语义转成 center-index 语义。
        point_coord_center_index = point_coord_local_voxel - 0.5
        # torch.Tensor, `(3,)`，按 `align_corners=True` 归一化时使用的分母。
        denom_xyz = torch.clamp(box_shape_xyz - 1.0, min=1.0)
        # torch.Tensor, `(N, 3)`，归一化到 `[-1, 1]` 后的采样坐标。
        grid_xyz = (2.0 * point_coord_center_index / denom_xyz) - 1.0

        single_axis_mask = box_shape_xyz <= 1
        if bool(single_axis_mask.any()):
            grid_xyz[:, single_axis_mask] = 0.0
        return grid_xyz

    @staticmethod
    def _centered_world_xyz_to_local_voxel_xyz(
        point_coord_centered_world: torch.Tensor,
        voxel_size_world: torch.Tensor,
        box_shape_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """
        将以 BOX 中心为原点的世界坐标，恢复为 BOX 内连续 voxel corner 坐标。

        输入参数:
            - point_coord_centered_world: torch.Tensor, `(N, 3)`, 当前点坐标，顺序为 `(x, y, z)`，单位 Å
            - voxel_size_world: torch.Tensor, `(3,)`, 当前 BOX 的 voxel size，顺序为 `(x, y, z)`，单位 Å
            - box_shape_zyx: torch.Tensor, `(3,)`, 当前 BOX 的体素尺寸，顺序为 `(Z, Y, X)`

        输出:
            - point_coord_local_voxel: torch.Tensor, `(N, 3)`, 当前点在 BOX 内的连续 voxel corner 坐标
        """
        # torch.Tensor, `(3,)`，当前 BOX 尺寸的 `(x, y, z)` 版本。
        box_shape_xyz = box_shape_zyx.to(
            device=point_coord_centered_world.device,
            dtype=point_coord_centered_world.dtype,
        )[[2, 1, 0]]
        # torch.Tensor, `(N, 3)`，BOX 中心坐标系 -> voxel corner 坐标系。
        return point_coord_centered_world / voxel_size_world.to(
            device=point_coord_centered_world.device,
            dtype=point_coord_centered_world.dtype,
        ) + (0.5 * box_shape_xyz)

    def set_input_channels(self, in_channels: int) -> None:
        """
        将输入通道设置请求透传给体素分支。

        输入参数:
            - in_channels: int, 体素输入通道数
        """
        if hasattr(self.voxel_backbone, "set_input_channels"):
            self.voxel_backbone.set_input_channels(int(in_channels))








    #  -------------------------------------------------------- 点与体素的融合逻辑 --------------------------------------------------------
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
            对单个 BOX 的点集采样一份体素特征。

            输入参数:
                - voxel_feat_one_box: torch.Tensor, `(1, C, D_f, H_f, W_f)` 或 `(C, D_f, H_f, W_f)`，单个 BOX 的体素特征图
                - point_coord_centered_world_one_box: torch.Tensor, `(N_i, 3)`, 当前点集的中心化世界坐标
                - voxel_size_world_one_box: torch.Tensor, `(3,)`, 当前 BOX 的 voxel size
                - box_shape_zyx_one_box: torch.Tensor, `(3,)`, 当前 BOX 的尺寸
                - fusion_mode: str, 当前命名融合项使用的融合策略；当前实现 `concat_linear`
                - sampler_mode: str, 当前命名融合项使用的采样策略；当前实现 `trilinear` / `nearest`

            输出:
                - sampled_feat_one_box: torch.Tensor, `(N_i, C)`, 当前点集对应的采样体素特征
        """
        if voxel_feat_one_box.ndim == 4:
            voxel_feat_one_box = voxel_feat_one_box.unsqueeze(0)
        point_count = int(point_coord_centered_world_one_box.shape[0])
        if point_count == 0:
            return voxel_feat_one_box.new_empty((0, int(voxel_feat_one_box.shape[1])))
        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        grid_sample_mode = "bilinear" if sampler_mode == "trilinear" else "nearest"


        # torch.Tensor, `(N_i, 3)`，当前点在 BOX 内的连续 voxel corner 坐标。
        point_coord_local_voxel = self._centered_world_xyz_to_local_voxel_xyz(
            point_coord_centered_world=point_coord_centered_world_one_box,
            voxel_size_world=voxel_size_world_one_box,
            box_shape_zyx=box_shape_zyx_one_box,
        )
        #  `(N_i, 3)`，按原始几何定义换算得到的 `grid_sample` 坐标(可能略微超出[-1,1]), 这里保持坐标变换本身不做 clamp
        grid_xyz = self._voxel_xyz_to_grid_sample_xyz(
            point_coord_local_voxel=point_coord_local_voxel,
            box_shape_zyx=box_shape_zyx_one_box,
        )


        # torch.Tensor, `(N_i, 3)`，真正送入采样算子的网格坐标, 仅在三线性插值前夹到 `[-1, 1]`，让恰落在 BOX 边角的点按边界 voxel 处理
        grid_xyz_for_sampling = grid_xyz.clamp(-1.0, 1.0) if sampler_mode == "trilinear" else grid_xyz
        # torch.Tensor, `(1, N_i, 1, 1, 3)`，适配 5D `grid_sample` 的采样网格
        grid = grid_xyz_for_sampling.view(1, point_count, 1, 1, 3)
        # torch.Tensor, `(1, C, N_i, 1, 1)`，单个 BOX 内的点采样结果
        sampled = F.grid_sample(
            input=voxel_feat_one_box,
            grid=grid,
            mode=grid_sample_mode,
            padding_mode="zeros",
            align_corners=True,
        )
        # torch.Tensor, `(N_i, C)`，去掉冗余维度后的点特征
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
            对 batch 内所有点采样一份指定命名体素特征。

            输入参数:
                - voxel_feat: torch.Tensor, `(B, C, D_f, H_f, W_f)`, 体素特征图
                - point_coord_centered_world: torch.Tensor, `(N_current, 3)`, 当前点集坐标
                - point_batch_index: torch.Tensor, `(N_current,)`, 当前点集所属 batch 索引
                - voxel_size_world: torch.Tensor, `(B, 3)`, 每个 BOX 的 voxel size
                - box_shape_zyx: torch.Tensor, `(B, 3)`, 每个 BOX 的尺寸
                - fusion_mode: str, 当前命名融合项使用的融合策略；当前实现 `concat_linear`
                - sampler_mode: str, 当前命名融合项使用的采样策略

            输出:
                - sampled_feat: torch.Tensor, `(N_current, C)`, 当前点集对应的采样体素特征
        """
        point_count = int(point_coord_centered_world.shape[0])
        if point_count == 0:
            return voxel_feat.new_empty((0, int(voxel_feat.shape[1])))

        if fusion_mode != "concat_linear":
            raise ValueError(f"Unsupported point fusion mode={fusion_mode}")
        batch_size = int(box_shape_zyx.shape[0])
        # torch.Tensor, `(N_current, C)`, 整个batch的采样体素特征（顺序不变）
        sampled_feat = voxel_feat.new_empty((point_count, int(voxel_feat.shape[1])))

        for box_idx in range(batch_size):
            # torch.Tensor, `(N_current,)`, 当前 BOX 下的点掩码
            point_mask = point_batch_index == box_idx
            # torch.Tensor, `(N_i, 3)`, 当前 BOX 内的点坐标
            point_coord_one_box = point_coord_centered_world[point_mask]
            sampled_feat_one_box = self._sample_voxel_feature_single_box(
                voxel_feat_one_box=voxel_feat[box_idx : box_idx + 1],
                point_coord_centered_world_one_box=point_coord_one_box,
                voxel_size_world_one_box=voxel_size_world[box_idx],
                box_shape_zyx_one_box=box_shape_zyx[box_idx],
                fusion_mode=fusion_mode,
                sampler_mode=sampler_mode,
            )
            sampled_feat[point_mask] = sampled_feat_one_box.to(dtype=sampled_feat.dtype)

        return sampled_feat

    def _fuse_point_variable(
        self,
        feature_name: str,
        point_like: Any,
        voxel_output_dict: dict[str, Any],
        batch: dict[str, Any],
        sampled_point_fusion_feat_dict: dict[str, torch.Tensor],
    ) -> Any:
        """
            对某个命名点变量执行一次即时 voxel->point 融合。

            输入参数:
                - feature_name: str, 当前点变量名(按照字典 self.point_fusion_map 决定的规则选择是否融合、融合的对应体素特征、融合策略)
                - point_like: Any, 当前点对象；`point.feat` 形状为 `(N_current, C_point)`
                - voxel_output_dict: dict[str, Any], 当前轮体素分支输出
                - batch: dict[str, Any], 当前 batch 字典
                - sampled_point_fusion_feat_dict: dict[str, torch.Tensor], 记录采样的体素特征. str为点变量名，值为对应的采样体素特征

            输出:
                - point_like: Any, 融合后的点对象
        """
        if feature_name not in self.point_fusion_map:
            return point_like

        voxel_name = self.point_fusion_map[feature_name]
        fusion_mode = self.fusion_mode_by_point_name[feature_name]
        sampler_mode = self.sampler_mode_by_point_name[feature_name]

        # torch.Tensor, `(N_current, C_voxel)`, 当前点集按该点变量绑定的采样策略采样得到的体素特征。
        sampled_voxel_feat = self._sample_voxel_feature_batch(
            voxel_feat=voxel_output_dict["voxel_features"][voxel_name],
            point_coord_centered_world=point_like.coord,
            point_batch_index=point_like.batch,
            voxel_size_world=batch["voxel_size_world"],
            box_shape_zyx=batch["box_shape_zyx"],
            fusion_mode=fusion_mode,
            sampler_mode=sampler_mode,
        )
        sampled_point_fusion_feat_dict[feature_name] = sampled_voxel_feat

        # torch.Tensor, `(N_current, C_point + C_voxel)`, 点特征与采样体素特征的拼接结果。
        fusion_input = torch.cat([point_like.feat, sampled_voxel_feat], dim=-1)
        point_like.feat = self.point_fusion_modules[feature_name](fusion_input)  # 目前实际调用上个函数 def _sample_voxel_feature_batch
        return point_like








    #  -------------------------------------------------------- forward --------------------------------------------------------
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
            模型的总 forward

            输入:
                - batch: dict[str, Any], `box_point_collate()` 返回的 batch 字典

            输出:
                - output_dict: dict[str, Any], 包含 atom logits、中间点变量融合信息与 recycle 信息
        """
        if not self.enable_recycling:  # int, 未启用 recycle 时固定只执行 1 轮前向。
            recycle_steps = 1
        elif self.training and self.randomize_recycles:  # int, 训练态可随机采样 1~max_recycles 轮
            recycle_steps = int(torch.randint(1, self.max_recycles + 1, (1,)).item())
        else:  # int, 推理态或关闭随机 recycle 时的固定轮数。
            recycle_steps = self.max_recycles

        voxel_recycle_in: torch.Tensor | None = None
        point_recycle_in: torch.Tensor | None = None

        final_output_dict: dict[str, Any] = {}
        for _ in range(recycle_steps):
            # ------------------------------------- 体素分支 -------------------------------------
            # dict[str, Any], 当前轮体素分支原始输出，包含命名体素特征、辅助 logits 与 recycle 输出。
            voxel_output_dict = self.voxel_backbone(
                voxel_grid=batch["voxel_grid"],
                recycle_in=voxel_recycle_in,
                return_feature_keys=self.voxel_feature_names_to_return,
            )




            # ------------------------------------- 点分支与融合 -------------------------------------
            # dict[str, torch.Tensor], 记录每个命名点变量实际采样到的体素特征，仅用于可视化与调试。
            sampled_point_fusion_feat_dict: dict[str, torch.Tensor] = {}
            if getattr(self.point_backbone, "backend", None) == "zeros":
                # dict[str, Any], zeros 后端直接提供全零占位点特征；此路径不执行 point forward，也不做 voxel->point 融合。
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
                        sampled_point_fusion_feat_dict=sampled_point_fusion_feat_dict,
                    )
                # dict[str, Any], 当前轮点分支原始输出，包含点特征、点状态与中间命名点变量。
                point_output_dict = self.point_backbone(
                    atom_feat=batch["atom_feat"],
                    atom_coord_centered_world=batch["atom_coord_centered_world"],
                    atom_batch_index=batch["atom_batch_index"],
                    atom_offsets=batch["atom_offsets"],
                    recycle_in=point_recycle_in,
                    point_feature_hook=point_feature_hook,
                    return_feature_names=self.point_feature_names_to_return,
                )
            # torch.Tensor, `(sumN, C_point)`, 注意, 最后的点融合也自动完成
            fused_point_feat = point_output_dict["point_feat"]




            # ------------------------------------- atom head -------------------------------------
            # torch.Tensor, `(sumN, C_point + 3 + 1)`, atom head 输入 token，由点特征、中心化世界坐标与 atom 有效标记拼接而成。
            atom_tokens = torch.cat(
                [
                    fused_point_feat,
                    batch["atom_coord_centered_world"],
                    batch["atom_valid_mask"].to(dtype=fused_point_feat.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            # torch.Tensor, `(sumN, C_head)`，投影到 atom head 隐空间后的 token 特征。
            atom_hidden = self.atom_token_proj(atom_tokens)
            # torch.Tensor, `(sumN, C_head)`，经过 k 层 SerializedAttention 后的 atom 隐藏特征。
            atom_hidden = self.atom_attention_stack(
                point_state=point_output_dict["point_state"],
                token_feat=atom_hidden,
            )
            # torch.Tensor, `(sumN, C_logit)`，atom 分类 logits。
            atom_logits = self.atom_logit_head(atom_hidden)




            # ------------------------------------- 循环逻辑 -------------------------------------
            # torch.Tensor | None, 下一轮体素 recycle 输入；按配置决定是否截断跨轮梯度。
            voxel_recycle_in = voxel_output_dict["voxel_recycle_out"]
            if voxel_recycle_in is not None and self.detach_recycle_states:
                voxel_recycle_in = voxel_recycle_in.detach()

            # torch.Tensor | None, 下一轮点分支 recycle 输入；按配置决定是否截断跨轮梯度。
            point_recycle_in = point_output_dict["point_recycle_out"]
            if point_recycle_in is not None and self.detach_recycle_states:
                point_recycle_in = point_recycle_in.detach()

            final_output_dict = {
                "fused_point_feat": fused_point_feat,   # atom_tokens 去掉3维点坐标和1维监督mask
                "atom_tokens": atom_tokens,
                "atom_hidden": atom_hidden,
                "atom_logits": atom_logits,
                
                "sampled_point_fusion_feat_dict": sampled_point_fusion_feat_dict,   # dict[str, torch.Tensor], 记录每个命名点变量实际采样到的体素特征
                "voxel_logits_aux": voxel_output_dict["voxel_logits_aux"],
                "voxel_outputs": voxel_output_dict,
                "point_outputs": point_output_dict,

                # 下2个变量仅作为辅助, 返回的时候不管(所以注释里只写了前8个)
                "voxel_recycle_out": voxel_output_dict["voxel_recycle_out"],        # 体素分支这一轮的体素特征
                "point_recycle_out": point_output_dict["point_recycle_out"],        # 点分支这一轮的点特征
            }

        # int, 本次 forward 实际执行的 recycle 轮数。
        final_output_dict["recycle_passes_used"] = recycle_steps
        return final_output_dict
